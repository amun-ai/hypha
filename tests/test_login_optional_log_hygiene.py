"""#0021 — ``login_optional`` must not emit a per-request INFO line on the hot auth path.

``hypha/core/store.py`` logged, for *every* token-bearing HTTP/MCP request::

    logger.info(f"login_optional: parsed user_info: id={...}, is_anonymous={...}")

In a prod deployment with ``HYPHA_LOGLEVEL=INFO`` this fires on the stateless
``/rpc`` + ``/mcp`` auth boundary once per request — measured at ~39% of all prod
log lines / ~915k lines/day (graceful-salmon nightly review 2026-08-11). It has no
steady-state diagnostic value: it merely restates that a valid token parsed. The
excess volume shortens the ``kubectl logs`` forensic horizon (the retained window
sawtooths to ~1 min right after a kubelet rotation), which directly obstructed
root-causing a ~106s edge outage on 2026-08-10.

Fix: downgrade the line to ``logger.debug`` — kept for deep debugging, silent at
INFO. Same log-hygiene family as #0008 (JWT-expiry) and #0020 (mcp.py DEBUG).

These are real tests: a genuine Docker-free ``RedisStore`` (FakeRedis), a genuine
minted auth token, and a genuine ``login_optional`` call under ``caplog`` — no
mocks. One test proves the line is silent at INFO (the regression guard); a second
proves it is NOT deleted, only downgraded (still present at DEBUG).
"""
import logging

import pytest

from hypha.core import UserInfo
from hypha.core.auth import create_scope, generate_auth_token
from hypha.core.store import RedisStore

_MARKER = "login_optional: parsed user_info"
_LOGGER = "redis-store"


class _FakeRequest:
    """Minimal stand-in carrying the one attribute ``login_optional`` reads."""

    def __init__(self, scope):
        self.scope = scope


async def _authenticated_request():
    """A request scope carrying a valid ``Authorization: Bearer <token>`` header,
    the exact shape ``extract_token_from_scope`` reads (auth.py:356-366)."""
    ws = "ws-user-loghygiene"
    user_info = UserInfo(
        id="loghygiene-user",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scope=create_scope(f"{ws}#a", current_workspace=ws),
        expires_at=None,
    )
    token = await generate_auth_token(user_info, 3600)
    return _FakeRequest(
        {"headers": [(b"authorization", f"Bearer {token}".encode())]}
    )


@pytest.mark.asyncio
async def test_login_optional_does_not_log_parsed_user_info_at_info(caplog):
    """REGRESSION GUARD: a token-bearing login_optional must emit NOTHING containing
    the parse-confirmation marker at INFO level."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        request = await _authenticated_request()
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            user_info = await store.login_optional(request)
        assert user_info is not None and not user_info.is_anonymous
        offending = [
            r for r in caplog.records
            if r.name == _LOGGER and r.levelno >= logging.INFO and _MARKER in r.getMessage()
        ]
        assert not offending, (
            f"login_optional emitted {len(offending)} INFO+ record(s) containing "
            f"{_MARKER!r} — this per-request line must be DEBUG (see #0021). "
            f"Messages: {[r.getMessage() for r in offending]}"
        )
    finally:
        await store.teardown()


@pytest.mark.asyncio
async def test_login_optional_still_logs_parsed_user_info_at_debug(caplog):
    """The line is DOWNGRADED, not DELETED: at DEBUG it must still be present so the
    diagnostic is available when explicitly debugging."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        request = await _authenticated_request()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            await store.login_optional(request)
        present = [
            r for r in caplog.records
            if r.name == _LOGGER and _MARKER in r.getMessage()
        ]
        assert present, (
            f"Expected the {_MARKER!r} line to still be emitted at DEBUG "
            "(downgraded, not removed)."
        )
        assert all(r.levelno == logging.DEBUG for r in present), (
            "The parse-confirmation line must be logged at DEBUG level exactly."
        )
    finally:
        await store.teardown()
