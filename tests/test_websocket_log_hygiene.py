"""Log-hygiene regression tests (0.21.107): benign WebSocket lifecycle events must
not surface as ERROR and mask real errors.

Flagged by the nightly prod reviews on 0.21.106:

[1] uvicorn logs "Exception in ASGI application" at ERROR (full traceback) whenever
    a websocket ASGI task ends with `asyncio.CancelledError` — which happens on
    EVERY normal client disconnect (uvicorn cancels `run_asgi`) and on our own
    supersede-cancel of a stale generation (~238/24h steady state, ~320/min during
    a rollout reconnect wave). This is the dominant benign-at-ERROR source and it
    originates at the uvicorn layer, so it is quieted with a logging filter on the
    `uvicorn.error` logger — without touching cancellation semantics.

[2] A send-after-close race in the teardown path: `disconnect()` sent a text frame
    after the peer already closed → `RuntimeError: Cannot call "send" once a close
    message has been sent`.

Plus three benign-at-ERROR log lines downgraded (disconnect / send-fail / close-race).
"""
import asyncio
import logging
import sys

import pytest

from hypha.core.store import RedisStore
from hypha.websocket import (
    _CancelledErrorASGIFilter,
    install_asgi_cancellederror_filter,
)

pytestmark = pytest.mark.asyncio


class _FakeState:
    name = "CONNECTED"


def _record_with_exc(exc_type, exc):
    """Build a LogRecord carrying exc_info like uvicorn's 'Exception in ASGI application'."""
    try:
        raise exc
    except exc_type:
        return logging.LogRecord(
            "uvicorn.error", logging.ERROR, __file__, 1,
            "Exception in ASGI application", (), sys.exc_info(),
        )


# ── [1] uvicorn.error CancelledError filter ─────────────────────────────────


def test_cancellederror_asgi_filter_drops_only_cancellederror():
    f = _CancelledErrorASGIFilter()

    # CancelledError terminal → suppressed.
    rec = _record_with_exc(asyncio.CancelledError, asyncio.CancelledError())
    assert f.filter(rec) is False

    # A real exception → passes through (real errors must still surface).
    rec2 = _record_with_exc(ValueError, ValueError("a real bug"))
    assert f.filter(rec2) is True

    # A record with no exc_info → passes through.
    rec3 = logging.LogRecord(
        "uvicorn.error", logging.ERROR, __file__, 1, "some message", (), None
    )
    assert f.filter(rec3) is True


async def test_filter_installed_on_uvicorn_logger_idempotently():
    from fastapi import FastAPI
    from hypha.websocket import WebsocketServer

    uvicorn_logger = logging.getLogger("uvicorn.error")
    # Clean slate for a deterministic assertion.
    for flt in [f for f in uvicorn_logger.filters if isinstance(f, _CancelledErrorASGIFilter)]:
        uvicorn_logger.removeFilter(flt)

    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        store._app = FastAPI()
        WebsocketServer(store)  # __init__ installs the filter
        installed = [
            f for f in uvicorn_logger.filters
            if isinstance(f, _CancelledErrorASGIFilter)
        ]
        assert len(installed) == 1, "filter must be installed exactly once"

        # Idempotent: installing again does not duplicate it.
        install_asgi_cancellederror_filter()
        installed = [
            f for f in uvicorn_logger.filters
            if isinstance(f, _CancelledErrorASGIFilter)
        ]
        assert len(installed) == 1, "filter install must be idempotent"
    finally:
        await store.teardown()


# ── [2] send-after-close in disconnect(), + log-level downgrades ────────────


class _SendAfterCloseWS:
    """send_text raises like Starlette after the peer already closed."""

    def __init__(self):
        self.client_state = _FakeState()
        self.application_state = _FakeState()
        self.closed = False

    async def send_text(self, data):
        raise RuntimeError(
            'Cannot call "send" once a close message has been sent.'
        )

    async def close(self, code=None, reason=None):
        self.closed = True


async def _make_server(store):
    from fastapi import FastAPI
    from hypha.websocket import WebsocketServer

    store._app = FastAPI()
    return WebsocketServer(store)


async def test_disconnect_survives_send_after_close():
    """disconnect() must not raise when the teardown-path send races a close."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws = _SendAfterCloseWS()
        # Must not raise despite send_text raising the close-race RuntimeError.
        await server.disconnect(ws, "reason", 1000)
        assert ws.closed, "disconnect() should still close the socket"
    finally:
        await store.teardown()


async def test_disconnect_logs_at_info_not_error(caplog):
    """WebsocketServer.disconnect() logs its lifecycle line at INFO, not ERROR."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws = _SendAfterCloseWS()
        with caplog.at_level(logging.DEBUG, logger="websocket-server"):
            await server.disconnect(ws, "Client normal close", 1000)

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, (
            f"disconnect() must not log at ERROR: {[r.getMessage() for r in errors]}"
        )
        assert any(
            "Disconnecting, reason" in r.getMessage() and r.levelno == logging.INFO
            for r in caplog.records
        ), "the disconnect lifecycle line must be logged at INFO"
    finally:
        await store.teardown()
