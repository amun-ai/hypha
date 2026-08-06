"""Tests for #0019: server->client RPC method-call timeout log hygiene.

When a service call over HTTP (e.g. ``services.outpost.poll`` to a half-open
laptop client) times out because the callee stopped answering, the server should
log a WARNING and return HTTP 504 — NOT an ERROR + full traceback + 500. The
traceback for such a timeout names nothing actionable (just the internal rpc
await chain), and these are high-volume on production.

Two layers of coverage:

1. Unit tests on the pure classifier ``_is_expected_service_timeout`` — these pin
   the carve-out to EXACTLY the hypha-rpc method-call timeout message and prove it
   cannot silently widen to swallow OS-level (OSError-family) timeouts or the rpc
   session-expiry timer. Fast and deterministic; construct real builtin
   ``TimeoutError`` instances (never route through ``asyncio.wait_for``, so the
   3.10-vs-3.11 builtin/asyncio TimeoutError aliasing cannot false-pass).

2. One genuine full-stack test: a callee client whose ``poll`` handler BLOCKS its
   own event loop (so it stops heartbeating) drives the server's real 30s rpc
   method-call Timer to fire, and the HTTP request must come back 504.
"""

import asyncio
import threading
import time

import httpx
import pytest
from hypha_rpc import connect_to_server

from . import SIO_PORT_SQLITE, SERVER_URL_SQLITE
from hypha.http import _is_expected_service_timeout

WS_SERVER_URL_SQLITE = f"ws://127.0.0.1:{SIO_PORT_SQLITE}/ws"


# ---------------------------------------------------------------------------
# Unit tests: the pure classifier. No network, no fixtures.
# ---------------------------------------------------------------------------


def test_classifier_matches_rpc_method_timeout():
    """The exact hypha-rpc method-call timeout message -> True (byte-for-byte
    shape of a real production line)."""
    exc = TimeoutError(
        "Method call timed out: ws-user-github|478667/machine-WeiOMacMini-fb9cfb45:"
        "services.outpost.poll, context: some-description"
    )
    assert _is_expected_service_timeout(exc) is True


def test_classifier_rejects_non_timeout_type():
    """A non-timeout exception in the same arm still gets 500 + traceback —
    proves the downgrade wasn't widened to all errors."""
    assert _is_expected_service_timeout(ValueError("boom")) is False


def test_classifier_rejects_generic_timeout_message():
    """A bare TimeoutError with a different message must NOT be downgraded."""
    assert _is_expected_service_timeout(TimeoutError("timed out")) is False


def test_classifier_rejects_oserror_family_timeout():
    """TimeoutError is a subclass of OSError; an OS-level socket/DNS/S3 timeout
    raised INSIDE a service function must keep its traceback (it names the dead
    dependency), so it must NOT match."""
    # TimeoutError.__mro__ == [TimeoutError, OSError, Exception, BaseException, object]
    assert TimeoutError.__mro__[1] is OSError  # guard the premise
    assert _is_expected_service_timeout(OSError("Connection timed out")) is False


def test_classifier_rejects_session_expiry_timeout():
    """The rpc SESSION-expiry timer (rpc.py:1779, "Session expired (TTL=...)") is
    a rare diagnostic client-leak signal and stays ERROR + 500 + traceback."""
    assert (
        _is_expected_service_timeout(
            TimeoutError("Session expired (TTL=300s): some-session-key")
        )
        is False
    )


# ---------------------------------------------------------------------------
# Full-stack test: a real server-side rpc method-call timeout -> HTTP 504.
# ---------------------------------------------------------------------------

# Must exceed the server-side rpc method_timeout (hard-coded 30s in
# store.py::create_rpc). The callee blocks its loop for this long so it stops
# heartbeating and the server's Timer fires a genuine builtin TimeoutError.
_BLOCK_SECONDS = 40


def _run_blocked_callee(ready_evt, stop_evt, ws_url, token, result):
    """Runs in a background thread with its OWN event loop.

    Registers an ``outpost`` service whose ``poll`` handler blocks this thread's
    event loop (``time.sleep``), which stops the auto-heartbeat and forces the
    server's method-call Timer to time out. NOT ``run_in_executor`` on purpose:
    the handler MUST block the loop, not a worker thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def main():
        api = await connect_to_server(
            {
                "name": "outpost-callee",
                "server_url": ws_url,
                "method_timeout": 30,
                "token": token,
            }
        )
        result["workspace"] = api.config["workspace"]

        def poll(*args, **kwargs):
            # Block THIS client's event loop: the heartbeat task cannot run, so
            # the server sees no pings and its 30s method Timer fires.
            time.sleep(_BLOCK_SECONDS)
            return "pong"

        svc = await api.register_service(
            {
                "id": "outpost",
                "name": "outpost",
                "config": {"visibility": "public"},
                "poll": poll,
            }
        )
        result["service_id"] = svc.id.split("/")[1]
        ready_evt.set()

        while not stop_evt.is_set():
            await asyncio.sleep(0.1)
        await api.disconnect()

    loop.run_until_complete(main())
    loop.close()


@pytest.mark.asyncio
async def test_service_method_timeout_returns_504(
    fastapi_server_sqlite, test_user_token
):
    """A genuine server->client method-call timeout returns HTTP 504 (not 500)."""
    ready_evt = threading.Event()
    stop_evt = threading.Event()
    result = {}

    thread = threading.Thread(
        target=_run_blocked_callee,
        args=(ready_evt, stop_evt, WS_SERVER_URL_SQLITE, test_user_token, result),
        daemon=True,
    )
    thread.start()
    assert ready_evt.wait(timeout=30), "callee failed to register service in time"

    workspace = result["workspace"]
    service_id = result["service_id"]
    url = f"{SERVER_URL_SQLITE}/{workspace}/services/{service_id}/poll"

    # Client timeout must exceed the server's 30s method timeout so the server
    # produces the 504 rather than the client aborting first.
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url)

    status = resp.status_code
    text = resp.text
    body = resp.json() if resp.headers.get("content-type", "").startswith(
        "application/json"
    ) else {}

    # Tear the callee down before asserting (no try/finally): stop the loop, let
    # the in-flight blocking sleep drain, join the thread.
    stop_evt.set()
    thread.join(timeout=_BLOCK_SECONDS + 10)

    assert status == 504, f"expected 504, got {status}: {text}"
    assert body.get("success") is False, body
    assert "timed out" in body.get("detail", "").lower(), body
