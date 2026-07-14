"""Regression tests for two caught-but-noisy WebSocket-lifecycle bugs.

Reported by the nightly prod review (crimson-spider, 2026-07-07) on 0.21.105.
Both are caught (no crash, no leak) but burn CPU + log volume and mask real
errors, so they are hardened here.

[1] Teardown race — a redis event reaches `filtered_handler` AFTER the connection
    was torn down and `self._subscriptions` was set to None, raising
    `TypeError: argument of type 'NoneType' is not iterable`
    (~640/24h, the top ERROR). Root: `EventBus.emit` snapshots the handler list
    and creates the handler coroutine *before* awaiting it, so a handler removed
    by `disconnect()` mid-flight still runs — after `_subscriptions` was niled.

[2] recv/send-after-close — the receive loop keeps calling the ASGI transport
    after the peer already sent a `websocket.disconnect`, raising
    `RuntimeError: Cannot call "receive" once a disconnect message has been
    received.` (~260/24h), and a symmetric send-after-close (~156/24h). Root:
    the loop doesn't detect the disconnect message and break cleanly.
"""
import asyncio
import os

import msgpack
import pytest

from hypha.core import RedisRPCConnection, UserInfo
from hypha.core.auth import create_scope
from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


def _user(ws):
    return UserInfo(
        id="hardening",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scope=create_scope(f"{ws}#a", current_workspace=ws),
        expires_at=None,
    )


# ── Bug [1]: filtered_handler after teardown ────────────────────────────────


async def test_filtered_handler_ignores_events_after_teardown():
    """A broadcast event dispatched to filtered_handler after disconnect() has
    niled self._subscriptions must be a no-op, not a TypeError."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        event_bus = store.get_event_bus()
        ws_name = "ws-user-teardown"
        conn = RedisRPCConnection(
            event_bus, ws_name, "clientA", _user(ws_name), store.get_manager_id()
        )

        received = []

        def send_handler(msg):
            received.append(msg)

        conn.on_message(send_handler)
        conn.subscribe("evt")  # so filtered_handler would otherwise forward "evt"

        # Capture the filtered_handler BEFORE teardown (disconnect() nils it +
        # off()s it, but an already-created coroutine from EventBus.emit's
        # snapshot would still run — this simulates that late dispatch).
        filtered_handler = conn._filtered_handler
        assert filtered_handler is not None

        await conn.disconnect("disconnected")
        assert conn._subscriptions is None  # teardown niled it

        # The late broadcast event must NOT raise and must NOT be forwarded.
        late_event = msgpack.packb({"type": "evt", "data": 1})
        await filtered_handler(late_event)  # pre-fix: TypeError on `in None`

        assert received == [], "a torn-down connection must not forward late events"
    finally:
        await store.teardown()


# ── Bug [2]: recv/send-after-close ──────────────────────────────────────────


class _FakeState:
    name = "CONNECTED"


class _DisconnectingWS:
    """receive() returns a websocket.disconnect once, then raises like Starlette."""

    def __init__(self):
        self.client_state = _FakeState()
        self.application_state = _FakeState()
        self.recv_count = 0
        self.closed = False
        self.sent = []

    async def receive(self):
        self.recv_count += 1
        if self.recv_count == 1:
            return {"type": "websocket.disconnect", "code": 1000}
        # Starlette raises this once a disconnect has been delivered.
        raise RuntimeError(
            'Cannot call "receive" once a disconnect message has been received.'
        )

    async def send_bytes(self, data):
        self.sent.append(data)

    async def send_text(self, data):
        self.sent.append(data)

    async def close(self, code=None, reason=None):
        self.closed = True


async def _make_server(store):
    from fastapi import FastAPI
    from hypha.websocket import WebsocketServer

    store._app = FastAPI()
    return WebsocketServer(store)


def test_expected_close_error_classifier():
    """The expected-close RuntimeErrors are classified benign (→ DEBUG); real
    errors are not."""
    from hypha.websocket import _is_expected_close_error

    assert _is_expected_close_error(
        RuntimeError('Cannot call "receive" once a disconnect message has been received.')
    )
    assert _is_expected_close_error(
        RuntimeError('Cannot call "send" once a close message has been sent.')
    )
    # Real errors must NOT be downgraded.
    assert not _is_expected_close_error(RuntimeError("something actually broke"))
    assert not _is_expected_close_error(ValueError("not even a RuntimeError"))


async def test_receive_after_disconnect_breaks_cleanly():
    """When the peer sends websocket.disconnect, the receive loop must break
    cleanly (calling receive() exactly once) instead of re-entering receive()
    and raising 'Cannot call receive once a disconnect message has been
    received.'"""
    os.environ["HYPHA_WS_IDLE_TIMEOUT"] = "600"
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws_name = "ws-user-disc"
        user = _user(ws_name)
        await store.register_workspace(
            dict(name=ws_name, description="x", owners=["hardening"],
                 persistent=False, read_only=False),
            overwrite=True,
        )

        closed_before = RedisRPCConnection._closed_total_int
        ws = _DisconnectingWS()

        # Pre-fix: this raises RuntimeError (loop re-enters receive()).
        # Post-fix: returns cleanly.
        await server.establish_websocket_communication(ws, ws_name, "c1", user)

        assert ws.recv_count == 1, (
            f"receive() must be called exactly once (broke on disconnect), "
            f"got {ws.recv_count} — the loop re-entered receive() after disconnect"
        )
        # The connection must still be cleaned up.
        assert RedisRPCConnection._closed_total_int - closed_before >= 1
    finally:
        await store.teardown()
        os.environ.pop("HYPHA_WS_IDLE_TIMEOUT", None)


async def test_register_client_task_after_teardown_no_error():
    """The background register_client() task must not crash if the connection is
    torn down (self._event_bus set to None) before it runs.

    Repro of a prod teardown race (core/__init__.py): on a reconnect via
    reconnection_token followed by an immediate 1001 disconnect, the
    register_client() task is scheduled just as disconnect() nils
    self._event_bus, so `await self._event_bus.subscribe_to_client_events(...)`
    raised `AttributeError: 'NoneType' object has no attribute
    subscribe_to_client_events`. The guard must make it exit cleanly instead.
    """
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        event_bus = store.get_event_bus()
        conn = RedisRPCConnection(
            event_bus, "ws-user-reconnrace", "clientA",
            _user("ws-user-reconnrace"), store.get_manager_id(),
        )
        # on_message() schedules register_client() as conn._registration_task but
        # does not await it, so it has not run yet when control returns here.
        conn.on_message(lambda m: None)
        task = conn._registration_task
        assert task is not None and not task.done()

        # Simulate teardown winning the race: disconnect() sets _event_bus = None.
        conn._event_bus = None

        # Now let the scheduled task run — it must complete without raising.
        await task
        assert task.done()
        assert task.exception() is None, (
            f"register_client raised {task.exception()!r} after teardown"
        )
    finally:
        await store.teardown()
