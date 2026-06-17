"""F6 incident #2 (2026-06-17): reject-storm to a disconnected peer at N>=2.

When a client with in-flight RPCs disconnects, the server cleans up its pending
promises by routing reject/result callbacks back to the (now-gone) client. Each
such fire-and-forget message hit the dead-peer check and RAISED "Target peer is
not connected" — a storm (52 per gone client observed) that churned the client
into a reconnect loop at N>=2.

Fix: the dead-peer check fast-fails ONLY primary calls awaiting a response
(which carry a "session"); fire-and-forget traffic (no "session") to a gone peer
is dropped silently. Distinct from the cross-pod migration fix (#969).

Docker-free: drives RedisRPCConnection.emit_message directly with a minimal
fake event bus that reports the target as recently-disconnected.
"""
import msgpack
import pytest

from hypha.core import RedisRPCConnection

pytestmark = pytest.mark.asyncio


class _FakeBus:
    def __init__(self):
        self.emitted = []

    def is_local_client(self, ws, client_id):
        return False  # target is not on this pod

    def is_recently_disconnected(self, ws, client_id):
        return True  # target recently disconnected from this pod

    async def emit(self, name, data):
        self.emitted.append(name)


def _conn(bus):
    c = RedisRPCConnection.__new__(RedisRPCConnection)
    c._workspace = "ws1"
    c._client_id = "sender"
    c._readonly = False
    c._user_info = {}
    c._event_bus = bus
    c._stop = False
    return c


async def test_primary_call_to_gone_peer_fast_fails():
    """A primary call (has 'session') to a gone peer must raise so the caller's
    await fails fast instead of timing out (preserves issue #868 behavior)."""
    bus = _FakeBus()
    conn = _conn(bus)
    data = msgpack.packb(
        {"to": "ws1/gone", "type": "method", "method": "do", "session": "s1"}
    )
    with pytest.raises(Exception, match="not connected"):
        await conn.emit_message(data)
    assert bus.emitted == [], "must not deliver to a gone peer"


async def test_fire_and_forget_to_gone_peer_drops_silently():
    """A fire-and-forget message (no 'session' — e.g. a reject/result callback
    to a gone caller) must be dropped silently, NOT raise. This is the fix for
    the reject-storm."""
    bus = _FakeBus()
    conn = _conn(bus)
    data = msgpack.packb(
        {"to": "ws1/gone", "type": "method", "method": "reject"}  # no session
    )
    # Must NOT raise (no storm) ...
    await conn.emit_message(data)
    # ... and must not be routed to the gone peer.
    assert bus.emitted == []


async def test_many_rejects_to_gone_peer_no_storm():
    """Simulate the cleanup storm: many fire-and-forget rejects to a gone client
    — none should raise."""
    bus = _FakeBus()
    conn = _conn(bus)
    for i in range(52):
        data = msgpack.packb(
            {"to": "ws1/gone", "type": "method", "method": f"cb{i}"}
        )
        await conn.emit_message(data)  # no raise on any of them
    assert bus.emitted == []
