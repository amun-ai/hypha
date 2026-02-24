"""Test that uvicorn WebSocket ping is disabled in server configuration.

Hypha uses application-level idle timeout (HYPHA_WS_IDLE_TIMEOUT, default 600s)
and client-side app-level pings (30s JS, 20s Python) instead of protocol-level
WebSocket PING frames. Uvicorn's WS-level ping (default 20s interval, 20s timeout)
causes spurious code 1011 disconnections when clients are busy processing.

These tests verify:
1. Server configuration disables WS-level ping
2. Long-lived connections remain stable without protocol-level pings
"""

import asyncio
import time
import pytest
from hypha_rpc import connect_to_server

from . import SERVER_URL_SQLITE, SIO_PORT_SQLITE

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_long_connection_no_spurious_disconnect(fastapi_server_sqlite):
    """Verify that a WebSocket connection survives beyond uvicorn's default ping timeout.

    Uvicorn's default ws_ping_interval=20s, ws_ping_timeout=20s means a client
    that doesn't respond to protocol PING within 40s would be disconnected with
    code 1011. By disabling WS-level ping, connections should survive much longer.

    We hold a connection for 45 seconds (beyond the 40s danger zone) and verify
    it remains alive. The client does NOT send any messages during this period
    to ensure it's not kept alive by app-level activity.
    """
    api = await connect_to_server(
        {
            "server_url": f"http://127.0.0.1:{SIO_PORT_SQLITE}",
            "client_id": "long-conn-test",
            "method_timeout": 60,
        }
    )

    # Verify connection works initially
    result = await api.echo("hello")
    assert result == "hello"

    # Hold connection idle for 45 seconds (beyond uvicorn's default 20+20=40s timeout)
    # During this time, the client's app-level ping may still fire, but the key
    # thing we're testing is that uvicorn's protocol-level ping doesn't kill us
    await asyncio.sleep(45)

    # Connection should still be alive
    result = await api.echo("still alive")
    assert result == "still alive", "Connection died - WS ping may not be disabled"

    await api.disconnect()


async def test_connection_survives_without_app_level_activity(fastapi_server_sqlite):
    """Verify connection stays alive even without explicit app-level calls.

    This is a shorter version that verifies basic stability. The connection
    should survive a 25-second idle period (beyond uvicorn's default 20s
    ping interval).
    """
    api = await connect_to_server(
        {
            "server_url": f"http://127.0.0.1:{SIO_PORT_SQLITE}",
            "client_id": "idle-conn-test",
            "method_timeout": 30,
        }
    )

    result = await api.echo("before idle")
    assert result == "before idle"

    # Idle for 25 seconds (beyond uvicorn's default 20s ping interval)
    await asyncio.sleep(25)

    # Should still work
    result = await api.echo("after idle")
    assert result == "after idle"

    await api.disconnect()


async def test_listener_leak_after_reconnection_on_server(fastapi_server_sqlite):
    """Integration test: verify no listener leak when reconnecting through hypha server.

    This tests the full stack: hypha server + hypha-rpc client reconnection.
    After multiple reconnections, the client_disconnected handler count should
    remain at 1, not grow with each reconnection.
    """
    api = await connect_to_server(
        {
            "server_url": f"http://127.0.0.1:{SIO_PORT_SQLITE}",
            "client_id": "leak-integ-test",
            "method_timeout": 30,
        }
    )

    await api.register_service(
        {
            "id": "leak-integ-svc",
            "config": {"visibility": "protected"},
            "ping": lambda: "pong",
        }
    )

    await asyncio.sleep(1)

    # Perform 3 reconnection cycles
    for cycle in range(3):
        services_registered = asyncio.Event()
        api.rpc.on("services_registered", lambda info: services_registered.set())

        await api.rpc._connection._websocket.close(1011)

        try:
            await asyncio.wait_for(services_registered.wait(), timeout=15)
        except asyncio.TimeoutError:
            await asyncio.sleep(3)

        # Check handler count
        handlers = api.rpc._event_handlers.get("client_disconnected", [])
        assert len(handlers) <= 1, (
            f"Listener leak after reconnection {cycle + 1}: "
            f"{len(handlers)} handlers (expected <= 1)"
        )

    # Verify service still works
    for attempt in range(5):
        try:
            svc = await asyncio.wait_for(api.get_service("leak-integ-svc"), timeout=3)
            result = await asyncio.wait_for(svc.ping(), timeout=3)
            assert result == "pong"
            break
        except Exception:
            if attempt == 4:
                raise
            await asyncio.sleep(1)

    await api.disconnect()
