"""Test that register_service works after WebSocket reconnection.

Reproduces GitHub issue: https://github.com/oeway/hypha-rpc/issues/144

The bug: after reconnection, the wm proxy refresh handler copies ALL
functions from the fresh remote manager onto the wm object, overwriting
locally-bound methods like register_service/unregister_service.  The
overwritten remote register_service sends raw service dicts to the server
without constructing the proper {client_id}:{service_id} format, causing
assertion errors on the server side.

The bug requires TWO reconnections to trigger because the
_is_initial_refresh flag is not consumed during the initial connection
(the handler is registered after the initial manager_refreshed event),
so it eats the first reconnection event.

Run: pytest tests/test_reconnection_register_service.py -v --timeout=120
"""

import asyncio

import pytest
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL

pytestmark = pytest.mark.asyncio


async def _force_reconnect(server):
    """Force-close the WebSocket and wait for reconnection + re-registration."""
    services_registered = asyncio.Event()

    def on_services_registered(info):
        services_registered.set()

    server.rpc.on("services_registered", on_services_registered)

    ws = server.rpc._connection._websocket
    if hasattr(ws, "close"):
        await ws.close()
    else:
        ws.close()

    await asyncio.wait_for(services_registered.wait(), timeout=30)
    server.rpc.off("services_registered", on_services_registered)
    # Give the async manager_refreshed handler time to complete
    await asyncio.sleep(2)


async def test_register_service_survives_reconnection(fastapi_server):
    """register_service must work after multiple WebSocket reconnections.

    The bug in issue #144 overwrites wm.register_service with the remote
    proxy version on the second reconnection.  This test verifies that:
    - register_service remains the local wrapper after reconnection
    - Registering new services works correctly
    - Previously registered services survive reconnection
    """
    server = await connect_to_server(
        {
            "name": "reconnect-register-test",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
        }
    )

    # Register a service before any reconnection
    svc1 = await server.register_service(
        {
            "id": "pre-reconnect-svc",
            "name": "Pre Reconnect Service",
            "hello": lambda name: f"hello {name}",
        }
    )
    assert ":" in svc1["id"]

    # Capture the original register_service identity
    original_register_service = server.register_service

    # First reconnection — consumes _is_initial_refresh flag
    await _force_reconnect(server)
    assert server.register_service is original_register_service, (
        "register_service should not change after first reconnection"
    )

    # Second reconnection — this is where the bug triggered
    await _force_reconnect(server)
    assert not hasattr(server.register_service, "_encoded_method"), (
        "register_service should be the local wrapper, not a RemoteFunction"
    )

    # THE CRITICAL TEST: Register a NEW service after reconnection.
    # Before the fix, this failed with either:
    #   AssertionError: Service id info must contain ':'
    #   AttributeError: 'NoneType' object has no attribute 'workspace'
    svc2 = await server.register_service(
        {
            "id": "post-reconnect-svc",
            "name": "Post Reconnect Service",
            "greet": lambda name: f"hi {name}",
        }
    )
    assert ":" in svc2["id"]

    # Verify the new service works end-to-end
    found = await server.get_service(svc2["id"])
    result = await found.greet("world")
    assert result == "hi world"

    # Verify echo still works (remote methods retargeted correctly)
    assert await server.echo("test") == "test"

    await server.disconnect()
