"""Test the token."""

import pytest
from hypha_rpc import connect_to_server

from . import (
    WS_SERVER_URL,
    find_item,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_generate_token(fastapi_server):
    """Test connecting to the server with a token."""
    api1 = await connect_to_server(
        {"name": "my app", "server_url": WS_SERVER_URL, "client_id": "my-app"}
    )
    token = await api1.generate_token()
    try:
        api2 = await connect_to_server(
            {
                "name": "my app",
                "server_url": WS_SERVER_URL,
                "client_id": "my-app",
                "token": token,
                "workspace": api1.config.workspace,
            }
        )
        assert await api2.echo("hello") == "hello"
        await api2.disconnect()
    except Exception as e:
        assert "Client already exists and is active:" in str(e)
    await api1.disconnect()


async def test_revoke_token(fastapi_server):
    """Test connecting to the server with a revoked token."""
    async with connect_to_server(
        {"name": "my app", "server_url": WS_SERVER_URL, "client_id": "my-app"}
    ) as api1:
        token = await api1.generate_token()

        async with connect_to_server(
            {
                "name": "my app",
                "server_url": WS_SERVER_URL,
                "token": token,
                "workspace": api1.config.workspace,
            }
        ) as api2:
            assert await api2.echo("hello") == "hello"

            await api1.revoke_token(token)

            try:
                async with connect_to_server(
                    {
                        "name": "my app",
                        "server_url": WS_SERVER_URL,
                        "token": token,
                        "workspace": api1.config.workspace,
                    }
                ):
                    assert False, "Should have raised an exception"
            except Exception as e:
                assert "Token has been revoked" in str(e)


async def test_generate_token_with_custom_scope(fastapi_server):
    """Test token with custom scope."""
    async with connect_to_server(
        {"name": "my app", "server_url": WS_SERVER_URL, "client_id": "my-app"}
    ) as api1:
        token = await api1.generate_token({"extra_scopes": ["my-data:read"]})

        async with connect_to_server(
            {
                "name": "my app",
                "server_url": WS_SERVER_URL,
                "client_id": "my-app-2",
                "token": token,
                "workspace": api1.config.workspace,
            }
        ) as api2:
            assert "my-data:read" in api2.config["user"]["scope"]["extra_scopes"]


async def test_generate_token_with_client_id_restriction(fastapi_server):
    """Test token with client_id restriction."""
    async with connect_to_server(
        {"name": "my app", "server_url": WS_SERVER_URL, "client_id": "my-app"}
    ) as api1:
        # Generate a token restricted to a specific client_id
        token = await api1.generate_token({"client_id": "allowed-client"})

        # Try to connect with the allowed client_id - should succeed
        async with connect_to_server(
            {
                "name": "allowed app",
                "server_url": WS_SERVER_URL,
                "client_id": "allowed-client",
                "token": token,
                "workspace": api1.config.workspace,
            }
        ) as api2:
            assert await api2.echo("hello") == "hello"
            # Verify the client_id is in the scope
            assert api2.config["user"]["scope"]["client_id"] == "allowed-client"

        # Try to connect with a different client_id - should fail
        try:
            async with connect_to_server(
                {
                    "name": "unauthorized app",
                    "server_url": WS_SERVER_URL,
                    "client_id": "unauthorized-client",
                    "token": token,
                    "workspace": api1.config.workspace,
                }
            ):
                assert (
                    False
                ), "Should have raised an exception due to client_id mismatch"
        except Exception as e:
            assert "Client id mismatch" in str(e) and "allowed-client" in str(e)


async def test_generate_token_without_client_id_restriction(fastapi_server):
    """Test token without client_id restriction works with any client."""
    async with connect_to_server(
        {"name": "my app", "server_url": WS_SERVER_URL, "client_id": "my-app"}
    ) as api1:
        # Generate a token without client_id restriction
        token = await api1.generate_token()

        # Should work with any client_id
        async with connect_to_server(
            {
                "name": "any app",
                "server_url": WS_SERVER_URL,
                "client_id": "any-client",
                "token": token,
                "workspace": api1.config.workspace,
            }
        ) as api2:
            assert await api2.echo("hello") == "hello"
            # The client_id should be set to the connecting client
            assert api2.config["user"]["scope"]["client_id"] == "any-client"
