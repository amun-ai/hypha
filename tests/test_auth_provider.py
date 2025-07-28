"""Test custom authentication provider functionality."""

import asyncio
import pytest
import httpx
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SIO_PORT

# URLs for auth server
AUTH_SERVER_URL = f"http://127.0.0.1:{SIO_PORT + 10}"
AUTH_WS_SERVER_URL = AUTH_SERVER_URL.replace("http://", "ws://") + "/ws"

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_custom_auth_provider(fastapi_server, test_user_token):
    """Test that custom auth providers can only be registered during startup."""
    # Connect as a regular user
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    # Attempting to register auth-provider from normal connection should fail
    try:
        await api.register_service({
            "id": "test-auth-provider",
            "type": "auth-provider",
            "config": {
                "visibility": "public",
                "singleton": True,
            },
            "validate_token": lambda token: {"error": "Should not work"},
        })
        assert False, "Should not be able to register auth-provider from normal connection"
    except Exception as e:
        assert "can only be registered during startup functions" in str(e)

    await api.disconnect()


async def test_auth_provider_limitations(fastapi_server_with_auth, test_user_token):
    """Document the limitations of auth providers registered during startup.

    IMPORTANT: This test documents a known architectural limitation:
    - Auth providers registered during startup cannot work properly
    - The client connection that registers them closes after startup
    - When the server tries to call auth provider methods, the connection is already closed
    - This makes custom auth providers effectively non-functional in the current architecture
    """
    # Give the server time to start
    await asyncio.sleep(2)

    # Test HTTP authentication - custom auth won't work
    async with httpx.AsyncClient() as client:
        # Without any auth - anonymous
        response = await client.get(f"{AUTH_SERVER_URL}/assets/config.json")
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "http-anonymous"

        # With CF-Token header - still anonymous because auth provider can't be called
        headers = {"CF-Token": "test-cloudflare-token-123"}
        response = await client.get(f"{AUTH_SERVER_URL}/assets/config.json", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "http-anonymous"  # Auth provider doesn't work

        # Standard auth still works
        headers = {"Authorization": f"Bearer {test_user_token}"}
        response = await client.get(f"{AUTH_SERVER_URL}/assets/config.json", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "user-1"


async def test_auth_provider_asgi_service(fastapi_server_with_auth):
    """Test that ASGI services registered by auth provider work fine."""
    await asyncio.sleep(2)  # Let services register

    # ASGI services work because they don't require the client connection
    async with httpx.AsyncClient() as client:
        # Test the custom login app
        response = await client.get(f"{AUTH_SERVER_URL}/public/apps/custom-auth-login/")
        assert response.status_code == 200
        assert b"Custom Cloudflare Login" in response.content

        # Test login API
        response = await client.post(
            f"{AUTH_SERVER_URL}/public/apps/custom-auth-login/api/login",
            json={"token": "test-cloudflare-token-123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user"]["id"] == "cf-user-123"
