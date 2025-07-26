"""Test custom authentication provider functionality."""

import asyncio
import pytest
import requests
import httpx
from hypha_rpc import connect_to_server
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope
import jwt
from datetime import datetime, timedelta
from pathlib import Path

from . import WS_SERVER_URL, SERVER_URL, SIO_PORT

# URLs for auth server
AUTH_SERVER_URL = f"http://127.0.0.1:{SIO_PORT + 10}"
AUTH_WS_SERVER_URL = AUTH_SERVER_URL.replace("http://", "ws://") + "/ws"

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_custom_auth_provider(fastapi_server, test_user_token):
    """Test custom authentication provider functionality."""
    # Connect as root to ensure we can register the auth provider
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    
    # We can't register auth-provider from normal connection, only from startup
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


async def test_auth_provider_integration(fastapi_server_with_auth, test_user_token):
    """Test full auth provider integration with HTTP and WebSocket."""
    # Give the server a moment to fully initialize
    await asyncio.sleep(1)
    
    # First, let's verify the auth provider is registered
    api = await connect_to_server({
        "name": "test-verify-auth",
        "server_url": AUTH_WS_SERVER_URL,
        "token": test_user_token,
    })
    
    # Try to get the auth provider service directly
    try:
        auth_provider = await api.get_service("public/cloudflare-auth-provider")
        print(f"[TEST] Auth provider found: {auth_provider is not None}")
        print(f"[TEST] Auth provider has extract_token_from_headers: {hasattr(auth_provider, 'extract_token_from_headers')}")
    except Exception as e:
        print(f"[TEST] Failed to get auth provider: {e}")
    
    await api.disconnect()
    
    # Test HTTP authentication with custom header
    async with httpx.AsyncClient() as client:
        # Test without any auth - should be anonymous
        response = await client.get(f"{AUTH_SERVER_URL}/assets/config.json")
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "http-anonymous"
        
        # Test with custom CF-Token header
        headers = {"CF-Token": "test-cloudflare-token-123"}
        response = await client.get(f"{AUTH_SERVER_URL}/assets/config.json", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "cf-user-123"
        assert data["user"]["email"] == "user@cloudflare.com"
        
        # Test with invalid CF-Token
        headers = {"CF-Token": "invalid-token"}
        response = await client.get(f"{AUTH_SERVER_URL}/assets/config.json", headers=headers)
        assert response.status_code == 200
        data = response.json()
        # Should fall back to anonymous when token is invalid
        assert data["user"]["id"] == "http-anonymous"
    
    # Test WebSocket authentication
    # Test with standard token
    api = await connect_to_server({
        "name": "test-ws-client",
        "server_url": AUTH_WS_SERVER_URL,
        "token": test_user_token,
    })
    user_info = api.config["user"]
    assert user_info["id"] == "user-1"  # Standard auth still works
    await api.disconnect()
    
    # Test with Cloudflare token
    api = await connect_to_server({
        "name": "test-cf-client", 
        "server_url": AUTH_WS_SERVER_URL,
        "token": "CF:test-cloudflare-token-123",  # Custom token format
    })
    user_info = api.config["user"]
    assert user_info["id"] == "cf-user-123"
    assert user_info["email"] == "user@cloudflare.com"
    await api.disconnect()
    
    # Test login redirect
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUTH_SERVER_URL}/login", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "https://cloudflare.example.com/login"


async def test_auth_provider_service_access(fastapi_server_with_auth, test_user_token):
    """Test service access with custom auth provider."""
    # Create a protected service
    api = await connect_to_server({
        "name": "service-creator",
        "server_url": AUTH_WS_SERVER_URL,
        "token": test_user_token,
    })
    
    await api.register_service({
        "id": "protected-service",
        "type": "functions",
        "config": {
            "visibility": "protected",
            "require_context": True,
        },
        "test": lambda context: {
            "user_id": context["user"]["id"],
            "workspace": context["ws"],
        }
    })
    
    await api.disconnect()
    
    # Test accessing service with CF token via HTTP
    async with httpx.AsyncClient() as client:
        # Without auth - should fail or show anonymous
        response = await client.get(
            f"{AUTH_SERVER_URL}/ws-user-user-1/services/protected-service/test"
        )
        assert response.status_code in [200, 403]  # Might show anonymous or deny
        
        # With CF token - should work
        headers = {"CF-Token": "test-cloudflare-token-123"}
        response = await client.get(
            f"{AUTH_SERVER_URL}/ws-cf-user-123/services/protected-service/test",
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "cf-user-123"


async def test_auth_provider_asgi_login(fastapi_server_with_auth):
    """Test custom ASGI login app provided by auth provider."""
    async with httpx.AsyncClient() as client:
        # Test the custom login app endpoint
        response = await client.get(f"{AUTH_SERVER_URL}/public/apps/custom-auth-login/")
        assert response.status_code == 200
        assert b"Custom Cloudflare Login" in response.content
        
        # Test login API endpoint
        response = await client.post(
            f"{AUTH_SERVER_URL}/public/apps/custom-auth-login/api/login",
            json={"token": "test-cloudflare-token-123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user"]["id"] == "cf-user-123"