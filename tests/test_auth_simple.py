"""Simple test to verify auth provider works within single server."""

import asyncio
import pytest
import httpx
from hypha_rpc import connect_to_server
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope

# URLs for the test server
TEST_PORT = 19527
SERVER_URL = f"http://127.0.0.1:{TEST_PORT}"
WS_SERVER_URL = f"ws://127.0.0.1:{TEST_PORT}/ws"

pytestmark = pytest.mark.asyncio


async def test_auth_provider_in_same_server(fastapi_server, test_user_token):
    """Test auth provider within the same server instance."""
    # Connect to the main test server
    api = await connect_to_server({
        "name": "test-auth-setup",
        "server_url": f"ws://127.0.0.1:{38283}/ws",  # Using the main test server
        "token": test_user_token,
    })
    
    # Register a simple auth provider
    async def extract_token_from_headers(headers):
        """Extract token from custom headers."""
        return headers.get("x-test-token")
    
    async def validate_token(token):
        """Validate the token and return UserInfo."""
        if token == "test-token-123":
            return UserInfo(
                id="test-user-123",
                is_anonymous=False,
                email="test@example.com",
                parent=None,
                roles=["test-user"],
                scope=create_scope(
                    workspaces={
                        "ws-test-user-123": UserPermission.admin,
                        "public": UserPermission.read,
                    },
                    current_workspace="ws-test-user-123",
                ),
                expires_at=None,
            )
        raise ValueError("Invalid token")
    
    # This should fail because we're not in startup context
    try:
        await api.register_service({
            "id": "test-auth-provider",
            "name": "Test Auth Provider",
            "type": "auth-provider",
            "config": {"visibility": "public"},
            "extract_token_from_headers": extract_token_from_headers,
            "validate_token": validate_token,
        })
        assert False, "Should not be able to register auth provider outside startup"
    except Exception as e:
        assert "startup functions" in str(e)
    
    await api.disconnect()
    
    # Now test HTTP requests to verify default auth still works
    async with httpx.AsyncClient() as client:
        # Test without auth - should be anonymous
        response = await client.get(f"http://127.0.0.1:38283/assets/config.json")
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "http-anonymous"
        
        # Test with standard auth token
        headers = {"Authorization": f"Bearer {test_user_token}"}
        response = await client.get(f"http://127.0.0.1:38283/assets/config.json", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "user-1"