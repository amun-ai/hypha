"""Test custom authentication with a real server."""
import pytest
import asyncio
from hypha_rpc import connect_to_server
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope, generate_auth_token


@pytest.mark.asyncio
async def test_custom_auth_server_basic(custom_auth_server):
    """Test that custom auth server starts and accepts connections."""
    server_url = custom_auth_server.replace("http://", "ws://") + "/ws"
    
    # First, try to connect without auth (should work for anonymous)
    async with connect_to_server(
        {
            "client_id": "test-client-anon",
            "server_url": server_url,
        }
    ) as api:
        # Verify we can access public services
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        # Service IDs include the full path, so check if any ends with our service name
        custom_auth_service_found = any("custom-auth-test" in sid for sid in service_ids)
        assert custom_auth_service_found, f"Custom auth test service not found. Available: {service_ids}"
        
        # Find the full service ID
        custom_auth_service_id = next(sid for sid in service_ids if "custom-auth-test" in sid)
        
        # Test the echo service
        echo_service = await api.get_service(custom_auth_service_id)
        result = await echo_service.echo("hello")
        assert result == "Custom auth server says: hello"


@pytest.mark.asyncio
async def test_custom_auth_token_generation_and_validation(custom_auth_server):
    """Test custom token generation and validation through the server."""
    server_url = custom_auth_server.replace("http://", "ws://") + "/ws"
    
    # Connect as anonymous first
    async with connect_to_server(
        {
            "client_id": "test-client-gen",
            "server_url": server_url,
        }
    ) as api:
        # Get the current user's workspace (anonymous user gets a default workspace)
        workspace_name = api.config.workspace
        
        # Generate a token using the server's token generation
        # This should use our custom generate_token function
        custom_token = await api.generate_token(
            config={
                "workspace": workspace_name,
                "expires_in": 3600
            }
        )
        
        # Verify the token has our custom format
        assert custom_token.startswith("CUSTOM:"), f"Token should start with CUSTOM: but got {custom_token[:20]}..."
        
    # Now connect with the custom token
    async with connect_to_server(
        {
            "client_id": "test-client-custom",
            "server_url": server_url,
            "token": custom_token,
        }
    ) as api_with_token:
        # Verify we're connected with the custom token by accessing services
        services = await api_with_token.list_services("public")
        assert len(services) > 0
        
        # We should be able to access the workspace from the token
        assert api_with_token.config.workspace == workspace_name


@pytest.mark.asyncio
async def test_custom_auth_with_workspace_operations(custom_auth_server):
    """Test that custom auth works with workspace operations."""
    server_url = custom_auth_server.replace("http://", "ws://") + "/ws"
    
    # Connect as anonymous
    async with connect_to_server(
        {
            "client_id": "test-workspace-client",
            "server_url": server_url,
        }
    ) as api:
        # Use the anonymous user's default workspace
        ws_name = api.config.workspace
        
        # Generate a custom token for this workspace
        custom_token = await api.generate_token(
            config={
                "workspace": ws_name,
                "expires_in": 1800
            }
        )
        
        # Verify token format
        assert custom_token.startswith("CUSTOM:"), "Should be a custom token"
        parts = custom_token.split(":")
        assert len(parts) == 4, f"Custom token should have 4 parts, got {len(parts)}"
        assert parts[2] == ws_name, f"Token should contain workspace name {ws_name}"
    
    # Use the custom token to connect
    async with connect_to_server(
        {
            "client_id": "test-custom-token-client",
            "server_url": server_url,
            "token": custom_token,
            "workspace": ws_name,
        }
    ) as api_custom:
        # Register a service in the workspace
        service_info = await api_custom.register_service(
            {
                "id": "test-service-custom-auth",
                "name": "Test Service with Custom Auth",
                "type": "test",
                "test": lambda x: x * 2,
            }
        )
        
        # List services to verify it was registered
        services = await api_custom.list_services(ws_name)
        service_ids = [s["id"] for s in services]
        # Check if any service ID contains our test service name
        assert any("test-service-custom-auth" in sid for sid in service_ids)
        
        # Find the full service ID and get the service
        full_service_id = next(sid for sid in service_ids if "test-service-custom-auth" in sid)
        test_service = await api_custom.get_service(full_service_id)
        result = await test_service.test(21)
        assert result == 42


@pytest.mark.asyncio
async def test_custom_auth_with_server_apps(custom_auth_server):
    """Test that server apps functionality works with custom authentication."""
    server_url = custom_auth_server.replace("http://", "ws://") + "/ws"
    
    # Connect with anonymous user
    async with connect_to_server(
        {
            "client_id": "test-apps-client",
            "server_url": server_url,
        }
    ) as api:
        # Generate a custom token
        workspace = api.config.workspace
        custom_token = await api.generate_token(
            config={
                "workspace": workspace,
                "expires_in": 3600
            }
        )
        
        # Verify custom token format
        assert custom_token.startswith("CUSTOM:"), f"Should be custom token: {custom_token[:30]}"
    
    # Connect with custom token
    async with connect_to_server(
        {
            "client_id": "test-apps-auth-client",
            "server_url": server_url,
            "token": custom_token,
        }
    ) as api_auth:
        # Should still be able to list services with auth
        services_with_auth = await api_auth.list_services("public")
        assert len(services_with_auth) > 0
        
        # Register a test service
        await api_auth.register_service(
            {
                "id": "test-app-service",
                "name": "Test App Service",
                "type": "functions",
                "config": {
                    "visibility": "public",
                },
                "test_method": lambda: "Test app service works!",
            }
        )
        
        # List services in the user's workspace to verify service was registered
        user_workspace = api_auth.config.workspace
        updated_services = await api_auth.list_services(user_workspace)
        
        # Check that our test service is in the list
        service_ids = [s["id"] for s in updated_services]
        assert any("test-app-service" in sid for sid in service_ids), f"Test service should be registered in {user_workspace}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])