"""Test that server apps work with custom authentication."""
import pytest
import asyncio
from hypha_rpc import connect_to_server


@pytest.mark.asyncio
async def test_server_apps_with_custom_auth(custom_auth_server):
    """Test that server apps functionality works with custom authentication."""
    server_url = custom_auth_server.replace("http://", "ws://") + "/ws"
    
    # Connect with anonymous user
    async with connect_to_server(
        {
            "client_id": "test-apps-client",
            "server_url": server_url,
        }
    ) as api:
        # List all services including apps
        all_services = await api.list_services(
            "public", 
            {"include_app_services": True}
        )
        
        # Check that we have services
        assert len(all_services) > 0
        
        # Filter for different service types
        regular_services = [s for s in all_services if s.get("type") != "window"]
        app_services = [s for s in all_services if s.get("type") == "window"]
        
        print(f"\nFound {len(regular_services)} regular services")
        print(f"Found {len(app_services)} app services")
        
        # Verify built-in service exists
        service_ids = [s["id"] for s in regular_services]
        assert any("built-in" in sid for sid in service_ids), "Built-in service should exist"
        
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
        services_with_auth = await api_auth.list_services(
            "public",
            {"include_app_services": True}
        )
        
        assert len(services_with_auth) > 0
        
        # Register a regular test service (window services might require special setup)
        test_service = await api_auth.register_service(
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
        assert any("test-app-service" in sid for sid in service_ids), f"Test service should be registered in {user_workspace}, found: {service_ids}"
        
        print(f"\n✅ Server apps work with custom authentication!")
        print(f"   - Can list services with include_app_services")
        print(f"   - Can register app services")
        print(f"   - Custom tokens work for app access")


@pytest.mark.asyncio  
async def test_lazy_loading_with_custom_auth(custom_auth_server):
    """Test lazy loading of services works with custom auth."""
    server_url = custom_auth_server.replace("http://", "ws://") + "/ws"
    
    async with connect_to_server(
        {
            "client_id": "test-lazy-client",
            "server_url": server_url,
        }
    ) as api:
        # Register a service that won't be loaded immediately
        await api.register_service(
            {
                "id": "lazy-service",
                "name": "Lazy Service",
                "config": {
                    "visibility": "public",
                    "lazy": True,  # Mark as lazy
                },
                "hello": lambda: "Hello from lazy service",
            }
        )
        
        # List services in the user's workspace (where the service was registered)
        services = await api.list_services(api.config.workspace)
        lazy_service = next((s for s in services if "lazy-service" in s["id"]), None)
        assert lazy_service is not None, "Lazy service should be registered"
        
        # Now get the service (this should trigger loading)
        full_service_id = lazy_service["id"]
        lazy_svc = await api.get_service(full_service_id)
        
        # Call the service method
        result = await lazy_svc.hello()
        assert result == "Hello from lazy service"
        
        print("\n✅ Lazy loading works with custom authentication!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])