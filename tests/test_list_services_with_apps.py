"""Test list_services with include_app_services parameter."""

import pytest
import asyncio
from hypha_rpc import connect_to_server
from . import SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_list_services_with_app_services_simple(
    minio_server, fastapi_server, test_user_token
):
    """Test that list_services can discover services from installed application manifests."""
    
    # Connect to server
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    
    # Get the apps controller
    apps = await api.get_service("public/server-apps")
    
    # Create a test application with services in its manifest
    # Use stage=True to avoid commit and app start
    app_info = await apps.install(
        source="<html><body>Test App</body></html>",
        manifest={
            "name": "Test Discovery App",
            "type": "window",
            "version": "1.0.0",
            "services": [
                {
                    "id": f"{api.config.workspace}/*:test-service-1",
                    "name": "Test Service 1",
                    "type": "computation",
                    "description": "Test service for discovery",
                    "config": {
                        "visibility": "public"
                    }
                },
                {
                    "id": f"{api.config.workspace}/*:test-service-2",
                    "name": "Test Service 2",
                    "type": "data-processing",
                    "description": "Another test service",
                    "config": {
                        "visibility": "protected",
                        "authorized_workspaces": [api.config.workspace]
                    }
                },
            ]
        },
        stage=True,  # Don't commit, just stage the artifact
        overwrite=True
    )
    
    app_id = app_info["id"] if isinstance(app_info, dict) else app_info.id
    
    try:
        # Commit the app to save it in artifacts
        await apps.commit_app(app_id, wait_for_service=False)
        
        # Test 1: List services WITHOUT include_app_services (should not find app services)
        services_without_apps = await api.list_services(
            query={"workspace": api.config.workspace},
            include_app_services=False
        )
        
        # Should not find services from the app manifest
        test_svc_1 = find_item(
            services_without_apps,
            lambda s: s.get("id", "").endswith(":test-service-1")
        )
        assert test_svc_1 is None, "Should not find test-service-1 when include_app_services=False"
        
        # Test 2: List services WITH include_app_services (should find app services)
        services_with_apps = await api.list_services(
            query={"workspace": api.config.workspace},
            include_app_services=True
        )
        
        # Debug: Print all services found
        print(f"Found {len(services_with_apps)} services with include_app_services=True")
        for svc in services_with_apps:
            print(f"  - Service: {svc.get('id')}, source: {svc.get('_source', 'active')}")
        
        # Should find the public service
        test_svc_1 = find_item(
            services_with_apps, 
            lambda s: s.get("id", "").endswith(":test-service-1")
        )
        assert test_svc_1 is not None, f"Should find test-service-1. Found services: {[s.get('id') for s in services_with_apps]}"
        assert test_svc_1.get("name") == "Test Service 1"
        assert test_svc_1.get("_source") == "app_manifest", "Should indicate source is app manifest"
        assert test_svc_1.get("app_id") == app_id, f"Should have app_id={app_id}"
        
        # Should find the protected service
        test_svc_2 = find_item(
            services_with_apps,
            lambda s: s.get("id", "").endswith(":test-service-2")
        )
        assert test_svc_2 is not None, "Should find test-service-2"
        assert test_svc_2.get("name") == "Test Service 2"
        assert test_svc_2.get("_source") == "app_manifest"
        
        # Test 3: Query with type filter
        computation_services = await api.list_services(
            query={"workspace": api.config.workspace, "type": "computation"},
            include_app_services=True
        )
        
        # Should find test-service-1 which has type "computation"
        test_svc_1_filtered = find_item(
            computation_services,
            lambda s: s.get("id", "").endswith(":test-service-1")
        )
        assert test_svc_1_filtered is not None, "Should find test-service-1 with type filter"
        
        # Should not find test-service-2 which has type "data-processing"
        test_svc_2_filtered = find_item(
            computation_services,
            lambda s: s.get("id", "").endswith(":test-service-2")
        )
        assert test_svc_2_filtered is None, "Should not find test-service-2 with computation type filter"
        
        print("✅ All tests passed successfully!")
        
    finally:
        # Clean up - uninstall the app
        try:
            await apps.uninstall(app_id)
        except:
            pass  # Ignore cleanup errors
    
    await api.disconnect()