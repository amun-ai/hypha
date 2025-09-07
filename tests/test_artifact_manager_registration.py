"""Test ArtifactController service registration."""

import pytest
import pytest_asyncio
from hypha_rpc import connect_to_server
from . import SERVER_URL_REDIS_1


@pytest.mark.asyncio
async def test_artifact_manager_service_available(fastapi_server_redis_1):
    """Test that artifact-manager service is properly registered and accessible."""
    
    # Connect to the server (without needing a token for this test)
    async with connect_to_server({
        "name": "test artifact controller registration", 
        "server_url": SERVER_URL_REDIS_1,
    }) as api:
        
        # Try to get the artifact-manager service
        try:
            artifact_manager = await api.get_service("public/artifact-manager")
            print(f"SUCCESS: Found artifact-manager service: {type(artifact_manager)}")
            
            # Verify it has the expected methods
            expected_methods = ['create', 'read', 'edit', 'delete', 'commit', 'list', 'put_file']
            for method in expected_methods:
                assert hasattr(artifact_manager, method), f"Missing method: {method}"
            
            print("SUCCESS: All expected methods are present on artifact-manager service")
            
        except KeyError as e:
            # List all available services to debug what's registered
            services = await api.list_services()
            print(f"Available services ({len(services)}):")
            for service in services:
                service_id = service.get('id', 'unknown')
                service_name = service.get('name', 'N/A')
                print(f"  - {service_id}: {service_name}")
            
            # Check specifically for public services
            public_services = [s for s in services if s.get('id', '').startswith('public/')]
            print(f"\nPublic services ({len(public_services)}):")
            for service in public_services:
                service_id = service.get('id', 'unknown')
                service_name = service.get('name', 'N/A')
                print(f"  - {service_id}: {service_name}")
            
            # Look for any services containing 'artifact' in the name
            artifact_services = [s for s in services if 'artifact' in s.get('id', '').lower() or 'artifact' in s.get('name', '').lower()]
            print(f"\nServices with 'artifact' in name ({len(artifact_services)}):")
            for service in artifact_services:
                service_id = service.get('id', 'unknown')
                service_name = service.get('name', 'N/A')
                print(f"  - {service_id}: {service_name}")
                
            # This should fail the test
            pytest.fail(f"Could not get artifact-manager service: {e}. Available services listed above.")