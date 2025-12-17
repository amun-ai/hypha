"""Test ArtifactController service registration."""

import pytest
import pytest_asyncio
import requests
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


@pytest.mark.asyncio
async def test_batch_put_file(
    minio_server, fastapi_server_redis_1, test_user_token
):
    """Test batch put_file operations with list input."""
    api = await connect_to_server(
        {"name": "test-client-batch-put", "server_url": SERVER_URL_REDIS_1, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a test collection
    collection = await artifact_manager.create(
        alias="batch-put-test",
        type="collection",
        manifest={"name": "Batch Put Test"},
        config={"permissions": {"*": "rw+"}},
        version="stage",
    )

    try:
        dataset = await artifact_manager.create(
            alias="batch-dataset",
            type="dataset",
            parent_id=collection.id,
            manifest={"name": "Batch Dataset"},
            version="stage",
        )

        # Test batch put_file with list input
        file_paths = ["file1.txt", "file2.txt", "file3.txt"]
        urls = await artifact_manager.put_file(
            artifact_id=dataset.id,
            file_path=file_paths,  # List input
        )

        # Verify we got a dict back
        assert isinstance(urls, dict), f"Expected dict, got {type(urls)}"
        assert len(urls) == 3, f"Expected 3 URLs, got {len(urls)}"

        # Verify all file paths are in the result
        for path in file_paths:
            assert path in urls, f"Missing URL for {path}"
            assert isinstance(urls[path], str), f"URL for {path} is not a string"

        # Upload files to verify URLs work
        for i, path in enumerate(file_paths):
            response = requests.put(urls[path], data=f"content {i}")
            assert response.ok, f"Failed to upload {path}: {response.text}"

        # Commit and verify
        await artifact_manager.commit(dataset.id)
        files = await artifact_manager.list_files(dataset.id)
        assert len(files) == 3, f"Expected 3 files, got {len(files)}"

    finally:
        await artifact_manager.delete(collection.id)
        await api.disconnect()


@pytest.mark.asyncio
async def test_batch_get_file(
    minio_server, fastapi_server_redis_1, test_user_token
):
    """Test batch get_file operations with list input."""
    api = await connect_to_server(
        {"name": "test-client-batch-get", "server_url": SERVER_URL_REDIS_1, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a test collection and dataset with files
    # Use the same pattern as test_batch_put_file which works
    collection = await artifact_manager.create(
        alias="batch-get-test-collection",
        type="collection",
        manifest={"name": "Batch Get Test"},
        config={"permissions": {"*": "rw+"}},
    )

    try:
        dataset = await artifact_manager.create(
            alias="batch-get-dataset",
            type="dataset",
            parent_id=collection.id,
            manifest={"name": "Batch Get Dataset"},
            version="stage",
        )

        # Upload some test files
        file_paths = ["data1.txt", "data2.txt", "data3.txt"]
        put_urls = await artifact_manager.put_file(
            artifact_id=dataset.id,
            file_path=file_paths,
        )

        for i, path in enumerate(file_paths):
            response = requests.put(put_urls[path], data=f"test data {i}")
            assert response.ok

        await artifact_manager.commit(dataset.id)

        # Test batch get_file with list input
        get_urls = await artifact_manager.get_file(
            artifact_id=dataset.id,
            file_path=file_paths,  # List input
        )

        # Verify we got a dict back
        assert isinstance(get_urls, dict), f"Expected dict, got {type(get_urls)}"
        assert len(get_urls) == 3, f"Expected 3 URLs, got {len(get_urls)}"

        # Verify all file paths are in the result
        for path in file_paths:
            assert path in get_urls, f"Missing URL for {path}"
            assert isinstance(get_urls[path], str), f"URL for {path} is not a string"

        # Download files to verify URLs work
        for i, path in enumerate(file_paths):
            response = requests.get(get_urls[path])
            assert response.ok, f"Failed to download {path}: {response.text}"
            assert response.text == f"test data {i}", f"Content mismatch for {path}"
    except Exception as exp:
        raise exp
    finally:
        await api.disconnect()