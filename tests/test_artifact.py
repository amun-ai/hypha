"""Test Artifact services."""
import pytest
import requests
from hypha_rpc.websocket_client import connect_to_server

from . import SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_artifact_controller_with_collection(minio_server, fastapi_server):
    """Test artifact controller with collections."""
    api = await connect_to_server(
        {"name": "test deploy client", "server_url": SERVER_URL}
    )
    artifact_controller = await api.get_service("public/*:artifact")

    # Create a collection
    collection_manifest = {
        "id": "test-collection",
        "name": "test-collection",
        "description": "A test collection",
        "type": "collection",
        "collection": [],
    }

    await artifact_controller.create(prefix="collections/test-collection", manifest=collection_manifest)

    # Ensure that the collection exists
    collections = await artifact_controller.list(prefix="collections", pending=True)
    item = find_item(collections, "name", "test-collection")
    assert item and item["pending"] == True

    # Create an artifact (a dataset in this case) within the collection
    dataset_manifest = {
        "id": "test-dataset",
        "name": "test-dataset",
        "description": "A test dataset in the collection",
        "type": "dataset",
        "files": [],
    }

    await artifact_controller.create(prefix="collections/test-collection/test-dataset", manifest=dataset_manifest)

    # Ensure that the dataset appears in the collection's index
    collection = await artifact_controller.read(prefix="collections/test-collection")
    assert find_item(collection["collection"], "_id", "test-collection/test-dataset")

    # Add a file to the artifact (test file)
    source = "file contents of test.txt"
    put_url = await artifact_controller.put_file(
        prefix="collections/test-collection/test-dataset", file_path="test.txt"
    )

    # Use the pre-signed URL to upload the file
    response = requests.put(put_url, data=source)
    assert response.ok

    # Ensure that the file is included in the manifest
    manifest_data = await artifact_controller.read(prefix="collections/test-collection/test-dataset")
    assert find_item(manifest_data["files"], "path", "test.txt")

    # Validate the artifact (finalize it)
    await artifact_controller.validate(prefix="collections/test-collection/test-dataset")

    # Ensure that the manifest.yaml is finalized and the artifact is validated
    artifacts = await artifact_controller.list(prefix="collections/test-collection")
    assert find_item(artifacts, "name", "test-collection/test-dataset")

    # Retrieve the file via the get_file method
    get_url = await artifact_controller.get_file(
        prefix="collections/test-collection/test-dataset", path="test.txt"
    )
    response = requests.get(get_url)
    assert response.ok
    assert response.text == source

    # Test overwriting the existing artifact within the collection
    new_dataset_manifest = {
        "id": "test-dataset",
        "name": "test-dataset",
        "description": "Overwritten test dataset in collection",
        "type": "dataset",
        "files": [],
    }

    await artifact_controller.create(
        prefix="collections/test-collection/test-dataset", manifest=new_dataset_manifest, overwrite=True
    )

    # Ensure that the old manifest has been overwritten
    manifest_data = await artifact_controller.read(prefix="collections/test-collection/test-dataset")
    assert manifest_data["description"] == "Overwritten test dataset in collection"
    assert len(manifest_data["files"]) == 0  # files should have been cleared

    # Remove the dataset artifact
    await artifact_controller.delete(prefix="collections/test-collection/test-dataset")

    # Ensure that the dataset artifact is removed
    artifacts = await artifact_controller.list(prefix="collections/test-collection")
    assert not find_item(artifacts, "name", "test-collection/test-dataset")

    # Ensure the collection is updated after removing the dataset
    collection = await artifact_controller.read(prefix="collections/test-collection")
    assert not find_item(collection["collection"], "_id", "test-collection/test-dataset")

    # Clean up by deleting the collection
    await artifact_controller.delete(prefix="collections/test-collection")


async def test_artifact_edge_cases_with_collection(minio_server, fastapi_server):
    """Test artifact edge cases with collections."""
    api = await connect_to_server(
        {"name": "test deploy client", "server_url": SERVER_URL}
    )
    artifact_controller = await api.get_service("public/*:artifact")

    # Create a collection first
    collection_manifest = {
        "id": "edge-case-collection",
        "name": "edge-case-collection",
        "description": "A test collection for edge cases",
        "type": "collection",
        "collection": [],
    }
    await artifact_controller.create(prefix="collections/edge-case-collection", manifest=collection_manifest)

    # Try to create an artifact that already exists within the collection without overwriting
    manifest = {
        "id": "edge-case-dataset",
        "name": "edge-case-dataset",
        "description": "Edge case test dataset",
        "type": "dataset",
        "files": [],
    }

    # Create the artifact first
    await artifact_controller.create(
        prefix="collections/edge-case-collection/edge-case-dataset", manifest=manifest
    )

    # Attempt to create the same artifact without overwrite
    with pytest.raises(
        Exception,
        match=r".*FileExistsError: Artifact under prefix .*",
    ):
        await artifact_controller.create(
            prefix="collections/edge-case-collection/edge-case-dataset", manifest=manifest, overwrite=False
        )

    # Add a file to the artifact
    put_url = await artifact_controller.put_file(
        prefix="collections/edge-case-collection/edge-case-dataset", file_path="example.txt"
    )
    source = "some content"
    response = requests.put(put_url, data=source)
    assert response.ok

    # Validate the artifact
    await artifact_controller.validate(prefix="collections/edge-case-collection/edge-case-dataset")

    # Ensure that the collection index is updated
    collection = await artifact_controller.read(prefix="collections/edge-case-collection")
    assert find_item(collection["collection"], "_id", "edge-case-collection/edge-case-dataset")

    # Test validation without uploading a file
    incomplete_manifest = {
        "id": "incomplete-dataset",
        "name": "incomplete-dataset",
        "description": "This dataset is incomplete",
        "type": "dataset",
        "files": [{"path": "missing.txt"}],
    }

    await artifact_controller.create(
        prefix="collections/edge-case-collection/incomplete-dataset", manifest=incomplete_manifest
    )

    # Validate should raise an error due to the missing file
    with pytest.raises(
        Exception,
        match=r".*FileNotFoundError: .*",
    ):
        await artifact_controller.validate(prefix="collections/edge-case-collection/incomplete-dataset")

    # Clean up the artifacts
    await artifact_controller.delete(prefix="collections/edge-case-collection/edge-case-dataset")
    await artifact_controller.delete(prefix="collections/edge-case-collection/incomplete-dataset")
    await artifact_controller.delete(prefix="collections/edge-case-collection")


def find_item(items, key, value):
    """Utility function to find an item in a list of dictionaries."""
    return next((item for item in items if item.get(key) == value), None)
