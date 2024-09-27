"""Test Artifact services."""
import pytest
import requests
from hypha_rpc.websocket_client import connect_to_server

from . import SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio

import pytest
import requests
from hypha_rpc.websocket_client import connect_to_server
from jsonschema import validate, ValidationError

from . import SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


"""Test Artifact services."""
import pytest
import requests
from hypha_rpc.websocket_client import connect_to_server

from . import SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_edit_existing_artifact(minio_server, fastapi_server):
    """Test editing an existing artifact."""
    api = await connect_to_server(
        {"name": "test deploy client", "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection first
    collection_manifest = {
        "id": "edit-test-collection",
        "name": "edit-test-collection",
        "description": "A test collection for editing artifacts",
        "type": "collection",
        "collection": [],
    }
    await artifact_manager.create(
        prefix="collections/edit-test-collection",
        manifest=collection_manifest,
        stage=False,
    )

    # Create an artifact (a dataset in this case) within the collection
    dataset_manifest = {
        "id": "edit-test-dataset",
        "name": "edit-test-dataset",
        "description": "A test dataset to edit",
        "type": "dataset",
        "files": [],
    }

    await artifact_manager.create(
        prefix="collections/edit-test-collection/edit-test-dataset",
        manifest=dataset_manifest,
        stage=True,
    )

    # Commit the dataset artifact (finalize it)
    await artifact_manager.commit(
        prefix="collections/edit-test-collection/edit-test-dataset"
    )

    # Ensure that the dataset appears in the collection's index
    collection = await artifact_manager.read(
        prefix="collections/edit-test-collection"
    )
    assert find_item(collection["collection"], "_id", "edit-test-dataset")

    # Edit the artifact's manifest
    edited_manifest = {
        "id": "edit-test-dataset",
        "name": "edit-test-dataset",
        "description": "Edited description of the test dataset",
        "type": "dataset",
        "files": [{"path": "example.txt"}],  # Adding a file in the edit process
    }

    # Call edit on the artifact
    await artifact_manager.edit(
        prefix="collections/edit-test-collection/edit-test-dataset",
        manifest=edited_manifest,
    )

    # Ensure that the changes are reflected in the manifest (read the artifact in stage mode)
    updated_manifest = await artifact_manager.read(
        prefix="collections/edit-test-collection/edit-test-dataset", stage=True
    )
    assert updated_manifest["description"] == "Edited description of the test dataset"
    assert find_item(updated_manifest["files"], "path", "example.txt")

    # Add a file to the artifact
    source = "file contents of example.txt"
    put_url = await artifact_manager.put_file(
        prefix="collections/edit-test-collection/edit-test-dataset",
        file_path="example.txt",
    )

    # Use the pre-signed URL to upload the file
    response = requests.put(put_url, data=source)
    assert response.ok

    # Commit the artifact again after editing
    await artifact_manager.commit(
        prefix="collections/edit-test-collection/edit-test-dataset"
    )

    # Ensure that the file is included in the manifest
    manifest_data = await artifact_manager.read(
        prefix="collections/edit-test-collection/edit-test-dataset"
    )
    assert find_item(manifest_data["files"], "path", "example.txt")

    # Clean up by deleting the dataset and collection
    await artifact_manager.delete(
        prefix="collections/edit-test-collection/edit-test-dataset"
    )
    await artifact_manager.delete(prefix="collections/edit-test-collection")


async def test_artifact_schema_validation(minio_server, fastapi_server):
    """Test schema validation when committing artifacts to a collection."""
    api = await connect_to_server(
        {"name": "test deploy client", "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Define the schema for the collection
    collection_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "type": {"type": "string", "enum": ["dataset", "model", "application"]},
            "files": {
                "type": "array",
                "items": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        "required": ["id", "name", "description", "type", "files"],
    }

    # Create a collection with the schema
    collection_manifest = {
        "id": "schema-test-collection",
        "name": "Schema Test Collection",
        "description": "A test collection with schema validation",
        "type": "collection",
        "collection_schema": collection_schema,
        "collection": [],
    }

    await artifact_manager.create(
        prefix="collections/schema-test-collection",
        manifest=collection_manifest,
        stage=False,
    )

    # Ensure that the collection exists
    collections = await artifact_manager.list(prefix="collections")
    assert "schema-test-collection" in collections

    # Create a valid dataset artifact that conforms to the schema
    valid_dataset_manifest = {
        "id": "valid-dataset",
        "name": "Valid Dataset",
        "description": "A dataset that conforms to the collection schema",
        "type": "dataset",
        "files": [{"path": "valid_file.txt"}],
    }

    await artifact_manager.create(
        prefix="collections/schema-test-collection/valid-dataset",
        manifest=valid_dataset_manifest,
        stage=True,
    )

    # Add a file to the valid dataset artifact
    source = "file contents of valid_file.txt"
    put_url = await artifact_manager.put_file(
        prefix="collections/schema-test-collection/valid-dataset",
        file_path="valid_file.txt",
    )

    # Use the pre-signed URL to upload the file
    response = requests.put(put_url, data=source)
    assert response.ok

    # Commit the valid dataset artifact (finalize it)
    await artifact_manager.commit(
        prefix="collections/schema-test-collection/valid-dataset"
    )

    # Ensure that the dataset appears in the collection's index
    collection = await artifact_manager.read(
        prefix="collections/schema-test-collection"
    )
    assert find_item(collection["collection"], "_id", "valid-dataset")

    # Now, create an invalid dataset artifact that does not conform to the schema (missing required fields)
    invalid_dataset_manifest = {
        "id": "invalid-dataset",
        "name": "Invalid Dataset",
        # "description" and "files" fields are missing, which are required by the schema
        "type": "dataset",
    }

    await artifact_manager.create(
        prefix="collections/schema-test-collection/invalid-dataset",
        manifest=invalid_dataset_manifest,
        stage=True,
    )

    # Commit should raise a ValidationError due to the schema violation
    with pytest.raises(
        Exception,
        match=r".*ValidationError: 'description' is a required property.*",
    ):
        await artifact_manager.commit(
            prefix="collections/schema-test-collection/invalid-dataset"
        )

    # Clean up by deleting the artifacts and the collection
    await artifact_manager.delete(
        prefix="collections/schema-test-collection/valid-dataset"
    )
    await artifact_manager.delete(
        prefix="collections/schema-test-collection/invalid-dataset"
    )
    await artifact_manager.delete(prefix="collections/schema-test-collection")


async def test_artifact_manager_with_collection(minio_server, fastapi_server):
    """Test artifact controller with collections."""
    api = await connect_to_server(
        {"name": "test deploy client", "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection
    collection_manifest = {
        "id": "test-collection",
        "name": "test-collection",
        "description": "A test collection",
        "type": "collection",
        "collection": [],
    }

    await artifact_manager.create(
        prefix="collections/test-collection", manifest=collection_manifest, stage=False
    )

    # Ensure that the collection exists
    collections = await artifact_manager.list(prefix="collections")
    assert "test-collection" in collections

    # Create an artifact (a dataset in this case) within the collection
    dataset_manifest = {
        "id": "test-dataset",
        "name": "test-dataset",
        "description": "A test dataset in the collection",
        "type": "dataset",
        "files": [],
    }

    await artifact_manager.create(
        prefix="collections/test-collection/test-dataset",
        manifest=dataset_manifest,
        stage=True,
    )

    # Add a file to the artifact (test file)
    source = "file contents of test.txt"
    put_url = await artifact_manager.put_file(
        prefix="collections/test-collection/test-dataset", file_path="test.txt"
    )

    # Use the pre-signed URL to upload the file
    response = requests.put(put_url, data=source)
    assert response.ok

    # Ensure that the file is included in the manifest
    manifest_data = await artifact_manager.read(
        prefix="collections/test-collection/test-dataset", stage=True
    )
    assert find_item(manifest_data["files"], "path", "test.txt")

    # Commit the artifact (finalize it)
    await artifact_manager.commit(prefix="collections/test-collection/test-dataset")

    # Ensure that the dataset appears in the collection's index
    collection = await artifact_manager.read(prefix="collections/test-collection")
    assert find_item(collection["collection"], "_id", "test-dataset")

    # Ensure that the manifest.yaml is finalized and the artifact is validated
    artifacts = await artifact_manager.list(prefix="collections/test-collection")
    assert "test-dataset" in artifacts

    # Retrieve the file via the get_file method
    get_url = await artifact_manager.get_file(
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

    await artifact_manager.create(
        prefix="collections/test-collection/test-dataset",
        manifest=new_dataset_manifest,
        overwrite=True,
        stage=False,
    )

    # Ensure that the old manifest has been overwritten
    manifest_data = await artifact_manager.read(
        prefix="collections/test-collection/test-dataset"
    )
    assert manifest_data["description"] == "Overwritten test dataset in collection"
    assert len(manifest_data["files"]) == 0  # files should have been cleared

    # Remove the dataset artifact
    await artifact_manager.delete(prefix="collections/test-collection/test-dataset")

    # Ensure that the dataset artifact is removed
    artifacts = await artifact_manager.list(prefix="collections/test-collection")
    assert "test-dataset" not in artifacts

    # Ensure the collection is updated after removing the dataset
    collection = await artifact_manager.read(prefix="collections/test-collection")
    assert not find_item(collection["collection"], "_id", "test-dataset")

    # Clean up by deleting the collection
    await artifact_manager.delete(prefix="collections/test-collection")


async def test_artifact_edge_cases_with_collection(minio_server, fastapi_server):
    """Test artifact edge cases with collections."""
    api = await connect_to_server(
        {"name": "test deploy client", "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection first
    collection_manifest = {
        "id": "edge-case-collection",
        "name": "edge-case-collection",
        "description": "A test collection for edge cases",
        "type": "collection",
        "collection": [],
    }
    await artifact_manager.create(
        prefix="collections/edge-case-collection",
        manifest=collection_manifest,
        stage=False,
    )

    # Try to create an artifact that already exists within the collection without overwriting
    manifest = {
        "id": "edge-case-dataset",
        "name": "edge-case-dataset",
        "description": "Edge case test dataset",
        "type": "dataset",
        "files": [],
    }

    # Create the artifact first
    await artifact_manager.create(
        prefix="collections/edge-case-collection/edge-case-dataset",
        manifest=manifest,
        stage=True,
    )

    # Attempt to create the same artifact without overwrite
    with pytest.raises(
        Exception,
        match=r".*FileExistsError: Artifact under prefix .*",
    ):
        await artifact_manager.create(
            prefix="collections/edge-case-collection/edge-case-dataset",
            manifest=manifest,
            overwrite=False,
            stage=True,
        )

    # Add a file to the artifact
    put_url = await artifact_manager.put_file(
        prefix="collections/edge-case-collection/edge-case-dataset",
        file_path="example.txt",
    )
    source = "some content"
    response = requests.put(put_url, data=source)
    assert response.ok

    # Commit the artifact
    await artifact_manager.commit(
        prefix="collections/edge-case-collection/edge-case-dataset"
    )

    # Ensure that the collection index is updated
    collection = await artifact_manager.read(
        prefix="collections/edge-case-collection"
    )
    assert find_item(collection["collection"], "_id", "edge-case-dataset")

    # Test validation without uploading a file
    incomplete_manifest = {
        "id": "incomplete-dataset",
        "name": "incomplete-dataset",
        "description": "This dataset is incomplete",
        "type": "dataset",
        "files": [{"path": "missing.txt"}],
    }

    await artifact_manager.create(
        prefix="collections/edge-case-collection/incomplete-dataset",
        manifest=incomplete_manifest,
        stage=True,
    )

    # Commit should raise an error due to the missing file
    with pytest.raises(
        Exception,
        match=r".*FileNotFoundError: .*",
    ):
        await artifact_manager.commit(
            prefix="collections/edge-case-collection/incomplete-dataset"
        )

    # Clean up the artifacts
    await artifact_manager.delete(
        prefix="collections/edge-case-collection/edge-case-dataset"
    )
    await artifact_manager.delete(
        prefix="collections/edge-case-collection/incomplete-dataset"
    )
    await artifact_manager.delete(prefix="collections/edge-case-collection")
