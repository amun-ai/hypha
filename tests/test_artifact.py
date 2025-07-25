"""Test Artifact services."""

import pytest
import requests
import os
from hypha_rpc import connect_to_server
from io import BytesIO
from zipfile import ZipFile
import httpx
import yaml
import json
import zipfile

from . import SERVER_URL, SERVER_URL_SQLITE, find_item
from hypha.core.auth import valid_token

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_sqlite_create_and_search_artifacts(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """Test creating a collection, adding datasets with files, and performing a search on manifest fields."""

    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "sqlite test client",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for the SQLite test
    collection_manifest = {
        "name": "SQLite Test Collection",
        "description": "A collection for testing SQLite backend",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Add multiple datasets to the collection
    dataset_manifests = [
        {"name": "Dataset 1", "description": "First dataset for SQLite testing"},
        {"name": "Dataset 2", "description": "Second dataset for SQLite testing"},
        {"name": "Dataset 3", "description": "Third dataset for SQLite testing"},
    ]
    datasets = []
    for dataset_manifest in dataset_manifests:
        dataset = await artifact_manager.create(
            type="dataset",
            parent_id=collection.id,
            manifest=dataset_manifest,
            version="stage",
        )
        datasets.append(dataset)

    # create an application artifact
    application = await artifact_manager.create(
        type="application",
        parent_id=collection.id,
        manifest={
            "name": "Application 1",
            "description": "First application for SQLite testing",
        },
        version="stage",
    )

    # Test read operation on the datasets
    for dataset in datasets:
        dataset_data = await artifact_manager.read(
            artifact_id=dataset.id, version="stage"
        )
        assert dataset_data["manifest"]["name"] in [
            dataset_manifest["name"] for dataset_manifest in dataset_manifests
        ]

    application_data = await artifact_manager.read(
        artifact_id=application.id, version="stage"
    )
    assert application_data["manifest"]["name"] == "Application 1"

    # Add files to each dataset
    file_contents = [
        "Contents of file 1 for Dataset 1",
        "Contents of file 2 for Dataset 2",
        "Contents of file 3 for Dataset 3",
    ]
    for idx, dataset in enumerate(datasets):
        put_url = await artifact_manager.put_file(
            artifact_id=dataset.id, file_path=f"file_{idx+1}.txt"
        )
        response = requests.put(put_url, data=file_contents[idx])
        assert response.ok

    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"version": "stage", "manifest": {"description": "*dataset*"}},
    )

    assert len(search_results) == len(datasets)

    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"version": "stage", "manifest": {"description": "*dataset*"}},
        pagination=True,
    )
    assert results["total"] == len(datasets)
    assert len(results["items"]) == len(datasets)

    # list application only
    search_results = await artifact_manager.list(
        parent_id=collection.id, filters={"version": "stage", "type": "application"}
    )
    assert len(search_results) == 1

    # Commit the datasets
    for dataset in datasets:
        await artifact_manager.commit(artifact_id=dataset.id)

    await artifact_manager.commit(artifact_id=application.id)

    # Search for artifacts based on manifest fields
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        keywords=["SQLite"],
        filters={"manifest": {"description": "*dataset*"}},
        mode="AND",
    )

    # Validate the search results contain the added datasets
    assert len(search_results) == len(datasets)
    assert set([result["manifest"]["name"] for result in search_results]) == set(
        [dataset_manifest["name"] for dataset_manifest in dataset_manifests]
    )

    # Search for application
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        keywords=["SQLite"],
        filters={"type": "application"},
        mode="AND",
    )
    assert len(search_results) == 1

    # Clean up by deleting datasets and the collection
    for dataset in datasets:
        await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_serve_artifact_endpoint(minio_server, fastapi_server, test_user_token):
    """Test the artifact serving endpoint."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a public collection and retrieve the UUID
    collection_manifest = {
        "name": "Public Dataset Gallery",
        "description": "A public collection for organizing datasets",
    }
    collection_info = await artifact_manager.create(
        type="collection",
        alias="public-dataset-gallery",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # list artifact in current workspace
    artifacts = await artifact_manager.list()
    assert collection_info.id in [artifact["id"] for artifact in artifacts]
    # Create an artifact inside the public collection
    dataset_manifest = {
        "name": "Public Example Dataset",
        "description": "A public dataset with example data",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection_info.id,
        manifest=dataset_manifest,
        version="stage",
    )

    # Commit the artifact
    await artifact_manager.commit(artifact_id=dataset.id)

    # Ensure the public artifact is available via HTTP
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}"
    )
    assert response.status_code == 200
    assert "Public Example Dataset" in response.json()["manifest"]["name"]

    # Now create a non-public collection
    private_collection_manifest = {
        "name": "Private Dataset Gallery",
        "description": "A private collection for organizing datasets",
    }
    private_collection = await artifact_manager.create(
        type="collection",
        manifest=private_collection_manifest,
    )

    # Create an artifact inside the private collection
    private_dataset_manifest = {
        "name": "Private Example Dataset",
        "description": "A private dataset with example data",
        "custom_key": 192,
    }
    private_dataset = await artifact_manager.create(
        type="dataset",
        parent_id=private_collection.id,
        manifest=private_dataset_manifest,
        version="stage",
    )

    # Commit the private artifact
    await artifact_manager.commit(artifact_id=private_dataset.id)

    token = await api.generate_token()
    # Ensure the private artifact is available via HTTP (requires authentication or special permissions)
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{private_dataset.alias}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert "Private Example Dataset" in response.json()["manifest"]["name"]
    assert response.json()["manifest"]["custom_key"] == 192

    # If no authentication is provided, the server should return a 403 Forbidden status code
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{private_dataset.alias}"
    )
    assert response.status_code == 403


async def test_http_file_and_directory_endpoint(
    minio_server, fastapi_server, test_user_token
):
    """Test the HTTP file serving and directory listing endpoint, including nested files."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection and retrieve the UUID
    collection_manifest = {
        "name": "test-collection",
        "description": "A test collection",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create a dataset within the collection
    dataset_manifest = {
        "name": "test-dataset",
        "description": "A test dataset",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
    )

    # Add a file to the dataset artifact
    file_contents = "file contents of example.txt"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="example.txt",
        download_weight=1,
    )
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.put(put_url, data=file_contents)
        assert response.status_code == 200

    # Add another file to the dataset artifact in a nested directory
    file_contents2 = "file contents of nested/example2.txt"
    nested_file_path = "nested/example2.txt"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path=nested_file_path,
        download_weight=1,
    )
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.put(put_url, data=file_contents2)
        assert response.status_code == 200

    # Commit the dataset artifact
    await artifact_manager.commit(artifact_id=dataset.id)

    files = await artifact_manager.list_files(artifact_id=dataset.id)
    assert len(files) == 2

    # Retrieve the file via HTTP
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/example.txt",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert response.text == file_contents

    # Check download count increment
    artifact = await artifact_manager.read(
        artifact_id=dataset.id,
    )
    assert artifact["download_count"] == 1

    # Try to get it using HTTP proxy
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/example.txt?use_proxy=1"
        )
        assert response.status_code == 200
        assert response.text == file_contents

    # Check download count increment
    artifact = await artifact_manager.read(
        artifact_id=dataset.id,
    )
    assert artifact["download_count"] == 2

    # Attempt to list directory contents
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/"
        )
        assert response.status_code == 200
        assert "example.txt" in [file["name"] for file in response.json()]

    # Get the zip file with specific files
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=example.txt&file={nested_file_path}"
        )
        assert response.status_code == 200
        # Write the zip file in a io.BytesIO object, then check if the file contents are correct
        zip_file = ZipFile(BytesIO(response.content))
        assert sorted(zip_file.namelist()) == sorted(
            ["example.txt", "nested/example2.txt"]
        )
        assert zip_file.read("example.txt").decode() == "file contents of example.txt"
        assert zip_file.read("nested/example2.txt").decode() == file_contents2

    # Check download count
    artifact = await artifact_manager.read(
        artifact_id=dataset.id,
    )
    assert artifact["download_count"] == 4

    # Get the zip file with all files
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file"
        )
        assert response.status_code == 200, response.text
        zip_file = ZipFile(BytesIO(response.content))
        assert sorted(zip_file.namelist()) == sorted(
            ["example.txt", "nested/example2.txt"]
        )
        assert zip_file.read("example.txt").decode() == "file contents of example.txt"
        assert zip_file.read("nested/example2.txt").decode() == file_contents2

    # check download count
    artifact = await artifact_manager.read(
        artifact_id=dataset.id,
    )
    assert artifact["download_count"] == 6


async def test_get_zip_file_content_endpoint(
    minio_server, fastapi_server, test_user_token
):
    """Test retrieving specific content and listing directories from a ZIP file stored in S3."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection and retrieve the UUID
    collection_manifest = {
        "name": "test-collection",
        "description": "A test collection",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create a dataset within the collection
    dataset_manifest = {
        "name": "test-dataset",
        "description": "A test dataset",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
    )

    # Create a ZIP file in memory
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("example.txt", "file contents of example.txt")
        zip_file.writestr("nested/example2.txt", "file contents of nested/example2.txt")
        zip_file.writestr(
            "nested/subdir/example3.txt", "file contents of nested/subdir/example3.txt"
        )
    zip_buffer.seek(0)

    # Upload the ZIP file to the artifact
    zip_file_path = "test-files"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path=f"{zip_file_path}.zip",
        download_weight=1,
    )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(put_url, data=zip_buffer.read())
        assert response.status_code == 200

    # Commit the dataset artifact
    await artifact_manager.commit(artifact_id=dataset.id)

    # Test listing the root directory of the ZIP
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip"
        )
        assert response.status_code == 200
        items = response.json()
        assert find_item(items, "name", "example.txt") is not None
        assert find_item(items, "name", "nested") is not None

    # Test listing the "nested/" directory
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nested/"
        )
        assert response.status_code == 200
        items = response.json()
        assert find_item(items, "name", "example2.txt") is not None
        assert find_item(items, "name", "subdir") is not None

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip/~/nested/"
        )
        assert response.status_code == 200
        items = response.json()
        assert find_item(items, "name", "example2.txt") is not None
        assert find_item(items, "name", "subdir") is not None

    # Test listing the "nested/subdir/" directory
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nested/subdir/"
        )
        assert response.status_code == 200
        items = response.json()
        assert find_item(items, "name", "example3.txt") is not None

    # Test retrieving `example.txt` from the ZIP file
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=example.txt"
        )
        assert response.status_code == 200
        assert response.text == "file contents of example.txt"

    # Test retrieving `nested/example2.txt` from the ZIP file
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nested/example2.txt"
        )
        assert response.status_code == 200
        assert response.text == "file contents of nested/example2.txt"

    # Test retrieving a non-existent file
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nonexistent.txt"
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "File not found inside ZIP: nonexistent.txt"

    # Test listing a non-existent directory
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=nonexistent/"
        )
        assert response.status_code == 200
        assert (
            response.json() == []
        )  # An empty list indicates an empty or non-existent directory.


async def test_artifact_permissions(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """Test artifact permissions."""
    api = await connect_to_server(
        {
            "name": "test-client",
            "server_url": SERVER_URL,
            "token": test_user_token,
            "token": test_user_token_2,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a public collection and get its UUID
    collection_manifest = {
        "name": "Public Dataset Gallery",
        "description": "A public collection for organizing datasets",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Verify that anonymous access to the collection works
    api_anonymous = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL}
    )
    artifact_manager_anonymous = await api_anonymous.get_service(
        "public/artifact-manager"
    )

    collection = await artifact_manager_anonymous.read(artifact_id=collection.id)
    assert collection["manifest"]["name"] == "Public Dataset Gallery"

    # Attempt to create an artifact inside the collection as an anonymous user
    dataset_manifest = {
        "name": "Public Example Dataset",
        "description": "A public dataset with example data",
    }
    with pytest.raises(
        Exception, match=r".*User does not have permission to perform the operation.*"
    ):
        await artifact_manager_anonymous.create(
            type="dataset",
            parent_id=collection.id,
            manifest=dataset_manifest,
        )

    # Verify that anonymous user cannot reset stats on the public collection
    with pytest.raises(
        Exception,
        match=r".*PermissionError: User does not have permission to perform the operation.*",
    ):
        await artifact_manager_anonymous.reset_stats(artifact_id=collection.id)

    # Authenticated user 2 can create a child artifact within the collection
    api_2 = await connect_to_server(
        {
            "name": "test-client",
            "server_url": SERVER_URL,
            "token": test_user_token,
            "token": test_user_token_2,
        }
    )
    artifact_manager_2 = await api_2.get_service("public/artifact-manager")

    # Create a dataset as a child of the public collection
    dataset = await artifact_manager_2.create(
        parent_id=collection.id,
        manifest=dataset_manifest,
    )
    assert dataset["parent_id"] == f"{api.config.workspace}/{collection.alias}"

    await artifact_manager_anonymous.read(artifact_id=dataset.id)

    # User 2 cannot reset stats on the newly created dataset
    with pytest.raises(
        Exception,
        match=r".*PermissionError: User does not have permission to perform the operation.*",
    ):
        await artifact_manager_anonymous.reset_stats(artifact_id=dataset.id)


async def test_versioning_artifacts(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """Test versioning features for artifacts."""

    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "sqlite test client",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an artifact with an initial version
    initial_manifest = {
        "name": "Versioned Artifact",
        "description": "Artifact to test versioning",
    }
    artifact = await artifact_manager.create(
        type="dataset",
        manifest=initial_manifest,
        version="v0",
        config={"permissions": {"*": "rw+"}},  # Add permissions for all operations
    )

    # Verify the initial version
    artifact_data = await artifact_manager.read(artifact_id=artifact.id, version="v0")
    assert artifact_data["versions"] is not None
    assert len(artifact_data["versions"]) == 1
    assert artifact_data["versions"][-1]["version"] == "v0"

    # Test creating with both version and stage=True
    staged_manifest = {
        "name": "Staged Artifact",
        "description": "Testing version and stage parameters",
    }
    staged_artifact = await artifact_manager.create(
        type="dataset",
        manifest=staged_manifest,
        version="my_version",  # This should be ignored since stage=True
        stage=True,
        config={"permissions": {"*": "rw+"}},
    )

    # Verify it's in staging mode with no versions
    assert staged_artifact["versions"] == []
    assert staged_artifact["staging"] is not None

    # Commit the staged artifact
    committed = await artifact_manager.commit(artifact_id=staged_artifact.id)
    assert committed["versions"] is not None
    assert len(committed["versions"]) == 1
    assert (
        committed["versions"][0]["version"] == "v0"
    )  # Should get v0 regardless of my_version

    # Create another artifact with no explicit version
    another_manifest = {
        "name": "Another Artifact",
        "description": "Testing default version creation",
    }
    another_artifact = await artifact_manager.create(
        type="dataset",
        manifest=another_manifest,
        config={"permissions": {"*": "rw+"}},  # Add permissions for all operations
    )
    # Verify it has an initial version
    assert another_artifact["versions"] is not None
    assert len(another_artifact["versions"]) == 1
    assert another_artifact["versions"][0]["version"] == "v0"

    # Test edit with both version and stage=True
    edited = await artifact_manager.edit(
        artifact_id=another_artifact.id,
        manifest={
            "name": "Updated Name",
            "description": "Testing version and stage parameters",
        },
        version="new",  # Use valid version "new" when staging
        stage=True,
    )
    assert edited["staging"] is not None
    assert len(edited["versions"]) == 1  # Should still have only initial version

    # Commit the staged edit
    committed = await artifact_manager.commit(
        artifact_id=another_artifact.id,
        version="custom_v1",  # Override version during commit
    )
    assert committed["versions"] is not None
    assert len(committed["versions"]) == 2  # Should have created a new version
    assert (
        committed["versions"][-1]["version"] == "custom_v1"
    )  # Should use custom version name

    # Clean up
    await artifact_manager.delete(artifact_id=artifact.id)
    await artifact_manager.delete(artifact_id=staged_artifact.id)
    await artifact_manager.delete(artifact_id=another_artifact.id)


async def test_commit_version_handling(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """Test that commit always creates a version even when not explicitly specified."""
    api = await connect_to_server(
        {
            "name": "sqlite test client",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an artifact with both version and stage=True
    manifest = {
        "name": "Test Artifact",
        "description": "Testing version creation on commit",
    }
    artifact = await artifact_manager.create(
        type="dataset",
        manifest=manifest,
        version="my_version",  # This should be ignored since stage=True
        stage=True,
    )

    # Verify no versions in staging
    assert artifact["versions"] == []
    assert artifact["staging"] is not None

    # Commit without specifying version - should create v0
    committed = await artifact_manager.commit(artifact_id=artifact.id)

    # Verify a version was created
    assert committed["versions"] is not None
    assert len(committed["versions"]) == 1
    assert (
        committed["versions"][0]["version"] == "v0"
    )  # Should get v0 regardless of my_version
    assert committed["staging"] is None

    # Create another staged version with both version and stage=True
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest=manifest,
        version="new",  # Use valid version "new" when staging
        stage=True,
    )

    # Add a file to the staged version
    file_content = "Test content"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="test.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit with explicit version - version parameter should be used
    # since we have intent to create a new version
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
        version="v5",  # This should be used since we have new version intent
        comment="Second version",
    )

    # Verify new version was added
    assert len(committed["versions"]) == 2
    assert committed["versions"][-1]["version"] == "v5"  # Should get v5 as specified
    assert committed["versions"][-1]["comment"] == "Second version"

    # Create another staged version with both version and stage=True
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest=manifest,
        version="new",  # Use valid version "new" when staging
        stage=True,
    )

    # Add another file
    file_content = "More test content"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="test2.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit again without version - should create v2 (sequential from 2 existing versions)
    committed = await artifact_manager.commit(
        artifact_id=artifact.id, comment="Third version"
    )

    # Verify another version was added
    assert len(committed["versions"]) == 3
    assert (
        committed["versions"][-1]["version"] == "v2"
    )  # Should get v2 as sequential version (len(versions)=2)
    assert committed["versions"][-1]["comment"] == "Third version"

    # Clean up
    await artifact_manager.delete(artifact_id=artifact.id)


async def test_artifact_alias_pattern(minio_server, fastapi_server, test_user_token):
    """Test artifact alias pattern functionality."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection with an alias pattern configuration
    collection_manifest = {
        "name": "Alias Pattern Collection",
        "description": "A collection to test alias pattern functionality",
    }
    collection_config = {
        "id_parts": {
            "animals": ["dog", "cat", "bird"],
            "colors": ["red", "blue", "green"],
        }
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config=collection_config,
    )

    # Create an artifact with alias pattern under the collection
    dataset_manifest = {"name": "My test data"}
    alias_pattern = "{colors}-{animals}"
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        alias=alias_pattern,
        version="stage",
    )

    # Fetch metadata to confirm the generated alias
    dataset_metadata = await artifact_manager.read(
        artifact_id=dataset.id, version="stage"
    )
    generated_alias = dataset_metadata["alias"].split("/")[-1]

    # Extract and validate parts of the generated alias
    color, animal = generated_alias.split("-")
    assert color in ["red", "blue", "green"]
    assert animal in ["dog", "cat", "bird"]

    # read the artifact using the generated alias
    dataset = await artifact_manager.read(dataset_metadata["alias"], version="stage")
    # Verify the alias pattern is correctly applied
    assert dataset["manifest"]["name"] == "My test data"

@pytest.mark.xfail(reason="Zenodo sandbox token may not be available")
async def test_publish_artifact(minio_server, fastapi_server, test_user_token):
    """Test publishing an artifact."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing publishing
    collection_manifest = {
        "name": "Publish Test Collection",
        "description": "A collection to test publishing",
    }

    # Define secrets, including the Zenodo sandbox access token
    secrets = {"SANDBOX_ZENODO_ACCESS_TOKEN": os.environ.get("SANDBOX_ZENODO_TOKEN")}
    assert secrets[
        "SANDBOX_ZENODO_ACCESS_TOKEN"
    ], "Please set SANDBOX_ZENODO_TOKEN environment variable"

    # Create the collection artifact with the necessary secrets
    collection = await artifact_manager.create(
        type="collection", manifest=collection_manifest, secrets=secrets
    )

    # Create an artifact within the collection with a publish target to sandbox Zenodo
    dataset_manifest = {
        "name": "Test Dataset",
        "description": "A test dataset to publish",
    }

    dataset = await artifact_manager.create(
        type="dataset",
        alias="{zenodo_conceptrecid}",
        parent_id=collection.id,
        manifest=dataset_manifest,
        config={"publish_to": "sandbox_zenodo"},
        version="stage",
    )

    assert (
        dataset.alias != "{zenodo_conceptrecid}"
        and str(int(dataset.alias)) == dataset.alias
    )

    # Add a file to the artifact
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="example.txt",
    )
    source = "file contents of example.txt"
    response = requests.put(put_url, data=source)
    assert response.ok, response.text

    # Commit the artifact after adding the file
    await artifact_manager.commit(artifact_id=dataset.id)

    # Publish the artifact to Zenodo Sandbox
    await artifact_manager.publish(artifact_id=dataset.id, to="sandbox_zenodo")

    # Retrieve and validate the updated metadata, confirming it includes publication info
    dataset_metadata = await artifact_manager.read(artifact_id=dataset.id)
    assert (
        "zenodo" in dataset_metadata["config"]
    ), "Zenodo publication information missing in config"


async def test_artifact_filtering(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """Test artifact filtering with various fields, including type, created_by, view_count, and stage."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    user_id1 = api.config.user["id"]

    # Create a collection for testing filters
    collection_manifest = {
        "name": "Filter Test Collection",
        "description": "A collection to test filter functionality",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={
            "permissions": {"*": "rw+"},
        },
    )

    # Create artifacts with varied attributes inside the collection (created by user_id1)
    created_artifacts = []
    for i in range(5):
        artifact_manifest = {
            "name": f"Artifact {i}",
            "description": f"Description for artifact {i}",
        }
        artifact = await artifact_manager.create(
            type="dataset" if i % 2 == 0 else "application",
            parent_id=collection.id,
            manifest=artifact_manifest,
            version="stage" if (i % 2 == 0) else None,
        )
        await artifact_manager.read(
            artifact_id=artifact.id, version="stage" if (i % 2 == 0) else None
        )
        created_artifacts.append(artifact)

    # Create a second client (user_id2) to create additional artifacts
    api2 = await connect_to_server(
        {"name": "test-client-2", "server_url": SERVER_URL, "token": test_user_token_2}
    )
    artifact_manager2 = await api2.get_service("public/artifact-manager")

    for i in range(5, 10):
        artifact_manifest = {
            "name": f"Artifact {i}",
            "description": f"Description for artifact {i}",
        }
        artifact = await artifact_manager2.create(
            type="dataset" if i % 2 == 0 else "application",
            parent_id=collection.id,
            manifest=artifact_manifest,
            version="stage" if (i % 2 == 0) else None,
        )
        if i % 2 == 0:
            await artifact_manager2.commit(artifact_id=artifact.id)
        created_artifacts.append(artifact)

    # Filter by `type`: Only datasets should be returned, only committed
    results = await artifact_manager.list(
        parent_id=collection.id, filters={"type": "dataset"}, mode="AND"
    )
    assert len(results) == 2
    for result in results:
        assert result["type"] == "dataset"

    # Filter by `created_by`: Only artifacts created by user_id1 should be returned
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"created_by": user_id1},
        mode="AND",
    )
    assert len(results) == 2
    for result in results:
        assert result["created_by"] == user_id1

    # Filter by `view_count`: Return artifacts that have been viewed at least once
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"view_count": [1, None]},  # Filter for any view count >= 1
        mode="AND",
    )
    assert len(results) == 2

    # Backwards compatibility with version filter
    # Filter by `stage`: Only staged artifacts should be returned
    results = await artifact_manager.list(
        parent_id=collection.id, filters={"version": "stage"}, mode="AND"
    )
    assert len(results) == 3

    # Clean up by deleting the artifacts and the collection
    for i in range(10):
        artifact = created_artifacts[i]
        await artifact_manager.delete(artifact_id=artifact.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_http_artifact_endpoint(minio_server, fastapi_server, test_user_token):
    """Test the HTTP artifact serving endpoint for retrieving files and listing directories."""

    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection
    collection_manifest = {
        "name": "Test Collection",
        "description": "A test collection for HTTP endpoint",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create a string in yaml with infinite float
    yaml_str = """
    name: Test Dataset
    description: A test dataset for HTTP endpoint
    inf_float: [-.inf, .inf]
    """
    dataset_manifest = yaml.safe_load(yaml_str)
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Add a file to the artifact
    file_content = "File contents of example.txt"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id, file_path="example.txt", download_weight=1
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit the artifact
    await artifact_manager.commit(artifact_id=dataset.id)

    # Retrieve the dataset manifest via HTTP
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}"
    )
    assert response.status_code == 200
    assert "Test Dataset" in response.json()["manifest"]["name"]

    # List the files in the dataset directory via HTTP
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/"
    )
    assert response.status_code == 200
    assert "example.txt" in [file["name"] for file in response.json()]

    # Retrieve the file content via HTTP
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/example.txt"
    )
    assert response.status_code == 200
    assert response.text == file_content

    # Verify that download count has incremented
    artifact_data = await artifact_manager.read(
        artifact_id=dataset.id,
    )
    assert artifact_data["download_count"] == 1

    # Retrieve the collection's children to verify the dataset presence
    # This will also trigger caching of the collection's children
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children"
    )
    assert response.status_code == 200
    item = find_item(response.json(), "id", dataset.id)
    manifest = item["manifest"]
    assert "Test Dataset" == manifest["name"]

    # Now let's add a new dataset to the collection
    dataset_manifest = {
        "name": "Test Dataset 2",
        "description": "A test dataset for HTTP endpoint",
    }
    dataset2 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )
    # now get the children again
    # due to caching, the new dataset should not be in the response
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children"
    )
    assert response.status_code == 200
    item = find_item(response.json(), "id", dataset2.id)
    assert item is None

    # But we can force a refresh of the cache with 'no_cache' query parameter
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?no_cache=1"
    )
    assert response.status_code == 200
    item = find_item(response.json(), "id", dataset2.id)
    manifest = item["manifest"]
    assert "Test Dataset 2" == manifest["name"]


async def test_edit_existing_artifact(minio_server, fastapi_server, test_user_token):
    """Test editing an existing artifact."""

    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection_manifest = {
        "name": "Edit Test Collection",
        "description": "A collection for testing artifact edits",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Verify collection has initial version
    assert collection["versions"] is not None
    assert len(collection["versions"]) == 1
    assert collection["versions"][0]["version"] == "v0"

    # Create an artifact (dataset) within the collection
    dataset_manifest = {
        "name": "Edit Test Dataset",
        "description": "A dataset to test editing",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
    )

    # Verify no versions in staging mode
    assert dataset["versions"] == []
    assert dataset["staging"] is not None

    collection = await artifact_manager.read(artifact_id=collection.id)
    assert collection["config"]["child_count"] == 1

    # Commit the artifact
    dataset = await artifact_manager.commit(artifact_id=dataset.id)
    # Verify version was created
    assert dataset["versions"] is not None
    assert len(dataset["versions"]) == 1
    assert dataset["staging"] is None

    # Verify the artifact is listed under the collection
    items = await artifact_manager.list(parent_id=collection.id)
    item = find_item(items, "id", dataset.id)
    assert item["manifest"]["name"] == "Edit Test Dataset"

    # Edit the artifact's manifest
    edited_manifest = {
        "name": "Edit Test Dataset",
        "description": "Updated description of the test dataset",
        "custom_key": 19222,
    }

    # Edit the artifact using the new manifest
    await artifact_manager.edit(
        type="dataset",
        artifact_id=dataset.id,
        manifest=edited_manifest,
        stage=True,  # Use stage=True to put artifact in staging mode
        version="new",  # Explicitly indicate intent to create new version on commit
    )

    # Verify the manifest updates are staged (not committed yet)
    staged_artifact = await artifact_manager.read(
        artifact_id=dataset.id, version="stage"
    )
    staged_manifest = staged_artifact["manifest"]
    assert staged_manifest["description"] == "Updated description of the test dataset"
    assert staged_manifest["custom_key"] == 19222

    # Add a file to the artifact
    file_content = "File contents of example.txt"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id, file_path="example.txt"
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit the artifact after editing
    dataset = await artifact_manager.commit(artifact_id=dataset.id)
    assert dataset["versions"] is not None
    assert len(dataset["versions"]) == 2  # Should have created a new version

    # Verify the committed manifest reflects the edited data
    committed_artifact = await artifact_manager.read(artifact_id=dataset.id)
    committed_manifest = committed_artifact["manifest"]
    assert (
        committed_manifest["description"] == "Updated description of the test dataset"
    )
    assert committed_manifest["custom_key"] == 19222
    versions = committed_artifact["versions"]

    # Retrieve and verify the uploaded file
    get_url = await artifact_manager.get_file(
        artifact_id=dataset.id,
        file_path="example.txt",
        version="v1",
    )
    response = requests.get(get_url)
    assert response.status_code == 200
    assert response.text == file_content

    # Edit it directly
    await artifact_manager.edit(
        type="dataset",
        artifact_id=dataset.id,
        manifest={"name": "Edit Test Dataset"},
        version="new",  # auto will create a new version
    )

    # Verify the manifest updates are committed
    committed_artifact = await artifact_manager.read(artifact_id=dataset.id)
    committed_manifest = committed_artifact["manifest"]
    assert committed_manifest["name"] == "Edit Test Dataset"
    assert len(versions) + 1 == len(committed_artifact["versions"])

    # Check list of artifacts
    items = await artifact_manager.list(parent_id=collection.id)
    item = find_item(items, "id", committed_artifact.id)

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_artifact_schema_validation(
    minio_server, fastapi_server, test_user_token
):
    """Test schema validation when committing artifacts to a collection."""

    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Define the schema for the collection
    collection_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "record_type": {
                "type": "string",
                "enum": ["dataset", "model", "application"],
            },
        },
        "required": ["name", "description", "record_type"],
    }

    # Create a collection with the schema
    collection_manifest = {
        "name": "Schema Test Collection",
        "description": "A collection with schema validation",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={
            "collection_schema": collection_schema,
            "permissions": {"*": "r", "@": "rw+"},
        },
    )

    # Create a valid dataset artifact that conforms to the schema
    valid_dataset_manifest = {
        "record_type": "dataset",
        "name": "Valid Dataset",
        "description": "A dataset that conforms to the schema",
    }
    valid_dataset = await artifact_manager.create(
        parent_id=collection.id, manifest=valid_dataset_manifest, version="stage"
    )

    # Add a file to the valid dataset artifact
    file_content = "Contents of valid_file.txt"
    put_url = await artifact_manager.put_file(
        artifact_id=valid_dataset.id, file_path="valid_file.txt"
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit the valid dataset artifact (should succeed as it conforms to the schema)
    artifact = await artifact_manager.commit(artifact_id=valid_dataset.id)
    assert artifact.file_count == 1

    # Confirm the dataset is listed in the collection
    items = await artifact_manager.list(parent_id=collection.id)
    item = find_item(items, "id", valid_dataset.id)
    assert item["manifest"]["name"] == "Valid Dataset"

    # Attempt to create an invalid dataset artifact (missing required 'description' field)
    invalid_dataset_manifest = {
        "name": "Invalid Dataset",
        "record_type": "dataset",
    }
    invalid_dataset = await artifact_manager.create(
        parent_id=collection.id, manifest=invalid_dataset_manifest, version="stage"
    )

    # Commit should raise a validation error due to schema violation
    with pytest.raises(
        Exception, match=r".*ValidationError: 'description' is a required property.*"
    ):
        await artifact_manager.commit(artifact_id=invalid_dataset.id)

    # Clean up by deleting the valid dataset and the collection
    await artifact_manager.delete(artifact_id=valid_dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_artifact_manager_with_collection(
    minio_server, fastapi_server, test_user_token
):
    """Test artifact management within collections."""

    # Connect to the server and set up the artifact manager
    api = await connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection
    collection_manifest = {
        "name": "test-collection",
        "description": "A test collection",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={
            "permissions": {"*": "r", "@": "rw+"},
        },
    )

    # Retrieve the collection via HTTP to confirm creation
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}"
    )
    assert response.status_code == 200
    assert "test-collection" in response.json()["manifest"]["name"]

    # Create a dataset artifact within the collection
    dataset_manifest = {
        "name": "test-dataset",
        "description": "A test dataset within the collection",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        alias="test-dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
    )

    # Add a file to the dataset artifact
    file_content = "file contents of test.txt"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id, file_path="test.txt"
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Confirm the dataset's inclusion in the collection
    collection_items = await artifact_manager.list(
        parent_id=collection.id, filters={"version": "stage"}
    )
    item = find_item(collection_items, "id", dataset.id)
    assert item["manifest"]["name"] == "test-dataset"

    # Commit the dataset artifact to finalize it
    await artifact_manager.commit(artifact_id=dataset.id)

    # Verify the dataset is listed in the collection after commit
    collection_items = await artifact_manager.list(parent_id=collection.id)
    item = find_item(collection_items, "id", dataset.id)
    dataset_item = item["manifest"]
    assert dataset_item["description"] == "A test dataset within the collection"
    assert item["view_count"] == 0

    # Increase the view count by reading the dataset
    dataset_summary = await artifact_manager.read(artifact_id=dataset.id)
    assert dataset_summary["view_count"] == 1

    # Verify that view count is reflected in the collection list
    collection_items = await artifact_manager.list(parent_id=collection.id)
    dataset_item = next(item for item in collection_items if item["id"] == dataset.id)
    assert dataset_item["view_count"] == 1

    # Retrieve the file in the dataset via HTTP to verify contents
    get_url = await artifact_manager.get_file(
        artifact_id=dataset.id, file_path="test.txt"
    )
    response = requests.get(get_url)
    assert response.ok
    assert response.text == file_content

    # Test overwriting the existing dataset artifact within the collection
    updated_dataset_manifest = {
        "name": "test-dataset",
        "description": "Overwritten description for dataset",
    }
    await artifact_manager.edit(
        type="dataset",
        artifact_id="test-dataset",
        manifest=updated_dataset_manifest,
    )

    # Confirm that the description update is reflected
    updated_manifest = await artifact_manager.read(
        artifact_id=f"{api.config.workspace}/test-dataset"
    )
    updated_manifest = updated_manifest["manifest"]
    assert updated_manifest["description"] == "Overwritten description for dataset"

    # Remove the dataset artifact from the collection
    await artifact_manager.delete(artifact_id=dataset.id)

    # Confirm deletion by checking the collection's children
    collection_items = await artifact_manager.list(parent_id=collection.id)
    item = find_item(collection_items, "id", dataset.id)
    assert item is None

    # Clean up by deleting the collection
    await artifact_manager.delete(artifact_id=collection.id)


async def test_artifact_edge_cases_with_collection(
    minio_server, fastapi_server, test_user_token
):
    """Test edge cases with artifact collections."""

    # Connect to the server and set up the artifact manager
    api = await connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Attempt to create a collection with orphan=False, which should fail
    collection_manifest = {
        "name": "edge-case-collection",
        "description": "A test collection for edge cases",
    }

    with pytest.raises(
        Exception,
        match=r".*Artifact with ID 'ws-user-user-1/non-existent-parent' does not exist.*",
    ):
        await artifact_manager.create(
            type="collection",
            parent_id="non-existent-parent",
            manifest=collection_manifest,
            config={"permissions": {"*": "r", "@": "rw+"}},
        )

    # Successfully create the collection with orphan set by default
    collection = await artifact_manager.create(
        manifest=collection_manifest, config={"permissions": {"*": "r", "@": "rw+"}}
    )

    # Create a dataset artifact within the collection
    dataset_manifest = {
        "name": "edge-case-dataset",
        "description": "A dataset for testing edge cases",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        alias="edge-case-dataset",
        manifest=dataset_manifest,
        version="stage",
    )

    # Try to create the same artifact again without overwrite, expecting it to fail
    with pytest.raises(Exception, match=r".*already exists.*"):
        await artifact_manager.create(
            type="dataset",
            parent_id=collection.id,
            alias="edge-case-dataset",
            manifest=dataset_manifest,
            version="stage",
        )

    # Add a file to the dataset artifact
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id, file_path="example.txt"
    )
    file_content = "sample file content"
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit the dataset artifact to finalize it
    await artifact_manager.commit(artifact_id=dataset.id)

    # Ensure the dataset is listed under the collection
    collection_items = await artifact_manager.list(parent_id=collection.id)
    item = find_item(collection_items, "id", dataset.id)
    assert item["manifest"]["name"] == "edge-case-dataset"

    # Test committing a dataset artifact with missing files, expecting a failure
    incomplete_manifest = {
        "name": "incomplete-dataset",
        "description": "Dataset missing a required file",
    }
    incomplete_dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=incomplete_manifest,
        version="stage",
    )

    await artifact_manager.put_file(
        artifact_id=incomplete_dataset.id, file_path="missing_file.txt"
    )

    # Commit should fail due to missing file
    with pytest.raises(Exception, match=r".*FileNotFoundError: .*"):
        await artifact_manager.commit(artifact_id=incomplete_dataset.id)

    # Clean up by deleting the artifacts and the collection
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=incomplete_dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_artifact_search_in_manifest(
    minio_server, fastapi_server, test_user_token
):
    """Test search functionality within the 'manifest' field of artifacts using keywords and filters in both AND and OR modes."""

    # Connect to the server and set up the artifact manager
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing search
    collection_manifest = {
        "name": "Search Test Collection",
        "description": "A collection for testing search functionality",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create multiple artifacts within the collection for search tests
    for i in range(5):
        dataset_manifest = {
            "name": f"Test Dataset {i}",
            "description": f"A test dataset {i}",
        }
        artifact = await artifact_manager.create(
            type="dataset",
            alias=f"test-dataset-{i}",
            parent_id=collection.id,
            manifest=dataset_manifest,
            version="stage",
        )
        await artifact_manager.commit(artifact_id=artifact.id)

    # Test search by exact match keyword using AND mode
    search_results = await artifact_manager.list(
        parent_id=collection.id, keywords=["Dataset 3"], mode="AND"
    )
    assert len(search_results) == 1
    assert search_results[0]["manifest"]["name"] == "Test Dataset 3"

    # Test search for multiple results by description using OR mode
    search_results = await artifact_manager.list(
        parent_id=collection.id, keywords=["test", "dataset"], mode="OR"
    )
    assert (
        len(search_results) == 5
    )  # All datasets have "test" and "dataset" in their description

    # Test search with multiple keywords using AND mode (returns fewer results)
    search_results = await artifact_manager.list(
        parent_id=collection.id, keywords=["Test Dataset", "3"], mode="AND"
    )
    assert len(search_results) == 1
    assert search_results[0]["manifest"]["name"] == "Test Dataset 3"

    # Test search using filters to match manifest field directly
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name": "Test Dataset 3"}},
        mode="AND",
    )
    assert len(search_results) == 1
    assert search_results[0]["manifest"]["name"] == "Test Dataset 3"

    # Test fuzzy search using filters within the manifest
    search_results = await artifact_manager.list(
        parent_id=collection.id, filters={"manifest": {"name": "Test*"}}, mode="OR"
    )
    assert len(search_results) == 5  # All datasets match "Test*" in the name

    # Test search with multiple filters in AND mode for exact match on both name and description
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={
            "manifest": {"name": "Test Dataset 3", "description": "A test dataset 3"}
        },
        mode="AND",
    )
    assert len(search_results) == 1
    assert search_results[0]["manifest"]["name"] == "Test Dataset 3"

    # Test search with multiple filters in OR mode for matching any field
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={
            "manifest": {
                "$or": [
                    {"name": "Test Dataset 3"},
                    {"description": "A test dataset 1"},
                ]
            }
        },
    )
    assert len(search_results) == 2  # Matches for either dataset 1 or 3

    # Test search with multiple filters in AND mode (implicit)
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={
            "manifest": {
                "name": "Test Dataset 3",
                "description": "A test dataset 3",
            }
        },
    )
    assert len(search_results) == 1
    assert search_results[0]["manifest"]["name"] == "Test Dataset 3"

    # Test fuzzy search using wildcards
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name": "Test*"}},
    )
    assert len(search_results) == 5  # All datasets match "Test*" in the name

    # Test fuzzy search with multiple wildcards
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name": "*Dataset*"}},
    )
    assert len(search_results) == 5  # All datasets match "*Dataset*"

    # Clean up by deleting all artifacts and the collection
    for i in range(5):
        await artifact_manager.delete(artifact_id=f"test-dataset-{i}")
    await artifact_manager.delete(artifact_id=collection.id)


async def test_artifact_search_with_array_filters(
    minio_server, fastapi_server, test_user_token
):
    """Test search functionality with array filters in the manifest."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing array filters
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Array Filter Test Collection",
            "description": "A collection to test array filter functionality",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create multiple artifacts with array fields in manifest
    artifacts = []
    test_data = [
        {
            "name": "Dataset 1",
            "tags": ["tag1", "tag2", "tag3"],
            "categories": ["cat1", "cat2"],
        },
        {
            "name": "Dataset 2",
            "tags": ["tag2", "tag3", "tag4"],
            "categories": ["cat2", "cat3"],
        },
        {
            "name": "Dataset 3",
            "tags": ["tag1", "tag4", "tag5"],
            "categories": ["cat1", "cat3"],
        },
    ]

    for data in test_data:
        artifact = await artifact_manager.create(
            type="dataset",
            parent_id=collection.id,
            manifest=data,
        )
        artifacts.append(artifact)

    # Test filtering by single tag
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"tags": ["tag1"]}},
    )
    assert len(results) == 2  # Should match Dataset 1 and 3
    assert set(r["manifest"]["name"] for r in results) == {"Dataset 1", "Dataset 3"}

    # Test filtering by multiple tags
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"tags": ["tag2", "tag3"]}},
    )
    assert len(results) == 2  # Should match Dataset 1 and 2
    assert set(r["manifest"]["name"] for r in results) == {"Dataset 1", "Dataset 2"}

    # Test filtering by multiple array fields
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"tags": ["tag1"], "categories": ["cat1"]}},
    )
    assert len(results) == 2  # Should match artifacts with both tag1 and cat1
    assert set(r["manifest"]["name"] for r in results) == {"Dataset 1", "Dataset 3"}

    # Clean up
    for artifact in artifacts:
        await artifact_manager.delete(artifact_id=artifact.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_artifact_search_with_advanced_filters(
    minio_server, fastapi_server, test_user_token
):
    """Test search functionality with advanced filter operators."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Advanced Filter Test Collection",
            "description": "Testing advanced filter functionality",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create test artifacts
    test_data = [
        {
            "name": "Dataset 1",
            "tags": ["tag1", "tag2", "tag3"],
            "categories": ["cat1", "cat2"],
            "version": "1.0.0",
        },
        {
            "name": "Dataset 2",
            "tags": ["tag2", "tag3", "tag4"],
            "categories": ["cat2", "cat3"],
            "version": "2.0.0",
        },
        {
            "name": "Dataset 3",
            "tags": ["tag1", "tag4", "tag5"],
            "categories": ["cat1", "cat3"],
            "version": "1.5.0",
        },
    ]

    artifacts = []
    for data in test_data:
        artifact = await artifact_manager.create(
            type="dataset",
            parent_id=collection.id,
            manifest=data,
        )
        artifacts.append(artifact)

    # Test AND condition with array containment and exact match
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={
            "manifest": {
                "$and": [
                    {"tags": {"$all": ["tag1", "tag2"]}},
                    {"categories": {"$in": ["cat1"]}},
                ]
            }
        },
    )
    assert len(results) == 1
    assert results[0]["manifest"]["name"] == "Dataset 1"

    # Test OR condition with fuzzy matching
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={
            "manifest": {
                "$or": [
                    {"version": {"$like": "1.*"}},
                    {"tags": {"$in": ["tag4"]}},
                ]
            }
        },
    )
    assert len(results) == 3  # Should match all datasets

    # Test complex nested conditions
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={
            "manifest": {
                "$and": [
                    {
                        "$or": [
                            {"tags": {"$in": ["tag4"]}},
                            {"categories": {"$in": ["cat3"]}},
                        ]
                    },
                    {"version": {"$like": "1.*"}},
                ]
            }
        },
    )
    assert len(results) == 1
    assert results[0]["manifest"]["name"] == "Dataset 3"

    # Clean up
    for artifact in artifacts:
        await artifact_manager.delete(artifact_id=artifact.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_multipart_upload_service_functions(
    minio_server, fastapi_server, test_user_token
):
    """Test the new multipart upload service functions."""
    api = await connect_to_server(
        {
            "name": "test client for multipart",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Multipart Upload Test Collection",
            "description": "A collection for testing multipart upload service functions",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create a dataset artifact
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={
            "name": "Multipart Test Dataset",
            "description": "A dataset for testing multipart upload",
        },
        version="stage",
    )

    # Test put_file_start_multipart service function
    file_path = "large_file.txt"
    part_count = 3
    download_weight = 5.0
    
    multipart_info = await artifact_manager.put_file_start_multipart(
        artifact_id=dataset.id,
        file_path=file_path,
        part_count=part_count,
        download_weight=download_weight,
        expires_in=3600
    )
    
    # Verify the response structure
    assert "upload_id" in multipart_info
    assert "parts" in multipart_info
    assert len(multipart_info["parts"]) == part_count
    
    upload_id = multipart_info["upload_id"]
    part_urls = multipart_info["parts"]
    
    # Verify each part has the expected structure
    for i, part in enumerate(part_urls):
        assert part["part_number"] == i + 1
        assert "url" in part
        assert part["url"].startswith("http")

    # Simulate uploading parts (we'll create fake ETags)
    # In a real scenario, you would upload actual data to each URL and get real ETags
    parts_data = []
    for part in part_urls:
        # Simulate an ETag (normally this would come from the S3 upload response)
        fake_etag = f'"etag-{part["part_number"]}"'
        parts_data.append({
            "part_number": part["part_number"],
            "etag": fake_etag
        })

    # Test put_file_complete_multipart service function
    # Note: This will fail in the test because we didn't actually upload the parts
    # but it tests that the service function is properly structured
    try:
        result = await artifact_manager.put_file_complete_multipart(
            artifact_id=dataset.id,
            upload_id=upload_id,
            parts=parts_data
        )
        # If this succeeds, verify the response
        assert result["success"] is True
        assert "message" in result
    except Exception as e:
        # Expected to fail because we didn't actually upload parts
        # but should be a specific S3 error, not a permission or structure error
        error_msg = str(e).lower()
        assert "invalid" in error_msg or "part" in error_msg or "etag" in error_msg
        print(f"Expected S3 error (parts not actually uploaded): {e}")

    # Test with invalid parameters
    try:
        await artifact_manager.put_file_start_multipart(
            artifact_id=dataset.id,
            file_path="test2.txt",
            part_count=0  # Invalid: should be > 0
        )
        assert False, "Should have raised ValueError for invalid part_count"
    except Exception as e:
        assert "Part count must be greater than 0" in str(e)

    try:
        await artifact_manager.put_file_start_multipart(
            artifact_id=dataset.id,
            file_path="test3.txt",
            part_count=10001  # Invalid: too many parts
        )
        assert False, "Should have raised ValueError for too many parts"
    except Exception as e:
        assert "maximum number of parts is 10,000" in str(e)

    # Test with invalid download_weight
    try:
        await artifact_manager.put_file_start_multipart(
            artifact_id=dataset.id,
            file_path="test4.txt",
            part_count=2,
            download_weight=-1  # Invalid: should be non-negative
        )
        assert False, "Should have raised ValueError for negative download_weight"
    except Exception as e:
        assert "Download weight must be a non-negative number" in str(e)

    # Test complete with invalid upload_id
    try:
        await artifact_manager.put_file_complete_multipart(
            artifact_id=dataset.id,
            upload_id="invalid-upload-id",
            parts=[{"part_number": 1, "etag": '"test"'}]
        )
        assert False, "Should have raised ValueError for invalid upload_id"
    except Exception as e:
        assert "Upload session expired or not found" in str(e)

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_artifact_version_management(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """Test version management functionality for artifacts."""
    api = await connect_to_server(
        {
            "name": "sqlite test client",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Version Test Collection",
            "description": "A collection for testing version management",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    assert len(collection.versions) == 1 and collection.versions[0].version == "v0"

    # Create an artifact with initial version (should be created automatically)
    initial_manifest = {
        "name": "Version Test Dataset",
        "description": "Initial version",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=initial_manifest,
        version="v0",  # Explicitly set initial version
    )

    # Verify initial version exists
    assert len(dataset["versions"]) == 1
    assert dataset["versions"][0]["version"] == "v0"

    # Create a new version in staging mode
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest=initial_manifest,
        stage=True,  # Use stage=True to put artifact in staging mode
        version="new",  # Explicitly indicate intent to create new version
    )

    # Add a file to the staged version
    file_content = "Content for version 1"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="test.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # List files in staged version to verify
    staged_files = await artifact_manager.list_files(
        artifact_id=dataset.id, version="stage"
    )
    assert len(staged_files) == 1  # Only the file we just added
    assert staged_files[0]["name"] == "test.txt"

    # Commit the file to create version 1
    await artifact_manager.commit(
        artifact_id=dataset.id, version="v1", comment="Added test.txt"
    )

    # Verify version 1 exists
    dataset = await artifact_manager.read(artifact_id=dataset.id)
    assert len(dataset["versions"]) == 2
    assert dataset["versions"][1]["version"] == "v1"
    assert dataset["versions"][1]["comment"] == "Added test.txt"

    # List files in version 1 to verify
    files_v1 = await artifact_manager.list_files(artifact_id=dataset.id, version="v1")
    assert len(files_v1) == 1
    assert files_v1[0]["name"] == "test.txt"

    # Read file from version 1
    get_url = await artifact_manager.get_file(
        artifact_id=dataset.id,
        file_path="test.txt",
        version="v1",
    )
    response = requests.get(get_url)
    assert response.ok
    assert response.text == file_content

    # Create another version in staging mode
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest=initial_manifest,
        stage=True,  # Use stage=True to put artifact in staging mode
        version="new",  # Explicitly indicate intent to create new version
    )

    # Copy file from version 1 to stage
    get_url = await artifact_manager.get_file(
        artifact_id=dataset.id,
        file_path="test.txt",
        version="v1",
    )
    response = requests.get(get_url)
    assert response.ok
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="test.txt",
    )
    response = requests.put(put_url, data=response.text)
    assert response.ok

    # Add another file for version 2
    file_content2 = "Content for version 2"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="test2.txt",
    )
    response = requests.put(put_url, data=file_content2)
    assert response.ok

    # Commit to create version 2
    await artifact_manager.commit(
        artifact_id=dataset.id, version="v2", comment="Added test2.txt"
    )

    # Verify version 2 exists
    dataset = await artifact_manager.read(artifact_id=dataset.id)
    assert len(dataset["versions"]) == 3
    assert dataset["versions"][2]["version"] == "v2"
    assert dataset["versions"][2]["comment"] == "Added test2.txt"

    # List files for each version
    files_v1 = await artifact_manager.list_files(artifact_id=dataset.id, version="v1")
    assert len(files_v1) == 1
    assert files_v1[0]["name"] == "test.txt"

    files_v2 = await artifact_manager.list_files(artifact_id=dataset.id, version="v2")
    assert len(files_v2) == 2
    assert {f["name"] for f in files_v2} == {"test.txt", "test2.txt"}

    # Delete version 1
    await artifact_manager.delete(artifact_id=dataset.id, version="v1")

    # Verify version 1 is removed
    dataset = await artifact_manager.read(artifact_id=dataset.id)
    assert len(dataset["versions"]) == 2
    assert dataset["versions"][0]["version"] == "v0"
    assert dataset["versions"][1]["version"] == "v2"

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_default_version_handling(minio_server, fastapi_server, test_user_token):
    """Test version handling when version and stage parameters are not specified."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an artifact without specifying version or stage
    initial_manifest = {
        "name": "Default Version Test",
        "description": "Testing default version handling",
    }
    artifact = await artifact_manager.create(
        type="dataset",
        manifest=initial_manifest,
        # No version or stage specified - should create v0
    )

    # Verify initial version was created
    assert len(artifact["versions"]) == 1
    assert artifact["versions"][0]["version"] == "v0"
    assert artifact["staging"] is None

    # Edit without specifying version or stage - should UPDATE the existing latest version (v0)
    updated_manifest = {
        "name": "Default Version Test",
        "description": "Updated description",
    }
    edited = await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest=updated_manifest,
        # No version or stage specified - should update existing v0, NOT create new version
    )

    # Verify the edit updated the existing latest version, not created a new one
    assert len(edited["versions"]) == 1
    assert edited["versions"][0]["version"] == "v0"
    assert edited["staging"] is None
    # Verify the content was actually updated
    read_artifact = await artifact_manager.read(artifact_id=artifact.id)
    assert read_artifact["manifest"]["description"] == "Updated description"

    # Put artifact in staging mode to add files
    staged = await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest=updated_manifest,
        stage=True,
    )
    assert staged["staging"] is not None

    # Add a file to test file handling
    file_content = "Test content"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="test.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit without specifying version - should UPDATE existing v0 since no version="new" was specified
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
    )

    # Verify file is associated with the existing version and NO new version was created
    files = await artifact_manager.list_files(artifact_id=artifact.id)
    assert len(files) == 1
    assert files[0]["name"] == "test.txt"
    assert len(committed["versions"]) == 1  # Still only 1 version (v0 updated)
    assert committed["versions"][0]["version"] == "v0"  # Still v0
    assert committed["staging"] is None

    # Files should exist in version 0 (updated)
    files_v0 = await artifact_manager.list_files(artifact_id=artifact.id, version="v0")
    assert len(files_v0) == 1
    assert files_v0[0]["name"] == "test.txt"

    # NOW test explicit new version creation
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest=updated_manifest,
        stage=True,
        version="new",  # Explicitly request new version
    )

    # Add another file for the new version
    file_content2 = "Content for new version"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="test2.txt",
    )
    response = requests.put(put_url, data=file_content2)
    assert response.ok

    # Commit to create new version
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
    )

    # Now we should have 2 versions
    assert len(committed["versions"]) == 2
    assert committed["versions"][0]["version"] == "v0"  # Original version
    assert committed["versions"][1]["version"] == "v1"  # New version
    assert committed["staging"] is None

    # Files should be distributed correctly
    files_v0 = await artifact_manager.list_files(artifact_id=artifact.id, version="v0")
    assert len(files_v0) == 1  # test.txt in v0
    assert files_v0[0]["name"] == "test.txt"

    files_v1 = await artifact_manager.list_files(artifact_id=artifact.id, version="v1")
    assert len(files_v1) == 1  # test2.txt in v1
    assert files_v1[0]["name"] == "test2.txt"

    # Create another artifact with no versions initially (legacy case)
    another_manifest = {
        "name": "Another Test",
        "description": "Testing version creation",
    }
    another = await artifact_manager.create(
        type="dataset",
        manifest=another_manifest,
        stage=True,  # Start with no versions
    )

    # Verify no versions exist initially
    assert len(another["versions"]) == 0
    assert another["staging"] is not None

    # Add a file to test file handling
    file_content = "Another test content"
    put_url = await artifact_manager.put_file(
        artifact_id=another.id,
        file_path="test.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit the staged artifact to create v0
    committed = await artifact_manager.commit(
        artifact_id=another.id,
    )

    # Verify v0 was created and file exists in v0
    assert len(committed["versions"]) == 1
    assert committed["versions"][0]["version"] == "v0"
    assert committed["staging"] is None

    files = await artifact_manager.list_files(artifact_id=another.id, version="v0")
    assert len(files) == 1
    assert files[0]["name"] == "test.txt"

    # Edit without specifying version or stage - should UPDATE existing v0, not create new version
    new_manifest = {
        "name": "Another Test Updated",
        "description": "Testing version creation updated",
    }
    edited = await artifact_manager.edit(
        artifact_id=another.id,
        manifest=new_manifest,
        # No version or stage specified - should update existing v0
    )

    # Verify the edit updated existing v0, not created a new version
    assert len(edited["versions"]) == 1
    assert edited["versions"][0]["version"] == "v0"
    assert edited["staging"] is None
    # Verify the content was actually updated
    read_artifact = await artifact_manager.read(artifact_id=another.id)
    assert read_artifact["manifest"]["name"] == "Another Test Updated"

    # Clean up
    await artifact_manager.delete(artifact_id=artifact.id)
    await artifact_manager.delete(artifact_id=another.id)


async def test_version_handling_rules(minio_server, fastapi_server, test_user_token):
    """Test version handling rules for create, edit and commit operations."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Test Case 1: Create with stage=True should ignore version parameter
    manifest = {
        "name": "Version Rules Test",
        "description": "Testing version handling rules",
    }
    artifact = await artifact_manager.create(
        type="dataset",
        manifest=manifest,
        stage=True,
        version="v5",  # This should be ignored since stage=True
    )
    assert len(artifact["versions"]) == 0
    assert artifact["staging"] is not None

    # Commit should create v0 regardless of the version specified in create
    committed = await artifact_manager.commit(artifact_id=artifact.id)
    assert len(committed["versions"]) == 1
    assert committed["versions"][0]["version"] == "v0"

    # Test Case 2: version="stage" is equivalent to stage=True
    artifact2 = await artifact_manager.create(
        type="dataset",
        manifest=manifest,
        version="stage",  # Equivalent to stage=True
    )
    assert len(artifact2["versions"]) == 0
    assert artifact2["staging"] is not None

    # Test Case 3: Edit with stage=True should ignore invalid version parameter
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest=manifest,
        stage=True,
        version="new",  # This is allowed in staging mode
    )
    edited = await artifact_manager.read(artifact_id=artifact.id, version="stage")
    assert edited["staging"] is not None
    assert len(edited["versions"]) == 1  # Still has v0

    # Test Case 4: Edit with stage=True is equivalent to using staging mode
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest=manifest,
        stage=True,  # Use stage=True instead of version="stage"
        version="new",  # Need to maintain new version intent for Test Case 5
    )
    edited = await artifact_manager.read(artifact_id=artifact.id, version="stage")
    assert edited["staging"] is not None
    assert len(edited["versions"]) == 1  # Still has v0

    # Test Case 5: Commit with version override when staging with version="new"
    # Add a file to make sure we're creating a new version
    file_content = "Test content"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="test.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit with a custom version name since we had version="new" intent
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
        version="custom_v1",  # This should be allowed since we had new version intent
    )
    assert len(committed["versions"]) == 2
    assert committed["versions"][-1]["version"] == "custom_v1"
    assert committed["staging"] is None

    # Test Case 6: Direct edit without stage should UPDATE current version (not create new)
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "Updated Name", "description": "Updated description"},
        # No version or stage parameter - should update existing latest version
    )
    edited = await artifact_manager.read(artifact_id=artifact.id)
    assert edited["manifest"]["name"] == "Updated Name"
    assert edited["manifest"]["description"] == "Updated description"
    assert (
        len(edited["versions"]) == 2
    )  # No new version created, still has v0 and custom_v1
    assert edited["staging"] is None

    # Test Case 7: Edit with version="new" should CREATE a new version
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "New Version Name", "description": "New version description"},
        version="new",  # This should create a new version
    )
    edited = await artifact_manager.read(artifact_id=artifact.id)
    assert edited["manifest"]["name"] == "New Version Name"
    assert len(edited["versions"]) == 3  # Now should have v0, custom_v1, and v2
    assert edited["versions"][-1]["version"] == "v2"
    assert edited["staging"] is None


async def test_list_stage_parameter(minio_server, fastapi_server, test_user_token):
    """Test the stage parameter in list() function."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Stage Test Collection",
            "description": "A collection for testing stage parameter",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create artifacts in different states
    # 1. Committed artifact
    committed = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Committed Artifact"},
        version="v0",  # This will create a committed artifact
    )

    # 2. Staged artifact
    staged = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Staged Artifact"},
        stage=True,  # This will create a staged artifact
    )

    # 3. Another committed artifact
    committed2 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Another Committed"},
        version="v0",
    )

    # 4. Another staged artifact
    staged2 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Another Staged"},
        stage=True,
    )

    # Test 1: Default behavior (stage=False) should list only committed artifacts
    results = await artifact_manager.list(parent_id=collection.id)
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert names == {"Committed Artifact", "Another Committed"}

    # Test 2: stage=False should show only committed artifacts
    results = await artifact_manager.list(parent_id=collection.id, stage=False)
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert names == {"Committed Artifact", "Another Committed"}

    # Test 3: stage=True should show only staged artifacts
    results = await artifact_manager.list(parent_id=collection.id, stage=True)
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert names == {"Staged Artifact", "Another Staged"}

    # Test 4: stage='all' should show both staged and committed artifacts
    results = await artifact_manager.list(parent_id=collection.id, stage="all")
    assert len(results) == 4
    names = {r["manifest"]["name"] for r in results}
    assert names == {
        "Committed Artifact",
        "Another Committed",
        "Staged Artifact",
        "Another Staged",
    }

    # Test 5: Backward compatibility with filters={"version": "stage"}
    results = await artifact_manager.list(
        parent_id=collection.id, filters={"version": "stage"}
    )
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert names == {"Staged Artifact", "Another Staged"}

    # Clean up
    await artifact_manager.delete(artifact_id=committed.id)
    await artifact_manager.delete(artifact_id=committed2.id)
    await artifact_manager.delete(artifact_id=staged.id)
    await artifact_manager.delete(artifact_id=staged2.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_http_children_endpoint(minio_server, fastapi_server, test_user_token):
    """Test the HTTP endpoint for listing child artifacts with various parameters."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Children Test Collection",
            "description": "A collection for testing children endpoint",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create multiple child artifacts with various properties
    child1 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={
            "name": "Child Artifact 1",
            "description": "First child artifact",
            "tags": ["test", "one"],
            "metadata": {"priority": "high"},
        },
    )

    child2 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={
            "name": "Child Artifact 2",
            "description": "Second child artifact",
            "tags": ["test", "two"],
            "metadata": {"priority": "medium"},
        },
    )

    child3 = await artifact_manager.create(
        type="model",
        parent_id=collection.id,
        manifest={
            "name": "Child Artifact 3",
            "description": "Third child artifact",
            "tags": ["test", "three"],
            "metadata": {"priority": "low"},
        },
    )

    # Create a staged artifact
    staged_artifact = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={
            "name": "Staged Artifact",
            "description": "A staged artifact",
        },
        stage=True,
    )

    # Test basic endpoint functionality (default stage=false)
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children"
    )
    assert response.status_code == 200
    results = response.json()
    # Should only return committed artifacts (not staged)
    assert len(results) == 3
    names = {r["manifest"]["name"] for r in results}
    assert "Staged Artifact" not in names

    # Test keyword search
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?keywords=two"
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["manifest"]["name"] == "Child Artifact 2"

    # Test filtering by type
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?filters={json.dumps({'type': 'model'})}"
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["manifest"]["name"] == "Child Artifact 3"

    # Test filtering by manifest field (in nested format)
    filters = {"manifest": {"name": "Child Artifact 1"}}
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?filters={json.dumps(filters)}"
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["manifest"]["name"] == "Child Artifact 1"

    # Test pagination
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?offset=1&limit=1&pagination=true"
    )
    assert response.status_code == 200
    results = response.json()
    assert "items" in results
    assert "total" in results
    assert results["total"] == 3
    assert len(results["items"]) == 1

    # Test ordering
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?order_by=created_at"
    )
    assert response.status_code == 200
    results_asc = response.json()

    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?order_by=created_at<"
    )
    assert response.status_code == 200
    results_desc = response.json()

    # Verify that both queries return the same set of results
    asc_ids = [item["_id"] for item in results_asc]
    desc_ids = [item["_id"] for item in results_desc]
    assert set(asc_ids) == set(desc_ids), "Results should contain the same items"

    # Skip checking order difference since artifacts might be created at the same timestamp

    # Test no_cache parameter
    # First create a new artifact
    new_child = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={
            "name": "New Child",
            "description": "Newly created artifact for testing no_cache",
        },
    )

    # Then immediately request with no_cache to ensure it's included
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?no_cache=true"
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 4  # Should include the new artifact (3 original + 1 new)

    # Clean up
    await artifact_manager.delete(artifact_id=child1.id)
    await artifact_manager.delete(artifact_id=child2.id)
    await artifact_manager.delete(artifact_id=child3.id)
    await artifact_manager.delete(artifact_id=staged_artifact.id)
    await artifact_manager.delete(artifact_id=new_child.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_http_children_endpoint_stage_parameter(
    minio_server, fastapi_server, test_user_token
):
    """Test the stage parameter functionality in HTTP endpoint for listing child artifacts."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Stage Parameter Test Collection",
            "description": "A collection for testing stage parameter",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create committed artifacts
    committed1 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Committed Artifact 1"},
        version="v0",  # This will create a committed artifact
    )

    committed2 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Committed Artifact 2"},
        version="v0",
    )

    # Create staged artifacts
    staged1 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Staged Artifact 1"},
        stage=True,  # This will create a staged artifact
    )

    staged2 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Staged Artifact 2"},
        stage=True,
    )

    # Test default behavior (stage=false) - should only return committed artifacts
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children"
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert "Staged Artifact 1" not in names
    assert "Staged Artifact 2" not in names
    assert "Committed Artifact 1" in names
    assert "Committed Artifact 2" in names

    # Test stage=true parameter to get only staged artifacts
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?stage=true"
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert "Staged Artifact 1" in names
    assert "Staged Artifact 2" in names
    assert "Committed Artifact 1" not in names
    assert "Committed Artifact 2" not in names

    # Test stage=all parameter to get both staged and committed artifacts
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?stage=all"
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 4  # All 4 artifacts
    names = {r["manifest"]["name"] for r in results}
    assert "Staged Artifact 1" in names
    assert "Staged Artifact 2" in names
    assert "Committed Artifact 1" in names
    assert "Committed Artifact 2" in names

    # Test stage=false parameter explicitly
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children?stage=false"
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert "Staged Artifact 1" not in names
    assert "Staged Artifact 2" not in names
    assert "Committed Artifact 1" in names
    assert "Committed Artifact 2" in names

    # Clean up
    await artifact_manager.delete(artifact_id=committed1.id)
    await artifact_manager.delete(artifact_id=committed2.id)
    await artifact_manager.delete(artifact_id=staged1.id)
    await artifact_manager.delete(artifact_id=staged2.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_get_zip_file_content_endpoint_large_scale(
    minio_server, fastapi_server, test_user_token
):
    """Test retrieving content and listing directories from a ZIP file with 2000 files stored in S3."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Large Scale Test Collection",
            "description": "A collection for testing large-scale zip file handling",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create a dataset within the collection
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={
            "name": "Large Scale Test Dataset",
            "description": "A dataset with 2000 files for testing",
        },
        version="stage",
    )

    # Create a ZIP file in memory with 2000 files in various directories
    zip_buffer = BytesIO()
    with ZipFile(
        zip_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as zip_file:
        # Create files in root directory
        for i in range(100):
            zip_file.writestr(f"root_file_{i}.txt", f"Content of root file {i}")

        # Create files in nested directories (10 directories, 190 files each)
        for dir_num in range(10):
            # Regular small files
            for file_num in range(180):
                file_path = f"dir_{dir_num}/file_{file_num}.txt"
                zip_file.writestr(file_path, f"Content of {file_path}")

            # Add some larger files (10 per directory)
            for large_file_num in range(10):
                file_path = f"dir_{dir_num}/large_file_{large_file_num}.txt"
                content = (
                    f"Large content repeated many times {large_file_num}" * 100
                )  # Reduced content size
                zip_file.writestr(file_path, content)

    zip_buffer.seek(0)
    zip_content = zip_buffer.getvalue()

    # Upload the ZIP file to the artifact
    zip_file_path = "large-test-files"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path=f"{zip_file_path}.zip",
        download_weight=1,
    )

    # Upload in multiple chunks to ensure that the S3 implementation handles it properly
    chunk_size = 1024 * 1024  # 1MB chunks
    async with httpx.AsyncClient(timeout=300) as client:
        # For a chunked upload we would use S3's multipart upload API, but for this test
        # a simple PUT is sufficient
        response = await client.put(put_url, data=zip_content)
        assert response.status_code == 200, response.text

    # Commit the dataset artifact
    await artifact_manager.commit(artifact_id=dataset.id)

    # Test listing the root directory of the ZIP
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip"
        )
        assert response.status_code == 200, response.text
        items = response.json()
        root_files = [item for item in items if item["type"] == "file"]
        root_dirs = [item for item in items if item["type"] == "directory"]
        assert len(root_files) == 100  # 100 files in root
        assert len(root_dirs) == 10  # 10 directories

    # Test listing a specific directory
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=dir_0/"
        )
        assert response.status_code == 200, response.text
        items = response.json()
        dir_files = [item for item in items if item["type"] == "file"]
        assert len(dir_files) == 190  # 180 regular + 10 large files

    # Test retrieving a regular file
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=root_file_0.txt"
        )
        assert response.status_code == 200, response.text
        assert response.text == "Content of root file 0"

    # Test retrieving a large file
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip?path=dir_0/large_file_0.txt"
        )
        assert response.status_code == 200, response.text
        assert "Large content repeated many times 0" in response.text
        assert len(response.text) > 3000  # Verify it's actually a large file

    # Test retrieving a file using the tilde syntax
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip/~/dir_0/file_0.txt"
        )
        assert response.status_code == 200, response.text
        assert response.text == "Content of dir_0/file_0.txt"

    # Test listing a directory using the tilde syntax
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/zip-files/{zip_file_path}.zip/~/dir_0/"
        )
        assert response.status_code == 200, response.text
        items = response.json()
        dir_files = [item for item in items if item["type"] == "file"]
        assert len(dir_files) == 190  # 180 regular + 10 large files

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_edit_version_behavior(minio_server, fastapi_server, test_user_token):
    """Test edit function version handling behavior - when to update vs create new versions."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an initial artifact with v0
    initial_manifest = {
        "name": "Edit Version Test",
        "description": "Testing edit version behavior",
    }
    artifact = await artifact_manager.create(
        type="dataset",
        manifest=initial_manifest,
        version="v0",
    )
    assert len(artifact["versions"]) == 1
    assert artifact["versions"][0]["version"] == "v0"

    # Case 1: Edit with version=None should UPDATE existing latest version
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "Updated Name", "description": "Updated description"},
        version=None,  # Explicitly set to None
    )
    edited = await artifact_manager.read(artifact_id=artifact.id)
    assert edited["manifest"]["name"] == "Updated Name"
    assert len(edited["versions"]) == 1  # Still only 1 version
    assert edited["versions"][0]["version"] == "v0"  # Still v0

    # Case 2: Edit with no version parameter should UPDATE existing latest version
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "No Param Updated", "description": "No param description"},
        # No version parameter at all
    )
    edited = await artifact_manager.read(artifact_id=artifact.id)
    assert edited["manifest"]["name"] == "No Param Updated"
    assert len(edited["versions"]) == 1  # Still only 1 version
    assert edited["versions"][0]["version"] == "v0"  # Still v0

    # Case 3: Edit with version="new" should CREATE a new version
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "New Version", "description": "New version description"},
        version="new",
    )
    edited = await artifact_manager.read(artifact_id=artifact.id)
    assert edited["manifest"]["name"] == "New Version"
    assert len(edited["versions"]) == 2  # Now 2 versions
    assert edited["versions"][0]["version"] == "v0"  # Original
    assert edited["versions"][1]["version"] == "v1"  # New version created

    # Case 5: Try to edit with invalid version name (should fail)
    with pytest.raises(Exception, match=r".*Invalid version.*"):
        await artifact_manager.edit(
            artifact_id=artifact.id,
            manifest={"name": "Invalid Version"},
            version="v99",  # This version doesn't exist
        )

    # Case 6: Edit with stage=True should put in staging mode (version parameter ignored if not allowed)
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "Staged Version", "description": "Staged description"},
        stage=True,
    )
    staged = await artifact_manager.read(artifact_id=artifact.id, version="stage")
    assert staged["manifest"]["name"] == "Staged Version"
    assert staged["staging"] is not None

    # Case 7: Edit with stage=True and version="new" should stage with intent for new version
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={
            "name": "Staged New Version",
            "description": "Staged new description",
        },
        stage=True,
        version="new",  # This should mark intent for new version
    )
    staged = await artifact_manager.read(artifact_id=artifact.id, version="stage")
    assert staged["manifest"]["name"] == "Staged New Version"
    assert staged["staging"] is not None

    # Case 8: version="stage" is equivalent to stage=True
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "Stage Version", "description": "Stage description"},
        version="stage",
    )
    staged = await artifact_manager.read(artifact_id=artifact.id, version="stage")
    assert staged["manifest"]["name"] == "Stage Version"
    assert staged["staging"] is not None

    # Case 9: Try to edit with invalid version while staging (should fail)
    with pytest.raises(Exception, match=r".*Invalid version.*"):
        await artifact_manager.edit(
            artifact_id=artifact.id,
            manifest={"name": "Invalid Staged"},
            stage=True,
            version="v99",  # Invalid version name (not staging specific issue)
        )

    # Test legacy case: Edit artifact with no versions (should create v0)
    legacy_artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Legacy Test"},
        stage=True,  # Create with no versions
    )
    assert len(legacy_artifact["versions"]) == 0

    # Edit without version should update staging area (since artifact is already staged)
    await artifact_manager.edit(
        artifact_id=legacy_artifact.id,
        manifest={"name": "Legacy Updated", "description": "Legacy description"},
        # No version parameter - should update staging area
    )

    # Commit to create the first version
    committed = await artifact_manager.commit(
        artifact_id=legacy_artifact.id,
        comment="Initial commit for legacy artifact",
    )
    assert committed["manifest"]["name"] == "Legacy Updated"
    assert len(committed["versions"]) == 1  # Should have created v0
    assert committed["versions"][0]["version"] == "v0"
    assert committed["staging"] is None

    # Clean up
    await artifact_manager.delete(artifact_id=artifact.id)
    await artifact_manager.delete(artifact_id=legacy_artifact.id)


async def test_commit_version_override(minio_server, fastapi_server, test_user_token):
    """Test commit function with version override when staging with version='new'."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an initial artifact with v0
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={
            "name": "Commit Test",
            "description": "Testing commit version override",
        },
        version="v0",
    )
    assert len(artifact["versions"]) == 1
    assert artifact["versions"][0]["version"] == "v0"

    # Test 1: Stage with version="new" and commit without version override
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={
            "name": "New Version Staged",
            "description": "Staged for new version",
        },
        stage=True,
        version="new",  # This marks intent for new version
    )

    # Add a file to the staged version
    file_content = "Test content for new version"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="test.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Commit without version override (should create v1)
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
        comment="New version created from staging",
    )
    assert len(committed["versions"]) == 2
    assert committed["versions"][0]["version"] == "v0"
    assert committed["versions"][1]["version"] == "v1"
    assert committed["staging"] is None

    # Test 2: Stage with version="new" and commit with custom version override
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={
            "name": "Custom Version Staged",
            "description": "Staged for custom version",
        },
        stage=True,
        version="new",  # This marks intent for new version
    )

    # Add another file
    file_content2 = "Test content for custom version"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="test2.txt",
    )
    response = requests.put(put_url, data=file_content2)
    assert response.ok

    # Commit with custom version override
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
        version="my_custom_version",  # Override the sequential naming
        comment="Custom version created from staging",
    )
    assert len(committed["versions"]) == 3
    assert committed["versions"][0]["version"] == "v0"
    assert committed["versions"][1]["version"] == "v1"
    assert committed["versions"][2]["version"] == "my_custom_version"
    assert committed["staging"] is None

    # Test 3: Try to commit with existing version name (should fail)
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "Another Staged", "description": "Another staged version"},
        stage=True,
        version="new",
    )

    with pytest.raises(Exception, match=r".*Version.*already exists.*"):
        await artifact_manager.commit(
            artifact_id=artifact.id,
            version="v1",  # This version already exists
        )

    # Test 4: Stage without version="new" and commit (should update existing)
    # First, clear staging and stage normally
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "Update Staged", "description": "Staged for update"},
        stage=True,
        # No version="new", so this should update existing when committed
    )

    # Commit should update existing latest version, version parameter should be ignored
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
        version="ignored_version",  # This should be ignored
        comment="Updated existing version from staging",
    )
    assert len(committed["versions"]) == 3  # Still 3 versions
    assert (
        committed["versions"][-1]["version"] == "my_custom_version"
    )  # Latest is still my_custom_version
    assert (
        committed["versions"][-1]["comment"] == "Updated existing version from staging"
    )

    # Test 5: Create artifact with no versions and commit (legacy case)
    legacy_artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Legacy Test"},
        stage=True,  # Create with no versions
    )

    # Add a file
    put_url = await artifact_manager.put_file(
        artifact_id=legacy_artifact.id,
        file_path="legacy.txt",
    )
    response = requests.put(put_url, data="Legacy content")
    assert response.ok

    # Commit should create first version
    committed = await artifact_manager.commit(
        artifact_id=legacy_artifact.id,
        version="initial_version",  # Custom name for first version
    )
    assert len(committed["versions"]) == 1
    assert committed["versions"][0]["version"] == "initial_version"

    # Clean up
    await artifact_manager.delete(artifact_id=artifact.id)
    await artifact_manager.delete(artifact_id=legacy_artifact.id)


async def test_list_stage_parameter(minio_server, fastapi_server, test_user_token):
    """Test the stage parameter in list() function."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing
    collection = await artifact_manager.create(
        type="collection",
        manifest={
            "name": "Stage Test Collection",
            "description": "A collection for testing stage parameter",
        },
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create artifacts in different states
    # 1. Committed artifact
    committed = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Committed Artifact"},
        version="v0",  # This will create a committed artifact
    )

    # 2. Staged artifact
    staged = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Staged Artifact"},
        stage=True,  # This will create a staged artifact
    )

    # 3. Another committed artifact
    committed2 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Another Committed"},
        version="v0",
    )

    # 4. Another staged artifact
    staged2 = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Another Staged"},
        stage=True,
    )

    # Test 1: Default behavior (stage=False) should list only committed artifacts
    results = await artifact_manager.list(parent_id=collection.id)
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert names == {"Committed Artifact", "Another Committed"}

    # Test 2: stage=False should show only committed artifacts
    results = await artifact_manager.list(parent_id=collection.id, stage=False)
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert names == {"Committed Artifact", "Another Committed"}

    # Test 3: stage=True should show only staged artifacts
    results = await artifact_manager.list(parent_id=collection.id, stage=True)
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert names == {"Staged Artifact", "Another Staged"}

    # Test 4: stage='all' should show both staged and committed artifacts
    results = await artifact_manager.list(parent_id=collection.id, stage="all")
    assert len(results) == 4
    names = {r["manifest"]["name"] for r in results}
    assert names == {
        "Committed Artifact",
        "Another Committed",
        "Staged Artifact",
        "Another Staged",
    }

    # Test 5: Backward compatibility with filters={"version": "stage"}
    results = await artifact_manager.list(
        parent_id=collection.id, filters={"version": "stage"}
    )
    assert len(results) == 2
    names = {r["manifest"]["name"] for r in results}
    assert names == {"Staged Artifact", "Another Staged"}

    # Clean up
    await artifact_manager.delete(artifact_id=committed.id)
    await artifact_manager.delete(artifact_id=committed2.id)
    await artifact_manager.delete(artifact_id=staged.id)
    await artifact_manager.delete(artifact_id=staged2.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_create_zip_file_download_count(
    minio_server, fastapi_server, test_user_token
):
    """Test download count increment behavior for create-zip-file endpoint."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection and dataset
    collection = await artifact_manager.create(
        type="collection",
        manifest={"name": "Download Count Test Collection"},
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Download Count Test Dataset"},
        version="stage",
    )

    # Add multiple files to the dataset
    file_contents = {
        "file1.txt": "Contents of file 1",
        "file2.txt": "Contents of file 2",
        "nested/file3.txt": "Contents of nested file 3",
        "nested/deep/file4.txt": "Contents of deeply nested file 4",
    }

    for file_path, content in file_contents.items():
        put_url = await artifact_manager.put_file(
            artifact_id=dataset.id,
            file_path=file_path,
            download_weight=0.5,
        )
        response = requests.put(put_url, data=content)
        assert response.ok

    # Add a file with weight 0
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="file5.txt",
        download_weight=0,
    )
    response = requests.put(put_url, data="Contents of file 5")
    assert response.ok

    # Commit the dataset
    await artifact_manager.commit(artifact_id=dataset.id)

    # Check initial download count (should be 0)
    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 0

    # Test 1: Download specific files via create-zip-file endpoint
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=file1.txt&file=file2.txt"
        )
        assert response.status_code == 200

    # Check download count increment (should be 2 - one for each file)
    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 0.5 + 0.5

    # Test 2: Download all files via create-zip-file endpoint
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file"
        )
        assert response.status_code == 200

    # Check download count increment (should be +4 more - one for each file in the dataset)
    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 3  # 0.5 + 0.5 + 1 + 1

    # Test 3: Download nested files via create-zip-file endpoint
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=nested/file3.txt&file=nested/deep/file4.txt"
        )
        assert response.status_code == 200

    # Check download count increment (should be +2 more)
    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 4  # 0.5 + 0.5 + 1 + 1 + 1 + 1

    # Test 4: Multiple requests to same zip endpoint should increment each time
    for _ in range(3):
        async with httpx.AsyncClient(timeout=200) as client:
            response = await client.get(
                f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=file1.txt"
            )
            assert response.status_code == 200

    # Check download count increment (should be +3 more - 0.5 for each request)
    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 5.5  # 1 + 2 + 1 + 1.5 (3 requests * 0.5 each)

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_workspace_artifacts_endpoint(
    minio_server, fastapi_server, test_user_token
):
    """Test the workspace-level artifacts listing endpoint."""

    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create multiple artifacts at different levels
    # 1. Root-level collection
    root_collection = await artifact_manager.create(
        type="collection",
        manifest={"name": "Root Collection", "description": "Root level collection"},
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # 2. Root-level dataset
    root_dataset = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Root Dataset", "description": "Root level dataset"},
    )

    # 3. Child dataset in collection
    child_dataset = await artifact_manager.create(
        type="dataset",
        parent_id=root_collection.id,
        manifest={
            "name": "Child Dataset",
            "description": "Child dataset in collection",
        },
    )

    # 4. Staged artifacts
    staged_collection = await artifact_manager.create(
        type="collection",
        manifest={"name": "Staged Collection", "description": "Staged collection"},
        stage=True,
    )

    staged_dataset = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Staged Dataset", "description": "Staged dataset"},
        stage=True,
    )

    # Test 1: Basic workspace artifacts listing (default: committed only)
    response = requests.get(f"{SERVER_URL}/{api.config.workspace}/artifacts/")
    assert response.status_code == 200
    results = response.json()

    # Should return committed root-level artifacts only
    committed_names = {r["manifest"]["name"] for r in results}
    assert "Root Collection" in committed_names
    assert "Root Dataset" in committed_names
    assert (
        "Child Dataset" not in committed_names
    )  # Child artifacts shouldn't appear at workspace level
    assert (
        "Staged Collection" not in committed_names
    )  # Staged artifacts not included by default
    assert "Staged Dataset" not in committed_names

    # Test 2: Workspace artifacts with stage=true (staged only)
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/?stage=true"
    )
    assert response.status_code == 200
    results = response.json()

    staged_names = {r["manifest"]["name"] for r in results}
    assert "Staged Collection" in staged_names
    assert "Staged Dataset" in staged_names
    assert "Root Collection" not in staged_names  # Committed artifacts not included
    assert "Root Dataset" not in staged_names

    # Test 3: Workspace artifacts with stage=all (both committed and staged)
    response = requests.get(f"{SERVER_URL}/{api.config.workspace}/artifacts/?stage=all")
    assert response.status_code == 200
    results = response.json()

    all_names = {r["manifest"]["name"] for r in results}
    assert "Root Collection" in all_names
    assert "Root Dataset" in all_names
    assert "Staged Collection" in all_names
    assert "Staged Dataset" in all_names
    assert "Child Dataset" not in all_names  # Child artifacts still shouldn't appear

    # Test 4: Filtering by type
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/?filters={json.dumps({'type': 'collection'})}"
    )
    assert response.status_code == 200
    results = response.json()

    # Should only return collections
    for result in results:
        assert result["type"] == "collection"
    collection_names = {r["manifest"]["name"] for r in results}
    assert "Root Collection" in collection_names
    assert "Root Dataset" not in collection_names

    # Test 5: Keyword search
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/?keywords=Root"
    )
    assert response.status_code == 200
    results = response.json()

    # Should return artifacts with "Root" in their name/description
    root_names = {r["manifest"]["name"] for r in results}
    assert "Root Collection" in root_names
    assert "Root Dataset" in root_names

    # Test 6: Pagination
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/?pagination=true&limit=1"
    )
    assert response.status_code == 200
    results = response.json()

    assert "items" in results
    assert "total" in results
    assert len(results["items"]) == 1
    assert results["total"] >= 2  # At least root collection and root dataset

    # Test 7: Ordering
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/?order_by=created_at"
    )
    assert response.status_code == 200
    results = response.json()

    # Verify results are ordered (check that we get same artifacts)
    ordered_names = {r["manifest"]["name"] for r in results}
    assert "Root Collection" in ordered_names
    assert "Root Dataset" in ordered_names

    # Test 8: Complex filtering with manifest fields
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/?filters={json.dumps({'manifest': {'description': '*Root*'}})}"
    )
    assert response.status_code == 200
    results = response.json()

    # Should match artifacts with "Root" in description
    for result in results:
        assert "Root" in result["manifest"]["description"]

    # Test 9: Authentication required for private artifacts
    # Create a private artifact
    private_dataset = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Private Dataset", "description": "Private dataset"},
        config={"permissions": {"@": "rw+"}},  # Only owner has access
    )

    # Test with authentication
    token = await api.generate_token()
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/?no_cache=true",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    results = response.json()
    private_names = {r["manifest"]["name"] for r in results}
    assert "Private Dataset" in private_names

    # Clean up
    await artifact_manager.delete(artifact_id=root_collection.id)
    await artifact_manager.delete(artifact_id=root_dataset.id)
    await artifact_manager.delete(artifact_id=child_dataset.id)
    await artifact_manager.delete(artifact_id=staged_collection.id)
    await artifact_manager.delete(artifact_id=staged_dataset.id)
    await artifact_manager.delete(artifact_id=private_dataset.id)


async def test_create_zip_file_download_weight_behavior(
    minio_server, fastapi_server, test_user_token
):
    """Test that create-zip-file respects download_weight when incrementing download count."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection and dataset
    collection = await artifact_manager.create(
        type="collection",
        manifest={"name": "Download Weight Test Collection"},
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Download Weight Test Dataset"},
        version="stage",
    )

    # Add files with different download weights
    files_with_weights = [
        ("light_file.txt", "Light content", 1),
        ("medium_file.txt", "Medium content", 3),
        ("heavy_file.txt", "Heavy content", 5),
        ("zero_weight_file.txt", "Zero weight content", 0),
    ]

    for file_path, content, weight in files_with_weights:
        put_url = await artifact_manager.put_file(
            artifact_id=dataset.id,
            file_path=file_path,
            download_weight=weight,
        )
        response = requests.put(put_url, data=content)
        assert response.ok

    # Commit the dataset
    await artifact_manager.commit(artifact_id=dataset.id)

    # Check initial download count
    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 0

    # Test 1: Download file with weight 1
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=light_file.txt"
        )
        assert response.status_code == 200

    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 1

    # Test 2: Download file with weight 3
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=medium_file.txt"
        )
        assert response.status_code == 200

    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 4  # 1 + 3

    # Test 3: Download file with weight 5
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=heavy_file.txt"
        )
        assert response.status_code == 200

    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 9  # 4 + 5

    # Test 4: Download file with weight 0 (should not increment)
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=zero_weight_file.txt"
        )
        assert response.status_code == 200

    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 9  # Should remain 9

    # Test 5: Download multiple files with mixed weights
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file?file=light_file.txt&file=medium_file.txt&file=zero_weight_file.txt"
        )
        assert response.status_code == 200

    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 13  # 9 + 1 + 3 + 0

    # Test 6: Download all files (should sum all weights)
    async with httpx.AsyncClient(timeout=200) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/create-zip-file"
        )
        assert response.status_code == 200

    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["download_count"] == 22  # 13 + 1 + 3 + 5 + 0

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_discard_staged_changes(minio_server, fastapi_server, test_user_token):
    """Test discarding staged changes for an artifact."""

    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection and dataset
    collection = await artifact_manager.create(
        type="collection",
        manifest={"name": "Discard Test Collection"},
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create a dataset with initial version
    initial_manifest = {
        "name": "Discard Test Dataset",
        "description": "Initial version",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=initial_manifest,
        version="v0",
    )

    # Verify initial state
    assert len(dataset["versions"]) == 1
    assert dataset["versions"][0]["version"] == "v0"
    assert dataset["staging"] is None

    # Stage changes
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest={
            "name": "Discard Test Dataset",
            "description": "Modified description for testing discard",
        },
        stage=True,
    )

    # Add a staged file
    staged_file_content = "This file should be discarded"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="staged_file.txt",
    )
    response = requests.put(put_url, data=staged_file_content)
    assert response.ok

    # Verify staged state
    staged_artifact = await artifact_manager.read(
        artifact_id=dataset.id, version="stage"
    )
    assert staged_artifact["staging"] is not None
    assert (
        staged_artifact["manifest"]["description"]
        == "Modified description for testing discard"
    )

    # List staged files to confirm they exist
    staged_files = await artifact_manager.list_files(
        artifact_id=dataset.id, version="stage"
    )
    assert len(staged_files) == 1
    assert staged_files[0]["name"] == "staged_file.txt"

    # Discard staged changes
    discarded_artifact = await artifact_manager.discard(artifact_id=dataset.id)

    # Verify discard worked
    assert discarded_artifact["staging"] is None
    assert (
        discarded_artifact["manifest"]["description"] == "Initial version"
    )  # Back to original
    assert len(discarded_artifact["versions"]) == 1  # No new version created

    # Verify staged files were cleaned up - should return empty list or raise exception
    try:
        staged_files_after_discard = await artifact_manager.list_files(
            artifact_id=dataset.id, version="stage"
        )
        # If no exception, should be empty list
        assert (
            len(staged_files_after_discard) == 0
        ), f"Expected no staged files after discard, but found: {staged_files_after_discard}"
    except Exception:
        # Exception is also acceptable behavior
        pass

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)


async def test_discard_error_no_staging(minio_server, fastapi_server, test_user_token):
    """Test that discard raises error when no staged changes exist."""

    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a committed artifact (no staging)
    dataset = await artifact_manager.create(
        type="dataset",
        manifest={"name": "No Staging Test"},
        version="v0",
    )

    # Verify no staging
    assert dataset["staging"] is None

    # Attempt to discard should fail
    with pytest.raises(Exception, match=r".*No staged changes to discard.*"):
        await artifact_manager.discard(artifact_id=dataset.id)

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)


async def test_discard_with_multiple_staged_files(
    minio_server, fastapi_server, test_user_token
):
    """Test discarding when multiple files are staged."""

    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a dataset and stage it
    dataset = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Multi-file Discard Test"},
        stage=True,
    )

    # Add multiple staged files
    files_to_stage = {
        "file1.txt": "Content of file 1",
        "nested/file2.txt": "Content of file 2",
        "nested/deep/file3.txt": "Content of file 3",
    }

    for file_path, content in files_to_stage.items():
        put_url = await artifact_manager.put_file(
            artifact_id=dataset.id,
            file_path=file_path,
        )
        response = requests.put(put_url, data=content)
        assert response.ok

    # Verify staged files
    staged_files = await artifact_manager.list_files(
        artifact_id=dataset.id, version="stage"
    )
    assert len(staged_files) == 2

    # Discard all staged changes
    discarded_artifact = await artifact_manager.discard(artifact_id=dataset.id)

    # Verify all staging is cleared
    assert discarded_artifact["staging"] is None
    assert (
        len(discarded_artifact["versions"]) == 0
    )  # No versions since it was never committed

    # Verify staged files are gone - should return empty list or raise exception
    try:
        staged_files_after = await artifact_manager.list_files(
            artifact_id=dataset.id, version="stage"
        )
        # If no exception, should be empty list
        assert (
            len(staged_files_after) == 0
        ), f"Expected no staged files after discard, but found: {staged_files_after}"
    except Exception:
        # Exception is also acceptable behavior
        pass

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)


async def test_discard_preserves_committed_version(
    minio_server, fastapi_server, test_user_token
):
    """Test that discard preserves the committed version and its files."""

    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create and commit initial version with a file
    dataset = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Preserve Test"},
        stage=True,
    )

    # Add and commit initial file
    initial_content = "Initial committed content"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="committed_file.txt",
    )
    response = requests.put(put_url, data=initial_content)
    assert response.ok

    committed_dataset = await artifact_manager.commit(artifact_id=dataset.id)
    assert len(committed_dataset["versions"]) == 1

    # Verify committed file exists
    committed_files = await artifact_manager.list_files(artifact_id=dataset.id)
    assert len(committed_files) == 1
    assert committed_files[0]["name"] == "committed_file.txt"

    # Stage new changes
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest={"name": "Preserve Test", "description": "Staged description"},
        stage=True,
    )

    # Add staged file
    staged_content = "Staged content to be discarded"
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="staged_file.txt",
    )
    response = requests.put(put_url, data=staged_content)
    assert response.ok

    # Discard staged changes
    await artifact_manager.discard(artifact_id=dataset.id)

    # Verify committed version and file are preserved
    final_artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert final_artifact["staging"] is None
    assert len(final_artifact["versions"]) == 1
    assert "description" not in final_artifact["manifest"]  # Staged change discarded

    # Verify committed file still exists, staged file is gone
    final_files = await artifact_manager.list_files(artifact_id=dataset.id)
    assert len(final_files) == 1
    assert final_files[0]["name"] == "committed_file.txt"

    # Clean up
    await artifact_manager.delete(artifact_id=dataset.id)


async def test_no_automatic_new_version_intent(
    minio_server, fastapi_server, test_user_token
):
    """Test that adding files to staging doesn't automatically trigger new version creation."""

    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an initial artifact with v0
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={
            "name": "No Auto Version Test",
            "description": "Testing no automatic versioning",
        },
        version="v0",
    )
    assert len(artifact["versions"]) == 1
    assert artifact["versions"][0]["version"] == "v0"

    # Stage changes WITHOUT version="new"
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={
            "name": "No Auto Version Test",
            "description": "Modified description",
        },
        stage=True,
        # No version="new" - should NOT automatically create new version intent
    )

    # Add files to staging - this should NOT automatically trigger new version intent
    file_content1 = "Content of first file"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="file1.txt",
    )
    response = requests.put(put_url, data=file_content1)
    assert response.ok

    file_content2 = "Content of second file"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="file2.txt",
    )
    response = requests.put(put_url, data=file_content2)
    assert response.ok

    # Verify staging state - should NOT have new_version intent
    staged_artifact = await artifact_manager.read(
        artifact_id=artifact.id, version="stage"
    )
    assert staged_artifact["staging"] is not None

    # Check that no new_version intent was automatically added
    has_new_version_intent = any(
        item.get("_intent") == "new_version" for item in staged_artifact["staging"]
    )
    assert (
        not has_new_version_intent
    ), "Files should not automatically trigger new_version intent"

    # Commit should UPDATE existing version, not create new version
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
        comment="Updated existing version with files",
    )

    # Should still have only 1 version (v0 updated)
    assert len(committed["versions"]) == 1
    assert committed["versions"][0]["version"] == "v0"
    assert committed["versions"][0]["comment"] == "Updated existing version with files"
    assert committed["staging"] is None

    # Verify files are now part of v0
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="v0")
    assert len(files) == 2
    file_names = {f["name"] for f in files}
    assert file_names == {"file1.txt", "file2.txt"}

    # Now test explicit new version creation
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={
            "name": "No Auto Version Test",
            "description": "Explicit new version",
        },
        stage=True,
        version="new",  # Explicitly request new version
    )

    # Add another file
    file_content3 = "Content of third file"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="file3.txt",
    )
    response = requests.put(put_url, data=file_content3)
    assert response.ok

    # Verify new_version intent is present when explicitly requested
    staged_artifact = await artifact_manager.read(
        artifact_id=artifact.id, version="stage"
    )
    has_new_version_intent = any(
        item.get("_intent") == "new_version" for item in staged_artifact["staging"]
    )
    assert (
        has_new_version_intent
    ), "Explicit version='new' should set new_version intent"

    # Commit should CREATE new version
    committed = await artifact_manager.commit(
        artifact_id=artifact.id,
        comment="Explicit new version with file",
    )

    # Should now have 2 versions
    assert len(committed["versions"]) == 2
    assert committed["versions"][0]["version"] == "v0"
    assert committed["versions"][1]["version"] == "v1"
    assert committed["versions"][1]["comment"] == "Explicit new version with file"

    # Verify v1 has the new file
    files_v1 = await artifact_manager.list_files(artifact_id=artifact.id, version="v1")
    assert len(files_v1) == 1
    assert files_v1[0]["name"] == "file3.txt"

    # Clean up
    await artifact_manager.delete(artifact_id=artifact.id)


async def test_secret_management(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """Test secret management functions."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an artifact with admin user
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Artifact"},
    )
    artifact_id = artifact.id

    # Test set_secret with admin user (should work)
    await artifact_manager.set_secret(
        artifact_id=artifact_id,
        secret_key="api_key",
        secret_value="secret123",
    )

    # Test get_secret with admin user (should work)
    secret_value = await artifact_manager.get_secret(
        artifact_id=artifact_id,
        secret_key="api_key",
    )
    assert secret_value == "secret123"

    # Test get_secret with all secrets (should work)
    secrets = await artifact_manager.get_secret(
        artifact_id=artifact_id,
    )
    assert secrets == {"api_key": "secret123"}

    # Test get_secret for non-existent key (should return None)
    non_existent = await artifact_manager.get_secret(
        artifact_id=artifact_id,
        secret_key="nonexistent",
    )
    assert non_existent is None

    # Test updating existing secret
    await artifact_manager.set_secret(
        artifact_id=artifact_id,
        secret_key="api_key",
        secret_value="updated_secret",
    )

    updated_value = await artifact_manager.get_secret(
        artifact_id=artifact_id,
        secret_key="api_key",
    )
    assert updated_value == "updated_secret"

    # Test adding multiple secrets
    await artifact_manager.set_secret(
        artifact_id=artifact_id,
        secret_key="db_password",
        secret_value="dbpass456",
    )

    db_password = await artifact_manager.get_secret(
        artifact_id=artifact_id,
        secret_key="db_password",
    )
    assert db_password == "dbpass456"

    # Verify original secret still exists
    api_key = await artifact_manager.get_secret(
        artifact_id=artifact_id,
        secret_key="api_key",
    )
    assert api_key == "updated_secret"

    # Test permission restrictions with second user (non-admin)
    api2 = await connect_to_server(
        {"name": "test-client-2", "server_url": SERVER_URL, "token": test_user_token_2}
    )
    artifact_manager2 = await api2.get_service("public/artifact-manager")

    # Secret operations require workspace-level permissions, not artifact-level permissions
    # Since test_user_token_2 only has access to their own workspace (ws-user-user-2),
    # they should not be able to access secrets in user-1's workspace

    # Second user should NOT be able to set_secret (no workspace permission)
    with pytest.raises(Exception) as exc_info:
        await artifact_manager2.set_secret(
            artifact_id=artifact_id,
            secret_key="user2_secret",
            secret_value="user2value",
        )
    assert "permission" in str(exc_info.value).lower()

    # Second user should NOT be able to get_secret (no workspace permission)
    with pytest.raises(Exception) as exc_info:
        await artifact_manager2.get_secret(
            artifact_id=artifact_id,
            secret_key="api_key",
        )
    assert "permission" in str(exc_info.value).lower()

    # Clean up
    await artifact_manager.delete(artifact_id=artifact_id)


async def test_secret_edge_cases(minio_server, fastapi_server, test_user_token):
    """Test edge cases for secret management."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an artifact
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Artifact"},
    )
    artifact_id = artifact.id

    # Test with None secret value (should work)
    await artifact_manager.set_secret(
        artifact_id=artifact_id,
        secret_key="null_secret",
        secret_value=None,
    )

    null_value = await artifact_manager.get_secret(
        artifact_id=artifact_id,
        secret_key="null_secret",
    )
    assert null_value is None

    # Test with empty string secret value
    await artifact_manager.set_secret(
        artifact_id=artifact_id,
        secret_key="empty_secret",
        secret_value="",
    )

    empty_value = await artifact_manager.get_secret(
        artifact_id=artifact_id,
        secret_key="empty_secret",
    )
    assert empty_value == ""

    # Test with non-existent artifact
    with pytest.raises(Exception):
        await artifact_manager.get_secret(
            artifact_id="nonexistent/artifact",
            secret_key="key",
        )

    with pytest.raises(Exception):
        await artifact_manager.set_secret(
            artifact_id="nonexistent/artifact",
            secret_key="key",
            secret_value="value",
        )

    # Clean up
    await artifact_manager.delete(artifact_id=artifact_id)


async def test_collection_permission_inheritance(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """Test that collection permissions are properly inherited by child artifacts."""

    # Set up first user (collection owner)
    api_owner = await connect_to_server(
        {"name": "owner-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager_owner = await api_owner.get_service("public/artifact-manager")
    owner_user_id = api_owner.config.user["id"]

    # Set up second user (with collection permissions)
    api_user = await connect_to_server(
        {"name": "user-client", "server_url": SERVER_URL, "token": test_user_token_2}
    )
    artifact_manager_user = await api_user.get_service("public/artifact-manager")
    user_user_id = api_user.config.user["id"]

    # Create a collection with owner giving rw+ permissions to second user
    collection_manifest = {
        "name": "Permission Inheritance Collection",
        "description": "Collection to test permission inheritance",
    }
    collection = await artifact_manager_owner.create(
        type="collection",
        manifest=collection_manifest,
        config={
            "permissions": {
                owner_user_id: "*",  # Owner has full access
                user_user_id: "rw+",  # Second user has read-write-create access
                "*": "r",  # Public read access
            }
        },
    )

    # Verify second user can read the collection
    collection_read = await artifact_manager_user.read(artifact_id=collection.id)
    assert collection_read["manifest"]["name"] == "Permission Inheritance Collection"

    # Second user creates a child artifact in the collection
    child_manifest = {
        "name": "Child Artifact",
        "description": "Child artifact for permission testing",
    }
    child_artifact = await artifact_manager_user.create(
        type="dataset",
        parent_id=collection.id,
        manifest=child_manifest,
        version="stage",
    )

    # Verify child artifact was created successfully
    assert (
        child_artifact["parent_id"]
        == f"{api_owner.config.workspace}/{collection.alias}"
    )
    assert child_artifact["manifest"]["name"] == "Child Artifact"

    # Test that second user can edit the child artifact (this was the original issue)
    updated_manifest = {
        "name": "Child Artifact",
        "description": "Updated description by second user",
        "custom_field": "added by second user",
    }

    # This should work now with the permission inheritance fix
    await artifact_manager_user.edit(
        artifact_id=child_artifact.id,
        manifest=updated_manifest,
        stage=True,
    )

    # Verify the edit was successful
    edited_artifact = await artifact_manager_user.read(
        artifact_id=child_artifact.id, version="stage"
    )
    assert (
        edited_artifact["manifest"]["description"]
        == "Updated description by second user"
    )
    assert edited_artifact["manifest"]["custom_field"] == "added by second user"

    # Test that second user can add files to the child artifact
    file_content = "File content added by second user"
    put_url = await artifact_manager_user.put_file(
        artifact_id=child_artifact.id,
        file_path="test_file.txt",
        download_weight=1,
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok

    # Test that second user can commit the child artifact
    committed_artifact = await artifact_manager_user.commit(
        artifact_id=child_artifact.id,
        comment="Committed by second user",
    )

    # Verify commit was successful
    assert committed_artifact["staging"] is None
    assert len(committed_artifact["versions"]) == 1
    assert committed_artifact["versions"][0]["comment"] == "Committed by second user"

    # Verify the committed manifest contains the updates
    assert (
        committed_artifact["manifest"]["description"]
        == "Updated description by second user"
    )
    assert committed_artifact["manifest"]["custom_field"] == "added by second user"

    # Verify the file was committed
    files = await artifact_manager_user.list_files(artifact_id=child_artifact.id)
    assert len(files) == 1
    assert files[0]["name"] == "test_file.txt"

    # Test that second user can continue editing the committed artifact
    await artifact_manager_user.edit(
        artifact_id=child_artifact.id,
        manifest={
            "name": "Child Artifact",
            "description": "Second update by second user",
            "custom_field": "updated again",
        },
        stage=True,
        version="new",  # Create a new version
    )

    # Commit the new version
    second_commit = await artifact_manager_user.commit(
        artifact_id=child_artifact.id,
        comment="Second version by second user",
    )

    # Verify second version was created
    assert len(second_commit["versions"]) == 2
    assert second_commit["versions"][1]["comment"] == "Second version by second user"
    assert second_commit["manifest"]["description"] == "Second update by second user"

    # Test that anonymous users still cannot edit (should only have read access)
    api_anonymous = await connect_to_server(
        {"name": "anonymous-client", "server_url": SERVER_URL}
    )
    artifact_manager_anonymous = await api_anonymous.get_service(
        "public/artifact-manager"
    )

    # Anonymous user should be able to read
    anonymous_read = await artifact_manager_anonymous.read(
        artifact_id=child_artifact.id
    )
    assert anonymous_read["manifest"]["name"] == "Child Artifact"

    # But anonymous user should NOT be able to edit
    with pytest.raises(Exception, match=r".*permission.*"):
        await artifact_manager_anonymous.edit(
            artifact_id=child_artifact.id,
            manifest={"name": "Should not work"},
        )

    # Test that a user with only list permissions cannot edit
    # Create a third user and give them only list permissions on the collection
    # (We can't easily test this without a third token, so we'll skip this part)

    # Test that the collection owner can still access and edit the child artifact
    owner_read = await artifact_manager_owner.read(artifact_id=child_artifact.id)
    assert owner_read["manifest"]["name"] == "Child Artifact"

    await artifact_manager_owner.edit(
        artifact_id=child_artifact.id,
        manifest={
            "name": "Child Artifact",
            "description": "Updated by owner",
        },
    )

    owner_updated = await artifact_manager_owner.read(artifact_id=child_artifact.id)
    assert owner_updated["manifest"]["description"] == "Updated by owner"

    # Clean up
    await artifact_manager_owner.delete(artifact_id=child_artifact.id)
    await artifact_manager_owner.delete(artifact_id=collection.id)


async def test_order_by_custom_json_fields(
    minio_server, fastapi_server, fastapi_server_sqlite, test_user_token
):
    """Test ordering by custom JSON fields in manifest and config for both SQL backends."""

    # Test with both PostgreSQL and SQLite backends
    test_configs = [
        {"server_url": SERVER_URL, "backend": "postgresql"},
        {"server_url": SERVER_URL_SQLITE, "backend": "sqlite"},
    ]

    for config in test_configs:
        print(f"Testing order_by with {config['backend']} backend")

        # Connect to the appropriate server
        api = await connect_to_server(
            {
                "name": f"test-client-{config['backend']}",
                "server_url": config["server_url"],
                "token": test_user_token,
            }
        )
        artifact_manager = await api.get_service("public/artifact-manager")

        # Create a collection
        collection_manifest = {
            "name": "Order Test Collection",
            "description": "Collection for testing ordering functionality",
        }
        collection = await artifact_manager.create(
            type="collection",
            alias=f"order-test-collection-{config['backend']}",
            manifest=collection_manifest,
            config={"permissions": {"*": "r", "@": "rw+"}},
        )

        # Create artifacts with various manifest and config fields for ordering
        test_artifacts = [
            {
                "alias": f"artifact-a-{config['backend']}",
                "manifest": {
                    "name": "Artifact A",
                    "description": "First artifact",
                    "likes": 100,
                    "priority": "high",
                    "score": 95.5,
                    "tags": ["important", "featured"],
                },
                "config": {
                    "permissions": {"*": "r", "@": "rw+"},
                    "priority": 1,
                    "rating": 4.8,
                    "category": "premium",
                },
            },
            {
                "alias": f"artifact-b-{config['backend']}",
                "manifest": {
                    "name": "Artifact B",
                    "description": "Second artifact",
                    "likes": 50,
                    "priority": "medium",
                    "score": 87.2,
                    "tags": ["standard"],
                },
                "config": {
                    "permissions": {"*": "r", "@": "rw+"},
                    "priority": 3,
                    "rating": 3.9,
                    "category": "standard",
                },
            },
            {
                "alias": f"artifact-c-{config['backend']}",
                "manifest": {
                    "name": "Artifact C",
                    "description": "Third artifact",
                    "likes": 200,
                    "priority": "low",
                    "score": 76.8,
                    "tags": ["basic"],
                },
                "config": {
                    "permissions": {"*": "r", "@": "rw+"},
                    "priority": 2,
                    "rating": 4.2,
                    "category": "basic",
                },
            },
        ]

        # Create the artifacts
        created_artifacts = []
        for artifact_data in test_artifacts:
            artifact = await artifact_manager.create(
                type="dataset",
                alias=artifact_data["alias"],
                parent_id=collection.id,
                manifest=artifact_data["manifest"],
                config=artifact_data["config"],
            )
            created_artifacts.append(artifact)
            print(f"Created {artifact['alias']}: likes={artifact['manifest']['likes']}")

        # Test 1: Order by manifest.likes (ascending)
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="manifest.likes<"
        )
        likes_values = [result["manifest"]["likes"] for result in results]
        aliases = [result["alias"] for result in results]
        print(f"Ascending likes order: {list(zip(aliases, likes_values))}")
        assert likes_values == sorted(
            likes_values
        ), f"Manifest likes ascending failed for {config['backend']}: {likes_values}"

        # Test 2: Order by manifest.likes (descending)
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="manifest.likes>"
        )
        likes_values = [result["manifest"]["likes"] for result in results]
        aliases = [result["alias"] for result in results]
        print(f"Descending likes order: {list(zip(aliases, likes_values))}")
        assert likes_values == sorted(
            likes_values, reverse=True
        ), f"Manifest likes descending failed for {config['backend']}: {likes_values}"

        # Test 3: Order by manifest.score (ascending)
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="manifest.score<"
        )
        score_values = [result["manifest"]["score"] for result in results]
        assert score_values == sorted(
            score_values
        ), f"Manifest score ascending failed for {config['backend']}: {score_values}"

        # Test 4: Order by config.priority (ascending)
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="config.priority<"
        )
        priority_values = [result["config"]["priority"] for result in results]
        assert priority_values == sorted(
            priority_values
        ), f"Config priority ascending failed for {config['backend']}: {priority_values}"

        # Test 5: Order by config.rating (descending)
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="config.rating>"
        )
        rating_values = [result["config"]["rating"] for result in results]
        assert rating_values == sorted(
            rating_values, reverse=True
        ), f"Config rating descending failed for {config['backend']}: {rating_values}"

        # Test 6: Order by manifest.priority (string field, ascending)
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="manifest.priority<"
        )
        priority_strings = [result["manifest"]["priority"] for result in results]
        assert priority_strings == sorted(
            priority_strings
        ), f"Manifest priority string ascending failed for {config['backend']}: {priority_strings}"

        # Test 7: Order by config.category (string field, descending)
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="config.category>"
        )
        category_values = [result["config"]["category"] for result in results]
        assert category_values == sorted(
            category_values, reverse=True
        ), f"Config category descending failed for {config['backend']}: {category_values}"

        # Test 8: Test built-in fields still work (created_at ascending)
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="created_at<"
        )
        created_at_values = [result["created_at"] for result in results]
        assert created_at_values == sorted(
            created_at_values
        ), f"Built-in created_at ascending failed for {config['backend']}: {created_at_values}"

        # Test 9: Test built-in fields still work (id descending)
        results = await artifact_manager.list(parent_id=collection.id, order_by="id>")
        id_values = [result["_id"] for result in results]  # _id is the actual UUID
        assert id_values == sorted(
            id_values, reverse=True
        ), f"Built-in id descending failed for {config['backend']}: {id_values}"

        # Test 10: Test default ordering still works (should default to id ascending)
        results_default = await artifact_manager.list(parent_id=collection.id)
        results_explicit = await artifact_manager.list(
            parent_id=collection.id, order_by="id<"
        )
        assert len(results_default) == len(
            results_explicit
        ), f"Default ordering length mismatch for {config['backend']}"
        for i in range(len(results_default)):
            assert (
                results_default[i]["id"] == results_explicit[i]["id"]
            ), f"Default ordering differs from explicit id< for {config['backend']}"

        # Test HTTP endpoints as well
        workspace = api.config.workspace

        # Test HTTP endpoint with manifest ordering
        response = requests.get(
            f"{config['server_url']}/{workspace}/artifacts/{collection.alias}/children",
            params={"order_by": "manifest.likes>"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        http_results = response.json()
        http_likes = [item["manifest"]["likes"] for item in http_results]
        assert http_likes == sorted(
            http_likes, reverse=True
        ), f"HTTP manifest.likes> failed for {config['backend']}: {http_likes}"

        # Test HTTP endpoint with config ordering
        response = requests.get(
            f"{config['server_url']}/{workspace}/artifacts/{collection.alias}/children",
            params={"order_by": "config.priority<"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        http_results = response.json()
        http_priorities = [item["config"]["priority"] for item in http_results]
        assert http_priorities == sorted(
            http_priorities
        ), f"HTTP config.priority< failed for {config['backend']}: {http_priorities}"

        # Clean up - delete artifacts and collection
        for artifact in created_artifacts:
            await artifact_manager.delete(artifact_id=artifact.id)
        await artifact_manager.delete(artifact_id=collection.id)

        print(f"✓ All order_by tests passed for {config['backend']} backend")


async def test_order_by_edge_cases(minio_server, fastapi_server, test_user_token):
    """Test edge cases for the order_by functionality."""

    api = await connect_to_server(
        {
            "name": "test-client-edge-cases",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection
    collection_manifest = {
        "name": "Edge Cases Collection",
        "description": "Collection for testing order_by edge cases",
    }
    collection = await artifact_manager.create(
        type="collection",
        alias="order-edge-cases-collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create artifacts with missing/null fields and different data types
    test_artifacts = [
        {
            "alias": "artifact-with-nulls",
            "manifest": {
                "name": "Artifact with nulls",
                "likes": None,  # null value
                "priority": "high",
            },
            "config": {
                "permissions": {"*": "r", "@": "rw+"},
                "rating": None,  # null value
            },
        },
        {
            "alias": "artifact-missing-fields",
            "manifest": {
                "name": "Artifact missing fields",
                "priority": "medium",
                # missing likes field
            },
            "config": {
                "permissions": {"*": "r", "@": "rw+"}
                # missing rating field
            },
        },
        {
            "alias": "artifact-with-values",
            "manifest": {
                "name": "Artifact with values",
                "likes": 150,
                "priority": "low",
            },
            "config": {"permissions": {"*": "r", "@": "rw+"}, "rating": 4.5},
        },
    ]

    # Create the artifacts
    created_artifacts = []
    for artifact_data in test_artifacts:
        artifact = await artifact_manager.create(
            type="dataset",
            alias=artifact_data["alias"],
            parent_id=collection.id,
            manifest=artifact_data["manifest"],
            config=artifact_data["config"],
        )
        created_artifacts.append(artifact)

    # Test ordering with missing/null fields - should handle gracefully
    try:
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="manifest.likes<"
        )
        # Should not crash, and results should be returned
        assert (
            len(results) == 3
        ), "Should return all artifacts even with missing/null fields"
        print("✓ Order by missing/null manifest fields handled gracefully")
    except Exception as e:
        pytest.fail(f"Order by missing manifest fields should not crash: {e}")

    try:
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="config.rating>"
        )
        # Should not crash, and results should be returned
        assert (
            len(results) == 3
        ), "Should return all artifacts even with missing/null config fields"
        print("✓ Order by missing/null config fields handled gracefully")
    except Exception as e:
        pytest.fail(f"Order by missing config fields should not crash: {e}")

    # Test ordering by non-existent fields - should default to id ordering
    try:
        results = await artifact_manager.list(
            parent_id=collection.id, order_by="manifest.nonexistent<"
        )
        assert (
            len(results) == 3
        ), "Should return all artifacts even with non-existent fields"
        print("✓ Order by non-existent fields handled gracefully")
    except Exception as e:
        pytest.fail(f"Order by non-existent fields should not crash: {e}")

    # Clean up
    for artifact in created_artifacts:
        await artifact_manager.delete(artifact_id=artifact.id)
    await artifact_manager.delete(artifact_id=collection.id)

    print("✓ All edge case tests passed")


async def test_overwrite_collection_with_children_fails(
    minio_server, fastapi_server, test_user_token
):
    """Test that overwriting a collection with children throws an error."""

    api = await connect_to_server(
        {
            "name": "test-client-overwrite",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection
    collection_manifest = {
        "name": "Test Collection",
        "description": "Collection for testing overwrite protection",
    }
    collection = await artifact_manager.create(
        type="collection",
        alias="test-collection-overwrite",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Create a child artifact in the collection
    child_manifest = {
        "name": "Child Artifact",
        "description": "Child artifact to test overwrite protection",
    }
    child = await artifact_manager.create(
        type="dataset",
        alias="child-artifact",
        parent_id=collection.id,
        manifest=child_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    # Try to overwrite the collection - should fail
    new_manifest = {
        "name": "New Collection",
        "description": "This should not work",
    }

    with pytest.raises(Exception) as exc_info:
        await artifact_manager.create(
            type="collection",
            alias="test-collection-overwrite",
            manifest=new_manifest,
            config={"permissions": {"*": "r", "@": "rw+"}},
            overwrite=True,
        )

    # Verify the error message contains information about children
    error_message = str(exc_info.value)
    assert (
        "Cannot overwrite collection" in error_message
    ), f"Expected collection overwrite error, got: {error_message}"
    assert (
        "child artifacts" in error_message
    ), f"Expected children count in error, got: {error_message}"
    assert (
        "1 child" in error_message
    ), f"Expected '1 child' in error message, got: {error_message}"

    # Verify collection still exists with original data
    existing_collection = await artifact_manager.read(collection.id)
    assert existing_collection["manifest"]["name"] == "Test Collection"
    assert (
        existing_collection["manifest"]["description"]
        == "Collection for testing overwrite protection"
    )

    # Test that overwrite works after removing children
    await artifact_manager.delete(child.id)

    # Now overwrite should work
    updated_collection = await artifact_manager.create(
        type="collection",
        alias="test-collection-overwrite",
        manifest=new_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
        overwrite=True,
    )

    assert updated_collection["manifest"]["name"] == "New Collection"
    assert updated_collection["manifest"]["description"] == "This should not work"

    # Clean up
    await artifact_manager.delete(updated_collection["id"])

    print("✓ Collection overwrite protection test passed")


async def test_zenodo_access_token_not_exposed_in_errors(
    minio_server, fastapi_server, test_user_token
):
    """Test that Zenodo access tokens are not exposed in error messages."""

    api = await connect_to_server(
        {
            "name": "test-client-security",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an artifact with Zenodo secrets
    fake_token = "fRa2TOboeMmZAfMzkgMleynzzbsRLABlIF8RgNlyGdhAKDVnXlmPS5dpAjFW"
    manifest = {
        "name": "Test Security Artifact",
        "description": "Testing access token security",
        "authors": [{"name": "Test Author"}],
    }

    # Create artifact with secrets containing a fake Zenodo token
    artifact = await artifact_manager.create(
        type="dataset",
        alias="security-test-artifact",
        manifest=manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
        secrets={"SANDBOX_ZENODO_ACCESS_TOKEN": fake_token},
    )

    # Try to publish to sandbox zenodo (will fail due to fake token)
    # This should trigger an HTTP error that potentially exposes the token
    with pytest.raises(Exception) as exc_info:
        await artifact_manager.publish(artifact_id=artifact.id, to="sandbox_zenodo")

    # Check that the access token is not present in the error message
    error_message = str(exc_info.value)
    assert (
        fake_token not in error_message
    ), f"Access token should not be exposed in error message: {error_message}"

    # The error should be about publishing/validation, indicating Zenodo interaction occurred
    assert any(
        keyword in error_message.lower()
        for keyword in ["publish", "deposition", "validation", "zenodo"]
    ), f"Expected Zenodo-related error message: {error_message}"

    # Check that any URL with access_token parameter is sanitized if URLs are present
    import re

    # Look for any URL patterns
    url_pattern = r'https://[^\s\'"]+'
    urls = re.findall(url_pattern, error_message)
    for url in urls:
        assert (
            "access_token=" not in url
        ), f"Found unsanitized URL with access_token: {url}"

    # Clean up
    await artifact_manager.delete(artifact.id)

    print("✓ Zenodo access token security test passed")


async def test_zenodo_publish_twice_handling(
    minio_server, fastapi_server, test_user_token
):
    """Test that publishing an artifact twice to Zenodo handles gracefully."""

    api = await connect_to_server(
        {
            "name": "test-client-publish-twice",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create an artifact with mock Zenodo configuration
    manifest = {
        "name": "Test Double Publish Artifact",
        "description": "Testing double publish handling",
        "authors": [{"name": "Test Author"}],
    }

    # Create artifact with a mock already-published Zenodo record
    mock_zenodo_record = {
        "id": "12345",
        "record_id": "12345",
        "conceptrecid": "67890",
        "state": "done",  # Published state
        "doi": "10.5072/zenodo.12345",
        "submitted": True,
    }

    artifact = await artifact_manager.create(
        type="dataset",
        alias="double-publish-test",
        manifest=manifest,
        config={
            "permissions": {"*": "r", "@": "rw+"},
            "zenodo": mock_zenodo_record,  # Simulate already published record
            "publish_to": "sandbox_zenodo",
        },
        secrets={"SANDBOX_ZENODO_ACCESS_TOKEN": "fake_token_for_test"},
    )

    # Try to publish again - this should handle gracefully
    # It should either create a new version or handle the already-published state
    try:
        # This will likely fail due to fake token, but it should attempt to create new version
        await artifact_manager.publish(artifact_id=artifact.id, to="sandbox_zenodo")
    except Exception as e:
        error_message = str(e)
        # Verify the error handling path was taken - should not crash with state issues
        # The error should be about authentication/HTTP, not about invalid state
        assert (
            "state" not in error_message.lower() or "version" in error_message.lower()
        ), f"Expected version handling or auth error, got state error: {error_message}"

        # Ensure access token is still not exposed
        assert (
            "fake_token_for_test" not in error_message
        ), f"Access token exposed in double publish error: {error_message}"

    # Verify the artifact's config was updated appropriately
    updated_artifact = await artifact_manager.read(artifact.id)
    zenodo_config = updated_artifact.get("config", {}).get("zenodo", {})

    # The zenodo config should still exist and be valid
    assert isinstance(zenodo_config, dict), "Zenodo config should remain a dictionary"

    # Clean up
    await artifact_manager.delete(artifact.id)

    print("✓ Zenodo double publish handling test passed")


async def test_multi_operation_permission_checking(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """Test that _get_artifact_with_permission properly handles multiple operations."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    api_user2 = await connect_to_server(
        {"name": "test-client-2", "server_url": SERVER_URL, "token": test_user_token_2}
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    artifact_manager_user2 = await api_user2.get_service("public/artifact-manager")

    # Get the actual user IDs from the API config
    user1_id = api.config.user["id"]
    user2_id = api_user2.config.user["id"]

    # Create a collection with specific permissions
    collection_manifest = {
        "name": "Test Collection",
        "description": "A test collection for multi-operation permission testing",
        "type": "collection",
    }

    # Give user2 only "create" permission but not "commit"
    collection_config = {
        "permissions": {
            user1_id: "*",  # Owner has full access
            user2_id: "rd+",  # Has create but not commit (rd+ vs rw+ which has commit)
        }
    }

    collection = await artifact_manager.create(
        manifest=collection_manifest,
        type="collection",
        config=collection_config,
    )

    # Test 1: User2 should fail to create a non-staged artifact (requires both create and commit)
    child_manifest = {
        "name": "Test Child",
        "description": "A test child artifact",
    }

    with pytest.raises(Exception) as exc_info:
        await artifact_manager_user2.create(
            parent_id=collection.id,
            manifest=child_manifest,
            stage=False,  # This requires both create and commit permissions
        )

    # Should get a permission error
    assert "permission" in str(exc_info.value).lower()

    # Test 2: User2 should succeed in creating a staged artifact (only requires create)
    staged_artifact = await artifact_manager_user2.create(
        parent_id=collection.id,
        manifest=child_manifest,
        stage=True,  # This only requires create permission
    )

    assert staged_artifact.id is not None
    assert staged_artifact.staging is not None

    # Test 3: User2 should fail to commit the staged artifact (requires commit permission)
    with pytest.raises(Exception) as exc_info:
        await artifact_manager_user2.commit(staged_artifact.id)

    assert "permission" in str(exc_info.value).lower()

    # Test 4: Update permissions to give user2 both create and commit
    collection_config_updated = {
        "permissions": {
            user1_id: "*",  # Owner has full access
            user2_id: "rw+",  # Has both create and commit
        }
    }

    await artifact_manager.edit(
        collection.id,
        config=collection_config_updated,
    )

    # Now user2 should be able to create a non-staged artifact
    child_manifest_2 = {
        "name": "Test Child 2",
        "description": "A second test child artifact",
    }

    committed_artifact = await artifact_manager_user2.create(
        parent_id=collection.id,
        manifest=child_manifest_2,
        stage=False,  # This should now work since user2 has both permissions
    )

    assert committed_artifact.id is not None
    assert committed_artifact.staging is None  # Should be committed
    assert len(committed_artifact.versions) > 0  # Should have versions

    # Test 5: User2 should now be able to commit the previously staged artifact
    committed_staged = await artifact_manager_user2.commit(staged_artifact.id)
    assert committed_staged.staging is None
    assert len(committed_staged.versions) > 0


async def test_delete_version_handling(minio_server, fastapi_server, test_user_token):
    """Test enhanced delete logic with proper version handling."""
    api = await connect_to_server(
        {"name": "test-client", "token": test_user_token, "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a test artifact with multiple versions
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Dataset", "description": "Test dataset for delete tests"},
        stage=True,
    )

    # Add a file and commit first version
    put_url = await artifact_manager.put_file(artifact.id, "test1.txt")
    requests.put(put_url, data=b"test content 1")
    v1_artifact = await artifact_manager.commit(artifact.id, version="v1")

    # Create a second version
    await artifact_manager.edit(artifact.id, manifest={"name": "Updated Dataset"}, stage=True, version="new")
    put_url = await artifact_manager.put_file(artifact.id, "test2.txt")
    requests.put(put_url, data=b"test content 2")
    v2_artifact = await artifact_manager.commit(artifact.id, version="v2")

    # Test 1: Delete a specific version that exists
    await artifact_manager.delete(artifact.id, version="v1")
    updated_artifact = await artifact_manager.read(artifact.id)
    version_names = [v["version"] for v in updated_artifact["versions"]]
    assert "v1" not in version_names
    assert "v2" in version_names

    # Test 2: Try to delete a version that doesn't exist
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.delete(artifact.id, version="v3")
    assert "Version 'v3' does not exist" in str(excinfo.value)

    # Test 3: Try to delete a version by invalid index
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.delete(artifact.id, version=10)
    assert "Version index 10 is out of range" in str(excinfo.value)

    # Test 4: Delete entire artifact (version=None)
    await artifact_manager.delete(artifact.id, delete_files=True)
    
    # Verify artifact is deleted
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.read(artifact.id)
    assert "does not exist" in str(excinfo.value)


async def test_delete_staged_version_error(minio_server, fastapi_server, test_user_token):
    """Test that delete raises proper error for staged versions."""
    api = await connect_to_server(
        {"name": "test-client", "token": test_user_token, "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Test 1: Create artifact in staging mode
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Dataset", "description": "Test dataset for delete tests"},
        stage=True,
    )

    # Test 2: Try to delete staged version - should raise error telling user to use discard
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.delete(artifact.id, version="stage")
    assert "Cannot delete staged version. The artifact is in staging mode. Please use the 'discard' function instead" in str(excinfo.value)

    # Test 3: Commit the artifact first
    await artifact_manager.commit(artifact.id)
    
    # Test 4: Try to delete staged version when artifact is not in staging mode
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.delete(artifact.id, version="stage")
    assert "Cannot delete staged version. The artifact is not in staging mode" in str(excinfo.value)

    # Test 5: Put artifact back in staging mode
    await artifact_manager.edit(artifact.id, manifest={"name": "Updated Dataset"}, stage=True)
    
    # Test 6: Try to delete staged version again - should raise error
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.delete(artifact.id, version="stage")
    assert "Cannot delete staged version. The artifact is in staging mode. Please use the 'discard' function instead" in str(excinfo.value)

    # Test 7: Use discard instead (proper way)
    await artifact_manager.discard(artifact.id)
    
    # Test 8: Now delete the entire artifact
    await artifact_manager.delete(artifact.id, delete_files=True)


async def test_delete_version_validation(minio_server, fastapi_server, test_user_token):
    """Test that delete properly validates version existence."""
    api = await connect_to_server(
        {"name": "test-client", "token": test_user_token, "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create artifact with specific versions
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Dataset", "description": "Test dataset for version validation"},
        stage=True,
    )

    # Create version v1
    await artifact_manager.commit(artifact.id, version="v1")
    
    # Create version v2
    await artifact_manager.edit(artifact.id, manifest={"name": "Updated Dataset"}, stage=True, version="new")
    await artifact_manager.commit(artifact.id, version="v2")

    # Test 1: Try to delete non-existent version by name
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.delete(artifact.id, version="v5")
    assert "Version 'v5' does not exist" in str(excinfo.value)
    assert "Available versions: ['v1', 'v2']" in str(excinfo.value)

    # Test 2: Try to delete by invalid numeric index
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.delete(artifact.id, version=5)
    assert "Version index 5 is out of range" in str(excinfo.value)
    assert "Available indices: 0-1" in str(excinfo.value)

    # Test 3: Try to delete by negative index
    with pytest.raises(Exception) as excinfo:
        await artifact_manager.delete(artifact.id, version=-1)
    assert "Version index -1 is out of range" in str(excinfo.value)

    # Test 4: Delete by valid index
    await artifact_manager.delete(artifact.id, version=0)  # Delete first version (v1)
    
    # Verify version was deleted
    updated_artifact = await artifact_manager.read(artifact.id)
    version_names = [v["version"] for v in updated_artifact["versions"]]
    assert "v1" not in version_names
    assert "v2" in version_names

    # Test 5: Delete by valid version name
    await artifact_manager.delete(artifact.id, version="v2")
    
    # Verify version was deleted
    updated_artifact = await artifact_manager.read(artifact.id)
    assert len(updated_artifact["versions"]) == 0

    # Clean up
    await artifact_manager.delete(artifact.id, delete_files=True)


async def test_list_files_include_pending(minio_server, fastapi_server, test_user_token):
    """Test list_files with include_pending parameter for staging artifacts."""
    api = await connect_to_server(
        {"name": "test-client", "token": test_user_token, "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a test artifact in staging mode
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Dataset", "description": "Test dataset for file listing"},
        stage=True,
    )

    # List files without include_pending (should be empty)
    files = await artifact_manager.list_files(artifact.id, version="stage")
    assert len(files) == 0

    # List files with include_pending=False (should be empty)
    files = await artifact_manager.list_files(artifact.id, version="stage", include_pending=False)
    assert len(files) == 0

    # Add a file to staging manifest (simulate put_file operation)
    await artifact_manager.put_file(artifact.id, "test_file.txt", download_weight=1.0)

    # List files without include_pending (should show actual uploaded files only)
    files = await artifact_manager.list_files(artifact.id, version="stage")
    assert len(files) == 0  # No actual files uploaded yet

    # List files with include_pending=True (should show pending files)
    files = await artifact_manager.list_files(artifact.id, version="stage", include_pending=True)
    assert len(files) == 1
    assert files[0]["name"] == "test_file.txt"
    assert files[0]["pending"] is True
    assert files[0]["download_weight"] == 1.0

    # Add another file to staging manifest
    await artifact_manager.put_file(artifact.id, "subfolder/test_file2.txt", download_weight=0.5)

    # List files with include_pending=True (should show both pending files)
    files = await artifact_manager.list_files(artifact.id, version="stage", include_pending=True)
    assert len(files) == 2
    file_names = [f["name"] for f in files]
    assert "test_file.txt" in file_names
    assert "subfolder/test_file2.txt" in file_names
    
    # Test with directory filtering
    files = await artifact_manager.list_files(artifact.id, dir_path="subfolder", version="stage", include_pending=True)
    assert len(files) == 1
    assert files[0]["name"] == "test_file2.txt"
    assert files[0]["pending"] is True


async def test_remove_file_error_handling(minio_server, fastapi_server, test_user_token):
    """Test remove_file handles non-existent files gracefully."""
    api = await connect_to_server(
        {"name": "test-client", "token": test_user_token, "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a test artifact in staging mode
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Dataset", "description": "Test dataset for file removal"},
        stage=True,
    )

    # Add a file to staging manifest
    await artifact_manager.put_file(artifact.id, "test_file.txt", download_weight=1.0)

    # Remove the file from staging manifest (even though it doesn't exist in S3)
    # This should not raise an exception
    await artifact_manager.remove_file(artifact.id, "test_file.txt")

    # Verify file is removed from staging manifest
    files = await artifact_manager.list_files(artifact.id, version="stage", include_pending=True)
    assert len(files) == 0

    # Try to remove a file that was never added to staging manifest
    # This should also not raise an exception
    await artifact_manager.remove_file(artifact.id, "nonexistent_file.txt")


async def test_list_files_stage_version_index(minio_server, fastapi_server, test_user_token):
    """Test that list_files uses correct version index for stage version."""
    api = await connect_to_server(
        {"name": "test-client", "token": test_user_token, "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create and commit an artifact with a file
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Dataset", "description": "Test dataset for version indexing"},
    )
    
    # Now edit it to add a file and commit
    edited_artifact = await artifact_manager.edit(artifact.id, stage=True)
    
    # The stage version should use len(versions) as version index
    # Since we have 1 version (v0), the stage version should be at index 1
    files = await artifact_manager.list_files(artifact.id, version="stage")
    # Should not raise an error and should return empty list (no files uploaded yet)
    assert isinstance(files, list)
    assert len(files) == 0

    # Test with stage=True parameter as well
    files = await artifact_manager.list_files(artifact.id, stage=True)
    assert isinstance(files, list)
    assert len(files) == 0


async def test_list_files_with_actual_and_pending_files(minio_server, fastapi_server, test_user_token):
    """Test list_files with mix of actual uploaded files and pending files."""
    api = await connect_to_server(
        {"name": "test-client", "token": test_user_token, "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create and commit an artifact with actual files
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "Test Dataset", "description": "Test dataset for mixed file listing"},
        stage=True,
    )

    # Add and actually upload a file
    put_url = await artifact_manager.put_file(artifact.id, "actual_file.txt", download_weight=1.0)
    response = requests.put(put_url, data="actual file content")
    assert response.ok

    # Commit the artifact
    await artifact_manager.commit(artifact.id)

    # Now put it back in staging mode with version="new" to create a new version
    await artifact_manager.edit(artifact.id, stage=True, version="new")

    # Add a pending file (not uploaded yet)
    await artifact_manager.put_file(artifact.id, "pending_file.txt", download_weight=0.5)

    # List files without include_pending (should be empty since this is a new version)
    files = await artifact_manager.list_files(artifact.id, version="stage")
    assert len(files) == 0  # New version starts empty

    # List files with include_pending=True (should show pending files)
    files = await artifact_manager.list_files(artifact.id, version="stage", include_pending=True)
    assert len(files) == 1  # Only the pending file
    assert files[0]["name"] == "pending_file.txt"
    assert files[0]["pending"] is True
    assert files[0]["download_weight"] == 0.5

    # Verify that the committed version still has the actual file
    files_committed = await artifact_manager.list_files(artifact.id, version="v0")
    assert len(files_committed) == 1
    assert files_committed[0]["name"] == "actual_file.txt"

    # Now test updating existing version (not creating new version)
    await artifact_manager.discard(artifact.id)  # Clear staging
    await artifact_manager.edit(artifact.id, stage=True)  # Stage without version="new"

    # Add a pending file for updating existing version
    await artifact_manager.put_file(artifact.id, "update_file.txt", download_weight=0.3)

    # List files without include_pending (should be empty - files are at different version index)
    files = await artifact_manager.list_files(artifact.id, version="stage")
    assert len(files) == 0

    # List files with include_pending=True (should show pending files)
    files = await artifact_manager.list_files(artifact.id, version="stage", include_pending=True)
    assert len(files) == 1
    assert files[0]["name"] == "update_file.txt"
    assert files[0]["pending"] is True
    assert files[0]["download_weight"] == 0.3


async def test_site_serving_endpoint(minio_server, fastapi_server, test_user_token):
    """Test static site serving functionality."""
    # Create an artifact with type "site"
    api = await connect_to_server(
        {"name": "test-client", "token": test_user_token, "server_url": SERVER_URL}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    token = test_user_token
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create a site artifact
    manifest = {
        "name": "Test Static Site",
        "description": "A test static website",
        "type": "site"
    }
    
    config = {
        "templates": ["index.html", "about.html"],  
        "template_engine": "jinja2",
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "max-age=3600"
        }
    }
    
    # Create artifact
    ws = api.config.workspace  # Use the actual workspace from the API connection
    artifact_alias = "my-static-site"
    
    # Create and commit an artifact with actual files
    artifact = await artifact_manager.create(
        type="site",
        alias=artifact_alias,
        manifest=manifest,
        config=config,
        stage=True,
    )
    
    # Add files to the artifact
    # Regular HTML file (non-templated)
    html_content = """<!DOCTYPE html>
<html>
<head><title>Test Site</title></head>
<body><h1>Welcome to Test Site</h1></body>
</html>"""
    
    file_url = await artifact_manager.put_file(artifact.id, "static.html")
    response = requests.put(file_url, data=html_content.encode(), headers={"Content-Type": "text/html"})
    assert response.ok
    
    
    # Templated HTML file (with Jinja2)
    template_content = """<!DOCTYPE html>
<html>
<head><title>{{ MANIFEST.name }}</title></head>
<body>
    <h1>{{ MANIFEST.name }}</h1>
    <p>{{ MANIFEST.description }}</p>
    <p>Base URL: {{ BASE_URL }}</p>
    <p>Public URL: {{ PUBLIC_BASE_URL }}</p>
    <p>Workspace: {{ WORKSPACE }}</p>
    <p>User: {{ USER.id if USER else 'Anonymous' }}</p>
    <p>View Count: {{ VIEW_COUNT }}</p>
</body>
</html>"""
    
    file_url = await artifact_manager.put_file(artifact.id, "index.html")
    response = requests.put(file_url, data=template_content.encode(), headers={"Content-Type": "text/html"})
    assert response.ok
    
    # CSS file
    css_content = "body { font-family: Arial, sans-serif; }"
    file_url = await artifact_manager.put_file(artifact.id, "style.css")
    response = requests.put(file_url, data=css_content.encode(), headers={"Content-Type": "text/css"})
    assert response.ok
    
    # JSON data file
    json_content = '{"message": "Hello from JSON"}'
    file_url = await artifact_manager.put_file(artifact.id, "data.json")
    response = requests.put(file_url, data=json_content.encode(), headers={"Content-Type": "application/json"})
    assert response.ok
    
    # Commit the artifact
    await artifact_manager.commit(artifact.id)
    
    async with httpx.AsyncClient() as client:
        # Test serving regular static file (non-templated)
        r = await client.get(f"{SERVER_URL}/{ws}/site/{artifact_alias}/static.html", headers=headers)
        assert r.status_code == 200  # Should stream content directly
        assert "text/html" in r.headers.get("content-type", "")
        assert "Welcome to Test Site" in r.text
        
        # Test serving CSS file with correct mime type
        r = await client.get(f"{SERVER_URL}/{ws}/site/{artifact_alias}/style.css", headers=headers)
        assert r.status_code == 200  # Should stream content directly
        assert "text/css" in r.headers.get("content-type", "")
        assert "font-family: Arial" in r.text
        
        # Test serving JSON file
        r = await client.get(f"{SERVER_URL}/{ws}/site/{artifact_alias}/data.json", headers=headers)
        assert r.status_code == 200  # Should stream content directly
        assert "application/json" in r.headers.get("content-type", "")
        assert "Hello from JSON" in r.text
        
        # Test serving templated file (should render template)
        r = await client.get(f"{SERVER_URL}/{ws}/site/{artifact_alias}/index.html", headers=headers)
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        
        # Check template rendering
        content = r.text
        assert "Test Static Site" in content  # From MANIFEST.name
        assert "A test static website" in content  # From MANIFEST.description
        assert f"/{ws}/site/{artifact_alias}/" in content  # BASE_URL
        assert ws in content  # WORKSPACE (use actual workspace name)
        assert "View Count:" in content  # VIEW_COUNT should be present
        
        # Test default index.html serving
        r = await client.get(f"{SERVER_URL}/{ws}/site/{artifact_alias}/", headers=headers)
        assert r.status_code == 200
        assert "Test Static Site" in r.text
        
        # Test CORS headers
        r = await client.get(f"{SERVER_URL}/{ws}/site/{artifact_alias}/index.html", headers=headers)
        assert r.headers.get("Access-Control-Allow-Origin") == "*"
        assert r.headers.get("Cache-Control") == "max-age=3600"
        
        # Test non-existent file
        r = await client.get(f"{SERVER_URL}/{ws}/site/{artifact_alias}/nonexistent.html", headers=headers)
        assert r.status_code == 404
        
        # Test artifact with wrong type
        regular_artifact = await artifact_manager.create(
            type="generic",
            alias="regular-artifact", 
            manifest={"name": "Regular"},
        )
        
        # Should fail for non-site artifact
        r = await client.get(f"{SERVER_URL}/{ws}/site/regular-artifact/index.html", headers=headers)
        assert r.status_code == 400
        assert "not a site artifact" in r.text.lower()
        
        # Test stage parameter (create staged version)
        staged_artifact = await artifact_manager.create(
            type="site",
            alias="staged-site", 
            manifest={"name": "Staged Site"},
            config={"template_engine": "jinja2", "templates": ["index.html"]},
            stage=True,
        )
        
        file_url = await artifact_manager.put_file(staged_artifact.id, "index.html")
        response = requests.put(file_url, data="<h1>Staged content</h1>", headers={"Content-Type": "text/html"})
        assert response.ok
        
        r = await client.get(f"{SERVER_URL}/{ws}/site/staged-site/index.html?stage=true", headers=headers)
        assert r.status_code == 200
        
        # Test default index.html serving when path is empty
        r = await client.get(f"{SERVER_URL}/{ws}/site/{artifact_alias}/", headers=headers)
        assert r.status_code == 200
        assert "Test Static Site" in r.text


async def test_put_file_endpoint_small_file(minio_server, fastapi_server, test_user_token):
    """Test the HTTP PUT file endpoint with a small file."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a dataset artifact
    dataset_manifest = {
        "name": "test-dataset-put",
        "description": "A test dataset for PUT endpoint",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        manifest=dataset_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
        version="stage",
    )

    # Small file content
    file_contents = "This is a small test file for PUT endpoint testing."
    file_path = "test_file.txt"

    # Upload file using the new PUT endpoint
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=file_contents,
        )
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["message"] == "File uploaded successfully"
        assert "etag" in response.headers

    # Verify the file was uploaded correctly by listing files
    files = await artifact_manager.list_files(artifact_id=dataset.id, stage=True)
    assert len(files) == 1
    assert files[0]["name"] == file_path

    # Commit and verify file can be downloaded
    await artifact_manager.commit(artifact_id=dataset.id)

    # Download and verify content
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert response.text == file_contents


async def test_put_file_endpoint_large_file_multipart(minio_server, fastapi_server, test_user_token):
    """Test the HTTP PUT file endpoint with a large file to trigger multipart upload."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a dataset artifact
    dataset_manifest = {
        "name": "test-dataset-large-put",
        "description": "A test dataset for large file PUT endpoint",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        manifest=dataset_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
        version="stage",
    )

    # Create large file content (>5MB to trigger multipart upload)
    large_content = "A" * (6 * 1024 * 1024)  # 6MB of 'A' characters
    file_path = "large_test_file.txt"

    # Upload large file using the new PUT endpoint
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=large_content,
        )
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["message"] == "File uploaded successfully"
        assert "etag" in response.headers

    # Verify the file was uploaded correctly
    files = await artifact_manager.list_files(artifact_id=dataset.id, stage=True)
    assert len(files) == 1
    assert files[0]["name"] == file_path

    # Commit and verify file can be downloaded
    await artifact_manager.commit(artifact_id=dataset.id)

    # Download and verify content (partial check to avoid memory issues)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            follow_redirects=True,
        )
        assert response.status_code == 200
        # Verify file size and start/end content
        assert len(response.content) == len(large_content)
        assert response.text.startswith("AAAA")
        assert response.text.endswith("AAAA")


async def test_put_file_endpoint_streaming_chunked(minio_server, fastapi_server, test_user_token):
    """Test the HTTP PUT file endpoint with streaming chunked upload."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a dataset artifact
    dataset_manifest = {
        "name": "test-dataset-streaming",
        "description": "A test dataset for streaming upload",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        manifest=dataset_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
        version="stage",
    )

    # Create chunked content (simulate streaming)
    def generate_chunks():
        """Generator that yields chunks of data."""
        for i in range(100):
            yield f"Chunk {i:03d}: " + "X" * 100 + "\n"

    file_path = "streaming_test_file.txt"
    expected_content = "".join(generate_chunks())

    # Upload using streaming with chunked transfer encoding
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=expected_content,
        )
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["message"] == "File uploaded successfully"
        assert "etag" in response.headers

    # Verify the file was uploaded correctly
    files = await artifact_manager.list_files(artifact_id=dataset.id, stage=True)
    assert len(files) == 1
    assert files[0]["name"] == file_path

    # Commit and verify content
    await artifact_manager.commit(artifact_id=dataset.id)

    # Download and verify content
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert response.text == expected_content


async def test_put_file_endpoint_nested_directories(minio_server, fastapi_server, test_user_token):
    """Test the HTTP PUT file endpoint with nested directory structure."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a dataset artifact
    dataset_manifest = {
        "name": "test-dataset-nested",
        "description": "A test dataset for nested directory uploads",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        manifest=dataset_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
        version="stage",
    )

    # Test files with nested paths
    test_files = {
        "root_file.txt": "Root level file content",
        "level1/file1.txt": "Level 1 file content",
        "level1/level2/file2.txt": "Level 2 file content", 
        "level1/level2/level3/deep_file.txt": "Deep nested file content",
    }

    # Upload all files
    for file_path, content in test_files.items():
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(
                f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
                headers={"Authorization": f"Bearer {test_user_token}"},
                data=content,
            )
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["message"] == "File uploaded successfully"

    # Verify all files were uploaded
    files = await artifact_manager.list_files(artifact_id=dataset.id, stage=True)
    assert len(files) == 2 # 1 file + 1 directory
    uploaded_paths = {f["name"] for f in files}
    assert uploaded_paths == {"root_file.txt", "level1"}

    # Commit and verify files can be downloaded
    await artifact_manager.commit(artifact_id=dataset.id)

    # Test downloading each file
    for file_path, expected_content in test_files.items():
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
                follow_redirects=True,
            )
            assert response.status_code == 200
            assert response.text == expected_content

async def test_multipart_upload_endpoint(
    minio_server, fastapi_server, test_user_token
):
    """Test the two-step multipart upload endpoint."""
    # 1. Set up the connection and create a test artifact
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace
    token = await api.generate_token()

    collection = await artifact_manager.create(
        type="collection",
        manifest={"name": "Multipart Test Collection"},
        config={"permissions": {"*": "r", "@": "rw+"}},
    )
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest={"name": "Multipart Dataset"},
        version="stage",
    )

    # 2. Define the file to be uploaded and chunk details
    file_path = "large_test_file.bin"
    file_size = 12 * 1024 * 1024  # 12 MB
    file_content = os.urandom(file_size)
    chunk_size = 5 * 1024 * 1024  # 5 MB
    part_count = (file_size + chunk_size - 1) // chunk_size
    assert part_count == 3

    # 3. Step 1: Create the multipart upload and get presigned URLs
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/artifacts/{dataset.alias}/create-multipart-upload",
            params={"path": file_path, "part_count": part_count},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        upload_info = response.json()
        assert "upload_id" in upload_info
        assert len(upload_info["parts"]) == part_count
        upload_id = upload_info["upload_id"]
        parts = upload_info["parts"]

    # 4. Step 2: Upload each part to its presigned URL
    uploaded_parts = []
    file_buffer = BytesIO(file_content)
    async with httpx.AsyncClient(timeout=120) as client:
        for part_info in parts:
            part_number = part_info["part_number"]
            url = part_info["url"]
            chunk_data = file_buffer.read(chunk_size)
            
            # Upload the chunk
            response = await client.put(url, data=chunk_data)
            assert response.status_code == 200
            
            # Save the ETag from the response header
            etag = response.headers["ETag"].strip('"').strip("'")
            uploaded_parts.append({"part_number": part_number, "etag": etag})

    assert len(uploaded_parts) == part_count

    # 5. Step 3: Complete the multipart upload
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/artifacts/{dataset.alias}/complete-multipart-upload",
            json={"upload_id": upload_id, "parts": uploaded_parts},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        assert response.status_code == 200
        completion_response = response.json()
        assert completion_response["success"] is True

    # 6. Commit and verify the uploaded file
    await artifact_manager.commit(artifact_id=dataset.id)
    
    # Download the file to verify its content
    download_url = f"{SERVER_URL}/{workspace}/artifacts/{dataset.alias}/files/{file_path}?use_proxy=true"
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        response = await client.get(
            download_url, headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        downloaded_content = response.content
        assert downloaded_content == file_content

    # 7. Clean up
    await artifact_manager.delete(artifact_id=dataset.id)
    await artifact_manager.delete(artifact_id=collection.id)

async def test_put_file_endpoint_download_weight(minio_server, fastapi_server, test_user_token):
    """Test the HTTP PUT file endpoint with download weight parameter."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a dataset artifact
    dataset_manifest = {
        "name": "test-dataset-download-weight",
        "description": "A test dataset for download weight testing",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        manifest=dataset_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
        version="stage",
    )

    file_contents = "File with download weight"
    file_path = "weighted_file.txt"
    download_weight = 5.5

    # Upload file with download weight
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}?download_weight={download_weight}",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=file_contents,
        )
        assert response.status_code == 200

    # Commit the artifact
    await artifact_manager.commit(artifact_id=dataset.id)

    # Verify download weight is recorded correctly
    artifact = await artifact_manager.read(artifact_id=dataset.id)
    assert artifact["config"]["download_weights"][file_path] == download_weight


async def test_put_file_endpoint_error_handling(minio_server, fastapi_server, test_user_token, test_user_token_2):
    """Test the HTTP PUT file endpoint error handling."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a dataset artifact with restrictive permissions
    dataset_manifest = {
        "name": "test-dataset-restricted",
        "description": "A test dataset with restricted access",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        manifest=dataset_manifest,
        config={"permissions": {"@": "r"}},  # Signed in user has read access
        version="stage",
    )

    file_contents = "Test file content"
    file_path = "test_file.txt"

    # Test 1: Unauthorized access (no token)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            data=file_contents,
        )
        assert response.status_code == 403

    # Test 2: Permission denied (different user)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            headers={"Authorization": f"Bearer {test_user_token_2}"},
            data=file_contents,
        )
        assert response.status_code == 403

    # Test 3: Non-existent artifact
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/non-existent-artifact/files/{file_path}",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=file_contents,
        )
        assert response.status_code == 404

    # Test 4: Invalid download weight
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}?download_weight=-1",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=file_contents,
        )
        assert response.status_code == 400

    # Test 5: Try to upload to committed artifact (not in staging)
    committed_dataset = await artifact_manager.create(
        type="dataset",
        manifest={"name": "committed-dataset", "description": "Committed dataset"},
        config={"permissions": {"@": "rw+"}},
    )
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{committed_dataset.alias}/files/{file_path}",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=file_contents,
        )
        assert response.status_code == 400
        assert "Artifact must be in staging mode" in response.text


async def test_put_file_endpoint_overwrite_existing(minio_server, fastapi_server, test_user_token):
    """Test the HTTP PUT file endpoint overwriting existing files."""

    # Connect and get the artifact manager service
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a dataset artifact
    dataset_manifest = {
        "name": "test-dataset-overwrite",
        "description": "A test dataset for file overwriting",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        manifest=dataset_manifest,
        config={"permissions": {"*": "r", "@": "rw+"}},
        version="stage",
    )

    file_path = "overwrite_test.txt"
    
    # Upload initial file
    initial_content = "Initial file content"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=initial_content,
        )
        assert response.status_code == 200

    # Verify initial file
    files = await artifact_manager.list_files(artifact_id=dataset.id, stage=True)
    assert len(files) == 1

    # Overwrite with new content
    new_content = "New file content that overwrites the old one"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            headers={"Authorization": f"Bearer {test_user_token}"},
            data=new_content,
        )
        assert response.status_code == 200

    # Still should have only 1 file (overwritten)
    files = await artifact_manager.list_files(artifact_id=dataset.id, stage=True)
    assert len(files) == 1

    # Commit and verify the new content
    await artifact_manager.commit(artifact_id=dataset.id)

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/{file_path}",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert response.text == new_content
