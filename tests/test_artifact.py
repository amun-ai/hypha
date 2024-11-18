"""Test Artifact services."""
import pytest
import requests
import os
from hypha_rpc import connect_to_server

from . import SERVER_URL, SERVER_URL_SQLITE, find_item

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
        config={"permissions": {"*": "r", "@": "r+"}},
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
        filters={"stage": True, "manifest": {"description": "*dataset*"}},
    )

    assert len(search_results) == len(datasets)

    # list application only
    search_results = await artifact_manager.list(
        parent_id=collection.id, filters={"stage": True, "type": "application"}
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
        config={"permissions": {"*": "r", "@": "r+"}},
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
    """Test the HTTP file serving and directory listing endpoint."""

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
        config={"permissions": {"*": "r", "@": "r+"}},
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
    response = requests.put(put_url, data=file_contents)
    assert response.ok

    # Commit the dataset artifact
    await artifact_manager.commit(artifact_id=dataset.id)

    files = await artifact_manager.list_files(artifact_id=dataset.id)
    assert len(files) == 1

    # Retrieve the file via HTTP
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/example.txt"
    )
    assert response.status_code == 200
    assert response.text == file_contents

    # Check download count increment
    artifact = await artifact_manager.read(
        artifact_id=dataset.id,
    )
    assert artifact["download_count"] == 1

    # Attempt to list directory contents (should be successful after attempting file)
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{dataset.alias}/files/"
    )
    assert response.status_code == 200
    assert "example.txt" in [file["name"] for file in response.json()]


async def test_artifact_permissions(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """Test artifact permissions."""
    api = await connect_to_server(
        {
            "name": "test-client",
            "server_url": SERVER_URL,
            "token": test_user_token,
            "token": test_user_token,
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
        config={"permissions": {"*": "r", "@": "r+"}},
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
        version="v1",
    )

    # Verify the initial version
    artifact_data = await artifact_manager.read(artifact_id=artifact.id, version="v1")
    assert artifact_data["versions"][-1]["version"] == "v1"

    # Stage a new version
    updated_manifest = {
        "name": "Versioned Artifact",
        "description": "Updated version description",
    }
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest=updated_manifest,
        version="stage",
    )

    # Read the staged version to confirm the update
    staged_data = await artifact_manager.read(artifact_id=artifact.id, version="stage")
    assert staged_data["manifest"]["description"] == "Updated version description"

    # Commit the staged version
    await artifact_manager.commit(artifact_id=artifact.id, version="v2")

    # Verify the staged version is accessible
    staged_data = await artifact_manager.read(artifact_id=artifact.id, version="v2")
    assert staged_data["manifest"]["description"] == "Updated version description"

    # Test reading the latest version
    latest_data = await artifact_manager.read(artifact_id=artifact.id, version="latest")
    assert latest_data["manifest"]["description"] == "Updated version description"

    # Attempt to read a non-existent version
    with pytest.raises(Exception, match=r".*Artifact version 'v999' does not exist.*"):
        await artifact_manager.read(artifact_id=artifact.id, version="v999")

    # Clean up by deleting the artifact
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
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
        publish_to="sandbox_zenodo",
    )

    # Add a file to the artifact
    put_url = await artifact_manager.put_file(
        artifact_id=dataset.id,
        file_path="example.txt",
    )
    source = "file contents of example.txt"
    response = requests.put(put_url, data=source)
    assert response.ok

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
            "permissions": {"*": "r+"},
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

    # Filter by `type`: Only datasets should be returned
    results = await artifact_manager.list(
        parent_id=collection.id, filters={"type": "dataset"}, mode="AND"
    )
    assert len(results) == 2
    for result in results:
        assert result["type"] == "dataset"

    # Filter by `created_by`: Only artifacts created by user_id1 should be returned
    results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"created_by": user_id1, "stage": False},
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
    for result in results:
        assert result["view_count"] >= 1

    # Filter by `stage`: Only staged artifacts should be returned
    results = await artifact_manager.list(
        parent_id=collection.id, filters={"stage": True}, mode="AND"
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
        config={"permissions": {"*": "r", "@": "r+"}},
    )

    # Create an artifact within the collection
    dataset_manifest = {
        "name": "Test Dataset",
        "description": "A test dataset for HTTP endpoint",
    }
    dataset = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
        config={"permissions": {"*": "r", "@": "r+"}},
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
    response = requests.get(
        f"{SERVER_URL}/{api.config.workspace}/artifacts/{collection.alias}/children"
    )
    assert response.status_code == 200
    item = find_item(response.json(), "id", dataset.id)
    manifest = item["manifest"]
    assert "Test Dataset" == manifest["name"]


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
        config={"permissions": {"*": "r", "@": "r+"}},
    )

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

    # Commit the artifact
    dataset = await artifact_manager.commit(artifact_id=dataset.id)
    versions = dataset["versions"]

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
        version="stage",
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

    # Commit the artifact after editing, no version specified, so it will overwrite the latest version
    dataset = await artifact_manager.commit(artifact_id=dataset.id)
    assert len(versions) == len(dataset["versions"])

    # Verify the committed manifest reflects the edited data
    committed_artifact = await artifact_manager.read(artifact_id=dataset.id)
    committed_manifest = committed_artifact["manifest"]
    assert (
        committed_manifest["description"] == "Updated description of the test dataset"
    )
    assert committed_manifest["custom_key"] == 19222

    # Retrieve and verify the uploaded file
    get_url = await artifact_manager.get_file(
        artifact_id=dataset.id, path="example.txt"
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

    # Clean up by deleting the dataset and collection
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
            "permissions": {"*": "r", "@": "r+"},
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
    await artifact_manager.commit(artifact_id=valid_dataset.id)

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
            "permissions": {"*": "r", "@": "r+"},
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
        parent_id=collection.id, filters={"stage": True}
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
    get_url = await artifact_manager.get_file(artifact_id=dataset.id, path="test.txt")
    response = requests.get(get_url)
    assert response.ok
    assert response.text == file_content

    # Test overwriting the existing dataset artifact within the collection
    updated_dataset_manifest = {
        "name": "test-dataset",
        "description": "Overwritten description for dataset",
    }
    await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        alias="test-dataset",
        manifest=updated_dataset_manifest,
        overwrite=True,
    )

    # Confirm that the description update is reflected
    updated_manifest = await artifact_manager.read(
        artifact_id=f"{api.config.workspace}/test-dataset"
    )
    updated_manifest = updated_manifest["manifest"]
    assert updated_manifest["description"] == "Overwritten description for dataset"

    # Remove the dataset artifact from the collection
    await artifact_manager.delete(artifact_id=dataset.id)

    # Confirm deletion by checking the collection’s children
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
            config={"permissions": {"*": "r", "@": "r+"}},
        )

    # Successfully create the collection with orphan set by default
    collection = await artifact_manager.create(
        manifest=collection_manifest, config={"permissions": {"*": "r", "@": "r+"}}
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
    with pytest.raises(
        Exception, match=r".*already exists. Use overwrite=True to overwrite.*"
    ):
        await artifact_manager.create(
            type="dataset",
            parent_id=collection.id,
            alias="edge-case-dataset",
            manifest=dataset_manifest,
            overwrite=False,
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
        config={"permissions": {"*": "r", "@": "r+"}},
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
            "manifest": {"name": "Test Dataset 3", "description": "A test dataset 1"}
        },
        mode="OR",
    )
    assert (
        len(search_results) == 2
    )  # Matches for either dataset 1 or 3 based on OR filter

    # Clean up by deleting all artifacts and the collection
    for i in range(5):
        await artifact_manager.delete(artifact_id=f"test-dataset-{i}")
    await artifact_manager.delete(artifact_id=collection.id)


async def test_artifact_search_with_filters(
    minio_server, fastapi_server, test_user_token
):
    """Test search functionality with specific key-value filters in the manifest, supporting multiple filters and AND/OR modes."""

    # Connect to the server and initialize the artifact manager
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing filter-based search
    collection_manifest = {
        "name": "Filter Test Collection",
        "description": "A collection to test filter functionality",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "r+"}},
    )

    # Create multiple artifacts with varied attributes inside the collection
    for i in range(5):
        dataset_manifest = {
            "name": f"Dataset {i}",
            "description": f"Description for dataset {i}",
        }
        await artifact_manager.create(
            type="dataset" if i % 2 == 0 else "application",
            alias=f"dataset-{i}",
            parent_id=collection.id,
            manifest=dataset_manifest,
        )

    # Filter by type to return only datasets
    search_results = await artifact_manager.list(
        parent_id=collection.id, filters={"type": "dataset"}, mode="AND"
    )
    assert len(search_results) == 3
    for result in search_results:
        assert result["type"] == "dataset"

    # Filter by a field in the manifest: exact name match
    search_results = await artifact_manager.list(
        parent_id=collection.id, filters={"manifest": {"name": "Dataset 3"}}, mode="AND"
    )
    assert len(search_results) == 1
    assert search_results[0]["manifest"]["name"] == "Dataset 3"

    # Filter by multiple fields in the manifest using OR mode
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={
            "manifest": {
                "name": "Dataset 3",
                "description": "Description for dataset 1",
            }
        },
        mode="OR",
    )
    assert (
        len(search_results) == 2
    )  # Matches for either dataset 1 or 3 based on OR filter

    # Filter by multiple fields in the manifest using AND mode
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={
            "manifest": {
                "name": "Dataset 3",
                "description": "Description for dataset 3",
            }
        },
        mode="AND",
    )
    assert len(search_results) == 1
    assert search_results[0]["manifest"]["name"] == "Dataset 3"

    # Filter with a fuzzy match in the description
    search_results = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"description": "Description for dataset*"}},
        mode="AND",
    )
    assert (
        len(search_results) == 5
    )  # All items match since the description pattern is common

    # Clean up by deleting the datasets and the collection
    for i in range(5):
        await artifact_manager.delete(artifact_id=f"dataset-{i}")
    await artifact_manager.delete(artifact_id=collection.id)


async def test_download_count(minio_server, fastapi_server, test_user_token):
    """Test the download count functionality for artifacts."""
    api = await connect_to_server(
        {"name": "test-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for testing download count
    collection_manifest = {
        "name": "Download Test Collection",
        "description": "A collection to test download count functionality",
    }
    collection = await artifact_manager.create(
        type="collection",
        manifest=collection_manifest,
        config={"permissions": {"*": "r", "@": "r+"}},
    )

    # Create an artifact inside the collection
    dataset_manifest = {
        "name": "Download Test Dataset",
        "description": "A test dataset for download count",
    }
    artifact = await artifact_manager.create(
        type="dataset",
        parent_id=collection.id,
        manifest=dataset_manifest,
        version="stage",
    )

    # Put a primary file in the artifact
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="example.txt",
        download_weight=0.5,  # Set as primary for download count
    )
    source = "file contents of example.txt"
    response = requests.put(put_url, data=source)
    assert response.ok

    # Put another non-primary file in the artifact
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id, file_path="example2.txt"
    )
    response = requests.put(put_url, data="file contents of example2.txt")
    assert response.ok

    # Commit the artifact
    await artifact_manager.commit(artifact_id=artifact.id)

    # Ensure that the download count is initially zero
    artifact = await artifact_manager.read(
        artifact_id=artifact.id,
    )
    assert artifact["download_count"] == 0

    # Increment the download count of the artifact by download the primary file
    get_url = await artifact_manager.get_file(
        artifact_id=artifact.id,
        path="example.txt",
    )
    response = requests.get(get_url)
    assert response.ok

    # Ensure that the download count is incremented
    artifact = await artifact_manager.read(
        artifact_id=artifact.id,
        silent=True,
    )
    assert artifact["download_count"] == 0.5

    # download twice will increment the download count by 1
    get_url = await artifact_manager.get_file(
        artifact_id=artifact.id,
        path="example.txt",
    )
    response = requests.get(get_url)
    assert response.ok

    # Ensure that the download count is incremented
    artifact = await artifact_manager.read(
        artifact_id=artifact.id,
        silent=True,
    )
    assert artifact["download_count"] == 1

    # If we get the example file in silent mode, the download count won't increment
    get_url = await artifact_manager.get_file(
        artifact_id=artifact.id,
        path="example.txt",
        silent=True,
    )
    response = requests.get(get_url)
    assert response.ok
    get_url = await artifact_manager.get_file(
        artifact_id=artifact.id,
        path="example.txt",
        silent=True,
    )
    response = requests.get(get_url)
    assert response.ok

    # Ensure that the download count is not incremented
    artifact = await artifact_manager.read(
        artifact_id=artifact.id,
    )
    assert artifact["download_count"] == 1

    # download example 2 won't increment the download count
    get_url = await artifact_manager.get_file(
        artifact_id=artifact.id,
        path="example2.txt",
    )
    response = requests.get(get_url)
    assert response.ok

    # Ensure that the download count is incremented
    artifact = await artifact_manager.read(
        artifact_id=artifact.id,
    )
    assert artifact["download_count"] == 1

    # Let's call reset_stats to reset the download count
    await artifact_manager.reset_stats(
        artifact_id=artifact.id,
    )

    # Ensure that the download count is reset
    artifact = await artifact_manager.read(
        artifact_id=artifact.id,
    )
    assert artifact["download_count"] == 0

    # Clean up by deleting the dataset and the collection
    await artifact_manager.delete(
        artifact_id=artifact.id,
        delete_files=True,
        recursive=True,
    )
    await artifact_manager.delete(artifact_id=collection.id)
