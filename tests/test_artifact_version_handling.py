"""Test artifact version handling with staging."""

import pytest
import requests
from hypha_rpc import connect_to_server
from . import SERVER_URL, SERVER_URL_SQLITE

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_version_index_handling(minio_server, fastapi_server_sqlite, test_user_token):
    """Test version index handling when editing existing artifact vs creating new version."""
    
    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "test version handling client",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    
    # Create an artifact with an initial version in staging mode
    initial_manifest = {
        "name": "Version Test Artifact",
        "description": "Artifact to test version index handling",
    }
    artifact = await artifact_manager.create(
        type="dataset",
        manifest=initial_manifest,
        config={"permissions": {"*": "rw+"}},
        stage=True,  # Start in staging mode
    )
    
    # Verify no versions yet (in staging)
    assert len(artifact["versions"]) == 0
    assert artifact["staging"] is not None
    
    # Add a file to the staging area
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="v0_file.txt",
        download_weight=1,
    )
    response = requests.put(put_url, data="Content for v0")
    assert response.ok
    
    # Commit to create the initial version
    artifact = await artifact_manager.commit(artifact_id=artifact.id)
    assert len(artifact["versions"]) == 1
    assert artifact["versions"][0]["version"] == "v0"
    
    # List files to verify it's in v0
    files = await artifact_manager.list_files(artifact_id=artifact.id)
    assert len(files) == 1
    assert files[0]["name"] == "v0_file.txt"
    
    # Scenario 1: Edit current version (stage=True without version="new")
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "Version Test Artifact", "description": "Edited in place"},
        stage=True,
    )
    
    # Add a file while editing current version
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="edited_file.txt",
        download_weight=1,
    )
    response = requests.put(put_url, data="Content added during edit")
    assert response.ok
    
    # List files in staging - should see both files (v0_file.txt and edited_file.txt)
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="stage")
    assert len(files) == 2
    file_names = [f["name"] for f in files]
    assert "v0_file.txt" in file_names
    assert "edited_file.txt" in file_names
    
    # Commit - should NOT create new version
    committed = await artifact_manager.commit(artifact_id=artifact.id)
    assert len(committed["versions"]) == 1  # Still only v0
    
    # Verify files are in v0
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="v0")
    assert len(files) == 2
    file_names = [f["name"] for f in files]
    assert "v0_file.txt" in file_names
    assert "edited_file.txt" in file_names
    
    # Scenario 2: Create new version (version="new" + stage=True)
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "Version Test Artifact", "description": "New version created"},
        version="new",
        stage=True,
    )
    
    # Add a file for the new version
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="v1_file.txt",
        download_weight=1,
    )
    response = requests.put(put_url, data="Content for v1")
    assert response.ok
    
    # List files in staging - should only see the new file (v1_file.txt)
    # The files from v0 should not be visible since we're creating a new version
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="stage")
    assert len(files) == 1
    assert files[0]["name"] == "v1_file.txt"
    
    # Commit with custom version name
    committed = await artifact_manager.commit(artifact_id=artifact.id, version="v1")
    assert len(committed["versions"]) == 2  # Now we have v0 and v1
    assert committed["versions"][1]["version"] == "v1"
    
    # Verify v0 still has its files
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="v0")
    assert len(files) == 2
    file_names = [f["name"] for f in files]
    assert "v0_file.txt" in file_names
    assert "edited_file.txt" in file_names
    
    # Verify v1 has only the new file
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="v1")
    assert len(files) == 1
    assert files[0]["name"] == "v1_file.txt"
    
    # Clean up
    await artifact_manager.delete(artifact_id=artifact.id, delete_files=True)


async def test_file_removal_version_handling(minio_server, fastapi_server_sqlite, test_user_token):
    """Test file removal with different version scenarios."""
    
    api = await connect_to_server(
        {
            "name": "test file removal client",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    
    # Create artifact with initial files in staging
    artifact = await artifact_manager.create(
        type="dataset",
        manifest={"name": "File Removal Test"},
        config={"permissions": {"*": "rw+"}},
        stage=True,
    )
    
    # Add two files to staging
    for filename in ["file1.txt", "file2.txt"]:
        put_url = await artifact_manager.put_file(
            artifact_id=artifact.id,
            file_path=filename,
        )
        response = requests.put(put_url, data=f"Content of {filename}")
        assert response.ok
    
    # Commit to create v0
    artifact = await artifact_manager.commit(artifact_id=artifact.id)
    assert len(artifact["versions"]) == 1
    
    # Scenario 1: Remove file while editing current version
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "File Removal Test", "description": "Removing file from current version"},
        stage=True,
    )
    
    # Remove file1.txt
    await artifact_manager.remove_file(artifact_id=artifact.id, file_path="file1.txt")
    
    # List files in staging - should only see file2.txt
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="stage")
    assert len(files) == 1
    assert files[0]["name"] == "file2.txt"
    
    # Commit
    committed = await artifact_manager.commit(artifact_id=artifact.id)
    assert len(committed["versions"]) == 1  # Still only v0
    
    # Verify v0 now only has file2.txt
    files = await artifact_manager.list_files(artifact_id=artifact.id)
    assert len(files) == 1
    assert files[0]["name"] == "file2.txt"
    
    # Scenario 2: Remove file while creating new version
    await artifact_manager.edit(
        artifact_id=artifact.id,
        manifest={"name": "File Removal Test", "description": "New version without file2"},
        version="new",
        stage=True,
    )
    
    # Add a new file for v1
    put_url = await artifact_manager.put_file(
        artifact_id=artifact.id,
        file_path="file3.txt",
    )
    response = requests.put(put_url, data="Content of file3.txt")
    assert response.ok
    
    # List files in staging - should only see file3.txt (new version doesn't inherit files)
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="stage")
    assert len(files) == 1
    assert files[0]["name"] == "file3.txt"
    
    # Commit
    committed = await artifact_manager.commit(artifact_id=artifact.id)
    assert len(committed["versions"]) == 2
    
    # Verify v0 still has file2.txt
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="v0")
    assert len(files) == 1
    assert files[0]["name"] == "file2.txt"
    
    # Verify v1 has file3.txt
    files = await artifact_manager.list_files(artifact_id=artifact.id, version="v1")
    assert len(files) == 1
    assert files[0]["name"] == "file3.txt"
    
    # Clean up
    await artifact_manager.delete(artifact_id=artifact.id, delete_files=True)