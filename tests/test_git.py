"""End-to-end tests for Git-enabled artifacts.

These tests use the actual Hypha server with Git and LFS support.
They require the git and git-lfs command-line tools.

NOTE: These tests require Docker (for postgres) to run. They use fastapi_server
fixture which depends on postgres_server. This ensures they run correctly in CI.
For local testing without Docker, see test_git_storage.py for unit tests.

For unit tests of the Git storage components, see test_git_storage.py.
"""

import os
import tempfile
import pytest
import requests

from . import SERVER_URL, WS_SERVER_URL, SIO_PORT

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


# Server-based Git CLI tests (standalone functions)


async def test_git_clone_empty_repo(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test cloning an empty repository with real git command."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    # Connect to the server and create a git-storage artifact
    api = await connect_to_server({
        "name": "git-clone-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-clone-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Clone Test"},
        config={"storage": "git"},
    )
    await artifact_manager.commit(artifact_id=artifact_alias)

    workspace = api.config.workspace
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "clone")

        # Git clone of empty repo should succeed (or give warning about empty)
        result = subprocess.run(
            ["git", "clone", base_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Empty repo clone might give warning but should not fail fatally
        if result.returncode != 0:
            assert "empty" in result.stderr.lower() or "warning" in result.stderr.lower(), \
                f"Unexpected error: {result.stderr}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)


async def test_git_ls_remote(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test git ls-remote on a repository."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-ls-remote-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-ls-remote-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git ls-remote Test"},
        config={"storage": "git"},
    )
    await artifact_manager.commit(artifact_id=artifact_alias)

    workspace = api.config.workspace
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    # Run git ls-remote
    result = subprocess.run(
        ["git", "ls-remote", base_url],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Empty repo should return 0
    assert result.returncode == 0, f"ls-remote failed: {result.stderr}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)


async def test_git_protocol_discovery(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that git can discover the repo via HTTP protocol."""
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-discovery-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-discovery-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Discovery Test"},
        config={"storage": "git"},
    )
    await artifact_manager.commit(artifact_id=artifact_alias)

    workspace = api.config.workspace
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    # Test info/refs endpoint directly
    info_refs_url = f"{base_url}/info/refs?service=git-upload-pack"
    response = requests.get(info_refs_url)

    assert response.status_code == 200
    assert "application/x-git-upload-pack-advertisement" in response.headers.get("content-type", "")
    assert b"# service=git-upload-pack" in response.content

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)


async def test_git_clone_add_commit_push(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test full git workflow: clone empty repo, add file, commit, and push."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-push-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-push-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Push Test"},
        config={"storage": "git"},
    )
    await artifact_manager.commit(artifact_id=artifact_alias)

    workspace = api.config.workspace
    port = SIO_PORT
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
    push_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "repo")

        # Step 1: Clone the empty repository
        result = subprocess.run(
            ["git", "clone", base_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            assert "empty" in result.stderr.lower() or "warning" in result.stderr.lower(), \
                f"Clone failed unexpectedly: {result.stderr}"
            os.makedirs(clone_dir, exist_ok=True)
            subprocess.run(["git", "init"], cwd=clone_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", base_url], cwd=clone_dir, check=True)

        # Step 2: Configure git user
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=clone_dir,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=clone_dir,
            check=True,
        )

        # Step 3: Create a new file
        test_file = os.path.join(clone_dir, "README.md")
        with open(test_file, "w") as f:
            f.write("# Test Repository\n\nThis is a test file.\n")

        # Step 4: Add and commit
        result = subprocess.run(
            ["git", "add", "README.md"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"git add failed: {result.stderr}"

        result = subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"git commit failed: {result.stderr}"

        # Step 5: Push with authentication
        result = subprocess.run(
            ["git", "push", push_url, "main"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )

        assert result.returncode == 0, \
            f"git push failed: stdout={result.stdout}, stderr={result.stderr}"

        # Step 6: Clone again to verify
        clone_dir2 = os.path.join(tmpdir, "repo2")
        result = subprocess.run(
            ["git", "clone", base_url, clone_dir2],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, \
            f"git clone (after push) failed: stdout={result.stdout}, stderr={result.stderr}"

        # Verify the cloned repository has the file
        cloned_file = os.path.join(clone_dir2, "README.md")
        assert os.path.exists(cloned_file), \
            f"Cloned repo should contain README.md"

        with open(cloned_file, "r") as f:
            content = f.read()
        assert "Test Repository" in content

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)


async def test_git_receive_pack_requires_auth(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that git-receive-pack endpoint returns 401 without auth."""
    import base64
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-auth-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-auth-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Auth Test"},
        config={"storage": "git"},
    )
    await artifact_manager.commit(artifact_id=artifact_alias)

    workspace = api.config.workspace
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    # Test 1: git-receive-pack without auth should return 401
    receive_pack_url = f"{base_url}/git-receive-pack"
    response = requests.post(receive_pack_url, data=b"", timeout=10)
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    assert "WWW-Authenticate" in response.headers
    assert "Basic" in response.headers["WWW-Authenticate"]

    # Test 2: git-receive-pack with invalid Basic auth should return 401
    invalid_creds = base64.b64encode(b"git:invalid-token").decode()
    response = requests.post(
        receive_pack_url,
        data=b"",
        headers={"Authorization": f"Basic {invalid_creds}"},
        timeout=10
    )
    assert response.status_code == 401

    # Test 3: git-receive-pack with valid auth should pass (200 status)
    valid_creds = base64.b64encode(f"git:{test_user_token}".encode()).decode()
    response = requests.post(
        receive_pack_url,
        data=b"",
        headers={"Authorization": f"Basic {valid_creds}"},
        timeout=5,
        stream=True,
    )
    # With valid auth, we should get 200 status code (auth passed)
    assert response.status_code == 200
    response.close()

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)


# Git LFS tests


@pytest.fixture
def check_git_lfs_installed():
    """Check if git-lfs is installed, skip test if not."""
    import subprocess
    import shutil

    if shutil.which("git-lfs") is None:
        pytest.skip("git-lfs not installed")

    result = subprocess.run(
        ["git", "lfs", "version"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        pytest.skip("git-lfs not available")


async def test_git_lfs_push_and_pull(
    minio_server,
    fastapi_server,
    test_user_token,
    check_git_lfs_installed,
):
    """Test pushing and pulling large files with Git LFS via the actual server.

    This test:
    1. Creates a git-storage artifact via artifact manager
    2. Clones the empty repo
    3. Adds a large file tracked by LFS
    4. Commits and pushes
    5. Clones to a new directory
    6. Verifies the large file is retrieved correctly
    """
    import subprocess
    import hashlib
    import uuid
    from hypha_rpc import connect_to_server

    # Connect to the server
    api = await connect_to_server({
        "name": "git-lfs-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    # Get artifact manager and create git-storage artifact
    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"lfs-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "LFS Test Repository", "description": "Test repo for Git LFS"},
        config={"storage": "git"},  # Enable git storage
    )

    # Commit the artifact to make it accessible
    await artifact_manager.commit(artifact_id=artifact_alias)

    workspace = api.config.workspace
    port = SIO_PORT

    with tempfile.TemporaryDirectory() as tmpdir:
        # URLs for git operations
        base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
        push_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

        # Step 1: Clone empty repo
        clone_dir = os.path.join(tmpdir, "repo")
        result = subprocess.run(
            ["git", "clone", base_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Empty repo clone may have warning but shouldn't fail fatally
        if result.returncode != 0:
            assert "empty" in result.stderr.lower() or "warning" in result.stderr.lower(), \
                f"Clone failed unexpectedly: {result.stderr}"
            # If clone failed due to empty repo, initialize manually
            os.makedirs(clone_dir, exist_ok=True)
            subprocess.run(["git", "init"], cwd=clone_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", base_url], cwd=clone_dir, check=True)

        # Step 2: Install and configure LFS
        result = subprocess.run(
            ["git", "lfs", "install"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"git lfs install failed: {result.stderr}"

        # Step 3: Track large files with LFS
        result = subprocess.run(
            ["git", "lfs", "track", "*.bin"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"git lfs track failed: {result.stderr}"

        # Step 4: Create a large file (1 MB for faster test)
        large_file_path = os.path.join(clone_dir, "large_data.bin")
        large_file_size = 1 * 1024 * 1024  # 1 MB
        large_file_content = os.urandom(large_file_size)
        large_file_hash = hashlib.sha256(large_file_content).hexdigest()

        with open(large_file_path, "wb") as f:
            f.write(large_file_content)

        # Also create a small regular file
        small_file_path = os.path.join(clone_dir, "README.md")
        with open(small_file_path, "w") as f:
            f.write("# Test Repository with LFS\n\nThis repo contains large files.\n")

        # Configure git user
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=clone_dir,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=clone_dir,
            check=True,
        )

        # Step 5: Add and commit
        subprocess.run(["git", "add", ".gitattributes"], cwd=clone_dir, check=True)
        subprocess.run(["git", "add", "large_data.bin"], cwd=clone_dir, check=True)
        subprocess.run(["git", "add", "README.md"], cwd=clone_dir, check=True)

        result = subprocess.run(
            ["git", "commit", "-m", "Add large file with LFS"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"git commit failed: {result.stderr}"

        # Step 6: Push with authentication
        result = subprocess.run(
            ["git", "push", push_url, "main"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )

        assert result.returncode == 0, \
            f"git push failed: stdout={result.stdout}, stderr={result.stderr}"

        # Step 7: Clone to a new directory
        clone_dir2 = os.path.join(tmpdir, "repo2")
        result = subprocess.run(
            ["git", "clone", push_url, clone_dir2],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )

        assert result.returncode == 0, \
            f"git clone (after LFS push) failed: stdout={result.stdout}, stderr={result.stderr}"

        # Step 8: Verify the large file
        cloned_large_file = os.path.join(clone_dir2, "large_data.bin")
        assert os.path.exists(cloned_large_file), \
            f"Cloned repo should contain large_data.bin, dir contents: {os.listdir(clone_dir2)}"

        # Verify size
        cloned_size = os.path.getsize(cloned_large_file)
        assert cloned_size == large_file_size, \
            f"Cloned file size mismatch: expected {large_file_size}, got {cloned_size}"

        # Verify content hash
        with open(cloned_large_file, "rb") as f:
            cloned_content = f.read()
        cloned_hash = hashlib.sha256(cloned_content).hexdigest()
        assert cloned_hash == large_file_hash, \
            f"Cloned file hash mismatch: expected {large_file_hash}, got {cloned_hash}"

    # Cleanup: delete the artifact
    await artifact_manager.delete(artifact_id=artifact_alias)


async def test_lfs_batch_api_via_http(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test the LFS batch API endpoint directly via HTTP."""
    import base64
    import uuid
    from hypha_rpc import connect_to_server

    # Connect to the server and create a git-storage artifact
    api = await connect_to_server({
        "name": "lfs-batch-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"lfs-batch-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "LFS Batch Test"},
        config={"storage": "git"},
    )
    await artifact_manager.commit(artifact_id=artifact_alias)

    workspace = api.config.workspace
    batch_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}/info/lfs/objects/batch"

    # Test batch download request for non-existent object
    test_oid = "d" * 64
    batch_request = {
        "operation": "download",
        "objects": [{"oid": test_oid, "size": 100}]
    }

    resp = requests.post(
        batch_url,
        json=batch_request,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        timeout=10,
    )

    assert resp.status_code == 200, f"Batch request failed: {resp.text}"
    data = resp.json()
    assert data["transfer"] == "basic"
    assert len(data["objects"]) == 1
    # Object doesn't exist, should have error
    assert data["objects"][0]["error"]["code"] == 404

    # Test batch upload request (requires auth)
    upload_request = {
        "operation": "upload",
        "objects": [{"oid": "e" * 64, "size": 1024}]
    }

    # Without auth should fail
    resp = requests.post(
        batch_url,
        json=upload_request,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        timeout=10,
    )
    assert resp.status_code == 401, f"Expected 401 without auth, got {resp.status_code}"

    # With auth should succeed
    auth_header = base64.b64encode(f"git:{test_user_token}".encode()).decode()
    resp = requests.post(
        batch_url,
        json=upload_request,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "Authorization": f"Basic {auth_header}",
        },
        timeout=10,
    )

    assert resp.status_code == 200, f"Batch upload request failed: {resp.text}"
    data = resp.json()
    assert len(data["objects"]) == 1
    obj = data["objects"][0]
    assert "upload" in obj["actions"]
    assert "verify" in obj["actions"]
    assert obj["actions"]["upload"]["href"] is not None

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
