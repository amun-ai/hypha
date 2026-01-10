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


def get_http_session():
    """Create an HTTP session that ignores ~/.netrc credentials.

    This prevents stale git credentials from being sent with HTTP requests,
    which would cause 401 errors when tokens don't match the test JWT_SECRET.
    """
    session = requests.Session()
    session.trust_env = False
    return session


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
    # Note: Git-storage artifacts don't use staging/commit - they're ready immediately

    workspace = api.config.workspace
    port = SIO_PORT
    # Use authenticated URL for clone (anonymous users can't access user workspaces)
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "clone")

        # Git clone of empty repo should succeed (or give warning about empty)
        result = subprocess.run(
            ["git", "clone", auth_url, clone_dir],
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
    await api.disconnect()


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
    # Note: Git-storage artifacts don't use staging/commit - they're ready immediately

    workspace = api.config.workspace
    port = SIO_PORT
    # Use authenticated URL (anonymous users can't access user workspaces)
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    # Run git ls-remote
    result = subprocess.run(
        ["git", "ls-remote", auth_url],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Empty repo should return 0
    assert result.returncode == 0, f"ls-remote failed: {result.stderr}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


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
    # Note: Git-storage artifacts don't use staging/commit - they're ready immediately

    workspace = api.config.workspace
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    # Test info/refs endpoint directly with authentication
    # Note: Use auth parameter instead of headers to avoid netrc overwriting the Authorization header
    info_refs_url = f"{base_url}/info/refs?service=git-upload-pack"
    response = requests.get(info_refs_url, auth=("git", test_user_token))
    assert response.status_code == 200
    assert "application/x-git-upload-pack-advertisement" in response.headers.get("content-type", "")
    assert b"# service=git-upload-pack" in response.content

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


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
    # Note: Git-storage artifacts don't use staging/commit - they're ready immediately

    workspace = api.config.workspace
    port = SIO_PORT
    # Use authenticated URL for all git operations (anonymous users can't access user workspaces)
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "repo")

        # Step 1: Clone the empty repository with authentication
        result = subprocess.run(
            ["git", "clone", auth_url, clone_dir],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            assert "empty" in result.stderr.lower() or "warning" in result.stderr.lower(), \
                f"Clone failed unexpectedly: {result.stderr}"
            os.makedirs(clone_dir, exist_ok=True)
            subprocess.run(["git", "init"], cwd=clone_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=clone_dir, check=True)

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
            ["git", "push", auth_url, "main"],
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
            ["git", "clone", auth_url, clone_dir2],
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
    await api.disconnect()


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
    # Note: Git-storage artifacts don't use staging/commit - they're ready immediately

    workspace = api.config.workspace
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
    receive_pack_url = f"{base_url}/git-receive-pack"

    # Test 1: git-receive-pack without auth should return 401
    # Note: The login_required decorator returns 401 for unauthenticated requests
    response = requests.post(receive_pack_url, data=b"", timeout=10)
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    assert "WWW-Authenticate" in response.headers
    assert "Basic" in response.headers["WWW-Authenticate"]

    # Test 2: git-receive-pack with invalid Basic auth should return 401
    # Note: Use auth parameter to avoid netrc overwriting the Authorization header
    response = requests.post(
        receive_pack_url,
        data=b"",
        auth=("git", "invalid-token"),
        timeout=10
    )
    assert response.status_code == 401

    # Test 3: git-receive-pack with valid auth should pass (200 status)
    # Note: Use auth parameter to avoid netrc overwriting the Authorization header
    response = requests.post(
        receive_pack_url,
        data=b"",
        auth=("git", test_user_token),
        timeout=5,
        stream=True,
    )
    # With valid auth, we should get 200 status code (auth passed)
    assert response.status_code == 200
    response.close()

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


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
    # Note: Git-storage artifacts don't use staging/commit - they're ready immediately

    workspace = api.config.workspace
    port = SIO_PORT

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use authenticated URL for all git operations (anonymous users can't access user workspaces)
        auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

        # Step 1: Clone empty repo with authentication
        clone_dir = os.path.join(tmpdir, "repo")
        result = subprocess.run(
            ["git", "clone", auth_url, clone_dir],
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
            subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=clone_dir, check=True)

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
            ["git", "push", auth_url, "main"],
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
            ["git", "clone", auth_url, clone_dir2],
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
    await api.disconnect()


async def test_lfs_batch_api_via_http(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test the LFS batch API endpoint directly via HTTP."""
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
    # Note: Git-storage artifacts don't use staging/commit - they're ready immediately

    workspace = api.config.workspace
    batch_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}/info/lfs/objects/batch"

    # Test batch download request for non-existent object
    test_oid = "d" * 64
    batch_request = {
        "operation": "download",
        "objects": [{"oid": test_oid, "size": 100}]
    }

    # Note: Use auth parameter instead of Authorization header to avoid netrc overwriting
    resp = requests.post(
        batch_url,
        json=batch_request,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        auth=("git", test_user_token),
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
    # Note: Use a custom Session with trust_env=False to prevent netrc from providing credentials
    session = requests.Session()
    session.trust_env = False
    resp = session.post(
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
    # Note: Use auth parameter instead of Authorization header to avoid netrc overwriting
    resp = requests.post(
        batch_url,
        json=upload_request,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        auth=("git", test_user_token),
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
    await api.disconnect()


# Tests for create-zip-file endpoint on git-storage artifacts


async def test_git_create_zip_file(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test creating a ZIP file from a git-storage artifact with pushed content."""
    import subprocess
    import uuid
    import zipfile
    import io
    from hypha_rpc import connect_to_server

    # Connect to the server and create a git-storage artifact
    api = await connect_to_server({
        "name": "git-zip-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-zip-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Zip Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize a local repo and push some files
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create some files
        with open(os.path.join(local_repo, "README.md"), "w") as f:
            f.write("# Test Repository\n\nThis is a test.")
        with open(os.path.join(local_repo, "main.py"), "w") as f:
            f.write('print("Hello, World!")\n')

        # Create a subdirectory with a file
        subdir = os.path.join(local_repo, "src")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "utils.py"), "w") as f:
            f.write('def helper():\n    return "help"\n')

        # Add and commit
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=local_repo, check=True, capture_output=True)

        # Configure remote with embedded credentials
        auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)

        # Push to the Hypha git repo
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Now test the create-zip-file endpoint
    zip_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/create-zip-file"

    # Request ZIP with auth token
    http = get_http_session()
    resp = http.get(zip_url, params={"token": test_user_token}, timeout=30)
    assert resp.status_code == 200, f"Create ZIP failed: {resp.text}"
    assert resp.headers.get("content-type") == "application/zip"

    # Verify ZIP contents
    zip_buffer = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_buffer, "r") as zf:
        names = set(zf.namelist())
        assert "README.md" in names
        assert "main.py" in names
        assert "src/utils.py" in names

        # Verify content
        readme_content = zf.read("README.md").decode()
        assert "Test Repository" in readme_content
        main_content = zf.read("main.py").decode()
        assert 'print("Hello, World!")' in main_content
        utils_content = zf.read("src/utils.py").decode()
        assert "def helper()" in utils_content

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_create_zip_file_with_specific_files(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test creating a ZIP file with specific files from a git-storage artifact."""
    import subprocess
    import uuid
    import zipfile
    import io
    from hypha_rpc import connect_to_server

    # Connect to the server and create a git-storage artifact
    api = await connect_to_server({
        "name": "git-zip-specific-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-zip-specific-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Zip Specific Files Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize a local repo and push some files
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create multiple files
        with open(os.path.join(local_repo, "file1.txt"), "w") as f:
            f.write("File 1 content")
        with open(os.path.join(local_repo, "file2.txt"), "w") as f:
            f.write("File 2 content")
        with open(os.path.join(local_repo, "file3.txt"), "w") as f:
            f.write("File 3 content")

        # Add and commit
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Multiple files"], cwd=local_repo, check=True, capture_output=True)

        # Configure remote with embedded credentials
        auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)

        # Push to the Hypha git repo
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Request ZIP with only specific files
    zip_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/create-zip-file"

    http = get_http_session()
    resp = http.get(
        zip_url,
        params={"token": test_user_token, "file": ["file1.txt", "file3.txt"]},
        timeout=30,
    )
    assert resp.status_code == 200, f"Create ZIP failed: {resp.text}"

    # Verify ZIP contains only requested files
    zip_buffer = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_buffer, "r") as zf:
        names = set(zf.namelist())
        assert names == {"file1.txt", "file3.txt"}

        assert zf.read("file1.txt").decode() == "File 1 content"
        assert zf.read("file3.txt").decode() == "File 3 content"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_create_zip_file_empty_repo(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that create-zip-file returns 404 for empty git-storage artifact."""
    import uuid
    from hypha_rpc import connect_to_server

    # Connect to the server and create a git-storage artifact
    api = await connect_to_server({
        "name": "git-zip-empty-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-zip-empty-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Zip Empty Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    zip_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/create-zip-file"

    # Request ZIP from empty repo should return 404
    http = get_http_session()
    resp = http.get(zip_url, params={"token": test_user_token}, timeout=30)
    assert resp.status_code == 404, f"Expected 404 for empty repo, got {resp.status_code}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_git_lfs_files_endpoint_streaming(
    minio_server,
    fastapi_server,
    test_user_token,
    check_git_lfs_installed,
):
    """Test that LFS-tracked files can be downloaded via /files/ endpoint.

    This test:
    1. Creates a git-storage artifact
    2. Pushes a repo with LFS-tracked large files
    3. Downloads the large file via /files/ endpoint (should stream from S3)
    4. Verifies the content matches the original
    """
    import io
    import subprocess
    import hashlib
    import uuid
    import zipfile
    from hypha_rpc import connect_to_server

    # Connect to the server
    api = await connect_to_server({
        "name": "git-lfs-files-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    # Create git-storage artifact
    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"lfs-files-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "LFS Files Endpoint Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    port = SIO_PORT

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup local repo with LFS
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)

        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Install and configure LFS
        subprocess.run(["git", "lfs", "install"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "lfs", "track", "*.bin"], cwd=local_repo, check=True, capture_output=True)

        # Create a large file (500 KB)
        large_file_path = os.path.join(local_repo, "large_data.bin")
        large_file_size = 500 * 1024  # 500 KB
        large_file_content = os.urandom(large_file_size)
        large_file_hash = hashlib.sha256(large_file_content).hexdigest()

        with open(large_file_path, "wb") as f:
            f.write(large_file_content)

        # Also create a small regular file
        small_file_path = os.path.join(local_repo, "README.md")
        small_content = "# Test Repo\n\nThis repo has LFS files."
        with open(small_file_path, "w") as f:
            f.write(small_content)

        # Add and commit
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add LFS file"], cwd=local_repo, check=True, capture_output=True)

        # Push to Hypha
        git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
        auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)

        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Test 1: Download LFS file via /files/ endpoint
    http = get_http_session()
    files_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/large_data.bin"
    resp = http.get(files_url, params={"token": test_user_token}, timeout=60)
    assert resp.status_code == 200, f"Failed to download LFS file: {resp.text}"

    # Verify content matches original
    downloaded_hash = hashlib.sha256(resp.content).hexdigest()
    assert downloaded_hash == large_file_hash, "Downloaded LFS content doesn't match original"
    assert len(resp.content) == large_file_size, f"Size mismatch: expected {large_file_size}, got {len(resp.content)}"

    # Test 2: Download small regular file (should still work)
    small_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/README.md"
    resp = http.get(small_url, params={"token": test_user_token}, timeout=30)
    assert resp.status_code == 200
    assert resp.text == small_content

    # Test 3: Create ZIP with LFS file (verify streaming works)
    zip_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/create-zip-file"
    resp = http.get(zip_url, params={"token": test_user_token}, timeout=60)
    assert resp.status_code == 200, f"Create ZIP failed: {resp.text}"

    # Verify ZIP contents
    zip_buffer = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_buffer, "r") as zf:
        names = set(zf.namelist())
        assert "large_data.bin" in names, f"LFS file not in ZIP: {names}"
        assert "README.md" in names

        # Verify LFS content in ZIP
        lfs_content = zf.read("large_data.bin")
        lfs_hash = hashlib.sha256(lfs_content).hexdigest()
        assert lfs_hash == large_file_hash, "LFS content in ZIP doesn't match original"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


# Tests for Git Artifact API methods (list_files, get_file, put_file, commit, remove_file)


async def test_git_artifact_list_files(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test list_files() on a git-storage artifact after git push."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-list-files-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-list-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git List Files Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize a local repo and push some files
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create files
        with open(os.path.join(local_repo, "README.md"), "w") as f:
            f.write("# Test Repository\n")
        with open(os.path.join(local_repo, "main.py"), "w") as f:
            f.write('print("Hello")\n')

        # Create a subdirectory with file
        subdir = os.path.join(local_repo, "src")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "utils.py"), "w") as f:
            f.write("def helper(): pass\n")

        # Add and commit
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=local_repo, check=True, capture_output=True)

        # Configure remote and push
        auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Test list_files on root
    files = await artifact_manager.list_files(artifact_id=artifact_alias)
    names = [f["name"] for f in files]
    assert "README.md" in names
    assert "main.py" in names
    assert "src" in names

    # Check types
    src_entry = [f for f in files if f["name"] == "src"][0]
    assert src_entry["type"] == "directory"

    readme_entry = [f for f in files if f["name"] == "README.md"][0]
    assert readme_entry["type"] == "file"
    assert readme_entry.get("size") is not None

    # Test list_files on subdirectory
    sub_files = await artifact_manager.list_files(artifact_id=artifact_alias, dir_path="src")
    sub_names = [f["name"] for f in sub_files]
    assert "utils.py" in sub_names

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_get_file(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test get_file() on a git-storage artifact to get file content."""
    import subprocess
    import uuid
    import base64
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-get-file-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-getfile-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Get File Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    file_content = "# Hello Git World\n\nThis is test content."

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize a local repo and push a file
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create a file
        with open(os.path.join(local_repo, "README.md"), "w") as f:
            f.write(file_content)

        # Add, commit, push
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=local_repo, check=True, capture_output=True)
        auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Test get_file - for git storage, regular files return dict with base64 content
    result = await artifact_manager.get_file(artifact_id=artifact_alias, file_path="README.md")

    # For git storage, regular files return a dict with content_base64
    if isinstance(result, dict) and result.get("_type") == "git_file_content":
        content = base64.b64decode(result["content_base64"]).decode()
        assert content == file_content
    elif isinstance(result, str) and result.startswith("http"):
        # If it's an LFS file or proxy URL, fetch it
        resp = requests.get(result, timeout=30)
        assert resp.status_code == 200
        assert file_content in resp.text
    else:
        # Direct content (shouldn't happen for git, but handle it)
        assert file_content in str(result)

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_list_files_empty_repo(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test list_files() on an empty git-storage artifact returns empty list."""
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-empty-list-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-empty-list-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Empty List Test"},
        config={"storage": "git"},
    )

    # list_files on empty repo should return empty list
    files = await artifact_manager.list_files(artifact_id=artifact_alias)
    assert files == []

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_put_file_and_commit(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test put_file() and commit() on a git-storage artifact."""
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-put-commit-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-put-commit-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Put/Commit Test"},
        config={"storage": "git"},
    )

    # Enter staging mode before uploading files
    await artifact_manager.edit(artifact_id=artifact_alias, stage=True)

    # Step 1: Get presigned URL for upload
    put_url = await artifact_manager.put_file(artifact_id=artifact_alias, file_path="test.txt")
    assert put_url is not None
    assert isinstance(put_url, str)
    assert "test.txt" in put_url or "staging" in put_url

    # Step 2: Upload content to the presigned URL
    file_content = b"Hello from put_file API test!"
    resp = requests.put(put_url, data=file_content, timeout=30)
    assert resp.status_code == 200, f"Upload failed: {resp.text}"

    # Step 3: Commit the staged file
    result = await artifact_manager.commit(artifact_id=artifact_alias, comment="Add test.txt via API")
    assert result is not None

    # Step 4: Verify file is now in the repository
    files = await artifact_manager.list_files(artifact_id=artifact_alias)
    names = [f["name"] for f in files]
    assert "test.txt" in names

    # Step 5: Get the file content to verify
    workspace = api.config.workspace
    file_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/test.txt"
    http = get_http_session()
    resp = http.get(file_url, params={"token": test_user_token}, timeout=30)
    assert resp.status_code == 200
    assert resp.content == file_content

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_remove_file(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test remove_file() on a git-storage artifact."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-remove-file-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-remove-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Remove File Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize a local repo and push files
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create files
        with open(os.path.join(local_repo, "keep.txt"), "w") as f:
            f.write("Keep this file\n")
        with open(os.path.join(local_repo, "delete.txt"), "w") as f:
            f.write("Delete this file\n")

        # Add, commit, push
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=local_repo, check=True, capture_output=True)
        auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Verify both files exist
    files = await artifact_manager.list_files(artifact_id=artifact_alias)
    names = [f["name"] for f in files]
    assert "keep.txt" in names
    assert "delete.txt" in names

    # Enter staging mode before removing files
    await artifact_manager.edit(artifact_id=artifact_alias, stage=True)

    # Step 1: Remove the file (stages it for removal)
    result = await artifact_manager.remove_file(artifact_id=artifact_alias, file_path="delete.txt")
    assert result is not None

    # Step 2: Commit the removal
    commit_result = await artifact_manager.commit(artifact_id=artifact_alias, comment="Remove delete.txt")
    assert commit_result is not None

    # Step 3: Verify file is gone
    files = await artifact_manager.list_files(artifact_id=artifact_alias)
    names = [f["name"] for f in files]
    assert "keep.txt" in names
    assert "delete.txt" not in names

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_version_resolution(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test list_files with version parameter (branch/tag) on git-storage artifact."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-version-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-version-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Version Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    port = SIO_PORT
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create first commit with v1 content
        with open(os.path.join(local_repo, "file.txt"), "w") as f:
            f.write("Version 1\n")
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "v1"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "tag", "v1.0"], cwd=local_repo, check=True, capture_output=True)

        # Create second commit with v2 content
        with open(os.path.join(local_repo, "file.txt"), "w") as f:
            f.write("Version 2\n")
        with open(os.path.join(local_repo, "new_file.txt"), "w") as f:
            f.write("New file in v2\n")
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "v2"], cwd=local_repo, check=True, capture_output=True)

        # Push with tags
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main", "--tags"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Test 1: List files at HEAD (should have both files)
    files_head = await artifact_manager.list_files(artifact_id=artifact_alias)
    names_head = [f["name"] for f in files_head]
    assert "file.txt" in names_head
    assert "new_file.txt" in names_head

    # Test 2: List files at tag v1.0 (should only have file.txt)
    files_v1 = await artifact_manager.list_files(artifact_id=artifact_alias, version="v1.0")
    names_v1 = [f["name"] for f in files_v1]
    assert "file.txt" in names_v1
    assert "new_file.txt" not in names_v1

    # Test 3: List files with "main" branch name
    files_main = await artifact_manager.list_files(artifact_id=artifact_alias, version="main")
    names_main = [f["name"] for f in files_main]
    assert "file.txt" in names_main
    assert "new_file.txt" in names_main

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_custom_default_branch(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test creating a git-storage artifact with a custom default branch."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-custom-branch-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-custom-branch-{uuid.uuid4().hex[:8]}"
    # Create with custom default branch "develop"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Custom Branch Test"},
        config={"storage": "git", "git_default_branch": "develop"},
    )

    workspace = api.config.workspace
    port = SIO_PORT
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init", "-b", "develop"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create a file
        with open(os.path.join(local_repo, "README.md"), "w") as f:
            f.write("# Custom Branch Repo\n")

        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=local_repo, check=True, capture_output=True)

        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "develop"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Verify files are listed (using default version which should be develop)
    files = await artifact_manager.list_files(artifact_id=artifact_alias)
    names = [f["name"] for f in files]
    assert "README.md" in names

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_get_file_by_commit_sha(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test get_file with a specific commit SHA as version."""
    import subprocess
    import uuid
    import base64
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-commit-sha-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-sha-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git SHA Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
    auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")

    first_commit_sha = None

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # First commit
        with open(os.path.join(local_repo, "file.txt"), "w") as f:
            f.write("Version 1 content\n")
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "v1"], cwd=local_repo, check=True, capture_output=True)

        # Get first commit SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=local_repo,
            capture_output=True,
            text=True,
        )
        first_commit_sha = result.stdout.strip()

        # Second commit with different content
        with open(os.path.join(local_repo, "file.txt"), "w") as f:
            f.write("Version 2 content\n")
        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "v2"], cwd=local_repo, check=True, capture_output=True)

        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Get file at HEAD (should be v2)
    result_head = await artifact_manager.get_file(artifact_id=artifact_alias, file_path="file.txt")
    if isinstance(result_head, dict) and result_head.get("_type") == "git_file_content":
        content_head = base64.b64decode(result_head["content_base64"]).decode()
    else:
        resp = requests.get(result_head, timeout=30)
        content_head = resp.text
    assert "Version 2" in content_head

    # Get file at first commit SHA (should be v1) - use full 40-character SHA
    # Note: Partial SHA matching is not currently implemented
    result_v1 = await artifact_manager.get_file(
        artifact_id=artifact_alias,
        file_path="file.txt",
        version=first_commit_sha  # Full 40-char SHA
    )
    if isinstance(result_v1, dict) and result_v1.get("_type") == "git_file_content":
        content_v1 = base64.b64decode(result_v1["content_base64"]).decode()
    else:
        resp = requests.get(result_v1, timeout=30)
        content_v1 = resp.text
    assert "Version 1" in content_v1

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_nested_directories(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test list_files with deeply nested directory structure."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-nested-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-nested-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Nested Dir Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
    auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create deeply nested structure
        deep_path = os.path.join(local_repo, "a", "b", "c", "d")
        os.makedirs(deep_path)
        with open(os.path.join(deep_path, "deep_file.txt"), "w") as f:
            f.write("Deep nested content\n")

        # Also add file at intermediate level
        mid_path = os.path.join(local_repo, "a", "b")
        with open(os.path.join(mid_path, "mid_file.txt"), "w") as f:
            f.write("Mid level content\n")

        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Nested structure"], cwd=local_repo, check=True, capture_output=True)

        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # List root - should have "a" directory
    files_root = await artifact_manager.list_files(artifact_id=artifact_alias)
    names = [f["name"] for f in files_root]
    assert "a" in names
    a_entry = [f for f in files_root if f["name"] == "a"][0]
    assert a_entry["type"] == "directory"

    # List a/b - should have "c" and "mid_file.txt"
    files_ab = await artifact_manager.list_files(artifact_id=artifact_alias, dir_path="a/b")
    names_ab = [f["name"] for f in files_ab]
    assert "c" in names_ab
    assert "mid_file.txt" in names_ab

    # List a/b/c/d - should have deep_file.txt
    files_deep = await artifact_manager.list_files(artifact_id=artifact_alias, dir_path="a/b/c/d")
    names_deep = [f["name"] for f in files_deep]
    assert "deep_file.txt" in names_deep

    # Get deep file content via files endpoint
    deep_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/a/b/c/d/deep_file.txt"
    http = get_http_session()
    resp = http.get(deep_url, params={"token": test_user_token}, timeout=30)
    assert resp.status_code == 200
    assert "Deep nested content" in resp.text

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_binary_file(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test handling binary files in git (non-LFS) storage."""
    import subprocess
    import uuid
    import hashlib
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-binary-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-binary-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Binary File Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    git_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
    auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")

    # Create small binary content (under LFS threshold)
    binary_content = os.urandom(1024)  # 1KB random binary data
    binary_hash = hashlib.sha256(binary_content).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create binary file
        with open(os.path.join(local_repo, "data.dat"), "wb") as f:
            f.write(binary_content)

        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add binary file"], cwd=local_repo, check=True, capture_output=True)

        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Verify file is listed
    files = await artifact_manager.list_files(artifact_id=artifact_alias)
    names = [f["name"] for f in files]
    assert "data.dat" in names

    # Download and verify binary content
    file_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/data.dat"
    http = get_http_session()
    resp = http.get(file_url, params={"token": test_user_token}, timeout=30)
    assert resp.status_code == 200

    downloaded_hash = hashlib.sha256(resp.content).hexdigest()
    assert downloaded_hash == binary_hash, "Binary file content mismatch"
    assert len(resp.content) == 1024

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_read_returns_git_url(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that artifact.read() returns git_url for git-storage artifacts."""
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-url-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-url-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git URL Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace

    # Read artifact info
    artifact_info = await artifact_manager.read(artifact_id=artifact_alias)

    # Should have git_url field
    assert "git_url" in artifact_info, "git_url should be present in artifact info"
    assert f"/{workspace}/git/{artifact_alias}" in artifact_info["git_url"]

    # Also verify config shows storage as git
    assert artifact_info.get("config", {}).get("storage") == "git"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()
