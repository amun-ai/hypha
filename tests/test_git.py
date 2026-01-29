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
    # Username must be 'git'; the token determines authorization
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
    # Verify symref capability is advertised (crucial for git clients to know default branch)
    assert b"symref=HEAD:refs/heads/main" in response.content, \
        f"symref capability not found in info/refs response. Content: {response.content[:500]}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_auth_requires_git_username(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that git authentication requires 'git' as the username.

    The username in Basic Auth must be 'git'. The token (password) contains
    the actual user identity for authorization. Using a fixed username avoids
    confusion and simplifies client configuration.
    """
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-username-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-username-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Username Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    port = SIO_PORT

    # Test with 'git' username - should succeed
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"
    result = subprocess.run(
        ["git", "ls-remote", auth_url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"ls-remote with 'git' username failed: {result.stderr}"

    # Test with wrong username - should fail with 401
    wrong_username_url = f"http://wrong-user:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"
    result = subprocess.run(
        ["git", "ls-remote", wrong_username_url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Git returns non-zero exit code on auth failure
    assert result.returncode != 0, f"ls-remote with wrong username should have failed"
    assert "401" in result.stderr or "Authentication failed" in result.stderr or "fatal" in result.stderr.lower(), \
        f"Expected auth error, got: {result.stderr}"

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
            # Use -b main to ensure consistent branch name across platforms
            subprocess.run(["git", "init", "-b", "main"], cwd=clone_dir, check=True)
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

        # Step 2.5: Ensure we're on the main branch
        # Some git versions may create a different default branch even with symref
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        current_branch = result.stdout.strip()
        if current_branch and current_branch != "main":
            # Rename the branch to main for consistency
            subprocess.run(
                ["git", "branch", "-m", current_branch, "main"],
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

        # Step 5: Push with authentication (push main to main on remote)
        result = subprocess.run(
            ["git", "push", auth_url, "main:main"],
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
        assert os.path.exists(cloned_file), (
            f"Cloned repo should contain README.md. "
            f"Clone stderr: {result.stderr}"
        )

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

    # Test 3: git-receive-pack with wrong username should return 401
    # Username must be 'git'
    response = requests.post(
        receive_pack_url,
        data=b"",
        auth=("wrong-user", test_user_token),
        timeout=5,
    )
    assert response.status_code == 401, "Wrong username should be rejected"

    # Test 4: git-receive-pack with valid auth should pass (200 status)
    # Note: Use auth parameter to avoid netrc overwriting the Authorization header
    # Username must be 'git', token provides authorization
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


async def test_git_clone_non_public_requires_auth(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that git clone of non-public repos requires authentication.

    Security test: Verifies that anonymous users cannot clone/fetch
    from git repositories that don't have public read permissions.
    """
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-security-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a NON-PUBLIC git artifact (no "*" permission)
    artifact_alias = f"git-private-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Private Git Repo"},
        config={
            "storage": "git",
            # No "*" permission means non-public
        },
    )

    workspace = api.config.workspace
    port = SIO_PORT

    # Test: Anonymous access should be denied
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
    info_refs_url = f"{base_url}/info/refs?service=git-upload-pack"

    # Anonymous request should get 401
    session = get_http_session()  # Don't use netrc
    response = session.get(info_refs_url, timeout=10)
    assert response.status_code == 401, \
        f"Anonymous access to non-public repo should return 401, got {response.status_code}"
    assert "WWW-Authenticate" in response.headers

    # Verify git clone without auth fails
    with tempfile.TemporaryDirectory() as tmpdir:
        git_url = f"http://127.0.0.1:{port}/{workspace}/git/{artifact_alias}"
        # Create a clean git environment that disables all credential helpers
        git_env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",  # Disable credential prompt
            "GIT_ASKPASS": "",  # Disable askpass program
            "SSH_ASKPASS": "",
            "GIT_CONFIG_NOSYSTEM": "1",  # Ignore system config
            "HOME": tmpdir,  # Use temp dir as home to avoid user's .gitconfig and credential store
            "XDG_CONFIG_HOME": tmpdir,  # Avoid XDG configs
        }
        result = subprocess.run(
            ["git", "-c", "credential.helper=", "clone", git_url, os.path.join(tmpdir, "repo")],
            capture_output=True,
            text=True,
            timeout=30,
            env=git_env,
        )
        # Clone should fail with authentication error
        assert result.returncode != 0, \
            f"Git clone of non-public repo without auth should fail, but succeeded. stderr: {result.stderr}"
        assert "401" in result.stderr or "authentication" in result.stderr.lower() or "fatal" in result.stderr.lower(), \
            f"Expected authentication error in stderr: {result.stderr}"

    # Test: Authenticated access should work (username must be 'git')
    response = session.get(
        info_refs_url,
        auth=("git", test_user_token),
        timeout=10
    )
    assert response.status_code == 200, \
        f"Authenticated access should succeed, got {response.status_code}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_clone_public_without_auth(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that git clone of PUBLIC repos works without authentication.

    Verifies that artifacts with "*" read permission can be cloned anonymously.
    """
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-public-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a PUBLIC git artifact with "*" read permission
    artifact_alias = f"git-public-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Public Git Repo"},
        config={
            "storage": "git",
            "permissions": {
                "*": "r"  # Public read permission
            }
        },
    )

    workspace = api.config.workspace
    port = SIO_PORT

    # Test: Anonymous access should work for public repos
    base_url = f"{SERVER_URL}/{workspace}/git/{artifact_alias}"
    info_refs_url = f"{base_url}/info/refs?service=git-upload-pack"

    session = get_http_session()  # Don't use netrc
    response = session.get(info_refs_url, timeout=10)
    assert response.status_code == 200, \
        f"Anonymous access to public repo should succeed, got {response.status_code}"

    # Verify anonymous git clone works for public repo
    with tempfile.TemporaryDirectory() as tmpdir:
        git_url = f"http://127.0.0.1:{port}/{workspace}/git/{artifact_alias}"
        result = subprocess.run(
            ["git", "clone", git_url, os.path.join(tmpdir, "repo")],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        # Clone should succeed (or show "empty repository" warning which is OK)
        # Note: An empty repo clone might return 0 or have "warning: You appear to have cloned an empty repository"
        assert result.returncode == 0 or "empty repository" in result.stderr.lower(), \
            f"Git clone of public repo should succeed: {result.stderr}"

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
            # Use -b main to ensure consistent branch name across platforms
            subprocess.run(["git", "init", "-b", "main"], cwd=clone_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=clone_dir, check=True)

        # Ensure we're on the main branch (some git versions may use different default)
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=clone_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        current_branch = result.stdout.strip()
        if current_branch and current_branch != "main":
            subprocess.run(
                ["git", "branch", "-m", current_branch, "main"],
                cwd=clone_dir,
                check=True,
            )

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

        # Step 6: Push with authentication (push main to main on remote)
        result = subprocess.run(
            ["git", "push", auth_url, "main:main"],
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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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

        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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

    # Test get_file - should always return a URL (same interface as raw storage)
    result = await artifact_manager.get_file(artifact_id=artifact_alias, file_path="README.md")

    # Result should be a URL string
    assert isinstance(result, str), f"Expected URL string, got {type(result)}"
    assert result.startswith("http"), f"Expected URL, got: {result}"

    # Fetch the file content via the URL
    http = get_http_session()
    resp = http.get(result, timeout=30)
    assert resp.status_code == 200
    assert file_content in resp.text

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
    # Verify staging is None after commit (critical for UI status display)
    assert result.get("staging") is None, f"Staging should be None after commit, got: {result.get('staging')}"

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


async def test_git_artifact_list_staged_files(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test list_files() with stage=True on a git-storage artifact.

    This test verifies that files uploaded to staging can be listed before commit.
    This is a critical feature for the web UI to show uploaded files before they
    are committed to the git repository.
    """
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-list-staged-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-list-staged-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git List Staged Test"},
        config={"storage": "git"},
    )

    # Enter staging mode before uploading files
    await artifact_manager.edit(artifact_id=artifact_alias, stage=True)

    # Upload multiple files to staging
    put_url1 = await artifact_manager.put_file(artifact_id=artifact_alias, file_path="file1.txt")
    resp = requests.put(put_url1, data=b"Content of file 1", timeout=30)
    assert resp.status_code == 200

    put_url2 = await artifact_manager.put_file(artifact_id=artifact_alias, file_path="file2.txt")
    resp = requests.put(put_url2, data=b"Content of file 2", timeout=30)
    assert resp.status_code == 200

    # Upload a file in a subdirectory
    put_url3 = await artifact_manager.put_file(artifact_id=artifact_alias, file_path="subdir/file3.txt")
    resp = requests.put(put_url3, data=b"Content of file 3 in subdir", timeout=30)
    assert resp.status_code == 200

    # List staged files (before commit) - this should work!
    staged_files = await artifact_manager.list_files(artifact_id=artifact_alias, stage=True)
    staged_names = [f["name"] for f in staged_files]

    # Verify all staged files are listed
    assert "file1.txt" in staged_names, f"file1.txt not in staged files: {staged_names}"
    assert "file2.txt" in staged_names, f"file2.txt not in staged files: {staged_names}"
    # Note: subdir/file3.txt might be listed as "subdir" directory or full path depending on implementation
    # At root level we should see either "subdir" directory or the files

    # Verify files have the staged marker
    for f in staged_files:
        if f["type"] == "file":
            assert f.get("staged") is True, f"File {f['name']} should have staged=True"

    # List files without stage=True - should be empty since nothing is committed yet
    committed_files = await artifact_manager.list_files(artifact_id=artifact_alias)
    # For git storage, an empty repo or no commits means empty list
    assert len(committed_files) == 0 or committed_files == [], f"Expected empty committed files, got: {committed_files}"

    # Now commit and verify files appear in committed listing
    result = await artifact_manager.commit(artifact_id=artifact_alias, comment="Add staged files")
    assert result is not None

    # List files after commit (without stage=True) - should now see files
    committed_files = await artifact_manager.list_files(artifact_id=artifact_alias)
    committed_names = [f["name"] for f in committed_files]
    assert "file1.txt" in committed_names, f"file1.txt not in committed files: {committed_names}"
    assert "file2.txt" in committed_names, f"file2.txt not in committed files: {committed_names}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_concurrent_put_file(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that concurrent put_file calls don't lose files due to race conditions.

    This is a critical test to verify that when multiple put_file calls are made
    in parallel (as the web UI does), all files are properly tracked in staging
    and committed to the git repository.

    Previously, this was broken because each put_file call would load the artifact,
    add its file to staging, and save - without proper locking, later calls would
    overwrite earlier ones.
    """
    import uuid
    import asyncio
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-concurrent-put-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-concurrent-put-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Concurrent Put Test"},
        config={"storage": "git"},
    )

    # Enter staging mode
    await artifact_manager.edit(artifact_id=artifact_alias, stage=True)

    # Simulate web UI behavior: call put_file for multiple files concurrently
    num_files = 5
    file_names = [f"concurrent_file_{i}.txt" for i in range(num_files)]

    async def upload_file(file_name):
        """Upload a single file (simulating web UI parallel upload)."""
        put_url = await artifact_manager.put_file(artifact_id=artifact_alias, file_path=file_name)
        content = f"Content of {file_name}".encode()
        resp = requests.put(put_url, data=content, timeout=30)
        assert resp.status_code == 200, f"Upload failed for {file_name}: {resp.text}"
        return file_name

    # Run all uploads concurrently (this is what the web UI does)
    upload_tasks = [upload_file(name) for name in file_names]
    uploaded = await asyncio.gather(*upload_tasks)
    assert len(uploaded) == num_files

    # List staged files - all files should be present
    staged_files = await artifact_manager.list_files(artifact_id=artifact_alias, stage=True)
    staged_names = [f["name"] for f in staged_files]

    # This is the critical assertion - all files should be in staging
    for file_name in file_names:
        assert file_name in staged_names, (
            f"RACE CONDITION: {file_name} was lost in staging! "
            f"Only found: {staged_names}"
        )

    # Commit and verify all files made it to git
    result = await artifact_manager.commit(artifact_id=artifact_alias, comment="Add concurrent files")
    assert result is not None
    assert result.get("file_count") == num_files, (
        f"Expected {num_files} files committed, got {result.get('file_count')}"
    )

    # List committed files
    committed_files = await artifact_manager.list_files(artifact_id=artifact_alias)
    committed_names = [f["name"] for f in committed_files]

    for file_name in file_names:
        assert file_name in committed_names, (
            f"RACE CONDITION: {file_name} was lost during commit! "
            f"Only found: {committed_names}"
        )

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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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

    # Get file at HEAD (should be v2) - always returns URL
    result_head = await artifact_manager.get_file(artifact_id=artifact_alias, file_path="file.txt")
    assert isinstance(result_head, str) and result_head.startswith("http")
    http = get_http_session()
    resp = http.get(result_head, timeout=30)
    assert resp.status_code == 200
    assert "Version 2" in resp.text

    # Get file at first commit SHA (should be v1) - use full 40-character SHA
    # Note: Partial SHA matching is not currently implemented
    result_v1 = await artifact_manager.get_file(
        artifact_id=artifact_alias,
        file_path="file.txt",
        version=first_commit_sha  # Full 40-char SHA
    )
    assert isinstance(result_v1, str) and result_v1.startswith("http")
    resp = http.get(result_v1, timeout=30)
    assert resp.status_code == 200
    assert "Version 1" in resp.text

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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
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


async def test_git_file_url_specialized_token_security(
    minio_server, fastapi_server, test_user_token
):
    """Test that git file URLs use specialized tokens that can't be used for WebSocket.

    This is a critical security test. The token generated for git file URLs should:
    1. Allow downloading files via the HTTP endpoint
    2. NOT allow connecting to WebSocket for general workspace access

    This prevents token leakage from file URLs being exploited for workspace access.
    """
    from hypha_rpc import connect_to_server
    import aiohttp
    import uuid
    from urllib.parse import urlparse, parse_qs

    api = await connect_to_server(
        {"name": "test-client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config.workspace
    artifact_alias = f"git-security-test-{uuid.uuid4().hex[:8]}"

    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a git artifact
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Security Test"},
        config={"storage": "git"},
    )

    # Enter staging mode and upload a file
    await artifact_manager.edit(artifact_id=artifact_alias, stage=True)
    put_url = await artifact_manager.put_file(artifact_id=artifact_alias, file_path="test.txt")
    async with aiohttp.ClientSession() as session:
        async with session.put(put_url, data=b"secret content") as resp:
            assert resp.status == 200

    await artifact_manager.commit(artifact_id=artifact_alias, comment="Add test file")

    # Get the file URL - this contains a specialized token
    file_url = await artifact_manager.get_file(
        artifact_id=artifact_alias, file_path="test.txt"
    )

    # Extract the token from the URL
    parsed_url = urlparse(file_url)
    query_params = parse_qs(parsed_url.query)
    token = query_params.get("token", [None])[0]
    assert token is not None, "Token should be present in file URL"

    # Verify the token works for file download
    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            assert response.status == 200
            content = await response.read()
            assert content == b"secret content"

    # Verify the token CANNOT be used for WebSocket connection
    # This is the critical security check
    try:
        api2 = await connect_to_server(
            {
                "name": "attacker-client",
                "server_url": WS_SERVER_URL,
                "token": token,
                "workspace": workspace,
            }
        )
        await api2.disconnect()
        raise AssertionError(
            "SECURITY FAILURE: Specialized file download token was able to connect to WebSocket!"
        )
    except Exception as e:
        error_msg = str(e)
        # Should contain error about specialized token or general workspace access
        assert (
            "Specialized token" in error_msg
            or "general workspace access" in error_msg
        ), f"Expected specialized token rejection error, got: {error_msg}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_file_url_token_cannot_access_other_files(
    minio_server, fastapi_server, test_user_token
):
    """Test that a file URL token cannot be used to download other files.

    This tests the token hijacking prevention - a token generated for one file
    should NOT work for downloading a different file, even in the same artifact.
    """
    from hypha_rpc import connect_to_server
    import aiohttp
    import uuid
    from urllib.parse import urlparse, parse_qs

    api = await connect_to_server(
        {"name": "test-client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config.workspace
    artifact_alias = f"git-hijack-test-{uuid.uuid4().hex[:8]}"

    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a git artifact with two files
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Token Hijack Test"},
        config={"storage": "git"},
    )

    # Enter staging mode and upload two files
    await artifact_manager.edit(artifact_id=artifact_alias, stage=True)

    put_url1 = await artifact_manager.put_file(artifact_id=artifact_alias, file_path="file1.txt")
    async with aiohttp.ClientSession() as session:
        async with session.put(put_url1, data=b"content of file 1") as resp:
            assert resp.status == 200

    put_url2 = await artifact_manager.put_file(artifact_id=artifact_alias, file_path="file2.txt")
    async with aiohttp.ClientSession() as session:
        async with session.put(put_url2, data=b"secret content of file 2") as resp:
            assert resp.status == 200

    await artifact_manager.commit(artifact_id=artifact_alias, comment="Add two files")

    # Get the URL for file1 - this contains a specialized token for file1 only
    file1_url = await artifact_manager.get_file(
        artifact_id=artifact_alias, file_path="file1.txt"
    )

    # Extract the token from file1's URL
    parsed_url = urlparse(file1_url)
    query_params = parse_qs(parsed_url.query)
    file1_token = query_params.get("token", [None])[0]
    assert file1_token is not None

    # Verify file1's token works for file1
    async with aiohttp.ClientSession() as session:
        async with session.get(file1_url) as response:
            assert response.status == 200
            content = await response.read()
            assert content == b"content of file 1"

    # Try to use file1's token to download file2 - this should FAIL
    file2_url_with_file1_token = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/file2.txt?token={file1_token}"
    async with aiohttp.ClientSession() as session:
        async with session.get(file2_url_with_file1_token) as response:
            # Should get 403 Forbidden or 500 Internal Server Error (permission denied)
            assert response.status in [403, 500], (
                f"Expected 403/500, got {response.status}. "
                f"SECURITY FAILURE: Token for file1 was able to download file2!"
            )
            # Verify error message mentions scope mismatch
            error_text = await response.text()
            assert "scope" in error_text.lower() or "permission" in error_text.lower(), (
                f"Expected scope/permission error, got: {error_text}"
            )

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


# Tests for read_file and write_file with git storage


async def test_git_read_write_file(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test read_file and write_file with git-storage artifact."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    # Connect to the server and create a git-storage artifact
    api = await connect_to_server({
        "name": "git-readwrite-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-rw-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Read Write Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    port = SIO_PORT
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize local repo and push initial files
        repo_dir = os.path.join(tmpdir, "repo")
        result = subprocess.run(
            ["git", "clone", auth_url, repo_dir],
            capture_output=True, check=False, timeout=30
        )

        # If clone failed (empty repo), initialize manually
        if result.returncode != 0:
            os.makedirs(repo_dir, exist_ok=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True, check=True)
            subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=repo_dir, capture_output=True, check=True)

        # Setup git config first
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True, check=True)

        # Ensure we're on the main branch (some git versions may use different default)
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        current_branch = branch_result.stdout.strip()
        if current_branch and current_branch != "main":
            subprocess.run(["git", "branch", "-m", current_branch, "main"], cwd=repo_dir, check=True)

        # Create test files
        with open(os.path.join(repo_dir, "readme.txt"), "w") as f:
            f.write("Hello, this is the initial readme content!")

        with open(os.path.join(repo_dir, "data.json"), "w") as f:
            f.write('{"key": "initial_value"}')

        # Add and commit
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "push", auth_url, "main:main", "--force"], cwd=repo_dir, capture_output=True, check=True, timeout=30)

    # Test 1: Read file as text
    result = await artifact_manager.read_file(
        artifact_id=artifact_alias,
        file_path="readme.txt",
        format="text"
    )
    assert result["name"] == "readme.txt"
    assert result["content"] == "Hello, this is the initial readme content!"

    # Test 2: Read file as base64
    result = await artifact_manager.read_file(
        artifact_id=artifact_alias,
        file_path="data.json",
        format="base64"
    )
    assert result["name"] == "data.json"
    import base64
    decoded = base64.b64decode(result["content"]).decode("utf-8")
    assert decoded == '{"key": "initial_value"}'

    # Test 3: Read file with partial read (offset/limit)
    result = await artifact_manager.read_file(
        artifact_id=artifact_alias,
        file_path="readme.txt",
        format="text",
        offset=7,  # Skip "Hello, "
        limit=4    # Read "this"
    )
    assert result["content"] == "this"

    # Test 4: Write file using staging
    await artifact_manager.edit(artifact_alias, stage=True)

    write_result = await artifact_manager.write_file(
        artifact_id=artifact_alias,
        file_path="new_file.txt",
        content="This is a new file created via write_file!",
        format="text"
    )
    assert write_result["success"] is True
    assert write_result["bytes_written"] == len("This is a new file created via write_file!")

    # Test 5: Read from staging before commit
    staged_result = await artifact_manager.read_file(
        artifact_id=artifact_alias,
        file_path="new_file.txt",
        format="text",
        stage=True
    )
    assert staged_result["content"] == "This is a new file created via write_file!"

    # Test 6: Write file with base64 format
    import base64
    binary_content = b"\x00\x01\x02\x03binary data"
    encoded_content = base64.b64encode(binary_content).decode("ascii")

    write_result = await artifact_manager.write_file(
        artifact_id=artifact_alias,
        file_path="binary_file.bin",
        content=encoded_content,
        format="base64"
    )
    assert write_result["success"] is True

    # Read it back and verify
    read_result = await artifact_manager.read_file(
        artifact_id=artifact_alias,
        file_path="binary_file.bin",
        format="base64",
        stage=True
    )
    assert base64.b64decode(read_result["content"]) == binary_content

    # Test 7: Commit and verify files are accessible
    await artifact_manager.commit(artifact_alias, comment="Added new files via write_file")

    # Read the committed file
    result = await artifact_manager.read_file(
        artifact_id=artifact_alias,
        file_path="new_file.txt",
        format="text"
    )
    assert result["content"] == "This is a new file created via write_file!"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_read_file_not_found(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test read_file returns proper error when file doesn't exist."""
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-read-notfound-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-notfound-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Not Found Test"},
        config={"storage": "git"},
    )

    workspace = api.config.workspace
    port = SIO_PORT
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, "repo")
        os.makedirs(repo_dir, exist_ok=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True, check=True)

        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True, check=True)

        with open(os.path.join(repo_dir, "exists.txt"), "w") as f:
            f.write("This file exists")

        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(["git", "push", auth_url, "main:main", "--force"], cwd=repo_dir, capture_output=True, check=True, timeout=30)

    # Try to read a file that doesn't exist
    try:
        await artifact_manager.read_file(
            artifact_id=artifact_alias,
            file_path="nonexistent.txt",
            format="text"
        )
        assert False, "Expected an error when reading nonexistent file"
    except Exception as e:
        # The error should mention "not found"
        assert "not found" in str(e).lower() or "404" in str(e)

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_write_file_requires_staging(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that write_file requires artifact to be in staging mode."""
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "write-staging-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"write-staging-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Write Staging Test"},
        config={"storage": "git"},
    )

    # Try to write without entering staging mode - should fail
    try:
        await artifact_manager.write_file(
            artifact_id=artifact_alias,
            file_path="test.txt",
            content="test content",
            format="text"
        )
        assert False, "Expected an error when writing without staging mode"
    except Exception as e:
        assert "staging" in str(e).lower()

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_storage_get_staged_file(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test downloading staged files from git-storage artifacts before commit.

    This tests the scenario where:
    1. A git-storage artifact is created
    2. The artifact is put into staging mode
    3. A file is uploaded to staging
    4. get_file is called with version='stage' to download the staged file

    Before the fix, this would fail with "Repository is empty (no commits)" because
    _get_git_file_url didn't handle version='stage' properly.
    """
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-staged-file-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    artifact_alias = f"git-staged-test-{uuid.uuid4().hex[:8]}"

    # Create a git-storage artifact
    artifact = await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Staged File Test"},
        config={"storage": "git"},
    )

    # Put artifact into staging mode
    await artifact_manager.edit(artifact_id=artifact_alias, stage=True)

    # Upload a file to staging using put_file
    file_content = "Hello from staging!"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact_alias,
        file_path="test.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok, f"Failed to upload file: {response.status_code}"

    # Verify the file appears in staged file list
    staged_files = await artifact_manager.list_files(
        artifact_id=artifact_alias,
        version="stage"
    )
    assert len(staged_files) >= 1, f"Expected at least 1 staged file, got {len(staged_files)}"
    file_names = [f["name"] for f in staged_files]
    assert "test.txt" in file_names, f"Expected 'test.txt' in staged files, got: {file_names}"

    # This is the key test: get_file with version='stage' should return a valid URL
    # Before the fix, this would fail with "Repository is empty"
    staged_file_url = await artifact_manager.get_file(
        artifact_id=artifact_alias,
        file_path="test.txt",
        version="stage"
    )

    # Download and verify the file content
    response = requests.get(staged_file_url)
    assert response.ok, f"Failed to download staged file: {response.status_code} - {response.text}"
    assert response.text == file_content, f"Expected '{file_content}', got '{response.text}'"

    # Now commit and verify we can still access the file
    await artifact_manager.commit(artifact_id=artifact_alias)

    # After commit, file should be accessible without version='stage'
    committed_file_url = await artifact_manager.get_file(
        artifact_id=artifact_alias,
        file_path="test.txt"
    )
    response = requests.get(committed_file_url)
    assert response.ok, f"Failed to download committed file: {response.status_code}"
    assert response.text == file_content

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_storage_get_staged_file_http_endpoint(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test the HTTP endpoint for staged files in git-storage artifacts.

    This tests the /files/{path} HTTP endpoint with stage=true parameter.
    """
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-staged-http-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    artifact_alias = f"git-staged-http-{uuid.uuid4().hex[:8]}"

    # Create a git-storage artifact
    await artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Staged HTTP Test"},
        config={"storage": "git"},
    )

    # Put artifact into staging mode
    await artifact_manager.edit(artifact_id=artifact_alias, stage=True)

    # Upload a file to staging
    file_content = "HTTP endpoint test content"
    put_url = await artifact_manager.put_file(
        artifact_id=artifact_alias,
        file_path="http_test.txt",
    )
    response = requests.put(put_url, data=file_content)
    assert response.ok, f"Failed to upload file: {response.status_code}"

    # Test the HTTP endpoint directly with stage=true
    http_url = f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/http_test.txt?stage=true"
    response = requests.get(
        http_url,
        headers={"Authorization": f"Bearer {test_user_token}"}
    )
    assert response.ok, f"HTTP endpoint failed: {response.status_code} - {response.text}"
    assert response.text == file_content

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias)
    await api.disconnect()


async def test_git_artifact_appears_in_list(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that git-storage artifacts appear in list() with default stage=False.

    This is a regression test for the bug where git-storage artifacts were not
    appearing in list() results because they lacked entries in the versions array.
    The list() method filters by json_array_length(versions) > 0 when stage=False,
    so git artifacts need a "git" version entry to be visible.

    Tests both:
    1. Git artifact created without staging (should have "git" version immediately)
    2. Git artifact created with staging then committed (should have "git" version after commit)
    """
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    api = await connect_to_server({
        "name": "git-list-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    # Test 1: Create EMPTY git artifact without staging - should appear in list immediately
    # This is the key regression test: empty git artifacts must be visible in list()
    artifact_alias_1 = f"git-list-test-1-{uuid.uuid4().hex[:8]}"
    result = await artifact_manager.create(
        alias=artifact_alias_1,
        manifest={"name": "Git List Test 1"},
        config={"storage": "git"},
    )

    # Verify the artifact has at least one version to be visible in list
    assert result.get("versions"), f"Git artifact should have versions array, got: {result}"
    assert len(result["versions"]) > 0, f"Git artifact should have at least one version to be visible in list"

    # Verify artifact is empty (no files pushed yet)
    assert result.get("file_count", 0) == 0, f"Newly created git artifact should be empty, got file_count: {result.get('file_count')}"

    # Verify EMPTY artifact appears in list with default stage=False
    children = await artifact_manager.list()
    child_aliases = [c.get("alias") for c in children]
    assert artifact_alias_1 in child_aliases, \
        f"Empty git artifact should appear in list() with default stage=False. Got aliases: {child_aliases}"

    # Test 2: Create git artifact with staging, commit, then check listing
    artifact_alias_2 = f"git-list-test-2-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=artifact_alias_2,
        manifest={"name": "Git List Test 2"},
        config={"storage": "git"},
        stage=True,  # Create in staging mode
    )

    # Before commit, artifact should NOT appear with stage=False
    children_before = await artifact_manager.list(stage=False)
    child_aliases_before = [c.get("alias") for c in children_before]
    assert artifact_alias_2 not in child_aliases_before, \
        f"Staged git artifact should NOT appear in list() with stage=False before commit"

    # Artifact should appear with stage=True
    children_staged = await artifact_manager.list(stage=True)
    child_aliases_staged = [c.get("alias") for c in children_staged]
    assert artifact_alias_2 in child_aliases_staged, \
        f"Staged git artifact should appear in list() with stage=True"

    # Push a file using git and commit
    port = SIO_PORT
    auth_url = f"http://git:{test_user_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias_2}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create a file
        with open(os.path.join(local_repo, "README.md"), "w") as f:
            f.write("# Test\n")

        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"

    # Commit the artifact
    commit_result = await artifact_manager.commit(artifact_id=artifact_alias_2)

    # Verify the committed artifact has "v0" version
    assert commit_result.get("versions"), f"Committed git artifact should have versions array"
    version_names_2 = [v.get("version") for v in commit_result["versions"]]
    assert "v0" in version_names_2, f"Committed git artifact should have 'v0' version, got: {version_names_2}"

    # After commit, artifact should appear with stage=False
    children_after = await artifact_manager.list(stage=False)
    child_aliases_after = [c.get("alias") for c in children_after]
    assert artifact_alias_2 in child_aliases_after, \
        f"Committed git artifact should appear in list() with stage=False. Got aliases: {child_aliases_after}"

    # Cleanup
    await artifact_manager.delete(artifact_id=artifact_alias_1)
    await artifact_manager.delete(artifact_id=artifact_alias_2)
    await api.disconnect()


async def test_git_child_token_preserves_parent_user(
    minio_server,
    fastapi_server,
    test_user_token,
):
    """Test that child tokens preserve parent user ID for git operations.

    When a child token is generated (via generate_token), the parent user ID
    should be preserved and accessible via get_effective_user_id(). This ensures
    git history and artifact tracking always reflects the actual user who
    generated the token chain, not the random child token ID.
    """
    import subprocess
    import uuid
    from hypha_rpc import connect_to_server

    # Connect with the main user token
    api = await connect_to_server({
        "name": "git-parent-test-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })

    artifact_manager = await api.get_service("public/artifact-manager")

    # Generate a child token using the API object's method
    child_token = await api.generate_token({
        "permission": "admin",
        "expires_in": 3600,
    })

    # Connect with the child token
    child_api = await connect_to_server({
        "name": "git-child-test-client",
        "server_url": WS_SERVER_URL,
        "token": child_token,
    })

    # Parse the child token to verify parent is set
    token_info = await child_api.parse_token(child_token)

    # Verify the child token has the parent user ID set
    assert token_info.get("parent") is not None, \
        f"Child token should have parent field set, got: {token_info}"
    assert token_info["parent"] == "user-1", \
        f"Parent should be 'user-1' (the test user), got: {token_info['parent']}"

    # Verify the child token has a different random ID
    assert token_info["id"] != "user-1", \
        f"Child token ID should be different from parent, got: {token_info['id']}"

    # Create a git artifact using the child token
    child_artifact_manager = await child_api.get_service("public/artifact-manager")
    artifact_alias = f"git-parent-test-{uuid.uuid4().hex[:8]}"
    artifact = await child_artifact_manager.create(
        alias=artifact_alias,
        manifest={"name": "Git Parent Test"},
        config={"storage": "git"},
    )

    # Verify the artifact was created with the child token's ID
    # (The actual attribution uses get_effective_user_id which returns parent)
    assert artifact is not None

    # Push content using the child token
    workspace = child_api.config.workspace
    port = SIO_PORT
    auth_url = f"http://git:{child_token}@127.0.0.1:{port}/{workspace}/git/{artifact_alias}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        subprocess.run(["git", "init", "-b", "main"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=local_repo, check=True, capture_output=True)

        # Create a file
        with open(os.path.join(local_repo, "README.md"), "w") as f:
            f.write("# Test with child token\n")

        subprocess.run(["git", "add", "."], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Child token commit"], cwd=local_repo, check=True, capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", auth_url], cwd=local_repo, check=True, capture_output=True)

        result = subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=local_repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Push with child token failed: {result.stderr}"

    # Cleanup
    await child_artifact_manager.delete(artifact_id=artifact_alias)
    await child_api.disconnect()
    await api.disconnect()
