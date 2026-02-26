"""Integration tests for per-branch worktree staging (overlay approach).

Tests cover:
- Branched staging format and migration from legacy format
- Per-branch edit/write/commit isolation
- Overlay reads (staging overrides committed git content)
- Concurrent multi-branch staging
- File removal in staging with overlay
- get_status and discard_changes operations
- Backward compatibility with existing workflows
"""

import pytest
import uuid
import os
import subprocess
import tempfile

from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SERVER_URL

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_run(args, cwd, check=True, timeout=30):
    """Run a git command and return the CompletedProcess."""
    return subprocess.run(
        args, cwd=cwd, check=check, capture_output=True, text=True, timeout=timeout,
    )


def _init_repo(tmpdir, auth_url):
    """Create a local git repo, configure it, and add 'origin' remote."""
    local_repo = os.path.join(tmpdir, "repo")
    os.makedirs(local_repo)
    _git_run(["git", "init", "-b", "main"], cwd=local_repo)
    _git_run(["git", "config", "user.email", "test@test.com"], cwd=local_repo)
    _git_run(["git", "config", "user.name", "Test"], cwd=local_repo)
    _git_run(["git", "remote", "add", "origin", auth_url], cwd=local_repo)
    return local_repo


def _write_commit_push(local_repo, files, message, branch="main", push=True):
    """Write *files* (dict path->content), commit, and optionally push."""
    for path, content in files.items():
        full = os.path.join(local_repo, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    _git_run(["git", "add", "."], cwd=local_repo)
    _git_run(["git", "commit", "-m", message], cwd=local_repo)
    if push:
        result = _git_run(
            ["git", "push", "-u", "origin", branch], cwd=local_repo, check=False,
        )
        assert result.returncode == 0, f"Push failed: {result.stderr}"


async def _setup_git_artifact(artifact_manager, workspace, test_user_token):
    """Create a git-storage artifact with initial files on main.

    Returns (alias, auth_url).
    """
    alias = f"git-staging-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=alias,
        manifest={"name": f"Staging Test {alias}"},
        config={"storage": "git"},
    )

    git_url = f"{SERVER_URL}/{workspace}/git/{alias}"
    auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = _init_repo(tmpdir, auth_url)
        _write_commit_push(
            local_repo,
            {
                "README.md": "# Test Project\n",
                "src/main.py": "print('hello')\n",
                "src/utils.py": "def helper(): pass\n",
            },
            "Initial commit",
        )

    return alias, auth_url


# ---------------------------------------------------------------------------
# Tests: Basic Per-Branch Staging
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_creates_branched_staging(minio_server, fastapi_server, test_user_token):
    """edit(stage=True, branch='feature') creates per-branch staging structure."""
    api = await connect_to_server({
        "name": "test-branched-staging-edit",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Create feature branch
    await artifact_manager.create_branch(alias, branch="feature", source="main")

    # Edit on feature branch
    await artifact_manager.edit(alias, stage=True, branch="feature")

    # Check staging status
    status = await artifact_manager.get_status(alias)
    assert status["staged"] is True
    assert status["active_branch"] == "feature"
    assert "feature" in status["branches"]

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_write_and_commit_on_branch(minio_server, fastapi_server, test_user_token):
    """Write files on a feature branch via RPC, commit, and verify isolation."""
    api = await connect_to_server({
        "name": "test-write-commit-branch",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Create feature branch
    await artifact_manager.create_branch(alias, branch="feature", source="main")

    # Edit, write, commit on feature
    await artifact_manager.edit(alias, stage=True, branch="feature")
    await artifact_manager.write_file(alias, "new_file.py", "print('new')")
    await artifact_manager.commit(alias, comment="Add new file", branch="feature")

    # File should exist on feature
    result = await artifact_manager.read_file(alias, "new_file.py", version="feature")
    assert result["content"] == "print('new')"

    # File should NOT exist on main
    with pytest.raises(Exception):
        await artifact_manager.read_file(alias, "new_file.py", version="main")

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


# ---------------------------------------------------------------------------
# Tests: Overlay Reads
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_overlay_read_staged_file(minio_server, fastapi_server, test_user_token):
    """Reading with stage=True returns staged file, not committed version."""
    api = await connect_to_server({
        "name": "test-overlay-read-staged",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Stage a modified version of an existing file on main
    await artifact_manager.edit(alias, stage=True, branch="main")
    await artifact_manager.write_file(alias, "README.md", "# Modified\n")

    # Overlay read: should return staged content
    result = await artifact_manager.read_file(alias, "README.md", stage=True, branch="main")
    assert result["content"] == "# Modified\n"

    # Non-staged read: should return committed content
    result = await artifact_manager.read_file(alias, "README.md", version="main")
    assert result["content"] == "# Test Project\n"

    # Cleanup
    await artifact_manager.discard_changes(alias)
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_overlay_read_fallthrough_to_git(minio_server, fastapi_server, test_user_token):
    """Reading a file not in staging falls through to committed git content."""
    api = await connect_to_server({
        "name": "test-overlay-fallthrough",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Stage on main (only stage one new file)
    await artifact_manager.edit(alias, stage=True, branch="main")
    await artifact_manager.write_file(alias, "new_file.txt", "new content")

    # Read file NOT in staging — should fall through to git
    result = await artifact_manager.read_file(alias, "src/main.py", stage=True, branch="main")
    assert result["content"] == "print('hello')\n"

    # Read staged file — should return staged content
    result = await artifact_manager.read_file(alias, "new_file.txt", stage=True, branch="main")
    assert result["content"] == "new content"

    # Cleanup
    await artifact_manager.discard_changes(alias)
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_overlay_list_files_merges_staging_and_git(minio_server, fastapi_server, test_user_token):
    """list_files(stage=True) merges staged files on top of git tree."""
    api = await connect_to_server({
        "name": "test-overlay-list-merge",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Stage a new file on main
    await artifact_manager.edit(alias, stage=True, branch="main")
    await artifact_manager.write_file(alias, "extra.txt", "extra content")

    # List with overlay — should show both committed and staged files
    items = await artifact_manager.list_files(alias, stage=True, branch="main")
    names = {item["name"] for item in items}
    assert "README.md" in names  # from git
    assert "extra.txt" in names  # from staging

    # Cleanup
    await artifact_manager.discard_changes(alias)
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_overlay_read_removed_file_404(minio_server, fastapi_server, test_user_token):
    """Reading a file marked for removal in staging returns 404 even if it exists in git."""
    api = await connect_to_server({
        "name": "test-overlay-removed-404",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Stage removal of an existing file
    await artifact_manager.edit(alias, stage=True, branch="main")
    await artifact_manager.remove_file(alias, "src/utils.py", branch="main")

    # Overlay read of removed file should fail
    with pytest.raises(Exception):
        await artifact_manager.read_file(alias, "src/utils.py", stage=True, branch="main")

    # Direct read from git should still work
    result = await artifact_manager.read_file(alias, "src/utils.py", version="main")
    assert "helper" in result["content"]

    # Cleanup
    await artifact_manager.discard_changes(alias)
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


# ---------------------------------------------------------------------------
# Tests: Concurrent Multi-Branch Staging
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_branch_staging_isolation(minio_server, fastapi_server, test_user_token):
    """Two branches can be staged concurrently without interference."""
    api = await connect_to_server({
        "name": "test-concurrent-staging",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Create two branches
    await artifact_manager.create_branch(alias, branch="feature-a", source="main")
    await artifact_manager.create_branch(alias, branch="feature-b", source="main")

    # Stage files on both branches
    await artifact_manager.edit(alias, stage=True, branch="feature-a")
    await artifact_manager.write_file(alias, "a.txt", "from branch a", branch="feature-a")

    await artifact_manager.edit(alias, stage=True, branch="feature-b")
    await artifact_manager.write_file(alias, "b.txt", "from branch b", branch="feature-b")

    # Check status — both branches should have staging
    status = await artifact_manager.get_status(alias)
    assert "feature-a" in status["branches"]
    assert "feature-b" in status["branches"]

    # Commit feature-a — should NOT affect feature-b
    await artifact_manager.commit(alias, comment="Add a.txt", branch="feature-a")

    # feature-a should be committed
    result = await artifact_manager.read_file(alias, "a.txt", version="feature-a")
    assert result["content"] == "from branch a"

    # feature-b staging should still exist
    status = await artifact_manager.get_status(alias)
    assert "feature-a" not in status.get("branches", {})
    assert "feature-b" in status["branches"]

    # Commit feature-b
    await artifact_manager.commit(alias, comment="Add b.txt", branch="feature-b")
    result = await artifact_manager.read_file(alias, "b.txt", version="feature-b")
    assert result["content"] == "from branch b"

    # No more staging
    status = await artifact_manager.get_status(alias)
    assert status["staged"] is False

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_commit_one_branch_preserves_other(minio_server, fastapi_server, test_user_token):
    """Committing branch A does not lose staging for branch B."""
    api = await connect_to_server({
        "name": "test-commit-preserves-other",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    await artifact_manager.create_branch(alias, branch="feat-x", source="main")
    await artifact_manager.create_branch(alias, branch="feat-y", source="main")

    # Stage on both
    await artifact_manager.edit(alias, stage=True, branch="feat-x")
    await artifact_manager.write_file(alias, "x.txt", "x content", branch="feat-x")

    await artifact_manager.edit(alias, stage=True, branch="feat-y")
    await artifact_manager.write_file(alias, "y.txt", "y content", branch="feat-y")

    # Commit feat-x
    await artifact_manager.commit(alias, comment="x", branch="feat-x")

    # feat-y should still be readable from staging
    result = await artifact_manager.read_file(alias, "y.txt", stage=True, branch="feat-y")
    assert result["content"] == "y content"

    # Now commit feat-y
    await artifact_manager.commit(alias, comment="y", branch="feat-y")
    result = await artifact_manager.read_file(alias, "y.txt", version="feat-y")
    assert result["content"] == "y content"

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


# ---------------------------------------------------------------------------
# Tests: get_status and discard_changes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_status_empty(minio_server, fastapi_server, test_user_token):
    """get_status returns staged=False when nothing is staged."""
    api = await connect_to_server({
        "name": "test-status-empty",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    status = await artifact_manager.get_status(alias)
    assert status["staged"] is False

    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_get_status_with_staged_files(minio_server, fastapi_server, test_user_token):
    """get_status shows files added and removed for a branch."""
    api = await connect_to_server({
        "name": "test-status-files",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    await artifact_manager.edit(alias, stage=True, branch="main")
    await artifact_manager.write_file(alias, "new.txt", "new")
    await artifact_manager.remove_file(alias, "src/utils.py", branch="main")

    status = await artifact_manager.get_status(alias, branch="main")
    assert status["staged"] is True
    assert "new.txt" in status["files_added"]
    assert "src/utils.py" in status["files_removed"]

    # Cleanup
    await artifact_manager.discard_changes(alias)
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_discard_specific_branch(minio_server, fastapi_server, test_user_token):
    """discard_changes(branch='x') only discards branch x, preserves branch y."""
    api = await connect_to_server({
        "name": "test-discard-branch",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    await artifact_manager.create_branch(alias, branch="feat-a", source="main")
    await artifact_manager.create_branch(alias, branch="feat-b", source="main")

    await artifact_manager.edit(alias, stage=True, branch="feat-a")
    await artifact_manager.write_file(alias, "a.txt", "a", branch="feat-a")

    await artifact_manager.edit(alias, stage=True, branch="feat-b")
    await artifact_manager.write_file(alias, "b.txt", "b", branch="feat-b")

    # Discard only feat-a
    await artifact_manager.discard_changes(alias, branch="feat-a")

    status = await artifact_manager.get_status(alias)
    assert "feat-a" not in status.get("branches", {})
    assert "feat-b" in status["branches"]

    # Cleanup
    await artifact_manager.discard_changes(alias)
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_discard_all(minio_server, fastapi_server, test_user_token):
    """discard_changes() without branch discards all staged changes."""
    api = await connect_to_server({
        "name": "test-discard-all",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    await artifact_manager.edit(alias, stage=True, branch="main")
    await artifact_manager.write_file(alias, "temp.txt", "temp")

    await artifact_manager.discard_changes(alias)

    status = await artifact_manager.get_status(alias)
    assert status["staged"] is False

    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


# ---------------------------------------------------------------------------
# Tests: Backward Compatibility
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_branch_staging(minio_server, fastapi_server, test_user_token):
    """edit(stage=True) without branch parameter uses default branch."""
    api = await connect_to_server({
        "name": "test-default-branch",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Edit without branch — should use "main"
    await artifact_manager.edit(alias, stage=True)
    await artifact_manager.write_file(alias, "file.txt", "content")
    await artifact_manager.commit(alias, comment="default branch commit")

    # File should exist on main
    result = await artifact_manager.read_file(alias, "file.txt", version="main")
    assert result["content"] == "content"

    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_overlay_on_new_branch_with_no_commits(minio_server, fastapi_server, test_user_token):
    """Overlay read on a branch that has no commits yet (newly created)."""
    api = await connect_to_server({
        "name": "test-overlay-new-branch",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Create a new branch
    await artifact_manager.create_branch(alias, branch="new-feat", source="main")

    # Stage files on the new branch
    await artifact_manager.edit(alias, stage=True, branch="new-feat")
    await artifact_manager.write_file(alias, "new.py", "# new feature", branch="new-feat")

    # Overlay read — staged file should be readable
    result = await artifact_manager.read_file(alias, "new.py", stage=True, branch="new-feat")
    assert result["content"] == "# new feature"

    # Overlay read of committed file from parent branch should work
    result = await artifact_manager.read_file(alias, "README.md", stage=True, branch="new-feat")
    assert "Test Project" in result["content"]

    # Commit and verify
    await artifact_manager.commit(alias, comment="new feature", branch="new-feat")
    result = await artifact_manager.read_file(alias, "new.py", version="new-feat")
    assert result["content"] == "# new feature"

    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_staging_overrides_committed_file(minio_server, fastapi_server, test_user_token):
    """Staged version of a file overrides the committed version in overlay reads."""
    api = await connect_to_server({
        "name": "test-staging-overrides",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    # Stage a modified version of README.md
    await artifact_manager.edit(alias, stage=True, branch="main")
    await artifact_manager.write_file(alias, "README.md", "# Updated README\n")

    # Overlay: staged version should win
    result = await artifact_manager.read_file(alias, "README.md", stage=True, branch="main")
    assert result["content"] == "# Updated README\n"

    # Non-overlay: committed version unchanged
    result = await artifact_manager.read_file(alias, "README.md", version="main")
    assert result["content"] == "# Test Project\n"

    # Commit and verify both views now show updated version
    await artifact_manager.commit(alias, comment="update readme", branch="main")
    result = await artifact_manager.read_file(alias, "README.md", version="main")
    assert result["content"] == "# Updated README\n"

    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_removal_and_add_on_same_branch(minio_server, fastapi_server, test_user_token):
    """Remove one file and add another on the same branch, then commit."""
    api = await connect_to_server({
        "name": "test-remove-and-add",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(artifact_manager, workspace, test_user_token)

    await artifact_manager.edit(alias, stage=True, branch="main")
    await artifact_manager.write_file(alias, "new_module.py", "# module")
    await artifact_manager.remove_file(alias, "src/utils.py", branch="main")
    await artifact_manager.commit(alias, comment="swap files", branch="main")

    # new_module.py should exist
    result = await artifact_manager.read_file(alias, "new_module.py", version="main")
    assert result["content"] == "# module"

    # src/utils.py should be gone
    with pytest.raises(Exception):
        await artifact_manager.read_file(alias, "src/utils.py", version="main")

    # Other files should still exist
    result = await artifact_manager.read_file(alias, "README.md", version="main")
    assert "Test Project" in result["content"]

    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()
