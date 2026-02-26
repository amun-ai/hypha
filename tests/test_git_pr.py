"""Integration tests for the git Pull Request system.

Tests cover branch management, diffs, pull request CRUD,
merge strategies (fast-forward, three-way, conflict), reviews,
and the full PR lifecycle.
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
    """Create a local git repo, configure it, and add 'origin' remote.

    Returns the local_repo path.
    """
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
    """Create a git-storage artifact and push an initial commit.

    Returns (artifact_alias, auth_url).
    """
    alias = f"git-pr-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=alias,
        manifest={"name": f"PR Test {alias}"},
        config={"storage": "git"},
    )

    git_url = f"{SERVER_URL}/{workspace}/git/{alias}"
    auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = _init_repo(tmpdir, auth_url)
        _write_commit_push(
            local_repo,
            {"README.md": "# Test\n"},
            "Initial commit",
        )

    return alias, auth_url


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_and_list_branches(minio_server, fastapi_server, test_user_token):
    """Create branches, list them, and verify duplicate creation fails."""
    api = await connect_to_server({
        "name": "test-create-list-branches",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    # Create two feature branches from main
    result_a = await artifact_manager.create_branch(alias, branch="feature-a", source="main")
    assert result_a["branch"] == "feature-a"
    assert len(result_a["sha"]) == 40

    result_b = await artifact_manager.create_branch(alias, branch="feature-b", source="main")
    assert result_b["branch"] == "feature-b"

    # List branches — expect main + 2 features
    branches = await artifact_manager.list_branches(alias)
    branch_names = {b["name"] for b in branches}
    assert branch_names == {"main", "feature-a", "feature-b"}

    # Verify main is default
    main_entry = [b for b in branches if b["name"] == "main"][0]
    assert main_entry.get("default") is True

    # Duplicate creation should fail
    with pytest.raises(Exception, match="already exists"):
        await artifact_manager.create_branch(alias, branch="feature-a", source="main")

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_delete_branch(minio_server, fastapi_server, test_user_token):
    """Delete a branch, verify protected default branch cannot be deleted."""
    api = await connect_to_server({
        "name": "test-delete-branch",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    await artifact_manager.create_branch(alias, branch="to-delete", source="main")

    # Delete the branch
    result = await artifact_manager.delete_branch(alias, branch="to-delete")
    assert result["status"] == "deleted"
    assert result["branch"] == "to-delete"

    # Verify it is gone
    branches = await artifact_manager.list_branches(alias)
    branch_names = {b["name"] for b in branches}
    assert "to-delete" not in branch_names

    # Deleting default branch should fail
    with pytest.raises(Exception, match="default branch"):
        await artifact_manager.delete_branch(alias, branch="main")

    # Deleting nonexistent branch should fail
    with pytest.raises(Exception, match="does not exist"):
        await artifact_manager.delete_branch(alias, branch="nonexistent")

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_write_to_branch_and_commit(minio_server, fastapi_server, test_user_token):
    """Write a file on a feature branch via RPC, commit, and verify isolation."""
    api = await connect_to_server({
        "name": "test-write-branch-commit",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    # Create feature branch
    await artifact_manager.create_branch(alias, branch="feature", source="main")

    # Edit in staging mode targeting the feature branch
    await artifact_manager.edit(alias, stage=True, branch="feature")

    # Write a file
    write_result = await artifact_manager.write_file(
        alias, "new_file.py", "print('hello')",
    )
    assert write_result["success"] is True

    # Commit to the feature branch
    commit_result = await artifact_manager.commit(
        alias, comment="Add new file", branch="feature",
    )
    assert commit_result is not None

    # Read file from feature branch — should exist
    file_data = await artifact_manager.read_file(
        alias, "new_file.py", version="feature",
    )
    assert file_data["content"] == "print('hello')"

    # Read file from main branch — should NOT exist
    with pytest.raises(Exception):
        await artifact_manager.read_file(alias, "new_file.py", version="main")

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_get_diff(minio_server, fastapi_server, test_user_token):
    """Compute diff between branches and verify stats, content, and filtering."""
    api = await connect_to_server({
        "name": "test-get-diff",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias = f"git-pr-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=alias,
        manifest={"name": f"Diff Test {alias}"},
        config={"storage": "git"},
    )

    git_url = f"{SERVER_URL}/{workspace}/git/{alias}"
    auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")

    # Push initial commit with multiple files
    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = _init_repo(tmpdir, auth_url)
        _write_commit_push(
            local_repo,
            {
                "README.md": "# Test Project\n",
                "existing.txt": "original content\n",
                "to_delete.txt": "this will be removed\n",
            },
            "Initial commit",
        )

    # Create a feature branch
    await artifact_manager.create_branch(alias, branch="feature", source="main")

    # Make changes on the feature branch via RPC
    await artifact_manager.edit(alias, stage=True, branch="feature")
    # Add a new file
    await artifact_manager.write_file(alias, "added.py", "print('new')\n")
    # Modify an existing file
    await artifact_manager.write_file(alias, "existing.txt", "modified content\n")
    # Delete a file
    await artifact_manager.remove_file(alias, "to_delete.txt")
    # Commit
    await artifact_manager.commit(alias, comment="Feature changes", branch="feature")

    # Get diff without content
    diff = await artifact_manager.get_diff(alias, base="main", head="feature")
    assert "files" in diff
    assert "stats" in diff

    # Verify stats
    stats = diff["stats"]
    # We should have additions (added.py), modifications (existing.txt), deletions (to_delete.txt)
    assert stats.get("additions", 0) >= 1, f"Expected at least 1 addition, stats={stats}"
    assert stats.get("modifications", 0) >= 1, f"Expected at least 1 modification, stats={stats}"
    assert stats.get("deletions", 0) >= 1, f"Expected at least 1 deletion, stats={stats}"

    # Get diff WITH content
    diff_content = await artifact_manager.get_diff(
        alias, base="main", head="feature", include_content=True,
    )
    # At least one file entry should have old_content or new_content
    has_content = any(
        f.get("old_content") is not None or f.get("new_content") is not None
        for f in diff_content["files"]
    )
    assert has_content, "Expected file content in diff when include_content=True"

    # Get diff with path_filter
    diff_filtered = await artifact_manager.get_diff(
        alias, base="main", head="feature", path_filter="added",
    )
    filtered_paths = [f["path"] for f in diff_filtered["files"]]
    # Only added.py should match the "added" filter
    assert any("added" in p for p in filtered_paths), f"Expected 'added.py' in filtered paths, got {filtered_paths}"
    # to_delete.txt and existing.txt should not be present
    for p in filtered_paths:
        assert "to_delete" not in p, f"Unexpected file in filtered diff: {p}"

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_create_and_list_prs(minio_server, fastapi_server, test_user_token):
    """Create PRs, list them with and without status filter."""
    api = await connect_to_server({
        "name": "test-create-list-prs",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    # Create a feature branch with a change
    await artifact_manager.create_branch(alias, branch="feature", source="main")
    await artifact_manager.edit(alias, stage=True, branch="feature")
    await artifact_manager.write_file(alias, "feature.py", "# feature code\n")
    await artifact_manager.commit(alias, comment="Add feature code", branch="feature")

    # Create first PR
    pr1 = await artifact_manager.create_pr(
        alias,
        title="Add feature",
        source_branch="feature",
        target_branch="main",
        description="This adds the feature module.",
    )
    assert pr1["pr_number"] == 1
    assert pr1["status"] == "open"
    assert pr1["title"] == "Add feature"
    assert pr1["source_branch"] == "feature"
    assert pr1["target_branch"] == "main"
    assert pr1["description"] == "This adds the feature module."
    assert pr1["created_by"] is not None
    assert pr1["created_at"] is not None

    # Create another feature branch and PR
    await artifact_manager.create_branch(alias, branch="feature-2", source="main")
    await artifact_manager.edit(alias, stage=True, branch="feature-2")
    await artifact_manager.write_file(alias, "feature2.py", "# feature 2\n")
    await artifact_manager.commit(alias, comment="Add feature 2", branch="feature-2")

    pr2 = await artifact_manager.create_pr(
        alias,
        title="Add feature 2",
        source_branch="feature-2",
        target_branch="main",
    )
    assert pr2["pr_number"] == 2

    # List all PRs — newest first
    all_prs = await artifact_manager.list_prs(alias)
    assert len(all_prs) == 2
    assert all_prs[0]["pr_number"] == 2  # newest first
    assert all_prs[1]["pr_number"] == 1

    # Filter by open
    open_prs = await artifact_manager.list_prs(alias, status="open")
    assert len(open_prs) == 2

    # Filter by merged — should be empty
    merged_prs = await artifact_manager.list_prs(alias, status="merged")
    assert len(merged_prs) == 0

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_update_and_close_pr(minio_server, fastapi_server, test_user_token):
    """Update PR fields, close it, and verify closed PR cannot be updated."""
    api = await connect_to_server({
        "name": "test-update-close-pr",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    # Create branch + PR
    await artifact_manager.create_branch(alias, branch="feature", source="main")
    await artifact_manager.edit(alias, stage=True, branch="feature")
    await artifact_manager.write_file(alias, "f.txt", "content")
    await artifact_manager.commit(alias, comment="commit", branch="feature")

    pr = await artifact_manager.create_pr(
        alias, title="Original title", source_branch="feature", target_branch="main",
    )
    assert pr["pr_number"] == 1

    # Update title
    updated = await artifact_manager.update_pr(
        alias, pr_number=1, title="Updated title",
    )
    assert updated["title"] == "Updated title"

    # Update description
    updated2 = await artifact_manager.update_pr(
        alias, pr_number=1, description="New description",
    )
    assert updated2["description"] == "New description"
    # Title should be preserved
    assert updated2["title"] == "Updated title"

    # Close PR
    closed = await artifact_manager.close_pr(
        alias, pr_number=1, comment="Superseded",
    )
    assert closed["status"] == "closed"
    assert closed["closed_at"] is not None

    # Updating a closed PR should fail
    with pytest.raises(Exception, match="Cannot update"):
        await artifact_manager.update_pr(alias, pr_number=1, title="Should fail")

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_merge_pr_fast_forward(minio_server, fastapi_server, test_user_token):
    """Merge PR via fast-forward when main has no new commits."""
    api = await connect_to_server({
        "name": "test-merge-ff",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    # Create feature branch and add a file (no new commits on main)
    await artifact_manager.create_branch(alias, branch="feature", source="main")
    await artifact_manager.edit(alias, stage=True, branch="feature")
    await artifact_manager.write_file(alias, "feature.py", "# fast forward\n")
    await artifact_manager.commit(alias, comment="Feature work", branch="feature")

    # Create PR
    pr = await artifact_manager.create_pr(
        alias, title="FF merge", source_branch="feature", target_branch="main",
    )

    # Merge — should be fast-forward since main hasn't diverged
    merge_result = await artifact_manager.merge_pr(alias, pr_number=1)
    assert merge_result["status"] == "merged"
    assert merge_result["merge_type"] == "fast-forward"
    assert merge_result["merge_sha"] is not None

    # The feature file should now be readable from main
    file_data = await artifact_manager.read_file(alias, "feature.py", version="main")
    assert file_data["content"] == "# fast forward\n"

    # PR status should be merged
    pr_after = await artifact_manager.get_pr(alias, pr_number=1)
    assert pr_after["status"] == "merged"
    assert pr_after["merged_at"] is not None

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_merge_pr_three_way(minio_server, fastapi_server, test_user_token):
    """Merge PR via three-way merge when both branches have diverged."""
    api = await connect_to_server({
        "name": "test-merge-3way",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    # Create feature branch
    await artifact_manager.create_branch(alias, branch="feature", source="main")

    # Add file on feature branch
    await artifact_manager.edit(alias, stage=True, branch="feature")
    await artifact_manager.write_file(alias, "feature_only.py", "# feature\n")
    await artifact_manager.commit(alias, comment="Feature file", branch="feature")

    # Add a DIFFERENT file on main (via subprocess push to ensure divergence)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        _git_run(["git", "clone", auth_url, local_repo], cwd=tmpdir)
        _git_run(["git", "config", "user.email", "test@test.com"], cwd=local_repo)
        _git_run(["git", "config", "user.name", "Test"], cwd=local_repo)
        _write_commit_push(
            local_repo,
            {"main_only.py": "# main\n"},
            "Main file",
            branch="main",
        )

    # Create PR
    pr = await artifact_manager.create_pr(
        alias, title="Three-way merge", source_branch="feature", target_branch="main",
    )

    # Merge
    merge_result = await artifact_manager.merge_pr(alias, pr_number=1)
    assert merge_result["status"] == "merged"
    assert merge_result["merge_type"] == "merge"

    # Both files should be on main after merge
    f1 = await artifact_manager.read_file(alias, "feature_only.py", version="main")
    assert f1["content"] == "# feature\n"
    f2 = await artifact_manager.read_file(alias, "main_only.py", version="main")
    assert f2["content"] == "# main\n"

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_merge_pr_conflict(minio_server, fastapi_server, test_user_token):
    """Merge should report conflict when same file is modified on both branches."""
    api = await connect_to_server({
        "name": "test-merge-conflict",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    # Create artifact with initial shared.txt
    alias = f"git-pr-test-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=alias,
        manifest={"name": f"Conflict Test {alias}"},
        config={"storage": "git"},
    )

    git_url = f"{SERVER_URL}/{workspace}/git/{alias}"
    auth_url = git_url.replace("http://", f"http://git:{test_user_token}@")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = _init_repo(tmpdir, auth_url)
        _write_commit_push(
            local_repo,
            {"shared.txt": "original line\n"},
            "Initial commit",
        )

    # Create feature branch
    await artifact_manager.create_branch(alias, branch="feature", source="main")

    # Modify shared.txt on feature
    await artifact_manager.edit(alias, stage=True, branch="feature")
    await artifact_manager.write_file(alias, "shared.txt", "feature version of the line\n")
    await artifact_manager.commit(alias, comment="Feature change", branch="feature")

    # Modify shared.txt on main (different content)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_repo = os.path.join(tmpdir, "repo")
        os.makedirs(local_repo)
        _git_run(["git", "clone", auth_url, local_repo], cwd=tmpdir)
        _git_run(["git", "config", "user.email", "test@test.com"], cwd=local_repo)
        _git_run(["git", "config", "user.name", "Test"], cwd=local_repo)
        _write_commit_push(
            local_repo,
            {"shared.txt": "main version of the line\n"},
            "Main change",
            branch="main",
        )

    # Create PR
    pr = await artifact_manager.create_pr(
        alias, title="Conflicting merge", source_branch="feature", target_branch="main",
    )

    # Attempt merge — should report conflict
    merge_result = await artifact_manager.merge_pr(alias, pr_number=1)
    assert merge_result["status"] == "conflict"
    assert "conflicts" in merge_result
    assert len(merge_result["conflicts"]) > 0

    # PR should still be open (not merged)
    pr_after = await artifact_manager.get_pr(alias, pr_number=1)
    assert pr_after["status"] == "open"

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_submit_and_list_reviews(minio_server, fastapi_server, test_user_token):
    """Submit reviews on a PR and list them."""
    api = await connect_to_server({
        "name": "test-reviews",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    # Create branch + PR
    await artifact_manager.create_branch(alias, branch="feature", source="main")
    await artifact_manager.edit(alias, stage=True, branch="feature")
    await artifact_manager.write_file(alias, "review_me.py", "# code\n")
    await artifact_manager.commit(alias, comment="code", branch="feature")

    await artifact_manager.create_pr(
        alias, title="Review me", source_branch="feature", target_branch="main",
    )

    # Submit two reviews
    r1 = await artifact_manager.submit_review(
        alias, pr_number=1, verdict="comment", comment="Looks interesting",
    )
    assert r1["verdict"] == "comment"
    assert r1["comment"] == "Looks interesting"
    assert r1["review_id"] is not None
    assert r1["reviewer"] is not None
    assert r1["created_at"] is not None

    r2 = await artifact_manager.submit_review(
        alias, pr_number=1, verdict="approve", comment="LGTM",
    )
    assert r2["verdict"] == "approve"

    # List reviews
    reviews = await artifact_manager.list_reviews(alias, pr_number=1)
    assert len(reviews) == 2
    # Ordered by created_at
    assert reviews[0]["verdict"] == "comment"
    assert reviews[1]["verdict"] == "approve"

    # Verify all expected fields
    for review in reviews:
        assert "review_id" in review
        assert "reviewer" in review
        assert "verdict" in review
        assert "comment" in review
        assert "created_at" in review

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()


@pytest.mark.asyncio
async def test_full_pr_lifecycle(minio_server, fastapi_server, test_user_token):
    """End-to-end lifecycle: branch, write, commit, PR, diff, review, merge."""
    api = await connect_to_server({
        "name": "test-full-lifecycle",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace

    # 1. Create git artifact and push initial commit
    alias, auth_url = await _setup_git_artifact(
        artifact_manager, workspace, test_user_token,
    )

    # 2. Create branch "feature-auth"
    branch_result = await artifact_manager.create_branch(
        alias, branch="feature-auth", source="main",
    )
    assert branch_result["branch"] == "feature-auth"

    # 3. Write files to feature-auth branch
    await artifact_manager.edit(alias, stage=True, branch="feature-auth")
    await artifact_manager.write_file(alias, "auth.py", "def login(): pass\n")
    await artifact_manager.write_file(alias, "auth_test.py", "def test_login(): pass\n")

    # 4. Commit to feature-auth
    await artifact_manager.commit(
        alias, comment="Add auth module", branch="feature-auth",
    )

    # Verify files exist on feature-auth
    auth_file = await artifact_manager.read_file(
        alias, "auth.py", version="feature-auth",
    )
    assert auth_file["content"] == "def login(): pass\n"

    # 5. Create PR
    pr = await artifact_manager.create_pr(
        alias,
        title="Add authentication module",
        source_branch="feature-auth",
        target_branch="main",
        description="Adds login functionality and tests.",
    )
    assert pr["pr_number"] == 1
    assert pr["status"] == "open"

    # 6. Get diff
    diff = await artifact_manager.get_diff(alias, base="main", head="feature-auth")
    assert len(diff["files"]) >= 2  # auth.py and auth_test.py
    diff_paths = [f["path"] for f in diff["files"]]
    assert "auth.py" in diff_paths
    assert "auth_test.py" in diff_paths

    # 7. Submit review (approve)
    review = await artifact_manager.submit_review(
        alias, pr_number=1, verdict="approve", comment="LGTM, good work!",
    )
    assert review["verdict"] == "approve"

    # 8. Merge PR (with delete_branch=True, which is the default)
    merge_result = await artifact_manager.merge_pr(
        alias, pr_number=1, delete_branch=True,
    )
    assert merge_result["status"] == "merged"
    assert merge_result["source_branch_deleted"] is True

    # 9. Verify main has the changes
    main_auth = await artifact_manager.read_file(alias, "auth.py", version="main")
    assert main_auth["content"] == "def login(): pass\n"

    main_test = await artifact_manager.read_file(alias, "auth_test.py", version="main")
    assert main_test["content"] == "def test_login(): pass\n"

    # 10. Verify feature-auth branch is deleted
    branches = await artifact_manager.list_branches(alias)
    branch_names = {b["name"] for b in branches}
    assert "feature-auth" not in branch_names

    # 11. PR status is merged
    pr_final = await artifact_manager.get_pr(alias, pr_number=1)
    assert pr_final["status"] == "merged"
    assert pr_final["merged_at"] is not None
    assert pr_final["merge_sha"] is not None

    # Cleanup
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()
