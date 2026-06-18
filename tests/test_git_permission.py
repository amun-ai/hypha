"""Regression tests for git smart-HTTP permission handling.

Two fixes (0.21.92):

1. UNDER-GRANT (Josh's bug): a user listed in an artifact's own `_permissions`
   could push via git only with their RAW login token. A token minted via
   generate_token() has a RANDOM ephemeral id (with `parent` = the human id),
   so `_check_permissions` (which matched only `user_info.id`) failed to match
   the `{"<human_id>": "*"}` permission key and the push was denied. The check
   now also matches the effective/parent id (UserInfo.get_effective_user_id()),
   so a generated/child token inherits the human's artifact permissions.

2. 404-masking: get_repo_callback caught PermissionError and re-raised it as
   KeyError -> the HTTP layer returned a misleading 404 "repository not found"
   for a genuine permission denial. PermissionError now propagates so the route
   returns 403.

Requires Docker (postgres) via fastapi_server + MinIO.
"""
import os
import subprocess
import tempfile
import uuid

import pytest
import requests
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SIO_PORT

pytestmark = pytest.mark.asyncio


def _git(args, cwd=None, timeout=30):
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


async def _make_git_artifact(artifact_manager, alias, manifest_name):
    await artifact_manager.create(
        alias=alias,
        manifest={"name": manifest_name},
        config={"storage": "git"},
    )


def _clone_init_commit(tmpdir, auth_url, filename, content):
    """Clone an (empty) git artifact, init if needed, create a local commit."""
    clone_dir = os.path.join(tmpdir, "repo-" + uuid.uuid4().hex[:6])
    result = _git(["git", "clone", auth_url, clone_dir])
    if result.returncode != 0:
        os.makedirs(clone_dir, exist_ok=True)
        _git(["git", "init", "-b", "main"], cwd=clone_dir)
        _git(["git", "remote", "add", "origin", auth_url], cwd=clone_dir)

    _git(["git", "config", "user.email", "test@example.com"], cwd=clone_dir)
    _git(["git", "config", "user.name", "Test User"], cwd=clone_dir)

    cur = _git(["git", "branch", "--show-current"], cwd=clone_dir).stdout.strip()
    if cur and cur != "main":
        _git(["git", "branch", "-m", cur, "main"], cwd=clone_dir)

    with open(os.path.join(clone_dir, filename), "w") as f:
        f.write(content)
    _git(["git", "add", filename], cwd=clone_dir)
    cm = _git(["git", "commit", "-m", "commit " + content], cwd=clone_dir)
    assert cm.returncode == 0, f"local commit failed: {cm.stderr}"
    return clone_dir


async def test_generated_token_inherits_artifact_permission_for_git_push(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """A generated/child token must inherit the human's artifact permissions.

    User A creates a git artifact and grants user B `*` on it (by B's stable id).
    B mints a token via generate_token() (random id, parent=B) and pushes via
    git. Before the fix this failed (id mismatch -> denied -> masked 404); it
    must now succeed.
    """
    api_a = await connect_to_server(
        {"name": "user-a", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    am_a = await api_a.get_service("public/artifact-manager")
    workspace = api_a.config.workspace

    alias = f"josh-{uuid.uuid4().hex[:8]}"
    await _make_git_artifact(am_a, alias, "Josh git push regression")

    api_b = await connect_to_server(
        {"name": "user-b", "server_url": WS_SERVER_URL, "token": test_user_token_2}
    )
    b_user_id = api_b.config.user["id"]
    assert b_user_id != api_a.config.user["id"]

    # Grant B full artifact-level permission by B's stable user id.
    art = await am_a.read(artifact_id=f"{workspace}/{alias}")
    config = dict(art.get("config") or {})
    perms = dict(config.get("permissions") or {})
    perms[b_user_id] = "*"
    config["permissions"] = perms
    await am_a.edit(artifact_id=f"{workspace}/{alias}", config=config)

    # B mints a generated (child) token — random id, parent=B — the realistic
    # credential a user pastes as their git password.
    gen_token = await api_b.generate_token({"permission": "admin", "expires_in": 3600})
    from hypha.core.auth import parse_auth_token

    parsed = await parse_auth_token(gen_token)
    assert parsed.id != b_user_id, "expected a generated token with an ephemeral id"
    assert parsed.get_effective_user_id() == b_user_id, "parent must be the human id"

    port = SIO_PORT
    auth_url = f"http://git:{gen_token}@127.0.0.1:{port}/{workspace}/git/{alias}"
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = _clone_init_commit(tmpdir, auth_url, "B_GEN.md", "push via generated token\n")
        push = _git(["git", "push", auth_url, "main:main"], cwd=clone_dir)

    await am_a.delete(artifact_id=f"{workspace}/{alias}")
    await api_a.disconnect()
    await api_b.disconnect()

    assert push.returncode == 0, (
        f"generated/child token push was denied (regression): "
        f"rc={push.returncode} stderr={push.stderr!r}"
    )


async def test_git_permission_denied_returns_403_not_404(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """A genuine permission denial must surface as 403, not a masked 404.

    User A creates a git artifact and grants B nothing. B authenticates (valid
    token) but has no permission on it. The receive-pack discovery endpoint must
    return 403 (permission denied), not 404 (repository not found).
    """
    api_a = await connect_to_server(
        {"name": "user-a-403", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    am_a = await api_a.get_service("public/artifact-manager")
    workspace = api_a.config.workspace
    alias = f"denied-{uuid.uuid4().hex[:8]}"
    await _make_git_artifact(am_a, alias, "403 regression")

    port = SIO_PORT
    # B authenticates (valid token, id=user-2) but has no permission on A's artifact.
    info_refs = (
        f"http://127.0.0.1:{port}/{workspace}/git/{alias}/info/refs"
        f"?service=git-receive-pack"
    )
    resp = requests.get(info_refs, auth=("git", test_user_token_2), timeout=15)

    await am_a.delete(artifact_id=f"{workspace}/{alias}")
    await api_a.disconnect()

    assert resp.status_code == 403, (
        f"permission denial should be 403, not masked: got {resp.status_code} "
        f"body={resp.text[:200]!r}"
    )
