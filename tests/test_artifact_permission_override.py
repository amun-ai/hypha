"""Per-artifact permission writes must win over the parent collection's entry (#1066).

`edit(config=...)` on a COMMITTED child used to fold the parent collection's
`config.permissions` over the caller's map, so any key the collection also
listed was silently reverted to the collection-level value while `edit()` still
reported success. These tests pin the write-time precedence and the rejection of
unknown permission codes.
"""

import pytest
from hypha_rpc import connect_to_server

from . import SERVER_URL

pytestmark = pytest.mark.asyncio

# An id that is not a real user: `config.permissions` keys are plain strings, so
# this is exactly what an administrator writes for a collaborator who has not
# connected yet (and what the issue reporter used).
REVIEWER = "github|999000111"
STRANGER = "github|999000222"


async def _make_collection_with_child(artifact_manager, reviewer_permission="rw+"):
    """A collection that grants REVIEWER something, plus one COMMITTED child."""
    collection = await artifact_manager.create(
        type="collection",
        manifest={"name": "Permission Override Collection"},
        config={"permissions": {REVIEWER: reviewer_permission, "*": "r"}},
    )
    child = await artifact_manager.create(
        type="model",
        parent_id=collection.id,
        manifest={"name": "perm-probe"},
        config={"permissions": {}},
        stage=True,
    )
    await artifact_manager.commit(artifact_id=child.id)
    return collection, child


async def _set_child_permission(artifact_manager, child_id, key, value):
    """Read-modify-write one permission key, the way a bulk script would."""
    config = (await artifact_manager.read(artifact_id=child_id))["config"]
    config.setdefault("permissions", {})[key] = value
    await artifact_manager.edit(artifact_id=child_id, config=config)
    return (await artifact_manager.read(artifact_id=child_id))["config"]["permissions"]


async def test_explicit_child_permission_overrides_collection_entry(
    minio_server, fastapi_server, test_user_token
):
    """The reported case: the collection's value must not clobber the write."""
    api = await connect_to_server(
        {"name": "perm-client", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    _, child = await _make_collection_with_child(artifact_manager)

    # REVIEWER is listed on the collection as "rw+" -- elevating them on this one
    # artifact must stick, not read back as the collection-level "rw+".
    permissions = await _set_child_permission(artifact_manager, child.id, REVIEWER, "*")
    assert permissions[REVIEWER] == "*"

    # An explicit operation list must survive too.
    ops = ["read", "get_file", "list", "delete"]
    permissions = await _set_child_permission(artifact_manager, child.id, REVIEWER, ops)
    assert permissions[REVIEWER] == ops

    # Restricting below the collection level must also stick.
    permissions = await _set_child_permission(artifact_manager, child.id, REVIEWER, "r")
    assert permissions[REVIEWER] == "r"

    await api.disconnect()


async def test_keys_absent_from_the_write_still_inherit_from_the_collection(
    minio_server, fastapi_server, test_user_token
):
    """Inheritance is preserved: the parent fills keys the caller did not mention."""
    api = await connect_to_server(
        {"name": "perm-inherit", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    _, child = await _make_collection_with_child(artifact_manager)

    # The caller writes an unrelated key and never mentions REVIEWER.
    permissions = await _set_child_permission(
        artifact_manager, child.id, STRANGER, "r"
    )
    assert permissions[STRANGER] == "r"
    assert permissions[REVIEWER] == "rw+", "collection entry should fill the gap"

    await api.disconnect()


async def test_controls_that_already_worked_still_work(
    minio_server, fastapi_server, test_user_token
):
    """The rows of the issue's table that were already correct must stay correct."""
    api = await connect_to_server(
        {"name": "perm-controls", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    collection, child = await _make_collection_with_child(artifact_manager)

    # A user absent from the collection map.
    permissions = await _set_child_permission(artifact_manager, child.id, STRANGER, "*")
    assert permissions[STRANGER] == "*"

    # A staged (uncommitted) child.
    staged = await artifact_manager.create(
        type="model",
        parent_id=collection.id,
        manifest={"name": "staged-probe"},
        config={"permissions": {REVIEWER: "*"}},
        stage=True,
    )
    staged_config = await artifact_manager.read(artifact_id=staged.id, version="stage")
    assert staged_config["config"]["permissions"][REVIEWER] == "*"

    await api.disconnect()


async def test_elevated_child_permission_actually_grants_the_operation(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """The stored value must take effect: `*` on one child grants an admin-only op."""
    api_owner = await connect_to_server(
        {"name": "perm-owner", "server_url": SERVER_URL, "token": test_user_token}
    )
    owner_manager = await api_owner.get_service("public/artifact-manager")

    api_user = await connect_to_server(
        {"name": "perm-user", "server_url": SERVER_URL, "token": test_user_token_2}
    )
    user_manager = await api_user.get_service("public/artifact-manager")
    user_id = api_user.config.user["id"]

    collection = await owner_manager.create(
        type="collection",
        manifest={"name": "Elevation Collection"},
        # "rw+" deliberately excludes reset_stats; only "*" grants it.
        config={"permissions": {user_id: "rw+", "*": "r"}},
    )
    children = []
    for name in ("elevated", "sibling"):
        child = await owner_manager.create(
            type="model",
            parent_id=collection.id,
            manifest={"name": name},
            config={"permissions": {}},
            stage=True,
        )
        await owner_manager.commit(artifact_id=child.id)
        children.append(child)
    elevated, sibling = children

    # The sibling is the baseline: collection-level "rw+" does not allow reset_stats.
    with pytest.raises(Exception, match=r".*permission.*"):
        await user_manager.reset_stats(artifact_id=sibling.id)

    # Elevate the user on one artifact only.
    permissions = await _set_child_permission(owner_manager, elevated.id, user_id, "*")
    assert permissions[user_id] == "*"

    # The elevation must be effective, and must not leak to the sibling.
    await user_manager.reset_stats(artifact_id=elevated.id)
    with pytest.raises(Exception, match=r".*permission.*"):
        await user_manager.reset_stats(artifact_id=sibling.id)

    await api_owner.disconnect()
    await api_user.disconnect()


async def test_unknown_permission_code_is_rejected(
    minio_server, fastapi_server, test_user_token
):
    """An unrecognised code grants nothing, so accepting it silently is a trap."""
    api = await connect_to_server(
        {"name": "perm-validate", "server_url": SERVER_URL, "token": test_user_token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    with pytest.raises(Exception, match=r".*admin.*"):
        await artifact_manager.create(
            type="collection",
            manifest={"name": "Invalid Code Collection"},
            config={"permissions": {REVIEWER: "admin"}},
        )

    collection, child = await _make_collection_with_child(artifact_manager)

    config = (await artifact_manager.read(artifact_id=child.id))["config"]
    config["permissions"][REVIEWER] = "admin"
    with pytest.raises(Exception, match=r".*admin.*"):
        await artifact_manager.edit(artifact_id=child.id, config=config)

    # An unknown operation name inside a list form is rejected too.
    config["permissions"][REVIEWER] = ["read", "teleport"]
    with pytest.raises(Exception, match=r".*teleport.*"):
        await artifact_manager.edit(artifact_id=child.id, config=config)

    # The staging path must reject it as well, or it lands on commit.
    config["permissions"][REVIEWER] = "admin"
    with pytest.raises(Exception, match=r".*admin.*"):
        await artifact_manager.edit(artifact_id=child.id, config=config, stage=True)

    # A valid code on the same key still works.
    config["permissions"][REVIEWER] = "rw"
    await artifact_manager.edit(artifact_id=child.id, config=config)
    stored = (await artifact_manager.read(artifact_id=child.id))["config"]["permissions"]
    assert stored[REVIEWER] == "rw"

    await api.disconnect()
