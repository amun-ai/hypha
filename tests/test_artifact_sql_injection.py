"""SQL-injection reproduction + regression for artifact-manager search/list filters.

The artifact-manager `search()`/`list()`/`list_children` are public
`@schema_method`s. Their `filters` (and `order_by`) arguments are user-supplied.
Before the fix, `_build_manifest_condition`, the `config.permissions` filter
branch, and the `order_by` manifest./config. path interpolated attacker-controlled
filter keys/values/order_by fields RAW into SQLAlchemy `text()` clauses, e.g.

    manifest->>'name' = 'published' OR '1'='1'

which lets any caller with read/list access break out of the quoted literal and
alter the WHERE clause (classic SQL injection). The reproduction below uses the
canonical `' OR '1'='1` boolean-injection payload: if the value is interpolated,
the condition becomes always-true and returns EVERY child even though no child's
manifest.name equals the payload; if the value is safely bound, it is treated as
a literal string and matches nothing.

Runs against BOTH the SQLite backend (json_extract path) and — via the shared
`fastapi_server` fixture in the sibling test module — the PostgreSQL backend
(manifest->> path). Both dialects share the same vulnerable code, so this file
targets the SQLite fixture for fast iteration; `test_artifact.py` continues to
exercise the Postgres path in CI.
"""

import pytest
from hypha_rpc import connect_to_server

from . import SERVER_URL, SERVER_URL_SQLITE

pytestmark = pytest.mark.asyncio


async def _make_collection_with_children(
    artifact_manager, prefix, n=5, child_permissions=None
):
    collection = await artifact_manager.create(
        type="collection",
        alias=f"{prefix}-collection",
        manifest={"name": "SQLi Test Collection", "description": "sqli"},
        config={"permissions": {"*": "r", "@": "rw+"}},
    )
    for i in range(n):
        child_config = None
        if child_permissions is not None:
            child_config = {"permissions": dict(child_permissions)}
        art = await artifact_manager.create(
            type="dataset",
            alias=f"{prefix}-ds-{i}",
            parent_id=collection.id,
            manifest={"name": f"Dataset {i}", "status": "committed"},
            config=child_config,
            version="stage",
        )
        await artifact_manager.commit(artifact_id=art.id)
    return collection


@pytest.mark.asyncio
async def test_manifest_filter_value_injection_blocked(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """A boolean-injection payload in a manifest filter VALUE must not leak rows."""
    api = await connect_to_server(
        {
            "name": "sqli-client",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    collection = await _make_collection_with_children(artifact_manager, "sqli-val", n=5)

    # Sanity: an exact legitimate value matches exactly one child.
    legit = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name": "Dataset 3"}},
        mode="AND",
    )
    assert len(legit) == 1
    assert legit[0]["manifest"]["name"] == "Dataset 3"

    # Injection: no child has this literal name. If the value is interpolated,
    # `= 'nope' OR '1'='1'` is always true -> ALL 5 children leak. If bound, 0.
    injected = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name": "nope' OR '1'='1"}},
        mode="AND",
    )
    assert len(injected) == 0, (
        f"SQL injection: boolean payload returned {len(injected)} rows "
        f"(expected 0 — value must be bound, not interpolated)"
    )

    # Keyword-search injection: keywords do a `manifest LIKE '%<kw>%'` match.
    # A boolean-injection payload in a keyword must be bound (0 rows), not
    # interpolated into an always-true clause (would leak all 5).
    kw_injected = await artifact_manager.list(
        parent_id=collection.id,
        keywords=["zzz' OR '1'='1"],
        mode="AND",
    )
    assert len(kw_injected) == 0, (
        f"SQL injection via keyword returned {len(kw_injected)} rows (expected 0)"
    )

    await api.disconnect()


@pytest.mark.asyncio
async def test_manifest_filter_key_injection_blocked(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """A boolean-injection payload in a manifest filter KEY must not leak rows."""
    api = await connect_to_server(
        {
            "name": "sqli-client-key",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    collection = await _make_collection_with_children(artifact_manager, "sqli-key", n=4)

    # Malicious KEY: if interpolated into json_extract(manifest, '$.<key>'),
    # a quote-breakout in the key alters the SQL. Bound/validated -> no leak.
    injected = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name') = json_extract(manifest,'$.name') OR ('1'='1": "x"}},
        mode="AND",
    )
    assert len(injected) == 0, (
        f"SQL injection via filter KEY returned {len(injected)} rows (expected 0)"
    )

    await api.disconnect()


@pytest.mark.asyncio
async def test_config_permissions_filter_injection_blocked(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """Injection via the config.permissions filter branch must not leak rows."""
    api = await connect_to_server(
        {
            "name": "sqli-client-perm",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    # Give each child a distinctive permission entry we control so we can assert
    # the filter actually matches on config['permissions'].
    collection = await _make_collection_with_children(
        artifact_manager, "sqli-perm", n=3, child_permissions={"probe-team": "r"}
    )

    # Positive: filtering by a permission the children actually have must return
    # them — this proves the config.permissions filter branch references the
    # correct nested `config['permissions']` path (it previously referenced a
    # non-existent bare `permissions` column and always raised).
    matched = await artifact_manager.list(
        parent_id=collection.id,
        filters={"config": {"permissions": {"probe-team": "r"}}},
        mode="AND",
    )
    assert len(matched) == 3, (
        f"config.permissions filter should match all 3 children with "
        f"'probe-team':'r' (got {len(matched)} — the branch must query "
        f"config['permissions'])"
    )

    # permissions ->> '<user_id>' = '<perm>' — inject via user_id.
    injected = await artifact_manager.list(
        parent_id=collection.id,
        filters={"config": {"permissions": {"nobody' OR '1'='1": "r"}}},
        mode="AND",
    )
    assert len(injected) == 0, (
        f"SQL injection via config.permissions returned {len(injected)} rows "
        f"(expected 0)"
    )

    await api.disconnect()


@pytest.mark.asyncio
async def test_order_by_injection_rejected_or_safe(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """A malicious order_by field must not inject SQL.

    order_by field names are single identifiers (e.g. 'created_at',
    'manifest.name'). A payload containing SQL metacharacters must either be
    rejected (ValueError) or treated inertly — never executed as SQL. We assert
    the call does not succeed with an injected ordering by requiring it to raise.
    """
    api = await connect_to_server(
        {
            "name": "sqli-client-order",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    collection = await _make_collection_with_children(artifact_manager, "sqli-order", n=3)

    with pytest.raises(Exception):
        await artifact_manager.list(
            parent_id=collection.id,
            order_by="manifest.name'); DROP TABLE artifacts; --",
        )

    # Legitimate order_by still works after the guard.
    ok = await artifact_manager.list(
        parent_id=collection.id, order_by="manifest.name"
    )
    assert len(ok) == 3

    await api.disconnect()


@pytest.mark.asyncio
async def test_postgres_filter_injection_blocked(
    minio_server, fastapi_server, test_user_token
):
    """The PostgreSQL dialect (manifest->>, (config->'permissions')->>) must be
    equally injection-safe. Exercises the postgres branches of
    `_build_manifest_condition` and the config.permissions filter against a real
    Postgres backend so the bound-param SQL is validated in CI on both dialects.
    """
    api = await connect_to_server(
        {
            "name": "sqli-client-pg",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    collection = await _make_collection_with_children(
        artifact_manager, "sqli-pg", n=4, child_permissions={"probe-team": "r"}
    )

    # Sanity: exact legitimate value matches exactly one child.
    legit = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name": "Dataset 2"}},
        mode="AND",
    )
    assert len(legit) == 1

    # manifest value injection -> bound -> 0 rows.
    injected = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name": "nope' OR '1'='1"}},
        mode="AND",
    )
    assert len(injected) == 0, (
        f"PG SQL injection (manifest value) returned {len(injected)} rows"
    )

    # manifest key injection -> bound -> 0 rows.
    injected_key = await artifact_manager.list(
        parent_id=collection.id,
        filters={"manifest": {"name'') = (manifest->>'name') OR ('1'='1": "x"}},
        mode="AND",
    )
    assert len(injected_key) == 0, (
        f"PG SQL injection (manifest key) returned {len(injected_key)} rows"
    )

    # config.permissions positive + injection.
    matched = await artifact_manager.list(
        parent_id=collection.id,
        filters={"config": {"permissions": {"probe-team": "r"}}},
        mode="AND",
    )
    assert len(matched) == 4, (
        f"PG config.permissions filter should match 4 (got {len(matched)})"
    )
    injected_perm = await artifact_manager.list(
        parent_id=collection.id,
        filters={"config": {"permissions": {"nobody' OR '1'='1": "r"}}},
        mode="AND",
    )
    assert len(injected_perm) == 0, (
        f"PG SQL injection (config.permissions) returned {len(injected_perm)} rows"
    )

    # order_by guard on postgres too.
    with pytest.raises(Exception):
        await artifact_manager.list(
            parent_id=collection.id,
            order_by="manifest.name'); DROP TABLE artifacts; --",
        )

    await api.disconnect()
