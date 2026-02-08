"""Test SQL injection vulnerabilities in database queries.

This test suite identifies SQL injection vectors in:
1. Manifest filter queries (CRITICAL)
2. Event log queries
3. Artifact metadata queries
"""
import pytest
import time
from hypha_rpc import connect_to_server


@pytest.mark.asyncio
async def test_manifest_filter_sql_injection_postgresql(hypha_server_with_postgres, test_user_token):
    """Test SQL injection via manifest filter in PostgreSQL backend.

    VULNERABILITY: _build_manifest_condition() uses string interpolation with user input
    directly in SQL queries without proper escaping.

    Location: hypha/artifact.py:8176-8230

    Attack Vector:
    manifest_filter = {"name": {"$like": "test' OR '1'='1"}}

    This gets converted to:
    manifest->>'name' ILIKE 'test' OR '1'='1%'

    Which allows bypassing workspace isolation to list artifacts from other workspaces.
    """
    server_url = hypha_server_with_postgres["server_url"]

    # Create a victim artifact in workspace A
    api_a = await connect_to_server(
        {"name": "client-a", "server_url": server_url, "workspace": "workspace-a", "token": test_user_token}
    )

    artifact_a = await api_a.create_artifact(
        alias="victim-artifact",
        manifest={"name": "confidential-data", "secret": "password123"}
    )

    # Create attacker client in workspace B
    api_b = await connect_to_server(
        {"name": "client-b", "server_url": server_url, "workspace": "workspace-b", "token": test_user_token}
    )

    # Attempt SQL injection to bypass workspace isolation
    # Inject SQL that closes the quote and adds OR condition
    injection_payload = "' OR manifest->>'secret' = 'password123' OR manifest->>'name' ILIKE '"

    try:
        # This should only return artifacts from workspace-b, but injection allows cross-workspace access
        results = await api_b.list_artifacts(
            filters={"manifest": {"name": {"$like": injection_payload}}}
        )

        # If we can see artifacts from workspace-a, injection succeeded
        workspace_a_artifacts = [r for r in results["items"] if r["workspace"] == "workspace-a"]

        if workspace_a_artifacts:
            pytest.fail(
                f"SQL INJECTION SUCCESSFUL: Accessed {len(workspace_a_artifacts)} artifacts "
                f"from workspace-a while authenticated to workspace-b. "
                f"Leaked artifact: {workspace_a_artifacts[0]}"
            )
    except Exception as e:
        # Even if it errors, the query was attempted
        print(f"Injection attempt caused error (may still be vulnerable): {e}")


@pytest.mark.asyncio
async def test_manifest_filter_sql_injection_sqlite(hypha_server, test_user_token):
    """Test SQL injection via manifest filter in SQLite backend.

    VULNERABILITY: Same as PostgreSQL but using SQLite syntax.

    Attack Vector:
    manifest_filter = {"name": "test' OR '1'='1"}

    This gets converted to:
    json_extract(manifest, '$.name') LIKE 'test' OR '1'='1%'
    """
    server_url = hypha_server["server_url"]

    # Create victim artifact
    api_a = await connect_to_server(
        {"name": "client-a", "server_url": server_url, "workspace": "workspace-a", "token": test_user_token}
    )

    victim = await api_a.create_artifact(
        alias="secret-artifact",
        manifest={"name": "classified", "level": "top-secret"}
    )

    # Attacker in different workspace
    api_b = await connect_to_server(
        {"name": "client-b", "server_url": server_url, "workspace": "workspace-b", "token": test_user_token}
    )

    # SQLite injection payload
    injection = "' OR json_extract(manifest, '$.level') = 'top-secret' OR json_extract(manifest, '$.name') LIKE '"

    try:
        results = await api_b.list_artifacts(
            filters={"manifest": {"name": injection}}
        )

        leaked = [r for r in results["items"] if r["workspace"] == "workspace-a"]

        if leaked:
            pytest.fail(
                f"SQL INJECTION in SQLite: Leaked {len(leaked)} artifacts from workspace-a. "
                f"Data: {leaked[0]['manifest']}"
            )
    except Exception as e:
        print(f"SQLite injection error: {e}")


@pytest.mark.asyncio
async def test_manifest_filter_time_based_sql_injection(hypha_server_with_postgres, test_user_token):
    """Test time-based blind SQL injection for data exfiltration.

    VULNERABILITY: Can use pg_sleep() to exfiltrate data bit by bit.

    Attack:
    name = "' OR (SELECT CASE WHEN (SELECT COUNT(*) FROM artifacts WHERE workspace != 'workspace-b') > 0
                  THEN pg_sleep(5) ELSE 0 END)::text = '0"

    If query takes >5 seconds, we know artifacts exist in other workspaces.
    """
    server_url = hypha_server_with_postgres["server_url"]

    api = await connect_to_server(
        {"name": "attacker", "server_url": server_url, "workspace": "attacker-ws", "token": test_user_token}
    )

    # Time-based payload to detect other workspaces
    # If other workspaces exist, this will cause a 3 second delay
    payload = (
        "' OR (SELECT CASE WHEN EXISTS(SELECT 1 FROM artifacts WHERE workspace != 'attacker-ws') "
        "THEN pg_sleep(3) ELSE 1 END) IS NOT NULL OR manifest->>'name' ILIKE '"
    )

    start = time.time()
    try:
        await api.list_artifacts(filters={"manifest": {"name": {"$like": payload}}})
        elapsed = time.time() - start

        if elapsed > 2.5:  # Account for network latency
            pytest.fail(
                f"TIME-BASED SQL INJECTION: Query took {elapsed:.2f}s, indicating "
                f"successful data exfiltration via pg_sleep(). Can enumerate all workspace data."
            )
    except Exception as e:
        print(f"Time-based injection error: {e}")


@pytest.mark.asyncio
async def test_manifest_key_injection(hypha_server, test_user_token):
    """Test SQL injection via manifest key names.

    VULNERABILITY: The manifest_key parameter is directly interpolated into SQL.

    Location: hypha/artifact.py:8182 and others
    text(f"manifest->>'{manifest_key}' ILIKE '{value}'")

    Attack: Use manifest_key = "name' OR '1'='1' --"
    """
    server_url = hypha_server["server_url"]

    api = await connect_to_server(
        {"name": "attacker", "server_url": server_url, "workspace": "test-ws", "token": test_user_token}
    )

    # Inject via the key name itself
    # This requires controlling the manifest filter key
    malicious_filter = {
        "name' OR '1'='1' -- ": {"$like": "anything"}
    }

    try:
        results = await api.list_artifacts(filters={"manifest": malicious_filter})

        # If this doesn't error, the injection was processed
        print(f"Key injection returned {len(results['items'])} items (may indicate vulnerability)")

    except Exception as e:
        # Expected - the key format might be validated
        print(f"Key injection blocked or errored: {e}")


@pytest.mark.asyncio
async def test_event_type_injection(hypha_server, test_user_token):
    """Test injection via event_type parameter.

    Location: hypha/core/workspace.py:502, 546
    query.filter(EventLog.event_type == event_type)

    This uses SQLAlchemy parameterization (SAFE), but test to confirm.
    """
    server_url = hypha_server["server_url"]

    api = await connect_to_server(
        {"name": "client", "server_url": server_url, "workspace": "test-ws", "token": test_user_token}
    )

    # Log a normal event
    await api.log_event(event_type="test_event", data={"key": "value"})

    # Attempt injection via event_type filter
    injection_event_type = "test_event' OR '1'='1"

    try:
        stats = await api.get_event_stats(event_type=injection_event_type)

        # If we get unexpected results, might be vulnerable
        if stats:
            print(f"Event type injection may be vulnerable: {stats}")
    except Exception as e:
        print(f"Event type injection test error: {e}")


@pytest.mark.asyncio
async def test_union_based_sql_injection(hypha_server_with_postgres, test_user_token):
    """Test UNION-based SQL injection to extract database schema.

    VULNERABILITY: Use UNION to append malicious SELECT queries.

    Attack:
    name = "' UNION SELECT id, workspace, alias, manifest, NULL, NULL, NULL, NULL, NULL, NULL
            FROM artifacts WHERE workspace != 'attacker-ws' --"
    """
    server_url = hypha_server_with_postgres["server_url"]

    api = await connect_to_server(
        {"name": "attacker", "server_url": server_url, "workspace": "attacker-ws", "token": test_user_token}
    )

    # UNION injection to extract all artifacts regardless of workspace
    union_payload = (
        "' UNION SELECT id, workspace, alias, manifest::text, staging::text, "
        "0.0, 0.0, 0, created_at, created_by, last_modified, NULL, NULL, NULL, NULL "
        "FROM artifacts WHERE workspace != 'attacker-ws' -- "
    )

    try:
        results = await api.list_artifacts(
            filters={"manifest": {"name": {"$like": union_payload}}}
        )

        # Check if we got artifacts from other workspaces
        other_ws = [r for r in results["items"] if r.get("workspace") != "attacker-ws"]

        if other_ws:
            pytest.fail(
                f"UNION-BASED SQL INJECTION: Extracted {len(other_ws)} artifacts "
                f"from other workspaces via UNION query. Full database compromise possible."
            )
    except Exception as e:
        print(f"UNION injection error (syntax or blocked): {e}")


@pytest.mark.asyncio
async def test_artifact_id_injection(hypha_server, test_user_token):
    """Test SQL injection via artifact_id parameter.

    Location: hypha/artifact.py:1903-1914 (_get_artifact_id_cond)

    The function splits by "/" and uses the parts directly:
    ws, alias = artifact_id.split("/")
    return and_(ArtifactModel.workspace == ws, ArtifactModel.alias == alias)

    SQLAlchemy should parameterize these (SAFE), but test to confirm.
    """
    server_url = hypha_server["server_url"]

    api = await connect_to_server(
        {"name": "client", "server_url": server_url, "workspace": "test-ws", "token": test_user_token}
    )

    # Create legitimate artifact
    artifact = await api.create_artifact(alias="test-artifact")

    # Attempt injection via workspace part of artifact ID
    malicious_id = "test-ws' OR '1'='1/test-artifact"

    try:
        result = await api.read_artifact(malicious_id)
        print(f"Artifact ID injection test returned: {result.get('alias')}")
    except Exception as e:
        print(f"Artifact ID injection blocked: {e}")


@pytest.mark.asyncio
async def test_json_extract_injection_sqlite(hypha_server, test_user_token):
    """Test injection via json_extract function in SQLite.

    VULNERABILITY: The manifest key is embedded in json_extract path.

    text(f"json_extract(manifest, '$.{manifest_key}')")

    If manifest_key = "name') OR json_extract(manifest, '$.secret') = ('password"
    Results in: json_extract(manifest, '$.name') OR json_extract(manifest, '$.secret') = ('password')
    """
    server_url = hypha_server["server_url"]

    # Create victim artifact with secret
    api_a = await connect_to_server(
        {"name": "victim", "server_url": server_url, "workspace": "victim-ws", "token": test_user_token}
    )

    await api_a.create_artifact(
        alias="secret-doc",
        manifest={"name": "public", "password": "admin123"}
    )

    # Attacker tries to extract password field
    api_b = await connect_to_server(
        {"name": "attacker", "server_url": server_url, "workspace": "attacker-ws", "token": test_user_token}
    )

    # This would require controlling the key name, which might not be possible
    # But demonstrates the vulnerability if manifest keys are user-controlled
    try:
        # Direct attack if we can control nested filter structure
        # This is a demonstration - actual exploitation depends on filter parsing
        malicious_key = "name') OR json_extract(manifest, '$.password') = ('admin123"
        results = await api_b.list_artifacts(
            filters={"manifest": {malicious_key: "anything"}}
        )

        if results.get("items"):
            print(f"JSON extract injection may be vulnerable: {len(results['items'])} items")
    except Exception as e:
        print(f"JSON extract injection error: {e}")


@pytest.mark.asyncio
async def test_second_order_sql_injection(hypha_server, test_user_token):
    """Test second-order SQL injection via stored manifest data.

    ATTACK SCENARIO:
    1. Store malicious payload in artifact manifest
    2. Payload gets executed when manifest is queried later

    This tests if stored manifest values are properly escaped when used in queries.
    """
    server_url = hypha_server["server_url"]

    api = await connect_to_server(
        {"name": "attacker", "server_url": server_url, "workspace": "test-ws", "token": test_user_token}
    )

    # Store malicious manifest
    malicious_manifest = {
        "name": "test' OR '1'='1",
        "description": "'; DROP TABLE artifacts; --"
    }

    artifact = await api.create_artifact(
        alias="poisoned",
        manifest=malicious_manifest
    )

    # Now search for artifacts - does the stored value get used in query unsafely?
    try:
        results = await api.list_artifacts(
            filters={"manifest": {"name": malicious_manifest["name"]}}
        )

        # If this works normally, the stored value was probably escaped
        print(f"Second-order injection test: found {len(results['items'])} items")
    except Exception as e:
        print(f"Second-order injection error: {e}")


def test_identify_vulnerable_code_locations():
    """Document all vulnerable code locations for reference.

    CRITICAL VULNERABILITIES:

    1. hypha/artifact.py:8182-8228 (_build_manifest_condition)
       - Direct string interpolation of user input into SQL
       - Affects both PostgreSQL and SQLite backends
       - Severity: CRITICAL

       Vulnerable lines:
       - Line 8186: f"manifest->>'{manifest_key}' ILIKE '{value.replace('*', '%')}'"
       - Line 8190: f"json_extract(manifest, '$.{manifest_key}') LIKE '{value.replace('*', '%')}'"
       - Line 8217: f"'{v}'" for v in value (array iteration)
       - And many more...

    2. hypha/artifact.py:8232-8280 (_process_manifest_filter)
       - Recursively processes manifest filters
       - No input validation before passing to _build_manifest_condition
       - Severity: CRITICAL

    POTENTIAL VULNERABILITIES (need confirmation):

    3. Event log queries use SQLAlchemy parameters (likely SAFE)
       - hypha/core/workspace.py:502, 546

    4. Artifact ID queries use SQLAlchemy parameters (likely SAFE)
       - hypha/artifact.py:1908-1911
    """
    pass
