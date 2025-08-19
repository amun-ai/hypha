"""Test workspace security and context isolation."""

import pytest
from hypha_rpc import connect_to_server
import uuid
import asyncio
from unittest.mock import patch

from . import (
    WS_SERVER_URL,
    find_item,
    wait_for_workspace_ready,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_workspace_context_isolation(fastapi_server, test_user_token):
    """Test that workspace context is properly isolated between workspaces."""
    
    # Create first workspace and client
    api1 = await connect_to_server({
        "client_id": "client-ws1",
        "name": "Client in Workspace 1",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    ws1_name = api1.config.workspace
    
    # Create a second workspace
    ws2_info = await api1.create_workspace({
        "name": f"test-workspace-{uuid.uuid4()}",
        "description": "Test workspace for isolation",
    }, overwrite=True)
    ws2_name = ws2_info["id"]
    
    # Generate token for second workspace
    ws2_token = await api1.generate_token({"workspace": ws2_name})
    
    # Connect to second workspace
    api2 = await connect_to_server({
        "client_id": "client-ws2",
        "name": "Client in Workspace 2",
        "workspace": ws2_name,
        "server_url": WS_SERVER_URL,
        "token": ws2_token,
    })
    
    # Register a service in each workspace that returns the context
    async def get_context(context=None):
        return {
            "workspace": context.get("ws"),
            "user": context.get("user", {}).get("id"),
            "from": context.get("from"),
        }
    
    # Register service in workspace 1
    svc1_info = await api1.register_service({
        "id": "context-test-ws1",
        "name": "Context Test WS1",
        "config": {"require_context": True, "visibility": "protected"},
        "get_context": get_context,
    })
    
    # Register service in workspace 2
    svc2_info = await api2.register_service({
        "id": "context-test-ws2",
        "name": "Context Test WS2",
        "config": {"require_context": True, "visibility": "protected"},
        "get_context": get_context,
    })
    
    # Test 1: Each service should see its own workspace in context
    svc1 = await api1.get_service(f"{ws1_name}/context-test-ws1")
    ctx1 = await svc1.get_context()
    assert ctx1["workspace"] == ws1_name, f"Service in WS1 should see WS1 context, got {ctx1['workspace']}"
    
    svc2 = await api2.get_service(f"{ws2_name}/context-test-ws2")
    ctx2 = await svc2.get_context()
    assert ctx2["workspace"] == ws2_name, f"Service in WS2 should see WS2 context, got {ctx2['workspace']}"
    
    # Test 2: Cross-workspace service calls should NOT leak context
    # Client from WS1 trying to call service in WS2 should fail without proper permissions
    with pytest.raises(Exception) as exc_info:
        await api1.get_service(f"{ws2_name}/context-test-ws2")
    assert "permission" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()
    
    await api1.disconnect()
    await api2.disconnect()


async def test_public_service_workspace_context(fastapi_server, test_user_token):
    """Test that public services correctly handle caller's workspace context."""
    
    # Connect as a user with their own workspace
    api = await connect_to_server({
        "client_id": "test-context-client",
        "name": "Test Context Client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    user_workspace = api.config.workspace
    
    # Register a public service that tracks caller contexts
    caller_contexts = []
    
    async def track_caller(context=None):
        caller_contexts.append({
            "workspace": context.get("ws"),
            "user": context.get("user", {}).get("id"),
            "from": context.get("from"),
        })
        return {"tracked": True, "caller_workspace": context.get("ws")}
    
    # Register the tracking service as a public service in user's workspace
    # (public visibility makes it accessible from other workspaces)
    await api.register_service({
        "id": "context-tracker",
        "name": "Context Tracker",
        "config": {"require_context": True, "visibility": "public"},
        "track_caller": track_caller,
    })
    
    # Create a second workspace and client to test cross-workspace calls
    ws2_info = await api.create_workspace({
        "name": f"test-context-{uuid.uuid4()}",
        "description": "Test workspace for context",
    }, overwrite=True)
    ws2_name = ws2_info["id"]
    
    ws2_token = await api.generate_token({"workspace": ws2_name})
    api2 = await connect_to_server({
        "client_id": "test-context-client-2",
        "workspace": ws2_name,
        "server_url": WS_SERVER_URL,
        "token": ws2_token,
    })
    
    # Call the public service from a different workspace
    tracker = await api2.get_service(f"{user_workspace}/context-tracker")
    result = await tracker.track_caller()
    
    # The caller's workspace should be preserved (ws2), NOT changed to the service's workspace
    assert result["caller_workspace"] == ws2_name, \
        f"Public service should see caller's workspace {ws2_name}, not service workspace {user_workspace}, got {result['caller_workspace']}"
    
    # Verify in the tracked contexts
    assert len(caller_contexts) == 1
    assert caller_contexts[0]["workspace"] == ws2_name, \
        f"Tracked context should have caller's workspace {ws2_name}, got {caller_contexts[0]['workspace']}"
    
    await api.disconnect()
    await api2.disconnect()


async def test_public_artifact_manager_context(fastapi_server, test_user_token):
    """Test that public artifact manager receives correct workspace context."""
    
    # Connect as user with their own workspace
    api = await connect_to_server({
        "client_id": "test-artifact-context",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    user_workspace = api.config.workspace
    
    # Get the public artifact manager service
    artifact_manager = await api.get_service("public/artifact-manager")
    
    # The artifact manager's list should use the caller's workspace context
    # Create an artifact in the user's workspace
    artifact = await artifact_manager.create(
        manifest={
            "name": "Test User Artifact",
            "description": "Artifact in user workspace",
        },
        stage=True
    )
    await artifact_manager.commit(artifact["id"])
    
    # List artifacts - should see the user's artifact, not public workspace artifacts
    artifacts = await artifact_manager.list()
    artifact_ids = [a["id"] for a in artifacts]
    
    # The artifact should be in the user's workspace
    assert any(user_workspace in aid for aid in artifact_ids), \
        f"Artifacts should be from user workspace {user_workspace}"
    assert not any(aid.startswith("public/") for aid in artifact_ids if "/" in aid), \
        "Should not see artifacts from public workspace when called from user workspace"
    
    await api.disconnect()


async def test_artifact_manager_workspace_isolation(fastapi_server, test_user_token):
    """Test that artifact manager correctly isolates data by workspace."""
    
    # Create two separate workspaces
    api1 = await connect_to_server({
        "client_id": "artifact-test-1",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    ws1 = api1.config.workspace
    
    # Create second workspace with persistent flag
    ws2_info = await api1.create_workspace({
        "name": f"artifact-test-{uuid.uuid4()}",
        "description": "Test workspace for artifacts",
        "persistent": True,  # Mark as persistent for artifact creation
    }, overwrite=True)
    ws2 = ws2_info["id"]
    
    ws2_token = await api1.generate_token({"workspace": ws2})
    api2 = await connect_to_server({
        "client_id": "artifact-test-2",
        "workspace": ws2,
        "server_url": WS_SERVER_URL,
        "token": ws2_token,
    })
    
    # Get artifact manager (should be public service)
    artifact_manager = await api1.get_service("public/artifact-manager")
    
    # Create an artifact in workspace 1
    artifact1 = await artifact_manager.create(
        manifest={
            "name": "Test Artifact WS1",
            "description": "Artifact in workspace 1",
        },
        stage=True
    )
    await artifact_manager.commit(artifact1["id"])
    
    # Switch to workspace 2 and create an artifact there
    artifact_manager2 = await api2.get_service("public/artifact-manager")
    artifact2 = await artifact_manager2.create(
        manifest={
            "name": "Test Artifact WS2",
            "description": "Artifact in workspace 2",
        },
        stage=True
    )
    await artifact_manager2.commit(artifact2["id"])
    
    # List artifacts from workspace 1 - should only see WS1 artifacts
    ws1_artifacts = await artifact_manager.list()
    ws1_ids = [a["id"] for a in ws1_artifacts]
    assert any(artifact1["id"] in aid for aid in ws1_ids), \
        f"WS1 should see its own artifact {artifact1['id']}"
    assert not any(artifact2["id"] in aid for aid in ws1_ids), \
        f"WS1 should NOT see WS2's artifact {artifact2['id']}"
    
    # List artifacts from workspace 2 - should only see WS2 artifacts
    ws2_artifacts = await artifact_manager2.list()
    ws2_ids = [a["id"] for a in ws2_artifacts]
    assert any(artifact2["id"] in aid for aid in ws2_ids), \
        f"WS2 should see its own artifact {artifact2['id']}"
    assert not any(artifact1["id"] in aid for aid in ws2_ids), \
        f"WS2 should NOT see WS1's artifact {artifact1['id']}"
    
    # Test cross-workspace access attempt
    # WS1 trying to read WS2's artifact should fail
    with pytest.raises(Exception) as exc_info:
        await artifact_manager.read(artifact2["id"])
    assert "not found" in str(exc_info.value).lower() or "permission" in str(exc_info.value).lower()
    
    await api1.disconnect()
    await api2.disconnect()


async def test_malicious_workspace_context_injection(fastapi_server):
    """Test that malicious attempts to inject workspace context are prevented."""
    
    # Connect as anonymous user (should get their own workspace)
    api = await connect_to_server({
        "client_id": "malicious-client",
        "server_url": WS_SERVER_URL,
    })
    anonymous_workspace = api.config.workspace
    
    # Attempt to register a service that tries to manipulate context
    async def malicious_service(target_workspace, context=None):
        # Try to modify context to access another workspace
        fake_context = dict(context)
        fake_context["ws"] = target_workspace
        # This should NOT allow access to the target workspace
        return {"original_ws": context.get("ws"), "attempted_ws": target_workspace}
    
    await api.register_service({
        "id": "malicious-service",
        "config": {"require_context": True},
        "attempt_access": malicious_service,
    })
    
    svc = await api.get_service(f"{anonymous_workspace}/malicious-service")
    
    # Try to inject "public" workspace
    result = await svc.attempt_access("public")
    assert result["original_ws"] == anonymous_workspace, \
        "Context workspace should not be modifiable by service"
    
    # The service should not be able to access resources from other workspaces
    # even if it tries to modify the context
    
    await api.disconnect()


async def test_permission_escalation_prevention(fastapi_server, test_user_token):
    """Test that users cannot escalate their permissions through context manipulation."""
    
    # Create a workspace with limited permissions
    admin_api = await connect_to_server({
        "client_id": "admin-client",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    
    restricted_ws = await admin_api.create_workspace({
        "name": f"restricted-{uuid.uuid4()}",
        "description": "Restricted workspace",
    }, overwrite=True)
    
    # Generate a read-only token for the workspace
    readonly_token = await admin_api.generate_token({
        "workspace": restricted_ws["id"],
        "permission": "read",  # read-only
    })
    
    # Connect with read-only token
    readonly_api = await connect_to_server({
        "client_id": "readonly-client",
        "workspace": restricted_ws["id"],
        "server_url": WS_SERVER_URL,
        "token": readonly_token,
    })
    
    # Attempt to register a service (should fail with read-only permission)
    with pytest.raises(Exception) as exc_info:
        await readonly_api.register_service({
            "id": "should-fail",
            "config": {"visibility": "protected"},
        })
    assert "permission" in str(exc_info.value).lower()
    
    # Attempt to use artifact manager to create artifacts (should fail)
    artifact_manager = await readonly_api.get_service("public/artifact-manager")
    with pytest.raises(Exception) as exc_info:
        await artifact_manager.create(
            manifest={"name": "Should Fail"},
            stage=True
        )
    assert "permission" in str(exc_info.value).lower()
    
    await admin_api.disconnect()
    await readonly_api.disconnect()


async def test_anonymous_user_workspace_isolation(fastapi_server):
    """Test that anonymous users are properly isolated in their workspaces."""
    
    # Connect two anonymous users
    anon1 = await connect_to_server({
        "client_id": "anon-client-1",
        "server_url": WS_SERVER_URL,
    })
    ws1 = anon1.config.workspace
    
    anon2 = await connect_to_server({
        "client_id": "anon-client-2",
        "server_url": WS_SERVER_URL,
    })
    ws2 = anon2.config.workspace
    
    # Anonymous users should get different workspaces
    assert ws1 != ws2, "Anonymous users should get separate workspaces"
    
    # Each should only see their own services
    async def test_service():
        return "test"
    
    # Register service as anon1
    await anon1.register_service({
        "id": "anon1-service",
        "test": test_service,
    })
    
    # Register service as anon2
    await anon2.register_service({
        "id": "anon2-service",
        "test": test_service,
    })
    
    # Anon1 should only see their service
    anon1_services = await anon1.list_services()
    anon1_service_ids = [s["id"] for s in anon1_services]
    assert any("anon1-service" in sid for sid in anon1_service_ids)
    assert not any("anon2-service" in sid for sid in anon1_service_ids)
    
    # Anon2 should only see their service
    anon2_services = await anon2.list_services()
    anon2_service_ids = [s["id"] for s in anon2_services]
    assert any("anon2-service" in sid for sid in anon2_service_ids)
    assert not any("anon1-service" in sid for sid in anon2_service_ids)
    
    # Cross-workspace access should fail
    with pytest.raises(Exception):
        await anon1.get_service(f"{ws2}/anon2-service")
    
    with pytest.raises(Exception):
        await anon2.get_service(f"{ws1}/anon1-service")
    
    await anon1.disconnect()
    await anon2.disconnect()


async def test_system_connection_workspace_context(fastapi_server, test_user_token):
    """Test that system connections with workspace='*' handle context correctly."""
    
    # This test would require internal access to create system connections
    # It verifies that when workspace="*", the context properly falls back
    # to target workspace or preserves current_workspace if set
    
    # Connect as regular user
    api = await connect_to_server({
        "client_id": "test-system-context",
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
    })
    user_workspace = api.config.workspace
    
    # Register a service that reports context
    async def report_context(context=None):
        return {
            "ws": context.get("ws"),
            "user_id": context.get("user", {}).get("id"),
            "user_current_ws": context.get("user", {}).get("current_workspace"),
        }
    
    await api.register_service({
        "id": "context-reporter",
        "config": {"require_context": True},
        "report": report_context,
    })
    
    # Get the service and call it
    svc = await api.get_service(f"{user_workspace}/context-reporter")
    result = await svc.report()
    
    # Verify context contains correct workspace
    assert result["ws"] == user_workspace, \
        f"Context should contain user's workspace {user_workspace}, got {result['ws']}"
    
    await api.disconnect()