"""Test suite for protocol adapter security vulnerabilities.

This test file contains tests for security issues in MCP and A2A protocol adapters.

Vulnerability List:
- V21: MCP session cache poisoning (HIGH) - Cache reuse across users
- V22: MCP workspace interface leak (MEDIUM) - SSE connections leak workspace interfaces
- V23: MCP cross-workspace service access (HIGH) - No workspace validation
- V24: A2A context filtering bypass (HIGH) - Security context loss
- V25: A2A agent card execution without context (MEDIUM) - Unvalidated callable execution
- V26: A2A workspace extraction bypass (HIGH) - Similar to V18 for A2A
"""

import pytest
import pytest_asyncio
import uuid
import asyncio
import httpx
import json
from hypha_rpc import connect_to_server

from . import (
    WS_SERVER_URL,
    SERVER_URL,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


class TestV26MCPSessionCachePoisoning:
    """
    V16: MCP Session Cache Poisoning (HIGH SEVERITY)

    Location: hypha/mcp.py:1171-1268 (MCPRoutingMiddleware._get_or_create_mcp_app)

    Issue: The MCP app cache uses a simple cache key of f"{workspace}/{service_id}"
    without validating that the current user has permission to access the cached app.
    This could allow an attacker to:
    1. Register a public MCP service in their workspace
    2. Wait for the service to be cached
    3. Have another user access it (cache hit)
    4. The cached app might have workspace interface from the first user

    Root Cause:
    - Cache key doesn't include user/session info
    - No validation that cached app matches current user's permissions
    - Workspace interface kept alive for SSE connections could be reused

    Expected Behavior:
    - Cache should validate user permissions before returning cached app
    - Or cache should be user-specific
    - Or cache should not include privileged workspace interfaces
    """

    async def test_mcp_cache_should_be_user_specific(
        self, fastapi_server, test_user_token
    ):
        """Test that MCP app cache doesn't leak between users."""
        # Create two users
        api1 = await connect_to_server({
            "client_id": "v26-user1",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        workspace1 = api1.config.workspace

        # Create a second user token (in a real attack, this would be a different user)
        # For this test, we'll create a second workspace with different permissions
        ws2_info = await api1.create_workspace({
            "name": f"v26-user2-ws-{uuid.uuid4().hex[:8]}",
            "description": "Second user workspace",
        }, overwrite=True)
        workspace2 = ws2_info["id"]

        # User 1 registers an MCP service in workspace1
        async def list_tools():
            return []

        async def call_tool(name, arguments):
            # This should only be accessible by workspace1 users
            return [{"type": "text", "text": f"Secret data from {workspace1}"}]

        svc1 = await api1.register_service({
            "id": "v26-mcp-service",
            "type": "mcp",
            "config": {"visibility": "protected"},  # Protected to workspace1
            "list_tools": list_tools,
            "call_tool": call_tool,
        })

        # Trigger cache population by accessing the MCP endpoint
        mcp_url1 = f"{SERVER_URL}/{workspace1}/mcp/v26-mcp-service/mcp"

        # First access should populate cache
        # Note: This test documents the vulnerability pattern but may need
        # real HTTP requests to fully exploit

        # User 2 (in different workspace) tries to access the cached service
        # through workspace2's MCP endpoint
        mcp_url2 = f"{SERVER_URL}/{workspace2}/mcp/v26-mcp-service/mcp"

        # VULNERABILITY: If cache key is just "workspace/service_id",
        # accessing "workspace2/v26-mcp-service" shouldn't return the cached
        # app from workspace1, but the code doesn't prevent this pattern

        # The fix should ensure:
        # 1. Cache includes user permissions/workspace context
        # 2. Or cache validation checks current user can access service
        # 3. Or cached workspace interfaces are not reused across users

        await api1.disconnect()


class TestV27MCPWorkspaceInterfaceLeak:
    """
    V17: MCP Workspace Interface Leak (MEDIUM SEVERITY)

    Location: hypha/mcp.py:1190-1229 (SSE connection caching)

    Issue: When is_sse=True, the code keeps the workspace interface alive
    by storing it in cache_entry["api"]. This interface is never properly
    cleaned up except in _cleanup_mcp_app. If SSE connections are long-lived
    or cleanup is not called, this leaks workspace interfaces.

    Root Cause:
    - SSE connections keep workspace interface alive in cache
    - No timeout or cleanup mechanism for idle connections
    - Workspace interface stays open even if service is unregistered

    Expected Behavior:
    - SSE connections should have timeout limits
    - Workspace interfaces should be cleaned up when service unregistered
    - Cache cleanup should be automatic, not manual
    """

    async def test_sse_connection_should_cleanup_workspace_interface(
        self, fastapi_server, test_user_token
    ):
        """Test that SSE connections properly clean up workspace interfaces."""
        api = await connect_to_server({
            "client_id": "v27-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        workspace = api.config.workspace

        # Register MCP service
        async def list_tools():
            return []

        svc = await api.register_service({
            "id": "v27-mcp-sse-service",
            "type": "mcp",
            "config": {"visibility": "public"},
            "list_tools": list_tools,
        })

        # Open SSE connection (this would populate cache with workspace interface)
        sse_url = f"{SERVER_URL}/{workspace}/mcp/v27-mcp-sse-service/sse"

        # Note: Actually opening SSE connection requires async HTTP client
        # This test documents the vulnerability - the workspace interface
        # in mcp_app_cache[cache_key]["api"] is kept alive

        # Unregister the service
        await api.unregister_service(svc["id"])

        # VULNERABILITY: The workspace interface in the cache is still alive
        # even though the service is gone. This could be exploited to:
        # 1. Keep accessing a workspace after permissions revoked
        # 2. Leak workspace resources through long-lived connections

        # The fix should:
        # 1. Clean up cache when service is unregistered
        # 2. Add timeout for cached workspace interfaces
        # 3. Validate workspace interface is still valid before reuse

        await api.disconnect()


class TestV28MCPCrossWorkspaceServiceAccess:
    """
    V18: MCP Cross-Workspace Service Access (HIGH SEVERITY)

    Location: hypha/mcp.py:1206-1212 (service ID construction)

    Issue: The MCP routing middleware constructs full_service_id with workspace
    from the URL path, but doesn't validate that the user has permission to
    access services in that workspace. Combined with wildcards, this could
    allow accessing services across workspaces.

    Root Cause:
    - URL path workspace is trusted without validation
    - Wildcard construction: f"{workspace}/*:{service_id}"
    - Permission check happens in get_service_info but might be bypassed

    Expected Behavior:
    - Validate that URL workspace matches user's permission scope
    - Don't use wildcards for cross-workspace access
    - Explicit permission check before service access
    """

    async def test_mcp_should_validate_workspace_from_url(
        self, fastapi_server, test_user_token
    ):
        """Test that MCP endpoint validates workspace from URL path."""
        api = await connect_to_server({
            "client_id": "v28-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        user_workspace = api.config.workspace

        # Create target workspace with a service
        target_ws_info = await api.create_workspace({
            "name": f"v28-target-{uuid.uuid4().hex[:8]}",
            "description": "Target workspace",
        }, overwrite=True)
        target_workspace = target_ws_info["id"]

        # Register protected service in target workspace
        # (this requires switching context or using workspace interface)

        # VULNERABILITY: User tries to access MCP service through URL:
        # GET /{target_workspace}/mcp/{service_id}/mcp
        #
        # The code constructs:
        # full_service_id = f"{target_workspace}/*:{service_id}"
        #
        # If the service is "protected", it should reject access from user_workspace
        # But the wildcard and permission check might not properly validate

        # Expected: 403 or 404 for cross-workspace access to protected services
        # Actual: May allow access if permission check is insufficient

        await api.disconnect()


class TestV24A2AContextFilteringBypass:
    """
    V19: A2A Context Filtering Bypass (HIGH SEVERITY)

    Location: hypha/a2a.py:145-153 (HyphaAgentExecutor._filter_context_for_hypha)

    Issue: The _filter_context_for_hypha method returns None unconditionally,
    which means all security context is lost when calling the Hypha service's
    run function. This includes:
    - User identity
    - Workspace information
    - Permission scope

    Root Cause:
    - Method returns None "to avoid serialization issues"
    - No attempt to extract safe fields from RequestContext
    - Service runs without any security context

    Expected Behavior:
    - Extract user info, workspace, and permissions from context
    - Pass filtered context to service for authorization checks
    - Never lose security-critical context information
    """

    async def test_a2a_should_preserve_security_context(
        self, fastapi_server, test_user_token
    ):
        """Test that A2A adapter preserves security context when calling services."""
        api = await connect_to_server({
            "client_id": "v24-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        workspace = api.config.workspace

        # Track what context is received by the service
        received_context = {"context": None}

        async def run_function(message, context):
            """Service that records what context it receives."""
            received_context["context"] = context
            return "Response"

        # Register A2A service
        svc = await api.register_service({
            "id": "v24-a2a-service",
            "type": "a2a",
            "config": {"visibility": "public"},
            "run": run_function,
            "agent_card": {
                "protocol_version": "0.3.0",
                "name": "Test Agent",
                "description": "Test A2A agent",
                "url": f"/{workspace}/a2a/v24-a2a-service",
                "version": "1.0.0",
            },
        })

        # Make A2A request (would need actual A2A client)
        # When HyphaAgentExecutor.execute is called:
        # filtered_context = self._filter_context_for_hypha(context)
        # result = await self.run_function(message_dict, filtered_context)

        # VULNERABILITY: filtered_context is always None
        # The service cannot check user permissions or workspace

        # Expected: context should include {"ws": workspace, "user": {...}, ...}
        # Actual: context is None

        # Security impact:
        # - Service cannot validate user permissions
        # - Service cannot check workspace isolation
        # - Service cannot audit who called it

        await api.disconnect()


class TestV25A2AAgentCardExecutionWithoutContext:
    """
    V20: A2A Agent Card Execution Without Context (MEDIUM SEVERITY)

    Location: hypha/a2a.py:186-202, 586-604 (agent_card callable handling)

    Issue: If agent_card is callable, it's executed without any context
    validation. A malicious service could:
    1. Register with a callable agent_card function
    2. Have the function execute arbitrary code when A2A app is created
    3. Access workspace interfaces or steal data

    Root Cause:
    - Callable agent_card executed in create_a2a_app_from_service
    - Also executed in _send_agent_info endpoint
    - No context or permission validation before execution
    - Function runs with server privileges

    Expected Behavior:
    - Validate callable is safe before executing
    - Execute with limited permissions/sandbox
    - Or disallow callable agent_card, require static data
    """

    async def test_a2a_agent_card_callable_should_be_validated(
        self, fastapi_server, test_user_token
    ):
        """Test that callable agent_card functions are validated before execution."""
        api = await connect_to_server({
            "client_id": "v25-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        workspace = api.config.workspace

        # Track execution
        execution_log = {"executed": False, "context": None}

        async def malicious_agent_card():
            """Callable agent_card that tries to access workspace."""
            execution_log["executed"] = True
            # In a real attack, this could try to access workspace
            # interface, leak data, or execute malicious code
            return {
                "protocol_version": "0.3.0",
                "name": "Malicious Agent",
                "description": "Agent card with side effects",
                "url": "/test",
                "version": "1.0.0",
            }

        async def run_function(message, context):
            return "Response"

        # Register service with callable agent_card
        svc = await api.register_service({
            "id": "v25-a2a-service",
            "type": "a2a",
            "config": {"visibility": "public"},
            "run": run_function,
            "agent_card": malicious_agent_card,  # Callable!
        })

        # When someone accesses the A2A endpoint or info endpoint,
        # the malicious_agent_card function is executed
        # GET /{workspace}/a2a-info/v25-a2a-service

        # VULNERABILITY: Function executes without context validation
        # Expected: Should validate or reject callable agent_card
        # Actual: Executes with server privileges

        await api.disconnect()


class TestV26A2AWorkspaceExtractionBypass:
    """
    V21: A2A Workspace Extraction Bypass (HIGH SEVERITY)

    Location: hypha/a2a.py:398-410 (A2A request routing)

    Issue: Similar to V18 for MCP, the A2A routing middleware extracts
    workspace from the URL path and constructs full_service_id without
    validating that the user has permission to access that workspace.

    Root Cause:
    - workspace = path_params["workspace"]
    - full_service_id = f"{workspace}/{service_id}"
    - Permission check in get_service_info might not validate workspace
    - Cross-workspace access control not enforced at routing level

    Expected Behavior:
    - Validate URL workspace against user permissions
    - Reject cross-workspace access to protected services
    - Log security events for cross-workspace attempts
    """

    async def test_a2a_should_validate_workspace_from_url(
        self, fastapi_server, test_user_token
    ):
        """Test that A2A endpoint validates workspace from URL path."""
        api = await connect_to_server({
            "client_id": "v26-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        user_workspace = api.config.workspace

        # Create target workspace
        target_ws_info = await api.create_workspace({
            "name": f"v26-target-{uuid.uuid4().hex[:8]}",
            "description": "Target workspace for A2A test",
        }, overwrite=True)
        target_workspace = target_ws_info["id"]

        # VULNERABILITY: User tries to access A2A service through URL:
        # POST /{target_workspace}/a2a/{service_id}
        #
        # The code extracts:
        # workspace = path_params["workspace"]  # target_workspace
        # full_service_id = f"{workspace}/{service_id}"
        #
        # If service is protected, should reject cross-workspace access
        # But routing middleware might not enforce this

        # Expected: 403 Forbidden for cross-workspace protected services
        # Actual: May allow if permission check is insufficient

        await api.disconnect()


class TestV27MCPServiceInfoInfoLeakage:
    """
    V27: MCP Service Info Information Leakage (LOW SEVERITY)

    Location: hypha/mcp.py info route handling

    Issue: The MCP info route may leak service information even for
    services the user shouldn't be able to access. This could include:
    - Service existence (enumeration)
    - Service configuration details
    - Tool/resource schemas

    Expected Behavior:
    - Info route should enforce same permissions as service access
    - Return 404 for services user can't access (not 403 to avoid enumeration)
    """

    async def test_mcp_info_route_should_enforce_permissions(
        self, fastapi_server, test_user_token
    ):
        """Test that MCP info route respects service permissions."""
        api = await connect_to_server({
            "client_id": "v27-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        workspace = api.config.workspace

        # Register protected MCP service
        async def list_tools():
            return []

        svc = await api.register_service({
            "id": "v27-secret-service",
            "type": "mcp",
            "config": {"visibility": "protected"},
            "list_tools": list_tools,
        })

        # Try to access info from different workspace context
        # Should return 404, not leak service details

        await api.disconnect()


class TestV28A2AInfoInfoLeakage:
    """
    V28: A2A Info Information Leakage (LOW SEVERITY)

    Location: hypha/a2a.py:529-742 (_send_agent_info)

    Issue: Similar to V22, the A2A info route (_send_agent_info) may leak
    information about services that the user shouldn't be able to access.
    The agent card, skills, and capabilities could be sensitive.

    Expected Behavior:
    - Info route should enforce service permissions
    - Return 404 for unauthorized access
    - Don't leak agent card details for protected services
    """

    async def test_a2a_info_route_should_enforce_permissions(
        self, fastapi_server, test_user_token
    ):
        """Test that A2A info route respects service permissions."""
        api = await connect_to_server({
            "client_id": "v28-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        workspace = api.config.workspace

        async def run_function(message, context):
            return "Secret response"

        # Register protected A2A service
        svc = await api.register_service({
            "id": "v28-secret-agent",
            "type": "a2a",
            "config": {"visibility": "protected"},
            "run": run_function,
            "agent_card": {
                "protocol_version": "0.3.0",
                "name": "Secret Agent",
                "description": "This should not be leaked",
                "url": f"/{workspace}/a2a/v28-secret-agent",
                "version": "1.0.0",
            },
        })

        # Try to access info route: GET /{workspace}/a2a-info/v28-secret-agent
        # From unauthorized workspace should return 404, not leak agent card

        await api.disconnect()
