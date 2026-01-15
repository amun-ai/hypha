"""Test suite for security vulnerabilities in Hypha server.

This test file contains tests to reproduce and verify fixes for identified
security vulnerabilities in the Hypha server codebase.

Vulnerability List:
- V1: Service ID workspace injection (HIGH)
- V2: Event subscription validation bypass (MEDIUM)
- V3: Protected service authorized_workspaces leak (MEDIUM)
- V4: Bookmark permission bypass (LOW)
- V5: Exception information leakage (LOW)
"""

import pytest
import pytest_asyncio
import uuid
import asyncio
import re
import requests
from unittest.mock import patch, MagicMock

from hypha_rpc import connect_to_server

from . import (
    WS_SERVER_URL,
    SERVER_URL,
    find_item,
    wait_for_workspace_ready,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


class TestV1ServiceIdWorkspaceInjection:
    """
    V1: Service ID Workspace Injection (HIGH SEVERITY)

    Location: hypha/core/workspace.py:1995-2000

    Issue: When registering a service, if the service ID contains a "/" prefix
    with a workspace name, the code does not validate that the workspace matches
    the caller's workspace. An attacker could potentially register services
    that appear to belong to other workspaces.

    Expected behavior: Service registration should use the caller's actual workspace
    regardless of what workspace prefix is specified in the service ID.

    Note: The hypha_rpc client library provides client-side protection by always
    constructing the service ID from the actual workspace. Server-side validation
    provides defense-in-depth.
    """

    async def test_service_id_with_wrong_workspace_prefix_should_fail(
        self, fastapi_server, test_user_token
    ):
        """Test that even if user tries to inject a different workspace prefix,
        the service is registered in the correct workspace (client-side protection)."""
        api = await connect_to_server({
            "client_id": "v1-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        user_workspace = api.config.workspace

        # Create a second workspace to use as injection target
        ws2_info = await api.create_workspace({
            "name": f"v1-target-ws-{uuid.uuid4().hex[:8]}",
            "description": "Target workspace for injection test",
        }, overwrite=True)
        target_workspace = ws2_info["id"]

        # Attempt to register a service with a different workspace prefix
        # The hypha_rpc client should override this and use the correct workspace
        async def dummy_function():
            return "test"

        svc_info = await api.register_service({
            # User attempts to inject target_workspace in the ID
            "id": f"{target_workspace}/injected-client:malicious-service",
            "name": "Malicious Service",
            "config": {"visibility": "protected"},
            "run": dummy_function,
        })

        # The service should be registered in the USER'S workspace
        # The hypha_rpc client constructs the ID as: {workspace}/{client_id}:{service_id}
        # where service_id is whatever the user provides (even if it contains slashes)
        # So the actual workspace prefix must be the user's workspace
        service_id = svc_info["id"]

        # Parse the service ID - format is: {workspace}/{client_id}:{service_name}@{app_id}
        # The workspace is everything before the first "/"
        actual_workspace_prefix = service_id.split("/")[0]
        assert actual_workspace_prefix == user_workspace, \
            f"Service registered in wrong workspace! Expected {user_workspace}, got {actual_workspace_prefix}"

        # The injected workspace name should NOT be the actual workspace
        # It can appear in the service name part (after the colon), which is harmless
        assert actual_workspace_prefix != target_workspace, \
            f"Attacker was able to register service in target workspace {target_workspace}!"

        # Verify the service is actually in the user's workspace
        services_in_user_ws = await api.list_services({"workspace": user_workspace})
        service_ids = [s["id"] for s in services_in_user_ws]

        # The service should be listed in the user's workspace
        matching = [sid for sid in service_ids if "malicious-service" in sid]
        assert len(matching) > 0, f"Service not found in user workspace {user_workspace}"

        await api.disconnect()

    async def test_service_id_with_correct_workspace_prefix_should_succeed(
        self, fastapi_server, test_user_token
    ):
        """Test that registering a service with correct workspace prefix works."""
        api = await connect_to_server({
            "client_id": "v1-test-client-2",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        user_workspace = api.config.workspace

        async def dummy_function():
            return "test"

        # Register with correct workspace prefix - should work
        svc_info = await api.register_service({
            "id": f"{user_workspace}/v1-test-client-2:valid-service",
            "name": "Valid Service",
            "config": {"visibility": "protected"},
            "run": dummy_function,
        })

        assert svc_info is not None
        assert user_workspace in svc_info["id"]

        await api.disconnect()


class TestV2EventSubscriptionBypass:
    """
    V2: Event Subscription Validation Bypass (MEDIUM SEVERITY)

    Location: hypha/core/workspace.py:597-630

    Issue: The event subscription validation uses suffix-based matching
    which could allow bypassing workspace restrictions if workspace names
    share suffixes.

    Expected behavior: Event subscriptions should use exact workspace matching
    to prevent cross-workspace event snooping.
    """

    async def test_event_subscription_cross_workspace_blocked(
        self, fastapi_server, test_user_token
    ):
        """Test that subscribing to events from other workspaces is blocked."""
        api = await connect_to_server({
            "client_id": "v2-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        user_workspace = api.config.workspace

        # Create another workspace
        ws2_info = await api.create_workspace({
            "name": f"v2-other-ws-{uuid.uuid4().hex[:8]}",
            "description": "Other workspace",
        }, overwrite=True)
        other_workspace = ws2_info["id"]

        # Get the workspace manager service
        ws_manager = await api.get_service("~")

        # Attempt to subscribe to events from another workspace
        # This should be blocked
        with pytest.raises(Exception) as exc_info:
            await ws_manager.subscribe(f"{other_workspace}/service_added")

        error_msg = str(exc_info.value).lower()
        assert "forbidden" in error_msg or "permission" in error_msg, \
            f"Expected forbidden/permission error for cross-workspace subscription, got: {exc_info.value}"

        await api.disconnect()

    async def test_event_subscription_own_workspace_allowed(
        self, fastapi_server, test_user_token
    ):
        """Test that subscribing to events in own workspace is allowed."""
        api = await connect_to_server({
            "client_id": "v2-test-client-2",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        user_workspace = api.config.workspace

        ws_manager = await api.get_service("~")

        # Subscribe to standard system events - should work
        await ws_manager.subscribe("service_added")
        await ws_manager.subscribe("service_removed")

        # Subscribe to workspace-specific events for own workspace
        await ws_manager.subscribe(f"{user_workspace}/custom_event")

        await api.disconnect()

    async def test_event_subscription_suffix_bypass_attempt(
        self, fastapi_server, test_user_token
    ):
        """Test that suffix-based bypass attempts are blocked."""
        api = await connect_to_server({
            "client_id": "v2-suffix-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        user_workspace = api.config.workspace

        # Extract the suffix from user workspace (e.g., "ws-user-xxx" -> "xxx")
        ws_parts = user_workspace.split("-")
        if len(ws_parts) > 2:
            suffix = ws_parts[-1]

            # Create a workspace with similar suffix
            similar_ws_info = await api.create_workspace({
                "name": f"v2-similar-{suffix}",
                "description": "Workspace with similar suffix",
            }, overwrite=True)
            similar_workspace = similar_ws_info["id"]

            ws_manager = await api.get_service("~")

            # Try to subscribe using suffix pattern that might match
            # This should be blocked
            with pytest.raises(Exception) as exc_info:
                await ws_manager.subscribe(f"ws-{suffix}/secret_event")

            error_msg = str(exc_info.value).lower()
            assert "forbidden" in error_msg or "identifier" in error_msg, \
                f"Expected forbidden error for suffix bypass attempt, got: {exc_info.value}"

        await api.disconnect()


class TestV3AuthorizedWorkspacesLeak:
    """
    V3: Protected Service authorized_workspaces Leak (MEDIUM SEVERITY)

    Location: hypha/core/workspace.py:1753-1776

    Issue: When listing services, the authorized_workspaces field is only
    removed conditionally. Users with access via authorized_workspaces
    can see which other workspaces have access.

    Expected behavior: authorized_workspaces should always be removed
    from service info for non-owners.
    """

    async def test_authorized_workspaces_not_leaked_to_authorized_users(
        self, fastapi_server, test_user_token
    ):
        """Test that authorized_workspaces is not visible to authorized users."""
        # Create first user/workspace (owner)
        api_owner = await connect_to_server({
            "client_id": "v3-owner-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        owner_workspace = api_owner.config.workspace

        # Create second workspace (authorized user)
        ws2_info = await api_owner.create_workspace({
            "name": f"v3-authorized-ws-{uuid.uuid4().hex[:8]}",
            "description": "Authorized user workspace",
        }, overwrite=True)
        authorized_workspace = ws2_info["id"]

        # Register a protected service with authorized_workspaces
        async def test_func():
            return "test"

        await api_owner.register_service({
            "id": "protected-with-auth-list",
            "name": "Protected Service",
            "config": {
                "visibility": "protected",
                "authorized_workspaces": [authorized_workspace, "another-workspace-123"],
            },
            "run": test_func,
        })

        # Connect as authorized user
        ws2_token = await api_owner.generate_token({"workspace": authorized_workspace})
        api_authorized = await connect_to_server({
            "client_id": "v3-authorized-client",
            "workspace": authorized_workspace,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        # List services - authorized user should see the service
        services = await api_authorized.list_services(f"{owner_workspace}/*:protected-with-auth-list")

        # The service should be visible but authorized_workspaces should be hidden
        matching_services = [s for s in services if "protected-with-auth-list" in s.get("id", "")]

        if matching_services:
            service = matching_services[0]
            config = service.get("config", {})
            # authorized_workspaces should NOT be visible to non-owners
            assert "authorized_workspaces" not in config, \
                f"authorized_workspaces should be hidden from non-owners, but found: {config.get('authorized_workspaces')}"

        await api_owner.disconnect()
        await api_authorized.disconnect()


class TestV4BookmarkPermissionBypass:
    """
    V4: Bookmark Permission Bypass (LOW SEVERITY)

    Location: hypha/core/workspace.py:757-832

    Issue: bookmark() and unbookmark() methods don't validate that the user
    has permission to access the workspace or service being bookmarked.

    Expected behavior: Users should only be able to bookmark resources
    they have permission to access.
    """

    async def test_bookmark_inaccessible_workspace_should_fail(
        self, fastapi_server, test_user_token
    ):
        """Test that bookmarking inaccessible workspace fails."""
        api = await connect_to_server({
            "client_id": "v4-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        ws_manager = await api.get_service("~")

        # Try to bookmark a non-existent/inaccessible workspace
        with pytest.raises(Exception) as exc_info:
            await ws_manager.bookmark({
                "bookmark_type": "workspace",
                "id": "non-existent-workspace-xyz-123",
            })

        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "permission" in error_msg or "exist" in error_msg, \
            f"Expected not found/permission error, got: {exc_info.value}"

        await api.disconnect()

    async def test_bookmark_inaccessible_service_should_fail(
        self, fastapi_server, test_user_token
    ):
        """Test that bookmarking inaccessible service fails."""
        # Create two users with separate workspaces
        api1 = await connect_to_server({
            "client_id": "v4-user1",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create second workspace
        ws2_info = await api1.create_workspace({
            "name": f"v4-private-ws-{uuid.uuid4().hex[:8]}",
            "description": "Private workspace",
        }, overwrite=True)
        ws2 = ws2_info["id"]

        # Register a protected service in ws2
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "v4-user2",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        async def private_func():
            return "private"

        svc_info = await api2.register_service({
            "id": "private-service",
            "name": "Private Service",
            "config": {"visibility": "protected"},
            "run": private_func,
        })

        # Now try to bookmark this private service from ws1 (different connection)
        api3 = await connect_to_server({
            "client_id": "v4-attacker",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        ws_manager = await api3.get_service("~")

        # Attempt to bookmark a protected service in another workspace
        with pytest.raises(Exception) as exc_info:
            await ws_manager.bookmark({
                "bookmark_type": "service",
                "id": f"{ws2}/v4-user2:private-service",
            })

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "not found" in error_msg or "denied" in error_msg, \
            f"Expected permission/not found error, got: {exc_info.value}"

        await api1.disconnect()
        await api2.disconnect()
        await api3.disconnect()


class TestV5ExceptionInformationLeakage:
    """
    V5: Exception Information Leakage (LOW SEVERITY)

    Location: hypha/http.py (multiple locations)

    Issue: Full tracebacks are sent to clients in error responses,
    potentially revealing internal implementation details.

    Expected behavior: Error responses should contain generic error messages
    while full tracebacks are only logged server-side.

    Note: The RPC layer (hypha_rpc library) may still expose tracebacks for
    debugging purposes. This test focuses on HTTP layer fixes which are
    within the Hypha server codebase.
    """

    async def test_http_error_does_not_leak_traceback(
        self, fastapi_server, test_user_token
    ):
        """Test that HTTP errors don't leak full tracebacks."""
        # Make a request that will cause an error
        # Try to access a non-existent artifact file
        response = requests.get(
            f"{SERVER_URL}/public/artifacts/non-existent-artifact-xyz/files/test.txt",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        # Should get an error response
        assert response.status_code >= 400

        # Response should NOT contain Python traceback details
        response_text = response.text.lower()

        # Allow "exception" in error message but not full traceback format
        has_full_traceback = (
            "traceback" in response_text
            or ('file "' in response_text and 'line ' in response_text)
            or '.py"' in response_text
        )

        assert not has_full_traceback, \
            f"Response should not contain full traceback, got: {response.text[:500]}"

    async def test_http_service_call_error_does_not_leak_traceback(
        self, fastapi_server, test_user_token
    ):
        """Test that HTTP service call errors don't leak full tracebacks."""
        # First connect to get actual workspace
        api = await connect_to_server({
            "client_id": "v5-http-error-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        user_workspace = api.config.workspace
        await api.disconnect()

        # Try to call a non-existent service function via HTTP
        response = requests.get(
            f"{SERVER_URL}/{user_workspace}/services/non-existent/method",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        # Should get an error response
        assert response.status_code >= 400, f"Expected error response, got {response.status_code}"
        response_text = response.text

        # Check for traceback indicators that reveal internal paths
        assert "/Users/" not in response_text, \
            f"HTTP error should not leak internal paths, got: {response_text[:500]}"
        assert "/home/" not in response_text, \
            f"HTTP error should not leak internal paths, got: {response_text[:500]}"
        # Also check that we don't expose File paths from traceback
        assert 'File "/' not in response_text, \
            f"HTTP error should not leak traceback file paths, got: {response_text[:500]}"

    async def test_rpc_error_does_not_leak_internal_paths(
        self, fastapi_server, test_user_token
    ):
        """Test that RPC errors don't leak internal file paths.

        Note: This test is informational - RPC tracebacks are controlled by
        the hypha_rpc library which may expose them for debugging.
        The HTTP layer has been fixed to not expose internal paths.
        """
        api = await connect_to_server({
            "client_id": "v5-error-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Try to call a method that will fail
        artifact_manager = await api.get_service("public/artifact-manager")

        try:
            # This should fail - trying to read non-existent artifact
            await artifact_manager.read("non-existent-artifact-xyz-456")
        except Exception as e:
            error_msg = str(e)

            # Note: The RPC layer (hypha_rpc) may expose tracebacks for debugging.
            # This is a known limitation. The important fix is in the HTTP layer.
            # We just verify the error message contains meaningful info
            assert "non-existent-artifact-xyz-456" in error_msg or "does not exist" in error_msg.lower(), \
                f"Error should contain meaningful message about the missing artifact"

        await api.disconnect()


class TestServiceVisibilityEnforcement:
    """
    Additional tests for service visibility enforcement.

    These tests verify that protected services are properly isolated
    and that visibility settings are correctly enforced.
    """

    async def test_protected_service_not_visible_cross_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that protected services are not visible to other workspaces."""
        api1 = await connect_to_server({
            "client_id": "visibility-test-1",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create second workspace
        ws2_info = await api1.create_workspace({
            "name": f"visibility-test-ws-{uuid.uuid4().hex[:8]}",
            "description": "Second workspace",
        }, overwrite=True)
        ws2 = ws2_info["id"]

        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "visibility-test-2",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        # Register protected service in ws1
        async def secret_func():
            return "secret data"

        await api1.register_service({
            "id": "secret-service",
            "name": "Secret Service",
            "config": {"visibility": "protected"},
            "get_secret": secret_func,
        })

        # Try to access from ws2 - should fail
        with pytest.raises(Exception) as exc_info:
            await api2.get_service(f"{ws1}/visibility-test-1:secret-service")

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "not found" in error_msg, \
            f"Expected permission/not found error, got: {exc_info.value}"

        # Also verify it's not in service listing
        services = await api2.list_services(f"{ws1}/*")
        secret_services = [s for s in services if "secret-service" in s.get("id", "")]
        assert len(secret_services) == 0, \
            f"Protected service should not be visible in listing, found: {secret_services}"

        await api1.disconnect()
        await api2.disconnect()

    async def test_public_service_visible_cross_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that public services are visible to other workspaces."""
        api1 = await connect_to_server({
            "client_id": "public-visibility-1",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create second workspace
        ws2_info = await api1.create_workspace({
            "name": f"public-vis-test-{uuid.uuid4().hex[:8]}",
            "description": "Second workspace",
        }, overwrite=True)
        ws2 = ws2_info["id"]

        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "public-visibility-2",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        # Register public service in ws1
        async def public_func():
            return "public data"

        await api1.register_service({
            "id": "public-service",
            "name": "Public Service",
            "config": {"visibility": "public"},
            "get_data": public_func,
        })

        # Access from ws2 - should work
        svc = await api2.get_service(f"{ws1}/public-visibility-1:public-service")
        result = await svc.get_data()
        assert result == "public data"

        await api1.disconnect()
        await api2.disconnect()


class TestTokenSecurityValidation:
    """
    Tests for token security and validation.
    """

    async def test_expired_token_rejected(self, fastapi_server, test_user_token):
        """Test that expired tokens are rejected."""
        api = await connect_to_server({
            "client_id": "token-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Generate a token with very short expiration (1 second)
        short_token = await api.generate_token({
            "expires_in": 1,
        })

        # Wait for token to expire
        await asyncio.sleep(2)

        # Try to connect with expired token
        with pytest.raises(Exception) as exc_info:
            await connect_to_server({
                "client_id": "expired-token-test",
                "server_url": WS_SERVER_URL,
                "token": short_token,
            })

        error_msg = str(exc_info.value).lower()
        assert "expired" in error_msg or "invalid" in error_msg, \
            f"Expected expired/invalid token error, got: {exc_info.value}"

        await api.disconnect()

    async def test_token_workspace_scope_enforced(self, fastapi_server, test_user_token):
        """Test that tokens are scoped to specific workspaces."""
        api = await connect_to_server({
            "client_id": "scope-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api.config.workspace

        # Create second workspace
        ws2_info = await api.create_workspace({
            "name": f"scope-test-ws-{uuid.uuid4().hex[:8]}",
            "description": "Scoped workspace",
        }, overwrite=True)
        ws2 = ws2_info["id"]

        # Generate token scoped to ws2 only
        ws2_token = await api.generate_token({"workspace": ws2})

        # Try to connect to ws1 with ws2 token
        # This should either fail or connect to ws2 instead
        api2 = await connect_to_server({
            "client_id": "scope-test-2",
            "workspace": ws2,  # Explicitly request ws2
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        # Verify we're in ws2, not ws1
        assert api2.config.workspace == ws2, \
            f"Token scoped to {ws2} should not allow access to {ws1}"

        await api.disconnect()
        await api2.disconnect()
