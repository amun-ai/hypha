"""Test HTTP cross-workspace access with generic and workspace-scoped tokens.

This test module validates that:
1. Generic tokens (no wid: in scope, e.g. Auth0 cookies) can access services
   in workspaces the user has permission for via the HTTP URL.
2. Workspace-scoped tokens (with wid: in scope) are restricted to their
   designated workspace.
3. Security: unauthorized cross-workspace access is properly denied.
"""

import asyncio
import pytest
import requests
from hypha_rpc import connect_to_server

from hypha.core import UserInfo, UserPermission, auth
from hypha.core.auth import generate_auth_token, create_scope

from . import WS_SERVER_URL, SERVER_URL

pytestmark = pytest.mark.asyncio


async def _generate_generic_token(user_id, email=None, roles=None):
    """Generate a generic token without wid: (simulates Auth0 token behavior).

    Generic tokens have workspace permissions but no current_workspace set,
    mimicking how Auth0 tokens are parsed (Auth0 scope is like 'openid
    profile email' which doesn't contain wid: prefixes).
    """
    user_info = UserInfo(
        id=user_id,
        is_anonymous=False,
        email=email or f"{user_id}@test.com",
        parent=None,
        roles=roles or [],
        # No current_workspace — this is what makes it "generic"
        scope=create_scope(
            workspaces={f"ws-user-{user_id}": UserPermission.admin}
        ),
        expires_at=None,
    )
    return await generate_auth_token(user_info, 3600)


def _http_post(url, json_data, token=None, cookie_token=None):
    """Make a blocking HTTP POST (to be used with run_in_executor)."""
    headers = {}
    cookies = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if cookie_token:
        cookies["access_token"] = cookie_token
    return requests.post(url, json=json_data, headers=headers, cookies=cookies)


async def test_generic_token_cross_workspace_http_access(
    minio_server, fastapi_server, test_user_token
):
    """Test that a generic token can access a protected service in another
    workspace the user owns, via the HTTP URL.

    This is the core scenario: a user has an Auth0 cookie (generic token)
    and visits https://server/workspace-B/services/svc/method where
    workspace-B is a workspace they own but is not their personal workspace.
    """
    loop = asyncio.get_event_loop()

    # Connect as user-1 to their personal workspace
    api = await connect_to_server(
        {
            "name": "owner-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )

    # Create a separate workspace owned by user-1
    target_ws = await api.create_workspace(
        {
            "name": "cross-ws-test",
            "description": "Test workspace for cross-workspace HTTP access",
        }
    )
    target_ws_name = target_ws["id"] if isinstance(target_ws, dict) else target_ws

    # Connect to the target workspace and register a protected service
    api2 = await connect_to_server(
        {
            "name": "service-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
            "workspace": target_ws_name,
        }
    )
    svc = await api2.register_service(
        {
            "id": "test-protected-svc",
            "name": "test-protected-svc",
            "config": {
                "visibility": "protected",
                "require_context": True,
            },
            "echo": lambda data, context=None: {
                "data": data,
                "ws": context.get("ws") if context else None,
            },
        }
    )
    service_id = svc["id"].split("/")[-1]

    # Generate a generic token (no wid:) for user-1
    generic_token = await _generate_generic_token("user-1")

    # Access the protected service via HTTP URL using the generic token.
    # This should succeed because user-1 owns the target workspace.
    url = f"{SERVER_URL}/{target_ws_name}/services/{service_id}/echo"
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(url, {"data": "hello-cross-workspace"}, token=generic_token),
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    result = response.json()
    assert result["data"] == "hello-cross-workspace"
    assert result["ws"] == target_ws_name

    # Also test with cookie (simulating browser access)
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(
            url, {"data": "hello-via-cookie"}, cookie_token=generic_token
        ),
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    result = response.json()
    assert result["data"] == "hello-via-cookie"
    assert result["ws"] == target_ws_name

    await api2.disconnect()
    await api.disconnect()


async def test_generic_token_access_public_workspace_services(
    minio_server, fastapi_server, test_user_token
):
    """Test that a generic token (Auth0 cookie) can access public services
    in the 'public' workspace via HTTP.

    This reproduces the login flow bug where:
    1. User authenticates via Auth0, gets a generic token (no wid:)
    2. Auth0 callback redirects to /public/services/hypha-login/report
    3. The HTTP handler checks workspace permissions for 'public'
    4. update_user_scope doesn't grant permission for 'public' to non-owners
    5. User gets 403 Forbidden instead of the report succeeding

    The 'public' workspace should be accessible by all authenticated users.
    """
    loop = asyncio.get_event_loop()

    # Generate a generic token (no wid:) for user-1 — simulates Auth0 cookie
    generic_token = await _generate_generic_token("user-1")

    # The 'public' workspace always exists and has the hypha-login service.
    # Try to access a public service in the 'public' workspace via HTTP GET.
    # Using the 'check' endpoint as a simple GET test (it will return an error
    # about invalid key, but NOT a 403 permission error).
    url = f"{SERVER_URL}/public/services/hypha-login/check?key=nonexistent-test-key&timeout=1"
    response = await loop.run_in_executor(
        None,
        lambda: requests.get(
            url,
            headers={"Authorization": f"Bearer {generic_token}"},
        ),
    )
    # The key "nonexistent-test-key" won't exist, so we expect a 400 (bad request)
    # from the check_login function — NOT a 403 permission denied.
    # If we get 403, it means the HTTP handler is blocking access to the 'public'
    # workspace for authenticated users with generic tokens.
    assert response.status_code != 403, (
        f"Got 403 when accessing public workspace service with generic token. "
        f"This means the HTTP handler incorrectly denies access to the 'public' "
        f"workspace for authenticated users. Response: {response.text}"
    )
    # Should be 400 (invalid key) not 403 (permission denied)
    assert response.status_code == 400, (
        f"Expected 400 (invalid key) but got {response.status_code}: {response.text}"
    )


async def test_generic_token_access_public_workspace_info(
    minio_server, fastapi_server, test_user_token
):
    """Test that a generic token (Auth0 cookie) can access the workspace
    info endpoint for the 'public' workspace.

    This tests the GET /{workspace}/info endpoint which had the same
    missing update_user_scope bug as the service_function handler.
    """
    loop = asyncio.get_event_loop()

    # Generate a generic token (no wid:) for user-1 — simulates Auth0 cookie
    generic_token = await _generate_generic_token("user-1")

    # Access the public workspace info endpoint
    url = f"{SERVER_URL}/public/info"
    response = await loop.run_in_executor(
        None,
        lambda: requests.get(
            url,
            headers={"Authorization": f"Bearer {generic_token}"},
        ),
    )
    # Should return workspace info, NOT 403
    assert response.status_code != 403, (
        f"Got 403 when accessing /public/info with generic token. "
        f"The endpoint incorrectly denies access to the 'public' workspace "
        f"for authenticated users. Response: {response.text}"
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data.get("id") == "public"


async def test_workspace_scoped_token_restricted_to_own_workspace(
    minio_server, fastapi_server, test_user_token
):
    """Test that a workspace-scoped token (with wid:) cannot access protected
    services in a different workspace via the HTTP URL.

    Workspace-scoped tokens are generated with a fixed workspace and should
    only work within that workspace for protected services.
    """
    loop = asyncio.get_event_loop()

    # Connect as user-1
    api = await connect_to_server(
        {
            "name": "scoped-token-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )
    user_workspace = api.config["workspace"]

    # Create a target workspace
    target_ws = await api.create_workspace(
        {
            "name": "scoped-token-target-ws",
            "description": "Target workspace for scoped token test",
        }
    )
    target_ws_name = target_ws["id"] if isinstance(target_ws, dict) else target_ws

    # Register services in the target workspace
    api2 = await connect_to_server(
        {
            "name": "svc-provider",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
            "workspace": target_ws_name,
        }
    )
    public_svc = await api2.register_service(
        {
            "id": "test-public-svc",
            "name": "test-public-svc",
            "config": {"visibility": "public"},
            "echo": lambda data: data,
        }
    )
    public_service_id = public_svc["id"].split("/")[-1]

    protected_svc = await api2.register_service(
        {
            "id": "test-protected-svc-scoped",
            "name": "test-protected-svc-scoped",
            "config": {"visibility": "protected"},
            "echo": lambda data: data,
        }
    )
    protected_service_id = protected_svc["id"].split("/")[-1]

    # Generate a workspace-scoped token for user-1's personal workspace
    scoped_token = await api.generate_token()

    # Public service should work with scoped token (visibility check skips
    # workspace mismatch for public services)
    url = f"{SERVER_URL}/{target_ws_name}/services/{public_service_id}/echo"
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(url, {"data": "scoped-public"}, token=scoped_token),
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )

    # Protected service with scoped token for DIFFERENT workspace should fail.
    # The scoped token creates an RPC in the user's personal workspace,
    # causing workspace mismatch for the protected service.
    url = f"{SERVER_URL}/{target_ws_name}/services/{protected_service_id}/echo"
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(url, {"data": "should-fail"}, token=scoped_token),
    )
    # Should fail — workspace mismatch for protected service
    assert response.status_code != 200 or (
        response.json().get("success") is False
    ), f"Expected failure but got: {response.text}"

    await api2.disconnect()
    await api.disconnect()


async def test_generic_token_no_permission_denied(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """Test that a generic token CANNOT access a workspace the user does
    NOT have permission for.

    Security test: ensure that just putting a workspace name in the URL
    doesn't grant access — the user must actually have permission.
    """
    loop = asyncio.get_event_loop()

    # User-1 creates a workspace and registers a protected service
    api = await connect_to_server(
        {
            "name": "ws-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )
    target_ws = await api.create_workspace(
        {
            "name": "private-ws-test",
            "description": "Private workspace for security test",
        }
    )
    target_ws_name = target_ws["id"] if isinstance(target_ws, dict) else target_ws

    api2 = await connect_to_server(
        {
            "name": "svc-provider",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
            "workspace": target_ws_name,
        }
    )
    svc = await api2.register_service(
        {
            "id": "secret-svc",
            "name": "secret-svc",
            "config": {"visibility": "protected"},
            "echo": lambda data: data,
        }
    )
    service_id = svc["id"].split("/")[-1]

    # User-2 generates a generic token and tries to access user-1's workspace
    generic_token_user2 = await _generate_generic_token("user-2")

    url = f"{SERVER_URL}/{target_ws_name}/services/{service_id}/echo"
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(url, {"data": "unauthorized"}, token=generic_token_user2),
    )
    # Should be denied — user-2 has no permission for user-1's workspace
    assert response.status_code == 403, (
        f"Expected 403 but got {response.status_code}: {response.text}"
    )
    result = response.json()
    assert result.get("success") is False
    assert "permission" in result.get("detail", "").lower() or "denied" in result.get(
        "detail", ""
    ).lower(), f"Expected permission error, got: {result}"

    await api2.disconnect()
    await api.disconnect()


async def test_generic_token_nonexistent_workspace(
    minio_server, fastapi_server, test_user_token
):
    """Test that accessing a nonexistent workspace via HTTP returns 404."""
    loop = asyncio.get_event_loop()
    generic_token = await _generate_generic_token("user-1")

    url = f"{SERVER_URL}/nonexistent-workspace-xyz/services/some-svc/echo"
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(url, {"data": "test"}, token=generic_token),
    )
    assert response.status_code == 404, (
        f"Expected 404 but got {response.status_code}: {response.text}"
    )
    result = response.json()
    assert result.get("success") is False
    assert "not found" in result.get("detail", "").lower()


async def test_generic_token_same_workspace_still_works(
    minio_server, fastapi_server, test_user_token
):
    """Test that a generic token accessing its own workspace via HTTP
    still works correctly (regression test).
    """
    loop = asyncio.get_event_loop()

    # Connect as user-1 to their personal workspace
    api = await connect_to_server(
        {
            "name": "same-ws-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )
    user_workspace = api.config["workspace"]

    # Register a protected service
    svc = await api.register_service(
        {
            "id": "my-protected-svc",
            "name": "my-protected-svc",
            "config": {"visibility": "protected"},
            "echo": lambda data: data,
        }
    )
    service_id = svc["id"].split("/")[-1]

    # Access with generic token — URL workspace matches the user's workspace
    generic_token = await _generate_generic_token("user-1")

    url = f"{SERVER_URL}/{user_workspace}/services/{service_id}/echo"
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(url, {"data": "same-workspace"}, token=generic_token),
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    assert response.json() == "same-workspace"

    await api.disconnect()


async def test_workspace_scoped_token_same_workspace_works(
    minio_server, fastapi_server, test_user_token
):
    """Test that a workspace-scoped token accessing its own workspace
    works correctly (regression test).
    """
    loop = asyncio.get_event_loop()

    api = await connect_to_server(
        {
            "name": "scoped-same-ws",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )
    user_workspace = api.config["workspace"]

    svc = await api.register_service(
        {
            "id": "my-scoped-svc",
            "name": "my-scoped-svc",
            "config": {"visibility": "protected"},
            "echo": lambda data: data,
        }
    )
    service_id = svc["id"].split("/")[-1]

    # Generate a workspace-scoped token for user-1's workspace
    scoped_token = await api.generate_token()

    url = f"{SERVER_URL}/{user_workspace}/services/{service_id}/echo"
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(url, {"data": "scoped-same-workspace"}, token=scoped_token),
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    assert response.json() == "scoped-same-workspace"

    await api.disconnect()


async def test_anonymous_user_public_service_cross_workspace(
    minio_server, fastapi_server, test_user_token
):
    """Test that anonymous users (no token) can still access public services
    via HTTP URLs.
    """
    loop = asyncio.get_event_loop()

    api = await connect_to_server(
        {
            "name": "anon-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )
    user_workspace = api.config["workspace"]

    svc = await api.register_service(
        {
            "id": "public-svc-anon",
            "name": "public-svc-anon",
            "config": {"visibility": "public"},
            "echo": lambda data: data,
        }
    )
    service_id = svc["id"].split("/")[-1]

    # Access without any token
    url = f"{SERVER_URL}/{user_workspace}/services/{service_id}/echo"
    response = await loop.run_in_executor(
        None,
        lambda: _http_post(url, {"data": "anonymous-test"}),
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    assert response.json() == "anonymous-test"

    await api.disconnect()
