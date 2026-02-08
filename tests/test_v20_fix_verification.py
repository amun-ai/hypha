"""
Tests to verify V20 fix is effective.

This test suite verifies that the fixes for V20 (Missing HTTP Input Validation)
properly prevent server crashes and path traversal attacks.

Tests verify:
1. Malformed query strings don't crash the server
2. Workspace names are properly validated
3. Path traversal is prevented
4. Service types are normalized correctly
"""

import pytest
import httpx
import uuid
from hypha_rpc import connect_to_server

# Constants
WS_SERVER_URL = "ws://127.0.0.1:9527/ws"
HTTP_SERVER_URL = "http://127.0.0.1:9527"

pytestmark = pytest.mark.asyncio


class TestV20FixQueryStringParsing:
    """Verify fix for query string parsing crash (V20)."""

    async def test_malformed_query_strings_do_not_crash_server(
        self, fastapi_server, test_user_token
    ):
        """
        Verify that malformed query strings are handled gracefully.

        Before fix: Server crashed with ValueError on queries like "?param_without_equals"
        After fix: Server handles gracefully with parse_qs
        """
        api = await connect_to_server({
            "client_id": f"v20-fix-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Create a simple ASGI service to test against
        async def test_asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"Test service OK",
                }

        svc = await api.register_service({
            "id": "v20-test-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": test_asgi_app,
        })

        service_id = svc["id"]

        # These query strings previously crashed the server
        malformed_queries = [
            "malformed",  # No "=" - this was the primary crash
            "key1&key2",  # Multiple params without "="
            "=value",  # Empty key
            "key1=value&malformed&key2=value",  # Mixed
            "a&b&c&d&e",  # Multiple malformed
            "",  # Empty query (should work)
        ]

        async with httpx.AsyncClient() as client:
            for query in malformed_queries:
                url_with_query = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"
                if query:
                    url_with_query += f"?{query}"

                try:
                    response = await client.get(url_with_query, timeout=5.0)

                    # Server should NOT crash - either process request or return error
                    # Status should NOT be 500 (Internal Server Error from crash)
                    assert response.status_code in [200, 400, 403, 404], \
                        f"Query '{query}' caused unexpected status {response.status_code}"

                    # If we got 500, check it's not the crash we're fixing
                    if response.status_code == 500:
                        error_text = response.text.lower()
                        assert "valueerror" not in error_text, \
                            f"Query '{query}' still causes ValueError crash"
                        assert "split" not in error_text, \
                            f"Query '{query}' still causes split error"

                except httpx.TimeoutException:
                    pytest.fail(f"Server hung on query '{query}' - may have crashed")
                except httpx.ConnectError:
                    pytest.fail(f"Server crashed on query '{query}'")

        await api.disconnect()

    async def test_mode_parameter_parsing_works_correctly(
        self, fastapi_server, test_user_token
    ):
        """Verify that _mode parameter is correctly extracted after fix."""
        api = await connect_to_server({
            "client_id": f"mode-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def test_asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"OK",
                }

        svc = await api.register_service({
            "id": "mode-test-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": test_asgi_app,
        })

        service_id = svc["id"]

        # Test valid _mode parameter
        async with httpx.AsyncClient() as client:
            url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test?_mode=default"
            response = await client.get(url)

            # Should process the request successfully
            assert response.status_code == 200, \
                f"Valid _mode parameter failed: {response.status_code}"

        await api.disconnect()


class TestV20FixWorkspaceValidation:
    """Verify fix for workspace name validation (V20)."""

    async def test_invalid_workspace_names_are_rejected(
        self, fastapi_server, test_user_token
    ):
        """
        Verify that workspace names with invalid characters are rejected.

        After fix: Workspace must match ^[a-zA-Z0-9_-]+$
        """
        api = await connect_to_server({
            "client_id": f"ws-val-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def test_asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"Should not be accessible",
                }

        svc = await api.register_service({
            "id": "ws-val-service",
            "type": "asgi",
            "config": {"visibility": "protected"},
            "serve": test_asgi_app,
        })

        service_id = svc["id"]

        # Invalid workspace names that should be rejected
        invalid_workspaces = [
            "../other-workspace",  # Path traversal
            "workspace/../admin",  # Directory traversal
            "workspace%2F..%2Fadmin",  # URL encoded traversal
            "workspace;admin",  # Semicolon
            "workspace||admin",  # Pipe characters
            "workspace/../../root",  # Deep traversal
            "workspace%00admin",  # Null byte
            "workspace<script>",  # XSS attempt
            "workspace'OR'1'='1",  # SQL injection attempt
        ]

        async with httpx.AsyncClient() as client:
            for invalid_ws in invalid_workspaces:
                url = f"{HTTP_SERVER_URL}/{invalid_ws}/apps/{service_id}/test"
                response = await client.get(url)

                # Should reject invalid workspace name with 400 Bad Request
                assert response.status_code in [400, 403, 404], \
                    f"Invalid workspace '{invalid_ws}' was not rejected (status: {response.status_code})"

                # Check error message mentions invalid workspace
                if response.status_code == 400:
                    assert b"workspace" in response.content.lower() or \
                           b"invalid" in response.content.lower(), \
                           f"Error message should mention invalid workspace"

        await api.disconnect()

    async def test_valid_workspace_names_are_accepted(
        self, fastapi_server, test_user_token
    ):
        """Verify that valid workspace names are accepted."""
        api = await connect_to_server({
            "client_id": f"ws-valid-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def test_asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"OK",
                }

        svc = await api.register_service({
            "id": "ws-valid-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": test_asgi_app,
        })

        service_id = svc["id"]

        # Valid workspace name (the actual workspace from token)
        async with httpx.AsyncClient() as client:
            url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"
            response = await client.get(url)

            # Valid workspace should work
            assert response.status_code == 200, \
                f"Valid workspace '{workspace}' was rejected"

        await api.disconnect()


class TestV20FixPathTraversalPrevention:
    """Verify fix for path traversal prevention (V20)."""

    async def test_path_traversal_is_blocked(
        self, fastapi_server, test_user_token
    ):
        """
        Verify that path traversal attempts are blocked.

        After fix: Paths are normalized and checked for ".."
        """
        api = await connect_to_server({
            "client_id": f"path-trav-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Track what paths the ASGI app receives
        received_paths = []

        async def path_echo_app(scope):
            """ASGI app that echoes back the path it received."""
            if scope["scope"]["type"] == "http":
                path = scope["scope"]["path"]
                received_paths.append(path)
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": f"Path: {path}".encode(),
                }

        svc = await api.register_service({
            "id": "path-echo-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": path_echo_app,
        })

        service_id = svc["id"]

        # Path traversal attempts that should be blocked
        traversal_paths = [
            "../../../etc/passwd",
            "legitimate/../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",  # URL encoded
            "....//....//etc//passwd",  # Double dots/slashes
            "./../admin",  # Relative traversal
            "test/../../root",  # Traversal in middle
        ]

        async with httpx.AsyncClient() as client:
            for trav_path in traversal_paths:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/{trav_path}"
                response = await client.get(url)

                # Should reject path traversal with 400 Bad Request
                assert response.status_code in [400, 403, 404], \
                    f"Path traversal '{trav_path}' was not rejected (status: {response.status_code})"

                # Check error message mentions path issue
                if response.status_code == 400:
                    content = response.content.lower()
                    assert b"path" in content or b"invalid" in content or b"traversal" in content, \
                        f"Error should mention path issue for '{trav_path}'"

        # Verify no traversal paths reached the ASGI app
        for received in received_paths:
            assert ".." not in received, \
                f"Path with '..' reached ASGI app: {received}"

        await api.disconnect()

    async def test_legitimate_paths_are_allowed(
        self, fastapi_server, test_user_token
    ):
        """Verify that legitimate paths (without traversal) are allowed."""
        api = await connect_to_server({
            "client_id": f"legit-path-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def test_asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"OK",
                }

        svc = await api.register_service({
            "id": "legit-path-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": test_asgi_app,
        })

        service_id = svc["id"]

        # Legitimate paths that should work
        legitimate_paths = [
            "index",
            "api/users",
            "api/users/123",
            "static/file.txt",
            "deeply/nested/path/file",
        ]

        async with httpx.AsyncClient() as client:
            for path in legitimate_paths:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/{path}"
                response = await client.get(url)

                # Legitimate paths should work
                assert response.status_code == 200, \
                    f"Legitimate path '{path}' was rejected: {response.status_code}"

        await api.disconnect()


class TestV19FixServiceTypeNormalization:
    """Verify fix for service type case normalization (V19)."""

    async def test_service_type_case_insensitive(
        self, fastapi_server, test_user_token
    ):
        """
        Verify that service types are normalized to lowercase.

        After fix: "ASGI", "asgi", "Asgi" all handled consistently
        """
        api = await connect_to_server({
            "client_id": f"type-norm-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def test_asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"ASGI OK",
                }

        # Test different case variations
        case_variations = ["ASGI", "asgi", "Asgi", "aSgI"]

        for type_variant in case_variations:
            try:
                svc = await api.register_service({
                    "id": f"type-test-{type_variant.lower()}",
                    "type": type_variant,
                    "config": {"visibility": "public"},
                    "serve": test_asgi_app,
                })

                service_id = svc["id"]

                # All variants should work consistently
                async with httpx.AsyncClient() as client:
                    url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"
                    response = await client.get(url)

                    assert response.status_code == 200, \
                        f"Service type '{type_variant}' not handled correctly: {response.status_code}"
                    assert b"ASGI OK" in response.content, \
                        f"Service type '{type_variant}' didn't invoke ASGI handler"

            except Exception as e:
                pytest.fail(f"Service type '{type_variant}' caused error: {e}")

        await api.disconnect()


class TestV20FixRegressionPrevention:
    """Verify that the fix doesn't break existing functionality."""

    async def test_normal_asgi_service_still_works(
        self, fastapi_server, test_user_token
    ):
        """Ensure fix doesn't break normal ASGI service functionality."""
        api = await connect_to_server({
            "client_id": f"regression-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def normal_asgi_app(scope):
            """Normal ASGI app that should continue working."""
            if scope["scope"]["type"] == "http":
                path = scope["scope"]["path"]
                query = scope["scope"].get("query_string", b"").decode()
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"application/json"]],
                    "body": f'{{"path": "{path}", "query": "{query}"}}'.encode(),
                }

        svc = await api.register_service({
            "id": "normal-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": normal_asgi_app,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # Test various normal requests
            test_cases = [
                ("/test", None, "/test"),
                ("/api/users", None, "/api/users"),
                ("/test", "key=value", "key=value"),
                ("/test", "key1=value1&key2=value2", "key1=value1&key2=value2"),
            ]

            for path, query, expected_query in test_cases:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}{path}"
                if query:
                    url += f"?{query}"

                response = await client.get(url)

                assert response.status_code == 200, \
                    f"Normal request failed: {path}?{query}"

                import json
                data = json.loads(response.text)
                assert data["path"] == path, \
                    f"Path mismatch: expected {path}, got {data['path']}"
                if expected_query:
                    assert data["query"] == expected_query, \
                        f"Query mismatch: expected {expected_query}, got {data['query']}"

        await api.disconnect()
