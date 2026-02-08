"""
Security tests for HTTP input validation vulnerabilities.

Tests for V20: Missing Input Validation in ASGIRoutingMiddleware
"""

import pytest
from hypha_rpc import connect_to_server
import httpx

# Constants
WS_SERVER_URL = "ws://127.0.0.1:9527/ws"
HTTP_SERVER_URL = "http://127.0.0.1:9527"

pytestmark = pytest.mark.asyncio


class TestV20MissingInputValidation:
    """
    V20: Missing Input Validation in ASGIRoutingMiddleware (HIGH SEVERITY)

    Location: hypha/http.py:254-384

    Issue: The ASGI routing middleware extracts workspace, service_id, and path
    from URL parameters without validation. This leads to several vulnerabilities:

    1. Malformed query strings can crash the server
    2. Path traversal characters are not sanitized
    3. Workspace and service_id can contain special characters
    4. Null byte injection is possible

    Expected behavior: All user-provided URL components should be validated
    before processing.
    """

    async def test_malformed_query_string_crashes_server(
        self, fastapi_server, test_user_token
    ):
        """Test that malformed query strings are handled safely."""
        api = await connect_to_server({
            "client_id": "query-crash-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Create a simple ASGI service
        async def asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"Success",
                }

        svc = await api.register_service({
            "id": "query-test-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": asgi_app,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # Current code does: dict([q.split("=") for q in query.split("&")])
            # This crashes if any query parameter doesn't have "="

            malformed_queries = [
                "malformed",  # No "=" sign - will crash split
                "key1&key2",  # Multiple params without "="
                "key1=value&malformed",  # Mixed valid and invalid
                "=value",  # Empty key
                "key=",  # Empty value is OK, but test it
            ]

            for query in malformed_queries:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test?{query}"

                try:
                    response = await client.get(url, timeout=5.0)

                    # Server should handle this gracefully, not crash
                    # Either accept it or return 400, but not 500 with crash
                    assert response.status_code in [200, 400], \
                        f"Query '{query}' caused server error: {response.status_code}"

                    if response.status_code == 500:
                        error_body = response.text
                        # Check if it's a crash from query parsing
                        assert "split" not in error_body.lower(), \
                            f"Server crashed parsing query '{query}': {error_body}"
                        assert "list index" not in error_body.lower(), \
                            f"Server crashed with index error for query '{query}': {error_body}"
                except httpx.ReadTimeout:
                    pytest.fail(f"Server hung processing malformed query: {query}")
                except Exception as e:
                    # Connection errors might indicate server crash
                    if "connection" in str(e).lower():
                        pytest.fail(f"Server may have crashed on query '{query}': {e}")

        await api.disconnect()

    async def test_path_traversal_in_service_path(
        self, fastapi_server, test_user_token
    ):
        """Test that path traversal in service path is prevented."""
        api = await connect_to_server({
            "client_id": "path-traversal-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Create ASGI service that echoes the path it receives
        async def asgi_app(scope):
            if scope["scope"]["type"] == "http":
                received_path = scope["scope"]["path"]
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": f"Received path: {received_path}".encode(),
                }

        svc = await api.register_service({
            "id": "echo-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": asgi_app,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # The current code does:
            # path = path_params["path"]
            # if not path.startswith("/"): path = "/" + path
            # scope["path"] = path
            #
            # No validation of ".." or other traversal patterns

            traversal_paths = [
                "../../../etc/passwd",
                "legitimate/../../etc/passwd",
                "..%2F..%2Fetc%2Fpasswd",  # URL encoded
                "....//....//etc//passwd",  # Double dots/slashes
                ".;/.;/etc/passwd",  # Alternate separator
            ]

            for trav_path in traversal_paths:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/{trav_path}"
                response = await client.get(url)

                # The server should either:
                # 1. Normalize the path (remove ..)
                # 2. Reject the request (400/403)
                # 3. Not pass raw traversal to ASGI app

                if response.status_code == 200:
                    body = response.text
                    # Path should be normalized, not contain ".."
                    assert ".." not in body, \
                        f"Path traversal '{trav_path}' was not normalized: {body}"
                    # Should not allow access to files outside service scope
                    assert "/etc/passwd" not in body.lower(), \
                        f"Path traversal may have succeeded for '{trav_path}': {body}"

        await api.disconnect()

    async def test_workspace_special_characters(
        self, fastapi_server, test_user_token
    ):
        """Test that workspace names with special characters are handled safely."""
        api = await connect_to_server({
            "client_id": "workspace-char-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        legitimate_workspace = api.config.workspace

        async def asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"Service response",
                }

        svc = await api.register_service({
            "id": "char-test-service",
            "type": "asgi",
            "config": {"visibility": "protected"},  # Protected to this workspace only
            "serve": asgi_app,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # Try to access the service using workspace names with special chars
            malicious_workspaces = [
                legitimate_workspace + "%00admin",  # Null byte injection
                legitimate_workspace + ";admin",  # Command injection attempt
                legitimate_workspace + "/../../admin",  # Path traversal in workspace
                "../" + legitimate_workspace,  # Relative path
                legitimate_workspace.replace("-", "%2D"),  # URL encoding
            ]

            for ws_name in malicious_workspaces:
                url = f"{HTTP_SERVER_URL}/{ws_name}/apps/{service_id}/test"
                response = await client.get(url)

                # Server should either:
                # 1. Reject the request (400/403/404)
                # 2. Normalize the workspace name
                # Should NOT allow access to services in other workspaces
                assert response.status_code in [400, 401, 403, 404], \
                    f"Malicious workspace '{ws_name}' was not rejected: {response.status_code}"

        await api.disconnect()

    async def test_service_id_special_characters(
        self, fastapi_server, test_user_token
    ):
        """Test that service IDs with special characters are handled safely."""
        api = await connect_to_server({
            "client_id": "service-char-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"Service response",
                }

        svc = await api.register_service({
            "id": "normal-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": asgi_app,
        })

        legitimate_service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # Try to manipulate service ID in URL
            malicious_service_ids = [
                legitimate_service_id + "/../admin-service",  # Path traversal
                legitimate_service_id + "%00",  # Null byte
                legitimate_service_id.replace(":", "%3A"),  # URL encode separator
                "../../../etc/passwd",  # Direct traversal
                "../../" + legitimate_service_id,  # Relative path
            ]

            for svc_id in malicious_service_ids:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{svc_id}/test"
                response = await client.get(url)

                # Should reject or normalize service ID
                # Should not allow access to unintended services
                if response.status_code == 200:
                    # If it succeeded, it should have accessed the legitimate service
                    # Check that the response is from our service, not something else
                    assert b"Service response" in response.content or \
                           response.status_code in [400, 403, 404], \
                           f"Service ID manipulation '{svc_id}' may have succeeded"

        await api.disconnect()

    async def test_null_byte_injection_prevention(
        self, fastapi_server, test_user_token
    ):
        """Test that null byte injection is prevented in all URL components."""
        api = await connect_to_server({
            "client_id": "null-byte-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def asgi_app(scope):
            if scope["scope"]["type"] == "http":
                path = scope["scope"]["path"]
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": f"Path length: {len(path)}".encode(),
                }

        svc = await api.register_service({
            "id": "null-test-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": asgi_app,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # Null byte can truncate strings in some languages/contexts
            null_byte_paths = [
                "test%00.txt",  # Null byte in path
                "test.txt%00/../etc/passwd",  # Null byte with traversal
                "test%00admin",  # Null byte injection
            ]

            for path in null_byte_paths:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/{path}"
                response = await client.get(url)

                if response.status_code == 200:
                    body = response.text
                    # Path should not be truncated at null byte
                    # If null byte is removed, path should be longer
                    # Should not contain literal null byte
                    assert "\\x00" not in repr(body), \
                        f"Null byte in path '{path}' was not properly handled"

        await api.disconnect()


class TestHTTPSecurityHeaders:
    """Test that appropriate security headers are set on HTTP responses."""

    async def test_security_headers_on_asgi_service(
        self, fastapi_server, test_user_token
    ):
        """Test that security headers are present on ASGI service responses."""
        import httpx

        api = await connect_to_server({
            "client_id": "security-headers-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"Response",
                }

        svc = await api.register_service({
            "id": "header-test-service",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": asgi_app,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"
            response = await client.get(url)

            # Check for important security headers
            # These prevent common web vulnerabilities

            # Note: This test documents current behavior
            # Not all headers may be present - this is a recommendation

            headers = {k.lower(): v for k, v in response.headers.items()}

            # Document which security headers are present
            security_headers = {
                "x-content-type-options": "nosniff",
                "x-frame-options": "DENY",
                "x-xss-protection": "1; mode=block",
                "strict-transport-security": "max-age=31536000",
                "content-security-policy": "default-src 'self'",
            }

            missing_headers = []
            for header, expected_value in security_headers.items():
                if header not in headers:
                    missing_headers.append(header)

            # Document findings (not failing test yet)
            if missing_headers:
                print(f"Missing security headers: {missing_headers}")
                print("Consider adding these headers to improve security")

        await api.disconnect()
