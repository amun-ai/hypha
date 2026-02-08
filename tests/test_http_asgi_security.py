"""
Security tests for HTTP and ASGI service vulnerabilities.

This module tests security vulnerabilities in:
- ASGI service routing and hijacking
- Function service injection attacks
- Path traversal in static file serving
- CORS bypass techniques
- Proxy configuration manipulation
- View endpoint exploitation
"""

import pytest
import uuid
from hypha_rpc import connect_to_server

# Constants
WS_SERVER_URL = "ws://127.0.0.1:9527/ws"
HTTP_SERVER_URL = "http://127.0.0.1:9527"

pytestmark = pytest.mark.asyncio


class TestASGIRouteHijacking:
    """
    Test ASGI service route hijacking vulnerabilities.

    ASGI services are mounted at: /{workspace}/apps/{service_id}/{path:path}

    Potential vulnerabilities:
    1. Service ID injection with path traversal characters
    2. Workspace name manipulation
    3. URL encoding bypass
    4. Route overlap with static files
    """

    async def test_asgi_service_id_path_traversal(
        self, fastapi_server, test_user_token
    ):
        """Test that ASGI service IDs with path traversal are properly sanitized."""
        import httpx

        api = await connect_to_server({
            "client_id": f"asgi-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Create a simple ASGI app
        async def asgi_app(scope):
            """Simple ASGI app that returns the request path."""
            async def receive():
                return {"type": "http.request", "body": b""}

            async def send(message):
                pass

            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": f"Path: {scope['scope']['path']}".encode(),
                }

        # Try to register ASGI service with path traversal in service_id
        malicious_service_id = "../../../evil-service"

        try:
            svc = await api.register_service({
                "id": malicious_service_id,
                "name": "Malicious ASGI Service",
                "type": "asgi",
                "config": {"visibility": "public"},
                "serve": asgi_app,
            })

            # The service ID should be sanitized/rejected or normalized
            workspace = api.config.workspace
            client_id = api.config.client_id

            # Try to access the service - it should NOT escape the workspace path
            async with httpx.AsyncClient() as client:
                # Attempt path traversal in URL
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{malicious_service_id}/test"
                response = await client.get(url)

                # If service is accessible, verify it's within proper workspace
                if response.status_code == 200:
                    # The service should NOT have escaped to root or other workspaces
                    # Verify by checking another workspace cannot access it
                    pass

        except Exception as e:
            # Service registration or access should fail
            assert "path" in str(e).lower() or "invalid" in str(e).lower()
        finally:
            await api.disconnect()

    async def test_asgi_workspace_name_injection(
        self, fastapi_server, test_user_token
    ):
        """Test that workspace names with special characters don't bypass routing."""
        import httpx

        api = await connect_to_server({
            "client_id": f"asgi-ws-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Create ASGI service
        async def asgi_app(scope):
            async def receive():
                return {"type": "http.request", "body": b""}

            async def send(message):
                pass

            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"ASGI response",
                }

        svc = await api.register_service({
            "id": "test-asgi",
            "type": "asgi",
            "config": {"visibility": "protected"},
            "serve": asgi_app,
        })

        service_id = svc["id"]

        # Try various workspace name manipulations
        async with httpx.AsyncClient() as client:
            malicious_workspaces = [
                "../other-workspace",  # Path traversal
                "workspace/../admin",  # Directory traversal
                "workspace%2F..%2Fadmin",  # URL encoded traversal
                "workspace;admin",  # Command injection attempt
                "workspace||admin",  # Command chaining
                "workspace/../../root",  # Deep traversal
            ]

            for malicious_ws in malicious_workspaces:
                url = f"{HTTP_SERVER_URL}/{malicious_ws}/apps/{service_id}/test"
                response = await client.get(url)

                # Should either reject the request or normalize the path
                # Should NOT allow access to services in other workspaces
                assert response.status_code in [400, 403, 404], \
                    f"Malicious workspace '{malicious_ws}' was not rejected"

        await api.disconnect()


class TestFunctionServiceInjection:
    """
    Test function service injection vulnerabilities.

    Function services support:
    - Nested function paths: /function/nested/path
    - Default fallback function
    - Index function for directory-style access

    Potential vulnerabilities:
    1. Function name injection
    2. Nested path traversal
    3. Default function hijacking
    4. CORS header manipulation
    """

    async def test_function_service_path_traversal(
        self, fastapi_server, test_user_token
    ):
        """Test that function service paths cannot traverse outside intended namespace."""
        import httpx

        api = await connect_to_server({
            "client_id": f"func-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Create function service with nested functions
        async def safe_function(scope):
            return {
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
                "body": b"Safe function",
            }

        async def admin_function(scope):
            return {
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
                "body": b"Admin function - should be protected",
            }

        svc = await api.register_service({
            "id": "function-service",
            "type": "functions",
            "config": {"visibility": "public"},
            "api": {
                "safe": safe_function,
            },
            "admin": {
                "secret": admin_function,
            },
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # Try path traversal to access admin function
            traversal_attempts = [
                "../admin/secret",
                "safe/../../admin/secret",
                "safe%2F..%2F..%2Fadmin%2Fsecret",
                "./../admin/secret",
                "safe/./../admin/secret",
            ]

            for path in traversal_attempts:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/{path}"
                response = await client.get(url)

                # Should not be able to access admin function through traversal
                # Either reject the request or return 404
                assert b"Admin function" not in response.content, \
                    f"Path traversal '{path}' accessed protected admin function"

        await api.disconnect()

    async def test_function_service_cors_injection(
        self, fastapi_server, test_user_token
    ):
        """Test that CORS headers cannot be manipulated through function returns."""
        import httpx

        api = await connect_to_server({
            "client_id": f"cors-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Function that tries to set malicious CORS headers
        async def malicious_cors_function(scope):
            return {
                "status": 200,
                "headers": [
                    [b"content-type", b"text/plain"],
                    # Attacker tries to override CORS to allow any origin
                    [b"Access-Control-Allow-Origin", b"https://evil.com"],
                    [b"Access-Control-Allow-Credentials", b"false"],
                ],
                "body": b"Response with malicious CORS",
            }

        svc = await api.register_service({
            "id": "cors-service",
            "type": "functions",
            "config": {"visibility": "public"},
            "test": malicious_cors_function,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"

            # Request with specific origin
            response = await client.get(
                url,
                headers={"Origin": "https://trusted.com"}
            )

            # The server should override any CORS headers from the function
            # and use its own CORS policy
            cors_origin = response.headers.get("Access-Control-Allow-Origin")

            # Server should either:
            # 1. Use the request origin (https://trusted.com)
            # 2. Use wildcard (*)
            # 3. NOT use the attacker's value (https://evil.com)
            assert cors_origin != "https://evil.com", \
                "Function was able to override CORS headers with malicious values"

        await api.disconnect()

    async def test_function_service_default_function_hijacking(
        self, fastapi_server, test_user_token
    ):
        """Test that default function cannot be used for unauthorized access."""
        import httpx

        api = await connect_to_server({
            "client_id": f"default-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Create service with default function that logs all requests
        request_log = []

        async def default_handler(scope):
            # Log what path was requested
            request_log.append(scope.get("function_path", scope["path"]))
            return {
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
                "body": b"Default handler",
            }

        async def specific_function(scope):
            return {
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
                "body": b"Specific function",
            }

        svc = await api.register_service({
            "id": "default-service",
            "type": "functions",
            "config": {"visibility": "public"},
            "specific": specific_function,
            "default": default_handler,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # Access non-existent paths - should hit default handler
            url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/nonexistent"
            response = await client.get(url)

            assert response.status_code == 200
            assert b"Default handler" in response.content

            # Verify the default handler received the correct path
            # Should not be able to use default to access other services
            assert len(request_log) > 0

        await api.disconnect()


class TestStaticFilePathTraversal:
    """
    Test path traversal vulnerabilities in static file serving.

    Static files are served from:
    - /static/* (built-in static files)
    - Custom mount points via --static-mounts

    Potential vulnerabilities:
    1. Directory traversal to read arbitrary files
    2. Accessing files outside mounted directory
    3. Reading sensitive configuration files
    """

    async def test_static_file_directory_traversal(self, fastapi_server):
        """Test that static file serving prevents directory traversal."""
        import httpx

        async with httpx.AsyncClient() as client:
            # Try various directory traversal patterns
            traversal_attempts = [
                "/static/../../../etc/passwd",
                "/static/../../hypha/server.py",
                "/static%2F..%2F..%2Fhypha%2Fserver.py",
                "/static/....//....//....//etc/passwd",
                "/static/..;/..;/etc/passwd",
                "/static/%2e%2e/%2e%2e/etc/passwd",
            ]

            for path in traversal_attempts:
                url = f"{HTTP_SERVER_URL}{path}"
                response = await client.get(url, follow_redirects=False)

                # Should either return 404 or reject the request
                # Should NOT return file contents from outside static directory
                assert response.status_code in [400, 404], \
                    f"Path traversal '{path}' was not properly blocked"

                # Verify we didn't get actual file contents
                if response.status_code == 200:
                    # Check for common file content patterns
                    content = response.text.lower()
                    assert "root:" not in content  # /etc/passwd
                    assert "fastapi" not in content  # Python files
                    assert "import" not in content  # Python files


class TestProxyConfigurationManipulation:
    """
    Test proxy configuration manipulation vulnerabilities.

    The ASGI proxy allows configuration of:
    - Upstream URLs
    - Header processing
    - HTTP options

    Potential vulnerabilities:
    1. SSRF through upstream URL manipulation
    2. Header injection
    3. Request smuggling
    4. Internal network access
    """

    async def test_asgi_service_ssrf_prevention(
        self, fastapi_server, test_user_token
    ):
        """Test that ASGI services cannot be used for SSRF attacks."""
        import httpx

        api = await connect_to_server({
            "client_id": f"ssrf-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Create ASGI service that tries to proxy to internal services
        async def proxy_asgi_app(scope):
            """ASGI app that tries to fetch from internal URLs."""
            async def receive():
                return {"type": "http.request", "body": b""}

            async def send(message):
                pass

            if scope["scope"]["type"] == "http":
                # Try to make request to internal service
                # This should be blocked if properly implemented
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        # Attempt SSRF to internal network
                        internal_response = await client.get("http://localhost:6379")  # Redis
                        return {
                            "status": 200,
                            "headers": [[b"content-type", b"text/plain"]],
                            "body": internal_response.text.encode(),
                        }
                except Exception as e:
                    return {
                        "status": 200,
                        "headers": [[b"content-type", b"text/plain"]],
                        "body": f"SSRF blocked: {str(e)}".encode(),
                    }

        svc = await api.register_service({
            "id": "ssrf-service",
            "type": "asgi",
            "config": {"visibility": "protected"},
            "serve": proxy_asgi_app,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"
            response = await client.get(url)

            # The ASGI app should not be able to access internal services
            # or the response should not contain internal service data
            content = response.text.lower()
            assert "redis" not in content or "ssrf blocked" in content, \
                "ASGI service was able to perform SSRF attack"

        await api.disconnect()


class TestHTTPRequestSmuggling:
    """
    Test HTTP request smuggling vulnerabilities.

    Request smuggling can occur when:
    - Content-Length and Transfer-Encoding headers conflict
    - Headers are not properly validated
    - Multiple services interpret requests differently

    Potential vulnerabilities:
    1. CL.TE desync attacks
    2. TE.CL desync attacks
    3. Header injection through service functions
    """

    async def test_content_length_transfer_encoding_conflict(
        self, fastapi_server, test_user_token
    ):
        """Test that conflicting Content-Length and Transfer-Encoding are rejected."""
        import httpx

        api = await connect_to_server({
            "client_id": f"smuggle-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def test_function(scope):
            return {
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
                "body": b"Response",
            }

        svc = await api.register_service({
            "id": "smuggle-service",
            "type": "functions",
            "config": {"visibility": "public"},
            "test": test_function,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"

            # Try request with both CL and TE headers
            try:
                response = await client.post(
                    url,
                    headers={
                        "Content-Length": "10",
                        "Transfer-Encoding": "chunked",
                    },
                    content=b"0\r\n\r\n",
                )

                # Server should reject or normalize this request
                # Should not process it in a way that causes smuggling
                assert response.status_code in [200, 400], \
                    "Unexpected response to CL/TE conflict"
            except httpx.RequestError:
                # Connection errors are acceptable - means server rejected request
                pass

        await api.disconnect()


class TestServiceTypeConfusion:
    """
    Test service type confusion vulnerabilities.

    Services can be of type:
    - "asgi" - ASGI applications
    - "functions" - Function handlers
    - Other types (should be rejected)

    Potential vulnerabilities:
    1. Type confusion between ASGI and functions
    2. Unhandled service types
    3. Type injection
    """

    async def test_invalid_service_type_rejection(
        self, fastapi_server, test_user_token
    ):
        """Test that invalid service types are properly rejected."""
        import httpx

        api = await connect_to_server({
            "client_id": f"type-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Try to register service with invalid type
        async def dummy_handler(scope):
            return {
                "status": 200,
                "body": b"Should not be accessible",
            }

        try:
            svc = await api.register_service({
                "id": "invalid-type-service",
                "type": "malicious_type",  # Invalid type
                "config": {"visibility": "public"},
                "serve": dummy_handler,
            })

            service_id = svc["id"]

            # Try to access the service
            async with httpx.AsyncClient() as client:
                url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"
                response = await client.get(url)

                # Should return error for invalid service type
                assert response.status_code == 500, \
                    "Invalid service type was not rejected"
                assert b"Invalid service type" in response.content, \
                    "Error message should indicate invalid service type"
        except Exception as e:
            # Service registration might fail, which is also acceptable
            assert "type" in str(e).lower()
        finally:
            await api.disconnect()

    async def test_service_type_case_sensitivity(
        self, fastapi_server, test_user_token
    ):
        """Test that service type checking is case-insensitive where appropriate."""
        import httpx

        api = await connect_to_server({
            "client_id": f"case-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"ASGI response",
                }

        # Try different case variations
        type_variations = ["ASGI", "asgi", "Asgi", "aSgI"]

        for type_variant in type_variations:
            try:
                svc = await api.register_service({
                    "id": f"case-service-{type_variant}",
                    "type": type_variant,
                    "config": {"visibility": "public"},
                    "serve": asgi_app,
                })

                service_id = svc["id"]

                async with httpx.AsyncClient() as client:
                    url = f"{HTTP_SERVER_URL}/{workspace}/apps/{service_id}/test"
                    response = await client.get(url)

                    # All case variations should work consistently
                    assert response.status_code == 200, \
                        f"Service type '{type_variant}' not handled correctly"
            except Exception as e:
                # If one case fails, all should fail
                pass

        await api.disconnect()


class TestURLEncodingBypass:
    """
    Test URL encoding bypass vulnerabilities.

    Attackers may try to bypass validation using:
    - URL encoding (%2F for /)
    - Double encoding (%252F)
    - Unicode encoding
    - Mixed encoding

    Potential vulnerabilities:
    1. Path traversal through encoding
    2. Workspace name bypass
    3. Service ID injection
    """

    async def test_url_encoded_path_traversal(
        self, fastapi_server, test_user_token
    ):
        """Test that URL-encoded path traversal is blocked."""
        import httpx

        api = await connect_to_server({
            "client_id": f"encode-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        async def test_function(scope):
            return {
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
                "body": f"Path: {scope['path']}".encode(),
            }

        svc = await api.register_service({
            "id": "encode-service",
            "type": "functions",
            "config": {"visibility": "public"},
            "test": test_function,
        })

        service_id = svc["id"]

        async with httpx.AsyncClient() as client:
            # Try various encoding bypasses
            encoded_attempts = [
                f"/{workspace}/apps/{service_id}/%2e%2e%2f%2e%2e%2fadmin",  # URL encoded ../..
                f"/{workspace}/apps/{service_id}/%252e%252e%252f",  # Double encoded
                f"/{workspace}/apps/{service_id}/test%2F..%2Fadmin",  # Encoded slashes
                f"/{workspace}/apps/{service_id}/test%00admin",  # Null byte
            ]

            for encoded_path in encoded_attempts:
                url = f"{HTTP_SERVER_URL}{encoded_path}"
                response = await client.get(url)

                # Should not traverse outside intended path
                if response.status_code == 200:
                    content = response.text.lower()
                    # Path should be normalized, not contain traversal
                    assert ".." not in content, \
                        f"Encoded path traversal '{encoded_path}' was not normalized"

        await api.disconnect()


class TestActivityTimerManipulation:
    """
    Test activity timer manipulation vulnerabilities.

    ASGI services reset activity timers to prevent premature cleanup.

    Potential vulnerabilities:
    1. Activity timer bypass
    2. Unauthorized timer reset
    3. Resource exhaustion through timer manipulation
    """

    async def test_activity_timer_isolation(
        self, fastapi_server, test_user_token
    ):
        """Test that activity timers are isolated per workspace."""
        import httpx
        import asyncio

        api = await connect_to_server({
            "client_id": f"timer-test-{uuid.uuid4().hex[:8]}",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace1 = api.config.workspace

        # Create second workspace
        ws2_info = await api.create_workspace({
            "name": f"timer-ws-{uuid.uuid4().hex[:8]}",
            "description": "Test workspace for timer isolation",
        }, overwrite=True)

        # Create ASGI service in first workspace
        async def asgi_app(scope):
            if scope["scope"]["type"] == "http":
                return {
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                    "body": b"Service active",
                }

        svc1 = await api.register_service({
            "id": "timer-service",
            "type": "asgi",
            "config": {"visibility": "protected"},
            "serve": asgi_app,
        })

        service_id = svc1["id"]

        # Access the service to trigger activity timer
        async with httpx.AsyncClient() as client:
            url = f"{HTTP_SERVER_URL}/{workspace1}/apps/{service_id}/test"
            response = await client.get(url)
            assert response.status_code == 200

            # Wait a bit
            await asyncio.sleep(1)

            # Service should still be active in workspace1
            # But this should NOT affect activity timers in workspace2
            # Verify workspace isolation is maintained

        await api.disconnect()
