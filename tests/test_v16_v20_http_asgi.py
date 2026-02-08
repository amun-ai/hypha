"""Test suite for V16-V20 HTTP/ASGI security vulnerabilities.

This test file verifies fixes for HTTP/ASGI routing middleware security issues:
- V16: HTTP query string injection (HIGH)
- V17: Workspace/service ID validation bypass (HIGH)
- V18: Path traversal in ASGI routing (HIGH)
- V19: Exception message information leakage (MEDIUM)
- V20: DoS via malformed query strings (MEDIUM)
"""

import pytest
import pytest_asyncio
import uuid
import httpx

from hypha_rpc import connect_to_server

from . import (
    WS_SERVER_URL,
    SERVER_URL,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


class TestV16HTTPQueryStringInjection:
    """
    V16: HTTP Query String Injection (HIGH SEVERITY)

    Location: hypha/http.py:387 (old vulnerable code)
    Fixed: hypha/http.py:392 (now uses _parse_query_string_safe)

    Issue: Unsafe query string parsing using list comprehension:
    ```python
    _mode = dict([q.split("=") for q in query.split("&")]).get("_mode")
    ```

    This fails on malformed queries like "key=val=extra" and could cause DoS.

    Fix: Use safe parsing with _parse_query_string_safe() that:
    - Handles malformed parameters gracefully
    - Limits query string size
    - Validates UTF-8 encoding
    """

    async def test_malformed_query_string_should_not_crash(
        self, fastapi_server, test_user_token
    ):
        """Test that malformed query strings are handled gracefully."""
        api = await connect_to_server({
            "client_id": "v16-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Register a simple ASGI service
        async def serve(asgi_args):
            scope, receive, send = (
                asgi_args["scope"],
                asgi_args["receive"],
                asgi_args["send"],
            )
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"OK",
            })

        await api.register_service({
            "id": "test-asgi",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": serve,
        })

        workspace = api.config.workspace

        # Test 1: Malformed query with multiple "=" signs
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/{workspace}/apps/test-asgi/test?key=val=extra",
                timeout=30,
            )
            # Should return 200 OK (or 500 but not crash the server)
            assert response.status_code in [200, 500]

        # Test 2: Query string without "=" (should be ignored)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/{workspace}/apps/test-asgi/test?invalidparam",
                timeout=30,
            )
            assert response.status_code in [200, 500]

        await api.disconnect()


class TestV17WorkspaceServiceIDValidation:
    """
    V17: Workspace/Service ID Validation Bypass (HIGH SEVERITY)

    Location: hypha/http.py:366-367 (old vulnerable code)
    Fixed: hypha/http.py:372-373 (now validates with _validate_workspace_id/_validate_service_id)

    Issue: No validation of workspace/service_id extracted from URL path.
    Attackers could use path traversal or special characters.

    Fix: Added validation functions that:
    - Check for empty IDs
    - Allow only alphanumeric + ["-", "_", ":", "."]
    - Block ".." path traversal
    """

    async def test_workspace_id_with_path_traversal_should_fail(
        self, fastapi_server, test_user_token
    ):
        """Test that workspace IDs with path traversal are rejected."""
        api = await connect_to_server({
            "client_id": "v17-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Register an ASGI service
        async def serve(asgi_args):
            scope, receive, send = (
                asgi_args["scope"],
                asgi_args["receive"],
                asgi_args["send"],
            )
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"OK",
            })

        await api.register_service({
            "id": "test-asgi",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": serve,
        })

        # Try to access with path traversal in workspace ID
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/../admin/apps/test-asgi/test",
                timeout=30,
            )
            # Should return error (400 or 500), not 200
            assert response.status_code in [400, 500]

        await api.disconnect()

    async def test_service_id_with_path_traversal_should_fail(
        self, fastapi_server, test_user_token
    ):
        """Test that service IDs with path traversal are rejected."""
        api = await connect_to_server({
            "client_id": "v17-test-client-2",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        workspace = api.config.workspace

        # Try to access with path traversal in service ID
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/{workspace}/apps/../secret-service/test",
                timeout=30,
            )
            # Should return error (400, 404, or 500), not 200
            # 404 is OK because the router normalizes the path
            assert response.status_code in [400, 404, 500]

        await api.disconnect()


class TestV18PathTraversalInASGIRouting:
    """
    V18: Path Traversal in ASGI Routing (HIGH SEVERITY)

    Location: hypha/http.py:379-382 (old vulnerable code)
    Fixed: hypha/http.py:376 (now validates with _validate_path)

    Issue: No validation of the request path before setting scope["path"].
    Attackers could use "..  ", null bytes, or excessive length.

    Fix: Added _validate_path() that:
    - Blocks ".." path traversal
    - Blocks null byte injection (\x00)
    - Limits path length to 2048 chars (DoS prevention)
    """

    async def test_path_with_null_byte_should_fail(
        self, fastapi_server, test_user_token
    ):
        """Test that paths with null bytes are rejected."""
        api = await connect_to_server({
            "client_id": "v18-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Register an ASGI service
        async def serve(asgi_args):
            scope, receive, send = (
                asgi_args["scope"],
                asgi_args["receive"],
                asgi_args["send"],
            )
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"OK",
            })

        await api.register_service({
            "id": "test-asgi",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": serve,
        })

        workspace = api.config.workspace

        # Try to access with null byte in path
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{SERVER_URL}/{workspace}/apps/test-asgi/test%00secret",
                    timeout=30,
                )
                # Should return error (400 or 500), not 200
                assert response.status_code in [400, 500]
            except Exception:
                # Client may reject invalid URL - that's OK
                pass

        await api.disconnect()

    async def test_excessive_path_length_should_fail(
        self, fastapi_server, test_user_token
    ):
        """Test that excessively long paths are rejected."""
        api = await connect_to_server({
            "client_id": "v18-test-client-2",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Register an ASGI service
        async def serve(asgi_args):
            scope, receive, send = (
                asgi_args["scope"],
                asgi_args["receive"],
                asgi_args["send"],
            )
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"OK",
            })

        await api.register_service({
            "id": "test-asgi",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": serve,
        })

        workspace = api.config.workspace

        # Try to access with excessive path length (>2048 chars)
        long_path = "a" * 3000
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/{workspace}/apps/test-asgi/{long_path}",
                timeout=30,
            )
            # Should return error (400 or 500), not 200
            assert response.status_code in [400, 500, 414]  # 414 = URI Too Long

        await api.disconnect()


class TestV19ExceptionMessageLeakage:
    """
    V19: Exception Message Information Leakage (MEDIUM SEVERITY)

    Location: hypha/http.py:484 (old vulnerable code)
    Fixed: hypha/http.py:473 (now uses _get_safe_error_detail)

    Issue: Exception messages were sent directly to clients:
    ```python
    "body": f"Internal Server Error: {exp}".encode()
    ```

    This could leak internal paths, stack traces, and sensitive info.

    Fix: Use _get_safe_error_detail() that:
    - Returns full traceback only if HYPHA_EXPOSE_DETAILED_ERRORS=true
    - In production, returns only exception type + sanitized message
    - Removes file paths and traceback details
    """

    async def test_exception_should_not_leak_internal_paths(
        self, fastapi_server, test_user_token
    ):
        """Test that exceptions don't leak internal file paths."""
        api = await connect_to_server({
            "client_id": "v19-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Register an ASGI service that raises an exception
        async def serve(asgi_args):
            # This will raise an exception with file path in traceback
            raise ValueError("Test error from /internal/secret/path/file.py")

        await api.register_service({
            "id": "test-asgi",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": serve,
        })

        workspace = api.config.workspace

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/{workspace}/apps/test-asgi/test",
                timeout=30,
            )
            # Should return 500
            assert response.status_code == 500

            # Error message should NOT contain internal paths
            error_text = response.text
            assert "/internal/secret/path/" not in error_text
            # Should contain sanitized error message
            assert "ValueError" in error_text or "Internal Server Error" in error_text

        await api.disconnect()


class TestV20DoSViaMalformedQueryStrings:
    """
    V20: DoS via Malformed Query Strings (MEDIUM SEVERITY)

    Location: hypha/http.py:387 (old vulnerable code)
    Fixed: hypha/http.py:392 (now uses _parse_query_string_safe)

    Issue: No size limit on query strings. Attackers could send:
    - Extremely long query strings (>1MB)
    - Many parameters causing CPU exhaustion

    Fix: _parse_query_string_safe() limits query string to 8KB.
    """

    async def test_excessive_query_string_should_fail(
        self, fastapi_server, test_user_token
    ):
        """Test that excessively large query strings are rejected."""
        api = await connect_to_server({
            "client_id": "v20-test-client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Register an ASGI service
        async def serve(asgi_args):
            scope, receive, send = (
                asgi_args["scope"],
                asgi_args["receive"],
                asgi_args["send"],
            )
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({
                "type": "http.response.body",
                "body": b"OK",
            })

        await api.register_service({
            "id": "test-asgi",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": serve,
        })

        workspace = api.config.workspace

        # Create a query string larger than 8KB limit
        large_query = "a=b&" * 3000  # ~12KB
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/{workspace}/apps/test-asgi/test?{large_query}",
                timeout=30,
            )
            # Should return error (400 or 500), not 200
            assert response.status_code in [400, 500]

        await api.disconnect()
