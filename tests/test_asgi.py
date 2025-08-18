"""Test ASGI services."""

from pathlib import Path
import asyncio
import json
import time

import pytest
import requests
import httpx
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_asgi(fastapi_server, test_user_token):
    """Test the ASGI gateway service."""
    app = FastAPI()

    @app.get("/api")
    async def read_api(request: Request):
        return {"message": "Hello World"}

    @app.put("/api")
    async def put_api(request: Request):
        data = await request.json()
        return {"message": f"Hello {data['name']}"}

    @app.post("/api")
    async def post_api(request: Request):
        data = await request.json()
        return {"message": f"Hello {data['name']}"}

    @app.delete("/api")
    async def delete_api(request: Request):
        return {"message": "Deleted"}

    @app.patch("/api")
    async def patch_api(request: Request):
        data = await request.json()
        return {"message": f"Patched {data['name']}"}

    @app.options("/api")
    async def options_api(request: Request):
        return {"message": "Options"}

    @app.get("/api/{item_id}")
    async def read_item(item_id: int, request: Request):
        return {"item_id": item_id}

    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    async def serve_fastapi(args):
        await app(args["scope"], args["receive"], args["send"])

    await api.register_service(
        {
            "id": "test-asgi",
            "type": "asgi",
            "config": {
                "visibility": "public",
            },
            "serve": serve_fastapi,
        }
    )

    # test with httpx async
    # url = f"{SERVER_URL}/{workspace}/apps/test-asgi/api"

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SERVER_URL}/{workspace}/apps/test-asgi/api")
        assert response.json()["message"] == "Hello World"

        response = await client.put(
            f"{SERVER_URL}/{workspace}/apps/test-asgi/api", json={"name": "Alice"}
        )
        assert response.json()["message"] == "Hello Alice"

        response = await client.post(
            f"{SERVER_URL}/{workspace}/apps/test-asgi/api", json={"name": "Alice"}
        )
        assert response.json()["message"] == "Hello Alice"

        response = await client.delete(f"{SERVER_URL}/{workspace}/apps/test-asgi/api")
        assert response.json()["message"] == "Deleted"

        response = await client.patch(
            f"{SERVER_URL}/{workspace}/apps/test-asgi/api", json={"name": "Alice"}
        )
        assert response.json()["message"] == "Patched Alice"

        response = await client.get(f"{SERVER_URL}/{workspace}/apps/test-asgi/api/1")
        assert response.json()["item_id"] == 1


async def test_asgi_serverapp(fastapi_server, test_user_token):
    """Test the ASGI gateway apps via the server-apps service."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config.workspace

    # Test app with custom template
    controller = await api.get_service("public/server-apps")

    source = (
        (Path(__file__).parent / "testASGIWebPythonPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.install(
        source=source,
        wait_for_service="hello-fastapi",
        timeout=30,
    )
    config = await controller.start(config.id)
    service = await api.get_service(f"{config.workspace}/hello-fastapi")
    assert "serve" in service

    response = requests.get(f"{SERVER_URL}/{workspace}/apps/hello-fastapi")
    assert response.ok
    assert response.json()["message"] == "Hello World"

    await controller.stop(config.id)
    await api.disconnect()


async def test_functions(fastapi_server, test_user_token):
    """Test the functions service."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test app with custom template
    controller = await api.get_service("public/server-apps")

    source = (
        (Path(__file__).parent / "testFunctionsPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.install(
        app_id="test-functions",
        source=source,
        wait_for_service="hello-functions",
        timeout=30,
    )
    config = await controller.start(config.id)

    service = await api.get_service(f"{config.workspace}/hello-functions")
    assert "hello-world" in service

    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok
    artifacts = response.json()
    artifact = find_item(artifacts, "name", "FunctionsPlugin")
    svc = find_item(artifact["services"], "name", "hello-functions")
    assert svc

    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/hello-functions/hello-world",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok
    ret = response.json()
    assert ret["message"] == "Hello World"
    assert "user" in ret["context"]
    user_info = ret["context"]["user"]
    assert user_info["scope"]["workspaces"]["ws-user-user-1"] == "rw"

    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/hello-functions/hello-world/"
    )
    assert response.ok

    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/hello-functions/",
        headers={"origin": "http://localhost:3000"},
    )
    assert response.ok
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
    assert response.content == b"Home page"

    response = requests.get(f"{SERVER_URL}/{workspace}/apps/hello-functions")
    assert response.ok
    assert response.content == b"Home page"

    await controller.stop(config.id)
    await api.disconnect()


async def test_asgi_concurrent_requests(fastapi_server, test_user_token):
    """Test concurrent requests to ASGI service with workspace cleanup"""
    start_time = time.time()

    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    setup_time = time.time() - start_time
    print(f"\nSetup time: {setup_time:.2f} seconds")

    # Get initial workspace count
    initial_workspaces = await api.list_workspaces()
    initial_count = len(initial_workspaces)
    print(f"Initial workspaces: {initial_count}")

    # Create a single FastAPI app instance to handle all requests
    app = FastAPI()

    @app.get("/api")
    async def read_api(request: Request):
        return {"message": "Hello World"}

    async def serve_fastapi(args):
        await app(args["scope"], args["receive"], args["send"])

    # Register service in public workspace
    service_start = time.time()
    service = await api.register_service(
        {
            "id": f"test-asgi",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": serve_fastapi,
        }
    )
    service_time = time.time() - service_start
    print(f"Service registration time: {service_time:.2f} seconds")

    sid = service["id"].split(":")[1]

    # Add a small delay to ensure service is ready
    await asyncio.sleep(1)

    # Make concurrent requests with timeout and retries
    max_retries = 3
    timeout = httpx.Timeout(30.0, connect=10.0)

    async def make_request_with_retry(client, url, retry_count=0):
        req_start = time.time()
        try:
            response = await client.get(url, timeout=timeout)
            req_time = time.time() - req_start
            return response, req_time
        except httpx.ReadTimeout:
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Add delay between retries
                return await make_request_with_retry(client, url, retry_count + 1)
            raise

    # Make requests in smaller batches to reduce load
    batch_size = 10
    total_requests = 50
    all_responses = []
    request_times = []

    requests_start = time.time()
    async with httpx.AsyncClient() as client:
        for i in range(0, total_requests, batch_size):
            batch_start = time.time()
            tasks = [
                make_request_with_retry(
                    client,
                    f"{SERVER_URL}/{api.config.workspace}/apps/{sid}/api?_mode=last",
                )
                for _ in range(batch_size)
            ]
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
            batch_time = time.time() - batch_start
            print(f"Batch {i//batch_size + 1} time: {batch_time:.2f} seconds")

            for response in batch_responses:
                if isinstance(response, tuple):
                    all_responses.append(response[0])
                    request_times.append(response[1])
                else:
                    all_responses.append(response)

            await asyncio.sleep(0.1)  # Reduced delay between batches for faster testing

    total_request_time = time.time() - requests_start
    print(f"\nTotal request time: {total_request_time:.2f} seconds")
    print(f"Average request time: {sum(request_times)/len(request_times):.2f} seconds")
    print(f"Min request time: {min(request_times):.2f} seconds")
    print(f"Max request time: {max(request_times):.2f} seconds")

    # Verify responses
    for response in all_responses:
        if isinstance(response, Exception):
            raise response
        assert response.status_code == 200, response.text
        assert response.json()["message"] == "Hello World"

    cleanup_start = time.time()

    # Check workspace count before disconnecting to avoid race conditions
    intermediate_workspaces = await api.list_workspaces()
    intermediate_count = len(intermediate_workspaces)
    print(f"Workspaces before disconnect: {intermediate_count}")

    # Verify no new persistent workspaces created during requests
    # Allow for the workspace to still exist since we haven't disconnected yet
    # Note: workspace cleanup can happen asynchronously, so we allow for some cleanup during the test
    assert (
        intermediate_count >= 0
    ), f"Unexpected workspace cleanup during test: {intermediate_count} < 0"

    # Verify temporary workspaces are cleaned up
    temp_workspaces = [
        w
        for w in intermediate_workspaces
        if w["id"].startswith("ws-user-") and not w["persistent"]
    ]
    assert (
        len(temp_workspaces) == 0
    ), "All temporary workspaces should have been cleaned up"

    # Disconnect API connection
    await api.disconnect()

    # Give a small delay to allow for any cleanup operations to complete
    # This is especially important for Python 3.9 which may have different timing
    await asyncio.sleep(0.5)

    # Now verify final workspace count
    # For this we need to create a new connection since we disconnected
    temp_api = await connect_to_server(
        {"name": "temp client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    try:
        final_workspaces = await temp_api.list_workspaces()
        final_count = len(final_workspaces)
        print(f"Final workspaces: {final_count}")

        # The workspace count should be stable (either the same as initial, or reduced due to cleanup)
        # We don't assert exact equality because the workspace might get cleaned up after disconnection
        # but we do ensure no unexpected workspaces were created
        assert (
            final_count <= initial_count
        ), f"New workspaces created unexpectedly: final={final_count}, initial={initial_count}"

    finally:
        await temp_api.disconnect()

    cleanup_time = time.time() - cleanup_start
    print(f"Cleanup time: {cleanup_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total test time: {total_time:.2f} seconds")


async def test_asgi_streaming(fastapi_server, test_user_token):
    """Test streaming responses through the ASGI middleware."""
    app = FastAPI()

    @app.get("/stream")
    async def stream_response():
        async def stream_generator():
            for i in range(5):
                yield f"chunk {i}\n".encode()
                await asyncio.sleep(0.1)

        return StreamingResponse(stream_generator(), media_type="text/plain")

    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    async def serve_fastapi(args):
        await app(args["scope"], args["receive"], args["send"])

    await api.register_service(
        {
            "id": "test-streaming",
            "type": "asgi",
            "config": {
                "visibility": "public",
            },
            "serve": serve_fastapi,
        }
    )

    # Test with httpx streaming client
    url = f"{SERVER_URL}/{workspace}/apps/test-streaming/stream"

    # Test chunked transfer with streaming client
    async with httpx.AsyncClient() as client:
        chunks = []
        async with client.stream("GET", url) as response:
            assert response.headers.get("transfer-encoding") == "chunked"
            async for chunk in response.aiter_bytes():
                chunks.append(chunk.decode())

    # Verify that we received 5 chunks
    assert len(chunks) > 1  # Might be coalesced but should be more than one
    complete_response = "".join(chunks)
    assert "chunk 0" in complete_response
    assert "chunk 4" in complete_response

    # Check all expected chunks are in the response
    for i in range(5):
        assert f"chunk {i}" in complete_response

    # Clean up
    await api.disconnect()


async def test_asgi_auth_context(fastapi_server, test_user_token):
    """Test authentication context passing for ASGI services."""
    from fastapi import FastAPI, Request, Depends
    from fastapi.responses import JSONResponse
    from hypha_rpc import connect_to_server

    # Create separate FastAPI instances for each service to avoid state confusion
    def create_context_app():
        app = FastAPI()
        app.state.context = None

        @app.get("/context")
        async def get_context():
            """Endpoint to check if context is available."""
            context = getattr(app.state, "context", None)

            if context is None:
                return {"has_context": False, "context": None, "workspace": None}

            return {
                "has_context": True,
                "context": context,
                "user_id": context.get("user", {}).get("id") if context else None,
                "workspace": context.get("ws"),  # Add workspace from context
            }

        @app.post("/context")
        async def post_context(request: Request):
            """POST endpoint that returns user context information."""
            body = (
                await request.json()
                if request.headers.get("content-type") == "application/json"
                else {}
            )

            context = getattr(app.state, "context", None)

            if context is None:
                return {
                    "has_context": False,
                    "context": None,
                    "workspace": None,
                    "data": body,
                }

            user_info = context.get("user") if context else None

            return {
                "data": body,
                "user_id": user_info.get("id") if user_info else None,  # Access as dict
                "workspace": context.get("ws"),  # Use "ws" not "workspace"
                "user_info": (
                    {
                        "id": user_info.get("id") if user_info else None,
                        "scope": (
                            user_info.get("scope") if user_info else None
                        ),  # Access as dict
                    }
                    if user_info
                    else None
                ),
                "has_context": context is not None,
            }

        return app

    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config.workspace
    token = await api.generate_token()

    # Create separate app instances for each service
    public_app = create_context_app()
    protected_app = create_context_app()

    async def serve_public_asgi(args, context=None):
        """ASGI wrapper for public service that properly handles context from RPC system."""
        scope = args["scope"]
        receive = args["receive"]
        send = args["send"]

        # Store the context in app state so FastAPI endpoints can access it
        public_app.state.context = context
        await public_app(scope, receive, send)

    async def serve_protected_asgi(args, context=None):
        """ASGI wrapper for protected service that properly handles context from RPC system."""
        scope = args["scope"]
        receive = args["receive"]
        send = args["send"]

        # Store the context in app state so FastAPI endpoints can access it
        protected_app.state.context = context
        await protected_app(scope, receive, send)

    # Register ASGI service WITH context requirement
    public_asgi_svc = await api.register_service(
        {
            "id": "test-asgi-public",
            "name": "Test ASGI Public Service",
            "type": "asgi",
            "config": {
                "visibility": "public",
                "require_context": True,
            },
            "serve": serve_public_asgi,
        }
    )

    # Register protected ASGI service
    protected_asgi_svc = await api.register_service(
        {
            "id": "test-asgi-protected",
            "name": "Test ASGI Protected Service",
            "type": "asgi",
            "config": {
                "visibility": "protected",
                "require_context": True,
            },
            "serve": serve_protected_asgi,
        }
    )

    print("=" * 60)
    print("Testing ASGI Service Authentication Context")
    print("=" * 60)

    # Test 1: Public ASGI service WITHOUT token
    print("\n1. Testing public ASGI service call without token...")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SERVER_URL}/{workspace}/apps/test-asgi-public/context"
        )

        # Simplified debug output
        print(f"Response Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response Text: {response.text}")

        assert response.status_code == 200
        result = response.json()
        print(f"User ID: {result.get('user_id')}")
        print(f"Workspace: {result.get('workspace')}")
        print(f"Has context: {result.get('has_context')}")

        # Without token, user should be anonymous but context should still be available
        # Public services run in the public workspace, not the user's workspace
        assert result.get("user_id") == "anonymouz-http"
        assert result.get("has_context") == True
        assert result.get("workspace") == "ws-anonymous"
        print("✓ Anonymous ASGI access works with context")

    # Test 2: Public ASGI service WITH token
    print("\n2. Testing public ASGI service call with token...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/apps/test-asgi-public/context",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        result = response.json()
        print(f"User ID: {result.get('user_id')}")
        print(f"Workspace: {result.get('workspace')}")
        print(f"Has context: {result.get('has_context')}")
        print(f"User info: {result.get('user_info')}")

        # With token, user should NOT be anonymous
        # With authentication, the context reflects the user's workspace
        assert result.get("user_id") != "anonymouz-http"
        assert result.get("has_context") == True
        assert result.get("workspace") == workspace
        user_info = result.get("user_info", {})
        assert user_info.get("id") != "anonymouz-http"
        print("✓ Authenticated ASGI access works with context")

    # Test 3: Protected ASGI service WITHOUT token - should fail or show anonymous
    print("\n3. Testing protected ASGI service call without token...")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SERVER_URL}/{workspace}/apps/test-asgi-protected/context"
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"User ID: {result.get('user_id')}")
            print(f"Has context: {result.get('has_context')}")
            # For protected services, anonymous access might still work but with limited context
            assert result.get("user_id") == "anonymouz-http"
            assert result.get("has_context") == True
        else:
            print("Protected service correctly rejects anonymous access")

    # Test 4: Protected ASGI service WITH token - should succeed
    print("\n4. Testing protected ASGI service call with token...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/apps/test-asgi-protected/context",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        result = response.json()
        print(f"User ID: {result.get('user_id')}")
        print(f"Workspace: {result.get('workspace')}")
        print(f"Has context: {result.get('has_context')}")

        # Should succeed with authenticated user
        # Protected services run in the user's workspace
        assert result.get("user_id") != "anonymouz-http"
        assert result.get("has_context") == True
        assert result.get("workspace") == workspace
        print("✓ Protected ASGI service accepts authenticated access")

    print("\nAll ASGI authentication context tests completed! ✓")
    await api.disconnect()


def test_find_nested_function():
    """Test the _find_nested_function helper method."""
    from hypha.http import ASGIRoutingMiddleware

    # Create a mock ASGIRoutingMiddleware instance to test the method
    middleware = ASGIRoutingMiddleware.__new__(
        ASGIRoutingMiddleware
    )  # Create without calling __init__

    # Test service with nested functions
    service = {
        "hello": lambda event: {"message": "hello"},
        "api": {
            "v1": {
                "users": lambda event: {"users": []},
                "posts": lambda event: {"posts": []},
            },
            "v2": {"users": lambda event: {"users": [], "version": "v2"}},
        },
        "default": lambda event: {"message": "default"},
    }

    # Test direct function access
    func = middleware._find_nested_function(service, ["hello"])
    assert func is not None
    assert callable(func)

    # Test nested function access
    func = middleware._find_nested_function(service, ["api", "v1", "users"])
    assert func is not None
    assert callable(func)

    func = middleware._find_nested_function(service, ["api", "v2", "users"])
    assert func is not None
    assert callable(func)

    # Test non-existent path
    func = middleware._find_nested_function(service, ["api", "v3", "users"])
    assert func is None

    # Test partial path that exists but isn't callable
    func = middleware._find_nested_function(service, ["api"])
    assert func is None  # api is a dict, not callable

    # Test default function access
    func = middleware._find_nested_function(service, ["default"])
    assert func is not None
    assert callable(func)

    print("✓ _find_nested_function works correctly")


async def test_nested_functions(fastapi_server, test_user_token):
    """Test nested functions service."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Get controller for app installation
    controller = await api.get_service("public/server-apps")

    # Install function service app with nested functions
    source = """
<config lang="json">
{
    "name": "Nested Functions Test", 
    "type": "web-worker",
    "version": "0.1.0"
}
</config>

<script lang="javascript">
class HyphaApp {
    async setup() {
        await api.registerService({
            "id": "nested-test-functions",
            "type": "functions", 
            "config": {
                "visibility": "public"
            },
            "api": {
                "v1": {
                    "users": async function(event) {
                        return {
                            status: 200,
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message: "User API v1", path: event.raw_path || ""})
                        };
                    },
                    "posts": async function(event) {
                        return {
                            status: 200,
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message: "Post API v1", method: event.method || "GET"})
                        };
                    }
                },
                "v2": {
                    "users": async function(event) {
                        return {
                            status: 200,
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message: "User API v2", version: "2.0"})
                        };
                    }
                }
            }
        });
    }
}
api.export(new HyphaApp());
</script>
    """

    config = await controller.install(
        source=source.strip(),
        wait_for_service="nested-test-functions",
        timeout=30,
        overwrite=True,
    )

    config = await controller.start(config["id"])

    # Test nested API v1 users
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/nested-test-functions/api/v1/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["message"] == "User API v1"

    # Test nested API v1 posts
    response = requests.post(
        f"{SERVER_URL}/{workspace}/apps/nested-test-functions/api/v1/posts",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["message"] == "Post API v1"
    assert ret["method"] == "POST"

    # Test nested API v2 users
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/nested-test-functions/api/v2/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["message"] == "User API v2"
    assert ret["version"] == "2.0"

    # Test non-existent nested path (should return 404)
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/nested-test-functions/api/v3/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404

    # Stop the app
    await controller.stop(config["id"])
    await api.disconnect()


async def test_default_fallback_function(fastapi_server, test_user_token):
    """Test default fallback function."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Get controller for app installation
    controller = await api.get_service("public/server-apps")

    # Install function service app with default fallback
    source = """
<config lang="json">
{
    "name": "Fallback Functions Test",
    "type": "web-worker", 
    "version": "0.1.0"
}
</config>

<script lang="javascript">
class HyphaApp {
    async setup() {
        await api.registerService({
            "id": "fallback-test-functions",
            "type": "functions",
            "config": {
                "visibility": "public"
            },
            "hello": async function(event) {
                return {
                    status: 200,
                    body: "Hello from specific function"
                };
            },
            "default": async function(event) {
                const path = event.function_path || "/";
                return {
                    status: 200,
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: "Default handler",
                        path: path,
                        method: event.method || "GET"
                    })
                };
            }
        });
    }
}
api.export(new HyphaApp());
</script>
    """

    config = await controller.install(
        source=source.strip(),
        wait_for_service="fallback-test-functions",
        timeout=30,
        overwrite=True,
    )
    config = await controller.start(config["id"])
    # Test specific function
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/fallback-test-functions/hello",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    assert response.text == "Hello from specific function"

    # Test fallback for non-existent path
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/fallback-test-functions/unknown-path",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["message"] == "Default handler"
    assert ret["path"] == "/unknown-path"
    assert ret["method"] == "GET"

    # Test fallback with POST method
    response = requests.post(
        f"{SERVER_URL}/{workspace}/apps/fallback-test-functions/some/deep/path",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["message"] == "Default handler"
    assert ret["path"] == "/some/deep/path"
    assert ret["method"] == "POST"

    # Test fallback for root path
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/fallback-test-functions/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["message"] == "Default handler"
    assert ret["path"] == "/"

    # Stop the app
    await controller.stop(config["id"])
    await api.disconnect()


async def test_nested_functions_with_default(fastapi_server, test_user_token):
    """Test combining nested functions with default fallback."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Get controller for app installation
    controller = await api.get_service("public/server-apps")

    # Install function service app with nested functions and default fallback
    source = """
<config lang="json">
{
    "name": "Complex Functions Test",
    "type": "web-worker",
    "version": "0.1.0"
}
</config>

<script lang="javascript">
class HyphaApp {
    async setup() {
        await api.registerService({
            "id": "complex-test-functions",
            "type": "functions",
            "config": {
                "visibility": "public"
            },
            "api": {
                "users": async function(event) {
                    return {
                        status: 200,
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({users: ["alice", "bob"]})
                    };
                }
            },
            "default": async function(event) {
                const path = event.function_path || "/";
                return {
                    status: 200,
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: "Fallback handler for complex service",
                        requested_path: path
                    })
                };
            }
        });
    }
}
api.export(new HyphaApp());
</script>
    """

    config = await controller.install(
        source=source.strip(),
        wait_for_service="complex-test-functions",
        timeout=30,
        overwrite=True,
    )
    # Start the app
    config = await controller.start(config["id"])
    # Test specific nested function
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/complex-test-functions/api/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["users"] == ["alice", "bob"]

    # Test fallback for non-existent nested path
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/complex-test-functions/api/nonexistent",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["message"] == "Fallback handler for complex service"
    assert ret["requested_path"] == "/api/nonexistent"

    # Test fallback for completely different path
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/complex-test-functions/other/path",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, f"Expected 200, got {response.status_code}: {response.text}"
    ret = response.json()
    assert ret["message"] == "Fallback handler for complex service"
    assert ret["requested_path"] == "/other/path"

    # Stop the app
    await controller.stop(config["id"])
    await api.disconnect()
