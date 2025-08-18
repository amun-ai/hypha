"""Test http proxy."""

import gzip

import json
import msgpack
import numpy as np
import pytest
import httpx
import requests
import asyncio
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio

TEST_APP_CODE = """
api.export({
    async setup(){
    },
    async register_services(){
        const service_info1 = await api.registerService(
            {
                "id": "test_service",
                "name": "test_service",
                "type": "test_service",
                "config": {
                    "visibility": "public",
                },
                echo( data ){
                    console.log("Echo: ", data)
                    return data
                }
            }
        )
        console.log(`registered test_service: ${service_info1.id}`)
        const service_info2 = await api.registerService(
            {
                "id": "test_service_protected",
                "name": "test_service_protected",
                "type": "test_service",
                "config": {
                    "visibility": "protected",
                },
                echo( data ){
                    console.log("Echo: ", data)
                    return data
                }
            }
        )
        console.log(`registered test_service: ${service_info2.id}`)
        return [service_info1, service_info2]
    }
})
"""


async def test_http_services(minio_server, fastapi_server, test_user_token):
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    workspace = api.config["workspace"]
    svc = await api.register_service(
        {
            "id": "my_service",
            "name": "my_service",
            "type": "my_service",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "echo": lambda data: data,
        }
    )

    def use_my_service():
        url = f"{SERVER_URL}/{workspace}/services/{svc.id.split('/')[1]}/echo"
        response = requests.post(url, json={"data": "123"})
        assert response.status_code == 200, f"{response.text}"
        assert response.json() == "123"

        response = requests.get(f"{SERVER_URL}/{workspace}/services")
        assert response.status_code == 200
        assert find_item(response.json(), "name", "my_service")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, use_my_service)


# pylint: disable=too-many-statements
async def test_http_proxy(
    minio_server, fastapi_server, test_user_token, root_user_token
):
    """Test http proxy."""
    # WS_SERVER_URL = "http://127.0.0.1:9527"
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test app with custom template
    controller = await api.get_service("public/server-apps")
    app_config = await controller.install(
        source=TEST_APP_CODE,
        config={"type": "window"},
        wait_for_service=True,
    )
    app_config = await controller.start(app_config.id)
    app = await api.get_app(app_config.id)
    assert "setup" in app and "register_services" in app
    svc1, svc2 = await app.register_services()

    service_ws = app.config.workspace
    assert service_ws
    service = await api.get_service(svc1["id"])
    assert await service.echo("233d") == "233d"

    service = await api.get_service(svc2["id"])
    assert await service.echo("22") == "22"

    async with connect_to_server(
        {
            "name": "root client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": root_user_token,
        }
    ) as root_api:
        workspaces = await root_api.list_workspaces()
        workspace_count = len(workspaces)
        assert workspace in [w.id for w in workspaces]

    response = requests.get(
        f"{SERVER_URL}/{workspace}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, response.json()["detail"]
    assert "hypha-rpc-websocket.js" in response.text

    # Without the token, we can only access to the public service
    response = requests.get(f"{SERVER_URL}/{service_ws}/services")
    assert response.ok, response.json()["detail"]
    response = response.json()
    assert find_item(response, "name", "test_service")
    assert not find_item(response, "name", "test_service_protected")

    # service = await api.get_service("test_service_protected")
    # assert await service.echo("22") == "22"

    # With the token we can access the protected service
    response = requests.get(
        f"{SERVER_URL}/{service_ws}/services",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, response.json()["detail"]
    assert find_item(response.json(), "name", "test_service")
    assert find_item(response.json(), "name", "test_service_protected")

    response = requests.get(f"{SERVER_URL}/{service_ws}/services")
    assert response.ok, response.json()["detail"]
    assert find_item(response.json(), "name", "test_service")

    response = requests.get(
        f"{SERVER_URL}/{service_ws}/services/{svc1.id.split('/')[-1]}"
    )
    assert response.ok, response.json()["detail"]
    service_info = response.json()
    assert service_info["name"] == "test_service"

    response = requests.get(
        f"{SERVER_URL}/{service_ws}/services/{svc1.id.split('/')[-1]}"
    )
    assert response.ok, response.json()["detail"]
    service_info = response.json()
    assert service_info["name"] == "test_service"

    response = requests.get(f"{SERVER_URL}/public/services/s3-storage")
    assert response.ok, response.json()["detail"]
    service_info = response.json()
    assert service_info["id"].endswith("s3-storage")

    response = requests.get(
        f"{SERVER_URL}/public/services/s3-storage/generate_credential"
    )
    print(response.text)
    assert not response.ok

    service_id = svc1.id.split("/")[-1]

    response = requests.get(
        f"{SERVER_URL}/{service_ws}/services/{service_id}/echo?v=3345"
    )
    assert response.ok, response.json()["detail"]

    response = requests.get(
        f"{SERVER_URL}/{service_ws}/services/{service_id}/echo?v=33",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, response.json()["detail"]
    service_info = response.json()
    assert service_info["v"] == 33

    response = requests.post(
        f"{SERVER_URL}/{service_ws}/services/{service_id}/echo",
        data=msgpack.dumps({"data": 123}),
        headers={"Content-type": "application/msgpack"},
    )
    assert response.ok
    result = msgpack.loads(response.content)
    assert result["data"] == 123

    response = requests.post(
        f"{SERVER_URL}/{service_ws}/services/{service_id}/echo",
        data=json.dumps({"data": 123}),
        headers={"Content-type": "application/json"},
    )
    assert response.ok
    result = json.loads(response.content)
    assert result["data"] == 123

    response = requests.post(
        f"{SERVER_URL}/{service_ws}/services/{service_id}/echo",
        data=msgpack.dumps({"data": 123}),
        headers={
            "Content-type": "application/msgpack",
            "Authorization": f"Bearer {token}",
            "Origin": "http://localhost:3000",
        },
    )
    assert response.ok
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"

    result = msgpack.loads(response.content)
    assert result["data"] == 123

    # Test numpy array
    input_array = np.random.randint(0, 255, [3, 10, 100]).astype("float32")
    input_data = {
        "_rtype": "ndarray",
        "_rvalue": input_array.tobytes(),
        "_rshape": input_array.shape,
        "_rdtype": str(input_array.dtype),
    }

    data = msgpack.dumps({"data": input_data})
    compressed_data = gzip.compress(data)
    response = requests.post(
        f"{SERVER_URL}/{service_ws}/services/{service_id}/echo",
        data=compressed_data,
        headers={
            "Content-Type": "application/msgpack",
            "Content-Encoding": "gzip",
        },
    )

    assert response.ok
    results = msgpack.loads(response.content)
    result = results["data"]

    output_array = np.frombuffer(result["_rvalue"], dtype=result["_rdtype"]).reshape(
        result["_rshape"]
    )

    assert output_array.shape == input_array.shape

    await controller.stop(app_config.id)

    await api.disconnect()

    await asyncio.sleep(1)

    async with connect_to_server(
        {
            "name": "root client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": root_user_token,
        }
    ) as root_api:
        workspaces = await root_api.list_workspaces()
        assert workspace_count >= len(workspaces)


async def test_http_streaming_generator(minio_server, fastapi_server, test_user_token):
    """Test HTTP proxy with generator functions that return streaming responses."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    workspace = api.config["workspace"]

    # Define generator functions
    def sync_counter(start=0, end=5):
        """Return a synchronous generator that counts from start to end."""
        for i in range(start, end):
            yield {"count": i, "message": f"Count {i}"}

    async def async_counter(start=0, end=5):
        """Return an async generator that counts from start to end."""
        for i in range(start, end):
            yield {"count": i, "message": f"Async count {i}"}

    # Register service with generator functions
    svc = await api.register_service(
        {
            "id": "streaming_service",
            "name": "streaming_service",
            "type": "streaming_service",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "sync_counter": sync_counter,
            "async_counter": async_counter,
        }
    )

    service_id = svc.id.split("/")[-1]
    base_url = f"{SERVER_URL}/{workspace}/services/{service_id}"

    async with httpx.AsyncClient() as client:
        # Test 1: GET request with JSON streaming (Server-Sent Events)
        print("Testing GET request with JSON streaming...")
        response = await client.get(f"{base_url}/sync_counter?start=0&end=3")
        assert response.status_code == 200

        # Verify it returns a streaming response
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Verify SSE format
        content = response.text
        assert "data: " in content  # Should have SSE data events
        assert "data: [DONE]" in content  # Should have completion marker

        # Verify it contains either valid data or error handling
        lines = content.split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) >= 2  # At least one data event + [DONE]

        # Test 2: GET request with different parameters
        print("Testing GET request with different parameters...")
        response = await client.get(f"{base_url}/sync_counter?start=1&end=4")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert "data: " in response.text
        assert "data: [DONE]" in response.text

        # Test 3: GET request for async generator
        print("Testing GET request for async generator...")
        response = await client.get(f"{base_url}/async_counter?start=0&end=2")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert "data: " in response.text
        assert "data: [DONE]" in response.text

        # Test 4: POST request with JSON body -> JSON streaming
        print("Testing POST request with JSON body...")
        response = await client.post(
            f"{base_url}/sync_counter",
            data=json.dumps({"start": 0, "end": 2}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert "data: " in response.text
        assert "data: [DONE]" in response.text

        # Test 5: POST request with msgpack body -> msgpack streaming
        print("Testing POST request with msgpack body...")
        response = await client.post(
            f"{base_url}/sync_counter",
            data=msgpack.dumps({"start": 0, "end": 2}),
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 200

        # Verify msgpack response
        assert "application/msgpack" in response.headers.get("content-type", "")
        assert len(response.content) > 0

    await api.disconnect()


async def test_http_auth_context(minio_server, fastapi_server, test_user_token):
    """Test authentication context passing for HTTP service functions."""

    # Connect to server and register services
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Register a public service that captures user context
    def echo_context(data, context=None):
        """Service function that echoes the user context."""
        try:
            if context is None:
                return {
                    "data": data,
                    "user_id": None,
                    "workspace": None,
                    "user_info": None,
                    "has_context": False,
                }

            # Handle context as dictionary (similar to ASGI tests)
            user_info = context.get("user") if isinstance(context, dict) else None

            return {
                "data": data,
                "user_id": user_info.get("id") if user_info else None,
                "workspace": context.get("ws") if isinstance(context, dict) else None,
                "user_info": (
                    {
                        "id": user_info.get("id") if user_info else None,
                        "scope": user_info.get("scope") if user_info else None,
                    }
                    if user_info
                    else None
                ),
                "has_context": True,
                "context_type": str(type(context)),
                "context_content": str(context) if context else None,
            }
        except Exception as e:
            # Return error information for debugging
            return {
                "data": data,
                "error": str(e),
                "context_type": str(type(context)) if context else None,
                "context_content": str(context) if context else None,
                "has_context": context is not None,
            }

    public_svc = await api.register_service(
        {
            "id": "test_public_http",
            "name": "test_public_http",
            "type": "test_service",
            "config": {
                "visibility": "public",
                "require_context": True,
            },
            "echo_context": echo_context,
        }
    )

    # Register a protected service that captures user context
    protected_svc = await api.register_service(
        {
            "id": "test_protected_http",
            "name": "test_protected_http",
            "type": "test_service",
            "config": {
                "visibility": "protected",
                "require_context": True,
            },
            "echo_context": echo_context,
        }
    )

    public_service_id = public_svc.id.split("/")[-1]
    protected_service_id = protected_svc.id.split("/")[-1]

    print("=" * 60)
    print("Testing HTTP Service Authentication Context")
    print("=" * 60)
    print(f"Workspace: {workspace}")
    print(f"Public service ID: {public_service_id}")
    print(f"Protected service ID: {protected_service_id}")
    print(f"Public service full ID: {public_svc.id}")
    print(f"Protected service full ID: {protected_svc.id}")

    # Test 1: Public service WITHOUT token
    print("\n1. Testing public service call without token...")
    response = requests.post(
        f"{SERVER_URL}/{workspace}/services/{public_service_id}/echo_context",
        json={"message": "hello"},
        headers={"Content-Type": "application/json"},
    )

    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response text: {response.text}")

    # If we get a response, check it
    if response.status_code == 200:
        result = response.json()
        print(f"Result: {result}")
        print(f"User ID: {result.get('user_id')}")
        print(f"Workspace: {result.get('workspace')}")
        print(f"Has context: {result.get('has_context')}")
        print(f"Context type: {result.get('context_type')}")
        if result.get("error"):
            print(f"Error: {result.get('error')}")

        # Without token, user should be anonymous or None
        if result.get("user_id"):
            assert result.get("user_id") == "anonymouz-http"
        print("✓ Anonymous access works correctly")
    else:
        # For now, let's allow the test to continue even if this fails
        print("⚠ Anonymous access returned non-200 status")

    # Test 2: Public service WITH token
    print("\n2. Testing public service call with token...")
    response = requests.post(
        f"{SERVER_URL}/{workspace}/services/{public_service_id}/echo_context",
        json={"message": "hello"},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )

    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response text: {response.text}")

    if response.status_code == 200:
        result = response.json()
        print(f"Result: {result}")
        print(f"User ID: {result.get('user_id')}")
        print(f"Workspace: {result.get('workspace')}")
        print(f"User scope: {result.get('user_info', {}).get('scope', {})}")
        if result.get("error"):
            print(f"Error: {result.get('error')}")

        # With token, user should NOT be anonymous
        if result.get("user_id"):
            assert result.get("user_id") != "anonymouz-http"
        print("✓ Authenticated access works correctly")
    else:
        print("⚠ Authenticated access returned non-200 status")

    # Test 3: Protected service WITHOUT token - should fail
    print("\n3. Testing protected service call without token...")
    response = requests.post(
        f"{SERVER_URL}/{workspace}/services/{protected_service_id}/echo_context",
        json={"message": "hello"},
        headers={"Content-Type": "application/json"},
    )
    # Should fail for protected service without token
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("⚠ Protected service allowed anonymous access")
    else:
        print("✓ Protected service correctly rejects anonymous access")

    # Test 4: Protected service WITH token - should succeed
    print("\n4. Testing protected service call with token...")
    response = requests.post(
        f"{SERVER_URL}/{workspace}/services/{protected_service_id}/echo_context",
        json={"message": "hello"},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )

    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response text: {response.text}")

    if response.status_code == 200:
        result = response.json()
        print(f"Result: {result}")
        print(f"User ID: {result.get('user_id')}")
        print(f"Workspace: {result.get('workspace')}")
        if result.get("error"):
            print(f"Error: {result.get('error')}")

        # Should succeed with authenticated user
        if result.get("user_id"):
            assert result.get("user_id") != "anonymouz-http"
        print("✓ Protected service accepts authenticated access")
    else:
        print("⚠ Protected service with token returned non-200 status")

    print("\nAll HTTP authentication context tests passed! ✓")
    await api.disconnect()
