"""Test ASGI services."""

from pathlib import Path
import asyncio

import pytest
import requests
import httpx
from fastapi import FastAPI
from starlette.requests import Request
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
    config = await controller.launch(
        source=source,
        wait_for_service="hello-fastapi",
        timeout=30,
    )
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
    config = await controller.launch(
        source=source,
        wait_for_service="hello-functions",
        timeout=30,
    )

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
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    # Get initial workspace count
    initial_workspaces = await api.list_workspaces()

    # Create a single FastAPI app instance to handle all requests
    app = FastAPI()
    @app.get("/api")
    async def read_api(request: Request):
        return {"message": "Hello World"}

    async def serve_fastapi(args):
        await app(args["scope"], args["receive"], args["send"])

    # Register service in public workspace
    service = await api.register_service(
        {
            "id": f"test-asgi",
            "type": "asgi",
            "config": {"visibility": "public"},
            "serve": serve_fastapi,
        }
    )
    sid = service["id"].split(":")[1]

    # Add a small delay to ensure service is ready
    await asyncio.sleep(1)

    # Make concurrent requests with timeout and retries
    max_retries = 3
    timeout = httpx.Timeout(30.0, connect=10.0)
    
    async def make_request_with_retry(client, url, retry_count=0):
        try:
            return await client.get(url, timeout=timeout)
        except httpx.ReadTimeout:
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Add delay between retries
                return await make_request_with_retry(client, url, retry_count + 1)
            raise

    # Make requests in smaller batches to reduce load
    batch_size = 5
    all_responses = []
    
    async with httpx.AsyncClient() as client:
        for i in range(0, 10, batch_size):
            tasks = [
                make_request_with_retry(
                    client,
                    f"{SERVER_URL}/{api.config.workspace}/apps/{sid}/api?_mode=last"
                )
                for _ in range(batch_size)
            ]
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
            all_responses.extend(batch_responses)
            await asyncio.sleep(0.5)  # Add delay between batches

    # Verify responses
    for response in all_responses:
        if isinstance(response, Exception):
            raise response
        assert response.status_code == 200, response.text
        assert response.json()["message"] == "Hello World"

    # Verify no new persistent workspaces created
    final_workspaces = await api.list_workspaces()
    assert len(final_workspaces) == len(
        initial_workspaces
    ), "No new workspaces should be created for anonymous access"

    # Verify temporary workspaces are cleaned up
    temp_workspaces = [
        w
        for w in final_workspaces
        if w["id"].startswith("ws-user-") and not w["persistent"]
    ]
    assert (
        len(temp_workspaces) == 0
    ), "All temporary workspaces should have been cleaned up"

    await api.disconnect()

