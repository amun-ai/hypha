"""Test http proxy."""
import gzip

import json
import msgpack
import numpy as np
import pytest
import httpx
import requests
from imjoy_rpc.hypha.websocket_client import connect_to_server

from . import WS_SERVER_URL, SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio

TEST_APP_CODE = """
api.export({
    async setup(){
    },
    async register_services(){
        const service_info1 = await api.register_service(
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
        const service_info2 = await api.register_service(
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


async def test_services(minio_server, fastapi_server, test_user_token):
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 10,
            "token": test_user_token,
        }
    )
    workspace = api.config["workspace"]
    await api.register_service(
        {
            "id": "test_service",
            "name": "test_service",
            "type": "test_service",
            "config": {
                "visibility": "public",
            },
            "echo": lambda data: data,
        }
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        data = await client.get(f"{SERVER_URL}/services/openapi.json")
        data = data.json()
        assert data.get("detail") != "Not Found"
        assert data["info"]["title"] == "Hypha Services"
        paths = data["paths"]
        assert "/call" in paths
        assert "/list" in paths

        url = f"{SERVER_URL}/services/call"
        data = await client.post(
            url,
            json={
                "workspace": workspace,
                "service_id": "test_service",
                "function_key": "echo",
                "function_kwargs": {"data": "123"},
            },
        )
        assert data.status_code == 200
        assert data.json() == "123"

        url = (
            f"{SERVER_URL}/services/call?workspace={workspace}&service_id=test_service&function_key=echo&function_kwargs="
            + json.dumps({"data": "123"})
        )
        data = await client.get(url)
        assert data.status_code == 200
        assert data.json() == "123"

        url = f"{SERVER_URL}/{workspace}/services/test_service/echo"
        data = await client.post(url, json={"data": "123"})
        assert data.status_code == 200
        assert data.json() == "123"

        data = await client.get(f"{SERVER_URL}/services/list?workspace={workspace}")
        print(data.json())
        # [{'config': {'visibility': 'public', 'require_context': False, 'workspace': 'VRRVEdTF9of2y4cLmepzBw', 'flags': []}, 'id': '5XCPAyZrW72oBzywEk2oxP:test_service', 'name': 'test_service', 'type': 'test_service', 'description': '', 'docs': {}}]
        assert data.status_code == 200
        assert data.json()[0]["name"] == "test_service"

    await api.disconnect()


# pylint: disable=too-many-statements
async def test_http_proxy(minio_server, fastapi_server, test_user_token):
    """Test http proxy."""
    # WS_SERVER_URL = "http://127.0.0.1:9527"
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 10,
            "token": test_user_token,
        }
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test plugin with custom template
    controller = await api.get_service("server-apps")
    config = await controller.launch(
        source=TEST_APP_CODE,
        config={"type": "window"},
        wait_for_service=None,
    )
    plugin = await api.get_plugin(config.id)
    assert "setup" in plugin and "register_services" in plugin
    svc1, svc2 = await plugin.register_services()

    service_ws = plugin.config.workspace
    assert service_ws
    service = await api.get_service(svc1["id"])
    assert await service.echo("233d") == "233d"

    service = await api.get_service(svc2["id"])
    assert await service.echo("22") == "22"

    response = requests.get(f"{SERVER_URL}/workspaces/list")
    assert response.ok, response.json()["detail"]
    response = response.json()
    assert workspace in response

    response = requests.get(f"{SERVER_URL}/{workspace}/info")
    assert not response.ok
    assert response.json()["detail"].startswith("Permission denied")

    response = requests.get(
        f"{SERVER_URL}/{workspace}/info", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.ok, response.json()["detail"]
    response = response.json()
    assert response["name"] == workspace

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

    response = requests.get(f"{SERVER_URL}/{service_ws}/services/test_service")
    assert response.ok, response.json()["detail"]
    service_info = response.json()
    assert service_info["name"] == "test_service"

    response = requests.get(f"{SERVER_URL}/public/services/s3-storage")
    assert response.ok, response.json()["detail"]
    service_info = response.json()
    assert service_info["id"] == "public/workspace-manager:s3-storage"

    response = requests.get(
        f"{SERVER_URL}/public/services/s3-storage/generate_credential"
    )
    assert not response.ok

    response = requests.get(
        f"{SERVER_URL}/{service_ws}/services/test_service/echo?v=3345"
    )
    assert response.ok, response.json()["detail"]

    response = requests.get(
        f"{SERVER_URL}/{service_ws}/services/test_service/echo?v=33",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, response.json()["detail"]
    service_info = response.json()
    assert service_info["v"] == 33

    response = requests.post(
        f"{SERVER_URL}/{service_ws}/services/test_service/echo",
        data=msgpack.dumps({"data": 123}),
        headers={"Content-type": "application/msgpack"},
    )
    assert response.ok
    result = msgpack.loads(response.content)
    assert result["data"] == 123

    response = requests.post(
        f"{SERVER_URL}/{service_ws}/services/test_service/echo",
        data=json.dumps({"data": 123}),
        headers={"Content-type": "application/json"},
    )
    assert response.ok
    result = json.loads(response.content)
    assert result["data"] == 123

    response = requests.post(
        f"{SERVER_URL}/{service_ws}/services/test_service/echo",
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
        f"{SERVER_URL}/{service_ws}/services/test_service/echo",
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
