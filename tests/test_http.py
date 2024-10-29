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
    app_config = await controller.launch(
        source=TEST_APP_CODE,
        config={"type": "window"},
        wait_for_service=True,
    )
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
