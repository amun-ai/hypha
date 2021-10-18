"""Test http proxy."""
import msgpack
import pytest
import requests
from imjoy_rpc import connect_to_server

from . import SIO_SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio

TEST_APP_CODE = """
api.export({
    async setup(){
        await api.register_service(
            {
                "_rintf": true,
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
        await api.register_service(
            {
                "_rintf": true,
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
    }
})
"""


async def test_http_proxy(minio_server, socketio_server):
    """Test http proxy."""
    # SIO_SERVER_URL = "http://127.0.0.1:9527"
    api = await connect_to_server({"name": "test client", "server_url": SIO_SERVER_URL})
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test plugin with custom template
    controller = await api.get_service("server-apps")
    config = await controller.launch(
        source=TEST_APP_CODE,
        type="window-plugin",
        workspace=workspace,
        token=token,
    )
    plugin = await api.get_plugin(config.name)
    assert "setup" in plugin
    await plugin.setup()

    service_ws = plugin.config.workspace
    service = await api.get_service({"workspace": service_ws, "name": "test_service"})
    assert await service.echo("233d") == "233d"

    service = await api.get_service(
        {"workspace": service_ws, "name": "test_service_protected"}
    )
    assert await service.echo("22") == "22"

    # Without the token, we can only access to the protected service
    response = requests.get(f"{SIO_SERVER_URL}/services")
    assert response.ok
    response = response.json()
    assert find_item(response, "name", "test_service")
    assert not find_item(response, "name", "test_service_protected")

    service = await api.get_service(
        {"workspace": service_ws, "name": "test_service_protected"}
    )
    assert await service.echo("22") == "22"

    # With the token we can access the protected service
    response = requests.get(
        f"{SIO_SERVER_URL}/services",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok
    assert find_item(response.json(), "name", "test_service")
    assert find_item(response.json(), "name", "test_service_protected")

    response = requests.get(f"{SIO_SERVER_URL}/{service_ws}/services")
    assert response.ok
    assert find_item(response.json(), "name", "test_service")

    response = requests.get(f"{SIO_SERVER_URL}/{service_ws}/services/test_service")
    assert response.ok
    service_info = response.json()
    assert service_info["name"] == "test_service"

    response = requests.get(f"{SIO_SERVER_URL}/public/services/s3-storage")
    assert response.ok
    service_info = response.json()
    assert service_info["name"] == "s3-storage"

    response = requests.get(
        f"{SIO_SERVER_URL}/public/services/s3-storage/generate_credential"
    )
    assert not response.ok

    response = requests.get(
        f"{SIO_SERVER_URL}/{service_ws}/services/test_service/echo?v=3345"
    )
    assert response.ok, response.json()["detail"]

    response = requests.get(
        f"{SIO_SERVER_URL}/{service_ws}/services/test_service/echo?v=33",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok, response.json()["detail"]
    service_info = response.json()
    assert service_info["v"] == 33

    response = requests.post(
        f"{SIO_SERVER_URL}/{service_ws}/services/test_service/echo",
        data=msgpack.dumps({"data": 123}),
        headers={"Content-type": "application/msgpack"},
    )

    response = requests.post(
        f"{SIO_SERVER_URL}/{service_ws}/services/test_service/echo",
        data=msgpack.dumps({"data": 123}),
        headers={
            "Content-type": "application/msgpack",
            "Authorization": f"Bearer {token}",
        },
    )
    assert response.ok
    result = msgpack.loads(response.content)
    assert result["data"] == 123
