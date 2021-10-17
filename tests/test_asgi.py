"""Test ASGI services."""
from pathlib import Path

import pytest
import requests
from imjoy_rpc import connect_to_server

from . import SIO_SERVER_URL

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_asgi(socketio_server):
    """Test the ASGI gateway apps."""
    api = await connect_to_server({"name": "test client", "server_url": SIO_SERVER_URL})
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test plugin with custom template
    controller = await api.get_service("server-apps")

    source = (
        (Path(__file__).parent / "testASGIWebPythonPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.launch(
        source=source,
        workspace=workspace,
        token=token,
    )
    plugin = await api.get_plugin(config.name)
    await plugin.setup()
    service = await api.get_service(
        {"workspace": config.workspace, "name": "hello-fastapi"}
    )
    assert "serve" in service

    response = requests.get(f"{SIO_SERVER_URL}/{workspace}/apps/hello-fastapi/")
    assert response.ok
    assert response.json()["message"] == "Hello World"

    service = await api.get_service(
        {"workspace": config.workspace, "name": "hello-flask"}
    )
    assert "serve" in service
    response = requests.get(f"{SIO_SERVER_URL}/{workspace}/apps/hello-flask/")
    assert response.ok
    assert response.text == "<p>Hello, World!</p>"

    # TODO: If we repeat the request, it fails for Flask
    # response = requests.get(f"{SIO_SERVER_URL}/{workspace}/apps/hello-flask/")
    # assert response.ok
    # assert response.text == "<p>Hello, World!</p>"

    await controller.stop(config.id)
