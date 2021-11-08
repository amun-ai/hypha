"""Test server apps."""
from pathlib import Path
import asyncio
import requests

import pytest
from imjoy_rpc import connect_to_server

from . import SIO_SERVER_URL, find_item

# pylint: disable=too-many-statements

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio

TEST_APP_CODE = """
api.log('awesome!connected!');

api.export({
    async setup(){
        await api.log("initialized")
    },
    async check_webgpu(){
        if ("gpu" in navigator) {
            // WebGPU is supported! 🎉
            return true
        }
        else return false
    },
    async execute(a, b){
        return a + b
    }
})
"""


async def test_server_apps(socketio_server):
    """Test the server apps."""
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
    assert "app_id" in config
    plugin = await api.get_plugin(config.name)
    assert "execute" in plugin
    result = await plugin.execute(2, 4)
    assert result == 6
    webgpu_available = await plugin.check_webgpu()
    assert webgpu_available is True
    # only pass name so the app won't be removed
    await controller.stop(config.id)

    config = await controller.launch(
        source=TEST_APP_CODE, type="window-plugin", workspace=workspace, token=token
    )
    plugin = await api.get_plugin(config.name)
    assert "execute" in plugin
    result = await plugin.execute(2, 4)
    assert result == 6
    webgpu_available = await plugin.check_webgpu()
    assert webgpu_available is True
    await controller.stop(config.id)

    # Test window plugin
    source = (
        (Path(__file__).parent / "testWindowPlugin1.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    config = await controller.launch(
        source=source,
        type="imjoy",
        workspace=workspace,
        token=token,
    )
    assert "app_id" in config
    plugin = await api.get_plugin(config.name)
    assert "add2" in plugin
    result = await plugin.add2(4)
    assert result == 6
    await controller.stop(config.id)

    source = (
        (Path(__file__).parent / "testWebPythonPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.launch(
        source=source,
        type="imjoy",
        workspace=workspace,
        token=token,
    )
    plugin = await api.get_plugin(config.name)
    assert "add2" in plugin
    result = await plugin.add2(4)
    assert result == 6
    await controller.stop(config.id)

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.launch(
        source=source,
        type="imjoy",
        workspace=workspace,
        token=token,
    )
    plugin = await api.get_plugin(config.name)
    assert "add2" in plugin
    result = await plugin.add2(4)
    assert result == 6
    await controller.stop(config.id)

    config = await controller.launch(
        workspace=workspace,
        token=token,
        source="https://raw.githubusercontent.com/imjoy-team/"
        "ImJoy/master/web/src/plugins/webWorkerTemplate.imjoy.html",
    )
    assert config.name == "Untitled Plugin"
    apps = await controller.list()
    assert find_item(apps, "id", config.id)


async def test_readiness_liveness(socketio_server):
    """Test readiness and liveness probes."""
    api = await connect_to_server({"name": "test client", "server_url": SIO_SERVER_URL})
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test plugin with custom template
    controller = await api.get_service("server-apps")

    source = (
        (Path(__file__).parent / "testUnreliablePlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    config = await controller.launch(
        source=source,
        type="imjoy",
        workspace=workspace,
        token=token,
    )

    assert config.name == "Unreliable Plugin"
    plugin = await api.get_plugin(config.name)
    assert plugin
    await asyncio.sleep(10)

    plugin = await api.get_plugin(config.name)
    assert plugin

    await plugin.exit()


async def test_non_persistent_workspace(socketio_server):
    """Test non-persistent workspace."""
    api = await connect_to_server({"name": "test client", "server_url": SIO_SERVER_URL})
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test plugin with custom template
    controller = await api.get_service("server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    config = await controller.launch(
        source=source,
        type="imjoy",
        workspace=workspace,
        token=token,
    )

    plugin = await api.get_plugin(config.name)
    assert plugin is not None

    # It should exist in the stats
    response = requests.get(f"{SIO_SERVER_URL}/api/stats")
    assert response.status_code == 200
    stats = response.json()
    workspace_info = find_item(stats["workspaces"], "name", workspace)
    assert workspace_info is not None
    plugins = workspace_info["plugins"]
    assert find_item(plugins, "id", plugin.config["id"]) is not None

    await api.disconnect()

    # now it should disappear from the stats
    response = requests.get(f"{SIO_SERVER_URL}/api/stats")
    assert response.status_code == 200
    stats = response.json()
    workspace_info = find_item(stats["workspaces"], "name", workspace)
    assert workspace_info is None
