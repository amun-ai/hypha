"""Test server apps."""
from pathlib import Path

import pytest
from imjoy_rpc import connect_to_server

from . import SIO_SERVER_URL

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
    config = await controller.load(
        source=TEST_APP_CODE,
        template="window-plugin.html",
        app_id="test-window-plugin",
        overwrite=True,
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
    await controller.unload(name=config.name)

    with pytest.raises(Exception, match=r".*does not exists.*"):
        config = await controller.load(
            app_id="123/233", workspace=workspace, token=token
        )

    config = await controller.load(
        app_id=config.app_id, workspace=workspace, token=token
    )
    plugin = await api.get_plugin(config.name)
    assert "execute" in plugin
    result = await plugin.execute(2, 4)
    assert result == 6
    webgpu_available = await plugin.check_webgpu()
    assert webgpu_available is True
    await controller.unload(name=config.name, app_id=config.app_id)

    # Test window plugin
    source = (
        (Path(__file__).parent / "testWindowPlugin1.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    config = await controller.load(
        source=source,
        template="imjoy",
        overwrite=True,
        workspace=workspace,
        token=token,
    )
    assert "app_id" in config
    plugin = await api.get_plugin(config.name)
    assert "add2" in plugin
    result = await plugin.add2(4)
    assert result == 6
    await controller.unload(name=config.name, app_id=config.app_id)

    source = (
        (Path(__file__).parent / "testWebPythonPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.load(
        source=source,
        template="imjoy",
        overwrite=True,
        workspace=workspace,
        token=token,
    )
    plugin = await api.get_plugin(config.name)
    assert "add2" in plugin
    result = await plugin.add2(4)
    assert result == 6
    await controller.unload(name=config.name, app_id=config.app_id)

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.load(
        source=source,
        template="imjoy",
        overwrite=True,
        workspace=workspace,
        token=token,
    )
    plugin = await api.get_plugin(config.name)
    assert "add2" in plugin
    result = await plugin.add2(4)
    assert result == 6
    await controller.unload(name=config.name, app_id=config.app_id)

    config = await controller.load(
        workspace=workspace,
        token=token,
        source="https://raw.githubusercontent.com/imjoy-team/ImJoy/master/web/src/plugins/webWorkerTemplate.imjoy.html"
    )
    assert config.name == "Untitled Plugin"
