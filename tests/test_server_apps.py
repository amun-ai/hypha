"""Test server apps."""
from pathlib import Path
import asyncio
import requests

import pytest
from imjoy_rpc.hypha.websocket_client import connect_to_server

from . import WS_SERVER_URL, SERVER_URL, find_item

# pylint: disable=too-many-statements

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio

TEST_APP_CODE = """
api.log('awesome!connected!');

api.export({
    async setup(){
        console.log("this is a log");
        console.error("this is an error");
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
        console.log("executing", a, b);
        return a + b
    }
})
"""


async def test_server_apps_unauthorized(fastapi_server):
    """Test the server apps."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "method_timeout": 30}
    )
    controller = await api.get_service("server-apps")

    with pytest.raises(
        Exception, match=r".*AssertionError: User must be authenticated.*"
    ):
        await controller.launch(
            source=TEST_APP_CODE,
            config={"type": "window"},
            wait_for_service="default",
        )
    await api.disconnect()


async def test_server_apps(fastapi_server, test_user_token):
    """Test the server apps."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    workspace = api.config["workspace"]

    # objects = await api.list_remote_objects()
    # assert len(objects) == 1

    # Test plugin with custom template
    controller = await api.get_service("server-apps")

    # objects = await api.list_remote_objects()
    # assert len(objects) == 2

    config = await controller.launch(
        source=TEST_APP_CODE,
        config={"type": "window"},
        wait_for_service="default",
    )
    assert "app_id" in config
    plugin = await api.get_service(f"{workspace}/{config.id}:default")

    # objects = await api.list_remote_objects()
    # assert len(objects) == 3

    assert "execute" in plugin
    result = await plugin.execute(2, 4)
    assert result == 6
    webgpu_available = await plugin.check_webgpu()
    assert webgpu_available is True

    # Test logs
    logs = await controller.get_log(config.id)
    assert "log" in logs and "error" in logs
    logs = await controller.get_log(config.id, type="log", offset=0, limit=1)
    assert len(logs) == 1

    await controller.stop(config.id)

    # Test window plugin
    source = (
        (Path(__file__).parent / "testWindowPlugin1.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    config = await controller.launch(
        source=source,
    )
    assert "app_id" in config
    plugin = await api.get_service(f"{workspace}/{config.id}:default")
    assert "add2" in plugin
    result = await plugin.add2(4)
    assert result == 6
    await controller.stop(config.id)


async def test_web_python_apps(fastapi_server, test_user_token):
    """Test webpython plugin."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config["workspace"]

    controller = await api.get_service("server-apps")
    source = (
        (Path(__file__).parent / "testWebPythonPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.launch(
        source=source,
        wait_for_service="default",
    )
    assert config.name == "WebPythonPlugin"
    plugin = await api.get_service(f"{workspace}/{config.id}:default")
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
        wait_for_service="default",
    )
    plugin = await api.get_service(f"{workspace}/{config.id}:default")
    assert "add2" in plugin
    result = await plugin.add2(4)
    assert result == 6
    await controller.stop(config.id)

    config = await controller.launch(
        source="https://raw.githubusercontent.com/imjoy-team/"
        "ImJoy/master/web/src/plugins/webWorkerTemplate.imjoy.html",
    )
    # assert config.name == "Untitled Plugin"
    apps = await controller.list_running()
    assert find_item(apps, "id", config.id)


async def test_readiness_liveness(fastapi_server, test_user_token):
    """Test readiness and liveness probes."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config["workspace"]

    # Test plugin with custom template
    controller = await api.get_service("server-apps")

    source = (
        (Path(__file__).parent / "testUnreliablePlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    config = await controller.launch(
        source=source,
    )

    # assert config.name == "Unreliable Plugin"
    plugin = await api.get_service(f"{workspace}/{config.id}:default")
    assert plugin
    await asyncio.sleep(5)

    plugin = None
    failing_count = 0
    while plugin is None:
        try:
            plugin = await api.get_service(f"{workspace}/{config.id}:default")
        except Exception:  # pylint: disable=broad-except
            failing_count += 1
            if failing_count > 30:
                raise
            await asyncio.sleep(1)
    assert plugin


async def test_non_persistent_workspace(fastapi_server, test_user_token_temporary):
    """Test non-persistent workspace."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token_temporary,
        }
    )
    workspace = api.config["workspace"]

    # Test plugin with custom template
    controller = await api.get_service("server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    config = await controller.launch(
        source=source,
    )

    plugin = await api.get_service(f"{workspace}/{config.id}:default")
    assert plugin is not None

    # It should exist in the stats
    response = requests.get(f"{SERVER_URL}/api/stats")
    assert response.status_code == 200
    stats = response.json()
    count = stats["user_count"]
    workspace_info = find_item(stats["workspaces"], "name", workspace)
    assert workspace_info is not None

    # We don't need to stop manually, since it should be removed
    # when the parent client exits
    # await controller.stop(config.id)

    await api.disconnect()
    await asyncio.sleep(0.1)

    # now it should disappear from the stats
    response = requests.get(f"{SERVER_URL}/api/stats")
    assert response.status_code == 200
    stats = response.json()
    workspace_info = find_item(stats["workspaces"], "name", workspace)
    assert workspace_info is None
    assert stats["user_count"] == count - 2


async def test_lazy_plugin(fastapi_server, test_user_token):
    """Test lazy plugin loading."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 60,
            "token": test_user_token,
        }
    )

    # Test plugin with custom template
    controller = await api.get_service("server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    app_info = await controller.install(
        source=source,
    )

    apps = await controller.list_apps()
    assert find_item(apps, "id", app_info.id)

    plugin = await api.get_service({"client_name": app_info.name, "launch": True})
    assert plugin is not None

    await controller.uninstall(app_info.id)

    await api.disconnect()


async def test_lazy_service(fastapi_server, test_user_token):
    """Test lazy service loading."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 5,
            "token": test_user_token,
        }
    )

    # Test plugin with custom template
    controller = await api.get_service("server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    app_info = await controller.install(
        source=source,
    )

    service = await api.get_service({"id": "echo", "launch": True})
    assert service.echo is not None
    assert await service.echo("hello") == "hello"

    long_string = "h" * 10000000  # 10MB
    assert await service.echo(long_string) == long_string

    await controller.uninstall(app_info.id)

    await api.disconnect()
