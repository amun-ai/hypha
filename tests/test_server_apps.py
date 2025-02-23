"""Test server apps."""

from pathlib import Path
import asyncio
import requests
import time

import pytest
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SERVER_URL, find_item, wait_for_workspace_ready

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


async def test_server_apps_workspace_removal(
    fastapi_server, test_user_token_5, root_user_token
):
    """Test the server apps."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token_5,
        }
    )
    controller = await api.get_service("public/server-apps")

    app_info = await controller.launch(
        source=TEST_APP_CODE,
        config={"type": "window"},
        overwrite=True,
        wait_for_service="default",
    )

    # the workspace should exist in the stats
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "admin", "token": root_user_token}
    ) as root:
        admin = await root.get_service("admin-utils")
        workspaces = await admin.list_workspaces()
        workspace_info = find_item(workspaces, "name", api.config["workspace"])
        assert workspace_info

    await controller.stop(app_info.id)
    # Now disconnect it
    await api.disconnect()
    await asyncio.sleep(0.5)

    # now it should disappear from the stats
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "admin", "token": root_user_token}
    ) as root:
        admin = await root.get_service("admin-utils")
        workspaces = await admin.list_workspaces()
        workspace_info = find_item(workspaces, "name", api.config["workspace"])
        assert workspace_info is None


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

    # objects = await api.list_remote_objects()
    # assert len(objects) == 1

    # Test app with custom template
    controller = await api.get_service("public/server-apps")

    # objects = await api.list_remote_objects()
    # assert len(objects) == 2

    config = await controller.launch(
        source=TEST_APP_CODE,
        config={"type": "window"},
        wait_for_service="default",
        overwrite=True,
    )
    assert "id" in config.keys()
    app = await api.get_app(config.id)

    # objects = await api.list_remote_objects()
    # assert len(objects) == 3

    assert "execute" in app
    result = await app.execute(2, 4)
    assert result == 6
    webgpu_available = await app.check_webgpu()
    assert webgpu_available is True

    # Test logs
    logs = await controller.get_log(config.id)
    assert "log" in logs and "error" in logs
    logs = await controller.get_log(config.id, type="log", offset=0, limit=1)
    assert len(logs) == 1

    await controller.stop(config.id)

    # Test window app
    source = (
        (Path(__file__).parent / "testWindowPlugin1.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    config = await controller.launch(
        source=source,
        wait_for_service=True,
    )
    assert "app_id" in config
    app = await api.get_app(config.id)
    assert "add2" in app, str(app and app.keys())
    result = await app.add2(4)
    assert result == 6
    await controller.stop(config.id)


async def test_singleton_apps(fastapi_server, test_user_token):
    """Test the singleton apps."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    controller = await api.get_service("public/server-apps")

    # Launch the first instance of the singleton app
    config1 = await controller.launch(
        source=TEST_APP_CODE,
        config={"type": "window", "singleton": True},
        overwrite=True,
        wait_for_service="default",
    )
    assert "id" in config1.keys()

    # Try launching another instance of the same singleton app
    with pytest.raises(Exception, match="is a singleton app and already running"):
        await controller.launch(
            source=TEST_APP_CODE,
            config={"type": "window", "singleton": True},
            overwrite=True,
            wait_for_service="default",
        )

    # Stop the singleton app
    await controller.stop(config1.id)

    # Verify the singleton app is no longer running
    running_apps = await controller.list_running()
    assert not any(app["id"] == config1.id for app in running_apps)


async def test_daemon_apps(fastapi_server, test_user_token_6, root_user_token):
    """Test the daemon apps."""
    # First create a persistent workspace
    async with connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "token": test_user_token_6,
        }
    ) as api:
        workspace = await api.create_workspace(
            {
                "name": "ws-user-user-6",
                "description": "Test workspace for daemon apps",
                "persistent": True,
            },
            overwrite=True,
        )
        workspace_id = workspace["id"]

    # Now test the daemon app
    async with connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token_6,
            "workspace": workspace_id,
            "client_id": "test-client_daemon_1",
        }
    ) as api:
        controller = await api.get_service("public/server-apps")

        # Launch a daemon app
        config = await controller.launch(
            source=TEST_APP_CODE,
            config={"type": "window", "daemon": True},
            overwrite=True,
            wait_for_service="default",
        )
        assert "id" in config.keys()

        # Verify the daemon app is running
        running_apps = await controller.list_running()
        assert any(app["id"] == config.id for app in running_apps)

        apps = await controller.list_apps()
        assert find_item(apps, "id", config.app_id)

        # Store the workspace and app info for later verification
        app_id = config.app_id

        # Stop the app
        await controller.stop(config.id)
        print(f"app {config.id} stopped")

    # Add a small delay to allow workspace to fully initialize
    await asyncio.sleep(1)

    # Reconnect with the same token to verify daemon app persistence
    print("Reconnecting with the same token to verify daemon app persistence")
    async with connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token_6,
            "workspace": workspace_id,
            "client_id": "test-client",
        }
    ) as api:
        print("Waiting for the workspace to be ready")
        await wait_for_workspace_ready(api, timeout=30)
        controller = await api.get_service("public/server-apps")
        running_apps = await controller.list_running()

        # Verify the daemon app is still running
        assert any(
            app["app_id"] == app_id for app in running_apps
        ), "Daemon app should still be running"


async def test_web_python_apps(fastapi_server, test_user_token):
    """Test webpython app."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    controller = await api.get_service("public/server-apps")
    source = (
        (Path(__file__).parent / "testWebPythonPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = await controller.launch(
        source=source,
        wait_for_service="default",
        timeout=40,
    )
    assert config.name == "WebPythonPlugin"
    svc = await api.get_service(config.service_id)
    assert svc is not None
    app = await api.get_app(config.id)
    assert "add2" in app, str(app and app.keys())
    result = await app.add2(4)
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
    app = await api.get_app(config.id)
    assert "add2" in app, str(app and app.keys())
    result = await app.add2(4)
    assert result == 6
    await controller.stop(config.id)

    config = await controller.launch(
        source="https://raw.githubusercontent.com/imjoy-team/"
        "ImJoy/master/web/src/plugins/webWorkerTemplate.imjoy.html",
        wait_for_service=True,
    )
    # assert config.name == "Untitled Plugin"
    apps = await controller.list_running()
    assert find_item(apps, "id", config.id)


async def test_non_persistent_workspace(fastapi_server, root_user_token):
    """Test non-persistent workspace."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
        }
    )
    workspace = api.config["workspace"]

    # # Test app with custom template
    # controller = await api.get_service("public/server-apps")

    # source = (
    #     (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
    #     .open(encoding="utf-8")
    #     .read()
    # )

    # config = await controller.launch(
    #     source=source,
    #     wait_for_service="default",
    # )

    # app = await api.get_app(config.id)
    # assert app is not None

    # It should exist in the stats
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "admin", "token": root_user_token}
    ) as root:
        admin = await root.get_service("admin-utils")
        workspaces = await admin.list_workspaces()
        workspace_info = find_item(workspaces, "name", workspace)
        assert workspace_info is not None

    # We don't need to stop manually, since it should be removed
    # when the parent client exits
    # await controller.stop(config.id)

    await api.disconnect()
    await asyncio.sleep(0.1)

    # now it should disappear from the stats
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "admin", "token": root_user_token}
    ) as root:
        admin = await root.get_service("admin-utils")
        workspaces = await admin.list_workspaces()
        workspace_info = find_item(workspaces, "name", workspace)
        assert workspace_info is None


async def test_lazy_plugin(fastapi_server, test_user_token):
    """Test lazy app loading."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    # Test app with custom template
    controller = await api.get_service("public/server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    app_info = await controller.install(
        source=source,
        overwrite=True,
    )

    apps = await controller.list_apps()
    assert find_item(apps, "id", app_info.id)

    app = await api.get_service(f"echo@{app_info.id}", mode="first")
    assert app is not None

    await controller.uninstall(app_info.id)

    await api.disconnect()


async def test_stop_after_inactive(fastapi_server, test_user_token):
    """Test app with time limit."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    # Test app with custom template
    controller = await api.get_service("public/server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    app_info = await controller.install(
        source=source,
        overwrite=True,
    )

    apps = await controller.list_apps()
    assert find_item(apps, "id", app_info.id)

    # start the app
    app = await controller.start(app_info.id, stop_after_inactive=1)
    apps = await controller.list_running()
    assert find_item(apps, "id", app.id) is not None
    await asyncio.sleep(2)
    apps = await controller.list_running()
    assert find_item(apps, "id", app.id) is None
    await controller.uninstall(app_info.id)
    await api.disconnect()


async def test_lazy_service(fastapi_server, test_user_token):
    """Test lazy service loading."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    # Test app with custom template
    controller = await api.get_service("public/server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    app_info = await controller.install(
        source=source,
        timeout=30,
        overwrite=True,
    )

    service = await api.get_service("echo@" + app_info.id, mode="first")
    assert service.echo is not None
    assert await service.echo("hello") == "hello"

    long_string = "h" * 10000000
    assert await service.echo(long_string) == long_string

    await controller.uninstall(app_info.id)

    await api.disconnect()
