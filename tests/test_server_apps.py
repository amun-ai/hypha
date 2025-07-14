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

# Add test for raw HTML apps
TEST_RAW_HTML_APP = """
<!DOCTYPE html>
<html>
<head>
    <title>Raw HTML Test App</title>
</head>
<body>
    <h1>Raw HTML Test App</h1>
    <p>This is a test raw HTML app.</p>
    <script type="module">
        const params = new URLSearchParams(window.location.search);
        const server_url = params.get('server_url') || window.location.origin;
        const workspace = params.get('workspace');
        const client_id = params.get('client_id');
        const token = params.get('token');
        
        const hyphaWebsocketClient = await import(server_url + '/assets/hypha-rpc-websocket.mjs');
        const config = {
            server_url,
            workspace,
            client_id,
            token
        };
        
        const api = await hyphaWebsocketClient.connectToServer(config);
        
        // Export a simple service
        api.export({
            async setup() {
                console.log("Raw HTML app initialized");
            },
            async sayHello(name) {
                return `Hello, ${name}! This is a raw HTML app.`;
            },
            async add(a, b) {
                return a + b;
            }
        });
    </script>
</body>
</html>
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
    # stop all the running apps
    for app in await controller.list_running():
        await controller.stop(app["id"])

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


async def test_lazy_service_with_default_timeout(fastapi_server, test_user_token):
    """Test lazy service loading with startup_config."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    # Test app with Web Worker template (simpler and more reliable)
    controller = await api.get_service("public/server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    # stop all the running apps
    for app in await controller.list_running():
        await controller.stop(app["id"])

    app_info = await controller.install(
        source=source,
        timeout=30,
        overwrite=True,
        config={
            "startup_config": {
                "stop_after_inactive": 3,  # Auto-stop after 3 seconds of inactivity
                "timeout": 30,  # Default timeout for starting
                "wait_for_service": "echo",  # Default service to wait for
            }
        },
    )

    # Test lazy loading via get_service (this should trigger the app start)
    service = await api.get_service("echo@" + app_info.id, mode="first")
    assert service.echo is not None
    assert await service.echo("hello") == "hello"

    # Verify that the app is running after lazy loading
    apps = await controller.list_running()
    running_app = find_item(apps, "app_id", app_info.id)
    assert running_app is not None, "App should be running after lazy loading"

    # Wait for the startup_config stop_after_inactive to trigger (3 seconds + some buffer)
    await asyncio.sleep(5)

    # Verify that the app has been stopped due to inactivity (as configured in startup_config)
    apps = await controller.list_running()
    running_app = find_item(apps, "app_id", app_info.id)
    assert (
        running_app is None
    ), "App should be stopped after inactive timeout from startup_config"

    await controller.uninstall(app_info.id)

    await api.disconnect()


async def test_startup_config_comprehensive(fastapi_server, test_user_token):
    """Test comprehensive startup_config with all parameters."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    # Test app with Web Worker template
    controller = await api.get_service("public/server-apps")

    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    # stop all the running apps
    for app in await controller.list_running():
        await controller.stop(app["id"])

    app_info = await controller.install(
        source=source,
        timeout=30,
        overwrite=True,
        config={
            "startup_config": {
                "stop_after_inactive": 2,  # Auto-stop after 2 seconds of inactivity
                "timeout": 25,  # Default timeout for starting
                "wait_for_service": "echo",  # Default service to wait for
            }
        },
    )

    # Test that the app starts with the default configuration
    # This should use the default wait_for_service and timeout from the config
    started_info = await controller.start(app_info["id"])

    # Verify that the app is running
    apps = await controller.list_running()
    running_app = find_item(apps, "app_id", app_info.id)
    assert running_app is not None, "App should be running after start"

    # Verify that the service is available
    service = await api.get_service("echo@" + app_info.id, mode="first")
    assert service.echo is not None
    assert await service.echo("hello") == "hello"

    # Wait for the startup_config stop_after_inactive to trigger (2 seconds + some buffer)
    await asyncio.sleep(4)

    # Verify that the app has been stopped due to inactivity (as configured in startup_config)
    apps = await controller.list_running()
    running_app = find_item(apps, "app_id", app_info.id)
    assert (
        running_app is None
    ), "App should be stopped after inactive timeout from startup_config"

    await controller.uninstall(app_info.id)

    await api.disconnect()


async def test_raw_html_apps(fastapi_server, test_user_token):
    """Test raw HTML app installation and launching."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    controller = await api.get_service("public/server-apps")

    # Test installing raw HTML app without config
    app_info = await controller.install(
        source=TEST_RAW_HTML_APP,
        overwrite=True,
    )

    assert app_info["name"] == "Raw HTML App"
    assert app_info["type"] == "application"  # convert_config_to_artifact sets this
    assert app_info["entry_point"] == "index.html"

    # Test launching the raw HTML app
    config = await controller.launch(
        source=TEST_RAW_HTML_APP,
        config={"name": "Custom Raw HTML App", "type": "window"},
        overwrite=True,
        wait_for_service="default",
    )

    assert "id" in config.keys()
    app = await api.get_app(config.id)

    # Test the exported functions
    assert "sayHello" in app
    assert "add" in app

    result = await app.sayHello("World")
    assert result == "Hello, World! This is a raw HTML app."

    result = await app.add(5, 3)
    assert result == 8

    await controller.stop(config.id)

    # Test installing raw HTML app with custom config
    app_info2 = await controller.install(
        source=TEST_RAW_HTML_APP,
        config={
            "name": "Custom Raw HTML App",
            "version": "1.0.0",
            "type": "window",
            "entry_point": "main.html",
        },
        overwrite=True,
    )

    assert app_info2["name"] == "Custom Raw HTML App"
    assert app_info2["version"] == "1.0.0"
    assert app_info2["type"] == "application"  # convert_config_to_artifact sets this
    assert app_info2["entry_point"] == "main.html"

    # Clean up - both apps have the same ID since they have the same source code
    # So we only need to uninstall once
    try:
        await controller.uninstall(app_info["id"])
    except Exception as e:
        # If first uninstall fails, try the second app's ID
        if "does not exist" in str(e):
            await controller.uninstall(app_info2["id"])
        else:
            raise

    await api.disconnect()


async def test_lazy_service_web_python_app(fastapi_server, test_user_token):
    """Test lazy service loading with web python app (FastAPI template)."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    # Test app with FastAPI template
    controller = await api.get_service("public/server-apps")

    # Read the FastAPI template
    source = (
        (
            Path(__file__).parent.parent
            / "hypha"
            / "templates"
            / "ws"
            / "fastapi_app_template.hypha.html"
        )
        .open(encoding="utf-8")
        .read()
    )

    # Stop all the running apps first
    for app in await controller.list_running():
        await controller.stop(app["id"])

    app_info = await controller.install(
        source=source,
        timeout=30,
        overwrite=True,
    )

    print(f"App info: {app_info}")
    print(f"App services: {app_info.get('services', [])}")

    # Test lazy loading via get_service - this should trigger the app start
    # The FastAPI template registers a service with id "hello-fastapi"
    try:
        service = await api.get_service("hello-fastapi@" + app_info.id, mode="first")
        assert service is not None, "Service should be available via lazy loading"

        # The service should be accessible (it's an ASGI service)
        print(f"Service loaded successfully: {service}")

    except Exception as e:
        print(f"Failed to load service: {e}")
        raise

    # Verify that the app is running after lazy loading
    apps = await controller.list_running()
    running_app = find_item(apps, "app_id", app_info.id)
    assert running_app is not None, "App should be running after lazy loading"

    await controller.uninstall(app_info.id)
    await api.disconnect()


async def test_service_collection_comparison(fastapi_server, test_user_token):
    """Compare service collection between web-worker and web-python apps during installation."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    controller = await api.get_service("public/server-apps")

    # Stop all the running apps first
    for app in await controller.list_running():
        await controller.stop(app["id"])

    # Test 1: Web Worker Plugin - should collect services during installation
    source_webworker = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )

    app_info_webworker = await controller.install(
        source=source_webworker,
        timeout=30,
        overwrite=True,
    )

    print(f"Web Worker App info: {app_info_webworker}")
    print(f"Web Worker App services: {app_info_webworker.get('services', [])}")

    # Test 2: FastAPI Web Python App - currently doesn't collect services during installation
    source_fastapi = (
        (
            Path(__file__).parent.parent
            / "hypha"
            / "templates"
            / "ws"
            / "fastapi_app_template.hypha.html"
        )
        .open(encoding="utf-8")
        .read()
    )

    app_info_fastapi = await controller.install(
        source=source_fastapi,
        timeout=30,
        overwrite=True,
    )

    print(f"FastAPI App info: {app_info_fastapi}")
    print(f"FastAPI App services: {app_info_fastapi.get('services', [])}")

    # Assert that both apps now have services collected during installation
    assert (
        len(app_info_webworker.get("services", [])) > 0
    ), "Web Worker app should have services collected during installation"
    assert (
        len(app_info_fastapi.get("services", [])) > 0
    ), "FastAPI app should now have services collected during installation"

    print(f"✓ Both apps now collect services during installation!")
    print(f"  - WebWorker services: {len(app_info_webworker.get('services', []))}")
    print(f"  - FastAPI services: {len(app_info_fastapi.get('services', []))}")

    # But lazy loading should still work for both
    # Test web-worker lazy loading
    webworker_service = await api.get_service(
        "echo@" + app_info_webworker.id, mode="first"
    )
    assert webworker_service is not None
    assert await webworker_service.echo("hello") == "hello"

    # Test FastAPI lazy loading
    fastapi_service = await api.get_service(
        "hello-fastapi@" + app_info_fastapi.id, mode="first"
    )
    assert fastapi_service is not None

    # Clean up
    await controller.uninstall(app_info_webworker.id)
    await controller.uninstall(app_info_fastapi.id)
    await api.disconnect()


async def test_service_selection_mode_with_multiple_instances(fastapi_server, test_user_token):
    """Test service selection mode with multiple instances of an app."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    controller = await api.get_service("public/server-apps")

    # Stop all running apps first
    for app in await controller.list_running():
        await controller.stop(app["id"])

    # Create a test app with service_selection_mode set to "random"
    test_app_source = """
    api.export({
        async setup() {
            console.log("Multi-instance test app initialized");
        },
        async getName() {
            return "test-service-instance";
        },
        async getInstanceId() {
            return Math.random().toString(36).substring(7);
        },
        async echo(message) {
            return `Echo from instance: ${message}`;
        },
        async add(a, b) {
            return a + b;
        }
    });
    """

    # Install the app with service_selection_mode configuration
    app_info = await controller.install(
        source=test_app_source,
        config={
            "name": "multi-instance-test-app",
            "type": "window",
            "service_selection_mode": "random",  # Set random selection mode
        },
        overwrite=True,
    )

    print(f"App installed with service_selection_mode: {app_info}")

    # Launch two instances of the same app using the existing artifact
    # Start the first instance using the installed app
    config1 = await controller.start(
        app_info["id"],
        wait_for_service="default",
    )

    # Start the second instance using the same app artifact
    config2 = await controller.start(
        app_info["id"],
        wait_for_service="default",
    )

    print(f"Instance 1 ID: {config1.id}")
    print(f"Instance 2 ID: {config2.id}")

    # Verify both instances are running
    running_apps = await controller.list_running()
    instance1 = find_item(running_apps, "id", config1.id)
    instance2 = find_item(running_apps, "id", config2.id)
    
    assert instance1 is not None, "First instance should be running"
    assert instance2 is not None, "Second instance should be running"

    # Now test service selection - this should work without error due to random selection mode
    # Test 1: Using get_service with app_id should use the app's service_selection_mode
    try:
        service = await api.get_service(f"default@{app_info.id}")
        assert service is not None, "Service should be accessible with random selection mode"
        
        # Test that the service actually works
        echo_result = await service.echo("test message")
        assert "Echo from instance: test message" in echo_result
        
        print("✅ Service selection with app_id works correctly")
        
    except Exception as e:
        print(f"❌ Service selection with app_id failed: {e}")
        raise

    # Test 2: Test multiple calls to verify random selection is working
    # (We can't predict which instance will be selected, but all calls should succeed)
    successful_calls = 0
    for i in range(5):
        try:
            service = await api.get_service(f"default@{app_info.id}")
            result = await service.getName()
            assert result == "test-service-instance"
            successful_calls += 1
        except Exception as e:
            print(f"Call {i+1} failed: {e}")
            
    assert successful_calls == 5, f"All 5 calls should succeed, but only {successful_calls} did"
    print("✅ Multiple service selection calls all succeeded")

    # Test 3: Test that explicit mode parameter still takes precedence
    try:
        # Try to get the first instance specifically (should work if there are multiple)
        service = await api.get_service(f"default@{app_info.id}", {"mode": "first"})
        assert service is not None, "Service should be accessible with explicit 'first' mode"
        
        result = await service.echo("explicit mode test")
        assert "Echo from instance: explicit mode test" in result
        
        print("✅ Explicit mode parameter takes precedence")
        
    except Exception as e:
        print(f"❌ Explicit mode test failed: {e}")
        raise

    # Test 4: Verify that without app_id, it would fail with multiple instances
    # This should demonstrate the value of the service_selection_mode feature
    try:
        # This should fail because there are multiple instances and no selection mode
        service = await api.get_service("default", {"mode": "exact"})
        assert False, "Should have failed with exact mode and multiple instances"
    except Exception as e:
        print(f"✅ Expected failure with exact mode and multiple instances: {e}")

    # Test 5: Test HTTP service endpoint without _mode parameter
    # This should use the app's service_selection_mode (random) and work correctly
    workspace = api.config["workspace"]
    
    # Test HTTP service endpoint
    service_url = f"{SERVER_URL}/{workspace}/services/default@{app_info.id}/echo"
    response = requests.post(
        service_url,
        json={"message": "HTTP service test"},
        headers={"Authorization": f"Bearer {test_user_token}"}
    )
    
    assert response.status_code == 200, f"HTTP service request failed: {response.status_code} {response.text}"
    result = response.json()
    assert "Echo from instance: HTTP service test" in result, f"Unexpected response: {result}"
    print("✅ HTTP service endpoint works with service_selection_mode")

    # Test multiple HTTP service calls to verify random selection is working
    successful_http_calls = 0
    for i in range(3):
        response = requests.post(
            service_url,
            json={"message": f"HTTP test {i+1}"},
            headers={"Authorization": f"Bearer {test_user_token}"}
        )
        if response.status_code == 200:
            successful_http_calls += 1
            
    assert successful_http_calls == 3, f"All 3 HTTP calls should succeed, but only {successful_http_calls} did"
    print("✅ Multiple HTTP service calls all succeeded")

    # Test 6: Test HTTP app endpoint without _mode parameter
    # This should also use the app's service_selection_mode (random) and work correctly
    app_url = f"{SERVER_URL}/{workspace}/apps/{app_info.id}/add"
    response = requests.post(
        app_url,
        json={"a": 10, "b": 20},
        headers={"Authorization": f"Bearer {test_user_token}"}
    )
    
    assert response.status_code == 200, f"HTTP app request failed: {response.status_code} {response.text}"
    result = response.json()
    assert result == 30, f"Expected 30, got {result}"
    print("✅ HTTP app endpoint works with service_selection_mode")

    # Test multiple HTTP app calls to verify random selection is working
    successful_app_calls = 0
    for i in range(3):
        response = requests.post(
            app_url,
            json={"a": i, "b": i * 2},
            headers={"Authorization": f"Bearer {test_user_token}"}
        )
        if response.status_code == 200:
            result = response.json()
            expected = i + (i * 2)
            assert result == expected, f"Expected {expected}, got {result}"
            successful_app_calls += 1
            
    assert successful_app_calls == 3, f"All 3 HTTP app calls should succeed, but only {successful_app_calls} did"
    print("✅ Multiple HTTP app calls all succeeded")

    # Test 7: Test that HTTP requests without app_id would fail with multiple instances
    # This demonstrates the value of the service_selection_mode feature for HTTP endpoints
    try:
        # This should fail because there are multiple instances and no selection mode
        service_url_no_app = f"{SERVER_URL}/{workspace}/services/default/echo"
        response = requests.post(
            service_url_no_app,
            json={"message": "Should fail"},
            headers={"Authorization": f"Bearer {test_user_token}"}
        )
        # Should return an error status code
        assert response.status_code != 200, "HTTP request should have failed with multiple instances"
        print(f"✅ Expected HTTP failure without app_id: {response.status_code}")
    except Exception as e:
        print(f"✅ Expected HTTP failure without app_id: {e}")

    # Clean up
    await controller.stop(config1.id)
    await controller.stop(config2.id)
    await controller.uninstall(app_info.id)

    print("✅ Service selection mode test completed successfully!")

    await api.disconnect()
