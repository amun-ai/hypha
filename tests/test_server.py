"""Test the hypha server."""
import os
import subprocess
import sys
import asyncio


import pytest
from imjoy_rpc import connect_to_server
from . import SIO_PORT, SIO_PORT2, SIO_SERVER_URL

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_connect_to_server(socketio_server):
    """Test connecting to the server."""

    class ImJoyPlugin:
        """Represent a test plugin."""

        def __init__(self, ws):
            self._ws = ws

        async def setup(self):
            """Set up the plugin."""
            await self._ws.log("initialized")

        async def run(self, ctx):
            """Run the plugin."""
            await self._ws.log("hello world")

    # test workspace is an exception, so it can pass directly
    ws = await connect_to_server(
        {"name": "my plugin", "workspace": "public", "server_url": SIO_SERVER_URL}
    )
    with pytest.raises(Exception, match=r".*Workspace test does not exist.*"):
        ws = await connect_to_server(
            {"name": "my plugin", "workspace": "test", "server_url": SIO_SERVER_URL}
        )
    ws = await connect_to_server({"name": "my plugin", "server_url": SIO_SERVER_URL})
    await ws.export(ImJoyPlugin(ws))

    ws = await connect_to_server({"server_url": SIO_SERVER_URL})
    assert len(ws.config.name) == 36


def test_plugin_runner(socketio_server):
    """Test the plugin runner."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.runner",
            f"--server-url=http://127.0.0.1:{SIO_PORT}",
            "--quit-on-ready",
            os.path.join(os.path.dirname(__file__), "example_plugin.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        out, err = proc.communicate()
        assert err.decode("utf8") == ""
        output = out.decode("utf8")
        assert "Generated token: " in output and "@imjoy@" in output
        assert "echo: a message" in output


def test_plugin_runner_subpath(socketio_subpath_server):
    """Test the plugin runner with subpath server."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.runner",
            f"--server-url=http://127.0.0.1:{SIO_PORT2}/my/engine",
            "--quit-on-ready",
            os.path.join(os.path.dirname(__file__), "example_plugin.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        out, err = proc.communicate()
        assert err.decode("utf8") == ""
        output = out.decode("utf8")
        assert "Generated token: " in output and "@imjoy@" in output
        assert "echo: a message" in output


async def test_plugin_runner_workspace(socketio_server):
    """Test the plugin runner with workspace."""
    api = await connect_to_server(
        {
            "name": "my second plugin",
            "server_url": SIO_SERVER_URL,
        }
    )
    token = await api.generate_token()
    assert "@imjoy@" in token

    # The following code without passing the token should fail
    # Here we assert the output message contains "permission denied"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.runner",
            f"--server-url=http://127.0.0.1:{SIO_PORT}",
            f"--workspace={api.config['workspace']}",
            # f"--token={token}",
            "--quit-on-ready",
            os.path.join(os.path.dirname(__file__), "example_plugin.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        out, err = proc.communicate()
        assert proc.returncode == 1
        assert err.decode("utf8") == ""
        output = out.decode("utf8")
        assert "Permission denied for workspace:" in output

    # now with the token, it should pass
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.runner",
            f"--server-url=http://127.0.0.1:{SIO_PORT}",
            f"--workspace={api.config['workspace']}",
            f"--token={token}",
            "--quit-on-ready",
            os.path.join(os.path.dirname(__file__), "example_plugin.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        out, err = proc.communicate()
        assert proc.returncode == 0
        assert err.decode("utf8") == ""
        output = out.decode("utf8")
        assert "Generated token: " in output and "@imjoy@" in output
        assert "echo: a message" in output


async def test_workspace(socketio_server):
    """Test the plugin runner."""
    api = await connect_to_server({"name": "my plugin", "server_url": SIO_SERVER_URL})
    with pytest.raises(
        Exception, match=r".*Scopes must be empty or contains only the workspace name*"
    ):
        await api.generate_token({"scopes": ["test-workspace"]})
    token = await api.generate_token()
    assert "@imjoy@" in token

    ws = await api.create_workspace(
        {
            "name": "test-workspace",
            "owners": ["user1@imjoy.io", "user2@imjoy.io"],
            "allow_list": [],
            "deny_list": [],
            "visibility": "protected",  # or public
        }
    )
    await ws.log("hello")
    service_info = await ws.register_service(
        {
            "name": "test_service",
            "type": "#test",
        }
    )
    service = await ws.get_service(service_info)
    assert service["name"] == "test_service"

    def test(context=None):
        return context

    service_info = await ws.register_service(
        {
            "name": "test_service",
            "type": "#test",
            "config": {"require_context": True},
            "test": test,
        }
    )
    service = await ws.get_service(service_info)
    context = await service.test()
    assert "user_id" in context and "email" in context
    assert service["name"] == "test_service"

    # we should not get it because api is in another workspace
    ss2 = await api.list_services({"type": "#test"})
    assert len(ss2) == 0

    # let's generate a token for the test-workspace
    token = await ws.generate_token()

    # now if we connect directly to the workspace
    # we should be able to get the test-workspace services
    api2 = await connect_to_server(
        {
            "name": "my plugin 2",
            "workspace": "test-workspace",
            "server_url": SIO_SERVER_URL,
            "token": token,
        }
    )
    assert api2.config["workspace"] == "test-workspace"
    await api2.export({"foo": "bar"})
    ss3 = await api2.list_services({"type": "#test"})
    assert len(ss3) == 2

    plugin = await api2.get_plugin("my plugin 2")
    assert plugin.foo == "bar"

    await api2.export({"foo2": "bar2"})
    plugin = await api2.get_plugin("my plugin 2")
    assert plugin.foo is None
    assert plugin.foo2 == "bar2"

    with pytest.raises(Exception, match=r".*Plugin `my plugin 2` not found.*"):
        await api.get_plugin("my plugin 2")

    ws2 = await api.get_workspace("test-workspace")
    assert ws.config == ws2.config

    await ws2.set({"docs": "https://imjoy.io"})
    with pytest.raises(Exception, match=r".*Changing workspace name is not allowed.*"):
        await ws2.set({"name": "new-name"})

    with pytest.raises(Exception):
        await ws2.set({"covers": [], "non-exist-key": 999})

    state = asyncio.Future()

    def set_state(data):
        """Test function for set the state to a value."""
        state.set_result(data)

    await ws2.on("set-state", set_state)

    await ws2.emit("set-state", 9978)

    assert await state == 9978

    await ws2.off("set-state")

    await api.disconnect()
