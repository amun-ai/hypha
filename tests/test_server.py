"""Test the hypha server."""
import os
import subprocess
import sys
import asyncio

import pytest
import requests
from hypha_rpc import connect_to_server

from . import (
    SERVER_URL,
    SERVER_URL_REDIS_1,
    SERVER_URL_REDIS_2,
    SIO_PORT2,
    WS_SERVER_URL,
    find_item,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_connect_to_server(fastapi_server):
    """Test connecting to the server."""

    class HyphaApp:
        """Represent a test app."""

        def __init__(self, ws):
            self._ws = ws

        async def setup(self):
            """Set up the app."""
            await self._ws.log("initialized")

        async def run(self, ctx):
            """Run the app."""
            await self._ws.log("hello world")

    # test workspace is an exception, so it can pass directly
    # rpc = await connect_to_server(
    #     {"name": "my app", "workspace": "public", "server_url": WS_SERVER_URL}
    # )
    with pytest.raises(
        Exception,
        match=r".*does not exist or is not accessible.*",
    ):
        rpc = await connect_to_server(
            {"name": "my app", "workspace": "test", "server_url": WS_SERVER_URL}
        )
    wm = await connect_to_server({"name": "my app", "server_url": WS_SERVER_URL})
    assert wm.config.workspace is not None
    assert wm.config.hypha_version
    assert wm.config.public_base_url.startswith("http")
    rpc = wm.rpc
    service_info = await rpc.register_service(HyphaApp(wm))
    service = await wm.get_service(service_info.id)
    assert await service.run(None) is None
    await wm.log("hello")


async def test_schema(fastapi_server):
    """Test schema."""
    api = await connect_to_server(
        {"name": "my app", "server_url": WS_SERVER_URL, "client_id": "my-app"}
    )
    for k in api:
        if callable(api[k]):
            assert (
                hasattr(api[k], "__schema__") and api[k].__schema__ is not None
            ), f"Schema not found for {k}"
            assert api[k].__schema__.get("name") == k, f"Schema name not match for {k}"


async def test_connect_to_server_two_client_same_id(fastapi_server):
    """Test connecting to the server with the same client id."""
    api1 = await connect_to_server(
        {"name": "my app", "server_url": WS_SERVER_URL, "client_id": "my-app"}
    )
    token = await api1.generate_token()
    try:
        api2 = await connect_to_server(
            {
                "name": "my app",
                "server_url": WS_SERVER_URL,
                "client_id": "my-app",
                "token": token,
                "workspace": api1.config.workspace,
            }
        )
        await api2.disconnect()
    except Exception as e:
        assert "Client already exists and is active:" in str(e)
    await api1.disconnect()


def test_plugin_runner(fastapi_server):
    """Test the app runner."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.runner",
            f"--server-url={WS_SERVER_URL}",
            "--quit-on-ready",
            os.path.join(os.path.dirname(__file__), "example_plugin.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        out, err = proc.communicate()
        assert err.decode("utf8") == ""
        output = out.decode("utf8")
        assert "This is a test" in output
        assert "echo: a message" in output


async def test_plugin_runner_subpath(fastapi_subpath_server):
    """Test the app runner with subpath server."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.runner",
            f"--server-url=ws://127.0.0.1:{SIO_PORT2}/my/engine/ws",
            "--quit-on-ready",
            os.path.join(os.path.dirname(__file__), "example_plugin.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        out, err = proc.communicate()
        assert err.decode("utf8") == ""
        output = out.decode("utf8")
        assert "This is a test" in output
        assert "echo: a message" in output


async def test_extra_mounts(fastapi_server):
    """Test mounting extra static files."""
    response = requests.get(f"{SERVER_URL}/tests/testWindowPlugin1.imjoy.html")
    assert response.ok
    assert "Test Window Plugin" in response.text


async def test_plugin_runner_workspace(fastapi_server):
    """Test the app runner with workspace."""
    api = await connect_to_server(
        {
            "name": "my second app",
            "server_url": WS_SERVER_URL,
        }
    )
    # Issue an admin token so the plugin can generate new token
    token = await api.generate_token()

    # The following code without passing the token should fail
    # Here we assert the output message contains "permission denied"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.runner",
            f"--server-url={WS_SERVER_URL}",
            f"--workspace={api.config['workspace']}",
            # f"--token={token}",
            "--quit-on-ready",
            os.path.join(os.path.dirname(__file__), "example_plugin.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        out, err = proc.communicate()
        output = out.decode("utf8")
        assert proc.returncode == 1, err.decode("utf8")
        assert err.decode("utf8") == ""
        assert "Permission denied for " in output

    # now with the token, it should pass
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.runner",
            f"--server-url={WS_SERVER_URL}",
            f"--workspace={api.config['workspace']}",
            f"--token={token}",
            "--quit-on-ready",
            os.path.join(os.path.dirname(__file__), "example_plugin.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        out, err = proc.communicate()
        output = out.decode("utf8")
        assert proc.returncode == 0, err.decode("utf8")
        assert err.decode("utf8") == ""
        assert "This is a test" in output
        assert "echo: a message" in output


async def test_services(fastapi_server):
    """Test services."""
    api = await connect_to_server({"name": "my app", "server_url": WS_SERVER_URL})

    token = await api.generate_token()
    assert token is not None

    service_info = await api.register_service(
        {
            "name": "test_service",
            "id": "test_service",
            "type": "test-type",
            "idx": 1,
        }
    )
    service = await api.get_service(service_info.id)
    assert service["name"] == "test_service"
    services = await api.list_services(api.config.workspace)
    assert find_item(services, "name", "test_service")

    service_info = await api.register_service(
        {
            "name": "test_service",
            "id": "test_service",
            "type": "test-type",
            "idx": 2,
        },
        {"overwrite": True},
    )
    # It should be overwritten because it's from the same provider
    assert (
        len(
            await api.list_services(
                {"workspace": api.config.workspace, "id": "test_service"}
            )
        )
        == 1
    )

    api2 = await connect_to_server(
        {
            "name": "my app 2",
            "server_url": WS_SERVER_URL,
            "token": token,
            "workspace": api.config.workspace,
        }
    )
    assert api2.config["workspace"] == api.config["workspace"]
    service_info = await api2.register_service(
        {
            "name": "test_service",
            "id": "test_service",
            "type": "test-type",
            "idx": 3,
        }
    )
    # It should be co-exist because it's from a different provider
    assert (
        len(
            await api.list_services(
                {"workspace": api.config.workspace, "id": "test_service"}
            )
        )
        == 2
    )
    assert (
        len(
            await api2.list_services(
                {"workspace": api2.config.workspace, "id": "test_service"}
            )
        )
        == 2
    )


async def test_workspace_owners(
    fastapi_server, test_user_token, test_user_token_temporary, test_user_token_2
):
    """Test workspace owners."""
    api = await connect_to_server(
        {
            "client_id": "my-app-99",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )

    ws = await api.create_workspace(
        {
            "name": "my-test-workspace-owners",
            "description": "This is a test workspace",
            "owners": ["user-1@test.com", "user-2@test.com"],
        },
    )

    assert ws["name"] == "my-test-workspace-owners"

    api2 = await connect_to_server(
        {
            "server_url": WS_SERVER_URL,
            "token": test_user_token_2,
            "workspace": "my-test-workspace-owners",
        }
    )

    assert api2.config["workspace"] == "my-test-workspace-owners"

    try:
        api3 = await connect_to_server(
            {
                "server_url": WS_SERVER_URL,
                "token": test_user_token_temporary,
                "workspace": "my-test-workspace-owners",
            }
        )
        assert api3.config["workspace"] == "my-test-workspace-owners"
    except Exception as e:
        assert "Permission denied for workspace" in str(e)

    await api2.disconnect()
    await api.disconnect()


async def test_server_scalability(
    fastapi_server_redis_1, fastapi_server_redis_2, test_user_token
):
    """Test services."""
    api = await connect_to_server(
        {
            "client_id": "my-app-99",
            "server_url": SERVER_URL_REDIS_2,
            "token": test_user_token,
        }
    )

    ws = await api.create_workspace(
        {
            "name": "my-test-workspace",
            "description": "This is a test workspace",
            "owners": ["user1@imjoy.io", "user2@imjoy.io"],
        },
        overwrite=True,
    )
    assert ws["name"] == "my-test-workspace"

    token = await api.generate_token({"workspace": "my-test-workspace"})

    # Connect from two different servers
    api88 = await connect_to_server(
        {
            "client_id": "my-app-88",
            "server_url": SERVER_URL_REDIS_1,
            "workspace": "my-test-workspace",
            "token": token,
            "method_timeout": 30,
        }
    )

    api77 = await connect_to_server(
        {
            "client_id": "my-app-77",
            "server_url": SERVER_URL_REDIS_2,
            "workspace": "my-test-workspace",
            "token": token,
            "method_timeout": 30,
        }
    )
    await asyncio.sleep(0.2)

    clients = await api77.list_clients()
    assert find_item(clients, "id", "my-test-workspace/my-app-77")
    assert find_item(clients, "id", "my-test-workspace/my-app-88")

    await api77.register_service(
        {
            "id": "test-service",
            "add77": lambda x: x + 77,
        },
        {"overwrite": True},
    )

    svc = await api88.get_service("my-app-77:test-service")
    assert await svc.add77(1) == 78

    await api.disconnect()
    await api77.disconnect()
    await api88.disconnect()
