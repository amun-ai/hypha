"""Test the hypha server."""
import asyncio
import os
import subprocess
import sys
import time

import pytest
import requests
from imjoy_rpc.hypha.websocket_client import connect_to_server
from requests import RequestException

from . import SIO_PORT, WS_SERVER_URL
from hypha.utils import launch_external_services


@pytest.mark.asyncio
async def test_external_services(fastapi_server):
    """Test the external services."""
    server = await connect_to_server(
        {
            "name": "my second plugin",
            "server_url": WS_SERVER_URL,
        }
    )
    internal_service = await server.get_service("test-service")
    assert internal_service.id == "test-service"
    assert await internal_service.test(1) == 100


def test_module_startup_function():
    """Test the wrong service config."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT+9}",
            "--reset-redis",
            "--startup-function-uri=hypha.example_startup:hypha_startup",
        ],
        env=os.environ.copy(),
    ) as proc:
        timeout = 1
        while timeout > 0:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{SIO_PORT+9}/health/services_loaded"
                )
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(1)
        assert (
            timeout <= 0
        )  # The server should fail to start due to wrong service config
        proc.kill()
        proc.terminate()


@pytest.mark.asyncio
async def test_launch_external_services(fastapi_server):
    """Test the launch command utility fuction."""
    server = await connect_to_server(
        {
            "name": "my third plugin",
            "server_url": WS_SERVER_URL,
        }
    )
    proc = await launch_external_services(
        server,
        "python ./tests/example_service_script.py --server-url={server_url} --service-id=external-test-service --workspace={workspace} --token={token}",
        name="example_service_script",
        check_services=["external-test-service"],
    )
    external_service = await server.get_service("external-test-service")
    assert external_service.id == "external-test-service"
    assert await external_service.test(1) == 100
    await proc.kill()
    with pytest.raises(
        Exception, match=r".*IndexError: Service not found: external-test-service.*"
    ):
        await server.get_service("external-test-service")

    proc = await launch_external_services(
        server,
        "python -m http.server 9391",
        name="example_service_script",
        check_url="http://127.0.0.1:9391",
    )
    assert await proc.ready()
    await proc.kill()
