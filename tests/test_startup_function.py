"""Test the hypha server."""
import asyncio
import os
import subprocess
import sys
import time

import pytest
import requests
from hypha_rpc import connect_to_server
from requests import RequestException

from . import SIO_PORT, WS_SERVER_URL


@pytest.mark.asyncio
async def test_external_services(fastapi_server):
    """Test the external services."""
    server = await connect_to_server(
        {
            "name": "my second app",
            "server_url": WS_SERVER_URL,
        }
    )
    svc = await server.get_service("public/test-service")
    assert svc.id.endswith("test-service")
    assert await svc.test(1) == 100

    svc = await server.get_service("public/example-startup-service")
    assert await svc.test(3) == 22 + 3


def test_failed_startup_function():
    """Test failed startup function from a module."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT+10}",
            "--reset-redis",
            "--startup-functions=hypha.non_existing_module:non_existing_function",
        ],
        env=os.environ.copy(),
    ) as proc:
        timeout = 1
        while timeout > 0:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{SIO_PORT+10}/health/readiness", timeout=1
                )
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(1)
        proc.kill()
        proc.terminate()
        assert timeout < 0  # The server should fail
