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


@pytest.mark.asyncio
async def test_external_services(fastapi_server):
    """Test the external services."""
    api = await connect_to_server(
        {
            "name": "my second plugin",
            "server_url": WS_SERVER_URL,
        }
    )
    internal_service = await api.get_service("internal-test-service")
    assert internal_service.id == "internal-test-service"
    # Wait for the external service to be ready
    # The ready check function need at least 5 seconds to be called
    await asyncio.sleep(5)
    external_service = await api.get_service("external-test-service")
    assert external_service.id == "external-test-service"
    assert external_service.test("hello")


def test_wrong_service_config():
    """Test the wrong service config."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT+9}",
            "--reset-redis",
            "--services-config=./tests/test_services_config_failed.yaml",
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
