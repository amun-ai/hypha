"""Test rpc in js."""
from pathlib import Path
import asyncio
import requests

import pytest
from hypha.websocket_client import connect_to_server
from hypha.runner.browser import BrowserAppRunner

from . import WS_SERVER_URL, find_item

# pylint: disable=too-many-statements

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_server_apps(fastapi_server):
    """Test the server apps."""
    brwoser_runner = BrowserAppRunner(self.core_interface, in_docker=self.in_docker)
    await brwoser_runner.initialize()
    # api = await connect_to_server({"name": "test client", "server_url": WS_SERVER_URL})
    # workspace = api.config["workspace"]
    # token = await api.generate_token()
