"""Test the hypha server."""
import os
import subprocess
import sys
import asyncio

import pytest
import requests
from hypha_rpc.websocket_client import connect_to_server

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


async def test_connect_to_server_two_client_same_id(fastapi_server):
    """Test connecting to the server with the same client id."""
    api1 = await connect_to_server(
        {"name": "my app", "server_url": WS_SERVER_URL, "client_id": "my-app"}
    )
    token = await api1.generate_token()
    assert "@hypha@" in token
