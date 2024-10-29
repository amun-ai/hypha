"""Test the hypha messages."""
import asyncio

import pytest
from hypha_rpc import connect_to_server

from . import (
    WS_SERVER_URL,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_client_to_client(fastapi_server):
    api = await connect_to_server({"name": "my app", "server_url": WS_SERVER_URL})
    token = await api.generate_token()
    api2 = await connect_to_server(
        {
            "name": "my app 2",
            "server_url": WS_SERVER_URL,
            "workspace": api.config["workspace"],
            "token": token,
        }
    )
    fut = asyncio.Future()

    def on_message(data):
        fut.set_result(data)

    api2.on("boradcast-message", on_message)
    await api.emit({"type": "boradcast-message", "to": "*", "message": "hello world"})
    message = await fut
    assert message["type"] == "boradcast-message"


async def test_broadcasting(fastapi_server):
    """Test connecting to the server."""
    api = await connect_to_server({"name": "my app", "server_url": WS_SERVER_URL})
    fut = asyncio.Future()

    def on_message(data):
        fut.set_result(data)

    api.on("message", on_message)
    await api.emit({"type": "message", "to": "*", "message": "hello world"})
    message = await fut
    assert message["message"] == "hello world"


async def test_specific_client(fastapi_server):
    api = await connect_to_server({"name": "my app", "server_url": WS_SERVER_URL})
    fut = asyncio.Future()

    def on_message(data):
        fut.set_result(data)

    api.on("self-message", on_message)
    await api.emit(
        {"type": "self-message", "to": api.config.client_id, "message": "hello world"}
    )
    message = await fut
    assert message["type"] == "self-message"
