"""Test the hypha messages."""
import asyncio

import pytest
from imjoy_rpc.hypha.websocket_client import connect_to_server

from . import (
    WS_SERVER_URL,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio

async def test_queue(fastapi_server):
    api = await connect_to_server({"name": "my plugin", "server_url": WS_SERVER_URL})
    token = await api.generate_token()
    api2 = await connect_to_server({"name": "my plugin 2", "server_url": WS_SERVER_URL, "workspace": api.config["workspace"], "token": token})
    
    q = await api.get_service("queue")
    await q.push("test", {"hello": "world"})
    task = await q.pop("test")
    assert task["hello"] == "world"
    
    await q.push("test", {"hello": "world"})
    tasks = await q.peek("test")
    assert tasks[0]["hello"] == "world"
    
    length = await q.get_length("test")
    assert length == 1
    
    await q.push("test", {"hello": "world"})
    await q.push("test", {"hello": "world"})
    length = await q.get_length("test")
    assert length == 3
    
    q2 = await api2.get_service("queue")

    tasks = await q2.peek("test", 2)
    assert len(tasks) == 2
    assert tasks[0]["hello"] == "world"
    assert tasks[1]["hello"] == "world"
    
    task = await q2.pop("test")
    assert task["hello"] == "world"
    task = await q2.pop("test")
    assert task["hello"] == "world"
    task = await q2.pop("test")
    assert task["hello"] == "world"
