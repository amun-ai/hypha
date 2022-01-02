"""Provide an s3 interface."""
import logging
import sys
from contextvars import copy_context

import shortuuid
from fastapi import HTTPException
from fastapi import WebSocket
from imjoy_rpc.core_connection import BasicConnection
from typing import Optional
from imjoy_rpc.utils import ContextLocal
from hypha.core.rpc import RPC
from fastapi import Cookie, Depends, FastAPI, Query, WebSocket, status
from fastapi.responses import HTMLResponse
import websockets


import asyncio

from hypha.utils import dotdict
from hypha.core.store import store
import msgpack

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket")
logger.setLevel(logging.INFO)


async def connect_to_websocket(url: str, workspace: str, client_id: str, token: str):
    """Connect to RPC via a websocket server."""
    websocket = await websockets.connect(
        url + f"/ws?workspace={workspace}&client_id={client_id}&token={token}"
    )

    async def send(data):
        if "target" not in data:
            raise ValueError("target is required")
        target_id = data["target"].encode()
        encoded = len(target_id).to_bytes(1, "big") + target_id + msgpack.packb(data)
        await websocket.send(encoded)

    connection = BasicConnection(send)

    async def listen():
        while True:
            data = await websocket.recv()
            data = msgpack.unpackb(data)
            connection.handle_message(data)
            await asyncio.sleep(0.01)

    asyncio.ensure_future(listen())

    rpc = RPC(connection, client_id=client_id, default_target_id="core")
    return rpc


class WebsocketServer:
    """Represent an Websocket server."""

    # pylint: disable=too-many-statements

    def __init__(self, core_interface, path="/ws", allow_origins="*") -> None:
        """Set up the socketio server."""
        if allow_origins == ["*"]:
            allow_origins = "*"

        self.core_interface = core_interface
        app = core_interface._app

        @app.websocket(path)
        async def websocket_endpoint(
            websocket: WebSocket,
            workspace: str = Query(None),
            client_id: str = Query(None),
            token: str = Query(None),
        ):
            if workspace is None or client_id is None or token is None:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            try:
                workspace = store.get_workspace(workspace)
            except Exception:
                logger.info("Invalid workspace: %s", workspace)
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)

            await websocket.accept()
            ps = store.redis.pubsub()

            async def listen():
                while True:
                    msg = ps.get_message()
                    if msg and msg.get("type") == "message":
                        assert (
                            msg.get("channel")
                            == f"{workspace.name}:msg:{client_id}".encode()
                        )
                        await websocket.send_bytes(msg["data"])
                    await asyncio.sleep(0.01)

            asyncio.ensure_future(listen())
            ps.subscribe(f"{workspace.name}:msg:{client_id}")

            # Send interface
            # interface = store.get_workspace_interface(
            #     workspace.name, client_id=client_id
            # )
            # data = {"type": "setInterface", "api": interface}
            # await websocket.send_bytes(msgpack.packb(data))

            while True:
                data = await websocket.receive_bytes()
                target_len = int.from_bytes(data[0:1], "big")
                target_id = data[1 : target_len + 1].decode()
                store.redis.publish(
                    f"{workspace.name}:msg:{target_id}", data[target_len + 1 :]
                )

    async def is_alive(self):
        """Check if the server is alive."""
        return True
