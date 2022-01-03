"""Provide an s3 interface."""
import asyncio
import logging
import sys

import msgpack
import websockets
from fastapi import Query, WebSocket, status
from imjoy_rpc.core_connection import BasicConnection
from starlette.websockets import WebSocketDisconnect

from hypha.core.rpc import RPC
from hypha.core.store import RedisStore


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket")
logger.setLevel(logging.INFO)


async def connect_to_websocket(
    url: str, workspace: str, client_id: str, token: str, timeout=None
):
    """Connect to RPC via a websocket server."""
    connection = BasicConnection(None)
    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    async def listen():
        try:
            async for websocket in websockets.connect(
                url + f"/ws?workspace={workspace}&client_id={client_id}&token={token}"
            ):

                async def send(data):
                    if data.get("type") == "disconnect":
                        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                        return
                    if "target" not in data:
                        raise ValueError("target is required")
                    target_id = data["target"].encode()
                    encoded = (
                        len(target_id).to_bytes(1, "big")
                        + target_id
                        + msgpack.packb(data)
                    )
                    await websocket.send(encoded)

                connection._send = send

                if not fut.done():
                    rpc = RPC(
                        connection,
                        client_id=client_id,
                        root_target_id="/",
                        default_context={"connection_type": "websocket"},
                    )
                    fut.set_result(rpc)

                try:
                    while True:
                        data = await websocket.recv()
                        data = msgpack.unpackb(data)
                        connection.handle_message(data)
                except websockets.ConnectionClosed:
                    if websocket.close_code not in [status.WS_1000_NORMAL_CLOSURE]:
                        logger.error(
                            f"websocket client disconnect (code={websocket.close_code})"
                        )
                    else:
                        break
        except Exception as exp:
            if not fut.done():
                fut.set_exception(exp)

    asyncio.ensure_future(listen())
    return await asyncio.wait_for(fut, timeout=timeout)


class WebsocketServer:
    """Represent an Websocket server."""

    # pylint: disable=too-many-statements

    def __init__(self, core_interface, path="/ws", allow_origins="*") -> None:
        """Set up the socketio server."""
        if allow_origins == ["*"]:
            allow_origins = "*"

        store = RedisStore.get_instance()

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
                logger.error("Missing query parameters: workspace, client_id and token")
                await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
                return
            try:
                workspace = store.get_workspace(workspace)
            except Exception:
                logger.exception("Invalid workspace: %s", workspace)
                await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
                return

            # if store.check_client_exists(workspace.name + ":" + client_id):
            #     logger.error("Another client with the same id %s already connected to workspace: %s", client_id, workspace)
            #     await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
            #     # store.delete_client(workspace.name + ":" + client_id)
            #     return

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

            store.register_client({"id": workspace.name + ":" + client_id})
            # await websocket.close(code=1001)
            try:
                while True:
                    data = await websocket.receive_bytes()
                    target_len = int.from_bytes(data[0:1], "big")
                    target_id = data[1 : target_len + 1].decode()
                    store.redis.publish(
                        f"{workspace.name}:msg:{target_id}", data[target_len + 1 :]
                    )
            except WebSocketDisconnect as exp:
                if exp.code != status.WS_1000_NORMAL_CLOSURE:
                    logger.warning(
                        f"websocket disconnect from the server (code={exp.code})"
                    )
                store.delete_client(workspace.name + ":" + client_id)

    async def is_alive(self):
        """Check if the server is alive."""
        return True
