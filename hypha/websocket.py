"""Provide an s3 interface."""
import asyncio
import logging
import sys

import msgpack
import websockets
from fastapi import Query, WebSocket, status
from imjoy_rpc.core_connection import BasicConnection
from starlette.websockets import WebSocketDisconnect

from hypha.core import ClientInfo
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
                    try:
                        if data.get("type") == "disconnect":
                            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                            return
                        if "to" not in data:
                            raise ValueError("`to` is required")
                        target_id = data["to"].encode()
                        encoded = (
                            len(target_id).to_bytes(1, "big")
                            + target_id
                            + msgpack.packb(data)
                        )
                        await websocket.send(encoded)
                    except Exception:
                        logger.exception(f"Failed to send data to {data.get('to')}")

                connection._send = send

                if not fut.done():
                    rpc = RPC(
                        connection,
                        client_id=client_id,
                        root_target_id="workspace-manager",
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
            workspace_manager = await store.get_workspace_manager(workspace)
            if await workspace_manager.check_client_exists(client_id):
                logger.error(
                    "Another client with the same id %s already connected to workspace: %s",
                    client_id,
                    workspace,
                )
                await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
                # await workspace_manager.delete_client(client_id)
                return

            await websocket.accept()

            asyncio.ensure_future(
                workspace_manager.listen(
                    client_id,
                    websocket.send_bytes,
                    unpack=False,
                    is_async=True,
                )
            )

            await workspace_manager.register_client(ClientInfo(id=client_id))
            try:
                while True:
                    data = await websocket.receive_bytes()
                    await workspace_manager.send(data)
            except WebSocketDisconnect as exp:
                if exp.code != status.WS_1000_NORMAL_CLOSURE:
                    logger.warning(
                        f"websocket disconnect from the server (code={exp.code})"
                    )
                await workspace_manager.delete_client(client_id)

    async def is_alive(self):
        """Check if the server is alive."""
        return True
