"""Provide an s3 interface."""
import asyncio
import logging
import sys

from fastapi import Query, WebSocket, status
from starlette.websockets import WebSocketDisconnect

from hypha.core import ClientInfo
from hypha.core.store import RedisRPCConnection, RedisStore

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket")
logger.setLevel(logging.INFO)


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

            conn = RedisRPCConnection(
                workspace_manager._redis,
                workspace_manager._workspace,
                client_id,
            )
            conn.on_message(websocket.send_bytes)

            await workspace_manager.register_client(ClientInfo(id=client_id))
            try:
                while True:
                    data = await websocket.receive_bytes()
                    await conn.emit_message(data)
            except WebSocketDisconnect as exp:
                if exp.code != status.WS_1000_NORMAL_CLOSURE:
                    logger.warning(
                        f"websocket disconnect from the server (code={exp.code})"
                    )
                await workspace_manager.delete_client(client_id)

    async def is_alive(self):
        """Check if the server is alive."""
        return True
