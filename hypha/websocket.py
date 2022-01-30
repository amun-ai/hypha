"""Provide an s3 interface."""
import asyncio
import logging
import sys

from fastapi import Query, WebSocket, status
from starlette.websockets import WebSocketDisconnect

from hypha.core import ClientInfo, UserInfo
from hypha.core.store import RedisRPCConnection
from hypha.core.auth import parse_reconnection_token, parse_token
import shortuuid

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket-server")
logger.setLevel(logging.INFO)


class WebsocketServer:
    """Represent an Websocket server."""

    # pylint: disable=too-many-statements

    def __init__(self, store, path="/ws", allow_origins="*") -> None:
        """Set up the websocket server."""
        if allow_origins == ["*"]:
            allow_origins = "*"

        self.store = store
        app = store._app

        @app.websocket(path)
        async def websocket_endpoint(
            websocket: WebSocket,
            workspace: str = Query(None),
            client_id: str = Query(None),
            token: str = Query(None),
            reconnection_token: str = Query(None),
        ):
            async def disconnect(code):
                logger.info(f"Disconnecting {code}")
                await websocket.close(code)

            if client_id is None:
                logger.error("Missing query parameters: workspace, client_id")
                await disconnect(code=status.WS_1003_UNSUPPORTED_DATA)
                return

            if reconnection_token:
                logger.info(f"Reconnecting client via token: {reconnection_token}")
                uid, cid = parse_reconnection_token(reconnection_token)
                assert cid == client_id
                user_info = await store.get_user(uid)
                assert user_info is not None, "User not found: " + uid
                logger.info("Client successfully reconnected: %s", cid)
            else:
                if token:
                    try:
                        user_info = parse_token(token)
                        await store.register_user(user_info)
                        uid = user_info.id
                    except Exception:
                        logger.error("Invalid token: %s", token)
                        await disconnect(code=status.WS_1003_UNSUPPORTED_DATA)
                        return
                else:
                    uid = shortuuid.uuid()
                    user_info = UserInfo(
                        id=uid,
                        is_anonymous=True,
                        email=None,
                        parent=None,
                        roles=[],
                        scopes=[],
                        expires_at=None,
                    )
                    await store.register_user(user_info)
                    logger.info("Anonymized User connected: %s", uid)

            if workspace is None:
                workspace = uid
                persistent = not user_info.is_anonymous
                # If the user disconnected unexpectedly, the workspace will be preserved
                if not await store.get_user_workspace(uid):
                    try:
                        await store.register_workspace(
                            dict(
                                name=uid,
                                owners=[uid],
                                visibility="protected",
                                persistent=persistent,
                                read_only=False,
                            ),
                            overwrite=False,
                        )
                    except Exception as exp:
                        logger.error("Failed to create user workspace: %s", exp)
                        await disconnect(code=status.WS_1003_UNSUPPORTED_DATA)
                        return
            try:
                workspace_manager = await store.get_workspace_manager(
                    workspace, setup=True
                )
            except Exception as exp:
                logger.error(
                    "Failed to get workspace manager %s, error: %s", workspace, exp
                )
                await disconnect(code=status.WS_1003_UNSUPPORTED_DATA)
                return
            if not await workspace_manager.check_permission(user_info):
                logger.error(
                    "Permission denied (client: %s, workspace: %s)",
                    client_id,
                    workspace,
                )
                await disconnect(code=status.WS_1003_UNSUPPORTED_DATA)
                return

            if not reconnection_token and await workspace_manager.check_client_exists(
                client_id
            ):
                logger.error(
                    "Another client with the same id %s already connected to workspace: %s",
                    client_id,
                    workspace,
                )
                await disconnect(code=status.WS_1013_TRY_AGAIN_LATER)
                # await workspace_manager.delete_client(client_id)
                return

            await websocket.accept()

            conn = RedisRPCConnection(
                workspace_manager._redis,
                workspace_manager._workspace,
                client_id,
                user_info,
            )
            conn.on_message(websocket.send_bytes)

            if not reconnection_token:
                await workspace_manager.register_client(
                    ClientInfo(
                        id=client_id,
                        workspace=workspace_manager._workspace,
                        user_info=user_info,
                    )
                )
            try:
                while True:
                    data = await websocket.receive_bytes()
                    await conn.emit_message(data)
            except WebSocketDisconnect as exp:
                if exp.code in [
                    status.WS_1000_NORMAL_CLOSURE,
                    status.WS_1001_GOING_AWAY,
                ]:
                    # Clean up if the client is disconnected normally
                    # TODO: clean up if the client never come back
                    logger.info("Client disconnected: %s", client_id)
                    await workspace_manager.delete_client(client_id)

                    if user_info.is_anonymous:
                        remain_clients = await workspace_manager.list_user_clients(
                            {"user": user_info.dict()}
                        )
                        if len(remain_clients) <= 0:
                            await store.delete_user(user_info.id)
                            logger.info("Anonymous user (%s) disconnected.", uid)
                        else:
                            logger.info(
                                "Anonymous user (%s) client disconnected (remaining clients: %s)",
                                uid,
                                len(remain_clients),
                            )
                    workspace_info = await workspace_manager.get_workspace_info()
                    if user_info.is_anonymous:
                        await workspace_manager.delete()
                    elif not workspace_info.persistent:
                        await workspace_manager.delete_if_empty()
                else:
                    logger.error(
                        "Websocket (client=%s) disconnected unexpectedly: %s",
                        client_id,
                        exp,
                    )

    async def is_alive(self):
        """Check if the server is alive."""
        return True
