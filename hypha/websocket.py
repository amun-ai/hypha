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
    def __init__(self, store, path="/ws"):
        """Initialize websocket server with the store and set up the endpoint."""
        self.store = store
        app = store._app

        @app.websocket(path)
        async def websocket_endpoint(websocket: WebSocket, workspace: str = Query(None), client_id: str = Query(None),
                                     token: str = Query(None), reconnection_token: str = Query(None)):
            await self.handle_websocket_connection(websocket, workspace, client_id, token, reconnection_token)

    async def handle_websocket_connection(self, websocket, workspace, client_id, token, reconnection_token):
        if client_id is None:
            logger.error("Missing query parameters: client_id")
            await self.disconnect(websocket, code=status.WS_1003_UNSUPPORTED_DATA)
            return

        try:
            user_info, parent_client = await self.authenticate_user(token, reconnection_token, client_id, websocket)
            if user_info is None:
                # Authentication failed, disconnect handled within authenticate_user
                return

            workspace_manager = await self.setup_workspace_and_permissions(websocket, user_info, workspace)

            await self.establish_websocket_communication(websocket, workspace_manager, client_id, user_info, parent_client)
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {str(e)}")
            await self.disconnect(websocket, code=status.WS_1011_INTERNAL_ERROR)

    async def authenticate_user(self, token, reconnection_token, client_id, websocket):
        """Authenticate user and handle reconnection or token authentication."""
        # Ensure actual implementation calls for parse_reconnection_token and parse_token
        user_info, parent_client = None, None
        try:
            if reconnection_token:
                user_info, ws, cid = parse_reconnection_token(reconnection_token)
                if cid != client_id or not await self.store.disconnected_client_exists(f"{ws}/{cid}"):
                    logger.error("Reconnection token validation failed")
                    await self.disconnect(websocket, code=status.WS_1003_UNSUPPORTED_DATA)
                    return None, None
                await self.store.remove_disconnected_client(f"{ws}/{cid}")
            else:
                if token:
                    user_info = parse_token(token)
                    parent_client = user_info.get_metadata("parent_client") if user_info else None
                    await self.store.register_user(user_info)
                else:
                    user_info = UserInfo(
                        id=shortuuid.uuid(),
                        is_anonymous=True,
                        email=None,
                        parent=None,
                        roles=[],
                        scopes=[],
                        expires_at=None,
                    )
                    await self.store.register_user(user_info)
            return user_info, parent_client
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            await self.disconnect(websocket, code=status.WS_1003_UNSUPPORTED_DATA)
            return None, None

    async def setup_workspace_and_permissions(self, websocket, user_info, workspace):
        """Setup workspace and check permissions."""
        if workspace is None:
            workspace = user_info.id

        # Ensure calls to store for workspace existence and permissions check
        workspace_exists = await self.store.workspace_exists(workspace)
        if not workspace_exists:
            # Simplified logic for workspace creation, ensure this matches the actual store method signatures
            await self.store.register_workspace({
                "name": workspace,
                "persistent": False,  # Determine based on actual logic
                "owners": [user_info.id],
                "visibility": "protected",
                "read_only": user_info.is_anonymous,
            })
        workspace_manager = await self.store.get_workspace_manager(
            workspace, setup=True
        )

        # Permission check using actual store or workspace_manager methods
        if not await workspace_manager.check_permission(user_info):
            logger.error(f"Permission denied for workspace: {workspace}")
            await self.disconnect(websocket, code=status.WS_1003_UNSUPPORTED_DATA)
        else:
            return workspace_manager

    async def establish_websocket_communication(self, websocket, workspace_manager, client_id, user_info, parent_client):
        """Establish and manage websocket communication."""
        try:
            winfo = await workspace_manager.get_workspace_info()
            workspace = winfo.name
            await websocket.accept()
            conn = RedisRPCConnection(self.store._redis, workspace, client_id, user_info)
            conn.on_message(websocket.send_bytes)
            client_info = ClientInfo(
                id=client_id,
                parent=parent_client,
                workspace=workspace,
                user_info=user_info,
            )
            await workspace_manager.register_client(client_info)

            while True:
                data = await websocket.receive_bytes()
                await conn.emit_message(data)
        except WebSocketDisconnect as exp:
            logger.info(f"Client disconnected: {workspace}/{client_id}, code: {exp.code}")
            await self.handle_disconnection(workspace_manager, workspace, client_id, exp.code)
        except Exception as e:
            logger.error(f"Error handling WebSocket communication: {str(e)}")
            await self.disconnect(websocket, code=status.WS_1011_INTERNAL_ERROR)
    
    async def handle_disconnection(self, workspace_manager, workspace, client_id, code):
        """Handle client disconnection."""
        try:
            if code in [status.WS_1000_NORMAL_CLOSURE, status.WS_1001_GOING_AWAY]:
                await workspace_manager.delete_client(client_id)
            else:
                await self.store.add_disconnected_client(ClientInfo(id=client_id, workspace=workspace))
        except Exception as e:
            logger.error(f"Error handling disconnection: {str(e)}")

    async def disconnect(self, websocket, code=status.WS_1000_NORMAL_CLOSURE):
        """Disconnect the websocket connection."""
        await websocket.close(code)

    async def is_alive(self):
        """Check if the server is alive."""
        return True
