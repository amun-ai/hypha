import asyncio
import logging
import sys

from fastapi import Query, WebSocket, status
from starlette.websockets import WebSocketDisconnect

from hypha.core import ClientInfo, UserInfo
from hypha.core.store import RedisRPCConnection
from hypha.core.auth import parse_reconnection_token, parse_token
import shortuuid
import json

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket-server")
logger.setLevel(logging.INFO)


class WebsocketServer:
    def __init__(self, store, path="/ws"):
        """Initialize websocket server with the store and set up the endpoint."""
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
            await websocket.accept()
            # If none of the authentication parameters are provided, wait for the first message
            if not workspace and not client_id and not token and not reconnection_token:
                # Wait for the first message which should contain the authentication information
                auth_info = await websocket.receive_text()
                # Parse the authentication information, e.g., JSON with token and/or reconnection_token
                try:
                    auth_data = json.loads(auth_info)
                    token = auth_data.get("token")
                    reconnection_token = auth_data.get("reconnection_token")
                    # Optionally, you can also update workspace and client_id if they are sent in the first message
                    workspace = auth_data.get("workspace")
                    client_id = auth_data.get("client_id")
                except json.JSONDecodeError:
                    logger.error("Failed to decode authentication information")
                    await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
                    return
            else:
                logger.warning(
                    "For enhanced security, it is recommended to use the first message for authentication"
                )
            try:
                await self.handle_websocket_connection(
                    websocket, workspace, client_id, token, reconnection_token
                )
            except Exception as e:
                if websocket.client_state.name == 'CONNECTED' or websocket.client_state.name == 'CONNECTING':
                    logger.error(f"Error handling WebSocket connection: {str(e)}")
                    await self.disconnect(websocket, code=status.WS_1011_INTERNAL_ERROR)

    async def handle_websocket_connection(
        self, websocket, workspace, client_id, token, reconnection_token
    ):
        if client_id is None:
            logger.error("Missing query parameters: client_id")
            await self.disconnect(websocket, code=status.WS_1003_UNSUPPORTED_DATA)
            return

        try:
            user_info, parent_client, workspace = await self.authenticate_user(
                token, reconnection_token, client_id, websocket, workspace
            )
            if user_info is None:
                # Authentication failed, disconnect handled within authenticate_user
                return

            workspace_manager = await self.setup_workspace_and_permissions(
                websocket, user_info, workspace
            )

            await self.establish_websocket_communication(
                websocket,
                workspace_manager,
                client_id,
                user_info,
                parent_client,
                reconnection_token,
            )

        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {str(e)}")
            await self.disconnect(websocket, code=status.WS_1011_INTERNAL_ERROR)

    async def authenticate_user(
        self, token, reconnection_token, client_id, websocket, workspace
    ):
        """Authenticate user and handle reconnection or token authentication."""
        # Ensure actual implementation calls for parse_reconnection_token and parse_token
        user_info, parent_client = None, None
        try:
            if reconnection_token:
                user_info, ws, cid = parse_reconnection_token(reconnection_token)
                parent_client = (
                    user_info.get_metadata("parent_client") if user_info else None
                )
                if workspace and workspace != ws:
                    logger.error("Workspace mismatch, disconnecting")
                    await self.disconnect(
                        websocket, code=status.WS_1003_UNSUPPORTED_DATA
                    )
                    return None, None, None
                elif cid != client_id:
                    logger.error("Client id mismatch, disconnecting")
                    await self.disconnect(
                        websocket, code=status.WS_1003_UNSUPPORTED_DATA
                    )
                    return None, None, None
                elif not await self.store.disconnected_client_exists(f"{ws}/{cid}"):
                    logger.info(
                        "Client has already been removed, recreating user and workspace."
                    )
                    await self.store.register_user(user_info)
                    return user_info, parent_client, ws
                else:
                    await self.store.remove_disconnected_client(f"{ws}/{cid}")
                    logger.info(f"Reconnected, client id: {ws}/{cid}")
                    return user_info, parent_client, ws
            else:
                if token:
                    user_info = parse_token(token)
                    parent_client = (
                        user_info.get_metadata("parent_client") if user_info else None
                    )
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
                return user_info, parent_client, workspace
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            await self.disconnect(websocket, code=status.WS_1003_UNSUPPORTED_DATA)
            return None, None

    async def setup_workspace_and_permissions(self, websocket, user_info, workspace):
        """Setup workspace and check permissions."""
        if workspace is None:
            workspace = user_info.id
        elif user_info.is_anonymous and user_info.parent != "root":
            logger.error("Anonymous users are not allowed to create workspaces")
            raise PermissionError(
                "Anonymous users are not allowed to create workspaces"
            )
        # Anonymous and Temporary users are not allowed to create persistant workspaces
        persistent = not user_info.is_anonymous and "temporary" not in user_info.roles

        # Ensure calls to store for workspace existence and permissions check
        workspace_exists = await self.store.workspace_exists(workspace)
        if not workspace_exists:
            assert workspace == user_info.id, "Workspace does not exist."
            # Simplified logic for workspace creation, ensure this matches the actual store method signatures
            await self.store.register_workspace(
                {
                    "name": workspace,
                    "persistent": persistent,
                    "owners": [user_info.id],
                    "visibility": "protected",
                    "read_only": user_info.is_anonymous,
                }
            )
            logger.info(f"Created workspace: {workspace}")

        workspace_manager = await self.store.get_workspace_manager(
            workspace, setup=True
        )

        # Permission check using actual store or workspace_manager methods
        if not await workspace_manager.check_permission(user_info):
            logger.error(f"Permission denied for workspace: {workspace}")
            await self.disconnect(websocket, code=status.WS_1003_UNSUPPORTED_DATA)
        else:
            return workspace_manager

    async def establish_websocket_communication(
        self,
        websocket,
        workspace_manager,
        client_id,
        user_info,
        parent_client,
        reconnection_token,
    ):
        """Establish and manage websocket communication."""
        winfo = await workspace_manager.get_workspace_info()
        workspace = winfo.name
        try:
            conn = RedisRPCConnection(
                self.store._redis, workspace, client_id, user_info
            )
            conn.on_message(websocket.send_bytes)
            client_info = ClientInfo(
                id=client_id,
                parent=parent_client,
                workspace=workspace,
                user_info=user_info,
            )
            if reconnection_token:
                await self.store.remove_disconnected_client(f"{workspace}/{client_id}")
            else:
                await workspace_manager.register_client(client_info)

            asyncio.ensure_future(workspace_manager.update_client_services(client_info))
            while True:
                data = await websocket.receive_bytes()
                await conn.emit_message(data)
        except WebSocketDisconnect as exp:
            logger.info(
                f"Client disconnected: {workspace}/{client_id}, code: {exp.code}"
            )
            await self.handle_disconnection(
                workspace_manager,
                workspace,
                client_id,
                exp.code,
                parent_client,
                user_info,
            )
        except Exception as e:
            logger.error(f"Error handling WebSocket communication: {str(e)}")
            await self.handle_disconnection(
                workspace_manager,
                workspace,
                client_id,
                status.WS_1011_INTERNAL_ERROR,
                parent_client,
                user_info,
            )

    async def handle_disconnection(
        self, workspace_manager, workspace, client_id, code, parent_client, user_info
    ):
        """Handle client disconnection with delayed removal for unexpected disconnections."""
        try:
            if code in [status.WS_1000_NORMAL_CLOSURE, status.WS_1001_GOING_AWAY]:
                # Client disconnected normally, remove immediately
                logger.info(f"Client disconnected normally: {workspace}/{client_id}")
                try:
                    await workspace_manager.delete_client(client_id, workspace)
                except KeyError:
                    logger.info(
                        "Client already deleted: %s/%s",
                        workspace,
                        client_id,
                    )
                try:
                    # Clean up if the client is disconnected normally
                    await workspace_manager.delete()
                except KeyError:
                    logger.info("Workspace already deleted: %s", workspace)
            else:
                # Client disconnected unexpectedly, mark for delayed removal
                disconnected_client_info = ClientInfo(
                    id=client_id,
                    parent=parent_client,
                    workspace=workspace,
                    user_info=user_info,
                )
                await self.store.add_disconnected_client(disconnected_client_info)

                # Start a delayed task for removal
                asyncio.create_task(self.delayed_remove_client(workspace, client_id))
        except Exception as e:
            logger.error(f"Error handling disconnection: {str(e)}")

    async def delayed_remove_client(self, workspace, client_id):
        """Remove a disconnected client after a delay."""
        delay_period = (
            self.store.disconnect_delay
        )  # Assuming disconnect_delay is defined in the store
        await asyncio.sleep(delay_period)

        # Check if the client is still marked as disconnected before removal
        if await self.store.disconnected_client_exists(f"{workspace}/{client_id}"):
            logger.info(
                f"Removing disconnected client after delay: {workspace}/{client_id}"
            )
            await self.store.remove_disconnected_client(
                f"{workspace}/{client_id}", remove_workspace=True
            )
        else:
            logger.info(f"Client reconnected, not removing: {workspace}/{client_id}")

    async def disconnect(self, websocket, code=status.WS_1000_NORMAL_CLOSURE):
        """Disconnect the websocket connection."""
        await websocket.close(code)

    async def is_alive(self):
        """Check if the server is alive."""
        return True
