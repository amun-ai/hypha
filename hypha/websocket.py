import logging
import sys
import json

from fastapi import Query, WebSocket, status
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
from fastapi import HTTPException

from hypha import __version__
from hypha.core import UserInfo, UserPermission
from hypha.core.store import RedisRPCConnection, RedisStore
from hypha.core.auth import (
    parse_reconnection_token,
    generate_reconnection_token,
    parse_token,
    generate_anonymous_user,
    create_scope,
    update_user_scope,
)


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket-server")
logger.setLevel(logging.INFO)


class WebsocketServer:
    def __init__(self, store: RedisStore, path="/ws"):
        """Initialize websocket server with the store and set up the endpoint."""
        self.store = store
        app = store._app
        self.store.set_websocket_server(self)
        self._stop = False
        self._websockets = {}

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
                    reason = "Failed to decode authentication information"
                    await self.disconnect(
                        websocket,
                        reason=reason,
                        code=status.WS_1003_UNSUPPORTED_DATA,
                    )
                    return
            else:
                logger.warning("Rejecting legacy imjoy-rpc client (%s)", client_id)
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason="Connection rejected, please  upgrade to `hypha-rpc` which pass the authentication information in the first message",
                )
                return

            try:
                if token or reconnection_token:
                    user_info = await self.authenticate_user(
                        token, reconnection_token, client_id, workspace
                    )
                else:
                    user_info = generate_anonymous_user()
                    user_info.scope = create_scope(
                        workspaces={user_info.id: UserPermission.admin},
                        client_id=client_id,
                    )
                workspace_info, user_info = await self.setup_workspace_and_permissions(
                    user_info, workspace, client_id
                )
                user_info.scope = update_user_scope(
                    user_info, workspace_info, client_id
                )
                if not user_info.check_permission(
                    workspace_info.name, UserPermission.read
                ):
                    logger.error(f"Permission denied for workspace: {workspace}")
                    raise PermissionError(
                        f"Permission denied for workspace: {workspace}"
                    )
                workspace = workspace_info.name

                assert workspace_info and workspace and client_id, (
                    "Failed to authenticate user to workspace: %s" % workspace_info.name
                )
                # We operate as the root user to remove and add clients
                await self.check_client(client_id, workspace_info.name, user_info)
            except Exception as e:
                reason = f"Failed to establish connection: {str(e)}"
                await self.disconnect(
                    websocket,
                    reason=reason,
                    code=status.WS_1001_GOING_AWAY,
                )
                return

            try:
                await self.establish_websocket_communication(
                    websocket, workspace_info.name, client_id, user_info
                )
            except RuntimeError as exp:
                # this happens when the websocket is closed
                reason = f"WebSocket runtime error: {str(exp)}"
                await self.handle_disconnection(
                    websocket,
                    workspace,
                    client_id,
                    user_info,
                    status.WS_1001_GOING_AWAY,
                    exp,
                )
            except WebSocketDisconnect as exp:
                reason = f"WebSocket disconnected: {str(exp)}"
                await self.handle_disconnection(
                    websocket,
                    workspace,
                    client_id,
                    user_info,
                    status.WS_1001_GOING_AWAY,
                    exp,
                )
            except KeyError as exp:
                logger.error(f"KeyError: {str(exp)}")
                reason = f"KeyError: {str(exp)}"
                await self.handle_disconnection(
                    websocket,
                    workspace,
                    client_id,
                    user_info,
                    status.WS_1001_GOING_AWAY,
                    exp,
                )
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                await self.handle_disconnection(
                    websocket,
                    workspace,
                    client_id,
                    user_info,
                    status.status.WS_1001_GOING_AWAY,
                    e,
                )

    async def force_disconnect(self, workspace, client_id, code, reason):
        """Force disconnect a client."""
        assert f"{workspace}/{client_id}" in self._websockets, "Client not connected"
        websocket = self._websockets[f"{workspace}/{client_id}"]
        await websocket.close(code=code, reason=reason)

    async def check_client(self, client_id, workspace, user_info):
        """Check if the client is already connected."""
        # check if client already exists
        if await self.store.client_exists(client_id, workspace):
            async with self.store.connect_to_workspace(
                workspace, "check-client-exists", user_info, timeout=5, silent=True
            ) as ws:
                if await ws.ping(f"{workspace}/{client_id}") == "pong":
                    reason = (
                        f"Client already exists and is active: {workspace}/{client_id}"
                    )
                    logger.error(reason)
                    raise RuntimeError(reason)
                else:
                    logger.info(
                        f"Client already exists but is inactive: {workspace}/{client_id}"
                    )
            # remove dead client
            await self.store.remove_client(client_id, workspace, user_info)

    async def authenticate_user(self, token, reconnection_token, client_id, workspace):
        """Authenticate user and handle reconnection or token authentication."""
        # Ensure actual implementation calls for parse_reconnection_token and parse_token
        user_info = None
        try:
            if reconnection_token:
                user_info, ws, cid = parse_reconnection_token(reconnection_token)
                if workspace and workspace != ws:
                    logger.error("Workspace mismatch, disconnecting")
                    raise RuntimeError("Workspace mismatch, disconnecting")
                if cid != client_id:
                    logger.error("Client id mismatch, disconnecting")
                    raise RuntimeError("Client id mismatch, disconnecting")
                logger.info(f"Client reconnected: {ws}/{cid} using reconnection token")
            elif token:
                user_info = parse_token(token)
                # user token doesn't have client id, so we add that
                user_info.scope.client_id = client_id
            else:
                raise RuntimeError("No authentication information provided")
            return user_info
        except HTTPException as e:
            logger.error(f"Authentication error: {e.detail}")
            raise RuntimeError(f"Authentication error: {e.detail}")
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise RuntimeError(f"Authentication error: {str(e)}")

    async def setup_workspace_and_permissions(
        self, user_info: UserInfo, workspace, client_id
    ):
        """Setup workspace and check permissions."""
        if workspace is None:
            workspace = user_info.id

        assert workspace != "*", "Dynamic workspace is not allowed for this endpoint"
        # Anonymous and Temporary users are not allowed to create persistant workspaces
        persistent = (
            not user_info.is_anonymous and "temporary-test-user" not in user_info.roles
        )

        # Ensure calls to store for workspace existence and permissions check
        workspace_info = await self.store.get_workspace(workspace, load=True)
        if not workspace_info:
            assert (
                workspace == user_info.id
            ), "User can only connect to a pre-existing workspace or their own workspace"
            # Simplified logic for workspace creation, ensure this matches the actual store method signatures
            workspace_info = await self.store.register_workspace(
                {
                    "name": workspace,
                    "description": f"Default user workspace for {user_info.id}",
                    "persistent": persistent,
                    "owners": [user_info.id],
                    "read_only": user_info.is_anonymous,
                }
            )
            logger.info(f"Created workspace: {workspace}")
        return workspace_info, user_info

    async def establish_websocket_communication(
        self, websocket, workspace, client_id, user_info
    ):
        """Establish and manage websocket communication."""
        conn = None

        async def force_disconnect(_):
            logger.info(
                f"Unloading workspace, force disconnecting websocket client: {workspace}/{client_id}"
            )
            await self.disconnect(
                websocket, "Workspace unloaded: " + workspace, status.WS_1001_GOING_AWAY
            )

        event_bus = self.store.get_event_bus()
        conn = RedisRPCConnection(event_bus, workspace, client_id, user_info, None)
        self._websockets[f"{workspace}/{client_id}"] = websocket
        try:
            event_bus.on_local(f"unload:{workspace}", force_disconnect)

            async def send_bytes(data):
                try:
                    await websocket.send_bytes(data)
                except ConnectionClosedOK:
                    logger.warning("Failed to send message, closing redis connection")
                    await conn.disconnect("disconnected")

            conn.on_message(send_bytes)
            reconnection_token = generate_reconnection_token(
                user_info, expires_in=self.store.reconnection_token_life_time
            )
            conn_info = {
                "type": "connection_info",
                "hypha_version": __version__,
                "public_base_url": self.store.public_base_url,
                "local_base_url": self.store.local_base_url,
                "manager_id": self.store.manager_id,
                "workspace": workspace,
                "client_id": client_id,
                "user": user_info.model_dump(),
                "reconnection_token": reconnection_token,
            }
            await websocket.send_text(json.dumps(conn_info))
            while not self._stop:
                data = await websocket.receive()
                if "bytes" in data:
                    data = data["bytes"]
                    await conn.emit_message(data)
                elif "text" in data:
                    try:
                        data = data["text"]
                        if len(data) > 1000:
                            logger.warning(
                                "Ignoring long text message: %s...", data[:1000]
                            )
                            continue
                        data = json.loads(data)
                        if data.get("type") == "ping":
                            await websocket.send_text(json.dumps({"type": "pong"}))
                        else:
                            logger.info("Unknown message type: %s", data.get("type"))
                    except json.JSONDecodeError:
                        logger.error("Failed to decode message: %s...", data[:10])

            if self._stop:
                logger.info("Server is stopping, disconnecting client")
                await self.disconnect(
                    websocket, "Server is stopping", status.WS_1001_GOING_AWAY
                )
        except Exception as e:
            raise e
        finally:
            await conn.disconnect("disconnected")
            event_bus.off_local(f"unload:{workspace}", force_disconnect)
            if (
                workspace
                and client_id
                and f"{workspace}/{client_id}" in self._websockets
            ):
                del self._websockets[f"{workspace}/{client_id}"]

    async def handle_disconnection(
        self, websocket, workspace: str, client_id: str, user_info: UserInfo, code, exp
    ):
        """Handle client disconnection with delayed removal for unexpected disconnections."""
        try:
            await self.store.remove_client(client_id, workspace, user_info)
            if code in [status.WS_1000_NORMAL_CLOSURE, status.WS_1001_GOING_AWAY]:
                logger.info(f"Client disconnected normally: {workspace}/{client_id}")
            else:
                logger.info(
                    f"Client disconnected unexpectedly: {workspace}/{client_id}, code: {code}, error: {exp}"
                )
        except Exception as e:
            logger.error(
                f"Error handling disconnection, client: {workspace}/{client_id}, error: {str(e)}"
            )
        finally:
            await self.disconnect(
                websocket,
                "Client disconnected",
                code,
            )

    async def disconnect(self, websocket, reason, code=status.WS_1000_NORMAL_CLOSURE):
        """Disconnect the websocket connection."""
        logger.error("Disconnecting, reason: %s, code: %s", reason, code)
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_text(json.dumps({"type": "error", "message": reason}))
        try:
            if websocket.client_state.name not in ["CLOSED", "CLOSING", "DISCONNECTED"]:
                await websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.error(f"Error disconnecting websocket: {str(e)}")

    async def is_alive(self):
        """Check if the server is alive."""
        return True

    async def stop(self):
        """Stop the server."""
        self._stop = True
