import logging
import os
import sys
import json
import msgpack

from fastapi import Query, WebSocket, status
from starlette.websockets import WebSocketDisconnect
from fastapi import HTTPException
from prometheus_client import Gauge

from hypha import __version__
from hypha.core import UserInfo, UserPermission
from hypha.core.store import RedisRPCConnection, RedisStore
from hypha.core.auth import (
    generate_auth_token,
    generate_anonymous_user,
    create_scope,
    update_user_scope,
)


LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("websocket-server")
logger.setLevel(LOGLEVEL)

_gauge = Gauge(
    "websocket_connections", "Total number of websocket connections"
)


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
                try:
                    await websocket.close(
                        code=status.WS_1008_POLICY_VIOLATION,
                        reason="Connection rejected, please  upgrade to `hypha-rpc` which pass the authentication information in the first message",
                    )
                except GeneratorExit:
                    pass
                return

            # Track if we successfully authenticated for cleanup purposes
            authenticated = False
            user_info = None
            
            try:
                if token or reconnection_token:
                    user_info, workspace = await self.authenticate_user(
                        token, reconnection_token, client_id, workspace
                    )
                else:
                    user_info = generate_anonymous_user()
                    # Each anonymous user gets their own workspace based on their ID
                    user_workspace = user_info.get_workspace()
                    # Use the requested workspace if specified, otherwise use the user's own workspace
                    workspace = workspace or user_workspace
                    # Anonymous users get admin permission for their own workspace
                    user_info.scope = create_scope(
                        current_workspace=workspace,
                        workspaces={user_workspace: UserPermission.admin},
                        client_id=client_id,
                    )
                    logger.info(f"Created anonymous user {user_info.id} with workspace {user_workspace}")

                workspace_info = await self.store.load_or_create_workspace(
                    user_info, workspace
                )

                user_info.scope = update_user_scope(
                    user_info, workspace_info, client_id
                )
                if not user_info.check_permission(
                    workspace_info.id, UserPermission.read
                ):
                    logger.error(f"Permission denied for workspace: {workspace}")
                    raise PermissionError(
                        f"Permission denied for workspace: {workspace}"
                    )
                workspace = workspace_info.id

                assert workspace_info and workspace and client_id, (
                    "Failed to authenticate user to workspace: %s" % workspace_info.id
                )
                
                # Mark as authenticated before checking client
                authenticated = True
                
                # If the client is not reconnecting, check if it exists
                if not reconnection_token:
                    # We operate as the root user to remove and add clients
                    await self.check_client(client_id, workspace_info.id, user_info)
            except Exception as e:
                # Log the full exception for debugging
                logger.error(f"Exception during connection setup: {e}", exc_info=True)
                reason = f"Failed to establish connection: {str(e)}"
                
                # If we authenticated but failed later, try to clean up
                # This handles the case where check_client creates services but then fails
                if authenticated and user_info and workspace and client_id:
                    try:
                        # Only try to remove client if it might have been partially created
                        # check_client might have removed an old client and started creating a new one
                        if await self.store.client_exists(client_id, workspace):
                            await self.store.remove_client(client_id, workspace, user_info, unload=True)
                            logger.info(f"Cleaned up partially created client {workspace}/{client_id} after setup failure")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not cleanup client after setup failure: {cleanup_error}")
                
                await self.disconnect(
                    websocket,
                    reason=reason,
                    code=status.WS_1001_GOING_AWAY,
                )
                return

            try:
                await self.establish_websocket_communication(
                    websocket, workspace_info.id, client_id, user_info
                )
            except RuntimeError as exp:
                # this happens when the websocket is closed
                logger.error(f"RuntimeError in establish_websocket_communication: {str(exp)}")
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
                    status.WS_1001_GOING_AWAY,
                    e,
                )

    def get_websockets(self):
        """Get the active websockets."""
        return self._websockets

    async def force_disconnect(self, workspace, client_id, code, reason):
        """Force disconnect a client."""
        assert f"{workspace}/{client_id}" in self._websockets, "Client not connected"
        websocket = self._websockets[f"{workspace}/{client_id}"]
        try:
            await websocket.close(code=code, reason=reason)
        except GeneratorExit:
            pass  # Suppress GeneratorExit to avoid RuntimeError

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
            await self.store.remove_client(client_id, workspace, user_info, unload=True)

    async def authenticate_user(
        self, token: str, reconnection_token: str, client_id: str, workspace: str
    ):
        """Authenticate user and handle reconnection or token authentication."""
        user_info = None
        try:
            if reconnection_token:
                user_info = await self.store.parse_user_token(reconnection_token)
                scope = user_info.scope
                assert (
                    scope and scope.current_workspace
                ), "Invalid scope, current_workspace is required"
                if workspace:
                    assert (
                        workspace == scope.current_workspace
                    ), f"Invalid scope, workspace mismatch: {workspace} != {scope.current_workspace}"
                else:
                    workspace = scope.current_workspace
                assert scope.client_id, "Invalid scope, client_id is required"
                if not user_info.check_permission(
                    scope.current_workspace, UserPermission.read
                ):
                    logger.error(f"Permission denied for workspace: {workspace}")
                    raise PermissionError(
                        f"Permission denied for workspace: {workspace}"
                    )
                cid = scope.client_id
                if cid != client_id:
                    logger.error("Client id mismatch, disconnecting")
                    raise RuntimeError("Client id mismatch, disconnecting")
                logger.info(
                    f"Client reconnected: {workspace}/{cid} using reconnection token"
                )
            elif token:
                user_info = await self.store.parse_user_token(token)
                # Check if the token has a restricted client_id
                if user_info.scope and user_info.scope.client_id:
                    # Token is restricted to a specific client_id
                    if user_info.scope.client_id != client_id:
                        logger.error(
                            f"Client id mismatch: token restricted to '{user_info.scope.client_id}' but client '{client_id}' attempted to use it"
                        )
                        raise RuntimeError(
                            f"Client id mismatch: this token is restricted to client with client_id='{user_info.scope.client_id}'"
                        )
                # user token doesn't have client id, so we add that
                user_info.scope.client_id = client_id
                if user_info.scope.current_workspace and workspace:
                    assert (
                        workspace == user_info.scope.current_workspace
                    ), f"Current workspace encoded in the token ({user_info.scope.current_workspace}) does not match the specified workspace ({workspace})"
                if not workspace and user_info.scope.current_workspace:
                    workspace = user_info.scope.current_workspace
            else:
                raise RuntimeError("No authentication information provided")
            return user_info, workspace
        except HTTPException as e:
            logger.error(f"Authentication error: {e.detail}")
            raise RuntimeError(f"Authentication error: {e.detail}")
        except Exception as e:
            logger.error(f"Failed to authenticate user: {str(e)}")
            raise RuntimeError(f"Failed to authenticate user: {str(e)}")

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
        assert (
            user_info.scope.current_workspace == workspace
        ), f"Workspace mismatch: {workspace} != {user_info.scope.current_workspace}"
        
        # Determine if client has read-only permissions
        user_permission = user_info.get_permission(workspace)
        is_readonly = user_permission == UserPermission.read if user_permission else False
        
        conn = RedisRPCConnection(event_bus, workspace, client_id, user_info, self.store.get_manager_id(), readonly=is_readonly)
        self._websockets[f"{workspace}/{client_id}"] = websocket
        try:
            _gauge.inc()  # Increment total connections without workspace label
            event_bus.on_local(f"unload:{workspace}", force_disconnect)

            async def send_bytes(data):
                try:
                    # Handle both bytes and dict data
                    if isinstance(data, dict):
                        data = msgpack.packb(data)
                    elif isinstance(data, str):
                        data = data.encode('utf-8')
                    await websocket.send_bytes(data)
                except Exception as e:
                    logger.error("Failed to send message via websocket: %s", str(e))

            conn.on_message(send_bytes)
            reconnection_token = await generate_auth_token(
                user_info, expires_in=self.store.reconnection_token_life_time
            )
            conn_info = {
                "type": "connection_info",
                "hypha_version": __version__,
                "public_base_url": self.store.public_base_url,
                "local_base_url": self.store.local_base_url,
                "manager_id": self.store.get_manager_id(),
                "workspace": workspace,
                "client_id": client_id,
                "user": user_info.model_dump(),
                "reconnection_token": reconnection_token,
                "reconnection_token_life_time": self.store.reconnection_token_life_time,
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
                        elif data.get("type") == "refresh_token":
                            reconnection_token = await generate_auth_token(
                                user_info,
                                expires_in=self.store.reconnection_token_life_time,
                            )
                            conn_info = {"type": "reconnection_token"}
                            conn_info["reconnection_token"] = reconnection_token
                            await websocket.send_text(json.dumps(conn_info))
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
            # Decrement the gauge
            try:
                _gauge.dec()  # Decrement total connections without workspace label
            except Exception:
                pass  # Gauge may not have been incremented
            
            # Ensure proper cleanup even if conn was not created
            if conn:
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
        """Handle client disconnection with proper cleanup."""
        cleanup_succeeded = False
        try:
            await self.store.remove_client(client_id, workspace, user_info, unload=True)
            cleanup_succeeded = True
            if code in [status.WS_1000_NORMAL_CLOSURE, status.WS_1001_GOING_AWAY]:
                logger.info(f"Client disconnected normally: {workspace}/{client_id}")
            else:
                logger.info(
                    f"Client disconnected unexpectedly: {workspace}/{client_id}, code: {code}, error: {exp}"
                )
        except Exception as e:
            logger.error(
                f"Error handling disconnection, client: {workspace}/{client_id}, error: {str(e)}", 
                exc_info=True
            )
            # Re-raise if cleanup failed critically to ensure we're aware of persistent issues
            if not cleanup_succeeded:
                logger.critical(
                    f"CRITICAL: Failed to cleanup client {workspace}/{client_id} from Redis. "
                    f"This may leave orphaned services. Error: {str(e)}"
                )
        finally:
            await self.disconnect(
                websocket,
                f"Client {workspace}/{client_id} disconnected",
                code,
            )

    async def disconnect(self, websocket, reason, code=status.WS_1000_NORMAL_CLOSURE):
        """Disconnect the websocket connection."""
        logger.error("Disconnecting, reason: %s, code: %s", reason, code)
        if websocket.client_state.name == "CONNECTED":
            await websocket.send_text(json.dumps({"type": "error", "message": reason}))
        try:
            if websocket.client_state.name not in ["CLOSED", "CLOSING", "DISCONNECTED"]:
                try:
                    await websocket.close(code=code, reason=reason)
                except GeneratorExit:
                    pass  # Suppress GeneratorExit to avoid RuntimeError
        except Exception as e:
            logger.error(f"Error disconnecting websocket: {str(e)}")

    async def is_alive(self):
        """Check if the server is alive."""
        return True

    async def stop(self):
        """Stop the server."""
        self._stop = True
