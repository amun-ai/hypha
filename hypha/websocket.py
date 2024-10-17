import logging
import sys
import json

from fastapi import Query, WebSocket, status
from starlette.websockets import WebSocketDisconnect
from fastapi import HTTPException
from prometheus_client import Gauge

from hypha import __version__
from hypha.core import UserInfo, UserPermission
from hypha.core.store import RedisRPCConnection, RedisStore
from hypha.core.auth import (
    generate_reconnection_token,
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
        self._gauge = Gauge(
            "websocket_connections", "Number of websocket connections", ["workspace"]
        )

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
                    user_info, workspace = await self.authenticate_user(
                        token, reconnection_token, client_id, workspace
                    )
                else:
                    user_info = generate_anonymous_user()
                    user_info.scope = create_scope(
                        current_workspace=user_info.get_workspace(),
                        workspaces={user_info.get_workspace(): UserPermission.admin},
                        client_id=client_id,
                    )
                    workspace = workspace or user_info.get_workspace()
                    logger.info(f"Created anonymous user {user_info.id}")

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
                # We operate as the root user to remove and add clients
                await self.check_client(client_id, workspace_info.id, user_info)
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
                    websocket, workspace_info.id, client_id, user_info
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
                    status.WS_1001_GOING_AWAY,
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
                # user token doesn't have client id, so we add that
                user_info.scope.client_id = client_id
                if user_info.scope.current_workspace and workspace:
                    assert (
                        workspace == user_info.scope.current_workspace
                    ), f"Current workspace encoded in the token ({user_info.scope.current_workspace}) does not the specified workspace ({workspace})"
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
        ), f"Workspace mismatch: {workspace} != {user_info.current_workspace}"
        conn = RedisRPCConnection(event_bus, workspace, client_id, user_info, None)
        self._websockets[f"{workspace}/{client_id}"] = websocket
        try:
            self._gauge.labels(workspace=workspace).inc()
            event_bus.on_local(f"unload:{workspace}", force_disconnect)

            async def send_bytes(data):
                try:
                    await websocket.send_bytes(data)
                except Exception as e:
                    logger.error("Failed to send message via websocket: %s", str(e))

            conn.on_message(send_bytes)
            reconnection_token = generate_reconnection_token(
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
                            reconnection_token = generate_reconnection_token(
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
            self._gauge.labels(workspace=workspace).dec()
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
            await self.store.remove_client(client_id, workspace, user_info, unload=True)
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
