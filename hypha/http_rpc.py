"""HTTP Streaming RPC Server.

This module provides HTTP-based RPC transport as an alternative to WebSocket.
It uses:
- HTTP GET with streaming (NDJSON) for server-to-client messages (events, callbacks, responses)
- HTTP POST for client-to-server messages (method calls)

This is more resilient to network issues than WebSocket because:
1. Each POST request is independent (stateless)
2. GET stream can be easily reconnected
3. Works through more proxies and firewalls
"""

import asyncio
import io
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, Optional
import shortuuid

import msgpack
from fastapi import Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from hypha.core import RedisRPCConnection, UserInfo, UserPermission
from hypha.core.auth import (
    create_scope,
    generate_anonymous_user,
    generate_auth_token,
    update_user_scope,
)
from hypha.core.store import RedisStore

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("http-rpc")
logger.setLevel(LOGLEVEL)


class HTTPStreamingRPCServer:
    """HTTP Streaming RPC Server.

    Provides bidirectional RPC over HTTP using:
    - GET /{workspace}/rpc - Streaming endpoint for server->client messages
    - POST /{workspace}/rpc - Endpoint for client->server messages
    """

    def __init__(self, store: RedisStore, base_path: str = "/"):
        """Initialize the HTTP streaming RPC server."""
        self.store = store
        self.base_path = base_path
        # Track active streaming connections: {connection_key: queue}
        self._connections: Dict[str, asyncio.Queue] = {}
        # Track connection metadata
        self._connection_info: Dict[str, dict] = {}
        # Lock for connection management
        self._lock = asyncio.Lock()

    def _norm_url(self, url: str) -> str:
        """Normalize URL with base path."""
        return self.base_path.rstrip("/") + url

    async def _create_connection(
        self,
        workspace: str,
        client_id: str,
        user_info: UserInfo,
    ) -> tuple[RedisRPCConnection, asyncio.Queue]:
        """Create a new RPC connection with message queue."""
        event_bus = self.store.get_event_bus()

        # Determine if client has read-only permissions
        user_permission = user_info.get_permission(workspace)
        is_readonly = user_permission == UserPermission.read if user_permission else False

        conn = RedisRPCConnection(
            event_bus,
            workspace,
            client_id,
            user_info,
            self.store.get_manager_id(),
            readonly=is_readonly,
        )

        # Create message queue for this connection
        queue = asyncio.Queue()
        connection_key = f"{workspace}/{client_id}"

        async with self._lock:
            self._connections[connection_key] = queue
            self._connection_info[connection_key] = {
                "workspace": workspace,
                "client_id": client_id,
                "user_info": user_info,
                "created_at": time.time(),
            }

        # Set up message handler to push to queue
        async def send_to_queue(data):
            try:
                if isinstance(data, dict):
                    data = msgpack.packb(data)
                await queue.put(data)
            except Exception as e:
                logger.error(f"Failed to queue message: {e}")

        conn.on_message(send_to_queue)

        return conn, queue

    async def _remove_connection(self, workspace: str, client_id: str, conn: RedisRPCConnection):
        """Remove and cleanup a connection."""
        connection_key = f"{workspace}/{client_id}"
        async with self._lock:
            self._connections.pop(connection_key, None)
            self._connection_info.pop(connection_key, None)

        try:
            await conn.disconnect("disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

    async def _authenticate_with_reconnection_token(
        self, reconnection_token: str, client_id: str, workspace: str
    ) -> tuple[UserInfo, str]:
        """Authenticate using a reconnection token.

        Similar to WebSocket reconnection, this validates the reconnection token
        and ensures the client_id and workspace match the token's scope.

        Returns:
            tuple of (user_info, workspace)
        """
        user_info = await self.store.parse_user_token(reconnection_token)
        # Reject specialized tokens - they cannot be used for reconnection
        user_info.validate_for_general_access()

        scope = user_info.scope
        if not scope or not scope.current_workspace:
            raise ValueError("Invalid scope, current_workspace is required")

        # Validate workspace matches
        if workspace and workspace != scope.current_workspace:
            raise ValueError(
                f"Workspace mismatch: requested {workspace}, token has {scope.current_workspace}"
            )
        workspace = scope.current_workspace

        # Validate client_id matches
        if not scope.client_id:
            raise ValueError("Invalid scope, client_id is required")
        if scope.client_id != client_id:
            raise ValueError(
                f"Client ID mismatch: requested {client_id}, token has {scope.client_id}"
            )

        # Validate permissions
        if not user_info.check_permission(workspace, UserPermission.read):
            raise PermissionError(f"Permission denied for workspace: {workspace}")

        logger.info(f"HTTP client reconnected: {workspace}/{client_id} using reconnection token")
        return user_info, workspace

    def register_routes(self, app):
        """Register HTTP streaming RPC routes."""

        @app.get(self._norm_url("/{workspace}/rpc"))
        async def rpc_stream(
            workspace: str,
            request: Request,
            client_id: str = None,
            format: str = None,  # Optional format parameter: "json" or "msgpack"
            reconnection_token: str = None,  # Token for reconnecting to existing session
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Streaming endpoint for server-to-client messages.

            This endpoint supports two streaming formats:
            - NDJSON (default): Each line is a complete JSON object
            - msgpack: Length-prefixed msgpack frames for binary data support

            Format selection:
            - Use `format=msgpack` query parameter for binary data support
            - Use `Accept: application/x-ndjson` header for JSON (default)
            - Use `Accept: application/x-msgpack-stream` header for msgpack

            Reconnection:
            - Use `reconnection_token` query parameter to reconnect with same client_id
            - The token contains the original workspace and client_id
            - On reconnection, the server validates the token and re-uses the client identity

            The client should:
            1. Connect to this endpoint with a GET request
            2. Read messages continuously (lines for JSON, frames for msgpack)
            3. Parse each message
            4. Handle the message (method call, response, event)
            5. Reconnect if the connection is lost (using reconnection_token)
            """
            # Determine format from query param or Accept header
            accept_header = request.headers.get("accept", "")
            if format == "msgpack" or "application/x-msgpack" in accept_header:
                use_msgpack = True
            else:
                use_msgpack = False
            try:
                # Generate client_id if not provided
                if not client_id:
                    client_id = shortuuid.uuid()

                # Handle reconnection with token (takes priority over regular auth)
                is_reconnection = False
                if reconnection_token:
                    try:
                        user_info, workspace = await self._authenticate_with_reconnection_token(
                            reconnection_token, client_id, workspace
                        )
                        is_reconnection = True
                    except Exception as e:
                        logger.warning(f"Reconnection token validation failed: {e}")
                        # Fall back to regular authentication if reconnection fails
                        return JSONResponse(
                            status_code=401,
                            content={
                                "success": False,
                                "detail": f"Invalid reconnection token: {str(e)}",
                            },
                        )

                # Handle anonymous users - determine if they need their own workspace
                # Default anonymous users (without token) get redirected to their own workspace
                # Anonymous users WITH tokens (child users) use the token's workspace permissions
                if not is_reconnection:
                    user_workspace = user_info.get_workspace()
                    has_token_permissions = (
                        user_info.scope.workspaces
                        and any(
                            ws != "ws-anonymous" and ws.startswith("ws-user-")
                            for ws in user_info.scope.workspaces
                        )
                    )
                    if user_info.is_anonymous and not has_token_permissions:
                        # Redirect to user's own workspace
                        workspace = workspace if workspace and workspace != "public" else user_workspace
                        user_info.scope = create_scope(
                            current_workspace=workspace,
                            workspaces={user_workspace: UserPermission.admin},
                            client_id=client_id,
                        )

                # Load or create workspace (for both new connections and reconnections)
                workspace_info = await self.store.load_or_create_workspace(
                    user_info, workspace
                )
                workspace = workspace_info.id

                # Update user scope with workspace info (critical for permission checks)
                # For reconnections, this ensures the scope is properly updated
                user_info.scope = update_user_scope(
                    user_info, workspace_info, client_id
                )

                # Check permissions
                if not user_info.check_permission(workspace, UserPermission.read):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": f"Permission denied for workspace {workspace}",
                        },
                    )

                # Create connection
                conn, queue = await self._create_connection(
                    workspace, client_id, user_info
                )

                # Generate new reconnection token for this session
                # This refreshes the token on each connection/reconnection
                new_reconnection_token = await generate_auth_token(
                    user_info, expires_in=self.store.reconnection_token_life_time
                )

                def encode_message(message: dict, use_msgpack_format: bool) -> bytes:
                    """Encode a message in the appropriate format."""
                    if use_msgpack_format:
                        # msgpack with 4-byte length prefix for framing
                        data = msgpack.packb(message)
                        length = len(data)
                        return length.to_bytes(4, 'big') + data
                    else:
                        # NDJSON: JSON + newline
                        return (json.dumps(message) + "\n").encode('utf-8')

                async def stream_messages():
                    """Generator that yields messages in the requested format."""
                    nonlocal new_reconnection_token
                    try:
                        # Send connection info as first message
                        conn_info = {
                            "type": "connection_info",
                            "workspace": workspace,
                            "client_id": client_id,
                            "manager_id": self.store.get_manager_id(),
                            "reconnection_token": new_reconnection_token,
                            "reconnection_token_life_time": self.store.reconnection_token_life_time,
                            "public_base_url": self.store.public_base_url,
                        }
                        yield encode_message(conn_info, use_msgpack)

                        # Keep-alive interval (send ping every 30 seconds)
                        last_ping = time.time()
                        ping_interval = 30

                        while True:
                            try:
                                # Check if client disconnected
                                if await request.is_disconnected():
                                    logger.info(f"Client {workspace}/{client_id} disconnected")
                                    break

                                # Try to get message with timeout
                                try:
                                    data = await asyncio.wait_for(
                                        queue.get(), timeout=1.0
                                    )

                                    # Process the data based on its type
                                    if isinstance(data, bytes):
                                        if use_msgpack:
                                            # For msgpack output, forward the raw bytes with length prefix
                                            length = len(data)
                                            yield length.to_bytes(4, 'big') + data
                                        else:
                                            # For JSON output, decode msgpack first
                                            unpacker = msgpack.Unpacker(io.BytesIO(data))
                                            message = unpacker.unpack()
                                            yield encode_message(message, use_msgpack)
                                    elif isinstance(data, dict):
                                        yield encode_message(data, use_msgpack)
                                    else:
                                        yield encode_message({"data": data}, use_msgpack)

                                except asyncio.TimeoutError:
                                    # No message, check if we need to send ping
                                    if time.time() - last_ping > ping_interval:
                                        yield encode_message({"type": "ping"}, use_msgpack)
                                        last_ping = time.time()
                                    continue

                            except asyncio.CancelledError:
                                logger.info(f"Stream cancelled for {workspace}/{client_id}")
                                break
                            except Exception as e:
                                logger.error(f"Error in stream: {e}")
                                yield encode_message({"type": "error", "message": str(e)}, use_msgpack)
                                break

                    finally:
                        # Cleanup connection
                        await self._remove_connection(workspace, client_id, conn)

                media_type = "application/x-msgpack-stream" if use_msgpack else "application/x-ndjson"
                return StreamingResponse(
                    stream_messages(),
                    media_type=media_type,
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    },
                )

            except Exception as e:
                logger.error(f"Error in rpc_stream: {e}\n{traceback.format_exc()}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": str(e)},
                )

        @app.post(self._norm_url("/{workspace}/rpc"))
        async def rpc_post(
            workspace: str,
            request: Request,
            client_id: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Endpoint for client-to-server RPC messages.

            This endpoint accepts msgpack-encoded RPC messages.
            The message is routed through the event bus to the target service.

            For stateless calls (no streaming connection), the response is returned directly.
            For clients with an active stream, the response goes through the stream.
            """
            try:
                if not client_id:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "detail": "client_id is required",
                        },
                    )

                connection_key = f"{workspace}/{client_id}"

                # Check if this client has an active streaming connection
                async with self._lock:
                    has_stream = connection_key in self._connections

                if not has_stream:
                    # For stateless calls, we need to create a temporary connection
                    # Handle anonymous users
                    if user_info.is_anonymous:
                        user_workspace = user_info.get_workspace()
                        workspace = workspace or user_workspace
                        user_info.scope = create_scope(
                            current_workspace=workspace,
                            workspaces={user_workspace: UserPermission.admin},
                            client_id=client_id,
                        )

                    # Check permissions
                    if not user_info.check_permission(workspace, UserPermission.read_write):
                        return JSONResponse(
                            status_code=403,
                            content={
                                "success": False,
                                "detail": f"Permission denied for workspace {workspace}",
                            },
                        )

                # Read the msgpack body
                body = await request.body()
                if not body:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "detail": "Empty request body"},
                    )

                # For clients with streams, route through event bus
                if has_stream:
                    # Get the connection and emit message
                    event_bus = self.store.get_event_bus()

                    # Get the stored user_info from connection (has updated scope)
                    async with self._lock:
                        conn_info = self._connection_info.get(connection_key, {})
                        stored_user_info = conn_info.get("user_info", user_info)
                        queue = self._connections.get(connection_key)

                    # Parse the message to check for control messages
                    unpacker = msgpack.Unpacker(io.BytesIO(body))
                    message = unpacker.unpack()
                    pos = unpacker.tell()

                    # Handle control messages (similar to WebSocket)
                    msg_type = message.get("type")
                    if msg_type == "refresh_token":
                        # Generate new reconnection token and send via stream
                        new_token = await generate_auth_token(
                            stored_user_info,
                            expires_in=self.store.reconnection_token_life_time,
                        )
                        token_msg = {
                            "type": "reconnection_token",
                            "reconnection_token": new_token,
                        }
                        if queue:
                            await queue.put(token_msg)
                        return JSONResponse(
                            status_code=200,
                            content={"success": True, "detail": "Token refresh requested"},
                        )
                    elif msg_type == "ping":
                        # Respond with pong via stream
                        if queue:
                            await queue.put({"type": "pong"})
                        return JSONResponse(
                            status_code=200,
                            content={"success": True, "detail": "Pong sent"},
                        )

                    # Regular RPC message - add context
                    target_id = message.get("to")
                    source_id = f"{workspace}/{client_id}"

                    # Handle broadcast
                    if target_id == "*":
                        target_id = f"{workspace}/*"
                    elif "/" not in target_id:
                        target_id = f"{workspace}/{target_id}"

                    # Add context using stored user_info with proper scope
                    message.update({
                        "ws": workspace,
                        "to": target_id,
                        "from": source_id,
                        "user": stored_user_info.model_dump(),
                    })

                    # Re-pack and emit
                    packed_message = msgpack.packb(message) + body[pos:]
                    await event_bus.emit(f"{target_id}:msg", packed_message)

                    return JSONResponse(
                        status_code=200,
                        content={"success": True, "detail": "Message sent"},
                    )
                else:
                    # Stateless call - handle synchronously
                    # This is a simplified path for clients that don't need callbacks
                    return await self._handle_stateless_call(
                        workspace, client_id, user_info, body
                    )

            except Exception as e:
                logger.error(f"Error in rpc_post: {e}\n{traceback.format_exc()}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": str(e)},
                )

    async def _handle_stateless_call(
        self,
        workspace: str,
        client_id: str,
        user_info: UserInfo,
        body: bytes,
    ):
        """Handle a stateless RPC call without streaming.

        This creates a temporary connection, makes the call, waits for response,
        and returns it directly in the HTTP response.
        """
        try:
            # Parse the message
            unpacker = msgpack.Unpacker(io.BytesIO(body))
            message = unpacker.unpack()

            # For stateless calls, we expect a specific format
            # {"type": "method", "to": "target", "method": "name", "args": [...]}
            if message.get("type") != "method":
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "detail": "Stateless calls require type='method'",
                    },
                )

            # Use workspace interface for the call
            async with self.store.get_workspace_interface(user_info, workspace) as api:
                target = message.get("to", "").split(":")
                if len(target) != 2:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "detail": "Invalid target format, expected 'client_id:service_id'",
                        },
                    )

                service_id = target[0] + ":" + target[1]
                if "/" not in service_id:
                    service_id = f"{workspace}/{service_id}"

                service = await api.get_service(service_id)
                method_name = message.get("method")

                if not hasattr(service, method_name):
                    return JSONResponse(
                        status_code=404,
                        content={
                            "success": False,
                            "detail": f"Method '{method_name}' not found",
                        },
                    )

                method = getattr(service, method_name)
                args = message.get("args", [])
                kwargs = message.get("kwargs", {})

                result = await method(*args, **kwargs)

                # Encode result
                return JSONResponse(
                    status_code=200,
                    content={"success": True, "result": result},
                )

        except Exception as e:
            logger.error(f"Error in stateless call: {e}\n{traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "detail": str(e)},
            )


def setup_http_rpc(store: RedisStore, app, base_path: str = "/"):
    """Set up HTTP streaming RPC server.

    Args:
        store: The RedisStore instance
        app: The FastAPI application
        base_path: Base path for the RPC endpoints

    Returns:
        HTTPStreamingRPCServer instance
    """
    server = HTTPStreamingRPCServer(store, base_path)
    server.register_routes(app)
    logger.info("HTTP Streaming RPC server initialized")
    return server
