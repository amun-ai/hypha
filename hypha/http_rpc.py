"""HTTP Streaming RPC Server.

This module provides HTTP-based RPC transport as an alternative to WebSocket.
It uses:
- HTTP GET with streaming (msgpack) for server-to-client messages (events, callbacks, responses)
- HTTP POST for client-to-server messages (method calls)

This is more resilient to network issues than WebSocket because:
1. Each POST request is independent (stateless)
2. GET stream can be easily reconnected
3. Works through more proxies and firewalls
"""

import asyncio
import io
import logging
import os
import sys
import time
import traceback
from typing import Dict, Optional
import shortuuid

import msgpack
from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

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

# SECURITY: Limits to prevent DoS attacks
MAX_REQUEST_BODY_SIZE = 100 * 1024 * 1024  # 100 MB max request size
MAX_CONNECTIONS_PER_WORKSPACE = 1000  # Max concurrent connections per workspace
MESSAGE_QUEUE_SIZE = 100  # Max messages buffered per connection


class HTTPStreamingRPCServer:
    """HTTP Streaming RPC Server.

    Provides bidirectional RPC over HTTP using:
    - GET /rpc - Streaming endpoint for server->client messages
    - POST /rpc - Endpoint for client->server messages

    Workspace is resolved from:
    1. The `workspace` query parameter (if provided)
    2. The token's current_workspace scope
    3. The user's own workspace (ws-user-{user_id})
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

        # Create message queue for this connection with bounded size
        # SECURITY: Prevent unbounded memory growth from malicious clients
        # maxsize=100 limits buffered messages to prevent DoS attacks
        queue = asyncio.Queue(maxsize=100)
        connection_key = f"{workspace}/{client_id}"

        async with self._lock:
            # Check if there's an existing connection with the same key
            # If so, close it first to prevent queue mismatch between old stream and new messages
            if connection_key in self._connections:
                old_queue = self._connections[connection_key]
                # Signal old stream to stop by putting a sentinel value
                try:
                    old_queue.put_nowait(None)  # None signals stream to close
                except asyncio.QueueFull:
                    pass  # Old queue is full, it will be replaced anyway
                logger.info(f"[HTTP RPC] Closing old connection for {connection_key}")

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
                # SECURITY: Use put_nowait to prevent blocking event bus
                # If queue is full (slow/malicious client), drop messages
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    logger.warning(f"Message queue full for {connection_key}, dropping message")
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

    def _resolve_workspace(self, workspace: Optional[str], user_info: UserInfo, client_id: str):
        """Resolve workspace when not explicitly provided or when "public" is used.

        Resolution order:
        1. If workspace is explicitly provided and not "public" for non-admin users -> use it
        2. If token has current_workspace -> use it
        3. Fall back to user's own workspace (user_info.get_workspace())

        Returns:
            tuple of (workspace, user_info) - user_info may have updated scope for anonymous users
        """
        user_workspace = user_info.get_workspace()
        has_token_permissions = (
            user_info.scope.workspaces
            and any(
                ws != "ws-anonymous" and ws.startswith("ws-user-")
                for ws in user_info.scope.workspaces
            )
        )
        if user_info.is_anonymous and not has_token_permissions:
            # For anonymous users: use explicit workspace if provided and not "public",
            # otherwise fall back to their own workspace
            workspace = workspace if workspace and workspace != "public" else user_workspace
            user_info.scope = create_scope(
                current_workspace=workspace,
                workspaces={user_workspace: UserPermission.admin},
                client_id=client_id,
            )
        elif not workspace or workspace == "public":
            # Workspace-less route (workspace=None) or "public" placeholder:
            # Use the token's workspace or the user's own workspace.
            if user_info.scope.current_workspace and user_info.scope.current_workspace != "public":
                workspace = user_info.scope.current_workspace
            else:
                workspace = user_workspace
        # else: workspace is explicitly provided and not "public", use as-is

        return workspace, user_info

    def register_routes(self, app):
        """Register HTTP streaming RPC routes."""

        async def _handle_rpc_stream(
            workspace: Optional[str],
            request: Request,
            client_id: Optional[str],
            reconnection_token: Optional[str],
            user_info: UserInfo,
        ):
            """Shared handler for GET /rpc.

            Uses msgpack streaming format with length-prefixed frames.
            Each frame consists of a 4-byte big-endian length prefix followed by
            msgpack-encoded data.

            When workspace is None (from the workspace-less /rpc route), the workspace
            is resolved from the user's token or defaults to the user's own workspace.
            """
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

                # Handle workspace assignment for users
                if not is_reconnection:
                    workspace, user_info = self._resolve_workspace(workspace, user_info, client_id)

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

                def encode_message(message: dict) -> bytes:
                    """Encode a message as length-prefixed msgpack."""
                    data = msgpack.packb(message)
                    length = len(data)
                    return length.to_bytes(4, 'big') + data

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
                        yield encode_message(conn_info)

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

                                    # Check for sentinel value (None) indicating connection replacement
                                    if data is None:
                                        logger.info(f"[STREAM] Received close signal for {workspace}/{client_id}")
                                        break

                                    # Process the data based on its type
                                    if isinstance(data, bytes):
                                        # Forward raw msgpack bytes with length prefix
                                        length = len(data)
                                        yield length.to_bytes(4, 'big') + data
                                    elif isinstance(data, dict):
                                        yield encode_message(data)
                                    else:
                                        yield encode_message({"data": data})

                                except asyncio.TimeoutError:
                                    # No message, check if we need to send ping
                                    if time.time() - last_ping > ping_interval:
                                        yield encode_message({"type": "ping"})
                                        last_ping = time.time()
                                    continue

                            except asyncio.CancelledError:
                                logger.info(f"Stream cancelled for {workspace}/{client_id}")
                                break
                            except Exception as e:
                                logger.error(f"Error in stream: {e}")
                                yield encode_message({"type": "error", "message": str(e)})
                                break

                    finally:
                        # Cleanup connection
                        await self._remove_connection(workspace, client_id, conn)

                return StreamingResponse(
                    stream_messages(),
                    media_type="application/x-msgpack-stream",
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

        async def _handle_rpc_post(
            workspace: Optional[str],
            request: Request,
            client_id: Optional[str],
            user_info: UserInfo,
        ):
            """Shared handler for POST /rpc.

            Accepts msgpack-encoded RPC messages.
            The message is routed through the event bus to the target service.

            When workspace is None (from the workspace-less /rpc route), the workspace
            is resolved from the user's token or defaults to the user's own workspace.
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

                # Resolve workspace early if not provided
                if not workspace:
                    workspace, user_info = self._resolve_workspace(workspace, user_info, client_id)

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
                        return Response(status_code=200)
                    elif msg_type == "ping":
                        # Respond with pong via stream
                        if queue:
                            await queue.put({"type": "pong"})
                        return Response(status_code=200)

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

                    return Response(status_code=200)
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

        # --- Route registrations ---
        # Single /rpc endpoint; workspace is resolved from query param or token.

        @app.get(self._norm_url("/rpc"))
        async def rpc_stream(
            request: Request,
            workspace: str = None,
            client_id: str = None,
            reconnection_token: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Streaming endpoint for server-to-client messages.

            Workspace is resolved from the `workspace` query parameter,
            the token's scope, or defaults to the user's own workspace.
            """
            return await _handle_rpc_stream(
                workspace=workspace,
                request=request,
                client_id=client_id,
                reconnection_token=reconnection_token,
                user_info=user_info,
            )

        @app.post(self._norm_url("/rpc"))
        async def rpc_post(
            request: Request,
            workspace: str = None,
            client_id: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Endpoint for client-to-server RPC messages.

            Workspace is resolved from the `workspace` query parameter,
            the token's scope, or defaults to the user's own workspace.
            """
            return await _handle_rpc_post(
                workspace=workspace,
                request=request,
                client_id=client_id,
                user_info=user_info,
            )

        # Backward-compatible /{workspace}/rpc routes for older hypha-rpc clients.
        @app.get(self._norm_url("/{workspace}/rpc"))
        async def rpc_stream_compat(
            request: Request,
            workspace: str,
            client_id: str = None,
            reconnection_token: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Backward-compatible streaming endpoint with workspace in path."""
            return await _handle_rpc_stream(
                workspace=workspace,
                request=request,
                client_id=client_id,
                reconnection_token=reconnection_token,
                user_info=user_info,
            )

        @app.post(self._norm_url("/{workspace}/rpc"))
        async def rpc_post_compat(
            request: Request,
            workspace: str,
            client_id: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Backward-compatible POST endpoint with workspace in path."""
            return await _handle_rpc_post(
                workspace=workspace,
                request=request,
                client_id=client_id,
                user_info=user_info,
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
