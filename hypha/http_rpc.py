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
        # Last UPSTREAM (client->server POST) activity per connection. Used by the
        # liveness sweep below. Unlike WebSockets — which reap a dead peer via
        # wait_for(receive(), idle_timeout) — an HTTP-streaming half-open peer
        # (docker veth yanked, NO FIN) is never detected by request.is_disconnected(),
        # so its service registrations ghost indefinitely (only a server restart
        # clears them). #0011.
        self._last_activity: Dict[str, float] = {}
        # Lock for connection management
        self._lock = asyncio.Lock()
        # Liveness sweep config (#0011): a /rpc stream with no upstream POST for
        # STREAM_IDLE_THRESHOLD becomes a ping-verify CANDIDATE (idle-but-live is
        # NORMAL for streaming — an idle client sends nothing upstream — so idle
        # alone NEVER reaps; it only triggers an ACTIVE RPC-layer ping). Only
        # non-responders are reaped, so a healthy idle machine is never
        # false-reaped. WebSockets are untouched (they have their own reaper).
        self._stream_idle_threshold = float(
            os.environ.get("HYPHA_STREAM_IDLE_THRESHOLD", "60")
        )
        self._stream_sweep_interval = float(
            os.environ.get("HYPHA_STREAM_SWEEP_INTERVAL", "30")
        )
        self._stream_ping_timeout = float(
            os.environ.get("HYPHA_STREAM_PING_TIMEOUT", "5")
        )
        # Reap only after this many CONSECUTIVE ping misses — never on a single
        # miss. The streaming ping is a full HTTP POST round-trip (slower than a
        # WS frame), so a live-but-loaded machine can miss one; requiring N
        # consecutive misses (reset on any answer) prevents false-reaping it.
        self._stream_max_ping_misses = int(
            os.environ.get("HYPHA_STREAM_MAX_PING_MISSES", "3")
        )
        # connection_key -> consecutive ping-miss count
        self._ping_misses: Dict[str, int] = {}
        # Wedge self-heal (#0012): a client that stops reading its downstream
        # stream (StreamingResponse yield blocks on TCP backpressure) makes the
        # stream loop stick — it never drains the queue, which fills and drops
        # messages FOREVER, leaving the machine unresponsive with no auto-recovery
        # (only a fresh reconnect clears it). We count CONSECUTIVE downstream-queue
        # drops per connection (reset on any successful enqueue, so a slow-but-
        # draining client never trips it) and, once a connection is clearly not
        # draining, force-recycle it (cancel its stuck stream task) so the client
        # reconnects with a fresh empty queue.
        self._drop_counts: Dict[str, int] = {}
        self._stream_tasks: Dict[str, asyncio.Task] = {}
        self._stream_max_drops = int(
            os.environ.get("HYPHA_STREAM_MAX_QUEUE_DROPS", "200")
        )
        self._stop_sweep = False
        # Start the background liveness sweep (mirrors WebsocketServer's idle loop).
        try:
            asyncio.get_running_loop()
            self._sweep_task = asyncio.create_task(self._liveness_sweep_loop())
        except RuntimeError:
            # No running loop yet (e.g., constructed during startup/tests); the
            # loop will be created lazily on first stream connect.
            self._sweep_task = None

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
            # Seed liveness activity so a just-connected client is not swept
            # before it has had a chance to be active (#0011).
            self._last_activity[connection_key] = time.time()
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
                    # Successful enqueue ⇒ the stream IS draining; reset the
                    # consecutive-drop counter (a slow-but-draining client must
                    # not accumulate toward a force-recycle). #0012
                    if connection_key in self._drop_counts:
                        self._drop_counts.pop(connection_key, None)
                except asyncio.QueueFull:
                    # Count CONSECUTIVE drops. A persistently-full queue means the
                    # stream loop is stuck (client not reading) — the liveness
                    # sweep force-recycles it past _stream_max_drops. #0012
                    self._drop_counts[connection_key] = (
                        self._drop_counts.get(connection_key, 0) + 1
                    )
                    logger.warning(f"Message queue full for {connection_key}, dropping message")
            except Exception as e:
                logger.error(f"Failed to queue message: {e}")

        conn.on_message(send_to_queue)

        # F6 (multi-replica): announce this client to sibling pods so they clear
        # any stale recently-disconnected cache entry and don't false-reject
        # messages to it after a re-pin to this pod.
        try:
            await event_bus.notify_client_connected(workspace, client_id)
        except Exception as e:
            logger.debug(f"notify_client_connected failed for {connection_key}: {e}")

        # Ensure the liveness sweep is running (handles construction before a loop).
        self._ensure_sweep_started()

        return conn, queue

    async def _remove_connection(self, workspace: str, client_id: str, conn: RedisRPCConnection):
        """Remove and cleanup a connection."""
        connection_key = f"{workspace}/{client_id}"
        async with self._lock:
            self._connections.pop(connection_key, None)
            self._connection_info.pop(connection_key, None)
            self._last_activity.pop(connection_key, None)
            self._drop_counts.pop(connection_key, None)
            self._stream_tasks.pop(connection_key, None)

        try:
            await conn.disconnect("disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

    def _ensure_sweep_started(self):
        """Start the liveness sweep if it wasn't started at construction time
        (e.g. the server was constructed before an event loop existed)."""
        if self._sweep_task is None or self._sweep_task.done():
            try:
                self._sweep_task = asyncio.create_task(self._liveness_sweep_loop())
            except RuntimeError:
                pass  # still no running loop; try again on the next connect

    async def _liveness_sweep_loop(self):
        """Periodically ping-verify idle /rpc streams and reap non-responders.

        This closes the HTTP-streaming analog of the WebSocket idle reaper: a
        half-open streaming peer (no FIN, e.g. a container hard-removed) is never
        surfaced by ``request.is_disconnected()``, so without this its service
        registrations ghost forever. #0011.
        """
        while not self._stop_sweep:
            try:
                await asyncio.sleep(self._stream_sweep_interval)
                await self._do_liveness_sweep()
            except asyncio.CancelledError:
                break
            except Exception as e:  # never let the sweep loop die
                logger.warning("HTTP-streaming liveness sweep error: %s", e)

    async def _do_liveness_sweep(self):
        """One sweep pass: ping-verify idle streams, reap non-responders.

        A stream with no upstream POST for ``_stream_idle_threshold`` is only a
        CANDIDATE — idle-but-live is the NORMAL state for a streaming client (it
        sends nothing upstream when idle), so idle alone NEVER reaps. We actively
        RPC-ping each candidate and reap ONLY the ones that do not answer, so a
        healthy idle machine is never false-reaped.

        Also (#0012) force-recycles WEDGED streams: a connection whose downstream
        queue is persistently full (many CONSECUTIVE drops, i.e. not draining) is
        aborted so the client reconnects with a fresh empty queue.
        """
        # Wedge self-heal (#0012): recycle connections whose downstream queue is
        # persistently full (not draining). Distinct from the idle/dead reap
        # below — a wedged client is ALIVE (upstream-active), so we recycle its
        # stream rather than reap its services.
        async with self._lock:
            wedged = [
                key
                for key, drops in self._drop_counts.items()
                if drops >= self._stream_max_drops and key in self._connections
            ]
        for connection_key in wedged:
            await self._recycle_wedged_stream(connection_key)

        now = time.time()
        async with self._lock:
            candidates = [
                key
                for key, last in self._last_activity.items()
                if now - last > self._stream_idle_threshold
                and key in self._connections
            ]
        for connection_key in candidates:
            if "/" not in connection_key:
                continue
            workspace, client_id = connection_key.split("/", 1)
            alive = await self._ping_stream_client(workspace, client_id)
            if alive:
                # Live idle client — reset the miss counter and refresh so it is
                # re-pinged at most once per idle-threshold (bounds ping load to
                # genuinely-idle streams).
                async with self._lock:
                    self._ping_misses.pop(connection_key, None)
                    if connection_key in self._last_activity:
                        self._last_activity[connection_key] = time.time()
            else:
                # Do NOT reap on a single miss — a live-but-loaded streaming
                # client can miss one slow POST round-trip. Reap only after
                # _stream_max_ping_misses CONSECUTIVE misses.
                async with self._lock:
                    if connection_key not in self._connections:
                        continue  # gone during the ping
                    misses = self._ping_misses.get(connection_key, 0) + 1
                    self._ping_misses[connection_key] = misses
                if misses >= self._stream_max_ping_misses:
                    await self._reap_dead_stream(
                        workspace, client_id, connection_key
                    )

    async def _ping_stream_client(self, workspace: str, client_id: str) -> bool:
        """Active RPC-layer ping of a streaming client. True iff it answers 'pong'.

        Uses the workspace manager's ``ping_client`` (which pings the client's
        built-in service). A live client — even one that is idle and never sends
        upstream — answers the RPC ping; a half-open/dead client times out.
        """
        try:
            wm = self.store._workspace_manager
            if wm is None:
                return True  # cannot verify — do NOT reap (fail safe)
            root = self.store.get_root_user()
            context = {
                "ws": workspace,
                "user": root.model_dump(),
                "from": f"{workspace}/check-client-exists",
            }
            result = await asyncio.wait_for(
                wm.ping_client(
                    f"{workspace}/{client_id}",
                    timeout=self._stream_ping_timeout,
                    context=context,
                ),
                timeout=self._stream_ping_timeout + 2,
            )
            return result == "pong"
        except Exception:
            return False

    async def _reap_dead_stream(
        self, workspace: str, client_id: str, connection_key: str
    ):
        """Reap a confirmed-dead streaming client: remove its service
        registrations and close its (ghost) stream. #0011."""
        logger.warning(
            "Reaping dead HTTP-streaming client %s: idle and failed %d "
            "consecutive RPC pings — removing its service registrations "
            "(half-open peer, no FIN).",
            connection_key,
            self._stream_max_ping_misses,
        )
        # Remove the client's service registrations (delete_client removes the
        # services:* keys — the actual ghost).
        try:
            await self.store.remove_client(
                client_id, workspace, self.store.get_root_user(), unload=False
            )
        except Exception as e:
            logger.warning("Error removing dead client %s: %s", connection_key, e)
        # Drop local state immediately (don't rely on the ghost stream loop
        # consuming the sentinel — a half-open stream may be slow to notice) and
        # push the None sentinel to end stream_messages() if it is still running.
        async with self._lock:
            queue = self._connections.pop(connection_key, None)
            self._connection_info.pop(connection_key, None)
            self._last_activity.pop(connection_key, None)
            self._ping_misses.pop(connection_key, None)
        if queue is not None:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def _recycle_wedged_stream(self, connection_key: str):
        """Force-recycle a wedged streaming connection (#0012).

        The downstream queue is persistently full: the client stopped reading its
        stream, so the loop is stuck on a backpressured ``yield`` and never drains
        the queue (a None sentinel can't help — the loop isn't at ``queue.get``).
        Cancelling the stream's response task aborts the stuck yield; the GET
        stream ends, the client reconnects with a FRESH empty queue, and the
        machine is responsive again. The client is ALIVE (upstream-active), so we
        do NOT reap its services (it re-registers on reconnect) — we just recycle
        the connection. The cancelled generator's ``finally`` runs
        ``_remove_connection`` to clear local state.
        """
        async with self._lock:
            drops = self._drop_counts.pop(connection_key, 0)
            task = self._stream_tasks.get(connection_key)
        logger.warning(
            "Recycling wedged HTTP-streaming connection %s: downstream queue full "
            "and not draining (%d consecutive drops) — aborting the stuck stream "
            "so the client reconnects with a fresh queue.",
            connection_key,
            drops,
        )
        if task is not None and not task.done():
            task.cancel()

    async def close_all_streams(self) -> int:
        """Signal every active HTTP streaming connection to close cleanly.

        Used during graceful shutdown (SIGTERM/rollout). Pushes the ``None``
        sentinel that ``stream_messages()`` interprets as a close signal, which
        ends the StreamingResponse so the client observes a clean stream EOF and
        reconnects immediately — instead of waiting for its read/heartbeat
        timeout, which would otherwise produce a reconnection storm when a pod
        is redeployed.

        Returns the number of connections that were signaled.
        """
        # Stop the liveness sweep during shutdown (#0011).
        self._stop_sweep = True
        if self._sweep_task is not None and not self._sweep_task.done():
            self._sweep_task.cancel()
        async with self._lock:
            queues = list(self._connections.values())
        if not queues:
            return 0
        count = 0
        for queue in queues:
            try:
                queue.put_nowait(None)  # None signals stream to close
                count += 1
            except asyncio.QueueFull:
                # Queue backed up by a slow client; the stream will still end via
                # request.is_disconnected() once the socket drops. We cannot
                # block here during shutdown.
                logger.warning(
                    "Stream queue full during shutdown; relying on disconnect detection"
                )
        logger.info("Signaled %d HTTP streaming connection(s) to close", count)
        return count

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
                    # Register this response's task so the liveness sweep can
                    # force-recycle a wedged stream (cancel this task to abort a
                    # yield stuck on client backpressure). #0012
                    self._stream_tasks[f"{workspace}/{client_id}"] = (
                        asyncio.current_task()
                    )
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
                    if has_stream:
                        # Any upstream POST is a liveness signal — refresh the
                        # activity timestamp so the liveness sweep won't ping/reap
                        # an actively-communicating client (#0011).
                        self._last_activity[connection_key] = time.time()

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
