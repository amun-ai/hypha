import io
import logging
import os
import sys
import json
import asyncio
import time
import random
import msgpack

from fastapi import Query, WebSocket, status
from starlette.websockets import WebSocketDisconnect
from fastapi import HTTPException
from prometheus_client import Counter, Gauge

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
_connections_established = Counter(
    "websocket_connections_established", "Total number of websocket connections ever established"
)
_rate_limited_disconnects = Counter(
    "websocket_rate_limited_disconnects", "Total number of clients disconnected due to rate limiting"
)


class _CancelledErrorASGIFilter(logging.Filter):
    """Drop uvicorn's "Exception in ASGI application" ERROR when the terminal
    exception is asyncio.CancelledError.

    A websocket ASGI task is cancelled on every normal client disconnect (uvicorn
    cancels ``run_asgi``) and on our own supersede-cancel of a stale generation.
    uvicorn logs that cancellation as a full ERROR traceback (~238/24h steady
    state, spiking to ~320/min during a rollout's reconnect wave). Cancellation of
    a websocket handler is never a server error, so the traceback is pure noise
    that masks genuine ASGI errors. This filter suppresses ONLY CancelledError
    terminals — any other ASGI exception passes through unchanged, so real errors
    still surface. It touches nothing in the control/cancellation path; it only
    quiets the log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, asyncio.CancelledError):
            return False
        return True


def install_asgi_cancellederror_filter() -> None:
    """Idempotently install the CancelledError filter on the ``uvicorn.error``
    logger (the logger uvicorn uses for "Exception in ASGI application")."""
    uvicorn_logger = logging.getLogger("uvicorn.error")
    if not any(
        isinstance(f, _CancelledErrorASGIFilter) for f in uvicorn_logger.filters
    ):
        uvicorn_logger.addFilter(_CancelledErrorASGIFilter())


def _is_expected_close_error(exc: BaseException) -> bool:
    """True if `exc` is a benign ASGI lifecycle race, not a real server error.

    Starlette raises RuntimeError when the receive/send loop touches the transport
    after the peer's socket already closed:
      - 'Cannot call "receive" once a disconnect message has been received.'
      - 'Cannot call "send" once a close message has been sent.'
    These are expected when a client drops mid-message and should be logged at
    DEBUG, not ERROR — otherwise they bury genuine errors (~416/24h in prod).
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return (
        'once a disconnect message has been received' in msg
        or 'once a close message has been sent' in msg
    )


class AuthenticationError(Exception):
    """A client's credentials were invalid/expired — an EXPECTED client outcome.

    An expired or malformed CLIENT token is not a server fault: the client should
    refetch and retry. Raised by `authenticate_user` (from the `HTTPException` that
    token validation produces) so the connection-setup block can catch it BEFORE
    the generic `except Exception`, log a single WARNING (no traceback), and close
    the socket with 1008 — instead of letting it hit the `exc_info=True` catch-all,
    which logged a full multi-frame ERROR traceback for every routine token expiry
    (~30/24h of pure noise in prod). Genuine setup errors still raise through to the
    ERROR+traceback path. See issue #0008.
    """


class TokenBucketRateLimiter:
    """Simple token-bucket rate limiter for per-client message throttling."""

    __slots__ = ("rate", "burst", "_tokens", "_last")

    def __init__(self, rate: float, burst: int):
        self.rate = rate  # tokens refilled per second
        self.burst = burst  # max tokens (bucket capacity)
        self._tokens = float(burst)
        self._last = time.monotonic()

    def consume(self, tokens: int = 1) -> tuple:
        """Try to consume tokens. Returns (allowed, tokens_remaining)."""
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True, self._tokens
        return False, self._tokens


class WebsocketServer:
    def __init__(self, store: RedisStore, path="/ws"):
        """Initialize websocket server with the store and set up the endpoint."""
        self.store = store
        app = store._app
        self.store.set_websocket_server(self)
        # Quiet uvicorn's benign "Exception in ASGI application" CancelledError
        # tracebacks (normal ws disconnect / supersede-cancel) so ERROR stays
        # meaningful. Idempotent — safe if multiple servers are constructed.
        install_asgi_cancellederror_filter()
        self._stop = False
        self._websockets = {}  # {conn_key: websocket} — current websocket per client
        self._ws_tasks = {}  # {conn_key: asyncio.Task} — current receive-loop task per client
        self._ws_generation = {}  # {conn_key: int} — monotonic counter to detect stale cleanups
        # Track last activity per connection (workspace/client_id)
        self._last_seen = {}
        # Idle timeout in seconds (applied to all connections, especially effective for anonymous churn)
        self._idle_timeout = int(os.environ.get("HYPHA_WS_IDLE_TIMEOUT", "600"))
        # Per-client message rate limiting (token bucket)
        self._msg_rate_limit = float(os.environ.get("HYPHA_WS_MSG_RATE_LIMIT", "200"))
        self._msg_burst_limit = int(os.environ.get("HYPHA_WS_MSG_BURST_LIMIT", "2000"))
        # Connection admission control: limit concurrent connection setups to
        # prevent reconnection storms from saturating the event loop and Redis.
        self._max_concurrent_connections = int(
            os.environ.get("HYPHA_MAX_CONCURRENT_CONNECTIONS", "10")
        )
        self._connection_semaphore = asyncio.Semaphore(
            self._max_concurrent_connections
        )
        # Background task to clean up idle connections
        try:
            # Check if there's a running event loop before creating the coroutine
            loop = asyncio.get_running_loop()
            self._idle_cleanup_task = asyncio.create_task(self._idle_cleanup_loop())
        except RuntimeError:
            # No running loop yet (e.g., during startup tests); will be created later on first use
            self._idle_cleanup_task = None

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
                try:
                    # Avoid dangling sockets that never send auth payloads
                    auth_info = await asyncio.wait_for(websocket.receive_text(), timeout=5)
                except asyncio.TimeoutError:
                    reason = "Authentication payload timeout"
                    await self.disconnect(
                        websocket,
                        reason=reason,
                        code=status.WS_1008_POLICY_VIOLATION,
                    )
                    return
                except WebSocketDisconnect:
                    # Client went away before sending its auth payload (e.g. code
                    # 1005 — no close frame). The socket is already gone, so there
                    # is nothing to close; just return. Without this, the
                    # WebSocketDisconnect propagated uncaught out of the ASGI app
                    # and uvicorn logged it as "Exception in ASGI application". A
                    # client bailing mid-handshake is benign — log at DEBUG.
                    logger.debug(
                        "Client disconnected during auth handshake before sending credentials"
                    )
                    return
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

            # Admission control: wait for a semaphore slot before doing
            # the heavy auth + workspace-load + client-check work.
            # This prevents 50+ simultaneous reconnections from hammering
            # Redis with concurrent SCAN/DELETE operations.
            try:
                await asyncio.wait_for(
                    self._connection_semaphore.acquire(), timeout=30
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Connection admission timeout — too many concurrent connections"
                )
                await self.disconnect(
                    websocket,
                    reason="Server busy, please retry",
                    code=status.WS_1013_TRY_AGAIN_LATER,
                )
                return

            # Add jitter for reconnecting clients to spread the load
            if reconnection_token:
                jitter = random.uniform(0, 1.0)
                await asyncio.sleep(jitter)

            try:
                if token or reconnection_token:
                    user_info, workspace = await self.authenticate_user(
                        token, reconnection_token, client_id, workspace
                    )
                    # Don't default to "public" workspace for regular users.
                    # If the token doesn't explicitly scope to "public", redirect
                    # to the token's workspace or the user's own workspace.
                    if (
                        workspace == "public"
                        and user_info.scope.current_workspace != "public"
                    ):
                        workspace = (
                            user_info.scope.current_workspace
                            or user_info.get_workspace()
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

                # Check resource limits (blocked user/IP, connection limits)
                resource_limits = self.store.get_resource_limits()
                _fwd = websocket.headers.get("x-forwarded-for")
                client_ip = (
                    _fwd.split(",")[0].strip()
                    if _fwd
                    else (websocket.scope.get("client") or ("unknown",))[0]
                )
                rejection = await resource_limits.check_connection_allowed(
                    user_info.id, client_ip, workspace
                )
                if rejection:
                    await self.disconnect(
                        websocket,
                        reason=rejection,
                        code=status.WS_1008_POLICY_VIOLATION,
                    )
                    return

                # Mark as authenticated before checking client
                authenticated = True
                
                # If the client is not reconnecting, check if it exists
                if not reconnection_token:
                    # We operate as the root user to remove and add clients
                    await self.check_client(client_id, workspace_info.id, user_info)
            except AuthenticationError as e:
                # Expired/invalid client credentials: an EXPECTED client condition,
                # not a server error. Auth fails before any client/service state is
                # created (authenticated is still False), so no cleanup is needed —
                # emit a single WARNING (no traceback) and close 1008. This keeps
                # routine token expiry off the exc_info=True catch-all below. #0008
                logger.warning(
                    f"Authentication failed for client "
                    f"{workspace or '?'}/{client_id or '?'}: {e}"
                )
                await self.disconnect(
                    websocket,
                    reason=f"Authentication failed: {e}",
                    code=status.WS_1008_POLICY_VIOLATION,
                )
                return
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
            finally:
                # Release the admission semaphore after setup is done (or failed).
                # The long-running communication loop does not need to hold the slot.
                self._connection_semaphore.release()

            try:
                await self.establish_websocket_communication(
                    websocket, workspace_info.id, client_id, user_info
                )
                # Normal return (idle timeout or server stopping) — still need
                # to remove the client's services from Redis.
                await self.handle_disconnection(
                    websocket,
                    workspace,
                    client_id,
                    user_info,
                    status.WS_1000_NORMAL_CLOSURE,
                    None,
                )
            except RuntimeError as exp:
                # this happens when the websocket is closed
                if _is_expected_close_error(exp):
                    # Benign peer-closed-underneath-us race — cleanup still runs
                    # below; log at DEBUG so it doesn't mask real errors.
                    logger.debug(
                        "WebSocket %s/%s closed underneath handler: %s",
                        workspace, client_id, str(exp),
                    )
                else:
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
                # Reject specialized tokens - they cannot be used for WebSocket connections
                user_info.validate_for_general_access()
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
                # Reject specialized tokens - they cannot be used for WebSocket connections
                # Specialized tokens (with extra_scopes) are only valid for specific operations
                # like file downloads, not for general workspace access
                user_info.validate_for_general_access()
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
            # Expired/invalid client token — an EXPECTED client condition, not a
            # server error. Raise a dedicated type so the caller emits a single
            # WARNING (no traceback) and closes 1008, instead of this reaching the
            # exc_info=True setup catch-all (which logged a full ERROR traceback
            # for every routine token expiry). Do NOT log here — the caller owns
            # the single log line. See issue #0008.
            raise AuthenticationError(e.detail) from e
        except Exception as e:
            logger.error(f"Failed to authenticate user: {str(e)}")
            raise RuntimeError(f"Failed to authenticate user: {str(e)}")

    async def establish_websocket_communication(
        self, websocket, workspace, client_id, user_info
    ):
        """Establish and manage websocket communication."""
        conn = None
        # Bind early so the finally block can safely reference it even if setup
        # raises before the per-connection bookkeeping below.
        current_task_ref = asyncio.current_task()

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
        conn_key = f"{workspace}/{client_id}"
        # Bump generation counter so stale cleanups from old connections are skipped
        gen = self._ws_generation.get(conn_key, 0) + 1
        self._ws_generation[conn_key] = gen

        # Actively tear down any superseded connection for this client_id. Bumping
        # the generation and overwriting _websockets/_last_seen below is NOT enough:
        # the old connection's coroutine is parked in `await websocket.receive()`
        # (line ~520) and only exits — running its finally-cleanup (conn.disconnect)
        # — when that receive returns or the coroutine is cancelled. If the old
        # socket keeps delivering frames (client keepalive pings every 20s, or a
        # half-open socket behind a proxy), the per-receive idle timeout never
        # fires; and once _websockets/_last_seen point at the NEW generation, the
        # idle sweeper can't see the old one either. The old coroutine — and its
        # RedisRPCConnection + event-bus handlers + ~50 RPC method-proxy coroutines
        # — would then leak forever (prod OOM: live RedisRPCConnection >> registry
        # size). Cancel the old task so its finally runs cleanup now; uvicorn
        # closes the superseded socket when its handler task ends.
        old_task = self._ws_tasks.get(conn_key)
        self._websockets[conn_key] = websocket
        self._ws_tasks[conn_key] = current_task_ref
        if (
            old_task is not None
            and old_task is not current_task_ref
            and not old_task.done()
        ):
            logger.info(
                "Superseding stale websocket connection for %s (cancelling old generation)",
                conn_key,
            )
            old_task.cancel()

        # Tag websocket with its generation for use in handle_disconnection
        websocket._ws_gen = gen
        self._last_seen[conn_key] = time.time()
        rate_limiter = TokenBucketRateLimiter(self._msg_rate_limit, self._msg_burst_limit)
        # Track connection in resource limits manager
        self.store.get_resource_limits().on_client_connected(
            conn_key, user_info.id, workspace
        )
        try:
            _gauge.inc()  # Increment total connections without workspace label
            _connections_established.inc()  # Monotonic counter, never decrements
            event_bus.on_local(f"unload:{workspace}", force_disconnect)

            # F6 (multi-replica): announce this client to sibling pods so they
            # clear any stale recently-disconnected cache entry — otherwise, after
            # a re-pin to this pod, the old pod would false-reject messages to it.
            await event_bus.notify_client_connected(workspace, client_id)

            async def send_bytes(data):
                try:
                    # Handle both bytes and dict data
                    if isinstance(data, dict):
                        data = msgpack.packb(data)
                    elif isinstance(data, str):
                        data = data.encode('utf-8')
                    await websocket.send_bytes(data)
                except Exception as e:
                    # Best-effort send: a failure here means the socket is closing
                    # underneath us (expected during reconnect churn / teardown).
                    # The receive loop + idle timeout determine real liveness and
                    # reap the connection, so this is DEBUG, not an ERROR.
                    logger.debug("Failed to send message via websocket: %s", str(e))

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
                # Apply receive timeout to prevent stuck connections from lingering forever
                try:
                    data = await asyncio.wait_for(websocket.receive(), timeout=self._idle_timeout)
                except asyncio.TimeoutError:
                    # Idle timeout reached
                    await self.disconnect(
                        websocket,
                        reason=f"Idle timeout ({self._idle_timeout}s)",
                        code=status.WS_1001_GOING_AWAY,
                    )
                    break
                # Mark activity
                self._last_seen[conn_key] = time.time()
                # The peer closed the connection: the ASGI transport delivered a
                # `websocket.disconnect` message. Break cleanly instead of looping
                # back into websocket.receive(), which would raise
                # `RuntimeError: Cannot call "receive" once a disconnect message
                # has been received.` (~260/24h expected-close noise). The finally
                # block runs conn.disconnect() cleanup.
                if data.get("type") == "websocket.disconnect":
                    logger.debug(
                        "WebSocket client %s disconnected (code %s)",
                        conn_key,
                        data.get("code"),
                    )
                    break
                if "bytes" in data:
                    data = data["bytes"]
                    allowed, remaining = rate_limiter.consume()
                    if not allowed:
                        logger.warning(
                            "Rate limit exceeded for %s, disconnecting",
                            conn_key,
                        )
                        _rate_limited_disconnects.inc()
                        await self.disconnect(
                            websocket,
                            reason="Rate limit exceeded: too many messages per second",
                            code=status.WS_1008_POLICY_VIOLATION,
                        )
                        break
                    try:
                        await conn.emit_message(data)
                    except Exception as emit_err:
                        # Dead-peer detection: send RPC reject back to caller
                        # Uses the standard RPC protocol so no hypha-rpc changes needed
                        error_msg = str(emit_err)
                        logger.warning("Message routing failed: %s", error_msg)
                        try:
                            unpacker = msgpack.Unpacker(io.BytesIO(data))
                            msg = unpacker.unpack()
                            session_id = msg.get("session")
                            if session_id:
                                # Construct a method call to {session}.reject
                                # This invokes the caller's reject callback
                                reject_main = {
                                    "type": "method",
                                    "from": msg.get("to", ""),
                                    "to": msg.get("from", ""),
                                    "method": f"{session_id}.reject",
                                }
                                reject_extra = {
                                    "args": [
                                        {
                                            "_rtype": "error",
                                            "_rvalue": error_msg,
                                            "_rtrace": "",
                                        }
                                    ],
                                }
                                response = msgpack.packb(
                                    reject_main
                                ) + msgpack.packb(reject_extra)
                                await websocket.send_bytes(response)
                        except Exception:
                            logger.debug(
                                "Could not send reject response: %s", error_msg
                            )
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

            # Check if a newer connection has taken over this client_id.
            # If so, skip cleanup to avoid clobbering the new connection.
            is_current = self._ws_generation.get(conn_key) == gen

            # Ensure proper cleanup even if conn was not created
            if conn:
                await conn.disconnect("disconnected")
            event_bus.off_local(f"unload:{workspace}", force_disconnect)
            if is_current:
                if conn_key in self._websockets:
                    del self._websockets[conn_key]
                # Clear the task handle only if it is still ours (a newer
                # generation may have already replaced it).
                if self._ws_tasks.get(conn_key) is current_task_ref:
                    self._ws_tasks.pop(conn_key, None)
                # Clear last seen
                self._last_seen.pop(conn_key, None)
            else:
                logger.info(
                    "Skipping websocket dict cleanup for %s (superseded by newer connection, gen %s vs current %s)",
                    conn_key, gen, self._ws_generation.get(conn_key),
                )

    async def handle_disconnection(
        self, websocket, workspace: str, client_id: str, user_info: UserInfo, code, exp
    ):
        """Handle client disconnection with proper cleanup.

        If a newer WebSocket connection has already taken over for this
        client_id (reconnection race), skip the remove_client call to
        avoid deleting the new connection's services.
        """
        conn_key = f"{workspace}/{client_id}"
        # Use generation counter to detect if a newer connection has taken over.
        # Each new connection bumps the generation and tags it on the websocket.
        # If our generation doesn't match the current one, a reconnection happened.
        my_gen = getattr(websocket, "_ws_gen", 0)
        current_gen = self._ws_generation.get(conn_key, 0)
        if my_gen < current_gen:
            # A newer connection has taken over — do NOT remove services,
            # but DO decrement the resource counter for this old connection.
            logger.info(
                "Skipping remove_client for %s: superseded by newer connection (gen %s < %s)",
                conn_key, my_gen, current_gen,
            )
            # Still close this old websocket gracefully
            await self.disconnect(
                websocket,
                f"Old connection for {conn_key} superseded",
                code,
            )
            # Note: on_client_disconnected uses _conn_user/_conn_workspace which
            # were overwritten by the new connection's on_client_connected call,
            # so we decrement directly using the user_info we have.
            rl = self.store.get_resource_limits()
            rl._decrement_connection(user_info.id, workspace)
            return

        # Track WebSocket client as recently disconnected for dead-peer detection
        self.store._event_bus.track_recently_disconnected(workspace, client_id)
        # Decrement resource limits counters
        self.store.get_resource_limits().on_client_disconnected(conn_key)
        cleanup_succeeded = False
        try:
            # Re-check generation right before the destructive remove_client.
            # A reconnecting client may have bumped the generation while we
            # were doing earlier async work (event tracking, etc.).
            if self._ws_generation.get(conn_key, 0) > my_gen:
                logger.info(
                    "Skipping remove_client for %s: new connection during cleanup (gen %s -> %s)",
                    conn_key, my_gen, self._ws_generation.get(conn_key, 0),
                )
                return
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
        # A disconnect is a normal lifecycle event (idle timeout, client close,
        # supersede, server stop) — log at INFO so ERROR stays reserved for real
        # errors. Previously this fired at ERROR on EVERY disconnect and was the
        # loudest benign-at-ERROR line, masking genuine errors during triage.
        logger.info("Disconnecting, reason: %s, code: %s", reason, code)
        if websocket.client_state.name == "CONNECTED":
            try:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": reason})
                )
            except Exception as e:
                # Send-after-close race: the socket closed between the state check
                # and the send (e.g. RuntimeError "Cannot call send once a close
                # message has been sent"). Benign — proceed to close.
                logger.debug(
                    "Could not send disconnect notice (socket already closing): %s", e
                )
        try:
            if websocket.client_state.name not in ["CLOSED", "CLOSING", "DISCONNECTED"]:
                try:
                    await websocket.close(code=code, reason=reason)
                except GeneratorExit:
                    pass  # Suppress GeneratorExit to avoid RuntimeError
        except Exception as e:
            # The socket was already closing/gone when we tried to close it — a
            # benign close race, not a server error.
            logger.debug(f"Error disconnecting websocket: {str(e)}")

    async def is_alive(self):
        """Check if the server is alive."""
        return True

    async def close_all_clients(
        self,
        code=status.WS_1001_GOING_AWAY,
        reason="Server is shutting down",
    ):
        """Proactively send a close frame to every active websocket client.

        Used during graceful shutdown (SIGTERM/rollout): emitting an explicit
        "going away" close frame lets clients reconnect immediately instead of
        waiting out their heartbeat timeout, which prevents a thundering-herd
        reconnection storm when a pod is redeployed.

        The hypha-rpc client treats close codes 1000/1001 (when not
        user-initiated) as a signal to reconnect, so a single replica draining
        its connections hands clients straight to the surviving replica.

        Returns the number of clients that were signaled.
        """
        websockets = list(self._websockets.values())
        if not websockets:
            return 0
        logger.info(
            "Gracefully closing %d websocket client(s) with code %s",
            len(websockets),
            code,
        )

        async def _close(ws):
            try:
                if ws.client_state.name == "CONNECTED":
                    await ws.close(code=code, reason=reason)
            except GeneratorExit:
                pass  # Suppress GeneratorExit to avoid RuntimeError
            except Exception as e:
                # One unresponsive socket must not block draining the rest.
                logger.warning("Error closing websocket during shutdown: %s", e)

        await asyncio.gather(*[_close(ws) for ws in websockets])
        return len(websockets)

    async def stop(self):
        """Stop the server."""
        self._stop = True
        # Stop idle cleanup task
        if getattr(self, "_idle_cleanup_task", None):
            self._idle_cleanup_task.cancel()
            try:
                await self._idle_cleanup_task
            except asyncio.CancelledError:
                # Expected when cancelling the task
                pass
            self._idle_cleanup_task = None

    async def _do_idle_cleanup(self):
        """Core idle cleanup logic — extracted for testability.

        TOCTOU safety: after building the candidate list we re-read _last_seen
        using a fresh timestamp for each key before calling disconnect().  A
        connection that received a message between the snapshot and the actual
        close call is no longer idle and must not be forcibly terminated.
        """
        now = time.time()
        to_disconnect = [
            key
            for key, last in list(self._last_seen.items())
            if now - last > self._idle_timeout
        ]
        for key in to_disconnect:
            ws = self._websockets.get(key)
            if ws:
                # Re-check with a fresh timestamp: did the connection become
                # active after we snapshotted it?
                current_last = self._last_seen.get(key)
                if current_last is not None and time.time() - current_last <= self._idle_timeout:
                    continue  # No longer idle — skip
                try:
                    await self.disconnect(
                        ws,
                        f"Idle timeout ({self._idle_timeout}s)",
                        status.WS_1001_GOING_AWAY,
                    )
                    self._last_seen.pop(key, None)
                except Exception as e:
                    logger.debug("Error disconnecting idle WebSocket %s: %s", key, e)
                    self._last_seen.pop(key, None)
            else:
                # Ghost entry: in _last_seen but no active WebSocket — clean it up.
                # This can happen when a reconnection race leaves _last_seen populated
                # but the WebSocket was already removed from _websockets.
                logger.debug("Removing ghost _last_seen entry for %s (no active WebSocket)", key)
                self._last_seen.pop(key, None)
                # Ensure resource counters are decremented for this ghost entry
                if self.store is not None:
                    self.store.get_resource_limits().on_client_disconnected(key)

    async def _idle_cleanup_loop(self):
        """Periodically disconnect idle connections and reconcile counters."""
        reconcile_counter = 0
        while not self._stop:
            try:
                await asyncio.sleep(60)
                await self._do_idle_cleanup()
                # Reconcile resource counters every ~5 minutes (every 5th iteration)
                reconcile_counter += 1
                if reconcile_counter >= 5 and self.store is not None:
                    reconcile_counter = 0
                    active_keys = set(self._websockets.keys())
                    self.store.get_resource_limits().reconcile_connections(
                        active_keys
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                # Do not crash the loop
                pass
