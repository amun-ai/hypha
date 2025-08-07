import logging
import os
import sys
import json
import msgpack
import asyncio
from typing import Dict

from fastapi import Query, Request, status, HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import Response as StarletteResponse
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


LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("sse-server")
logger.setLevel(LOGLEVEL)

_gauge = Gauge(
    "sse_connections", "Number of SSE connections", ["workspace"]
)


class SSEServerTransport:
    """SSE server transport for a single client connection."""
    
    def __init__(self, response: StreamingResponse, workspace: str, client_id: str):
        """Initialize SSE transport."""
        self.response = response
        self.workspace = workspace
        self.client_id = client_id
        self.queue = asyncio.Queue()
        self._closed = False
        self._send_task = None
        
    async def send_bytes(self, data):
        """Send bytes to the client via SSE."""
        try:
            if isinstance(data, dict):
                data = msgpack.packb(data)
            elif isinstance(data, str):
                data = data.encode('utf-8')
            
            # Convert to base64 for SSE transmission
            import base64
            encoded = base64.b64encode(data).decode('utf-8')
            await self.queue.put(f"data: {encoded}\n\n")
        except Exception as e:
            logger.error("Failed to send message via SSE: %s", str(e))
    
    async def send_text(self, text):
        """Send text to the client via SSE."""
        try:
            await self.queue.put(f"data: {text}\n\n")
        except Exception as e:
            logger.error("Failed to send text via SSE: %s", str(e))
    
    async def close(self, code=None, reason=None):
        """Close the SSE connection."""
        self._closed = True
        # Send close event
        close_msg = json.dumps({"type": "close", "code": code, "reason": reason})
        await self.queue.put(f"data: {close_msg}\n\n")
        # Signal end of stream
        await self.queue.put(None)
        
    async def event_generator(self):
        """Generate SSE events."""
        try:
            while not self._closed:
                message = await self.queue.get()
                if message is None:
                    break
                yield message
        except asyncio.CancelledError:
            logger.info("SSE event generator cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in SSE event generator: {e}")
            raise


class SSEServer:
    def __init__(self, store: RedisStore, sse_path="/sse", messages_path="/messages"):
        """Initialize SSE server with the store and set up the endpoints."""
        self.store = store
        app = store._app
        self.store.set_sse_server(self)
        self._stop = False
        self._transports: Dict[str, SSEServerTransport] = {}
        self._connections: Dict[str, RedisRPCConnection] = {}
        
        @app.get(sse_path)
        async def sse_endpoint(
            request: Request,
            workspace: str = Query(None),
            client_id: str = Query(None),
            token: str = Query(None),
            reconnection_token: str = Query(None),
        ):
            """SSE endpoint for event streaming."""
            # Create transport
            transport = SSEServerTransport(None, workspace, client_id)
            transport_key = f"{workspace}/{client_id}"
            
            # Store transport for POST handler
            self._transports[transport_key] = transport
            
            try:
                # Authenticate user
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
                
                # Check if client exists
                if not reconnection_token:
                    await self.check_client(client_id, workspace_info.id, user_info)
                
                # Start establishing connection in background
                async def setup_and_stream():
                    """Setup connection and stream events."""
                    try:
                        # First establish the connection
                        await self.establish_sse_communication(
                            transport, workspace_info.id, client_id, user_info
                        )
                        # Then start streaming events
                        async for event in transport.event_generator():
                            yield event
                    except Exception as e:
                        logger.error(f"Error in SSE stream: {e}")
                        # Send error message and close
                        error_msg = json.dumps({"type": "error", "message": str(e)})
                        yield f"data: {error_msg}\n\n"
                    finally:
                        # Clean up when streaming ends
                        await self.handle_disconnection(
                            transport_key, workspace_info.id, client_id, user_info
                        )
                
                # Return SSE response
                return StreamingResponse(
                    setup_and_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",  # Disable Nginx buffering
                    }
                )
                
            except Exception as e:
                logger.error(f"Error in SSE endpoint: {e}")
                # Clean up transport
                if transport_key in self._transports:
                    del self._transports[transport_key]
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post(messages_path)
        async def messages_endpoint(
            request: Request,
            workspace: str = Query(None),
            client_id: str = Query(None),
        ):
            """POST endpoint for receiving messages from client."""
            transport_key = f"{workspace}/{client_id}"
            
            if transport_key not in self._connections:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No active SSE connection for {transport_key}"
                )
            
            try:
                # Get message body
                body = await request.body()
                
                # Get connection and emit message
                conn = self._connections[transport_key]
                await conn.emit_message(body)
                
                return {"status": "ok"}
                
            except Exception as e:
                logger.error(f"Error handling POST message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
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
    
    async def establish_sse_communication(
        self, transport, workspace, client_id, user_info
    ):
        """Establish and manage SSE communication."""
        conn = None
        transport_key = f"{workspace}/{client_id}"
        
        async def force_disconnect(_):
            logger.info(
                f"Unloading workspace, force disconnecting SSE client: {workspace}/{client_id}"
            )
            await transport.close(code=status.WS_1001_GOING_AWAY, reason=f"Workspace unloaded: {workspace}")
        
        event_bus = self.store.get_event_bus()
        assert (
            user_info.scope.current_workspace == workspace
        ), f"Workspace mismatch: {workspace} != {user_info.scope.current_workspace}"
        
        # Determine if client has read-only permissions
        user_permission = user_info.get_permission(workspace)
        is_readonly = user_permission == UserPermission.read if user_permission else False
        
        conn = RedisRPCConnection(event_bus, workspace, client_id, user_info, self.store.get_manager_id(), readonly=is_readonly)
        self._connections[transport_key] = conn
        
        try:
            _gauge.labels(workspace=workspace).inc()
            event_bus.on_local(f"unload:{workspace}", force_disconnect)
            
            # Set up message handler (this establishes the connection)
            conn.on_message(transport.send_bytes)
            
            # Generate reconnection token
            reconnection_token = generate_reconnection_token(
                user_info, expires_in=self.store.reconnection_token_life_time
            )
            
            # Send connection info
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
            await transport.send_text(json.dumps(conn_info))
            
        except Exception as e:
            # Clean up connection on error
            if transport_key in self._connections:
                del self._connections[transport_key]
            if transport_key in self._transports:
                del self._transports[transport_key]
            _gauge.labels(workspace=workspace).dec()
            raise e
    
    async def handle_disconnection(
        self, transport_key: str, workspace: str, client_id: str, user_info: UserInfo
    ):
        """Handle client disconnection."""
        try:
            await self.store.remove_client(client_id, workspace, user_info, unload=True)
            logger.info(f"Client disconnected: {workspace}/{client_id}")
        except Exception as e:
            logger.error(
                f"Error handling disconnection, client: {workspace}/{client_id}, error: {str(e)}"
            )
        finally:
            # Clean up transport and connection
            if transport_key in self._transports:
                del self._transports[transport_key]
            if transport_key in self._connections:
                conn = self._connections[transport_key]
                try:
                    await conn.disconnect("disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting RPC connection: {e}")
                del self._connections[transport_key]
                _gauge.labels(workspace=workspace).dec()
    
    def get_transports(self):
        """Get the active SSE transports."""
        return self._transports
    
    async def force_disconnect(self, workspace, client_id, code, reason):
        """Force disconnect a client."""
        transport_key = f"{workspace}/{client_id}"
        assert transport_key in self._transports, "Client not connected"
        transport = self._transports[transport_key]
        await transport.close(code=code, reason=reason)
    
    async def is_alive(self):
        """Check if the server is alive."""
        return True
    
    async def stop(self):
        """Stop the server."""
        self._stop = True
        # Close all active transports
        for transport_key in list(self._transports.keys()):
            transport = self._transports[transport_key]
            await transport.close(code=status.WS_1001_GOING_AWAY, reason="Server stopping")