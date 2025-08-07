"""Provide an SSE (Server-Sent Events) client."""

import asyncio
import inspect
import logging
import sys
import os
import json
import base64
import aiohttp
from aiohttp import ClientSession
from aiohttp_sse_client import client as sse_client

import shortuuid
from hypha_rpc.rpc import RPC
from hypha_rpc.utils.schema import schema_function

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("sse-client")
logger.setLevel(LOGLEVEL)

MAX_RETRY = 1000000


class SSERPCConnection:
    """Represent an SSE connection."""

    def __init__(
        self,
        server_url,
        client_id,
        workspace=None,
        token=None,
        reconnection_token=None,
        timeout=60,
        ssl=None,
        token_refresh_interval=2 * 60 * 60,
        additional_headers=None,
    ):
        """Set up instance."""
        self._sse_client = None
        self._session = None
        self._handle_message = None
        self._handle_disconnected = None  # Disconnection handler
        self._handle_connected = None  # Connection open handler
        self._last_message = None  # Store the last sent message
        assert server_url and client_id
        
        # Convert ws/wss URLs to http/https
        if server_url.startswith("ws://"):
            server_url = server_url.replace("ws://", "http://")
        elif server_url.startswith("wss://"):
            server_url = server_url.replace("wss://", "https://")
        
        # Convert /ws endpoint to /sse
        if server_url.endswith("/ws"):
            server_url = server_url[:-3] + "/sse"
        
        self._server_url = server_url
        self._messages_url = server_url.replace("/sse", "/messages")
        self._client_id = client_id
        self._workspace = workspace
        self._token = token
        self._reconnection_token = reconnection_token
        self._timeout = timeout
        self._closed = False
        self.connection_info = None
        self._enable_reconnect = False
        self._refresh_token_task = None
        self._listen_task = None
        self._token_refresh_interval = token_refresh_interval
        self._additional_headers = additional_headers or {}
        self._reconnect_tasks = set()  # Track reconnection tasks
        self._ssl = ssl
        self.manager_id = None

    def on_message(self, handler):
        """Handle message."""
        self._handle_message = handler
        self._is_async = inspect.iscoroutinefunction(handler)

    def on_disconnected(self, handler):
        """Register a disconnection event handler."""
        self._handle_disconnected = handler

    def on_connected(self, handler):
        """Register a connection open event handler."""
        self._handle_connected = handler
        assert inspect.iscoroutinefunction(
            handler
        ), "reconnect handler must be a coroutine"

    async def _send_refresh_token(self, token_refresh_interval):
        """Send refresh token at regular intervals."""
        try:
            await asyncio.sleep(2)
            while not self._closed and self._session:
                # Create the refresh token message
                refresh_message = json.dumps({"type": "refresh_token"})
                # Send the message to the server via POST
                await self.emit_message(refresh_message.encode('utf-8'))
                # Wait for the next refresh interval
                await asyncio.sleep(token_refresh_interval)
        except asyncio.CancelledError:
            # Task was cancelled, cleanup or exit gracefully
            logger.info("Refresh token task was cancelled.")
        except Exception as exp:
            logger.error(f"Failed to send refresh token: {exp}")

    async def open(self):
        """Open the SSE connection."""
        logger.info(
            "Creating a new SSE connection to %s", self._server_url
        )
        try:
            # Create session
            connector = aiohttp.TCPConnector(ssl=self._ssl)
            self._session = ClientSession(connector=connector)
            
            # Build query parameters
            params = {}
            if self._workspace:
                params["workspace"] = self._workspace
            if self._client_id:
                params["client_id"] = self._client_id
            if self._token:
                params["token"] = self._token
            if self._reconnection_token:
                params["reconnection_token"] = self._reconnection_token
            
            # Connect to SSE endpoint
            self._sse_client = sse_client.EventSource(
                self._server_url,
                session=self._session,
                params=params,
                headers=self._additional_headers,
                timeout=self._timeout
            )
            
            # Connect to the SSE endpoint
            await self._sse_client.connect()
            
            # Wait for connection info
            async for event in self._sse_client:
                if event.data:
                    try:
                        first_message = json.loads(event.data)
                        if first_message.get("type") == "connection_info":
                            self.connection_info = first_message
                            if self._workspace:
                                assert (
                                    self.connection_info.get("workspace") == self._workspace
                                ), f"Connected to the wrong workspace: {self.connection_info['workspace']}, expected: {self._workspace}"
                            if "reconnection_token" in self.connection_info:
                                self._reconnection_token = self.connection_info[
                                    "reconnection_token"
                                ]
                            if "reconnection_token_life_time" in self.connection_info:
                                if (
                                    self._token_refresh_interval
                                    > self.connection_info["reconnection_token_life_time"] / 1.5
                                ):
                                    logger.warning(
                                        f"Token refresh interval is too long ({self._token_refresh_interval}), setting it to 1.5 times of the token life time({self.connection_info['reconnection_token_life_time']})."
                                    )
                                    self._token_refresh_interval = (
                                        self.connection_info["reconnection_token_life_time"] / 1.5
                                    )
                            # Manager ID must always be present in connection_info
                            assert "manager_id" in self.connection_info, f"manager_id missing in connection_info: {self.connection_info}"
                            self.manager_id = self.connection_info["manager_id"]
                            logger.info(
                                f"Successfully connected to the server, workspace: {self.connection_info.get('workspace')}, manager_id: {self.manager_id}"
                            )
                            if "announcement" in self.connection_info:
                                print(self.connection_info["announcement"])
                            break
                        elif first_message.get("type") == "error":
                            error = first_message["message"]
                            logger.error("Failed to connect: %s", error)
                            raise ConnectionAbortedError(error)
                    except json.JSONDecodeError:
                        pass
            
            if self._token_refresh_interval > 0:
                self._refresh_token_task = asyncio.create_task(
                    self._send_refresh_token(self._token_refresh_interval)
                )
            self._listen_task = asyncio.ensure_future(self._listen())
            if self._handle_connected:
                await self._handle_connected(self.connection_info)
            return self.connection_info
        except Exception as exp:
            # Clean up any tasks that might have been created before the error
            try:
                await self._cleanup()
                if self._sse_client:
                    await self._sse_client.close()
                if self._session:
                    await self._session.close()
            except Exception as cleanup_error:
                logger.debug(f"Error during cleanup after connection failure: {cleanup_error}")
            logger.error("Failed to connect to %s", self._server_url)
            raise exp

    async def emit_message(self, data):
        """Emit a message via POST."""
        if self._closed:
            raise Exception("Connection is closed")
        if not self._session:
            await self.open()

        try:
            self._last_message = data  # Store the message before sending
            
            # Build query parameters for POST
            params = {
                "workspace": self._workspace or self.connection_info.get("workspace"),
                "client_id": self._client_id
            }
            
            async with self._session.post(
                self._messages_url,
                data=data,
                params=params,
                headers=self._additional_headers,
                timeout=self._timeout
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to send message: {response.status}")
            
            self._last_message = None  # Clear after successful send
        except Exception as exp:
            logger.error(f"Failed to send message: {exp}")
            raise exp

    async def _listen(self):
        """Listen to the SSE connection and handle disconnection."""
        self._enable_reconnect = True
        self._closed = False
        try:
            async for event in self._sse_client:
                if self._closed:
                    break
                
                if event.data:
                    try:
                        # Check if it's a close event
                        try:
                            data = json.loads(event.data)
                            if data.get("type") == "close":
                                logger.info(f"Server closed connection: {data.get('reason')}")
                                break
                            elif data.get("type") == "reconnection_token":
                                self._reconnection_token = data.get("reconnection_token")
                                continue
                        except json.JSONDecodeError:
                            pass
                        
                        # Try to decode as base64 (binary message)
                        try:
                            decoded = base64.b64decode(event.data)
                            if self._handle_message:
                                if self._is_async:
                                    await self._handle_message(decoded)
                                else:
                                    self._handle_message(decoded)
                        except Exception:
                            # If not base64, treat as text
                            if self._handle_message:
                                if self._is_async:
                                    await self._handle_message(event.data.encode('utf-8'))
                                else:
                                    self._handle_message(event.data.encode('utf-8'))
                    except Exception as exp:
                        logger.exception(
                            "Failed to handle message: %s, error: %s", event.data, exp
                        )
        except asyncio.CancelledError:
            logger.info("Listen task was cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in _listen: {e}")
            raise
        finally:
            # Handle unexpected disconnection
            if not self._closed and self._enable_reconnect:
                logger.warning("SSE connection closed unexpectedly")
                
                async def reconnect_with_retry():
                    retry = 0
                    base_delay = 1.0  # Start with 1 second
                    max_delay = 60.0  # Maximum delay of 60 seconds

                    while retry < MAX_RETRY and not self._closed:
                        try:
                            logger.warning(
                                "Reconnecting to %s (attempt #%s)",
                                self._server_url,
                                retry,
                            )
                            # Open the connection
                            connection_info = await self.open()

                            # Wait a short time for services to be registered
                            await asyncio.sleep(1.0)

                            # Resend last message if there was one
                            if self._last_message:
                                logger.info(
                                    "Resending last message after reconnection"
                                )
                                await self.emit_message(self._last_message)
                                self._last_message = None
                            logger.warning(
                                "Successfully reconnected to %s",
                                self._server_url,
                            )
                            # Emit reconnection success event
                            if self._handle_connected:
                                await self._handle_connected(connection_info)
                            break
                        except ConnectionAbortedError as e:
                            logger.warning("Server refuse to reconnect: %s", e)
                            break
                        except (ConnectionRefusedError, OSError) as e:
                            logger.error(
                                f"Failed to connect to {self._server_url}: {e}"
                            )
                        except asyncio.TimeoutError as e:
                            logger.error(
                                f"Connection timeout to {self._server_url}: {e}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Unexpected error during reconnection: {e}"
                            )

                        # Calculate exponential backoff
                        delay = min(base_delay * (2**retry), max_delay)
                        
                        logger.debug(
                            f"Waiting {delay:.2f}s before next reconnection attempt"
                        )

                        # Use tracked sleep task for cancellation
                        sleep_task = asyncio.create_task(asyncio.sleep(delay))
                        self._reconnect_tasks.add(sleep_task)
                        try:
                            await sleep_task
                        except asyncio.CancelledError:
                            logger.info("Reconnection cancelled")
                            self._reconnect_tasks.discard(sleep_task)
                            return  # Exit immediately on cancellation
                        finally:
                            self._reconnect_tasks.discard(sleep_task)

                        # Check if we were explicitly closed
                        if self._closed:
                            logger.info(
                                "Connection was closed, stopping reconnection"
                            )
                            break

                        retry += 1

                    if retry >= MAX_RETRY and not self._closed:
                        logger.error(
                            f"Failed to reconnect after {MAX_RETRY} attempts, giving up. Exiting process."
                        )
                        if self._handle_disconnected:
                            self._handle_disconnected(
                                "Max reconnection attempts exceeded"
                            )
                        # Exit process to prevent stuck event loop
                        import os
                        logger.error(
                            "Forcing process exit due to unrecoverable connection failure"
                        )
                        os._exit(1)

                # Create and track the reconnection task
                reconnect_task = asyncio.create_task(reconnect_with_retry())
                self._reconnect_tasks.add(reconnect_task)
                # Remove task from tracking when it completes
                reconnect_task.add_done_callback(
                    lambda t: self._reconnect_tasks.discard(t)
                )
            else:
                if self._handle_disconnected:
                    self._handle_disconnected("Connection closed")

    async def disconnect(self, reason=None):
        """Disconnect."""
        self._closed = True
        self._last_message = None
        
        # Use centralized cleanup to cancel all tasks
        try:
            await self._cleanup()
        except Exception as e:
            # Event loop might be closed during shutdown
            logger.warning(f"Error during cleanup: {e}")
        
        # Close SSE client and session after cleanup
        if self._sse_client:
            try:
                await self._sse_client.close()
            except Exception as e:
                logger.debug(f"Error closing SSE client: {e}")
        
        if self._session:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
                
        logger.info("SSE connection disconnected (%s)", reason)

    async def _cleanup(self):
        """Centralized cleanup method to cancel all tasks and prevent resource leaks."""
        try:
            # Check if event loop is running before cleanup
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                logger.debug("Event loop is closed, performing minimal cleanup")
                self._refresh_token_task = None
                self._listen_task = None
                self._reconnect_tasks.clear()
                return

            # Cancel token refresh task
            if self._refresh_token_task and not self._refresh_token_task.done():
                self._refresh_token_task.cancel()
                try:
                    await asyncio.wait_for(self._refresh_token_task, timeout=1.0)
                except (asyncio.CancelledError, RuntimeError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    logger.debug(f"Error waiting for refresh token task: {e}")
                self._refresh_token_task = None

            # Cancel listen task
            if self._listen_task and not self._listen_task.done():
                self._listen_task.cancel()
                try:
                    await asyncio.wait_for(self._listen_task, timeout=1.0)
                except (asyncio.CancelledError, RuntimeError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    logger.debug(f"Error waiting for listen task: {e}")
                self._listen_task = None

            # Cancel all reconnection tasks
            for task in list(self._reconnect_tasks):
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=0.5)
                    except (asyncio.CancelledError, RuntimeError, asyncio.TimeoutError):
                        pass
                    except Exception as e:
                        logger.debug(f"Error waiting for reconnect task: {e}")
                self._reconnect_tasks.discard(task)
            
            # Clear any remaining tasks
            self._reconnect_tasks.clear()
            
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.debug("Event loop closed during cleanup, performing minimal cleanup")
                self._refresh_token_task = None
                self._listen_task = None
                self._reconnect_tasks.clear()
            else:
                logger.warning(f"RuntimeError during cleanup: {e}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        finally:
            # Ensure tasks are marked as None even if cleanup fails
            self._refresh_token_task = None
            self._listen_task = None
            self._reconnect_tasks.clear()


def normalize_server_url(server_url):
    """Normalize the server url for SSE."""
    if not server_url:
        raise ValueError("server_url is required")

    if server_url.startswith("http://"):
        server_url = server_url.rstrip("/") + "/sse"
    elif server_url.startswith("https://"):
        server_url = server_url.rstrip("/") + "/sse"
    elif server_url.startswith("ws://"):
        server_url = server_url.replace("ws://", "http://").rstrip("/") + "/sse"
    elif server_url.startswith("wss://"):
        server_url = server_url.replace("wss://", "https://").rstrip("/") + "/sse"

    return server_url


async def connect_to_server(config):
    """Connect to RPC via SSE to a hypha server."""
    client_id = config.get("client_id")
    if client_id is None:
        client_id = shortuuid.uuid()

    server_url = normalize_server_url(config["server_url"])

    connection = SSERPCConnection(
        server_url,
        client_id,
        workspace=config.get("workspace"),
        token=config.get("token"),
        reconnection_token=config.get("reconnection_token"),
        timeout=config.get("method_timeout", 30),
        ssl=config.get("ssl"),
        token_refresh_interval=config.get("token_refresh_interval", 2 * 60 * 60),
        additional_headers=config.get("additional_headers"),
    )
    connection_info = await connection.open()
    assert connection_info, (
        "Failed to connect to the server, no connection info obtained."
    )
    # Manager ID must be set from connection_info
    assert connection.manager_id, "Manager ID must be available after connection is established"

    if config.get("workspace") and connection_info["workspace"] != config["workspace"]:
        raise Exception(
            f"Connected to the wrong workspace: {connection_info['workspace']}, expected: {config['workspace']}"
        )
    workspace = connection_info["workspace"]
    rpc = RPC(
        connection,
        client_id=client_id,
        workspace=workspace,
        default_context={"connection_type": "sse"},
        name=config.get("name"),
        method_timeout=config.get("method_timeout"),
        loop=config.get("loop"),
        app_id=config.get("app_id"),
        server_base_url=connection_info.get("public_base_url"),
        long_message_chunk_size=config.get("long_message_chunk_size"),
        enable_http_transmission=config.get("enable_http_transmission", True),
        http_transmission_threshold=config.get("http_transmission_threshold", 1024 * 1024),
        multipart_threshold=config.get("multipart_threshold", 10 * 1024 * 1024),
        multipart_size=config.get("multipart_size", 6 * 1024 * 1024),
        max_parallel_uploads=config.get("max_parallel_uploads", 5),
    )
    await rpc.wait_for("services_registered", timeout=config.get("method_timeout", 120))
    wm = await rpc.get_manager_service(
        {"timeout": config.get("method_timeout", 30), "case_conversion": "snake"}
    )
    wm.rpc = rpc

    def export(api: dict):
        """Export the api."""
        # Convert class instance to a dict
        if not isinstance(api, dict) and inspect.isclass(type(api)):
            api = {a: getattr(api, a) for a in dir(api)}
        api["id"] = "default"
        api["description"] = api.get("description") or config.get("description")
        return asyncio.ensure_future(rpc.register_service(api, {"overwrite": True}))

    async def get_app(client_id: str):
        """Get the app."""
        assert ":" not in client_id, "clientId should not contain ':'"
        if "/" not in client_id:
            client_id = connection_info["workspace"] + "/" + client_id
        assert (
            len(client_id.split("/")) == 2
        ), "clientId should be in the format of 'workspace/client_id'"
        return await wm.get_service(f"{client_id}:default")

    async def list_apps(workspace: str = None):
        """List the apps."""
        workspace = workspace or connection_info["workspace"]
        assert ":" not in workspace, "workspace should not contain ':'"
        assert "/" not in workspace, "workspace should not contain '/'"
        query = {"workspace": workspace, "service_id": "default"}
        return await wm.list_services(query)

    if connection_info:
        wm.config.update(connection_info)

    wm.export = schema_function(
        export,
        name="export",
        description="Export the api.",
        parameters={
            "properties": {
                "api": {"description": "The api to export", "type": "object"}
            },
            "required": ["api"],
            "type": "object",
        },
    )
    wm.get_app = schema_function(
        get_app,
        name="get_app",
        description="Get the app.",
        parameters={
            "properties": {
                "clientId": {
                    "default": "*",
                    "description": "The clientId",
                    "type": "string",
                }
            },
            "type": "object",
        },
    )
    wm.list_apps = schema_function(
        list_apps,
        name="list_apps",
        description="List the apps.",
        parameters={
            "properties": {
                "workspace": {
                    "default": workspace,
                    "description": "The workspace",
                    "type": "string",
                }
            },
            "type": "object",
        },
    )
    wm.disconnect = schema_function(
        rpc.disconnect,
        name="disconnect",
        description="Disconnect.",
        parameters={"properties": {}, "type": "object"},
    )
    wm.register_codec = schema_function(
        rpc.register_codec,
        name="register_codec",
        description="Register a codec",
        parameters={
            "type": "object",
            "properties": {
                "codec": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {},
                        "encoder": {"type": "function"},
                        "decoder": {"type": "function"},
                    },
                    "description": "codec",
                }
            },
            "required": ["codec"],
        },
    )
    wm.emit = schema_function(
        rpc.emit,
        name="emit",
        description="Emit a message.",
        parameters={
            "properties": {
                "data": {"description": "The data to emit", "type": "object"}
            },
            "required": ["data"],
            "type": "object",
        },
    )
    wm.on = schema_function(
        rpc.on,
        name="on",
        description="Register a message handler.",
        parameters={
            "properties": {
                "event": {"description": "The event to listen to", "type": "string"},
                "handler": {"description": "The handler function", "type": "function"},
            },
            "required": ["event", "handler"],
            "type": "object",
        },
    )

    wm.off = schema_function(
        rpc.off,
        name="off",
        description="Remove a message handler.",
        parameters={
            "properties": {
                "event": {"description": "The event to remove", "type": "string"},
                "handler": {"description": "The handler function", "type": "function"},
            },
            "required": ["event", "handler"],
            "type": "object",
        },
    )

    wm.once = schema_function(
        rpc.once,
        name="once",
        description="Register a one-time message handler.",
        parameters={
            "properties": {
                "event": {"description": "The event to listen to", "type": "string"},
                "handler": {"description": "The handler function", "type": "function"},
            },
            "required": ["event", "handler"],
            "type": "object",
        },
    )

    wm.get_service_schema = schema_function(
        rpc.get_service_schema,
        name="get_service_schema",
        description="Get the service schema.",
        parameters={
            "properties": {
                "service": {
                    "description": "The service to extract schema",
                    "type": "object",
                },
            },
            "required": ["service"],
            "type": "object",
        },
    )

    wm.register_service = schema_function(
        rpc.register_service,
        name="register_service",
        description="Register a service.",
        parameters={
            "properties": {
                "service": {"description": "The service to register", "type": "object"},
                "force": {
                    "default": False,
                    "description": "Force to register the service",
                    "type": "boolean",
                },
            },
            "required": ["service"],
            "type": "object",
        },
    )
    wm.unregister_service = schema_function(
        rpc.unregister_service,
        name="unregister_service",
        description="Unregister a service.",
        parameters={
            "properties": {
                "service": {
                    "description": "The service id to unregister",
                    "type": "string",
                },
                "notify": {
                    "default": True,
                    "description": "Notify the workspace manager",
                },
            },
            "required": ["service"],
            "type": "object",
        },
    )
    if connection.manager_id:

        async def handle_disconnect(message):
            if message["from"] == "*/" + connection.manager_id:
                logger.info(
                    "Disconnecting from server, reason: %s", message.get("reason")
                )
                await rpc.disconnect()

        rpc.on("force-exit", handle_disconnect)

    async def serve():
        await asyncio.Event().wait()

    wm.serve = schema_function(
        serve, name="serve", description="Run the event loop forever", parameters={}
    )

    async def register_probes(probes):
        probes["id"] = "probes"
        probes["name"] = "Probes"
        probes["config"] = {"visibility": "public"}
        probes["type"] = "probes"
        probes["description"] = (
            f"Probes Service, visit {server_url}/{workspace}services/probes for the available probes."
        )
        return await wm.register_service(probes, {"overwrite": True})

    wm.register_probes = schema_function(
        register_probes,
        name="register_probes",
        description="Register probes service",
        parameters={
            "properties": {
                "probes": {
                    "description": "The probes to register, e.g. {'liveness': {'type': 'function', 'description': 'Check the liveness of the service'}}",
                    "type": "object",
                }
            },
            "required": ["probes"],
            "type": "object",
        },
    )
    return wm