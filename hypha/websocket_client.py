import asyncio
import inspect
import logging
import sys

import msgpack
import websockets
import shortuuid

from hypha.core.rpc import RPC

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket")
logger.setLevel(logging.INFO)


class WebsocketRPCConnection:
    """Represent a websocket connection."""

    def __init__(self, server_url, client_id, workspace=None, token=None):
        """Set up instance."""
        self._websocket = None
        self._handle_message = None
        assert server_url and client_id
        server_url = server_url + f"?client_id={client_id}"
        if workspace is not None:
            server_url += f"&workspace={workspace}"
        if token:
            server_url += f"&token={token}"
        self._server_url = server_url

    def on_message(self, handler):
        self._handle_message = handler
        self._is_async = inspect.iscoroutinefunction(handler)

    async def open(self):
        try:
            self._websocket = await websockets.connect(self._server_url)
            asyncio.ensure_future(self._listen(self._websocket))
        except Exception as exp:
            if hasattr(exp, "status_code") and exp.status_code == 403:
                raise PermissionError(
                    f"Permission denied for {self._server_url}, error: {exp}"
                )
            else:
                raise Exception(
                    f"Failed to connect to {self._server_url}, error: {exp}"
                )

    async def emit_message(self, data):
        assert self._handle_message is not None, "No handler for message"
        if not self._websocket or self._websocket.closed:
            await self.open()
        try:
            await self._websocket.send(data)
        except Exception:
            data = msgpack.unpackb(data)
            logger.exception(f"Failed to send data to {data['to']}")
            raise

    async def _listen(self, ws):
        try:
            while not ws.closed:
                data = await ws.recv()
                if self._is_async:
                    await self._handle_message(data)
                else:
                    self._handle_message(data)
        except websockets.exceptions.ConnectionClosedError:
            logger.warning("Connecting is broken, reopening a new connection.")
            await self.open()
        except websockets.exceptions.ConnectionClosedOK:
            pass

    async def disconnect(self):
        ws = self._websocket
        self._websocket = None
        if ws and not ws.closed:
            await ws.close(code=1000)


async def connect_to_server(config):
    """Connect to RPC via a websocket server."""
    client_id = config.get("client_id")
    if client_id is None:
        client_id = shortuuid.uuid()
    connection = WebsocketRPCConnection(
        config["server_url"],
        client_id,
        workspace=config.get("workspace"),
        token=config.get("token"),
    )
    await connection.open()
    rpc = RPC(
        connection,
        client_id=client_id,
        name=config.get("name"),
        root_target_id="workspace-manager",
        default_context={"connection_type": "websocket"},
        method_timeout=config.get("method_timeout"),
    )
    wm = await rpc.get_remote_service("workspace-manager:default")
    wm.rpc = rpc

    def export(api):
        return asyncio.ensure_future(rpc.register_service(api))

    wm.export = export
    return wm
