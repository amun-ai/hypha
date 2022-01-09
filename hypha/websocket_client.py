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

    def __init__(self, url, client_id, workspace=None, token=None):
        """Set up instance."""
        self._websocket = None
        self._handle_message = None
        assert url and client_id
        url = url + f"/ws?client_id={client_id}"
        if workspace is not None:
            url += f"&workspace={workspace}"
        if token:
            url += f"&token={token}"
        self._url = url

    def on_message(self, handler):
        self._handle_message = handler
        self._is_async = inspect.iscoroutinefunction(handler)

    async def open(self):
        self._websocket = await websockets.connect(self._url)
        asyncio.ensure_future(self._listen(self._websocket))

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


async def connect_to_server(
    url: str, client_id: str=None, workspace: str=None, token: str=None, method_timeout=10
):
    """Connect to RPC via a websocket server."""
    if client_id is None:
        client_id = shortuuid.uuid()
    connection = WebsocketRPCConnection(url, client_id, workspace=workspace, token=token)
    rpc = RPC(
        connection,
        client_id=client_id,
        root_target_id="workspace-manager",
        default_context={"connection_type": "websocket"},
        method_timeout=method_timeout,
    )
    return rpc
