import asyncio
import inspect
import logging
import sys

import msgpack
import websockets

from hypha.core.rpc import RPC

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket")
logger.setLevel(logging.INFO)


class WebsocketRPCConnection:
    """Represent a websocket connection."""

    def __init__(self, url, workspace, client_id, token):
        """Set up instance."""
        self._websocket = None
        self._handle_message = None
        self._url = (
            url + f"/ws?workspace={workspace}&client_id={client_id}&token={token}"
        )

    def on_message(self, handler):
        self._handle_message = handler
        self._is_async = inspect.iscoroutinefunction(handler)

    async def emit_message(self, data):
        assert self._handle_message is not None, "No handler for message"
        if not self._websocket or self._websocket.closed:
            self._websocket = await websockets.connect(self._url)
            asyncio.ensure_future(self._listen(self._websocket))
        try:
            if "to" not in data:
                raise ValueError("`to` is required")
            target_id = data["to"].encode()
            encoded = (
                len(target_id).to_bytes(1, "big") + target_id + msgpack.packb(data)
            )
            await self._websocket.send(encoded)
        except Exception:
            logger.exception(f"Failed to send data to {data.get('to')}")
            raise

    async def _listen(self, ws):
        try:
            while not ws.closed:
                data = await ws.recv()
                data = msgpack.unpackb(data)
                if self._is_async:
                    await self._handle_message(data)
                else:
                    self._handle_message(data)
        except websockets.exceptions.ConnectionClosedOK:
            pass

    async def disconnect(self):
        ws = self._websocket
        self._websocket = None
        if ws and not ws.closed:
            await ws.close(code=1000)


async def connect_to_server(
    url: str, workspace: str, client_id: str, token: str, method_timeout=10
):
    """Connect to RPC via a websocket server."""
    connection = WebsocketRPCConnection(url, workspace, client_id, token)
    rpc = RPC(
        connection,
        client_id=client_id,
        root_target_id="workspace-manager",
        default_context={"connection_type": "websocket"},
        method_timeout=method_timeout,
    )
    return rpc
