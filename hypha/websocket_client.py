import asyncio
import logging
import sys

import msgpack
import websockets
from imjoy_rpc.core_connection import BasicConnection

from hypha.core.rpc import RPC

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("websocket")
logger.setLevel(logging.INFO)


class WebsocketConnection(BasicConnection):
    """Represent a base connection."""

    def __init__(self, url, workspace, client_id, token):
        """Set up instance."""
        super().__init__(self._send_message)
        self._websocket = None
        self._url = (
            url + f"/ws?workspace={workspace}&client_id={client_id}&token={token}"
        )

    async def open(self):
        self._websocket = await websockets.connect(self._url)
        asyncio.ensure_future(self._listen(self._websocket))

    async def _send_message(self, data):
        if not self._websocket or self._websocket.closed:
            await self.open()
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
                self.handle_message(data)
        except RuntimeError:
            pass
        except websockets.exceptions.ConnectionClosedOK:
            pass

    def disconnect(self):
        ws = self._websocket
        self._websocket = None
        if ws and not ws.closed:
            asyncio.ensure_future(ws.close(code=1000))
            self._fire("disconnected", None)


async def connect_to_server(
    url: str, workspace: str, client_id: str, token: str, method_timeout=10
):
    """Connect to RPC via a websocket server."""
    connection = WebsocketConnection(url, workspace, client_id, token)
    rpc = RPC(
        connection,
        client_id=client_id,
        root_target_id="workspace-manager",
        default_context={"connection_type": "websocket"},
        method_timeout=method_timeout,
    )
    return rpc
