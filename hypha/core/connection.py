"""Provide the connection."""
import asyncio
import logging
import sys
import time

from imjoy_rpc.utils import MessageEmitter, dotdict

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("core-connection")
logger.setLevel(logging.WARNING)

all_connections = {}


class BasicConnection(MessageEmitter):
    """Represent a base connection."""

    def __init__(self, socketio, plugin_id, session_id):
        """Set up instance."""
        super().__init__(logger)
        self.plugin_config = dotdict()
        self._socketio = socketio
        self._plugin_id = plugin_id
        self._session_id = session_id
        self._access_token = None
        self._expires_in = None
        self._plugin_origin = "*"
        self._refresh_token = None
        self.peer_id = None
        self.on("initialized", self._initialized)

    def _initialized(self, data):
        self.plugin_config = data["config"]
        # peer_id can only be set for once
        self.peer_id = data["peer_id"]
        self._plugin_origin = data.get("origin", "*")
        all_connections[self.peer_id] = self
        if self._plugin_origin != "*":
            logger.info(
                "Connection to the imjoy-rpc peer $%s is limited to origin %s.",
                self.peer_id,
                self._plugin_origin,
            )

        if not self.peer_id:
            raise Exception("Please provide a peer_id for the connection.")

        if self.plugin_config.get("auth"):
            if self._plugin_origin == "*":
                logger.error(
                    "Refuse to transmit the token without an explicit origin, "
                    "there is a security risk that you may leak the credential "
                    "to website from other origin. "
                    "Please specify the `origin` explicitly."
                )
                self._access_token = None
                self._refresh_token = None

            if self.plugin_config["auth"]["type"] != "jwt":
                logger.error(
                    "Unsupported authentication type: %s", self.plugin_config.auth.type
                )
            else:
                self._expires_in = self.plugin_config["auth"]["expires_in"]
                self._access_token = self.plugin_config["auth"]["access_token"]
                self._refresh_token = self.plugin_config["auth"]["refresh_token"]

    async def _send(self, data):
        await self._socketio.emit(
            "plugin_message",
            data,
            room=self._plugin_id,
        )

    def get_session_id(self):
        """Get session id."""
        return self._session_id

    def handle_message(self, data):
        """Handle a message."""
        target_id = data.get("target_id")
        if target_id and self.peer_id and target_id != self.peer_id:
            conn = all_connections[target_id]
            if conn:
                conn._fire(data["type"], data)  # pylint: disable=protected-access
            else:
                logger.warning(
                    "Connection with target_id %s not found, discarding data: %s",
                    target_id,
                    data,
                )
        else:
            self._fire(data["type"], data)

    def connect(self):
        """Connect."""
        self._fire("connected")

    async def execute(self, code):
        """Execute."""
        # pylint: disable=no-self-use
        raise PermissionError

    def emit(self, msg):
        """Send a message to the plugin site."""
        if self._access_token:
            if time.time() >= self._expires_in:
                # TODO: refresh access token
                raise Exception("Refresh token is not implemented.")

            msg["access_token"] = self._access_token
        msg["peer_id"] = msg.get("peer_id") or self.peer_id
        asyncio.ensure_future(self._send(msg))

    def disconnect(self):
        """Disconnect the plugin."""
        self.emit({"type": "disconnect"})
        if self.peer_id and self.peer_id in all_connections:
            del all_connections[self.peer_id]
