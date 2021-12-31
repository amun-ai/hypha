"""Provide an s3 interface."""
import logging
import sys
from contextvars import copy_context

import shortuuid
import socketio
from fastapi import HTTPException
from imjoy_rpc.core_connection import BasicConnection

from hypha.core.plugin import DynamicPlugin
from hypha.utils import dotdict

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("socketio")
logger.setLevel(logging.INFO)


class SocketIOServer:
    """Represent an SocketIO server."""

    # pylint: disable=too-many-statements

    def __init__(
        self, core_interface, socketio_path="/socket.io", allow_origins="*"
    ) -> None:
        """Set up the socketio server."""
        if allow_origins == ["*"]:
            allow_origins = "*"

        self.core_interface = core_interface
        sio = socketio.AsyncServer(
            async_mode="asgi", cors_allowed_origins=allow_origins
        )
        _app = socketio.ASGIApp(socketio_server=sio, socketio_path=socketio_path)
        app = core_interface._app
        app.mount("/", _app)

        # TODO: what is this for?
        app.sio = sio
        self.sio = sio

        event_bus = core_interface.event_bus

        @sio.event
        async def connect(sid, environ):
            """Handle event called when a socketio client is connected to the server."""
            # We don't do much until register_plugin is called
            # This allows us to use websocket transport directly
            # Without relying on the Authorization header
            logger.info("New session connected: %s", sid)

        @sio.event
        async def echo(sid, data):
            """Echo service for testing."""
            return data

        @sio.event
        async def register_plugin(sid, config):
            # Check if the plugin is already registered
            plugin = DynamicPlugin.get_plugin_by_session_id(sid)
            if plugin:
                if plugin.is_disconnected():
                    DynamicPlugin.remove_plugin(plugin)
                    logger.info("Removing disconnected plugin: %s", plugin.id)
                else:
                    core_interface.restore_plugin(plugin)
                    logger.info("Plugin has already been registered: %s", plugin.id)
                    return

            try:
                user_info = core_interface.get_user_info_from_token(config.get("token"))
            except HTTPException as exp:
                logger.warning("Failed to create user: %s", exp.detail)
                config = dotdict(config)
                config.detail = f"Failed to create user: {exp.detail}"
                DynamicPlugin.plugin_failed(config)
                return {"success": False, "detail": config.detail}
            except Exception as exp:  # pylint: disable=broad-except
                logger.warning("Failed to create user: %s", exp)
                config = dotdict(config)
                config.detail = f"Failed to create user: {exp}"
                DynamicPlugin.plugin_failed(config)
                return {"success": False, "detail": config.detail}

            ws = config.get("workspace") or user_info.id

            config["name"] = config.get("name") or shortuuid.uuid()
            workspace = await core_interface.get_workspace(ws)
            if workspace is None:
                if ws == user_info.id:
                    workspace = core_interface.create_user_workspace(
                        user_info, read_only=False  # user_info.is_anonymous
                    )
                else:
                    logger.error("Workspace %s does not exist", ws)
                    config = dotdict(config)
                    config.detail = f"Workspace {ws} does not exist"
                    DynamicPlugin.plugin_failed(config)
                    return {"success": False, "detail": config.detail}

            config["workspace"] = workspace.name
            config["public_base_url"] = core_interface.public_base_url

            logger.info(
                "Registering plugin (uid: %s, workspace: %s)",
                user_info.id,
                workspace.name,
            )

            if user_info.id != ws and not core_interface.check_permission(
                workspace, user_info
            ):
                logger.warning(
                    "Failed to register plugin (uid: %s, workspace: %s)"
                    " due to permission error",
                    user_info.id,
                    workspace.name,
                )
                config = dotdict(config)
                config.detail = f"Permission denied for workspace: {ws}"
                DynamicPlugin.plugin_failed(config)
                return {
                    "success": False,
                    "detail": config.detail,
                }

            config["id"] = config.get("id") or "plugin-" + sid
            plugin_id = config["id"]
            sio.enter_room(sid, plugin_id)

            plugin = DynamicPlugin.get_plugin_by_id(plugin_id)
            # Note: for restarting plugins, we need to mark it as unready first
            if plugin and plugin.get_status() != "restarting":
                logger.error("Duplicated plugin id: %s", plugin.id)
                config = dotdict(config)
                config.detail = f"Duplicated plugin id: {plugin.id}"
                DynamicPlugin.plugin_failed(config)
                return {
                    "success": False,
                    "detail": config.detail,
                }

            async def send(data):
                await sio.emit(
                    "plugin_message",
                    data,
                    room=plugin_id,
                )

            connection = BasicConnection(send)
            core_interface.add_user(user_info)
            core_interface.current_user.set(user_info)
            plugin = DynamicPlugin(
                config,
                await core_interface.get_workspace_interface(workspace.name),
                core_interface.get_codecs(),
                connection,
                workspace,
                user_info,
                event_bus,
                sid,
                core_interface.public_base_url,
            )
            user_info.add_plugin(plugin)
            workspace.add_plugin(plugin)
            event_bus.emit(
                "plugin_registered",
                plugin,
            )
            logger.info(
                "New plugin registered successfully (%s)",
                plugin_id,
            )
            return {"success": True, "plugin_id": plugin_id}

        @sio.event
        async def plugin_message(sid, data):
            plugin = DynamicPlugin.get_plugin_by_session_id(sid)
            # TODO: Do we need to check the permission of the user?
            if not plugin:
                return {"success": False, "detail": f"Plugin session not found: {sid}"}
            workspace = plugin.workspace
            core_interface.current_user.set(plugin.user_info)
            core_interface.current_plugin.set(plugin)
            core_interface.current_workspace.set(workspace)
            ctx = copy_context()
            ctx.run(plugin.connection.handle_message, data)
            return {"success": True}

        @sio.event
        async def disconnect(sid):
            """Event handler called when the client is disconnected."""
            core_interface.remove_plugin_temp(sid)
            logger.info("Session disconnected: %s", sid)

        event_bus.emit("socketio_ready", None)

    async def is_alive(self):
        """Check if the server is alive."""
        try:
            await self.sio.emit("liveness")
        except Exception:  # pylint: disable=broad-except
            return False
        else:
            return True
