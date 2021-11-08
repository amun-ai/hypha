"""Provide the server."""
import argparse
import logging
import sys
from contextvars import copy_context
from os import environ as env
from pathlib import Path
from typing import Union

import shortuuid
import socketio
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from hypha import __version__ as VERSION
from hypha.asgi import ASGIGateway
from hypha.core.connection import BasicConnection
from hypha.core.interface import CoreInterface
from hypha.core.plugin import DynamicPlugin
from hypha.http import HTTPProxy
from hypha.utils import dotdict

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


def initialize_socketio(sio, core_interface):
    """Initialize socketio."""
    # pylint: disable=too-many-statements, unused-variable
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
        workspace = core_interface.get_workspace(ws)
        if workspace is None:
            if ws == user_info.id:
                workspace = core_interface.create_user_workspace(
                    user_info, read_only=user_info.is_anonymous
                )
            else:
                logger.error("Workspace %s does not exist", ws)
                config = dotdict(config)
                config.detail = f"Workspace {ws} does not exist"
                DynamicPlugin.plugin_failed(config)
                return {"success": False, "detail": config.detail}

        config["workspace"] = workspace.name

        logger.info(
            "Registering plugin (uid: %s, workspace: %s)", user_info.id, workspace.name
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

        connection = BasicConnection(sio, plugin_id, sid)
        plugin = DynamicPlugin(
            config,
            core_interface.get_interface(),
            core_interface.get_codecs(),
            connection,
            workspace,
            user_info,
            event_bus,
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


def create_application(allow_origins) -> FastAPI:
    """Set up the server application."""
    # pylint: disable=unused-variable

    app = FastAPI(
        title="ImJoy Core Server",
        description=(
            "A server for managing imjoy plugins and \
                enabling remote procedure calls"
        ),
        version=VERSION,
    )

    static_folder = str(Path(__file__).parent / "static_files")
    app.mount("/static", StaticFiles(directory=static_folder), name="static")

    @app.middleware("http")
    async def add_cors_header(request: Request, call_next):
        headers = {}
        request_origin = request.headers.get("access-control-allow-origin")
        if request_origin and (
            allow_origins == "*"
            or allow_origins[0] == "*"
            or request_origin in allow_origins
        ):
            headers["access-control-allow-origin"] = request_origin
        headers["access-control-allow-credentials"] = "true"
        headers["access-control-allow-methods"] = ", ".join(["*"])
        headers["access-control-allow-headers"] = ", ".join(
            ["Content-Type", "Authorization"]
        )
        if (
            request.method == "OPTIONS"
            and "access-control-request-method" in request.headers
        ):
            return PlainTextResponse("OK", status_code=200, headers=headers)
        response = await call_next(request)
        # We need to first normalize the case of the headers
        # To avoid multiple values in the headers
        # See issue: https://github.com/encode/starlette/issues/1309
        # pylint: disable=protected-access
        items = response.headers._list
        # pylint: disable=protected-access
        response.headers._list = [
            (item[0].decode("latin-1").lower().encode("latin-1"), item[1])
            for item in items
        ]
        response.headers.update(headers)
        return response

    return app


def setup_socketio_server(
    app: FastAPI,
    core_interface: CoreInterface,
    port: int,
    base_path: str = "/",
    allow_origins: Union[str, list] = "*",
    enable_server_apps: bool = False,
    enable_s3: bool = False,
    endpoint_url: str = None,
    access_key_id: str = None,
    secret_access_key: str = None,
    workspace_bucket: str = "hypha-workspaces",
    rdf_bucket: str = "hypha-rdfs",
    apps_dir: str = "hypha-apps",
    executable_path: str = "",
    **kwargs,
) -> None:
    """Set up the socketio server."""
    # pylint: disable=too-many-arguments,too-many-locals

    def norm_url(url):
        return base_path.rstrip("/") + url

    HTTPProxy(core_interface)
    ASGIGateway(core_interface)

    @app.get(base_path)
    async def home():
        return {
            "name": "Hypha",
            "version": VERSION,
        }

    @app.get(norm_url("/api/stats"))
    async def stats():
        users = core_interface.get_all_users()
        client_count = len(users)
        return {
            "plugin_count": client_count,
            "workspace_count": len(core_interface.get_all_workspace()),
            "workspaces": [w.get_summary() for w in core_interface.get_all_workspace()],
            "users": [u.id for u in users],
        }

    if enable_server_apps:
        # pylint: disable=import-outside-toplevel
        from hypha.apps import ServerAppController

        ServerAppController(core_interface, port=port, apps_dir=apps_dir)

    if enable_s3:
        # pylint: disable=import-outside-toplevel
        from hypha.s3 import S3Controller
        from hypha.rdf import RDFController

        s3_controller = S3Controller(
            core_interface,
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            workspace_bucket=workspace_bucket,
            executable_path=executable_path,
        )

        RDFController(
            core_interface, s3_controller=s3_controller, rdf_bucket=rdf_bucket
        )

    @app.get(norm_url("/health/liveness"))
    async def liveness(req: Request) -> JSONResponse:
        try:
            await sio.emit("liveness")
        except Exception:  # pylint: disable=broad-except
            return JSONResponse({"status": "DOWN"}, status_code=503)
        return JSONResponse({"status": "OK"})

    if allow_origins == ["*"]:
        allow_origins = "*"
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=allow_origins)

    _app = socketio.ASGIApp(socketio_server=sio, socketio_path=norm_url("/socket.io"))

    app.mount("/", _app)
    app.sio = sio

    initialize_socketio(sio, core_interface)

    @app.on_event("startup")
    async def startup_event():
        core_interface.event_bus.emit("startup")

    @app.on_event("shutdown")
    def shutdown_event():
        core_interface.event_bus.emit("shutdown")

    return sio


def start_server(args):
    """Start the socketio server."""
    if args.allow_origin:
        args.allow_origin = args.allow_origin.split(",")
    else:
        args.allow_origin = env.get("ALLOW_ORIGINS", "*").split(",")
    application = create_application(args.allow_origin)
    core_interface = CoreInterface(application)
    setup_socketio_server(application, core_interface, **vars(args))
    if args.host in ("127.0.0.1", "localhost"):
        print(
            "***Note: If you want to enable access from another host, "
            "please start with `--host=0.0.0.0`.***"
        )
    uvicorn.run(application, host=args.host, port=int(args.port))


def get_argparser():
    """Return the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="host for the hypha server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9527,
        help="port for the hypha server",
    )
    parser.add_argument(
        "--allow-origin",
        type=str,
        default="*",
        help="origins for the hypha server",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="/",
        help="the base path for the server",
    )
    parser.add_argument(
        "--enable-server-apps",
        action="store_true",
        help="enable server applications",
    )
    parser.add_argument(
        "--enable-s3",
        action="store_true",
        help="enable S3 object storage",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="set endpoint URL for S3",
    )
    parser.add_argument(
        "--access-key-id",
        type=str,
        default=None,
        help="set AccessKeyID for S3",
    )
    parser.add_argument(
        "--secret-access-key",
        type=str,
        default=None,
        help="set SecretAccessKey for S3",
    )
    parser.add_argument(
        "--apps-dir",
        type=str,
        default="hypha-apps",
        help="temporary directory for storing installed apps",
    )
    parser.add_argument(
        "--rdf-bucket",
        type=str,
        default="hypha-rdfs",
        help="S3 bucket for storing RDF files",
    )
    parser.add_argument(
        "--workspace-bucket",
        type=str,
        default="hypha-workspaces",
        help="S3 bucket for storing workspaces",
    )
    parser.add_argument(
        "--executable-path",
        type=str,
        default="bin",
        help="temporary directory for storing executables (e.g. mc, minio)",
    )
    return parser


if __name__ == "__main__":
    arg_parser = get_argparser()
    opt = arg_parser.parse_args()
    start_server(opt)
