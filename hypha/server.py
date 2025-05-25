"""Provide the server."""

import argparse
import logging
import sys
import os
from os import environ as env
from pathlib import Path
import asyncio
import subprocess

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hypha import __version__
from hypha.core.auth import create_login_service
from hypha.core.store import RedisStore
from hypha.queue import create_queue_service
from hypha.http import HTTPProxy
from hypha.triton import TritonProxy
from hypha.utils import GZipMiddleware, GzipRoute, PatchedCORSMiddleware
from hypha.websocket import WebsocketServer
from hypha.minio import start_minio_server
from contextlib import asynccontextmanager

# Global variable to track the Minio server process
minio_proc = None

try:
    # For pyodide, we need to patch http
    import pyodide
    import pyodide_http

    pyodide_http.patch_all()  # Patch all libraries
except ImportError:
    pass


LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("server")
logger.setLevel(LOGLEVEL)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

ALLOW_HEADERS = [
    "Content-Type",
    "Authorization",
    "Access-Control-Allow-Headers",
    "Origin",
    "Accept",
    "X-Requested-With",
    "Access-Control-Request-Method",
    "Access-Control-Request-Headers",
    "Range",
    # for triton inference server
    "Inference-Header-Content-Length",
    "Accept-Encoding",
    "Content-Encoding",
]
ALLOW_METHODS = ["*"]
EXPOSE_HEADERS = [
    "Inference-Header-Content-Length",
    "Accept-Encoding",
    # "Content-Encoding",
    "Range",
    "Origin",
    "Content-Type",
]


def start_builtin_services(
    app: FastAPI,
    store: RedisStore,
    args: argparse.Namespace,
) -> None:
    """Set up the builtin services."""
    # pylint: disable=too-many-arguments,too-many-locals

    if args.triton_servers:
        TritonProxy(
            store,
            triton_servers=args.triton_servers.split(","),
            allow_origins=args.allow_origins,
        )

    store.register_public_service(create_queue_service(store))
    store.register_public_service(create_login_service(store))

    if args.enable_s3:
        # pylint: disable=import-outside-toplevel
        from hypha.s3 import S3Controller
        from hypha.artifact import ArtifactController

        s3_controller = S3Controller(
            store,
            endpoint_url=args.endpoint_url,
            access_key_id=args.access_key_id,
            secret_access_key=args.secret_access_key,
            endpoint_url_public=args.endpoint_url_public,
            region_name=args.region_name,
            s3_admin_type=args.s3_admin_type,
            enable_s3_proxy=args.enable_s3_proxy,
            workspace_bucket=args.workspace_bucket,
            executable_path=args.executable_path or args.cache_dir,
        )
        artifact_manager = ArtifactController(
            store,
            s3_controller=s3_controller,
            workspace_bucket=args.workspace_bucket,
        )

    if args.enable_server_apps:
        assert args.enable_s3, "Server apps require S3 to be enabled"
        # pylint: disable=import-outside-toplevel
        from hypha.apps import ServerAppController
        from hypha.runner import BrowserAppRunner

        BrowserAppRunner(store, in_docker=args.in_docker)
        ServerAppController(
            store,
            port=args.port,
            in_docker=args.in_docker,
            artifact_manager=artifact_manager,
        )

    HTTPProxy(
        store,
        app,
        endpoint_url=args.endpoint_url,
        access_key_id=args.access_key_id,
        secret_access_key=args.secret_access_key,
        region_name=args.region_name,
        workspace_bucket=args.workspace_bucket,
        base_path=args.base_path,
    )


def mount_static_files(app, new_route, directory, name="static"):
    # Get top level route paths
    top_level_route_paths = [
        route.path.split("/")[1] for route in app.routes if route.path.count("/") == 1
    ]

    # Make sure the new route starts with a "/"
    assert new_route.startswith("/"), "The new route must start with a '/'."

    assert "/" not in new_route.strip("/"), "Only top-level routes are supported."

    # Check if new route collides with existing top-level paths
    if new_route.strip("/") in top_level_route_paths:
        raise ValueError(f"The route '{new_route}' collides with an existing route.")

    # Check if the directory exists
    if not Path(directory).exists():
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    # If no collision, mount static files
    app.mount(new_route, StaticFiles(directory=directory), name=name)

    if Path(f"{directory}/index.html").exists():
        # Add a new route that serves index.html directly
        @app.get(new_route)
        async def root():
            return FileResponse(f"{directory}/index.html")


def norm_url(base_path, url):
    return base_path.rstrip("/") + url


def create_application(args):
    """Create a hypha application."""
    global minio_proc

    if args.from_env:
        logger.info("Loading arguments from environment variables")
        _args = get_args_from_env()
        # copy the _args to args
        for key, value in _args.__dict__.items():
            setattr(args, key, value)

    # Handle Minio server if requested
    minio_proc = None
    if args.start_minio_server:
        # Check if S3 settings are already configured
        if args.endpoint_url or args.access_key_id or args.secret_access_key:
            raise ValueError(
                "Cannot use --start-minio-server with S3 settings (--endpoint-url, --access-key-id, --secret-access-key)."
                " Please use either built-in Minio server or external S3 configuration."
            )

        # Start Minio server
        minio_proc, server_url, workdir = start_minio_server(
            executable_path=args.cache_dir or args.executable_path,
            workdir=args.minio_workdir,
            port=args.minio_port,
            root_user=args.minio_root_user,
            root_password=args.minio_root_password,
            minio_version=args.minio_version,
            mc_version=args.mc_version,
            file_system_mode=args.minio_file_system_mode,
        )

        if not minio_proc:
            raise RuntimeError("Failed to start Minio server")

        # Set S3 settings automatically
        args.endpoint_url = server_url
        args.endpoint_url_public = server_url
        args.access_key_id = args.minio_root_user
        args.secret_access_key = args.minio_root_password
        args.enable_s3 = True
        args.s3_admin_type = "minio"

        logger.info(f"Started built-in Minio server at {server_url}")

    if args.allow_origins and isinstance(args.allow_origins, str):
        args.allow_origins = args.allow_origins.split(",")
    else:
        args.allow_origins = env.get("ALLOW_ORIGINS", "*").split(",")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Here we can register all the startup functions
        args.startup_functions = args.startup_functions or []
        await store.init(args.reset_redis, startup_functions=args.startup_functions)
        yield
        # Emit the shutdown event
        await store.get_event_bus().emit_local("shutdown")
        await store.teardown()
        await asyncio.sleep(0.1)
        await websocket_server.stop()

        # Terminate Minio if we started it
        if minio_proc:
            logger.info("Shutting down built-in Minio server")
            minio_proc.terminate()
            try:
                # Wait for the process to terminate with a timeout
                minio_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                logger.warning(
                    "Minio server did not terminate gracefully, forcing termination"
                )
                minio_proc.kill()

    application = FastAPI(
        title="Hypha",
        lifespan=lifespan,
        docs_url="/api-docs",
        redoc_url="/api-redoc",
        description=(
            "A serverless application framework for \
                large-scale data management and AI model serving"
        ),
        version=__version__,
    )
    application.router.route_class = GzipRoute

    application.add_middleware(GZipMiddleware, minimum_size=1000)
    application.add_middleware(
        PatchedCORSMiddleware,
        allow_origins=args.allow_origins,
        allow_methods=ALLOW_METHODS,
        allow_headers=ALLOW_HEADERS,
        expose_headers=EXPOSE_HEADERS,
        allow_credentials=True,
    )

    local_base_url = f"http://127.0.0.1:{args.port}/{args.base_path.strip('/')}".strip(
        "/"
    )
    if args.public_base_url:
        public_base_url = args.public_base_url.strip("/")
    else:
        public_base_url = local_base_url

    store = RedisStore(
        application,
        server_id=args.server_id or env.get("HYPHA_SERVER_ID"),
        public_base_url=public_base_url,
        local_base_url=local_base_url,
        redis_uri=args.redis_uri,
        database_uri=args.database_uri,
        ollama_host=args.ollama_host,
        cache_dir=args.cache_dir,
        openai_config={
            "base_url": args.openai_base_url,
            "api_key": args.openai_api_key,
        },
        enable_service_search=args.enable_service_search,
        reconnection_token_life_time=float(
            env.get("RECONNECTION_TOKEN_LIFE_TIME", str(2 * 24 * 60 * 60))
        ),
        activity_check_interval=float(env.get("ACTIVITY_CHECK_INTERVAL", str(10))),
    )
    application.state.store = store

    websocket_server = WebsocketServer(store, path=norm_url(args.base_path, "/ws"))

    static_folder = str(Path(__file__).parent / "static_files")
    mount_static_files(application, "/static", directory=static_folder, name="static")

    if args.static_mounts:
        for index, mount in enumerate(args.static_mounts):
            mountpath, localdir = mount.split(":")
            mount_static_files(
                application, mountpath, localdir, name=f"static-mount-{index}"
            )

    start_builtin_services(application, store, args)

    if args.host in ("127.0.0.1", "localhost"):
        logger.info(
            "***Note: If you want to enable access from another host, "
            "please start with `--host=0.0.0.0`.***"
        )
    return application


def get_args_from_env():
    """Create a hypha application using environment variables."""
    # Retrieve the arguments from environment variables
    parser = get_argparser(add_help=False)
    args = parser.parse_args([])

    # Get the argument types from the parser
    arg_types = {
        action.dest: action.type
        for action in parser._actions
        if action.type is not None
    }
    arg_bools = {
        action.dest
        for action in parser._actions
        if isinstance(action, argparse._StoreTrueAction)
    }
    arg_lists = {action.dest for action in parser._actions if action.nargs == "*"}

    for arg_name in vars(args):
        env_var = "HYPHA_" + arg_name.upper().replace("-", "_")
        if env_var in env:
            value = env[env_var]

            # Handle boolean flags
            if arg_name in arg_bools:
                value = value.lower() in ("true", "1", "yes", "y", "on")
            # Handle other types using the parser's type information
            elif arg_name in arg_types:
                try:
                    # Special handling for lists
                    if arg_types[arg_name] == str and isinstance(value, str):
                        if arg_name in arg_lists:
                            value = value.split()
                    else:
                        value = arg_types[arg_name](value)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to convert environment variable {env_var}={value} "
                        f"to type {arg_types[arg_name]}: {str(e)}"
                    )
                    continue

            setattr(args, arg_name, value)

    return args


def get_argparser(add_help=True):
    """Return the argument parser."""
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument(
        "--from-env",
        action="store_true",
        help="load arguments from environment variables, the environment variables should be in the format of HYPHA_<ARG_NAME>_<ARG_NAME_UPPER>",
    )
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
        "--allow-origins",
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
        "--redis-uri",
        type=str,
        default=None,
        help="the URI (a URL or database file path) for the redis database",
    )
    parser.add_argument(
        "--reset-redis",
        action="store_true",
        help="reset and clear all the data in the redis database",
    )
    parser.add_argument(
        "--public-base-url",
        type=str,
        default=None,
        help="the public base URL for accessing the server",
    )
    parser.add_argument(
        "--triton-servers",
        type=str,
        default=None,
        help="A list of comma separated Triton servers to proxy",
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
        "--s3-admin-type",
        type=str,
        default="generic",
        help="set the S3 admin interface type, depending on s3 server implementation (e.g. minio, generic)",
    )
    parser.add_argument(
        "--in-docker",
        action="store_true",
        help="Indicate whether running in docker (e.g. "
        "server apps will run without sandboxing)",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="set endpoint URL for S3",
    )
    parser.add_argument(
        "--endpoint-url-public",
        type=str,
        default=None,
        help="set public endpoint URL for S3"
        "(if different from the local endpoint one)",
    )
    parser.add_argument(
        "--region-name",
        type=str,
        default="EU",
        help="set region name for S3",
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
        "--ollama-host",
        type=str,
        default=None,
        help="set host for the ollama server",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="set OpenAI API type",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="set OpenAI API key",
    )
    parser.add_argument(
        "--database-uri",
        type=str,
        default=None,
        help="set database URI for the artifact manager",
    )
    parser.add_argument(
        "--migrate-database",
        type=str,
        default=None,
        help="migrate the database using alembic to the specified revision, e.g. head",
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
    parser.add_argument(
        "--static-mounts",
        type=str,
        nargs="*",
        help="extra directories to serve static files in the form <mountpath>:<localdir>, (e.g. /mystatic:./static/)",
    )
    parser.add_argument(
        "--startup-functions",
        type=str,
        nargs="*",
        help="specifies one or more startup functions. Each URI should be in the format '<python module or script>:<entrypoint function name>'. These functions are executed at server startup to perform initialization tasks such as loading services, configuring the server, or launching additional processes.",
    )
    parser.add_argument(
        "--server-id",
        type=str,
        default=None,
        help="the server ID of this instance, used to distinguish between multiple instances of hypha server in a distributed environment",
    )
    parser.add_argument(
        "--enable-s3-proxy",
        action="store_true",
        help="enable S3 proxy for serving pre-signed URLs",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="set the cache directory for the server",
    )
    parser.add_argument(
        "--enable-service-search",
        action="store_true",
        help="enable semantic service search via vector database",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="start an interactive shell with the hypha store",
    )
    parser.add_argument(
        "--enable-server",
        action="store_true",
        help="enable server in interactive mode",
    )
    parser.add_argument(
        "--start-minio-server",
        action="store_true",
        help="start a built-in Minio server for S3 storage",
    )
    parser.add_argument(
        "--minio-workdir",
        type=str,
        default=None,
        help="working directory for the built-in Minio server data",
    )
    parser.add_argument(
        "--minio-port",
        type=int,
        default=9000,
        help="port for the built-in Minio server",
    )
    parser.add_argument(
        "--minio-root-user",
        type=str,
        default="minioadmin",
        help="root user for the built-in Minio server",
    )
    parser.add_argument(
        "--minio-root-password",
        type=str,
        default="minioadmin",
        help="root password for the built-in Minio server",
    )
    parser.add_argument(
        "--minio-version",
        type=str,
        default=None,
        help="specify the Minio server version to use",
    )
    parser.add_argument(
        "--mc-version",
        type=str,
        default=None,
        help="specify the Minio client (mc) version to use",
    )
    parser.add_argument(
        "--minio-file-system-mode",
        action="store_true",
        help="enable file system mode for Minio, which uses specific compatible versions",
    )
    return parser


if __name__ == "__main__":
    import uvicorn

    arg_parser = get_argparser()

    opt = arg_parser.parse_args()

    # Apply database migrations
    if opt.migrate_database is not None:
        from alembic.config import Config
        from alembic import command

        alembic_cfg = Config()
        migration_dir = Path(__file__).parent / "migrations"
        alembic_cfg.set_main_option("script_location", str(migration_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", opt.database_uri)
        command.upgrade(alembic_cfg, "head")

    app = create_application(opt)
    if opt.interactive:
        from hypha.interactive import start_interactive_shell

        asyncio.run(start_interactive_shell(app, opt))
    else:
        uvicorn.run(app, host=opt.host, port=int(opt.port))

else:
    # Create the app instance when imported by uvicorn
    import sys

    # Parse uvicorn command line arguments to get host and port
    args = sys.argv
    for i, arg in enumerate(args):
        if arg == "--host" and i + 1 < len(args):
            if "HYPHA_HOST" in env and env["HYPHA_HOST"] != args[i + 1]:
                raise ValueError(
                    f"HYPHA_HOST ({env['HYPHA_HOST']}) does not match --host argument ({args[i + 1]})"
                )
            env["HYPHA_HOST"] = args[i + 1]
        elif arg.startswith("--host="):
            host = arg.split("=")[1]
            if "HYPHA_HOST" in env and env["HYPHA_HOST"] != host:
                raise ValueError(
                    f"HYPHA_HOST ({env['HYPHA_HOST']}) does not match --host argument ({host})"
                )
            env["HYPHA_HOST"] = host
        elif arg == "--port" and i + 1 < len(args):
            if "HYPHA_PORT" in env and env["HYPHA_PORT"] != args[i + 1]:
                raise ValueError(
                    f"HYPHA_PORT ({env['HYPHA_PORT']}) does not match --port argument ({args[i + 1]})"
                )
            env["HYPHA_PORT"] = args[i + 1]
        elif arg.startswith("--port="):
            port = arg.split("=")[1]
            if "HYPHA_PORT" in env and env["HYPHA_PORT"] != port:
                raise ValueError(
                    f"HYPHA_PORT ({env['HYPHA_PORT']}) does not match --port argument ({port})"
                )
            env["HYPHA_PORT"] = port

    arg_parser = get_argparser(add_help=False)
    opt = arg_parser.parse_args(["--from-env"])
    app = create_application(opt)
    __all__ = ["app"]
