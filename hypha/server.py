"""Provide the server."""
import argparse
import logging
import sys
from os import environ as env
from pathlib import Path
import asyncio

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hypha import __version__
from hypha.core.auth import create_login_service
from hypha.core.store import RedisStore
from hypha.core.queue import create_queue_service
from hypha.http import HTTPProxy
from hypha.triton import TritonProxy
from hypha.utils import GZipMiddleware, GzipRoute, PatchedCORSMiddleware
from hypha.websocket import WebsocketServer
from contextlib import asynccontextmanager

try:
    # For pyodide, we need to patch http
    import pyodide
    import pyodide_http

    pyodide_http.patch_all()  # Patch all libraries
except ImportError:
    pass


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)

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
            executable_path=args.executable_path,
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

        ServerAppController(
            store,
            port=args.port,
            in_docker=args.in_docker,
            workspace_bucket=args.workspace_bucket,
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
        reconnection_token_life_time=float(
            env.get("RECONNECTION_TOKEN_LIFE_TIME", str(2 * 24 * 60 * 60))
        ),
    )

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


def create_application_from_env():
    """Create a hypha application using environment variables."""

    # Retrieve the arguments from environment variables
    parser = get_argparser(add_help=False)
    args = parser.parse_args([])
    for arg_name in vars(args):
        env_var = "HYPHA_" + arg_name.upper().replace("-", "_")
        if env_var in env:
            setattr(args, arg_name, env[env_var])

    return create_application(args)


def start_server(args):
    """Start the server."""
    import uvicorn

    app = create_application(args)
    uvicorn.run(app, host=args.host, port=int(args.port))


def get_argparser(add_help=True):
    """Return the argument parser."""
    parser = argparse.ArgumentParser(add_help=add_help)
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
    uvicorn.run(app, host=opt.host, port=int(opt.port))
