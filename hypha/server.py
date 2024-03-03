"""Provide the server."""
import argparse
import logging
import sys
from os import environ as env
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse

from hypha import __version__ as VERSION
from hypha.asgi import ASGIGateway
from hypha.core.store import RedisStore
from hypha.http import HTTPProxy
from hypha.sse import SSEProxy
from hypha.triton import TritonProxy
from hypha.utils import GZipMiddleware, GzipRoute, PatchedCORSMiddleware
from hypha.websocket import WebsocketServer

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

    def norm_url(url):
        return args.base_path.rstrip("/") + url

    WebsocketServer(store, path=norm_url("/ws"))

    HTTPProxy(store)
    SSEProxy(store)
    if args.triton_servers:
        TritonProxy(
            store,
            triton_servers=args.triton_servers.split(","),
            allow_origins=args.allow_origins,
        )
    ASGIGateway(
        store,
        allow_origins=args.allow_origins,
        allow_methods=ALLOW_METHODS,
        allow_headers=ALLOW_HEADERS,
        expose_headers=EXPOSE_HEADERS,
    )

    @app.get(args.base_path)
    async def home():
        return {
            "name": "Hypha",
            "version": VERSION,
        }

    @app.get(norm_url("/api/stats"))
    async def stats():
        users = await store.get_all_users()
        user_count = len(users)
        all_workspaces = await store.get_all_workspace()
        return {
            "user_count": user_count,
            "users": [u.id for u in users],
            "workspace_count": len(all_workspaces),
            "workspaces": [w.model_dump() for w in all_workspaces],
        }

    if args.enable_s3:
        # pylint: disable=import-outside-toplevel
        from hypha.rdf import RDFController
        from hypha.s3 import S3Controller

        s3_controller = S3Controller(
            store,
            endpoint_url=args.endpoint_url,
            access_key_id=args.access_key_id,
            secret_access_key=args.secret_access_key,
            endpoint_url_public=args.endpoint_url_public,
            workspace_bucket=args.workspace_bucket,
            executable_path=args.executable_path,
        )

        RDFController(store, s3_controller=s3_controller, rdf_bucket=args.rdf_bucket)

    if args.enable_server_apps:
        # pylint: disable=import-outside-toplevel
        from hypha.apps import ServerAppController

        ServerAppController(
            store,
            port=args.port,
            apps_dir=args.apps_dir,
            in_docker=args.in_docker,
            endpoint_url=args.endpoint_url,
            access_key_id=args.access_key_id,
            secret_access_key=args.secret_access_key,
            workspace_bucket=args.workspace_bucket,
        )

    @app.get(norm_url("/health/liveness"))
    async def liveness(req: Request) -> JSONResponse:
        if store.is_ready():
            return JSONResponse({"status": "OK"})

        return JSONResponse({"status": "DOWN"}, status_code=503)

    @app.on_event("startup")
    async def startup_event():
        # Here we can register all the startup functions
        args.startup_functions = args.startup_functions or []
        args.startup_functions.append("hypha.core.auth:register_login_service")
        await store.init(args.reset_redis, startup_functions=args.startup_functions)

    @app.on_event("shutdown")
    def shutdown_event():
        store.get_event_bus().emit("shutdown", target="local")


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


def create_application(args):
    """Create a hypha application."""
    if args.allow_origins and isinstance(args.allow_origins, str):
        args.allow_origins = args.allow_origins.split(",")
    else:
        args.allow_origins = env.get("ALLOW_ORIGINS", "*").split(",")

    application = FastAPI(
        title="Hypha",
        docs_url="/api-docs",
        redoc_url="/api-redoc",
        description=(
            "A serverless application framework for \
                large-scale data management and AI model serving"
        ),
        version=VERSION,
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
        public_base_url=public_base_url,
        local_base_url=local_base_url,
        redis_uri=args.redis_uri,
        disconnect_delay=float(env.get("DISCONNECT_DELAY", "30")),
    )

    start_builtin_services(application, store, args)

    static_folder = str(Path(__file__).parent / "static_files")
    mount_static_files(application, "/static", directory=static_folder, name="static")

    if args.static_mounts:
        for index, mount in enumerate(args.static_mounts):
            mountpath, localdir = mount.split(":")
            mount_static_files(
                application, mountpath, localdir, name=f"static-mount-{index}"
            )

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
        "--redis-port",
        type=int,
        default=6383,
        help="the port for the redis database",
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
        help="Specifies one or more startup functions. Each URI should be in the format '<python module or script>:<entrypoint function name>'. These functions are executed at server startup to perform initialization tasks such as loading services, configuring the server, or launching additional processes.",
    )

    return parser


if __name__ == "__main__":
    import uvicorn

    arg_parser = get_argparser()
    opt = arg_parser.parse_args()
    app = create_application(opt)

    uvicorn.run(app, host=opt.host, port=int(opt.port))
