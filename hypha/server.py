"""Provide the server."""
import argparse
import logging
import sys
from os import environ as env
from pathlib import Path

import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseSettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from hypha import __version__ as VERSION
from hypha.asgi import ASGIGateway
from hypha.core.store import RedisStore
from hypha.http import HTTPProxy
from hypha.triton import TritonProxy
from hypha.utils import GZipMiddleware, GzipRoute, PatchedCORSMiddleware
from hypha.websocket import WebsocketServer


class Settings(BaseSettings):
    """Settings for the server."""
    host: str = "127.0.0.1"  # "host for the hypha server"
    port: int = 9527  # "port for the hypha server"
    allow_origins: str = "*"  # "origins for the hypha server",
    base_path: str = "/"  # "the base path for the server",
    redis_uri: str = "/tmp/redis.db"  # "the URI (a URL or database file path) for the redis database",
    reset_redis: bool = False  # "reset and clear all the data in the redis database",
    redis_port: int = 6383  # "the port for the redis database",
    public_base_url: str = None  # "the public base URL for accessing the server",
    triton_servers: str = None  # "A list of comma separated Triton servers to proxy",
    enable_server_apps: bool = False  # "enable server applications",
    enable_s3: bool = False  # "enable S3 object storage",
    in_docker: bool = False  # "Indicate whether running in docker (e.g. server apps will run without sandboxing)"
    endpoint_url: str = None  # "set endpoint URL for S3",
    endpoint_url_public: str = None  # "set public endpoint URL for S3 (if different from the local endpoint one)"
    access_key_id: str = None  # "set access key ID for S3",
    secret_access_key: str = None  # "set secret access key for S3",
    apps_dir: str = "hypha-apps"  # "temporary directory for storing installed apps",
    rdf_bucket: str = "hypha-rdfs"  # "set the bucket name for the RDF store",
    workspace_bucket: str = (
        "hypha-workspaces"  # "set the bucket name for the workspaces",
    )
    executable_path: str = (
        "bin"  # "temporary directory for storing executables (e.g. mc, minio)"
    )


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
    app.router.route_class = GzipRoute

    static_folder = str(Path(__file__).parent / "static_files")
    app.mount("/static", StaticFiles(directory=static_folder), name="static")

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        PatchedCORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=ALLOW_METHODS,
        allow_headers=ALLOW_HEADERS,
        expose_headers=EXPOSE_HEADERS,
        allow_credentials=True,
    )
    return app


def start_builtin_services(
    app: FastAPI,
    store: RedisStore,
    settings: Settings,
) -> None:
    """Set up the builtin services."""
    # pylint: disable=too-many-arguments,too-many-locals

    def norm_url(url):
        return settings.base_path.rstrip("/") + url

    WebsocketServer(store, path=norm_url("/ws"))

    HTTPProxy(store)
    if settings.triton_servers:
        TritonProxy(
            store,
            triton_servers=settings.triton_servers.split(","),
            allow_origins=settings.allow_origins,
        )
    ASGIGateway(
        store,
        allow_origins=settings.allow_origins,
        allow_methods=ALLOW_METHODS,
        allow_headers=ALLOW_HEADERS,
        expose_headers=EXPOSE_HEADERS,
    )

    @app.get(settings.base_path)
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
            "workspaces": [w.dict() for w in all_workspaces],
        }

    if settings.enable_s3:
        # pylint: disable=import-outside-toplevel
        from hypha.rdf import RDFController
        from hypha.s3 import S3Controller

        s3_controller = S3Controller(
            store,
            endpoint_url=settings.endpoint_url,
            access_key_id=settings.access_key_id,
            secret_access_key=settings.secret_access_key,
            endpoint_url_public=settings.endpoint_url_public,
            workspace_bucket=settings.workspace_bucket,
            executable_path=settings.executable_path,
        )

        RDFController(store, s3_controller=s3_controller, rdf_bucket=settings.rdf_bucket)

    if settings.enable_server_apps:
        # pylint: disable=import-outside-toplevel
        from hypha.apps import ServerAppController

        ServerAppController(
            store,
            port=settings.port,
            apps_dir=settings.apps_dir,
            in_docker=settings.in_docker,
            endpoint_url=settings.endpoint_url,
            access_key_id=settings.access_key_id,
            secret_access_key=settings.secret_access_key,
            workspace_bucket=settings.workspace_bucket,
        )

    @app.get(norm_url("/health/liveness"))
    async def liveness(req: Request) -> JSONResponse:
        if store.is_ready():
            return JSONResponse({"status": "OK"})

        return JSONResponse({"status": "DOWN"}, status_code=503)

    @app.on_event("startup")
    async def startup_event():
        await store.init(settings.reset_redis)

    @app.on_event("shutdown")
    def shutdown_event():
        store.get_event_bus().emit("shutdown", target="local")


settings = Settings()
if settings.allow_origins:
    settings.allow_origins = settings.allow_origins.split(",")
else:
    settings.allow_origins = env.get("ALLOW_ORIGINS", "*").split(",")
application = create_application(settings.allow_origins)
local_base_url = f"http://127.0.0.1:{settings.port}/{settings.base_path.strip('/')}".strip(
    "/"
)
if settings.public_base_url:
    public_base_url = settings.public_base_url.strip("/")
else:
    public_base_url = local_base_url
store = RedisStore(
    application,
    public_base_url=public_base_url,
    local_base_url=local_base_url,
    redis_uri=settings.redis_uri,
    redis_port=settings.redis_port,
)

start_builtin_services(application, store, settings)

if __name__ == "__main__":
    if settings.host in ("127.0.0.1", "localhost"):
        print(
            "***Note: If you want to enable access from another host, "
            "please start with `--host=0.0.0.0`.***"
        )
    uvicorn.run(application, host=settings.host, port=int(settings.port))