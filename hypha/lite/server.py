# hypha/lite/server.py
"""Provide the minimal Hypha server."""

import argparse
import logging
import sys
import os
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles # Keep for now, maybe remove later

from hypha import __version__
from hypha.utils import GZipMiddleware, GzipRoute, PatchedCORSMiddleware
from hypha.websocket import WebsocketServer
from hypha.lite.store import RedisStore

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("lite-server")
logger.setLevel(LOGLEVEL)

# Simplified CORS settings
ALLOW_HEADERS = [
    "Content-Type",
    "Authorization",
    "Access-Control-Allow-Headers",
    "Origin",
    "Accept",
    "X-Requested-With",
    "Access-Control-Request-Method",
    "Access-Control-Request-Headers",
]
ALLOW_METHODS = ["*"]
EXPOSE_HEADERS = ["Origin", "Content-Type"]


def norm_url(base_path, url):
    """Normalize URL by joining base path and URL."""
    # Ensure no leading/trailing slashes cause issues
    base = base_path.rstrip('/')
    url_part = url.lstrip('/')
    # Handle case where base_path is just "/"
    if base == "" and base_path.startswith("/"):
        base = ""
    return f"{base}/{url_part}"


def create_application(args):
    """Create the minimal Hypha application."""

    if args.allow_origins and isinstance(args.allow_origins, str):
        args.allow_origins = args.allow_origins.split(",")
    else:
        args.allow_origins = ["*"]

    # Placeholder for the store instance
    store_instance: RedisStore | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal store_instance
        logger.info("Lite server starting up...")
        local_base_url = f"http://{args.host}:{args.port}{args.base_path}".rstrip("/")
        public_base_url = args.public_base_url.rstrip("/") if args.public_base_url else local_base_url

        store_instance = RedisStore(
            app=app, # Pass the FastAPI app instance to the store
            public_base_url=public_base_url,
            local_base_url=local_base_url,
            redis_uri=args.redis_uri,
        )
        app.state.store = store_instance

        # WebsocketServer registers its own route using the app instance via the store
        ws_path = norm_url(args.base_path, "/ws")
        logger.info(f"Initializing WebSocket server to handle path: {ws_path}")
        websocket_server = WebsocketServer(store_instance, path=ws_path) # Pass path back
        # store_instance.set_websocket_server(websocket_server) # WebsocketServer init does this
        # app.state.websocket_server = websocket_server # Not needed for routing

        await store_instance.init(args.reset_redis)
        logger.info("Lite store initialized.")

        yield

        logger.info("Lite server shutting down...")
        if store_instance:
            await store_instance.get_event_bus().emit_local("shutdown")
            await store_instance.teardown()
        # Websocket server shutdown might be handled internally or in store teardown
        # if websocket_server:
        #     await websocket_server.stop()
        logger.info("Lite server shutdown complete.")


    application = FastAPI(
        title="Hypha Lite",
        lifespan=lifespan,
        version=__version__,
        docs_url=None,
        redoc_url=None,
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

    # Minimal root endpoint
    # Use norm_url to ensure consistent path handling
    root_path = args.base_path.rstrip('/') or '/'
    @application.get(root_path)
    async def root():
        return {"message": "Hypha Lite Server is running."}

    logger.info(f"Hypha Lite server configured for: http://{args.host}:{args.port}{args.base_path}")
    if args.public_base_url:
        logger.info(f"Public base URL set to: {args.public_base_url}")

    return application


def get_argparser():
    """Return the minimal argument parser."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the Hypha Lite server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9527,
        help="Port for the Hypha Lite server.",
    )
    parser.add_argument(
        "--allow-origins",
        type=str,
        default="*",
        help="Allowed origins (comma-separated).",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="/",
        help="Base path for the server API.",
    )
    parser.add_argument(
        "--redis-uri",
        type=str,
        default=None, # Use fakeredis by default if None
        help="URI for the Redis database (e.g., redis://localhost:6379/0). If not provided, uses fakeredis.",
    )
    parser.add_argument(
        "--reset-redis",
        action="store_true",
        help="Reset and clear all data in the Redis database on startup.",
    )
    parser.add_argument(
        "--public-base-url",
        type=str,
        default=None,
        help="Publicly accessible base URL for the server (if different from host/port).",
    )
    # Removed many other arguments
    return parser

if __name__ == "__main__":
    import uvicorn

    arg_parser = get_argparser()
    opt = arg_parser.parse_args()

    # Basic host warning
    if opt.host in ("127.0.0.1", "localhost") and not opt.public_base_url:
        logger.info(
            "***Note: Server is only accessible locally. "
            "Use --host=0.0.0.0 or set --public-base-url to allow external access.***"
        )

    app = create_application(opt)
    uvicorn.run(app, host=opt.host, port=int(opt.port)) 