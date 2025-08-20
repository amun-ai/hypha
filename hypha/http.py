"""Provide the http proxy."""

import inspect
import json
import traceback
from typing import Any
import logging
import sys
from pathlib import Path
import httpx
import datetime
import msgpack
from fastapi import Depends, Request
from starlette.routing import Route, Match
from starlette.types import ASGIApp
from jinja2 import Environment, PackageLoader, select_autoescape
from prometheus_client import generate_latest
from fastapi.responses import (
    JSONResponse,
    Response,
    RedirectResponse,
    FileResponse,
    StreamingResponse,
)
import jose
import os

from starlette.datastructures import Headers, MutableHeaders

from hypha_rpc import RPC
from hypha import hypha_rpc_version
from hypha.core import UserPermission
from hypha.core.auth import AUTH0_DOMAIN, extract_token_from_scope
from hypha.core.store import RedisStore
from hypha.utils import safe_join, is_safe_path

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("http")
logger.setLevel(LOGLEVEL)


class MsgpackResponse(Response):
    """Response class for msgpack encoding."""

    media_type = "application/msgpack"

    def render(self, content: Any) -> bytes:
        """Render the response."""
        return msgpack.dumps(content)


def normalize(string):
    """Normalize numbers in URL query."""
    if string.isnumeric():
        return int(string)

    try:
        return float(string)
    except ValueError:
        return string


def serialize(obj):
    """Generate a serialized object for display."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, tuple, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: serialize(obj[k]) for k in obj}
    if isinstance(obj, list):
        return [serialize(k) for k in obj]
    if callable(obj):
        if hasattr(obj, "__schema__"):
            return {"type": "function", "function": obj.__schema__}
        else:
            return {"type": "function", "function": {"name": obj.__name__}}

    raise ValueError(f"unsupported data type: {type(obj)}")


def get_value(keys, service):
    """Get service function by a key string."""
    keys = keys.split(".")
    key = keys[0]
    value = service[key]
    if len(keys) > 1:
        for key in keys[1:]:
            value = value.get(key)
            if value is None:
                break
    return value


async def extracted_kwargs(
    request: Request,
    use_function_kwargs: bool = True,
):
    """Extract the kwargs from the request."""
    content_type = request.headers.get("content-type", "application/json")
    if request.method == "GET":
        kwargs = list(request.query_params.items())
        kwargs = {kwargs[k][0]: normalize(kwargs[k][1]) for k in range(len(kwargs))}
        if use_function_kwargs:
            kwargs = json.loads(kwargs.get("function_kwargs", "{}"))
        else:
            for key in list(kwargs.keys()):
                if key in ["workspace", "service_id", "function_key", "_mode"]:
                    del kwargs[key]
                else:
                    if isinstance(kwargs[key], str) and kwargs[key].startswith("{"):
                        kwargs[key] = json.loads(kwargs[key])

    elif request.method == "POST":
        if content_type == "application/msgpack":
            kwargs = msgpack.loads(await request.body())
        elif content_type == "application/json":
            kwargs = json.loads(await request.body())
        else:
            raise RuntimeError(
                "Invalid content-type (supported types: "
                "application/msgpack, application/json, text/plain)",
            )
        # Handle different JSON input types for HTTP service calls
        if isinstance(kwargs, list):
            # Convert positional arguments array to a dictionary for internal processing
            # e.g., ["arg1", "arg2"] becomes {"0": "arg1", "1": "arg2"}
            kwargs = {str(i): arg for i, arg in enumerate(kwargs)}
        elif not isinstance(kwargs, dict):
            # For single primitive values, treat as a single positional argument
            # e.g., "hello" becomes {"0": "hello"}
            if isinstance(kwargs, (str, int, float, bool)) or kwargs is None:
                kwargs = {"0": kwargs}
            else:
                raise RuntimeError(
                    f"Invalid JSON format for HTTP service call. "
                    f"Expected a dictionary (object), array, or primitive value, but got {type(kwargs).__name__}: {kwargs!r}. "
                    f"Examples: "
                    f"Array format: ['arg1', 'arg2'] "
                    f"Object format: {{'param1': 'value1', 'param2': 'value2'}} "
                    f"Single value: 'hello'"
                )
        
        if use_function_kwargs:
            kwargs = kwargs.get("function_kwargs", {})
        else:
            for key in ["workspace", "service_id", "function_key", "_mode"]:
                if key in list(kwargs.keys()):
                    del kwargs[key]
    else:
        raise RuntimeError(f"Invalid request method: {request.method}")

    return kwargs


async def extracted_call_info(request: Request):
    """Extract workspace, service_id, function_key from the request."""
    content_type = request.headers.get("content-type", "application/json")
    if request.method == "POST":
        if content_type == "application/msgpack":
            kwargs = msgpack.loads(await request.body())
        elif content_type == "application/json":
            kwargs = json.loads(await request.body())
        else:
            raise RuntimeError(
                "Invalid content-type (supported types: "
                "application/msgpack, application/json, text/plain)",
            )
        queries = list(request.query_params.items())
        queries = {kwargs[k][0]: normalize(queries[k][1]) for k in range(len(queries))}
        kwargs.update(queries)
        workspace = kwargs.get("workspace")
        service_id = kwargs.get("service_id")
        function_key = kwargs.get("function_key")
    else:
        raise RuntimeError(f"Invalid request method: {request.method}")
    return workspace, service_id, function_key


def detected_response_type(request: Request):
    """Detect the response type."""
    content_type = request.headers.get("content-type", "application/json")
    return content_type


class ASGIRoutingMiddleware:
    def __init__(self, app: ASGIApp, base_path=None, store=None):
        self.app = app
        assert base_path is not None, "Base path is required"
        assert store is not None, "Store is required"
        route = base_path.rstrip("/") + "/{workspace}/apps/{service_id}/{path:path}"
        # Define a route using Starlette's routing system for pattern matching
        self.route = Route(route, endpoint=None)
        self.store = store

    async def _get_token_from_scope(self, scope):
        """Extract token from scope using custom or default logic."""
        return await extract_token_from_scope(scope)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Check if the current request path matches the defined route
            match, params = self.route.matches(scope)
            path_params = params.get("path_params", {})
            if match == Match.FULL:
                # Extract workspace and service_id from the matched path
                workspace = path_params["workspace"]
                service_id = path_params["service_id"]
                path = path_params["path"]

                try:
                    scope = {
                        k: scope[k]
                        for k in scope
                        if isinstance(
                            scope[k],
                            (str, int, float, bool, tuple, list, dict, bytes),
                        )
                    }
                    if not path.startswith("/"):
                        path = "/" + path
                    scope["path"] = path
                    scope["raw_path"] = path.encode("latin-1")

                    # get _mode from query string
                    query = scope.get("query_string", b"").decode("utf-8")
                    if query:
                        _mode = dict([q.split("=") for q in query.split("&")]).get(
                            "_mode"
                        )
                        # TODO: remove _mode from query string
                    else:
                        _mode = None

                    # Create a Request object with the scope
                    request = Request(scope, receive=receive)
                    user_info = await self.store.login_optional(request)

                    # Here we must always use the user's current workspace for the interface
                    # This is a security measure to prevent unauthorized access to private workspaces
                    async with self.store.get_workspace_interface(
                        user_info, user_info.scope.current_workspace
                    ) as api:
                        # Call get_service to trigger lazy loading if needed
                        service = await api.get_service(
                            workspace + "/" + service_id, {"mode": _mode}
                        )
                        # intercept the request if it's an ASGI service
                        # Check multiple possible ways the service type might be stored
                        service_type = getattr(service, "type", None)
                        if service_type in ["asgi", "ASGI"]:
                            # Call the ASGI app with manually provided receive and send
                            await service.serve(
                                {
                                    "scope": scope,
                                    "receive": receive,
                                    "send": send,
                                }
                            )
                        elif service_type == "functions":
                            await self.handle_function_service(
                                service, path, scope, receive, send
                            )
                        else:
                            await send(
                                {
                                    "type": "http.response.start",
                                    "status": 500,
                                    "headers": [
                                        [b"content-type", b"text/plain"],
                                    ],
                                }
                            )
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": b"Invalid service type: "
                                    + str(service_type).encode(),
                                    "more_body": False,
                                }
                            )
                        return
                except Exception as exp:
                    logger.exception(f"Error in ASGI service: {exp}")
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 500,
                            "headers": [
                                [b"content-type", b"text/plain"],
                            ],
                        }
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": f"Internal Server Error: {exp}".encode(),
                            "more_body": False,
                        }
                    )
                    return

        await self.app(scope, receive, send)

    def _find_nested_function(self, service, path_parts):
        """Find a function in nested service structure."""
        current = service
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current if callable(current) else None

    async def handle_function_service(self, service, path, scope, receive, send):
        """Handle function service with support for nested functions and default fallback."""
        # Clean up the path
        path = path.strip("/")
        if not path:
            path = "index"

        path_parts = path.split("/")

        # Try to find the exact function match (including nested)
        func = self._find_nested_function(service, path_parts)

        # If no exact match, try index function for directory-style access
        if not func and len(path_parts) > 1:
            index_parts = path_parts + ["index"]
            func = self._find_nested_function(service, index_parts)

        # If still no match, try the default function as fallback
        if not func and "default" in service and callable(service["default"]):
            func = service["default"]
            # For default function, we pass the original path in the scope
            scope["function_path"] = "/" + path if path != "index" else "/"

        if func:
            scope["query_string"] = scope["query_string"].decode("utf-8")
            scope["raw_path"] = scope["raw_path"].decode("latin-1")
            scope["headers"] = dict(Headers(scope=scope).items())
            event = await receive()
            body = event["body"]
            while event.get("more_body"):
                body += await receive()["body"]
            scope["body"] = body or None

            try:
                result = await func(scope)
                headers = MutableHeaders(headers=result.get("headers"))
                origin = scope.get("headers", {}).get("origin")
                cors_headers = {
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Origin": origin or "*",
                    "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
                    "Access-Control-Allow-Headers": "Authorization,Content-Type",
                    "Access-Control-Expose-Headers": "Content-Disposition",
                }
                for key, value in cors_headers.items():
                    headers[key] = value

                body = result.get("body")
                status = result.get("status", 200)
                assert isinstance(status, int)
                start = {
                    "type": "http.response.start",
                    "status": status,
                    "headers": headers.raw,
                }
                if not body:
                    start["more_body"] = False
                await send(start)
                if body:
                    if not isinstance(body, bytes):
                        body = body.encode()
                    await send(
                        {
                            "type": "http.response.body",
                            "body": body,
                            "more_body": False,
                        }
                    )
            except Exception:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [
                            [b"content-type", b"text/plain"],
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"Internal Server Error: "
                        + traceback.format_exc().encode(),
                        "more_body": False,
                    }
                )
        else:
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [
                        [b"content-type", b"text/plain"],
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"Function not found: " + path.encode(),
                    "more_body": False,
                }
            )


class HTTPProxy:
    """A proxy for accessing services from HTTP."""

    def __init__(
        self,
        store: RedisStore,
        app: ASGIApp,
        endpoint_url=None,
        access_key_id=None,
        secret_access_key=None,
        region_name=None,
        workspace_bucket="hypha-workspaces",
        base_path="/",
        enable_a2a=False,
        enable_mcp=False,
    ) -> None:
        """Initialize the http proxy."""
        # pylint: disable=broad-except
        self.store = store
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name
        self.s3_enabled = endpoint_url is not None
        self.workspace_bucket = workspace_bucket
        self.ws_apps_dir = Path(__file__).parent / "templates/ws"
        self.ws_app_files = os.listdir(self.ws_apps_dir)
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_files = os.listdir(self.templates_dir)
        self.jinja_env = Environment(
            loader=PackageLoader("hypha"), autoescape=select_autoescape()
        )
        self.server_info = self.store.get_server_info()

        def norm_url(url):
            return base_path.rstrip("/") + url

        # download the hypha-rpc-websocket.js file from the CDN
        self.rpc_lib_esm_content = None
        self.rpc_lib_umd_content = None

        @app.get(norm_url("/assets/hypha-rpc-websocket.mjs"))
        async def hypha_rpc_websocket_mjs(request: Request):
            if not self.rpc_lib_esm_content:
                _rpc_lib_script = f"https://cdn.jsdelivr.net/npm/hypha-rpc@{hypha_rpc_version}/dist/hypha-rpc-websocket.mjs"
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(_rpc_lib_script)
                    response.raise_for_status()
                    self.rpc_lib_esm_content = response.content
            return Response(
                content=self.rpc_lib_esm_content, media_type="application/javascript"
            )

        @app.get(norm_url("/assets/hypha-rpc-websocket.js"))
        async def hypha_rpc_websocket_js(request: Request):
            if not self.rpc_lib_umd_content:
                _rpc_lib_script = f"https://cdn.jsdelivr.net/npm/hypha-rpc@{hypha_rpc_version}/dist/hypha-rpc-websocket.js"
                async with httpx.AsyncClient() as client:
                    response = await client.get(_rpc_lib_script)
                    response.raise_for_status()
                    self.rpc_lib_umd_content = response.content
            return Response(
                content=self.rpc_lib_umd_content, media_type="application/javascript"
            )

        @app.get(norm_url("/assets/config.json"))
        async def get_config(
            request: Request,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            return JSONResponse(
                status_code=200,
                content={"user": user_info.model_dump(), **self.server_info},
            )

        @app.post(norm_url("/oauth/token"))
        async def token_proxy(request: Request):
            form_data = await request.form()
            async with httpx.AsyncClient() as client:
                auth0_response = await client.post(
                    f"https://{AUTH0_DOMAIN}/oauth/token",
                    data=form_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                return JSONResponse(status_code=200, content=auth0_response.json())

        @app.get(norm_url("/oauth/authorize"))
        async def auth_proxy(request: Request):
            # Construct the full URL for the Auth0 authorize endpoint with the query parameters
            auth0_authorize_url = (
                f"https://{AUTH0_DOMAIN}/authorize?{request.query_params}"
            )

            # Redirect the client to the constructed URL
            return RedirectResponse(url=auth0_authorize_url)

        @app.get(norm_url("/{workspace}/info"))
        async def get_workspace_info(
            workspace: str,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for checking details of a workspace."""
            try:
                if not user_info.check_permission(workspace, UserPermission.read):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": (
                                f"{user_info.id} has no"
                                f" permission to access {workspace}"
                            ),
                        },
                    )
                info = await self.store.get_workspace_info(workspace, load=True)
                return JSONResponse(
                    status_code=200,
                    content=info.model_dump(),
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": str(exp)},
                )

        @app.get(norm_url("/{workspace}/services"))
        async def get_workspace_services_redirect(
            workspace: str,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            return RedirectResponse(url=norm_url(f"/{workspace}/services/"))

        @app.get(norm_url("/{workspace}/services/"))
        async def get_workspace_services(
            workspace: str,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for get services under a workspace."""
            try:
                async with self.store.get_workspace_interface(
                    user_info, workspace
                ) as manager:
                    services = await manager.list_services()
                info = serialize(services)
                # remove workspace prefix
                for service in info:
                    service["id"] = service["id"].split("/")[-1]
                return JSONResponse(
                    status_code=200,
                    content=info,
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": str(exp)},
                )

        @app.get(norm_url("/{workspace}/services/{service_id}"))
        async def get_service_info(
            workspace: str,
            service_id: str,
            _mode: str = None,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for checking details of a service."""
            try:
                async with self.store.get_workspace_interface(
                    user_info, workspace
                ) as api:
                    if service_id == "ws":
                        return serialize(api)
                    service_info = await api.get_service_info(
                        service_id, {"mode": _mode, "read_app_manifest": True}
                    )
                return JSONResponse(
                    status_code=200,
                    content=serialize(service_info),
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": str(exp)},
                )

        async def _call_service_function(func, kwargs):
            """Call a service function by keys."""
            _rpc = RPC(None, "anon")

            kwargs = _rpc.decode(kwargs)
            results = func(**kwargs)
            if inspect.isawaitable(results):
                results = await results

            # Check if the result is a generator (regular or async)
            if inspect.isgenerator(results) or inspect.isasyncgen(results):
                return results  # Return the generator directly without encoding

            results = _rpc.encode(results)
            return results

        async def _create_streaming_response(generator, response_type):
            """Create a streaming response from a generator."""
            _rpc = RPC(None, "anon")

            if response_type == "application/msgpack":
                # For msgpack, stream individual encoded items
                async def stream_msgpack():
                    try:
                        if inspect.isasyncgen(generator):
                            async for item in generator:
                                encoded_item = _rpc.encode(item)
                                yield msgpack.dumps(encoded_item)
                        else:
                            # Regular generator - convert to async
                            for item in generator:
                                encoded_item = _rpc.encode(item)
                                yield msgpack.dumps(encoded_item)
                    except Exception as e:
                        logger.error(f"Error in msgpack streaming: {e}")
                        # Send error as final chunk
                        error_data = {
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        yield msgpack.dumps(error_data)

                return StreamingResponse(
                    stream_msgpack(),
                    media_type="application/msgpack",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            else:
                # For JSON, use Server-Sent Events format
                async def stream_json():
                    try:
                        if inspect.isasyncgen(generator):
                            async for item in generator:
                                encoded_item = _rpc.encode(item)
                                json_data = json.dumps(encoded_item)
                                yield f"data: {json_data}\n\n"
                        else:
                            # Regular generator - convert to async
                            for item in generator:
                                encoded_item = _rpc.encode(item)
                                json_data = json.dumps(encoded_item)
                                yield f"data: {json_data}\n\n"

                        # Send end marker
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"Error in JSON streaming: {e}")
                        # Send error as final event
                        error_data = {
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        json_data = json.dumps(error_data)
                        yield f"data: {json_data}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_json(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Cache-Control",
                    },
                )

        @app.get(norm_url("/{workspace}/apps"))
        async def get_workspace_apps(
            workspace: str,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for get apps under a workspace."""
            try:
                async with self.store.get_workspace_interface(
                    user_info, workspace or user_info.scope.current_workspace
                ) as manager:
                    try:
                        controller = await manager.get_service("public/server-apps")
                    except KeyError:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "success": False,
                                "detail": "Server Apps service is not enabled.",
                            },
                        )
                    apps = await controller.list_apps()
                return JSONResponse(
                    status_code=200,
                    content=serialize(apps),
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": str(exp)},
                )

        @app.get(norm_url("/{workspace}/apps/ws/{path:path}"))
        async def get_ws_app_file(
            workspace: str,
            path: str,
            user_info: store.login_optional = Depends(store.login_optional),
        ) -> Response:
            """Route for getting browser app files."""
            if not user_info.check_permission(workspace, UserPermission.read):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": (
                            f"{user_info.id} has no"
                            f" permission to access {workspace}"
                        ),
                    },
                )

            if not path:
                return FileResponse(safe_join(str(self.ws_apps_dir), "index.html"))

            # check if the path is inside the built-in apps dir
            # get the jinja template from the built-in apps dir
            dir_path = path.split("/")[0]
            if dir_path in self.ws_app_files:
                file_path = safe_join(str(self.ws_apps_dir), path)
                if not is_safe_path(str(self.ws_apps_dir), file_path):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": f"Unsafe path: {file_path}",
                        },
                    )
                if not os.path.exists(file_path):
                    return JSONResponse(
                        status_code=404,
                        content={
                            "success": False,
                            "detail": f"File not found: {path}",
                        },
                    )
                return FileResponse(safe_join(str(self.ws_apps_dir), "ws", path))
            else:
                key = safe_join(workspace, path)
                assert self.s3_enabled, "S3 is not enabled."
                from hypha.s3 import FSFileResponse
                from aiobotocore.session import get_session

                s3_client = get_session().create_client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name=self.region_name,
                )
                return FSFileResponse(s3_client, self.workspace_bucket, key)

        app.add_middleware(
            ASGIRoutingMiddleware,
            base_path=base_path,
            store=store,
        )

        # Add A2A middleware for A2A services if enabled
        logger.info(f"enable_a2a = {enable_a2a}")
        if enable_a2a:
            try:
                from hypha.a2a import A2ARoutingMiddleware

                logger.info("Adding A2ARoutingMiddleware to the app")
                app.add_middleware(
                    A2ARoutingMiddleware,
                    base_path=base_path,
                    store=store,
                )
                logger.info("A2ARoutingMiddleware added successfully")
            except ImportError as e:
                logger.error(f"Failed to import A2ARoutingMiddleware: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to add A2ARoutingMiddleware: {e}")
                raise
        else:
            logger.info("A2A middleware not enabled (enable_a2a=False)")

        # Add MCP middleware for MCP services if enabled
        logger.info(f"enable_mcp = {enable_mcp}")
        if enable_mcp:
            try:
                from hypha.mcp import MCPRoutingMiddleware

                logger.info("Adding MCPRoutingMiddleware to the app")
                app.add_middleware(
                    MCPRoutingMiddleware,
                    base_path=base_path,
                    store=store,
                )
                logger.info("MCPRoutingMiddleware added successfully")
            except ImportError as e:
                logger.error(f"Failed to import MCPRoutingMiddleware: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to add MCPRoutingMiddleware: {e}")
                raise
        else:
            logger.info("MCP middleware not enabled (enable_mcp=False)")

        @app.get(norm_url("/{workspace}/apps/{service_id}"))
        async def get_app_info(
            workspace: str,
            service_id: str,
            request: Request,
            _mode: str = None,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for checking details of an app."""
            return RedirectResponse(
                url=f"{base_path.rstrip('/')}/{workspace}/apps/{service_id}/"
            )

        @app.get(norm_url("/{workspace}/services/{service_id}/{function_key}"))
        @app.post(norm_url("/{workspace}/services/{service_id}/{function_key}"))
        async def call_service_function(
            workspace: str,
            service_id: str,
            function_key: str,
            request: Request,
            _mode: str = None,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Run service function by keys."""
            function_kwargs = await extracted_kwargs(request, use_function_kwargs=False)
            response_type = detected_response_type(request)
            try:
                return await service_function(
                    (workspace, service_id, function_key),
                    function_kwargs,
                    response_type,
                    user_info,
                    _mode=_mode,
                )
            except Exception:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "detail": traceback.format_exc()},
                )

        async def service_function(
            function_info: extracted_call_info = Depends(extracted_call_info),
            function_kwargs: extracted_kwargs = Depends(extracted_kwargs),
            response_type: detected_response_type = Depends(detected_response_type),
            user_info: store.login_optional = Depends(store.login_optional),
            _mode: str = None,
        ):
            """Run service function by keys.

            It can contain dot to refer to deeper object.
            """
            try:
                workspace, service_id, function_key = function_info
                # The workspace should always be the user's current workspace
                # derived from the token
                # We should never trust the workspace specified in the url or query
                async with self.store.get_workspace_interface(
                    user_info, user_info.scope.current_workspace
                ) as api:
                    if service_id == "ws":
                        service = api
                    else:
                        service = await api.get_service(
                            workspace + "/" + service_id, {"mode": _mode}
                        )
                    func = get_value(function_key, service)
                    if not func:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "success": False,
                                "detail": f"{function_key} not found.",
                            },
                        )
                    if not callable(func):
                        return JSONResponse(status_code=200, content=serialize(func))

                    try:
                        results = await _call_service_function(func, function_kwargs)
                    except Exception:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "success": False,
                                "detail": traceback.format_exc(),
                            },
                        )

                    # Check if the result is a generator - if so, create streaming response
                    if inspect.isgenerator(results) or inspect.isasyncgen(results):
                        return await _create_streaming_response(results, response_type)

                    if response_type == "application/msgpack":
                        return MsgpackResponse(
                            status_code=200,
                            content=results,
                        )

                    return JSONResponse(
                        status_code=200,
                        content=results,
                    )

            except Exception:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": traceback.format_exc()},
                )

        @app.get(norm_url("/health/readiness"))
        async def readiness(req: Request) -> JSONResponse:
            """Used for readiness probe.
            Determines whether the application inside the container is ready to accept traffic or requests.
            Checks:
            1. Store initialization status
            2. Redis connectivity
            3. Event bus and pubsub functionality
            """
            try:
                # Check store readiness
                if not self.store.is_ready():
                    return JSONResponse(
                        {"status": "DOWN", "detail": "Store is not ready"},
                        status_code=503,
                    )

                # Check Redis connectivity
                try:
                    redis = self.store.get_redis()
                    await redis.ping()
                except Exception as e:
                    return JSONResponse(
                        {
                            "status": "DOWN",
                            "detail": f"Redis connection failed: {str(e)}",
                        },
                        status_code=503,
                    )

                # Check event bus status
                event_bus = self.store.get_event_bus()
                if event_bus._circuit_breaker_open:
                    return JSONResponse(
                        {
                            "status": "DOWN",
                            "detail": "Event bus circuit breaker is open",
                        },
                        status_code=503,
                    )

                # Check pubsub health
                pubsub_healthy = await event_bus._check_pubsub_health()
                if not pubsub_healthy:
                    return JSONResponse(
                        {"status": "DOWN", "detail": "Pubsub system is not healthy"},
                        status_code=503,
                    )

                return JSONResponse(
                    {
                        "status": "OK",
                        "detail": {
                            "store": "ready",
                            "redis": "connected",
                            "event_bus": "ready",
                            "pubsub": "healthy",
                        },
                    }
                )

            except Exception as e:
                logger.error(f"Readiness check failed: {str(e)}")
                return JSONResponse(
                    {"status": "DOWN", "detail": f"Readiness check failed: {str(e)}"},
                    status_code=503,
                )

        @app.get(norm_url("/health/liveness"))
        async def liveness(req: Request) -> JSONResponse:
            """Used for liveness probe.
            If the liveness probe fails, it means the app is in a failed state and needs to be restarted.
            Checks:
            1. Store health
            2. Event bus health
            3. Pubsub functionality
            """
            # Check store health
            if not self.store.is_ready():
                return JSONResponse(
                    {"status": "DOWN", "detail": "Store is not ready"}, status_code=503
                )
            try:
                if not await self.store.is_alive():
                    return JSONResponse(
                        {"status": "DOWN", "detail": "Server is not alive"},
                        status_code=503,
                    )
            except Exception as e:
                logger.error(f"Liveness check failed: {str(e)}")
                return JSONResponse(
                    {"status": "DOWN", "detail": f"Liveness check failed: {str(e)}"},
                    status_code=503,
                )

            try:

                # Get event bus status
                event_bus = self.store.get_event_bus()

                # Check event bus consecutive failures
                if event_bus._consecutive_failures >= event_bus._max_failures:
                    return JSONResponse(
                        {
                            "status": "DOWN",
                            "detail": f"Event bus has {event_bus._consecutive_failures} consecutive failures",
                        },
                        status_code=503,
                    )

                # Check pubsub health
                pubsub_healthy = await event_bus._check_pubsub_health()
                if not pubsub_healthy:
                    return JSONResponse(
                        {"status": "DOWN", "detail": "Pubsub system is not healthy"},
                        status_code=503,
                    )

                # Return detailed status
                return JSONResponse(
                    {
                        "status": "OK",
                        "detail": {
                            "store": "ready",
                            "event_bus_failures": event_bus._consecutive_failures,
                            "circuit_breaker": (
                                "open" if event_bus._circuit_breaker_open else "closed"
                            ),
                            "pubsub": "healthy",
                            "last_successful_connection": (
                                datetime.datetime.fromtimestamp(
                                    event_bus._last_successful_connection
                                ).isoformat()
                                if event_bus._last_successful_connection
                                else None
                            ),
                        },
                    }
                )

            except Exception as e:
                logger.error(f"Liveness check failed: {str(e)}")
                return JSONResponse(
                    {"status": "DOWN", "detail": f"Liveness check failed: {str(e)}"},
                    status_code=503,
                )

        @app.get(norm_url("/login"))
        async def login(request: Request):
            """Redirect to the login page."""
            return RedirectResponse(norm_url("/public/apps/hypha-login/"))

        @app.get(norm_url("/metrics"))
        async def metrics():
            """Expose Prometheus metrics."""
            return Response(generate_latest(), media_type="text/plain")

        @app.get(norm_url("/{page:path}"))
        async def get_pages(
            page: str,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for getting pages, if other matches not found, this will be the fallback."""
            if not page or page == "/":
                page = "index.html"
            dir_path = page.split("/")[0]
            if dir_path not in self.templates_files and "-" in dir_path:
                inner_path = "/".join(page.split("/")[1:])
                if not inner_path:
                    inner_path = "index.html"
                if not os.path.exists(safe_join(str(self.ws_apps_dir), inner_path)):
                    return JSONResponse(
                        status_code=404,
                        content={
                            "success": False,
                            "detail": f"File not found: {inner_path}",
                        },
                    )
                else:
                    # redirect /{workspace} to /{workspace}/
                    if not page.endswith("/") and inner_path == "index.html":
                        # redirect it to the version with trailing slash
                        return RedirectResponse(norm_url(f"/{dir_path}/"))
                return FileResponse(safe_join(str(self.ws_apps_dir), inner_path))
            file_path = safe_join(str(self.templates_dir), page)
            if not is_safe_path(str(self.templates_dir), file_path):
                return JSONResponse(
                    status_code=403,
                    content={"success": False, "detail": f"Unsafe path: {file_path}"},
                )
            if not os.path.exists(file_path):
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"File not found: {file_path}",
                    },
                )
            assert os.path.basename(file_path) not in [
                "apps",
                "ws",
            ], f"Invalid page name: {page}"
            # compile the jinja template
            file_path = safe_join(str(self.templates_dir), page)
            assert is_safe_path(
                str(self.templates_dir), file_path
            ), f"Unsafe path: {page}"
            if not os.path.exists(file_path):
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"File not found: {inner_path}",
                    },
                )
            return FileResponse(file_path)
