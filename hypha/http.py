"""Provide the http proxy."""
import inspect
import json
import traceback
from typing import Any
import asyncio
from pathlib import Path
import requests

import httpx
import msgpack
from fastapi import APIRouter, Depends, Request
from jinja2 import Environment, PackageLoader, select_autoescape
from fastapi.responses import (
    JSONResponse,
    Response,
    RedirectResponse,
    StreamingResponse,
    FileResponse,
)
import jose
import os

from starlette.datastructures import Headers

from hypha_rpc import RPC
from hypha import main_version
from hypha.core import UserPermission
from hypha.core.auth import AUTH0_DOMAIN
from hypha.core.store import RedisStore
from hypha.utils import GzipRoute, safe_join, is_safe_path


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
            for key in kwargs:
                if key in ["workspace", "service_id", "function_key"]:
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
        if use_function_kwargs:
            kwargs = kwargs.get("function_kwargs", {})
        else:
            for key in ["workspace", "service_id", "function_key"]:
                if key in kwargs:
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


class HTTPProxy:
    """A proxy for accessing services from HTTP."""

    def __init__(
        self,
        store: RedisStore,
        endpoint_url=None,
        access_key_id=None,
        secret_access_key=None,
        region_name=None,
        workspace_bucket="hypha-workspaces",
        workspace_etc_dir="etc",
        base_path="/",
    ) -> None:
        """Initialize the http proxy."""
        # pylint: disable=broad-except
        router = APIRouter()
        router.route_class = GzipRoute
        self.store = store
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name
        self.s3_enabled = endpoint_url is not None
        self.workspace_bucket = workspace_bucket
        self.workspace_etc_dir = workspace_etc_dir
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

        @router.get(norm_url("/assets/hypha-rpc-websocket.mjs"))
        async def hypha_rpc_websocket_mjs(request: Request):
            if not self.rpc_lib_esm_content:
                _rpc_lib_script = f"https://cdn.jsdelivr.net/npm/hypha-rpc@{main_version}/dist/hypha-rpc-websocket.mjs"
                response = requests.get(_rpc_lib_script)
                response.raise_for_status()
                self.rpc_lib_esm_content = response.content
            return Response(
                content=self.rpc_lib_esm_content, media_type="application/javascript"
            )

        @router.get(norm_url("/assets/hypha-rpc-websocket.js"))
        async def hypha_rpc_websocket_js(request: Request):
            if not self.rpc_lib_umd_content:
                _rpc_lib_script = f"https://cdn.jsdelivr.net/npm/hypha-rpc@{main_version}/dist/hypha-rpc-websocket.js"
                response = requests.get(_rpc_lib_script)
                response.raise_for_status()
                self.rpc_lib_umd_content = response.content
            return Response(
                content=self.rpc_lib_umd_content, media_type="application/javascript"
            )

        @router.get(norm_url("/assets/config.json"))
        async def get_config(
            request: Request,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            return JSONResponse(
                status_code=200,
                content={"user": user_info.model_dump(), **self.server_info},
            )

        @router.post(norm_url("/oauth/token"))
        async def token_proxy(request: Request):
            form_data = await request.form()
            async with httpx.AsyncClient() as client:
                auth0_response = await client.post(
                    f"https://{AUTH0_DOMAIN}/oauth/token",
                    data=form_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                return JSONResponse(status_code=200, content=auth0_response.json())

        @router.get(norm_url("/oauth/authorize"))
        async def auth_proxy(request: Request):
            # Construct the full URL for the Auth0 authorize endpoint with the query parameters
            auth0_authorize_url = (
                f"https://{AUTH0_DOMAIN}/authorize?{request.query_params}"
            )

            # Redirect the client to the constructed URL
            return RedirectResponse(url=auth0_authorize_url)

        @router.get(norm_url("/{workspace}/info"))
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

                info = await self.store.load_or_create_workspace(user_info, workspace)
                return JSONResponse(
                    status_code=200,
                    content=info.model_dump(),
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": str(exp)},
                )

        @router.get(norm_url("/{workspace}/services"))
        async def get_workspace_services(
            workspace: str,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for get services under a workspace."""
            try:
                async with self.store.get_workspace_interface(
                    workspace, user_info
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

        @router.get(norm_url("/{workspace}/services/{service_id}"))
        async def get_service_info(
            workspace: str,
            service_id: str,
            mode: str = None,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for checking details of a service."""
            try:
                async with self.store.get_workspace_interface(
                    workspace, user_info
                ) as api:
                    if service_id == "ws":
                        return serialize(api)
                    service_info = await api.get_service_info(
                        service_id, {"mode": mode}
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
            results = _rpc.encode(results)
            return results

        @router.get(norm_url("/{workspace}/apps"))
        async def get_workspace_apps(
            workspace: str,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for get apps under a workspace."""
            try:
                async with self.store.get_workspace_interface(
                    workspace, user_info
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
                    apps = await controller.list_apps(workspace)
                return JSONResponse(
                    status_code=200,
                    content=serialize(apps),
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": str(exp)},
                )

        @router.get(norm_url("/{workspace}/apps/{service_id}"))
        async def get_app_info(
            workspace: str,
            service_id: str,
            request: Request,
            path: str = None,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for checking details of an app."""
            if not path:
                path = "/"
            return await run_app(
                workspace=workspace,
                service_id=service_id,
                request=request,
                path=path,
                user_info=user_info,
            )

        @router.get(norm_url("/{workspace}/apps/{service_id}/{path:path}"))
        async def run_app(
            workspace: str,
            service_id: str,
            request: Request,
            path: str = None,
            mode: str = None,
            user_info: store.login_optional = Depends(store.login_optional),
        ) -> Response:
            if service_id == "ws":
                if not path:
                    return FileResponse(
                        safe_join(str(self.ws_apps_dir), "ws/index.html")
                    )

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

            # Serve dynamic apps
            try:
                scope = request.scope
                scope = {
                    k: scope[k]
                    for k in scope
                    if isinstance(
                        scope[k], (str, int, float, bool, tuple, list, dict, bytes)
                    )
                }
                if not path.startswith("/"):
                    path = "/" + path
                scope["path"] = path
                scope["raw_path"] = path.encode("latin-1")
                send_queue = asyncio.Queue()
                async with self.store.get_workspace_interface(
                    workspace, user_info
                ) as api:
                    info = await api.get_service_info(service_id, {"mode": mode})
                    if info.type == "ASGI":
                        service = await api.get_service(info.id)
                        # Call the ASGI app with manually provided receive and send
                        await service.serve(
                            {
                                "scope": scope,
                                "receive": request.body,
                                "send": send_queue.put,
                            }
                        )
                    elif info.type == "functions":
                        func_name = path.split("/", 1)[-1] or "index"
                        func_name = func_name.rstrip("/")
                        service = await api.get_service(info.id)
                        if func_name in service and callable(service[func_name]):
                            scope["query_string"] = scope["query_string"].decode(
                                "utf-8"
                            )
                            scope["raw_path"] = scope["raw_path"].decode("latin-1")
                            scope["headers"] = dict(Headers(scope=scope).items())
                            event = await request.receive()
                            body = event["body"]
                            while event.get("more_body"):
                                body += await request.receive()["body"]
                            scope["body"] = body or None
                            func = service[func_name]
                            try:
                                result = await func(scope)
                                headers = Headers(headers=result.get("headers"))
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
                                await send_queue.put(start)
                                if body:
                                    if not isinstance(body, bytes):
                                        body = body.encode()
                                    await send_queue.put(
                                        {
                                            "type": "http.response.body",
                                            "body": body,
                                            "more_body": False,
                                        }
                                    )
                            except Exception:
                                await send_queue.put(
                                    {
                                        "type": "http.response.start",
                                        "status": 500,
                                        "headers": [
                                            [b"content-type", b"text/plain"],
                                        ],
                                    }
                                )

                        else:
                            return JSONResponse(
                                status_code=404,
                                content={
                                    "success": False,
                                    "detail": f"Function not found: {func_name}",
                                },
                            )
                    else:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "success": False,
                                "detail": f"Service cannot be run as an app: {service_id}",
                            },
                        )

                    async def send_response(send_queue):
                        # This function will be called to send data as it's received from the ASGI app
                        while True:
                            message = await send_queue.get()
                            if message["type"] == "http.response.body":
                                yield message["body"]
                                if not message.get("more_body", False):
                                    break

                    # Extract initial status and headers from the queue
                    initial_message = await send_queue.get()
                    assert initial_message["type"] == "http.response.start"
                    status = initial_message.get("status", 200)
                    raw_headers = initial_message.get("headers", [])
                    headers = {
                        k.decode("latin-1"): v.decode("latin-1") for k, v in raw_headers
                    }

                    return StreamingResponse(
                        send_response(send_queue), status_code=status, headers=headers
                    )
            except KeyError:
                return Response(status_code=404)

        @router.get(norm_url("/{workspace}/server-apps/{app_id}/{path:path}"))
        async def get_browser_app_file(
            workspace: str, app_id: str, path: str, token: str = None
        ) -> Response:
            """Route for getting browser app files."""
            if token is None:
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": (f"Token not provided for {workspace}/{path}"),
                    },
                )
            try:
                user_info = await store.parse_user_token(token)
            except jose.exceptions.JWTError:
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": (
                            f"Invalid token not provided for {workspace}/{path}"
                        ),
                    },
                )
            if not user_info.check_permission(workspace, UserPermission.read):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": (
                            f"{user_info['username']} has no"
                            f" permission to access {workspace}"
                        ),
                    },
                )
            key = safe_join(self.workspace_etc_dir, workspace, app_id, path)
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

        @router.get(norm_url("/{workspace}/services/{service_id}/{function_key}"))
        @router.post(norm_url("/{workspace}/services/{service_id}/{function_key}"))
        async def call_service_function(
            workspace: str,
            service_id: str,
            function_key: str,
            request: Request,
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
        ):
            """Run service function by keys.

            It can contain dot to refer to deeper object.
            """
            try:
                workspace, service_id, function_key = function_info
                async with self.store.get_workspace_interface(
                    workspace, user_info
                ) as api:
                    if service_id == "ws":
                        service = api
                    else:
                        service = await api.get_service(service_id)
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

        @router.get(norm_url("/health/readiness"))
        async def readiness(req: Request) -> JSONResponse:
            """Used for readiness probe.
            t determines whether the application inside the container is ready to accept traffic or requests.
            """
            if store.is_ready():
                return JSONResponse({"status": "OK"})

            return JSONResponse({"status": "DOWN"}, status_code=503)

        @router.get(norm_url("/health/liveness"))
        async def liveness(req: Request) -> JSONResponse:
            """Used for liveness probe.
            If the liveness probe fails, it means the app is in a failed state and restarts it.
            """
            return JSONResponse({"status": "OK"})

        @router.get(norm_url("/{page:path}"))
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

        store.register_router(router)
