"""Provide the http proxy."""
import inspect
import json
import traceback
from typing import Any, Optional

import httpx
import msgpack
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, RedirectResponse

from imjoy_rpc.hypha import RPC
from hypha.core.auth import login_optional, AUTH0_DOMAIN
from hypha.core.store import RedisStore
from hypha.utils import GzipRoute
from hypha import __version__ as VERSION

SERVICES_OPENAPI_SCHEMA = {
    "openapi": "3.1.0",
    "info": {
        "title": "Hypha Services",
        "description": "Providing access to services in Hypha",
        "version": f"v{VERSION}",
    },
    "paths": {
        "/call": {
            "post": {
                "description": "Call a service function",
                "operationId": "CallServiceFunction",
                "requestBody": {
                    "required": True,
                    "description": "The request body type depends on each service function schema",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/ServiceFunctionSchema"
                            }
                        }
                    },
                },
                "deprecated": False,
            }
        },
        "/list": {
            "get": {
                "description": "List services under a workspace",
                "operationId": "ListServices",
                "parameters": [
                    {
                        "name": "workspace",
                        "in": "query",
                        "description": "Workspace name",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "deprecated": False,
            }
        },
    },
    "components": {
        "schemas": {
            "ServiceFunctionSchema": {
                "type": "object",
                "properties": {
                    "workspace": {
                        "description": "Workspace name, optional if the service_id contains workspace name",
                        "type": "string",
                    },
                    "service_id": {
                        "description": "Service ID, format: workspace/client_id:service_id",
                        "type": "string",
                    },
                    "function_key": {
                        "description": "Function key, can contain dot to refer to deeper object",
                        "type": "string",
                    },
                    "function_kwargs": {
                        "description": "Function keyword arguments, must be set according to the function schema",
                        "type": "object",
                    },
                },
                "required": ["service_id", "function_key"],
            }
        }
    },
}


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
        return {"type": "function", "name": obj.__name__}

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

    def __init__(self, store: RedisStore) -> None:
        """Initialize the http proxy."""
        # pylint: disable=broad-except
        router = APIRouter()
        router.route_class = GzipRoute
        self.store = store

        @router.get("/authorize")
        async def auth_proxy(request: Request):
            # Construct the full URL for the Auth0 authorize endpoint with the query parameters
            auth0_authorize_url = (
                f"https://{AUTH0_DOMAIN}/authorize?{request.query_params}"
            )

            # Redirect the client to the constructed URL
            return RedirectResponse(url=auth0_authorize_url)

        @router.post("/oauth/token")
        async def token_proxy(request: Request):
            form_data = await request.form()
            async with httpx.AsyncClient() as client:
                auth0_response = await client.post(
                    f"https://{AUTH0_DOMAIN}/oauth/token",
                    data=form_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                return JSONResponse(status_code=200, content=auth0_response.json())

        @router.get("/authorize")
        async def auth_proxy(request: Request):
            # Construct the full URL for the Auth0 authorize endpoint with the query parameters
            auth0_authorize_url = (
                f"https://{AUTH0_DOMAIN}/authorize?{request.query_params}"
            )

            # Redirect the client to the constructed URL
            return RedirectResponse(url=auth0_authorize_url)

        @router.post("/oauth/token")
        async def token_proxy(request: Request):
            form_data = await request.form()
            async with httpx.AsyncClient() as client:
                auth0_response = await client.post(
                    f"https://{AUTH0_DOMAIN}/oauth/token",
                    data=form_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                return JSONResponse(status_code=200, content=auth0_response.json())

        @router.get("/workspaces/list")
        async def list_all_workspaces(
            user_info: login_optional = Depends(login_optional),
        ):
            """Route for listing all the workspaces."""
            try:
                workspaces = await store.list_all_workspaces()
                return JSONResponse(
                    status_code=200,
                    content=workspaces,
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": str(exp)},
                )

        @router.get("/{workspace}/info")
        @router.get("/workspaces/info")
        async def get_workspace_info(
            workspace: str,
            user_info: login_optional = Depends(login_optional),
        ):
            """Route for get detailed info of a workspace."""
            try:
                if workspace != "public" and not await store.check_permission(
                    workspace, user_info
                ):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": f"Permission denied to workspace: {workspace}",
                        },
                    )
                workspace_info = await store.get_workspace(workspace)
                return JSONResponse(
                    status_code=200,
                    content=workspace_info.model_dump(),
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": str(exp)},
                )

        @router.get("/services/openapi.json")
        async def get_services_openapi(
            user_info: login_optional = Depends(login_optional),
        ):
            """Get the openapi schema for services."""
            schema = SERVICES_OPENAPI_SCHEMA.copy()
            schema["servers"] = [{"url": f"{store.public_base_url}/services"}]
            return schema

        @router.get("/services/call")
        async def call_service_function_get(
            service_id: str,
            function_key: str,
            workspace: Optional[str] = None,
            function_kwargs: extracted_kwargs = Depends(extracted_kwargs),
            response_type: detected_response_type = Depends(detected_response_type),
            user_info: login_optional = Depends(login_optional),
        ):
            """Call a service function by keys."""
            if "/" in service_id:
                _workspace, service_id = service_id.split("/")
                if workspace:
                    assert (
                        _workspace == workspace
                    ), f"workspace mismatch: {_workspace} != {workspace}"
                else:
                    workspace = _workspace
            assert (
                workspace
            ), "workspace should be included in the service_id or provided separately"
            return await service_function(
                (workspace, service_id, function_key),
                function_kwargs,
                response_type,
                user_info,
            )

        @router.post("/services/call")
        async def call_service_function_post(
            request: Request = None,
            user_info: login_optional = Depends(login_optional),
        ):
            """Call a service function by keys."""
            workspace, service_id, function_key = await extracted_call_info(request)
            function_kwargs = await extracted_kwargs(request)
            response_type = detected_response_type(request)
            if "/" in service_id:
                _workspace, service_id = service_id.split("/")
                if workspace:
                    assert (
                        _workspace == workspace
                    ), f"workspace mismatch: {_workspace} != {workspace}"
                else:
                    workspace = _workspace
            assert (
                workspace
            ), "workspace should be included in the service_id or provided separately"
            return await service_function(
                (workspace, service_id, function_key),
                function_kwargs,
                response_type,
                user_info,
            )

        @router.get("/services/list")
        async def list_services(
            workspace: str,
            user_info: login_optional = Depends(login_optional),
        ):
            """List services under a workspace."""
            return await get_workspace_services(workspace, user_info)

        @router.get("/{workspace}/services")
        async def get_workspace_services(
            workspace: str,
            user_info: login_optional = Depends(login_optional),
        ):
            """Route for get services under a workspace."""
            try:
                manager = await store.get_workspace_manager(workspace)
                services = await manager.list_services(context={"user": user_info})
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

        @router.get("/{workspace}/services/{service_id}")
        async def get_service_info(
            workspace: str,
            service_id: str,
            user_info: login_optional = Depends(login_optional),
        ):
            """Route for checking details of a service."""
            try:
                service = await store.get_service_as_user(
                    workspace, service_id, user_info
                )
                return JSONResponse(
                    status_code=200,
                    content=serialize(service),
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

        @router.get("/{workspace}/services/{service_id}/{function_key}")
        async def service_function_get(
            workspace: str,
            service_id: str,
            function_key: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            """Run service function by keys."""
            function_kwargs = await extracted_kwargs(request, use_function_kwargs=False)
            response_type = detected_response_type(request)
            return await service_function(
                (workspace, service_id, function_key),
                function_kwargs,
                response_type,
                user_info,
            )

        @router.post("/{workspace}/services/{service_id}/{function_key}")
        async def service_function_post(
            workspace: str,
            service_id: str,
            function_key: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            """Run service function by keys."""
            function_kwargs = await extracted_kwargs(request, use_function_kwargs=False)
            response_type = detected_response_type(request)
            return await service_function(
                (workspace, service_id, function_key),
                function_kwargs,
                response_type,
                user_info,
            )

        async def service_function(
            function_info: extracted_call_info = Depends(extracted_call_info),
            function_kwargs: extracted_kwargs = Depends(extracted_kwargs),
            response_type: detected_response_type = Depends(detected_response_type),
            user_info: login_optional = Depends(login_optional),
        ):
            """Run service function by keys.

            It can contain dot to refer to deeper object.
            """
            try:
                workspace, service_id, function_key = function_info
                service = await store.get_service_as_user(
                    workspace, service_id, user_info
                )
                func = get_value(function_key, service)
                if not func:
                    return JSONResponse(
                        status_code=200,
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
                        status_code=500,
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

        store.register_router(router)
