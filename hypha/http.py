"""Provide the http proxy."""
import inspect
import json
import traceback
from typing import Any

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
                "parameters": [
                    {
                        "name": "workspace",
                        "in": "query",
                        "description": "Workspace name",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "service_id",
                        "in": "query",
                        "description": "Service ID",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "function_key",
                        "in": "query",
                        "description": "Function key, can contain dot to refer to deeper object",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                ],
                "requestBody": {
                    "required": False,
                    "description": "The request body type depends on each service function schema",
                    "content": {
                        "application/json": {
                            "schema": {}
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
        "schemas": {}
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
                if workspace != "public" and not await store.check_permission(workspace, user_info):
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
                    content=workspace_info.dict(),
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
        @router.post("/services/call")
        async def call_service_function(
            workspace: str,
            service_id: str,
            function_key: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            """Call a service function by keys."""
            return await service_function(
                workspace, service_id, function_key, request, user_info
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

        @router.post("/{workspace}/services/{service_id}/{function_key}")
        async def service_function(
            workspace: str,
            service_id: str,
            function_key: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            """Run service function by keys.

            It can contain dot to refer to deeper object.
            """
            try:
                service = await store.get_service_as_user(
                    workspace, service_id, user_info
                )
                value = get_value(function_key, service)
                if not value:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "success": False,
                            "detail": f"{function_key} not found.",
                        },
                    )
                content_type = request.headers.get("content-type", "application/json")
                if request.method == "GET":
                    kwargs = list(request.query_params.items())
                    for key in ["workspace", "service_id", "function_key"]:
                        kwargs = [k for k in kwargs if k[0] != key]
                    kwargs = {
                        kwargs[k][0]: normalize(kwargs[k][1])
                        for k in range(len(kwargs))
                    }
                elif request.method == "POST":
                    if content_type == "application/msgpack":
                        kwargs = msgpack.loads(await request.body())
                    elif content_type == "application/json":
                        kwargs = json.loads(await request.body())
                    else:
                        return JSONResponse(
                            status_code=500,
                            content={
                                "success": False,
                                "detail": "Invalid content-type (supported types: "
                                "application/msgpack, application/json, text/plain)",
                            },
                        )
                else:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "success": False,
                            "detail": f"Invalid request method: {request.method}",
                        },
                    )
                if not callable(value):
                    return JSONResponse(status_code=200, content=serialize(value))
                _rpc = RPC(None, "anon")
                try:
                    kwargs = _rpc.decode(kwargs)
                    results = value(**kwargs)
                    if inspect.isawaitable(results):
                        results = await results
                    results = _rpc.encode(results)
                except Exception:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "success": False,
                            "detail": traceback.format_exc(),
                        },
                    )

                if request.method == "POST" and content_type == "application/msgpack":
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
