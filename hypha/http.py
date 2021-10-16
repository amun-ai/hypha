"""Provide the http proxy."""
import json
import traceback
from typing import Any

import msgpack
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from hypha.core.auth import login_optional
from hypha.core.interface import CoreInterface


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
        return f"<function: {str(obj)}>"

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

    def __init__(self, core_interface: CoreInterface) -> None:
        """Initialize the http proxy."""
        # pylint: disable=broad-except
        router = APIRouter()
        self.core_interface = core_interface

        @router.get("/services")
        def get_all_services(
            user_info: login_optional = Depends(login_optional),
        ):
            """Route for listing all the services."""
            try:
                core_interface.current_user.set(user_info)
                services = core_interface.list_services()
                info = serialize(services)
                return JSONResponse(
                    status_code=200,
                    content=info,
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": str(exp)},
                )

        @router.get("/{workspace}/services")
        def get_workspace_services(
            workspace: str,
            user_info: login_optional = Depends(login_optional),
        ):
            """Route for get services under a workspace."""
            try:
                core_interface.current_user.set(user_info)
                services = core_interface.list_services({"workspace": workspace})
                info = serialize(services)
                return JSONResponse(
                    status_code=200,
                    content=info,
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": str(exp)},
                )

        @router.get("/{workspace}/services/{service}")
        async def get_service_info(
            workspace: str,
            service: str,
            user_info: login_optional = Depends(login_optional),
        ):
            """Route for checking details of a service."""
            try:
                core_interface.current_user.set(user_info)
                ws = core_interface.get_workspace(workspace)
                service = ws.get_service_by_name(service)
                info = service.config.dict()
                return JSONResponse(
                    status_code=200,
                    content=info,
                )
            except Exception as exp:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": str(exp)},
                )

        @router.get("/{workspace}/services/{service}/{keys}")
        @router.post("/{workspace}/services/{service}/{keys}")
        async def service_function(
            workspace: str,
            service: str,
            keys: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            """Run service function by keys.

            It can contain dot to refer to deeper object.
            """
            try:
                core_interface.current_user.set(user_info)
                ws = core_interface.get_workspace(workspace)
                service = ws.get_service_by_name(service)
                value = get_value(keys, service)
                if not value:
                    return JSONResponse(
                        status_code=200,
                        content={"success": False, "detail": f"{keys} not found."},
                    )
                content_type = request.headers.get("content-type", "application/json")
                if request.method == "GET":
                    kwargs = list(request.query_params.items())
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
                try:
                    result = await value(**kwargs)
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
                        content=result,
                    )

                return JSONResponse(
                    status_code=200,
                    content=result,
                )

            except Exception:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "detail": traceback.format_exc()},
                )

        core_interface.register_router(router)
