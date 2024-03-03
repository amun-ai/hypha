"""Support ASGI web server apps."""
import asyncio
import logging
import sys
import traceback

from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send

from hypha.core import ServiceInfo
from hypha.core.auth import parse_token
from hypha.utils import PatchedCORSMiddleware

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("asgi")
logger.setLevel(logging.INFO)


class RemoteASGIApp:
    """Wrapper for a remote ASGI app."""

    def __init__(self, service: ServiceInfo) -> None:
        """Initialize the ASGI app."""
        self.service = service
        assert self.service.type in ["ASGI", "functions"]
        if self.service.type == "ASGI":
            assert not self.service.config.get(
                "require_context"
            ), "require_context must be False/None for ASGI apps"
            assert self.service.serve is not None, "No serve function defined"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle requests for the ASGI app."""
        scope = {
            k: scope[k]
            for k in scope
            if isinstance(scope[k], (str, int, float, bool, tuple, list, dict, bytes))
        }
        if self.service.type == "ASGI":
            interface = {
                "scope": scope,
                "receive": receive,
                "send": send,
            }
            await self.service.serve(interface)
        elif self.service.type == "functions":
            func_name = scope["path"].split("/", 1)[-1] or "index"
            func_name = func_name.rstrip("/")

            if func_name in self.service and callable(self.service[func_name]):
                scope["query_string"] = scope["query_string"].decode("utf-8")
                scope["raw_path"] = scope["raw_path"].decode("latin-1")
                scope["headers"] = dict(Headers(scope=scope).items())
                event = await receive()
                body = event["body"]
                while event.get("more_body"):
                    body += await receive()["body"]
                scope["body"] = body or None
                func = self.service[func_name]
                try:
                    if self.service.config.get("require_context"):
                        authorization = scope["headers"].get("authorization")
                        user_info = parse_token(authorization, allow_anonymouse=True)
                        result = await func(scope, {"user": user_info.model_dump()})
                    else:
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
                except Exception:  # pylint: disable=broad-except
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
                            "body": f"{traceback.format_exc()}".encode(),
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
                        "body": b"Not Found",
                        "more_body": False,
                    }
                )


class ASGIGateway:
    """ASGI gateway for running web servers in the browser apps."""

    def __init__(
        self,
        store,
        allow_origins=None,
        allow_methods=None,
        allow_headers=None,
        expose_headers=None,
    ):
        """Initialize the gateway."""
        self.store = store
        self.allow_origins = allow_origins
        self.allow_methods = allow_methods
        self.allow_headers = allow_headers
        self.expose_headers = expose_headers
        # TODO: query the current services and mount them
        event_bus = store.get_event_bus()
        event_bus.on(
            "service_registered",
            lambda service: asyncio.ensure_future(self.mount_asgi_app(service)),
        )
        event_bus.on(
            "service_unregistered",
            lambda service: asyncio.ensure_future(self.umount_asgi_app(service)),
        )

    async def mount_asgi_app(self, service: dict):
        """Mount the ASGI apps from new services."""
        service = ServiceInfo.model_validate(service)

        if service.type in ["ASGI", "functions"]:
            workspace = service.config.workspace
            # TODO: extract the user info and pass it
            # TODO: Support multiple worker processes
            service = await self.store.get_service_as_user(workspace, service.id)
            service_id = service.id
            if ":" in service_id:  # Remove client_id
                service_id = service_id.split(":")[-1]
            subpath = f"/{workspace}/apps/{service_id}"
            app = PatchedCORSMiddleware(
                RemoteASGIApp(service),
                allow_origins=self.allow_origins or ["*"],
                allow_methods=self.allow_methods or ["*"],
                allow_headers=self.allow_headers or ["*"],
                expose_headers=self.expose_headers or [],
                allow_credentials=True,
            )

            self.store.mount_app(subpath, app, priority=-1)
            logger.info("Mounted ASGI app %s", subpath)

    async def umount_asgi_app(self, service: dict):
        """Unmount the ASGI apps."""
        service = ServiceInfo.model_validate(service)
        if service.type in ["ASGI", "functions"]:
            service_id = service.id
            if ":" in service_id:  # Remove client_id
                service_id = service_id.split(":")[-1]
            subpath = f"/{service.config.workspace}/apps/{service_id}"
            self.store.unmount_app(subpath)
            logger.info("Unmounted ASGI app %s", subpath)
