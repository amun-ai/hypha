"""Support ASGI web server apps."""
import traceback

from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send

from hypha.core import ServiceInfo
from hypha.utils import PatchedCORSMiddleware


class RemoteASGIApp:
    """Wrapper for a remote ASGI app."""

    def __init__(self, service: ServiceInfo) -> None:
        """Initialize the ASGI app."""
        self.service = service
        assert self.service.type in ["ASGI", "functions"]
        if self.service.type == "ASGI":
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
                "_rintf": True,
            }
            await self.service.serve(interface)
            # clear the object store to avoid gabage collection issue
            # this means the service plugin cannot have extra interface registered
            self.service._provider.dispose_object(interface)
        elif self.service.type == "functions":
            func_name = scope["path"].split("/", 1)[-1] or "index"
            func_name = func_name.rstrip("/")
            service = self.service.dict()
            if func_name in service and callable(service[func_name]):
                scope["query_string"] = scope["query_string"].decode("utf-8")
                scope["raw_path"] = scope["raw_path"].decode("latin-1")
                scope["headers"] = dict(Headers(scope=scope).items())
                event = await receive()
                body = event["body"]
                while event.get("more_body"):
                    body += await receive()["body"]
                scope["body"] = body or None
                func = service[func_name]
                context = {}
                try:
                    result = await func(scope, context)
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
        core_interface,
        allow_origins=None,
        allow_methods=None,
        allow_headers=None,
        expose_headers=None,
    ):
        """Initialize the gateway."""
        self.core_interface = core_interface
        self.allow_origins = allow_origins
        self.allow_methods = allow_methods
        self.allow_headers = allow_headers
        self.expose_headers = expose_headers
        core_interface.event_bus.on("service_registered", self.mount_asgi_app)
        core_interface.event_bus.on("service_unregistered", self.umount_asgi_app)

    def mount_asgi_app(self, service):
        """Mount the ASGI apps from new services."""
        if service.type in ["ASGI", "functions"]:
            subpath = f"/{service.config.workspace}/apps/{service.name}"
            app = PatchedCORSMiddleware(
                RemoteASGIApp(service),
                allow_origins=self.allow_origins or ["*"],
                allow_methods=self.allow_methods or ["*"],
                allow_headers=self.allow_headers or ["*"],
                expose_headers=self.expose_headers or [],
                allow_credentials=True,
            )

            self.core_interface.mount_app(subpath, app, priority=-1)

    def umount_asgi_app(self, service):
        """Unmount the ASGI apps."""
        if service.type in ["ASGI", "functions"]:
            subpath = f"/{service.config.workspace}/apps/{service.name}"
            self.core_interface.umount_app(subpath)
