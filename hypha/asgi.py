"""Support ASGI web server apps."""
from starlette.types import Receive, Scope, Send

from hypha.core import ServiceInfo


class RemoteASGIApp:
    """Wrapper for a remote ASGI app."""

    def __init__(self, service: ServiceInfo) -> None:
        """Initialize the ASGI app."""
        self.service = service
        assert self.service.serve is not None, "No serve function defined"
        # super().__init__(**kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle requests for the ASGI app."""
        scope = {
            k: scope[k]
            for k in scope
            if isinstance(scope[k], (str, int, float, bool, tuple, list, dict, bytes))
        }
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


class ASGIGateway:
    """ASGI gateway for running web servers in the browser apps."""

    def __init__(self, core_interface):
        """Initialize the gateway."""
        self.core_interface = core_interface
        core_interface.event_bus.on("service_registered", self.mount_asgi_app)
        core_interface.event_bus.on("service_unregistered", self.umount_asgi_app)

    def mount_asgi_app(self, service):
        """Mount the ASGI apps from new services."""
        if service.type == "ASGI":
            subpath = f"/{service.config.workspace}/apps/{service.name}"
            self.core_interface.mount_app(subpath, RemoteASGIApp(service), priority=-1)

    def umount_asgi_app(self, service):
        """Unmount the ASGI apps."""
        if service.type == "ASGI":
            subpath = f"/{service.config.workspace}/apps/{service.name}"
            self.core_interface.umount_app(subpath)
