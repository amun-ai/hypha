"""A scalable state store based on Redis."""
import asyncio
import json
import logging
import sys
from typing import List, Union
from pydantic import BaseModel

from hypha_rpc import RPC
from starlette.routing import Mount

from hypha.core import (
    RedisEventBus,
    RedisRPCConnection,
    ServiceInfo,
    ServiceTypeInfo,
    UserInfo,
    WorkspaceInfo,
)
from hypha.core.auth import create_scope
from hypha.core.workspace import WorkspaceManager
from hypha.startup import run_startup_function
from hypha.utils import random_id

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("redis-store")
logger.setLevel(logging.INFO)


class WorkspaceInterfaceContextManager:
    """Workspace interface context manager."""

    def __init__(self, rpc, redis, workspace, timeout=10):
        self._rpc = rpc
        self._timeout = timeout
        self._wm = None
        self._redis = redis
        self._workspace = workspace

    async def __aenter__(self):
        return await self._get_workspace_manager()

    async def __aexit__(self, exc_type, exc, tb):
        await self._wm.disconnect()

    def __await__(self):
        return self._get_workspace_manager().__await__()

    async def _get_workspace_manager(self):
        # Check if workspace exists
        if not await self._redis.hexists("workspaces", self._workspace):
            raise KeyError(f"Workspace {self._workspace} does not exist")
        self._wm = await self._rpc.get_manager_service(self._timeout)
        self._wm.rpc = self._rpc
        self._wm.disconnect = self._rpc.disconnect
        self._wm.register_codec = self._rpc.register_codec
        self._wm.register_service = self._rpc.register_service
        return self._wm

    async def __aexit__(self, exc_type, exc, tb):
        await self._rpc.disconnect()


class RedisStore:
    """Represent a redis store."""

    # pylint: disable=no-self-use, too-many-instance-attributes, too-many-public-methods

    def __init__(
        self,
        app,
        public_base_url=None,
        local_base_url=None,
        redis_uri=None,
        reconnection_token_life_time=2 * 24 * 60 * 60,
    ):
        """Initialize the redis store."""
        self._workspace_loader = None
        self._app = app
        self._codecs = {}
        self._disconnected_plugins = []
        self._public_workspace_interface = None
        self._root_workspace_interface = None
        self.public_base_url = public_base_url
        self.local_base_url = local_base_url
        self._public_services: List[ServiceInfo] = []
        self._public_types: List[ServiceTypeInfo] = []
        self._ready = False
        self._workspace_manager = None
        self._websocket_server = None
        self.manager_id = None
        self.reconnection_token_life_time = reconnection_token_life_time
        self._server_info = {
            "public_base_url": self.public_base_url,
            "local_base_url": self.local_base_url,
        }

        if redis_uri and redis_uri.startswith("redis://"):
            from redis import asyncio as aioredis

            self._redis = aioredis.from_url(redis_uri)
        else:  #  Create a redis server with fakeredis
            from fakeredis import aioredis

            self._redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")

        self._root_user = None
        self._event_bus = RedisEventBus(self._redis)

    def set_websocket_server(self, websocket_server):
        """Set the websocket server."""
        assert self._websocket_server is None, "Websocket server already set"
        self._websocket_server = websocket_server

    def kickout_client(self, workspace, client_id, code, reason):
        """Force disconnect a client."""
        return self._websocket_server.force_disconnect(
            workspace, client_id, code, reason
        )

    def get_redis(self):
        return self._redis

    def get_event_bus(self):
        """Get the event bus."""
        return self._event_bus

    async def setup_root_user(self) -> UserInfo:
        """Setup the root user."""
        self._root_user = UserInfo(
            id="root",
            is_anonymous=False,
            email=None,
            parent=None,
            roles=["admin"],
            scope=create_scope("*#a"),
            expires_at=None,
        )
        return self._root_user

    def get_root_user(self):
        """Get the root user."""
        return self._root_user

    async def _run_startup_functions(self, startup_functions):
        """Run the startup functions in the background."""
        try:
            if startup_functions:
                for startup_function in startup_functions:
                    logger.info(f"Running startup function: {startup_function}")
                    await asyncio.sleep(0)
                    await run_startup_function(self, startup_function)
        except Exception as e:
            logger.exception(f"Error running startup function: {e}")
            # Stop the entire event loop if an error occurs
            asyncio.get_running_loop().stop()

    async def init(self, reset_redis, startup_functions=None):
        """Setup the store."""
        if reset_redis:
            logger.warning("RESETTING ALL REDIS DATA!!!")
            await self._redis.flushall()
        await self._event_bus.init()
        await self.setup_root_user()
        self._workspace_manager = await self.register_workspace_manager()

        self.manager_id = self._workspace_manager.get_client_id()
        try:
            await self.register_workspace(
                WorkspaceInfo.model_validate(
                    {
                        "name": "root",
                        "description": "Root workspace",
                        "persistent": True,
                        "owners": ["root"],
                        "read_only": False,
                    }
                ),
                overwrite=False,
            )
        except RuntimeError:
            logger.warning("Root workspace already exists.")

        try:
            await self.register_workspace(
                WorkspaceInfo.model_validate(
                    {
                        "name": "public",
                        "description": "Public workspace",
                        "persistent": True,
                        "owners": ["root"],
                        "read_only": True,
                    }
                ),
                overwrite=False,
            )
        except RuntimeError:
            logger.warning("Public workspace already exists.")
        await self._register_root_services()
        api = await self.get_public_api()
        for service_type in self._public_types:
            try:
                await api.register_service_type(
                    service_type.model_dump(),
                )
            except Exception:
                logger.exception(
                    "Failed to register public service type: %s", service_type
                )
                raise
        for service in self._public_services:
            try:
                await api.register_service(
                    service.model_dump(),
                    notify=True,
                )
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to register public service: %s", service)
                raise

        if startup_functions:
            await self._run_startup_functions(startup_functions)
        self._ready = True
        await self.get_event_bus().emit_local("startup")

    async def _register_root_services(self):
        """Register root services."""
        self._root_workspace_interface = await self.get_workspace_interface(
            "root", self._root_user
        )
        await self._root_workspace_interface.register_service(
            {
                "id": "admin-utils",
                "name": "Admin Utilities",
                "config": {
                    "visibility": "protected",
                    "require_context": False,
                },
                "kickout_client": self.kickout_client,
            }
        )

    async def get_public_api(self):
        """Get the public API."""
        if self._public_workspace_interface is None:
            self._public_workspace_interface = await self.get_workspace_interface(
                "public", self._root_user
            )
        return self._public_workspace_interface

    async def client_exists(self, client_id: str, workspace: str = None):
        """Check if a client exists."""
        assert workspace is not None, "Workspace must be provided."
        assert client_id and "/" not in client_id, "Invalid client id: " + client_id
        pattern = f"services:*:{workspace}/{client_id}:built-in@*"
        keys = await self._redis.keys(pattern)
        return bool(keys)

    async def remove_client(self, client_id, workspace, user_info):
        """Remove a client."""
        context = {"user": self._root_user.model_dump(), "ws": workspace}
        return await self._workspace_manager.delete_client(
            client_id, workspace, user_info, context=context
        )

    async def get_user_workspace(self, user_id: str):
        """Get a user."""
        workspace_info = await self._redis.hget("workspaces", user_id)
        if workspace_info is None:
            return None
        workspace_info = WorkspaceInfo.model_validate(
            json.loads(workspace_info.decode())
        )
        return workspace_info

    async def get_all_workspace(self):
        """Get all workspaces."""
        workspaces = await self._redis.hgetall("workspaces")
        return [
            WorkspaceInfo.model_validate(json.loads(v.decode()))
            for k, v in workspaces.items()
        ]

    async def workspace_exists(self, workspace_name: str):
        """Check if a workspace exists."""
        return await self._redis.hexists("workspaces", workspace_name)

    async def register_workspace(
        self, workspace_info: Union[dict, WorkspaceInfo], overwrite=False
    ) -> WorkspaceInfo:
        """Add a workspace."""
        info = await self._workspace_manager.create_workspace(
            workspace_info,
            overwrite=overwrite,
            context={"user": self._root_user, "ws": "public"},
        )
        return WorkspaceInfo.model_validate(info)

    def connect_to_workspace(
        self,
        workspace: str,
        client_id: str,
        user_info: UserInfo = None,
        timeout=10,
        silent=False,
    ):
        """Connect to a workspace."""
        # user get_workspace_interface
        user_info = user_info or self._root_user
        return self.get_workspace_interface(
            workspace, user_info, client_id=client_id, timeout=timeout, silent=silent
        )

    async def register_workspace_manager(self):
        """Register a workspace manager."""
        manager = WorkspaceManager(
            self._redis,
            self._root_user,
            self._event_bus,
            self._server_info,
            self._workspace_loader,
        )
        await manager.setup()
        return manager

    def get_workspace_interface(
        self,
        workspace: str,
        user_info: UserInfo,
        client_id=None,
        timeout=10,
        silent=False,
    ):
        """Get the interface of a workspace."""
        assert workspace, "Workspace name is required"
        assert user_info and isinstance(user_info, UserInfo), "User info is required"
        # the client will be hidden if client_id is None
        if silent is None:
            silent = client_id is None
        client_id = client_id or "workspace-client-" + random_id(readable=False)
        rpc = self.create_rpc(workspace, user_info, client_id=client_id, silent=silent)
        return WorkspaceInterfaceContextManager(
            rpc, self._redis, workspace, timeout=timeout
        )

    async def list_all_workspaces(self):
        """List all workspaces."""
        workspace_keys = await self._redis.hkeys("workspaces")
        return [k.decode() for k in workspace_keys]

    def create_rpc(
        self,
        workspace: str,
        user_info: UserInfo,
        client_id: str = None,
        default_context=None,
        silent=True,
    ):
        """Create a rpc object for a workspace."""
        client_id = client_id or "anonymous-client-" + random_id(readable=False)
        assert "/" not in client_id
        logger.info("Creating RPC for client %s", client_id)
        assert user_info is not None, "User info is required"
        connection = RedisRPCConnection(
            self._event_bus, workspace, client_id, user_info, manager_id=self.manager_id
        )
        rpc = RPC(
            connection,
            client_id=client_id,
            default_context=default_context,
            workspace=workspace,
            silent=silent,
        )
        rpc.register_codec(
            {
                "name": "pydantic-model",
                "type": BaseModel,
                "encoder": lambda x: x.model_dump(),
            }
        )
        return rpc

    async def get_workspace(self, workspace: str, load: bool = False):
        """Return the workspace information."""
        try:
            return await self._workspace_manager.load_workspace_info(
                workspace, load=load
            )
        except KeyError:
            return None

    async def set_workspace(
        self, workspace: WorkspaceInfo, user_info: UserInfo, overwrite: bool = False
    ):
        """Set the workspace information."""
        return await self._workspace_manager._update_workspace(
            workspace, user_info, overwrite=overwrite
        )

    def set_workspace_loader(self, loader):
        """Set the workspace loader."""
        self._workspace_loader = loader

    async def get_service_as_user(
        self,
        workspace_name: str,
        service_id: str,
        user_info: UserInfo = None,
    ):
        """Get service as a specified user."""
        user_info = user_info or self._root_user
        if "/" in service_id:
            assert (
                service_id.split("/")[0] == workspace_name
            ), f"Workspace name mismatch, {workspace_name} != {service_id.split('/')[0]}"
            service_id = service_id.split("/")[1]
        if ":" not in service_id:
            service_id = "*:" + service_id
        service_id = workspace_name + "/" + service_id
        wm = await self.get_workspace_interface("public", user_info)
        service = await wm.get_service(service_id)
        await wm.disconnect()
        return service

    def register_public_service(self, service: dict):
        """Register a service."""
        assert not self._ready, "Cannot register public service after ready"

        if "name" not in service or "id" not in service:
            raise Exception("Service should at least contain `name` and `id`")

        # TODO: check if it's already exists
        service["config"] = service.get("config", {})
        assert isinstance(
            service["config"], dict
        ), "service.config must be a dictionary"
        service["config"]["workspace"] = "public"
        assert (
            "visibility" not in service
        ), "`visibility` should be placed inside `config`"
        assert (
            "require_context" not in service
        ), "`require_context` should be placed inside `config`"
        formated_service = ServiceInfo.model_validate(service)
        # Note: service can set its `visibility` to `public` or `protected`
        self._public_services.append(formated_service)
        return {
            "id": formated_service.id,
            "workspace": "public",
            "name": formated_service.name,
        }

    def register_public_type(self, service_type: dict):
        """Register a service type."""
        assert not self._ready, "Cannot register public service after ready"

        if "name" not in service_type or "id" not in service_type:
            raise Exception("Service should at least contain `name` and `id`")
        assert "definition" in service_type, "Service type should contain `definition`"
        formatted_service_type = ServiceTypeInfo.model_validate(service_type)
        self._public_types.append(formatted_service_type)

    def is_ready(self):
        """Check if the server is alive."""
        return self._ready

    def register_router(self, router):
        """Register a router."""
        self._app.include_router(router)

    def mount_app(self, path, app, name=None, priority=-1):
        """Mount an app to fastapi."""
        route = Mount(path, app, name=name)
        # remove existing path
        routes_remove = [route for route in self._app.routes if route.path == path]
        for rou in routes_remove:
            self._app.routes.remove(rou)
        # The default priority is -1 which assumes the last one is websocket
        self._app.routes.insert(priority, route)

    def unmount_app(self, path):
        """Unmount an app to fastapi."""
        routes_remove = [route for route in self._app.routes if route.path == path]
        for route in routes_remove:
            self._app.routes.remove(route)

    def teardown(self):
        """Teardown the server."""
        pass
