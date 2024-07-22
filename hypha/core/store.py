"""A scalable state store based on Redis."""
import asyncio
import json
import logging
import random
import sys
import time
from typing import Dict, List, Union
from pydantic import BaseModel

import shortuuid
from hypha_rpc import RPC
from starlette.routing import Mount

from hypha.core import (
    RedisEventBus,
    RedisRPCConnection,
    ServiceInfo,
    UserInfo,
    WorkspaceInfo,
)
from hypha.core.workspace import WorkspaceManager
from hypha.startup import run_startup_function

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("redis-store")
logger.setLevel(logging.INFO)


class WorkspaceInterfaceContextManager:
    """Workspace interface context manager."""

    def __init__(self, rpc, timeout=10):
        self.rpc = rpc
        self.timeout = timeout
        self.wm = None

    async def __aenter__(self):
        return await self._get_workspace_manager()

    async def __aexit__(self, exc_type, exc, tb):
        await self.wm.disconnect()

    def __await__(self):
        return self._get_workspace_manager().__await__()

    async def _get_workspace_manager(self):
        self.wm = await self.rpc.get_manager_service(self.timeout)
        self.wm.rpc = self.rpc
        self.wm.disconnect = self.rpc.disconnect
        self.wm.register_codec = self.rpc.register_codec
        return self.wm

    async def __aexit__(self, exc_type, exc, tb):
        await self.rpc.disconnect()


class RedisStore:
    """Represent a redis store."""

    # pylint: disable=no-self-use, too-many-instance-attributes, too-many-public-methods

    def __init__(
        self,
        app,
        public_base_url=None,
        local_base_url=None,
        redis_uri=None,
        disconnect_delay=30,
    ):
        """Initialize the redis store."""
        self._all_users: Dict[str, UserInfo] = {}  # uid:user_info
        self._all_workspaces: Dict[str, WorkspaceInfo] = {}  # wid:workspace_info
        self._workspace_loader = None
        self._app = app
        self.disconnect_delay = disconnect_delay
        self._codecs = {}
        self._disconnected_plugins = []
        self._public_workspace_interface = None
        self.public_base_url = public_base_url
        self.local_base_url = local_base_url
        self._public_services: List[ServiceInfo] = []
        self._ready = False
        self._workspace_manager = None
        self.manager_id = None
        self._public_workspace = WorkspaceInfo.model_validate(
            {
                "name": "public",
                "persistent": True,
                "owners": ["root"],
                "allow_list": [],
                "deny_list": [],
                "visibility": "public",
                "read_only": True,
            }
        )
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
            scopes=[],
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
            await self.register_workspace(self._public_workspace, overwrite=False)
        except RuntimeError:
            logger.warning("Public workspace already exists.")

        api = await self.get_public_api()
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
            asyncio.create_task(self._run_startup_functions(startup_functions))
        self._ready = True

        await self.get_event_bus().emit_local("startup")

    async def get_public_api(self):
        """Get the public API."""
        if self._public_workspace_interface is None:
            self._public_workspace_interface = await self.get_workspace_interface(
                "public", self._root_user
            )
        return self._public_workspace_interface

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
    ):
        """Add a workspace."""
        return await self._workspace_manager.create_workspace(
            workspace_info,
            overwrite=overwrite,
            context={"user": self._root_user, "ws": "public"},
        )

    async def delete_client(self, workspace: str, client_id: str, user: UserInfo):
        """Delete a client."""
        return await self._workspace_manager.delete_client(
            client_id, context={"user": user.model_dump(), "ws": workspace}
        )

    async def check_permission(
        self, user: UserInfo, workspace: Union[str, WorkspaceInfo]
    ):
        """Check permission."""
        return await self._workspace_manager.check_permission(user, workspace)

    def connect_to_workspace(
        self, workspace: str, client_id: str, user_info: UserInfo = None, timeout=10
    ):
        """Connect to a workspace."""
        # user get_workspace_interface
        user_info = user_info or self._root_user
        return self.get_workspace_interface(
            workspace, user_info, client_id=client_id, timeout=timeout
        )

    async def register_workspace_manager(self):
        """Register a workspace manager."""
        manager = WorkspaceManager(
            self._redis,
            self._root_user,
            self._event_bus,
            self._server_info,
        )
        await manager.setup()
        return manager

    def get_workspace_interface(
        self, workspace: str, user_info: UserInfo, client_id=None, timeout=10
    ):
        """Get the interface of a workspace."""
        assert workspace, "Workspace name is required"
        assert user_info and isinstance(user_info, UserInfo), "User info is required"
        # the client will be hidden if client_id is None
        silent = client_id is None
        client_id = client_id or "workspace-client-" + shortuuid.uuid()
        rpc = self.create_rpc(workspace, user_info, client_id=client_id, silent=silent)
        return WorkspaceInterfaceContextManager(rpc, timeout=timeout)

    async def client_exists(self, workspace: str, client_id: str):
        """Check if a client exists."""
        pattern = f"services:*:{workspace}/{client_id}:built-in@*"
        keys = await self._redis.keys(pattern)
        return bool(keys)

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
        client_id = client_id or "anonymous-client-" + shortuuid.uuid()
        assert "/" not in client_id
        logger.info("Creating RPC for client %s", client_id)
        assert user_info is not None, "User info is required"
        connection = RedisRPCConnection(
            self._event_bus, workspace, client_id, user_info
        )
        rpc = RPC(
            connection,
            client_id=client_id,
            default_context=default_context,
            manager_id=self.manager_id,
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

    async def get_workspace(self, name, load=True):
        """Return the workspace information."""
        try:
            workspace_info = await self._redis.hget("workspaces", name)
            if workspace_info is None:
                raise KeyError(f"Workspace not found: {name}")
            workspace_info = WorkspaceInfo.model_validate(
                json.loads(workspace_info.decode())
            )
            return workspace_info
        except KeyError:
            if load and self._workspace_loader:
                try:
                    # TODO: register the loaded workspace
                    workspace = await self._workspace_loader(name, self._root_user)
                    if workspace:
                        self._all_workspaces[workspace.name] = workspace
                except Exception:  # pylint: disable=broad-except
                    logger.exception("Failed to load workspace %s", name)
                else:
                    return workspace
        return None

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

        if "name" not in service or "type" not in service:
            raise Exception("Service should at least contain `name` and `type`")

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
