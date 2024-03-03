"""A scalable state store based on Redis."""
import asyncio
import json
import logging
import random
import sys
import time
from typing import Dict, List, Union

import shortuuid
from imjoy_rpc.hypha import RPC
from starlette.routing import Mount

from hypha.core import (
    ClientInfo,
    RedisEventBus,
    RedisRPCConnection,
    ServiceInfo,
    UserInfo,
    VisibilityEnum,
    WorkspaceInfo,
)
from hypha.core.workspace import SERVICE_SUMMARY_FIELD, WorkspaceManager
from hypha.startup import run_startup_function

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("redis-store")
logger.setLevel(logging.INFO)


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
        self.public_base_url = public_base_url
        self.local_base_url = local_base_url
        self._public_services: List[ServiceInfo] = []
        self._ready = False
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
        self._public_workspace_interface = None
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

    def get_event_bus(self):
        """Get the event bus."""
        return self._event_bus

    async def setup_root_user(self) -> UserInfo:
        """Setup the root user."""
        if not self._root_user:
            self._root_user = UserInfo(
                id="root",
                is_anonymous=False,
                email=None,
                parent=None,
                roles=["admin"],
                scopes=[],
                expires_at=None,
            )
            await self.register_user(self._root_user)
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
            assert len(await self._redis.hgetall("public:services")) == 0
        await self._event_bus.init()
        await self.setup_root_user()

        try:
            await self.register_workspace(self._public_workspace, overwrite=False)
        except RuntimeError:
            logger.warning("Public workspace already exists.")
        manager = await self.get_workspace_manager("public")
        self._public_workspace_interface = await manager.get_workspace()
        self._event_bus.on_local(
            "service_registered",
            lambda svc: asyncio.ensure_future(self._register_public_service(svc)),
        )
        self._event_bus.on_local(
            "service_unregistered",
            lambda svc: asyncio.ensure_future(self._unregister_public_service(svc)),
        )
        await self.cleanup_disconnected_clients()
        for service in self._public_services:
            try:
                await self._public_workspace_interface.register_service(service.model_dump())
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to register public service: %s", service)
                raise

        if startup_functions:
            asyncio.create_task(self._run_startup_functions(startup_functions))
        self._ready = True

        self.get_event_bus().emit("startup", target="local")

    async def _register_public_service(self, service: dict):
        """Register the public service."""
        service = ServiceInfo.model_validate(service)
        assert ":" in service.id and service.config.workspace
        # Add public service to the registry
        if (
            service.config.visibility == VisibilityEnum.public
            and not service.id.endswith(":built-in")
        ):
            service_dict = service.model_dump()
            service_summary = {k: service_dict[k] for k in SERVICE_SUMMARY_FIELD}
            if "/" not in service.id:
                service_id = service.config.workspace + "/" + service.id
            else:
                service_id = service.id
            await self._redis.hset(
                "public:services", service_id, json.dumps(service_summary)
            )

    async def _unregister_public_service(self, service: dict):
        service = ServiceInfo.model_validate(service)
        if (
            service.config.visibility == VisibilityEnum.public
            and not service.id.endswith(":built-in")
        ):
            if "/" not in service.id:
                service_id = service.config.workspace + "/" + service.id
            else:
                service_id = service.id
            await self._redis.hdel("public:services", service_id)

    async def add_disconnected_client(self, client_info: ClientInfo):
        """Add a disconnected client."""
        await self._redis.hset(
            "clients:disconnected",
            f"{client_info.workspace}/{client_info.id}",
            json.dumps({"client": client_info.model_dump_json(), "timestamp": time.time()}),
        )

    async def remove_disconnected_client(
        self, full_client_id: str, remove_workspace=False
    ):
        """Remove a disconnected client."""
        assert "/" in full_client_id
        await self._redis.hdel("clients:disconnected", full_client_id)
        if remove_workspace:
            ws, client_id = full_client_id.split("/")
            workspace_manager = await self.get_workspace_manager(ws, setup=False)
            try:
                await workspace_manager.delete_client(client_id)
            except KeyError:
                logger.info("Client already deleted: %s", full_client_id)
            try:
                await workspace_manager.delete()
            except KeyError:
                logger.info("Workspace does not exists: %s", ws)

    async def disconnected_client_exists(self, full_client_id: str):
        """Check if a disconnected client exists."""
        assert "/" in full_client_id
        ret = await self._redis.hexists("clients:disconnected", full_client_id)
        return ret

    async def cleanup_disconnected_clients(self):
        """Cleanup disconnected clients."""
        clients = await self._redis.hgetall("clients:disconnected")
        clients = {k.decode(): json.loads(v.decode()) for k, v in clients.items()}
        # check if the disconnected clients are expired
        for key in clients.keys():
            if clients[key]["timestamp"] + self.disconnect_delay < time.time():
                await self.remove_disconnected_client(key, remove_workspace=True)

    async def register_user(self, user_info: UserInfo):
        """Register a user."""
        await self._redis.hset("users", user_info.id, user_info.model_dump_json())

    async def get_user(self, user_id: str):
        """Get a user."""
        user_info = await self._redis.hget("users", user_id)
        if user_info is None:
            return None
        return UserInfo.model_validate(json.loads(user_info.decode()))

    async def get_user_workspace(self, user_id: str):
        """Get a user."""
        workspace_info = await self._redis.hget("workspaces", user_id)
        if workspace_info is None:
            return None
        workspace_info = WorkspaceInfo.model_validate(json.loads(workspace_info.decode()))
        return workspace_info

    async def get_all_users(self):
        """Get all users."""
        users = await self._redis.hgetall("users")
        return [
            UserInfo.model_validate(json.loads(user.decode())) for user in users.values()
        ]

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

    async def register_workspace(self, workspace: dict, overwrite=False):
        """Add a workspace."""
        workspace = WorkspaceInfo.model_validate(workspace)
        if not overwrite and await self._redis.hexists("workspaces", workspace.name):
            raise RuntimeError(f"Workspace {workspace.name} already exists.")
        if overwrite:
            # clean up the clients and users in the workspace
            client_keys = await self._redis.hkeys(f"{workspace.name}:clients")
            for client_id in client_keys:
                client_id = client_id.decode()
                client_info = await self._redis.hget(
                    f"{workspace.name}:clients", client_id
                )
                if client_info is None:
                    raise KeyError(
                        f"Client does not exist: {workspace.name}/{client_id}"
                    )
                client_info = ClientInfo.model_validate(json.loads(client_info.decode()))
                await self._redis.srem(
                    f"user:{client_info.user_info.id}:clients", client_info.id
                )
                # assert ret >= 1, f"Client not found in user({client_info.user_info.id})'s clients list: {client_info.id}"
                await self._redis.hdel(f"{workspace.name}:clients", client_id)
            await self._redis.delete(f"{workspace}:clients")
        await self._redis.hset("workspaces", workspace.name, workspace.model_dump_json())
        await self.get_workspace_manager(workspace.name, setup=True)

        self._event_bus.emit("workspace_registered", workspace.model_dump())

    async def connect_to_workspace(self, workspace: str, client_id: str):
        """Connect to a workspace."""
        manager = await self.get_workspace_manager(workspace, setup=True)
        rpc = await manager.setup(client_id=client_id)
        wm = await rpc.get_remote_service(workspace + "/workspace-manager:default")
        wm.rpc = rpc
        return wm

    async def get_workspace_manager(self, workspace: str, setup=True):
        """Get a workspace manager."""
        manager = WorkspaceManager.get_manager(
            workspace,
            self._redis,
            await self.setup_root_user(),
            self._event_bus,
            self._server_info,
        )
        if setup:
            await manager.setup()
        return manager

    async def get_workspace_interface(self, workspace: str, context: dict = None):
        """Get the interface of a workspace."""
        manager = WorkspaceManager.get_manager(
            workspace,
            self._redis,
            await self.setup_root_user(),
            self._event_bus,
            self._server_info,
        )
        return await manager.get_workspace(context=context)

    async def list_all_workspaces(self):
        """List all workspaces."""
        workspace_keys = await self._redis.hkeys("workspaces")
        return [k.decode() for k in workspace_keys]

    def create_rpc(
        self, client_id: str, workspace: str, user_info: UserInfo, default_context=None
    ):
        """Create a rpc object for a workspace."""
        assert "/" not in client_id
        logger.info("Creating RPC for client %s", client_id)
        connection = RedisRPCConnection(self._redis, workspace, client_id, user_info)
        rpc = RPC(connection, client_id=client_id, default_context=default_context)
        return rpc

    async def check_permission(
        self, workspace: Union[str, WorkspaceInfo], user_info: UserInfo
    ):
        """Check user permission for a workspace."""
        if isinstance(workspace, WorkspaceInfo):
            workspace = workspace.name
        manager = await self.get_workspace_manager(workspace, setup=False)
        return await manager.check_permission(user_info, workspace)

    async def get_workspace(self, name, load=True):
        """Return the workspace."""
        try:
            manager = await self.get_workspace_manager(name, setup=False)
            return await manager.get_workspace_info(name)
        except KeyError:
            if load and self._workspace_loader:
                try:
                    workspace = await self._workspace_loader(
                        name, await self.setup_root_user()
                    )
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

    async def list_public_services(self, query=None):
        """List all public services."""
        return await self._public_workspace_interface.list_services(query)

    async def get_public_service(self, query=None):
        """Get public service."""
        return await self._public_workspace_interface.get_service(query)

    def get_public_workspace_interface(self):
        """Get public workspace interface."""
        return self._public_workspace_interface

    async def list_services_as_user(self, query, user_info: UserInfo = None):
        """List all services as user."""
        user_info = user_info or await self.setup_root_user()
        wm = await self.get_workspace_manager("public", setup=False)
        services = await wm.list_services(query, context={"user": user_info})
        return services

    async def get_service_as_user(
        self,
        workspace_name: str,
        service_id: str,
        user_info: UserInfo = None,
    ):
        """Get service as a specified user."""
        assert "/" not in service_id
        user_info = user_info or await self.setup_root_user()

        if ":" not in service_id:
            wm = await self.get_workspace_manager(workspace_name)
            services = await wm.list_services(context={"user": user_info})
            services = list(
                filter(
                    lambda service: service["id"].endswith(
                        ":" + service_id if ":" not in service_id else service_id
                    ),
                    services,
                )
            )
            if not services:
                raise KeyError(f"Service {service_id} not found")
            service = random.choice(services)
            service_id = service["id"]

        rpc = self.create_rpc(
            "http-client-" + shortuuid.uuid(), workspace_name, user_info=user_info
        )
        if "/" not in service_id:
            service_id = f"{workspace_name}/{service_id}"
        service = await rpc.get_remote_service(service_id, timeout=5)
        # Patch the workspace name
        service["config"]["workspace"] = workspace_name
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
        self._public_services.append(ServiceInfo.model_validate(formated_service))
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
