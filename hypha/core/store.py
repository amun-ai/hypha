"""A scalable state store based on Redis."""
import asyncio
import inspect
import json
import logging
import os
import sys
import io
import tempfile
from typing import Callable, Union, Optional
from hypha.utils import EventBus

import aioredis
import msgpack
import shortuuid
from redislite import Redis
import random

from hypha.core import (
    ClientInfo,
    ServiceInfo,
    UserInfo,
    WorkspaceInfo,
    VisibilityEnum,
    TokenConfig,
)
from hypha.core.rpc import RPC
from hypha.core.auth import generate_presigned_token

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("redis-store")
logger.setLevel(logging.INFO)


class RedisRPCConnection:
    """Represent a redis connection."""

    def __init__(
        self, redis: aioredis.Redis, workspace: str, client_id: str, user_info: UserInfo
    ):
        self._redis = redis
        self._handle_message = None
        self._workspace = workspace
        self._client_id = client_id
        assert "/" not in client_id
        self._user_info = user_info.dict()

    def on_message(self, handler: Callable):
        self._handle_message = handler
        self._is_async = inspect.iscoroutinefunction(handler)
        asyncio.ensure_future(self._subscribe_redis())

    async def emit_message(self, data: Union[dict, bytes]):
        assert self._handle_message is not None, "No handler for message"
        assert isinstance(data, bytes)
        unpacker = msgpack.Unpacker(io.BytesIO(data))
        message = unpacker.unpack()  # Only unpack the main message
        pos = unpacker.tell()
        target_id = message["to"]
        if "/" not in target_id:
            target_id = self._workspace + "/" + target_id
        source_id = self._workspace + "/" + self._client_id

        message.update(
            {
                "to": target_id,
                "from": source_id,
                "user": self._user_info,
            }
        )
        # Pack more context info to the package
        data = msgpack.packb(message) + data[pos:]
        await self._redis.publish(f"{target_id}:msg", data)

    async def disconnect(self, reason=None):
        pass

    async def _subscribe_redis(self):
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(f"{self._workspace}/{self._client_id}:msg")
        while True:
            msg = await pubsub.get_message()
            if msg and msg.get("type") == "message":
                assert (
                    msg.get("channel")
                    == f"{self._workspace}/{self._client_id}:msg".encode()
                )
                if self._is_async:
                    await self._handle_message(msg["data"])
                else:
                    self._handle_message(msg["data"])


class WorkspaceManager:

    _managers = {}

    @staticmethod
    def get_manager(
        workspace: str, redis: Redis, root_user: UserInfo, event_bus: EventBus
    ):
        if workspace in WorkspaceManager._managers:
            return WorkspaceManager._managers[workspace]
        else:
            return WorkspaceManager(workspace, redis, root_user, event_bus)

    def __init__(
        self, workspace: str, redis: Redis, root_user: UserInfo, event_bus: EventBus
    ):
        self._redis = redis
        self._workspace = workspace
        self._initialized = False
        self._rpc = None
        self._root_user = root_user
        self._event_bus = event_bus
        WorkspaceManager._managers[workspace] = self

    async def setup(
        self,
        client_id="workspace-manager",
        service_id="default",
        service_name="Default workspace management service",
    ):
        if self._rpc and client_id == self._rpc._client_id:
            return self._rpc
        await self._get_workspace_info()
        if await self._redis.hexists(f"{self._workspace}:clients", client_id):
            return self._rpc
        # if await self.check_client_exists(client_id):
        #     raise Exception(f"Another workspace-manager already exists")

        # Register an client as root
        await self.register_client(
            ClientInfo(
                id=client_id,
                workspace=self._workspace,
                user_info=self._root_user.dict(),
            )
        )
        rpc = await self.create_rpc(client_id)

        def save_client_info(_):
            return asyncio.create_task(
                self.update_client_info(
                    rpc.get_client_info(),
                    context={
                        "from": self._workspace + "/" + client_id,
                        "user": self._root_user.dict(),
                    },
                )
            )

        rpc.on("service-updated", save_client_info)
        management_service = self.create_service(service_id, service_name)
        await rpc.register_service(management_service, notify=False)
        await save_client_info(None)
        self._rpc = rpc
        self._initialized = True
        return rpc

    async def get_rpc(self, workspace=None):
        if not self._rpc:
            await self.setup()
        return self._rpc

    async def create_workspace(
        self, config: dict, overwrite=False, context: Optional[dict] = None
    ):
        """Create a new workspace."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        user_info = UserInfo.parse_obj(context["user"])
        config["persistent"] = config.get("persistent") or False
        if user_info.is_anonymous and config["persistent"]:
            raise Exception("Only registered user can create persistent workspace.")
        workspace = WorkspaceInfo.parse_obj(config)
        # workspace.set_global_event_bus(self._event_bus)
        # make sure we add the user's email to owners
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]
        user_info.scopes.append(workspace.name)

        if not overwrite and await self._redis.hexists("workspaces", workspace.name):
            raise Exception(f"Workspace {workspace.name} already exists.")
        await self._redis.hset("workspaces", workspace.name, workspace.json())
        # Clear the workspace
        await self._redis.delete(f"{workspace.name}:clients")
        return await self.get_workspace(workspace.name, context=context)
        # Remove the client if it's not responding
        # for client_id in await manager.list_clients(context=context):
        #     try:
        #         await rpc.ping(client_id)
        #     except asyncio.exceptions.TimeoutError:
        #         await manager.delete_client(client_id)

    async def register_service(self, service, context: Optional[dict] = None, **kwargs):
        """Register a service"""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        rpc = await self.get_rpc()
        sv = await rpc.get_remote_service(context["from"] + ":built-in")
        service["config"] = service.get("config", {})
        service["config"]["workspace"] = self._workspace
        service = await sv.register_service(service, **kwargs)
        source_workspace = context["from"].split("/")[0]
        assert (
            source_workspace == self._workspace
        ), "Service must be registered in the same workspace"
        assert "/" not in service["id"], "Service id must not contain '/'"
        service["id"] = self._workspace + "/" + service["id"]
        return service

    async def generate_token(
        self, config: Optional[dict] = None, context: Optional[dict] = None
    ):
        """Generate a token for the current workspace."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        user_info = UserInfo.parse_obj(context["user"])
        config = config or {}
        if "scopes" in config and config["scopes"] != [self._workspace]:
            raise Exception("Scopes must be empty or contains only the workspace name.")
        config["scopes"] = [self._workspace]
        token_config = TokenConfig.parse_obj(config)
        for ws in config["scopes"]:
            if not await self.check_permission(user_info, ws):
                raise PermissionError(f"User has no permission to workspace: {ws}")
        token = generate_presigned_token(user_info, token_config)
        return token

    async def update_client_info(self, client_info, context=None):
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        ws, client_id = context["from"].split("/")
        assert client_info["id"] == client_id
        assert ws == self._workspace
        client_info["user_info"] = user_info
        client_info["id"] = client_id
        client_info["workspace"] = self._workspace
        await self.register_client(ClientInfo.parse_obj(client_info), overwrite=True)

    async def get_client_info(self, client_id):
        assert "/" not in client_id
        client_info = await self._redis.hget(f"{self._workspace}:clients", client_id)
        assert (
            client_info is not None
        ), f"Failed to get client info for {self._workspace}/{client_id}"
        return client_info and json.loads(client_info.decode())

    async def list_clients(self, context: Optional[dict] = None):
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        client_keys = await self._redis.hkeys(f"{self._workspace}:clients")
        return [k.decode() for k in client_keys]

    async def list_services(
        self, query: Optional[dict] = None, context: Optional[dict] = None
    ):
        """Return a list of services based on the query."""
        assert context is not None
        # if workspace is not set, then it means current workspace
        # if workspace = *, it means search globally
        # otherwise, it search the specified workspace
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        if query is None:
            query = {}

        ws = query.get("workspace")
        if ws:
            del query["workspace"]
        if ws == "*":
            ret = []
            for workspace in await self.get_all_workspace():

                if not await self.check_permission(user_info, workspace):
                    continue

                ws = await self.get_workspace(workspace.name, context=context)
                for service in await ws.list_services():
                    # To access the service, it should be public or owned by the user
                    if service.config["visibility"] != "public":
                        continue
                    match = True
                    for key in query:
                        if getattr(service, key) != query[key]:
                            match = False
                    if match:
                        if "/" not in service.id:
                            service.id = self._workspace + "/" + service.id
                        ret.append(service.get_summary())
            return ret
        if ws is None:
            services = await self.get_services(context=context)
        else:
            ws = await self.get_workspace(ws, context=context)
            services = await ws.list_services()

        ret = []

        for service in services:
            match = True
            for key in query:
                if getattr(service, key) != query[key]:
                    match = False
            if match:
                if "/" not in service.id:
                    service.id = self._workspace + "/" + service.id
                ret.append(service.get_summary())

        return ret

    async def get_services(self, context=None):
        service_list = []
        client_ids = await self.list_clients(context=context)
        for client_id in client_ids:
            try:
                client_info = await self.get_client_info(client_id)
                for sinfo in client_info["services"]:
                    service_list.append(ServiceInfo.parse_obj(sinfo))
            # make sure we don't throw if a client is removed
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to get service info for client: %s", client_id)
        return service_list

    async def register_client(self, client_info: ClientInfo, overwrite: bool = False):
        """Add a client."""
        assert "/" not in client_info.id
        if not overwrite and await self._redis.hexists(
            f"{self._workspace}:clients", client_info.id
        ):
            raise KeyError(
                "Client already exists: %s/%s", self._workspace, client_info.id
            )

        await self._redis.hset(
            f"{self._workspace}:clients", client_info.id, client_info.json()
        )
        self._event_bus.emit("client_registered", client_info)

    async def check_client_exists(self, client_id: str, workspace: str = None):
        """Check if a client exists."""
        assert "/" not in client_id
        client_id = client_id
        workspace = workspace or self._workspace
        return await self._redis.hexists(f"{workspace}:clients", client_id)

    async def delete_client(self, client_id: str):
        """Delete a client."""
        # assert "/" not in client_id
        await self._redis.hdel(f"{self._workspace}:clients", client_id)

    async def create_rpc(self, client_id: str, default_context=None):
        """Create a rpc for the workspace."""
        assert "/" not in client_id
        logger.info("Creating RPC for client %s", client_id)
        connection = RedisRPCConnection(
            self._redis, self._workspace, client_id, self._root_user
        )
        rpc = RPC(connection, client_id=client_id, default_context=default_context)
        return rpc

    async def log(self, msg, context=None):
        """Log a plugin message."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        logger.info("%s: %s", context["from"], msg)

    async def info(self, msg, context=None):
        """Log a plugin message."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        logger.info("%s: %s", context["from"], msg)

    async def warning(self, msg, context=None):
        """Log a plugin message (warning)."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        logger.warning("WARNING: %s: %s", context["from"], msg)

    async def error(self, msg, context=None):
        """Log a plugin error message (error)."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        logger.error("%s: %s", context["from"], msg)

    async def critical(self, msg, context=None):
        """Log a plugin error message (critical)."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        logger.critical("%s: %s", context["from"], msg)

    async def _get_workspace_info(self, workspace: str = None):
        """Get info of the current workspace."""
        workspace = workspace or self._workspace
        workspace_info = await self._redis.hget("workspaces", workspace)
        if workspace_info is None:
            raise KeyError(f"Workspace not found: {workspace}")
        workspace_info = WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))
        return workspace_info

    async def get_service(self, query, context=None):
        # Note: No authorization required because the access is controlled by the client itself
        if isinstance(query, str):
            if ":" in query:
                query = {"id": query}
            else:
                query = {"name": query}

        if "id" in query:
            if "/" in query["id"]:
                workspace = query["id"].split("/")[0]
                if workspace != self._workspace:

                    ws = await self.get_workspace(workspace)
                    return await ws.get_service(query["id"], context=context)
            if "/" not in query["id"]:
                query["id"] = self._workspace + "/" + query["id"]
            rpc = await self.setup()
            service_api = await rpc.get_remote_service(query["id"])
        elif "name" in query:
            if "workspace" in query:
                assert await self._redis.hget("workspaces", query["workspace"])
                ws = await self.get_workspace(query["workspace"])
                services = await ws.list_services()
            else:
                services = await self.list_services(context=context)
            services = list(
                filter(lambda service: service["name"] == query["name"], services)
            )
            if not services:
                raise Exception(f"Service not found: {query['name']}")
            service_info = random.choice(services)
            rpc = await self.setup()
            service_api = await rpc.get_remote_service(service_info["id"])
        else:
            raise Exception("Please specify the service id or name to get the service")
        return service_api

    async def get_all_workspace(self):
        """Get all workspaces."""
        workspaces = await self._redis.hgetall("workspaces")
        return [
            WorkspaceInfo.parse_obj(json.loads(v.decode()))
            for k, v in workspaces.items()
        ]

    async def check_permission(
        self, user_info: UserInfo, workspace: Union[WorkspaceInfo, str] = None
    ):
        """Check user permission for the workspace."""
        # pylint: disable=too-many-return-statements
        if workspace is None:
            workspace = self._workspace
        if isinstance(workspace, str):
            workspace = await self._get_workspace_info(workspace)

        # Make exceptions for root user, the children of root and test workspace
        if (
            user_info.id == "root"
            or user_info.parent == "root"
            or workspace.name == "public"
        ):
            return True

        if workspace.name == user_info.id:
            return True

        if user_info.parent:
            parent = await self._redis.hget(f"users", user_info.parent)
            if not parent:
                return False
            parent = UserInfo.parse_obj(json.loads(parent.decode()))
            if not await self.check_permission(parent):
                return False
            # if the parent has access
            # and the workspace is in the scopes
            # then we allow the access
            if workspace.name in user_info.scopes:
                return True

        _id = user_info.email or user_info.id

        if _id in workspace.owners:
            return True

        if workspace.visibility == VisibilityEnum.public:
            if workspace.deny_list and user_info.email not in workspace.deny_list:
                return True
        elif workspace.visibility == VisibilityEnum.protected:
            if workspace.allow_list and user_info.email in workspace.allow_list:
                return True

        if "admin" in user_info.roles:
            logger.info(
                "Allowing access to %s for admin user %s", workspace.name, user_info.id
            )
            return True

        return False

    async def get_workspace(self, workspace: str = None, context=None):
        """Get the service api of the workspace manager."""
        if not workspace:
            workspace = self._workspace
        if context:
            user_info = UserInfo.parse_obj(context["user"])
            if not await self.check_permission(user_info, workspace):
                raise Exception(f"Permission denied for workspace {self._workspace}.")

        if await self.check_client_exists("workspace-manager", workspace):
            await self.setup()
            rpc = await self.get_rpc()
            try:
                wm = await rpc.get_remote_service(
                    workspace + "/workspace-manager:default", timeout=5
                )
                return wm
            except asyncio.exceptions.TimeoutError:
                logger.info(
                    "Failed to get the workspace manager service of %s", workspace
                )

        manager = WorkspaceManager.get_manager(
            workspace, self._redis, self._root_user, self._event_bus
        )
        await manager.setup()
        return await manager.get_workspace()

    async def _update_workspace(self, config: dict, context=None):
        """Update the workspace config."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise PermissionError(f"Permission denied for workspace {self._workspace}")

        workspace_info = await self._redis.hget("workspaces", self._workspace)
        workspace = WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))
        if "name" in config:
            raise Exception("Changing workspace name is not allowed.")

        # make sure all the keys are valid
        # TODO: verify the type
        for key in config:
            if key.startswith("_") or not hasattr(workspace, key):
                raise KeyError(f"Invalid key: {key}")

        for key in config:
            if not key.startswith("_") and hasattr(workspace, key):
                setattr(workspace, key, config[key])
        # make sure we add the user's email to owners
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]
        await self._redis.hset("workspaces", workspace.name, workspace.json())
        self._event_bus.emit("workspace_changed", workspace)

    def create_service(self, service_id, service_name=None):
        interface = {
            "id": service_id,
            "name": service_name or service_id,
            # Note: We make these services public by default, and assuming we will do authorization in each function
            "config": {
                "require_context": True,
                "workspace": self._workspace,
                "visibility": "public",
            },
            "log": self.log,
            "info": self.info,
            "error": self.error,
            "warning": self.warning,
            "critical": self.critical,
            "update_client_info": self.update_client_info,
            "listServices": self.list_services,
            "list_services": self.list_services,
            "listClients": self.list_clients,
            "list_clients": self.list_clients,
            "getService": self.get_service,
            "get_service": self.get_service,
            "generate_token": self.generate_token,
            "generateToken": self.generate_token,
            "create_workspace": self.create_workspace,
            "createWorkspace": self.create_workspace,
            "register_service": self.register_service,
            "registerService": self.register_service,
            "get_workspace": self.get_workspace,
            "getWorkspace": self.get_workspace,
            "set": self._update_workspace,
        }
        return interface


class RedisStore:
    """Represent a redis store."""

    _store = None

    @staticmethod
    def get_instance():
        if RedisStore._store is None:
            RedisStore._store = RedisStore()
        return RedisStore._store

    def __init__(self, uri="/tmp/redis.db", port=6383):
        """Set up the redis store."""
        if uri is None:
            uri = os.path.join(tempfile.mkdtemp(), "redis.db")
        Redis(uri, serverconfig={"port": str(port)})
        self._redis = aioredis.from_url(f"redis://127.0.0.1:{port}/0")
        self._root_user = None
        self._event_bus = EventBus()

    def get_event_bus(self):
        return self._event_bus

    async def setup_root_user(self):
        if not self._root_user:
            self._root_user = UserInfo(
                id="root",
                is_anonymous=False,
                email=None,
                parent=None,
                roles=[],
                scopes=[],
                expires_at=None,
            )
            await self.register_user(self._root_user)
        return self._root_user

    async def init(self):
        await self.setup_root_user()

    async def register_user(self, user_info: UserInfo):
        """Register a user."""
        await self._redis.hset(f"users", user_info.id, user_info.json())

    async def get_user(self, user_id: str):
        """Get a user."""
        user_info = await self._redis.hget(f"users", user_id)
        if user_info is None:
            return None
        return UserInfo.parse_obj(json.loads(user_info.decode()))

    async def get_all_users(self):
        """Get all users."""
        users = await self._redis.hgetall("users")
        return [
            UserInfo.parse_obj(json.loads(user.decode())) for user in users.values()
        ]

    async def delete_user(self, user_id: str):
        """Delete a user."""
        await self._redis.hdel(f"users", user_id)

    async def register_workspace(self, workspace: dict, overwrite=False):
        """Add a workspace."""
        workspace = WorkspaceInfo.parse_obj(workspace)
        if not overwrite and await self._redis.hexists("workspaces", workspace.name):
            raise Exception(f"Workspace {workspace.name} already exists.")
        await self.clear_workspace(workspace.name)
        await self._redis.hset("workspaces", workspace.name, workspace.json())
        await self.get_workspace_manager(workspace.name)
        self._event_bus.emit("workspace_registered", workspace)

    async def clear_workspace(self, workspace: str):
        """Clear a workspace."""
        await self._redis.hdel("workspaces", workspace)
        await self._redis.delete(f"{workspace}:clients")

    async def connect_to_workspace(self, workspace: str, client_id: str):
        """Connect to a workspace."""
        manager = WorkspaceManager(
            workspace, self._redis, await self.setup_root_user(), self._event_bus
        )
        await manager.setup(client_id=client_id)
        rpc = await manager.get_rpc()
        wm = await rpc.get_remote_service(workspace + "/workspace-manager:default")
        wm.rpc = rpc
        return wm

    async def get_workspace_manager(self, workspace: str):
        """Get a workspace manager."""
        manager = WorkspaceManager.get_manager(
            workspace, self._redis, await self.setup_root_user(), self._event_bus
        )
        await manager.setup()
        return manager

    async def get_workspace_interface(self, workspace: str):
        """Get the interface of a workspace."""
        manager = WorkspaceManager.get_manager(
            workspace, self._redis, await self.setup_root_user(), self._event_bus
        )
        return await manager.get_workspace()

    async def list_workspace(self):
        """List all workspaces."""
        workspace_keys = await self._redis.hkeys("workspaces")
        return [k.decode() for k in workspace_keys]

    async def delete_workspace(self, workspace: str):
        """Delete a workspace."""
        winfo = await self.get_workspace_info(workspace)
        await self._redis.hdel("workspaces", workspace)
        self._event_bus.emit("workspace_removed", winfo)

    async def get_workspace_info(self, workspace: str):
        """Get info of the current workspace."""
        workspace_info = await self._redis.hget("workspaces", workspace)
        if workspace_info is None:
            raise KeyError(f"Workspace not found: {workspace}")
        workspace_info = WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))
        return workspace_info
