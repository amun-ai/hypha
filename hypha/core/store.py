"""A scalable state store based on Redis."""
import asyncio
import inspect
import json
import logging
from multiprocessing.sharedctypes import Value
import os
import sys
import io
import tempfile
from typing import Callable, Union, Optional

from numpy import isin
from hypha.utils import EventBus

import aioredis
import msgpack
import shortuuid
from redislite import Redis
import random

from hypha.core import (
    ClientInfo,
    UserInfo,
    WorkspaceInfo,
    VisibilityEnum,
    TokenConfig,
    RDF,
)
from hypha.core.rpc import RPC
from hypha.core.auth import generate_presigned_token, generate_reconnection_token

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
        self._subscribe_task = None
        self._workspace = workspace
        self._client_id = client_id
        assert "/" not in client_id
        self._user_info = user_info.dict()

    def on_message(self, handler: Callable):
        self._handle_message = handler
        self._is_async = inspect.iscoroutinefunction(handler)
        self._subscribe_task = asyncio.ensure_future(self._subscribe_redis())

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
        self._redis = None
        self._handle_message = None
        if self._subscribe_task:
            self._subscribe_task.cancel()
            self._subscribe_task = None
        logger.info("Redis connection disconnected (%s)", reason)

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


class RedisEventBus(EventBus):
    """Represent a redis event bus."""

    def __init__(self, redis) -> None:
        super().__init__()
        self._redis = redis

    async def init(self, loop):
        self._ready = loop.create_future()
        self._loop = loop
        loop.create_task(self._subscribe_redis())
        await self._ready

    def emit(self, event_type, data=None):
        assert self._ready.result(), "Event bus not ready"
        self._loop.create_task(
            self._redis.publish(
                "global_event_bus", json.dumps({"type": event_type, "data": data})
            )
        )

    async def _subscribe_redis(self):
        pubsub = self._redis.pubsub()
        try:
            await pubsub.subscribe("global_event_bus")
            self._ready.set_result(True)
            while True:
                msg = await pubsub.get_message()
                if msg and msg.get("type") == "message":
                    data = json.loads(msg["data"].decode())
                    super().emit(data["type"], data["data"])
        except Exception as exp:
            self._ready.set_exception(exp)


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
        await self.get_workspace_info()
        if self._rpc and await self._redis.hexists(
            f"{self._workspace}:clients", client_id
        ):
            return self._rpc
        if await self.check_client_exists(client_id):
            rpc = self._create_rpc(client_id + "-" + shortuuid.uuid())
            try:
                await rpc.ping(client_id)
            except asyncio.exceptions.TimeoutError:
                logger.info("Removing a dead workspace-manager client: %s", client_id)
                await self.delete_client(client_id)
            else:
                logger.info("Reusing existing workspace-manager client: %s", client_id)
                self._rpc = rpc
                self._initialized = True
                return rpc

        # Register an client as root
        await self.register_client(
            ClientInfo(
                id=client_id,
                workspace=self._workspace,
                user_info=self._root_user.dict(),
            )
        )
        rpc = self._create_rpc(client_id)

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
        await self.remove_clients(workspace.name)
        return await self.get_workspace(workspace.name, context=context)
        # Remove the client if it's not responding
        # for client_id in await manager.list_clients(context=context):
        #     try:
        #         await rpc.ping(client_id)
        #     except asyncio.exceptions.TimeoutError:
        #         await manager.delete_client(client_id)

    async def remove_clients(self, workspace: str = None):
        """Remove all the clients in the workspace."""
        workspace = workspace or self._workspace
        client_keys = await self._redis.hkeys(f"{workspace}:clients")
        for k in client_keys:
            k = k.decode()
            await self.delete_client(k, workspace=workspace)
        await self._redis.delete(f"{workspace}:clients")

    async def install_application(self, rdf: dict, context: Optional[dict] = None):
        """Install an application to the workspace."""
        rdf = RDF.parse_obj(rdf)
        # TODO: check if the application is already installed
        workspace_info = await self.get_workspace_info()
        workspace_info.applications[rdf.id] = rdf.dict()
        await self._update_workspace(workspace_info.dict(), context)

    async def uninstall_application(self, rdf_id: str, context: Optional[dict] = None):
        """Uninstall a application from the workspace."""
        workspace_info = await self.get_workspace_info()
        if rdf_id not in workspace_info.applications:
            raise KeyError("Application not found: " + rdf_id)
        del workspace_info.applications[rdf_id]
        await self._update_workspace(workspace_info.dict(), context)

    async def register_service(self, service, context: Optional[dict] = None, **kwargs):
        """Register a service"""
        user_info = UserInfo.parse_obj(context["user"])
        source_workspace = context["from"].split("/")[0]
        assert (
            source_workspace == self._workspace
        ), "Service must be registered in the same workspace"
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        logger.info("Registering service %s to %s", service.id, self._workspace)
        # create a new rpc client as the user
        # rpc = self._create_rpc(
        #     "client-" + shortuuid.uuid(), self._workspace, user_info=user_info
        # )
        # try:
        #     sv = await rpc.get_remote_service(context["from"] + ":built-in")
        # except Exception as exp:
        #     raise Exception(f"Client not ready: {context['from']}") from exp
        rpc = await self.get_rpc()
        sv = await rpc.get_remote_service(context["from"] + ":built-in")
        service["config"] = service.get("config", {})
        service["config"]["workspace"] = self._workspace
        service = await sv.register_service(service, **kwargs)
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
        config = config or {}
        if "scopes" in config and not isinstance(config["scopes"], list):
            raise Exception(
                "Scopes must be empty or contains a list of workspace names."
            )
        token_config = TokenConfig.parse_obj(config)
        for ws in config["scopes"]:
            if not await self.check_permission(user_info, ws):
                raise PermissionError(f"User has no permission to workspace: {ws}")
        token = generate_presigned_token(user_info, token_config)
        return token

    async def update_client_info(self, client_info: dict, context=None):
        """Update the client info."""
        assert context is not None
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        ws, client_id = context["from"].split("/")
        assert client_info["id"] == client_id
        assert ws == self._workspace
        client_info["user_info"] = user_info
        client_info["workspace"] = self._workspace
        await self._update_client(ClientInfo.parse_obj(client_info))

    async def get_user_info(self, context=None):
        """Get the user info."""
        assert context is not None
        ws, client_id = context["from"].split("/")
        user_info = UserInfo.parse_obj(context["user"])
        # Generate a new toke for the client to reconnect
        logger.info(
            "Generating new reconnection token for client %s (user id: %s)",
            client_id,
            user_info.id,
        )
        token = generate_reconnection_token(user_info.id, client_id)
        return {
            "workspace": ws,
            "client_id": client_id,
            "user_info": context["user"],
            "reconnection_token": token,
        }

    async def _get_client_info(self, client_id):
        assert "/" not in client_id
        client_info = await self._redis.hget(f"{self._workspace}:clients", client_id)
        return client_info and json.loads(client_info.decode())

    async def list_clients(self, context: Optional[dict] = None):
        if context:
            user_info = UserInfo.parse_obj(context["user"])
            if not await self.check_permission(user_info):
                raise Exception(f"Permission denied for workspace {self._workspace}.")
        client_keys = await self._redis.hkeys(f"{self._workspace}:clients")
        return [k.decode() for k in client_keys]

    async def list_user_clients(self, context: Optional[dict] = None):
        """List all the clients belongs to the user."""
        user_info = UserInfo.parse_obj(context["user"])
        client_keys = await self._redis.smembers(f"user:{user_info.id}:clients")
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
        if query is None:
            query = {}

        ws = query.get("workspace")
        if ws:
            del query["workspace"]
        if ws == "*":
            ret = []
            for workspace in await self._get_all_workspace():
                can_access_workspace = await self.check_permission(user_info, workspace)
                ws = await self.get_workspace(workspace.name, context=context)
                for service in await ws.list_services():
                    assert isinstance(service, dict)
                    # To access the service, it should be public or owned by the user
                    if (
                        not can_access_workspace
                        and service["config"].get("visibility") != "public"
                    ):
                        continue
                    match = True
                    for key in query:
                        if service.get(key) != query[key]:
                            match = False
                    if match:
                        if "/" not in service["id"]:
                            service["id"] = workspace.name + "/" + service["id"]
                        ret.append(service)
            return ret

        if ws is None:
            can_access_workspace = await self.check_permission(user_info)
            services = await self._get_services()
            workspace_name = self._workspace
        else:
            can_access_workspace = await self.check_permission(user_info, ws)
            workspace_name = ws
            ws = await self.get_workspace(ws, context=context)
            services = await ws.list_services()

        ret = []

        for service in services:
            assert isinstance(service, dict)
            if (
                not can_access_workspace
                and service["config"].get("visibility") != "public"
            ):
                continue
            match = True
            for key in query:
                if service.get(key) != query[key]:
                    match = False
            if match:
                if "/" not in service["id"]:
                    service["id"] = workspace_name + "/" + service["id"]
                ret.append(service)

        return ret

    async def _get_services(self):
        service_list = []
        client_ids = await self.list_clients()
        for client_id in client_ids:
            try:
                client_info = await self._get_client_info(client_id)
                assert (
                    client_info is not None
                ), f"Failed to get client info for {self._workspace}/{client_id}"
                for sinfo in client_info["services"]:
                    service_list.append(sinfo)
            # make sure we don't throw if a client is removed
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to get service info for client: %s", client_id)
        return service_list

    async def _update_client(self, client_info: ClientInfo):
        """Update the client info."""
        assert "/" not in client_info.id
        if not await self._redis.hexists(f"{self._workspace}:clients", client_info.id):
            raise Exception(f"Client {client_info.id} not found.")
        await self._redis.hset(
            f"{self._workspace}:clients", client_info.id, client_info.json()
        )
        await self._redis.sadd(
            f"user:{client_info.user_info.id}:clients", client_info.id
        )
        logger.info(
            "Client %s updated (services: %s)",
            client_info.id,
            [s.id for s in client_info.services],
        )
        self._event_bus.emit("client_updated", client_info.dict())

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
        await self._redis.sadd(
            f"user:{client_info.user_info.id}:clients", client_info.id
        )
        self._event_bus.emit("client_registered", client_info.dict())
        logger.info("New client registered: %s/%s", self._workspace, client_info.id)

    async def check_client_exists(self, client_id: str, workspace: str = None):
        """Check if a client exists."""
        assert "/" not in client_id
        client_id = client_id
        workspace = workspace or self._workspace
        return await self._redis.hexists(f"{workspace}:clients", client_id)

    async def delete_client(self, client_id: str, workspace: str = None):
        """Delete a client."""
        # assert "/" not in client_id
        workspace = workspace or self._workspace
        client_info = await self._redis.hget(f"{workspace}:clients", client_id)
        if client_info is None:
            raise KeyError(f"Client does not exist: {workspace}/{client_id}")
        client_info = ClientInfo.parse_obj(json.loads(client_info.decode()))
        await self._redis.srem(
            f"user:{client_info.user_info.id}:clients", client_info.id
        )
        await self._redis.hdel(f"{workspace}:clients", client_id)
        logger.info("Client deleted: %s/%s", workspace, client_id)

    def _create_rpc(
        self, client_id: str, default_context=None, user_info: UserInfo = None
    ):
        """Create a rpc for the workspace.
        Note: Any request made through this rcp will be treated as a request from the root user.
        """
        assert "/" not in client_id
        logger.info("Creating RPC for client %s/%s", self._workspace, client_id)
        connection = RedisRPCConnection(
            self._redis, self._workspace, client_id, (user_info or self._root_user)
        )
        rpc = RPC(connection, client_id=client_id, default_context=default_context)
        return rpc

    async def echo(self, data, context=None):
        """Log a plugin message."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        return data

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

    async def get_workspace_info(self, workspace: str = None):
        """Get info of the current workspace."""
        workspace = workspace or self._workspace
        workspace_info = await self._redis.hget("workspaces", workspace)
        if workspace_info is None:
            raise KeyError(f"Workspace not found: {workspace}")
        workspace_info = WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))
        return workspace_info

    async def _launch_application_by_service(
        self,
        query: dict,
        timeout: float = 60,
        context: dict = None,
    ):
        """Launch an installed application by service name."""
        workspace, service_id, client_name = (
            query.get("workspace", self._workspace),
            query.get("service_id"),
            query.get("client_name"),
        )
        if not service_id or service_id == "*":
            service_id = "default"
        # Check if the user has permission to this workspace and the one to be launched
        token = await self.generate_token({"scopes": [workspace]}, context=context)
        workspace = await self.get_workspace_info(workspace)
        controller = await self.get_service("public/workspace-manager:server-apps")
        if not controller:
            raise Exception(
                "Plugin `{name}` not found and failed to"
                " launch the plugin (no server-apps service found)"
            )
        app_id = None
        for aid, app_info in workspace.applications.items():
            # find the app by service name
            if client_name and client_name == app_info.name:
                app_id = aid
                break
            if app_info.services and service_id in app_info.services:
                app_id = aid
                break
        if app_id is None:
            raise Exception(f"Service id {service_id} not found")

        client_id = shortuuid.uuid()
        config = await controller.start(
            app_id,
            workspace=workspace.name,
            token=token,
            client_id=client_id,
            timeout=timeout,
        )
        return await self.get_service(
            {
                "workspace": workspace.name,
                "client_id": config["id"],
                "service_id": service_id,
            },
            context=context,
        )

    async def get_service(self, query: Union[dict, str], context=None):
        # Note: No authorization required because the access is controlled by the client itself
        if isinstance(query, dict):
            if "id" not in query:
                service_id = query.get("service_id", "*")
            else:
                service_id = query["id"]
        else:
            assert isinstance(query, str)
            service_id = query
            query = {"id": service_id}
        assert isinstance(service_id, str)

        if "/" in service_id and ":" not in service_id:
            service_id = service_id + ":default"
            query["workspace"] = service_id.split("/")[0]
            if "client_id" in query and query["client_id"] != service_id.split("/")[1]:
                raise ValueError(
                    f"client_id ({query['client_id']}) does not match service_id ({service_id})"
                )
            query["client_id"] = service_id.split("/")[1]
        elif "/" not in service_id and ":" not in service_id:
            workspace = query.get("workspace", self._workspace)
            service_id = workspace + "/*:" + service_id
            query["workspace"] = workspace
            query["client_id"] = "*"
        elif "/" not in service_id:
            workspace = query.get("workspace", self._workspace)
            query["client_id"] = service_id.split(":")[0]
            service_id = workspace + "/" + service_id
            query["workspace"] = workspace
        else:
            assert "/" in service_id and ":" in service_id
            workspace = service_id.split("/")[0]
            query["client_id"] = service_id.split("/")[1].split(":")[0]
            query["workspace"] = workspace

        query["service_id"] = service_id.split("/")[1].split(":")[1]

        logger.info("Getting service via query %s", query)

        # service id with an abosolute client id
        if "*" not in service_id:
            if "/" in service_id:
                workspace = service_id.split("/")[0]
                if workspace != self._workspace:
                    ws = await self.get_workspace(workspace)
                    return await ws.get_service(service_id, context=context)
            if "/" not in service_id:
                service_id = self._workspace + "/" + service_id
                workspace = self._workspace

            # Make sure the client exists
            if await self._get_client_info(service_id.split("/")[1].split(":")[0]):
                rpc = await self.setup()
                service_api = await rpc.get_remote_service(service_id)
                service_api["config"]["workspace"] = workspace
                return service_api
            elif "launch" in query and query["launch"] == True:
                app_info = await self._launch_application_by_service(
                    service_id,
                    context=context,
                )
                # TODO: get the actual service
                return app_info
        else:
            workspace, sname = service_id.split("/")
            cid, sid = sname.split(":")
            assert cid == "*", "Specifying a client id is not supported: " + cid
            if sid == "*":  # Skip the client check if the service id is *
                services = []
            elif workspace == "*":
                services = await self.list_services({"workspace": "*"}, context=context)
            elif workspace != self._workspace:
                assert await self._redis.hget("workspaces", workspace)
                ws = await self.get_workspace(workspace)
                services = await ws.list_services()
            else:  # only the service id without client id
                services = await self.list_services(context=context)
                workspace = self._workspace

        services = list(
            filter(lambda service: service["id"].endswith(":" + sid), services)
        )
        if not services:
            if "launch" in query and query["launch"] == True:
                app_info = await self._launch_application_by_service(
                    query,
                    context=context,
                )
                # TODO: get the actual service
                return app_info
            raise Exception(f"Service not found: {sid} in {workspace}")
        service_info = random.choice(services)
        rpc = await self.setup()
        service_api = await rpc.get_remote_service(service_info["id"])
        service_api["config"]["workspace"] = service_info["id"].split("/")[0]
        return service_api

    async def _get_all_workspace(self):
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
            workspace = await self.get_workspace_info(workspace)

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
            rpc = await self.setup()
            try:
                wm = await rpc.get_remote_service(
                    workspace + "/workspace-manager:default", timeout=10
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
        if "name" in config and config["name"] != workspace.name:
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
        self._event_bus.emit("workspace_changed", workspace.dict())

    async def delete_if_empty(self):
        """Delete the workspace if it is empty."""
        client_keys = await self._redis.hkeys(f"{self._workspace}:clients")
        if not client_keys:
            await self.delete()

    async def delete(self):
        """Delete the workspace."""
        await self.remove_clients(self._workspace)
        winfo = await self.get_workspace_info(self._workspace)
        await self._redis.hdel("workspaces", self._workspace)
        self._event_bus.emit("workspace_removed", winfo.dict())

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
            "echo": self.echo,
            "log": self.log,
            "info": self.info,
            "error": self.error,
            "warning": self.warning,
            "critical": self.critical,
            "update_client_info": self.update_client_info,
            "get_user_info": self.get_user_info,
            "getUserInfo": self.get_user_info,
            "listServices": self.list_services,
            "list_services": self.list_services,
            "listClients": self.list_clients,
            "list_clients": self.list_clients,
            "list_user_clients": self.list_user_clients,
            "listUserClients": self.list_user_clients,
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
            "install_application": self.install_application,
            "installApplication": self.install_application,
            "uninstall_application": self.uninstall_application,
            "uninstallApplication": self.uninstall_application,
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
        self._event_bus = RedisEventBus(self._redis)

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

    async def init(self, loop):
        await self._event_bus.init(loop)
        await self.setup_root_user()

    async def register_user(self, user_info: UserInfo):
        """Register a user."""
        await self._redis.hset("users", user_info.id, user_info.json())

    async def get_user(self, user_id: str):
        """Get a user."""
        user_info = await self._redis.hget("users", user_id)
        if user_info is None:
            return None
        return UserInfo.parse_obj(json.loads(user_info.decode()))

    async def get_user_workspace(self, user_id: str):
        """Get a user."""
        workspace_info = await self._redis.hget("workspaces", user_id)
        if workspace_info is None:
            return None
        workspace_info = WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))
        return workspace_info

    async def get_all_users(self):
        """Get all users."""
        users = await self._redis.hgetall("users")
        return [
            UserInfo.parse_obj(json.loads(user.decode())) for user in users.values()
        ]

    async def delete_user(self, user_id: str):
        """Delete a user."""
        await self._redis.hdel("users", user_id)

    async def get_all_workspace(self):
        """Get all workspaces."""
        workspaces = await self._redis.hgetall("workspaces")
        return [
            WorkspaceInfo.parse_obj(json.loads(v.decode()))
            for k, v in workspaces.items()
        ]

    async def register_workspace(self, workspace: dict, overwrite=False):
        """Add a workspace."""
        workspace = WorkspaceInfo.parse_obj(workspace)
        if not overwrite and await self._redis.hexists("workspaces", workspace.name):
            raise Exception(f"Workspace {workspace.name} already exists.")
        if overwrite:
            # clean up the clients and users in the workspace
            client_keys = await self._redis.hkeys(f"{workspace.name}:clients")
            for k in client_keys:
                client_id = k.decode()
                client_info = await self._redis.hget(
                    f"{workspace.name}:clients", client_id
                )
                if client_info is None:
                    raise KeyError(
                        f"Client does not exist: {workspace.name}/{client_id}"
                    )
                client_info = ClientInfo.parse_obj(json.loads(client_info.decode()))
                await self._redis.srem(
                    f"user:{client_info.user_info.id}:clients", client_info.id
                )
                # assert ret >= 1, f"Client not found in user({client_info.user_info.id})'s clients list: {client_info.id}"
                await self._redis.hdel(f"{workspace.name}:clients", client_id)
            await self._redis.delete(f"{workspace}:clients")
        await self._redis.hset("workspaces", workspace.name, workspace.json())
        await self.get_workspace_manager(workspace.name, setup=True)
        self._event_bus.emit("workspace_registered", workspace.dict())

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
            workspace, self._redis, await self.setup_root_user(), self._event_bus
        )
        if setup:
            await manager.setup()
        return manager

    async def get_workspace_interface(self, workspace: str, context: dict = None):
        """Get the interface of a workspace."""
        manager = WorkspaceManager.get_manager(
            workspace, self._redis, await self.setup_root_user(), self._event_bus
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
