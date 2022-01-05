"""A scalable state store based on Redis."""
import asyncio
import inspect
import json
import logging
import os
import sys
import tempfile
from typing import Callable, Union

import aioredis
import msgpack
import shortuuid
from redislite import Redis

from hypha.core import ClientInfo, WorkspaceInfo
from hypha.core.rpc import RPC

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("redis-store")
logger.setLevel(logging.INFO)


class RedisRPCConnection:
    """Represent a redis connection."""

    def __init__(self, redis: aioredis.Redis, workspace: str, client_id: str):
        self._redis = redis
        self._handle_message = None
        self._workspace = workspace
        self._client_id = client_id

    def on_message(self, handler: Callable):
        self._handle_message = handler
        self._is_async = inspect.iscoroutinefunction(handler)
        asyncio.ensure_future(self._subscribe_redis())

    async def emit_message(self, data: Union[dict, bytes]):
        assert self._handle_message is not None, "No handler for message"
        if isinstance(data, dict):
            if "to" not in data:
                raise ValueError(f"`to` is required: {str(data)[:20]}")
            await self._redis.publish(
                f"{self._workspace}:msg:{data['to']}", msgpack.packb(data)
            )
        elif isinstance(data, bytes):
            target_len = int.from_bytes(data[0:1], "big")
            target_id = data[1 : target_len + 1].decode()
            await self._redis.publish(
                f"{self._workspace}:msg:{target_id}", data[target_len + 1 :]
            )
        else:
            raise TypeError("Invalid type")

    async def disconnect(self):
        pass

    async def _subscribe_redis(self):
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(f"{self._workspace}:msg:{self._client_id}")
        while True:
            msg = await pubsub.get_message()
            if msg and msg.get("type") == "message":
                assert (
                    msg.get("channel")
                    == f"{self._workspace}:msg:{self._client_id}".encode()
                )
                if self._is_async:
                    await self._handle_message(msg["data"])
                else:
                    self._handle_message(msg["data"])


class WorkspaceManager:

    _managers = {}

    @staticmethod
    def get_manager(workspace: str, redis: Redis):
        if workspace in WorkspaceManager._managers:
            return WorkspaceManager._managers[workspace]
        else:
            return WorkspaceManager(workspace, redis)

    def __init__(self, workspace: str, redis: Redis):
        self._redis = redis
        self._workspace = workspace
        self._initialized = False
        WorkspaceManager._managers[workspace] = self

    async def setup(
        self,
        client_id="workspace-manager",
        service_id="default",
        service_name="Default workspace management service",
    ):
        if await self.check_client_exists(client_id):
            # try to determine whether the client is still alive
            rpc = await self.create_rpc(client_id + "-" + shortuuid.uuid())
            try:
                svc = await rpc.get_remote_service(f"{client_id}:{service_id}")
                assert svc is not None
                # return directly without creating the client
                return
            except Exception:  # pylint: disable=broad-except
                logger.info(
                    "Another client with the same id (%s) exists but not responding.",
                    client_id,
                )

        await self.register_client(ClientInfo(id=client_id))
        rpc = await self.create_rpc(client_id)

        def save_client_info(_):
            return asyncio.create_task(
                self.update_client_info(
                    rpc.get_client_info(),
                    context={"client_id": rpc.client_id},
                )
            )

        rpc.on("serviceUpdated", save_client_info)
        management_service = self.create_service(service_id, service_name)
        await rpc.register_service(management_service, notify=False)
        await save_client_info(None)
        self._initialized = True

    async def update_client_info(self, client_info, context=None):
        assert context is not None and "client_id" in context
        client_info = ClientInfo.parse_obj(client_info)
        client_id = context["client_id"] if context["client_id"] != "/" else "/"
        await self._redis.hset(
            f"{self._workspace}:clients", client_id, client_info.json()
        )

    async def get_client_info(self, client_id, context=None):
        client_info = await self._redis.hget(f"{self._workspace}:clients", client_id)
        assert (
            client_info is not None
        ), f"Failed to get client info for {self._workspace}/{client_id}"
        return client_info and json.loads(client_info.decode())

    async def list_clients(
        self,
    ):
        client_keys = await self._redis.hkeys(f"{self._workspace}:clients")
        return [k.decode() for k in client_keys]

    async def list_services(self, context=None):
        service_list = []
        client_ids = await self.list_clients()
        for client_id in client_ids:
            try:
                client_info = await self.get_client_info(client_id)
                for sinfo in client_info["services"]:
                    sinfo["uri"] = f"{self._workspace}/{client_id}:{sinfo['id']}"
                    service_list.append(sinfo)
            # make sure we don't throw if a client is removed
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to get service info for client: %s", client_id)
        return service_list

    async def register_client(self, client_info: ClientInfo):
        """Add a client."""
        await self._redis.hset(
            f"{self._workspace}:clients", client_info.id, client_info.json()
        )

    async def check_client_exists(self, client_id: str):
        """Check if a client exists."""
        return await self._redis.hexists(f"{self._workspace}:clients", client_id)

    async def list_clients(self):
        """List all clients."""
        client_keys = await self._redis.hkeys(f"{self._workspace}:clients")
        return [k.decode() for k in client_keys]

    async def delete_client(self, client_id: str):
        """Delete a client."""
        await self._redis.hdel(f"{self._workspace}:clients", client_id)

    async def create_rpc(self, client_id: str, default_context=None):
        """Create a rpc for the workspace."""
        connection = RedisRPCConnection(self._redis, self._workspace, client_id)
        rpc = RPC(connection, client_id=client_id, default_context=default_context)
        return rpc

    def log(self, *args, context=None):
        print(context["client_id"], *args)

    def create_service(self, service_id, service_name=None):
        interface = {
            "id": service_id,
            "name": service_name or service_id,
            "config": {"require_context": True},
            "log": self.log,
            "update_client_info": self.update_client_info,
            "list_services": self.list_services,
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

    def __init__(self, uri="/tmp/redis.db", port=6381):
        """Set up the redis store."""
        if uri is None:
            uri = os.path.join(tempfile.mkdtemp(), "redis.db")
        Redis(uri, serverconfig={"port": str(port)})
        self._redis = aioredis.from_url(f"redis://127.0.0.1:{port}/0")

    async def register_workspace(self, workspace: dict, overwrite=False):
        """Add a workspace."""
        workspace = WorkspaceInfo.parse_obj(workspace)
        if not overwrite and await self._redis.hexists("workspaces", workspace.name):
            raise Exception(f"Workspace {workspace.name} already exists.")
        await self._redis.hset("workspaces", workspace.name, workspace.json())
        manager = WorkspaceManager.get_manager(workspace.name, self._redis)
        await manager.setup()

    async def clear_workspace(self, workspace: str):
        """Clear a workspace."""
        await self._redis.delete("workspaces", workspace)
        await self._redis.delete(f"{workspace}:clients")

    async def get_workspace(self, workspace: str):
        """Get a workspace."""
        workspace_info = await self._redis.hget("workspaces", workspace)
        if workspace_info is None:
            raise Exception(f"Workspace not found: {workspace}")
        return WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))

    async def get_workspace_manager(self, workspace: str):
        """Get a workspace manager."""
        manager = WorkspaceManager.get_manager(workspace, self._redis)
        await manager.setup()
        return manager

    async def connect_to_workspace(
        self, workspace: str, client_id: str, default_context: dict = None
    ):
        """Connect to a workspace."""
        manager = await self.get_workspace_manager(workspace)
        return await manager.create_rpc(client_id, default_context=default_context)

    async def get_all_workspaces(self):
        """Get all workspaces."""
        workspaces = await self._redis.hgetall("workspaces")
        return {
            k.decode(): WorkspaceInfo.parse_obj(json.loads(v.decode()))
            for k, v in workspaces.items()
        }

    async def list_workspace(self):
        """List all workspaces."""
        workspace_keys = await self._redis.hkeys("workspaces")
        return [k.decode() for k in workspace_keys]

    async def delete_workspace(self, workspace: str):
        """Delete a workspace."""
        await self._redis.hdel("workspaces", workspace)
