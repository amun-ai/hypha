"""A scalable state store based on Redis."""
import asyncio
import json
import os
import tempfile
from typing import Union

import msgpack
from imjoy_rpc.core_connection import BasicConnection
from redislite import Redis

from hypha.core import WorkspaceInfo
from hypha.core.rpc import RPC


class WorkspaceManager:
    def __init__(self, workspace: Union[str, WorkspaceInfo], redis: Redis):
        self.redis = redis
        if isinstance(workspace, str):
            workspace = self._get_workspace(workspace)
        self.workspace = workspace

    def _save_workspace(self, workspace: WorkspaceInfo):
        workspace_info = workspace.json()
        self.redis.hset("workspaces", workspace.name, workspace_info)

    def _get_workspace(self, workspace_name):
        """Get a workspace."""
        workspace_info = self.redis.hget("workspaces", workspace_name)
        if workspace_info is None:
            raise Exception(f"Workspace not found: {workspace_name}")
        return WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))

    def save_services(self, services, context=None):
        assert isinstance(services, list)
        for service in services:
            if "id" not in service or "name" not in service or "config" not in service:
                raise ValueError(
                    f"Invalid service(id={service.get('id')}, "
                    f"name={service.get('name')}), each service"
                    " should at least contain `id`, `name` and `config`."
                )
        client_id = "/" + context["client_id"] if context["client_id"] != "/" else "/"
        self.workspace.interfaces[client_id] = services
        self._save_workspace(self.workspace)

    def list_services(self, context=None):
        service_list = []
        for provider, services in self.workspace.interfaces.items():
            for service in services:
                service_list.append(
                    {
                        "id": provider + service["id"]
                        if provider != "/"
                        else service["id"],
                        "name": service["name"],
                        "config": service["config"],
                    }
                )
        return service_list

    def log(self, *args, context=None):
        print(context["client_id"], *args)

    def create_service(self, service_id, service_name):
        interface = {
            "id": service_id,
            "name": service_name or service_id,
            "config": {"require_context": True},
            "log": self.log,
            "save_services": self.save_services,
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

    def __init__(self, uri="/tmp/redis.db"):
        """Set up the redis store."""
        if uri is None:
            uri = os.path.join(tempfile.mkdtemp(), "redis.db")
        self.redis = Redis(uri)

    def create_rpc(
        self, workspace: Union[str, WorkspaceInfo], client_id: str, default_context=None
    ):
        """Create a rpc."""
        if isinstance(workspace, str):
            workspace = self.get_workspace(workspace)
        pubsub = self.redis.pubsub()

        async def send(data):
            if "target" not in data:
                raise ValueError(f"target is required: {data}")
            self.redis.publish(
                f"{workspace.name}:msg:{data['target']}", msgpack.packb(data)
            )

        connection = BasicConnection(send)

        async def listen():
            while True:
                msg = pubsub.get_message()
                if msg and msg.get("type") == "message":
                    assert (
                        msg.get("channel")
                        == f"{workspace.name}:msg:{client_id}".encode()
                    )
                    data = msgpack.unpackb(msg["data"])
                    print(
                        client_id,
                        "<====process===",
                        data.get("source"),
                        data["type"],
                        data,
                    )
                    connection.handle_message(data)

                await asyncio.sleep(0.01)

        asyncio.ensure_future(listen())

        pubsub.subscribe(f"{workspace.name}:msg:{client_id}")

        rpc = RPC(connection, client_id=client_id, default_context=default_context)
        return rpc

    async def register_workspace(self, workspace: WorkspaceInfo):
        """Add a workspace."""
        self._save_workspace(workspace)
        await self.setup_workspace_manager(
            workspace.name, client_id="/", service_id="/"
        )

    def _save_workspace(self, workspace: WorkspaceInfo):
        workspace_info = workspace.json()
        self.redis.hset("workspaces", workspace.name, workspace_info)

    def get_workspace(self, workspace_name):
        """Get a workspace."""
        workspace_info = self.redis.hget("workspaces", workspace_name)
        if workspace_info is None:
            raise Exception(f"Workspace not found: {workspace_name}")
        return WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))

    def connect_to_workspace(self, workspace_name, client_id):
        rpc = self.create_rpc(workspace_name, client_id=client_id, default_context={})
        return rpc

    async def setup_workspace_manager(
        self, workspace_name, client_id, service_id, service_name=None
    ):
        """Get the workspace interface."""
        manager = WorkspaceManager(workspace_name, self.redis)
        rpc = self.connect_to_workspace(workspace_name, client_id)

        def update_services(_):
            local_services = rpc.get_all_local_services()
            manager.save_services(
                [
                    {
                        "id": k,
                        "name": local_services[k]["name"],
                        "config": local_services[k]["config"],
                    }
                    for k in local_services.keys()
                ],
                context={"client_id": rpc.client_id},
            )

        rpc.on("serviceUpdated", update_services)
        management_service = manager.create_service(service_id, service_name)
        await rpc.register_service(management_service)

    def get_all_workspaces(self):
        """Get all workspaces."""
        return {
            k.decode(): WorkspaceInfo.parse_obj(json.loads(v.decode()))
            for k, v in self.redis.hgetall("workspaces").items()
        }

    def list_workspace(self):
        """List all workspaces."""
        return [k.decode() for k in self.redis.hkeys("workspaces")]

    def delete_workspace(self, workspace_name: str):
        """Delete a workspace."""
        self.redis.delete("workspaces", workspace_name)

    def register_client(self, client_info):
        """Add a client."""
        self.redis.hset("clients", client_info["id"], json.dumps(client_info))

    def get_client(self, client_id: str):
        """Get a client."""
        return json.loads(self.redis.hget("clients", client_id).decode())

    def check_client_exists(self, client_id: str):
        """Check if a client exists."""
        return self.redis.hexists("clients", client_id)

    def get_all_clients(self):
        """Get all clients."""
        return {
            k.decode(): json.loads(v.decode())
            for k, v in self.redis.hgetall("clients").items()
        }

    def list_clients(self):
        """List all clients."""
        return [k.decode() for k in self.redis.hkeys("clients")]

    def delete_client(self, client_id: str):
        """Delete a client."""
        self.redis.delete("clients", client_id)
