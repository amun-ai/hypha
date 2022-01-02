"""A scalable state store based on Redis."""
import asyncio
import json

import msgpack
from imjoy_rpc.core_connection import BasicConnection
from redislite import Redis

from hypha.core import RDF, WorkspaceInfo
from hypha.core.rpc import RPC


class RedisStore:
    """Represent a redis store."""

    def __init__(self, uri="/tmp/redis.db"):
        """Set up the redis store."""
        self.redis = Redis(uri)

    def create_rpc(
        self,
        workspace: WorkspaceInfo,
        client_id: str,
    ):
        """Create a rpc."""
        pubsub = self.redis.pubsub()

        async def send(data):
            if "target" not in data:
                raise ValueError("target is required")
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

        rpc = RPC(connection, client_id="core")
        return rpc

    def register_workspace(self, workspace: WorkspaceInfo):
        """Add a workspace."""

        def register_service(encoded):
            assert "id" in encoded
            workspace.interfaces["/" + encoded["id"]] = encoded
            self._save_workspace(workspace)

        def get_service(service_id):
            ws = self.get_workspace(workspace.name)
            interface = ws.interfaces["/" + service_id]
            return interface

        def list_services():
            ws = self.get_workspace(workspace.name)
            return list(ws.interfaces.keys())

        interface = {
            "config": {},
            "log": print,
            "register_service": register_service,
            "get_service": get_service,
            "list_services": list_services,
        }

        rpc = self.create_rpc(workspace, "core")
        workspace.interfaces["/"] = rpc.export(interface)
        self._save_workspace(workspace)

    def _save_workspace(self, workspace: WorkspaceInfo):
        workspace_info = workspace.json()
        self.redis.hset("workspaces", workspace.name, workspace_info)

    def get_workspace(self, workspace_name):
        """Get a workspace."""
        workspace_info = self.redis.hget("workspaces", workspace_name)
        return WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))

    def get_workspace_interface(self, workspace_name, client_id):
        """Get the workspace interface."""
        workspace = self.get_workspace(workspace_name)
        return workspace.interfaces["/"]

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

    def register_plugin(self, plugin_info: RDF):
        """Add a plugin."""
        self.redis.hset("plugins", plugin_info.id, plugin_info.json())

    def get_plugin(self, plugin_id: str):
        """Get a plugin."""
        return RDF.parse_obj(json.loads(self.redis.hget("plugins", plugin_id).decode()))

    def get_all_plugins(self):
        """Get all plugins."""
        return {
            k.decode(): RDF.parse_obj(json.loads(v.decode()))
            for k, v in self.redis.hgetall("plugins").items()
        }

    def list_plugins(self):
        """List all plugins."""
        return [k.decode() for k in self.redis.hkeys("plugins")]

    def delete_plugin(self, plugin_id: str):
        """Delete a plugin."""
        self.redis.delete("plugins", plugin_id)


store = RedisStore()
