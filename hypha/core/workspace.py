import asyncio
import json
import logging
import random
import sys
from typing import Optional, Union

import shortuuid
from fakeredis import aioredis

from hypha.core import (
    RDF,
    ClientInfo,
    TokenConfig,
    UserInfo,
    VisibilityEnum,
    WorkspaceInfo,
    RedisRPCConnection,
)
from hypha.core.auth import generate_presigned_token, generate_reconnection_token
from imjoy_rpc.hypha import RPC
from hypha.utils import EventBus

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("workspace")
logger.setLevel(logging.INFO)

SERVICE_SUMMARY_FIELD = ["id", "name", "type", "config"]


class WorkspaceManager:
    _managers = {}

    @staticmethod
    def get_manager(
        workspace: str,
        redis: aioredis.FakeRedis,
        root_user: UserInfo,
        event_bus: EventBus,
        server_info: dict,
    ):
        if workspace in WorkspaceManager._managers:
            return WorkspaceManager._managers[workspace]
        else:
            return WorkspaceManager(workspace, redis, root_user, event_bus, server_info)

    def __init__(
        self,
        workspace: str,
        redis: aioredis.FakeRedis,
        root_user: UserInfo,
        event_bus: EventBus,
        server_info: dict,
    ):
        self._redis = redis
        self._workspace = workspace
        self._initialized = False
        self._rpc = None
        self._root_user = root_user
        self._event_bus = event_bus
        self._server_info = server_info
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
            except asyncio.TimeoutError:
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

    async def get_rpc(self):
        if not self._rpc:
            await self.setup()
        return self._rpc

    async def get_summary(self, context: Optional[dict] = None) -> dict:
        """Get a summary about the workspace."""
        if context:
            user_info = UserInfo.parse_obj(context["user"])
            if not await self.check_permission(user_info):
                raise Exception(f"Permission denied for workspace {self._workspace}.")
        workspace_info = await self.get_workspace_info()
        workspace_info = workspace_info.dict()
        workspace_summary_fields = ["name", "description"]
        clients = await self.list_clients(context=context)
        services = await self.list_services(context=context)
        summary = {k: workspace_info[k] for k in workspace_summary_fields}
        summary.update(
            {
                "client_count": len(clients),
                "clients": clients,
                "service_count": len(services),
                "services": [
                    {k: service[k] for k in SERVICE_SUMMARY_FIELD}
                    for service in services
                ],
            }
        )
        return summary

    async def create_workspace(
        self, config: dict, overwrite=False, context: Optional[dict] = None
    ):
        """Create a new workspace."""
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        config["persistent"] = config.get("persistent") or False
        if user_info.is_anonymous and config["persistent"]:
            raise Exception("Only registered user can create persistent workspace.")
        workspace = WorkspaceInfo.parse_obj(config)
        # make sure we add the user's email to owners
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]

        if not overwrite and await self._redis.hexists("workspaces", workspace.name):
            raise Exception(f"Workspace {workspace.name} already exists.")
        await self._redis.hset("workspaces", workspace.name, workspace.json())
        # Clear the workspace
        await self.remove_clients(workspace.name)
        workspace_info = await self.get_workspace_info(workspace.name)
        return workspace_info.dict()

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
        assert source_workspace == self._workspace, (
            f"Service must be registered in the same workspace: "
            f"{source_workspace} != {self._workspace} (current workspace)."
        )

        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        logger.info("Registering service %s to %s", service.id, self._workspace)
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
        elif "scopes" not in config:
            config["scopes"] = [self._workspace]
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

    async def get_connection_info(self, context=None):
        """Get the connection info."""
        assert context is not None
        ws, client_id = context["from"].split("/")
        user_info = UserInfo.parse_obj(context["user"])
        # Generate a new toke for the client to reconnect
        logger.info(
            "Generating new reconnection token for client %s (user id: %s)",
            client_id,
            user_info.id,
        )
        expires_in = 60 * 60 * 3  #  3 hours
        token = generate_reconnection_token(
            user_info, client_id, ws, expires_in=expires_in
        )
        info = {
            "workspace": ws,
            "client_id": client_id,
            "user_info": context["user"],
            "reconnection_token": token,
            "reconnection_expires_in": expires_in,
        }
        info.update(self._server_info)
        return info

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
        self, query: Optional[Union[dict, str]] = None, context: Optional[dict] = None
    ):
        """Return a list of services based on the query."""
        assert context is not None
        # if workspace is not set, then it means current workspace
        # if workspace = *, it means search globally
        # otherwise, it search the specified workspace
        user_info = UserInfo.parse_obj(context["user"])
        if query is None:
            query = {}

        if isinstance(query, str):
            if query == "public":
                public_services = await self._redis.hgetall("public:services")
                ps = []
                for service_id, service in public_services.items():
                    sid = service_id.decode()
                    assert "/" in sid and ":" in sid
                    ws = sid.split("/")[0]
                    cid = sid.split("/")[1].split(":")[0]
                    # Disconnected clients will be removed
                    if await self.check_client_exists(cid, ws):
                        service = json.loads(service.decode())
                        # Make sure we have an absolute id with workspace
                        service["id"] = sid
                        ps.append(service)
                    else:
                        logger.info("Skipping unavailable service: %s", sid)
                return ps
            else:
                query = {"workspace": query}

        ws = query.get("workspace")
        if ws:
            del query["workspace"]
        if ws == "*":  # search public and the current workspace
            ret = []
            for workspace in [self._workspace, "public"]:
                can_access_workspace = await self.check_permission(user_info, workspace)
                ws = await self.get_workspace(workspace)
                for service in await ws.list_services():
                    assert isinstance(service, dict)
                    # To access the service, it should be public or owned by the user
                    if (
                        service["id"].endswith(":built-in")
                        or service["id"] == "workspace-manager:default"
                        or (
                            not can_access_workspace
                            and service["config"].get("visibility") != "public"
                        )
                    ):
                        continue
                    match = True
                    for key in query:
                        if service.get(key) != query[key]:
                            match = False
                    if match:
                        if "/" not in service["id"]:
                            service["id"] = workspace + "/" + service["id"]
                        ret.append(service)
            return ret

        if ws is None:
            can_access_workspace = await self.check_permission(user_info)
            services = await self._get_services()
            workspace_name = self._workspace
        else:
            can_access_workspace = await self.check_permission(user_info, ws)
            workspace_name = ws
            ws = await self.get_workspace(ws)
            services = await ws.list_services()

        ret = []

        for service in services:
            assert isinstance(service, dict)
            if (
                service["id"].endswith(":built-in")
                or service["id"] == "workspace-manager:default"
                or (
                    not can_access_workspace
                    and service["config"].get("visibility") != "public"
                )
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

        for service in client_info.services:
            # Add workspace info to the service
            service.config.workspace = self._workspace
            if ":" not in service.id:
                service.id = client_info.id + ":" + service.id

        # Store previous services for detecting changes
        previous_client_info = await self._redis.hget(
            f"{self._workspace}:clients", client_info.id
        )
        previous_client_info = ClientInfo.parse_obj(
            json.loads(previous_client_info.decode())
        )

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

        # Detect changes
        service_ids = [service.id for service in client_info.services]
        previous_service_ids = [service.id for service in previous_client_info.services]
        # Detect removed services
        for service in previous_client_info.services:
            if service.id not in service_ids:
                service.config.workspace = self._workspace
                self._event_bus.emit("service_unregistered", service.dict())
        # Detect new services
        for service in client_info.services:
            if service.id not in previous_service_ids:
                service.config.workspace = self._workspace
                self._event_bus.emit("service_registered", service.dict())
                # Try to execute the setup function for the default service
                if service.id.endswith(":default"):
                    rpc = await self.setup()
                    wm = await rpc.get_remote_service(
                        self._workspace + "/" + service.id, timeout=10
                    )
                    if callable(wm.setup):
                        await wm.setup()

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
        if client_info.parent:
            # parent must be an absolute client id
            assert "/" in client_info.parent
            await self._redis.sadd(
                f"client:{client_info.parent}:children",
                self._workspace + "/" + client_info.id,
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
        # Unregister all the services
        for service in client_info.services:
            service.config.workspace = workspace
            self._event_bus.emit("service_unregistered", service.dict())
        logger.info("Client deleted: %s/%s", workspace, client_id)

        if client_info.parent:
            # Remove from the parent's children
            await self._redis.srem(
                f"client:{client_info.parent}:children",
                workspace + "/" + client_info.id,
            )

        # Remove children if the user is anonymous
        if client_info.user_info.is_anonymous:
            client_keys = await self._redis.smembers(
                f"client:{workspace}/{client_info.id}:children"
            )
            children = [k.decode() for k in client_keys]
            logger.info(
                "Removing children client of %s (count: %d)",
                client_info.id,
                len(children),
            )
            for client_id in children:
                ws, cid = client_id.split("/")
                try:
                    await self.delete_client(cid, ws)
                except Exception as exp:
                    logger.warning("Failed to remove client: %s, error: %s", cid, exp)

        user_info = client_info.user_info
        if user_info.is_anonymous:
            remain_clients = await self.list_user_clients({"user": user_info.dict()})
            if len(remain_clients) <= 0:
                await self._redis.hdel("users", user_info.id)
                logger.info("Anonymous user (%s) removed.", user_info.id)
            else:
                logger.info(
                    "Anonymous user (%s) client removed (remaining clients: %s)",
                    user_info.id,
                    len(remain_clients),
                )
        self._event_bus.emit("client_deleted", client_info.dict())

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

    async def get_workspace_info(self, workspace: str = None) -> WorkspaceInfo:
        """Get info of the current workspace."""
        workspace = workspace or self._workspace
        workspace_info = await self._redis.hget("workspaces", workspace)
        if workspace_info is None:
            raise KeyError(f"Workspace not found: {workspace}")
        workspace_info = WorkspaceInfo.parse_obj(json.loads(workspace_info.decode()))
        return workspace_info

    async def _get_workspace_info_dict(
        self, workspace: str = None, context=None
    ) -> dict:
        user_info = UserInfo.parse_obj(context["user"])
        if not await self.check_permission(user_info):
            raise Exception(f"Permission denied for workspace {self._workspace}.")
        info = await self.get_workspace_info(workspace)
        return info.dict()

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
        if workspace == "*":
            workspace = self._workspace
        # Check if the user has permission to this workspace and the one to be launched
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
            client_id=client_id,
            timeout=timeout,
            wait_for_service=service_id,
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
        # Note: No authorization required
        # because the access is controlled by the client itself
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
                    f"client_id ({query['client_id']}) does"
                    f" not match service_id ({service_id})"
                )
            query["client_id"] = service_id.split("/")[1]
        elif "/" not in service_id and ":" not in service_id:
            # workspace=* means the current workspace or the public workspace
            workspace = query.get("workspace", "*")
            service_id = workspace + "/*:" + service_id
            query["workspace"] = workspace
            query["client_id"] = "*"
        elif "/" not in service_id and ":" in service_id:
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

        logger.info("Getting service: %s", query)

        # service id with an abosolute client id
        if "*" not in service_id:
            if "/" in service_id:
                workspace = service_id.split("/")[0]
                if workspace != self._workspace:
                    rpc = await self.setup()
                    service_api = await rpc.get_remote_service(service_id)
                    return service_api
            if "/" not in service_id:
                service_id = self._workspace + "/" + service_id
                workspace = self._workspace

            # Make sure the client exists
            client_id = service_id.split("/")[1].split(":")[0]
            if await self._get_client_info(client_id):
                rpc = await self.setup()
                service_api = await rpc.get_remote_service(service_id)
                return self.patch_service_config(workspace, service_api)
            elif "launch" in query and query["launch"] is True:
                service_api = await self._launch_application_by_service(
                    query,
                    context=context,
                )
                # No need to patch the service config
                # because the service is already patched
                return service_api
            else:
                raise Exception(f"Client not found: {client_id}")
        else:
            workspace, sname = service_id.split("/")
            cid, sid = sname.split(":")
            assert cid == "*", "Specifying a client id is not supported: " + cid
            if sid == "*":  # Skip the client check if the service id is *
                services = []
            elif workspace == "*":
                # Check if it's in the current workspace
                services = await self.list_services(
                    {"workspace": self._workspace}, context=context
                )
                services = list(
                    filter(lambda service: service["id"].endswith(":" + sid), services)
                )
                if not services:
                    # Try to find the service in "public"
                    services = await self.list_services("public", context=context)
            elif workspace != self._workspace:
                assert await self._redis.hget("workspaces", workspace)
                ws = await self.get_workspace(workspace)
                services = await ws.list_services()
            else:  # only the service id without client id
                workspace = self._workspace
                services = await self.list_services(context=context)
            services = list(
                filter(lambda service: service["id"].endswith(":" + sid), services)
            )

        if not services:
            if "launch" in query and query["launch"] is True:
                service_api = await self._launch_application_by_service(
                    query,
                    context=context,
                )
                # No need to patch the service config
                # because the service is already patched
                return service_api
            raise IndexError(f"Service not found: {sid} in workspace {workspace}")
        service_info = random.choice(services)
        rpc = await self.setup()
        service_api = await rpc.get_remote_service(service_info["id"])
        return self.patch_service_config(service_info["id"].split("/")[0], service_api)

    def patch_service_config(self, workspace, service_api):
        service_api["config"]["workspace"] = workspace
        return service_api

    async def _get_all_workspace(self):
        """Get all workspaces."""
        workspaces = await self._redis.hgetall("workspaces")
        return [
            WorkspaceInfo.parse_obj(json.loads(v.decode())) for v in workspaces.values()
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
        if user_info.id == "root" or user_info.parent == "root":
            return True

        if workspace.name == user_info.id:
            return True

        if user_info.parent:
            parent = await self._redis.hget("users", user_info.parent)
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
            except asyncio.TimeoutError:
                logger.info(
                    "Failed to get the workspace manager service of %s", workspace
                )

        manager = WorkspaceManager.get_manager(
            workspace,
            self._redis,
            self._root_user,
            self._event_bus,
            self._server_info,
        )
        await manager.setup(client_id="workspace-manager")
        rpc = await self.setup()
        wm = await rpc.get_remote_service(
            workspace + "/workspace-manager:default", timeout=10
        )
        return wm

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
        if b"workspace-manager" in client_keys:
            client_keys.remove(b"workspace-manager")
        if not client_keys:
            await self.delete()

    async def delete(self, force: bool = False):
        """Delete the workspace."""
        winfo = await self.get_workspace_info(self._workspace)
        if force:
            await self.remove_clients(self._workspace)
            await self._redis.hdel("workspaces", self._workspace)
            self._event_bus.emit("workspace_removed", winfo.dict())
        else:
            client_keys = await self._redis.hkeys(f"{self._workspace}:clients")
            if b"workspace-manager" in client_keys:
                client_keys.remove(b"workspace-manager")
            if not client_keys and not winfo.persistent:
                await self.delete(force=True)

    def create_service(self, service_id, service_name=None):
        interface = {
            "id": service_id,
            "name": service_name or service_id,
            # Note: We make these services public by default, and assuming we will do authorization in each function
            "config": {
                "require_context": True,
                "workspace": self._workspace,
                "visibility": "protected" if self._workspace != "public" else "public",
            },
            "echo": self.echo,
            "log": self.log,
            "info": self.info,
            "error": self.error,
            "warning": self.warning,
            "critical": self.critical,
            "update_client_info": self.update_client_info,
            "get_connection_info": self.get_connection_info,
            "getConnectionInfo": self.get_connection_info,
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
            # "get_workspace": self.get_workspace,
            # "getWorkspace": self.get_workspace,
            "get_workspace_info": self._get_workspace_info_dict,
            "getWorkspaceInfo": self._get_workspace_info_dict,
            "set": self._update_workspace,
            "install_application": self.install_application,
            "installApplication": self.install_application,
            "uninstall_application": self.uninstall_application,
            "uninstallApplication": self.uninstall_application,
            "get_summary": self.get_summary,
            "getSummary": self.get_summary,
        }
        interface["config"].update(self._server_info)
        return interface
