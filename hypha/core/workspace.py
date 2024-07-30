import asyncio
import re
import json
import logging
import random
import sys
from typing import Optional, Union, List
from contextlib import asynccontextmanager

from fakeredis import aioredis
from hypha_rpc import RPC
from pydantic import BaseModel

from hypha.core import (
    Card,
    RedisRPCConnection,
    UserInfo,
    WorkspaceInfo,
    ServiceInfo,
    UserPermission,
)
from hypha.core.auth import generate_presigned_token, create_scope
from hypha.utils import EventBus, random_id

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("workspace")
logger.setLevel(logging.INFO)

SERVICE_SUMMARY_FIELD = ["id", "name", "type", "description", "config"]

# Ensure the client_id is safe
_allowed_characters = re.compile(r"^[a-zA-Z0-9-_/*]*$")


def validate_key_part(key_part: str):
    """Ensure key parts only contain safe characters."""
    if not _allowed_characters.match(key_part):
        raise ValueError(f"Invalid characters in query part: {key_part}")


class WorkspaceManager:
    def __init__(
        self,
        redis: aioredis.FakeRedis,
        root_user: UserInfo,
        event_bus: EventBus,
        server_info: dict,
        workspace_loader: Optional[callable] = None,
    ):
        self._redis = redis
        self._initialized = False
        self._rpc = None
        self._root_user = root_user
        self._event_bus = event_bus
        self._server_info = server_info
        self._client_id = None
        self._workspace_loader = workspace_loader

    def get_client_id(self):
        assert self._client_id is not None, "Manager client id not set."
        return self._client_id

    async def setup(
        self,
        service_id="default",
        service_name="Default workspace management service",
    ):
        """Setup the workspace manager."""
        if self._initialized:
            return self._rpc
        self._client_id = "workspace-manager-" + random_id(readable=False)
        rpc = self._create_rpc(self._client_id)
        self._rpc = rpc
        rpc.on("service-added", self._add_service_handler)
        rpc.on("service-removed", self._remove_service_handler)
        management_service = self.create_service(service_id, service_name)
        await rpc.register_service(management_service, notify=False)
        self._initialized = True
        return rpc

    async def get_summary(self, context: Optional[dict] = None) -> dict:
        """Get a summary about the workspace."""
        assert context is not None
        ws = context["ws"]
        workspace_info = await self.load_workspace_info(ws)
        workspace_info = workspace_info.model_dump()
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
                    {k: service[k] for k in SERVICE_SUMMARY_FIELD if k in service}
                    for service in services
                ],
            }
        )
        return summary

    def _validate_workspace_name(self, name):
        """Validate the workspace name."""
        if not name:
            raise ValueError("Workspace name must not be empty.")
        # only allow numbers, letters in lower case, hyphens, underscores and |
        # use a regex to validate the workspace name
        pattern = re.compile(r"^[a-z0-9-_|]*$")
        if not pattern.match(name):
            raise ValueError(f"Invalid workspace name: {name}, must match {pattern}")
        if name in [
            "protected",
            "private",
            "default",
            "built-in",
            "all",
            "admin",
            "system",
            "server",
        ]:
            raise ValueError("Invalid workspace name: " + name)
        return name

    async def _bookmark_workspace(
        self,
        workspace: WorkspaceInfo,
        user_info: UserInfo,
        context: Optional[dict] = None,
    ):
        """Bookmark the workspace for the user."""
        assert isinstance(workspace, WorkspaceInfo) and isinstance(user_info, UserInfo)
        user_workspace = await self.load_workspace_info(user_info.id)
        user_workspace.config = user_workspace.config or {}
        if "bookmarks" not in user_workspace.config:
            user_workspace.config["bookmarks"] = []
        user_workspace.config["bookmarks"].append(
            {
                "type": "workspace",
                "name": workspace.name,
                "description": workspace.description,
            }
        )
        await self._update_workspace(user_workspace, user_info)

    async def _get_bookmarked_workspaces(
        self, user_info: UserInfo, context: Optional[dict] = None
    ) -> List[dict]:
        """Get the bookmarked workspaces for the user."""
        try:
            user_workspace = await self.load_workspace_info(user_info.id)
            return user_workspace.config.get("bookmarks", [])
        except KeyError:
            return []

    async def create_workspace(
        self,
        config: Union[dict, WorkspaceInfo],
        overwrite=False,
        context: Optional[dict] = None,
    ):
        """Create a new workspace."""
        assert context is not None
        if isinstance(config, WorkspaceInfo):
            config = config.model_dump()
        assert "name" in config, "Workspace name must be provided."
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if user_info.is_anonymous:
            raise Exception("Only registered user can create workspace.")
        if await self._redis.hexists("workspace", config["name"]):
            if not overwrite:
                raise KeyError(f"Workspace already exists: {ws}")

        config["persistent"] = config.get("persistent") or False
        if user_info.is_anonymous and config["persistent"]:
            raise Exception("Only registered user can create persistent workspace.")
        workspace = WorkspaceInfo.model_validate(config)
        if user_info.id not in workspace.owners:
            workspace.owners.append(user_info.id)
        self._validate_workspace_name(workspace.name)
        # make sure we add the user's email to owners
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]

        # user workspace, let's store all the created workspaces
        if user_info.id == workspace.name:
            workspace.config = workspace.config or {}
            workspace.config["bookmarks"] = [
                {
                    "type": "workspace",
                    "name": workspace.name,
                    "description": workspace.description,
                }
            ]
        await self._redis.hset(
            "workspaces", workspace.name, workspace.model_dump_json()
        )
        await self._event_bus.emit("workspace_loaded", workspace.model_dump())
        if user_info.id != workspace.name:
            await self._bookmark_workspace(workspace, user_info, context=context)
        return workspace.model_dump()

    async def install_application(self, card: dict, context: Optional[dict] = None):
        """Install an application to the workspace."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        card = Card.model_validate(card)
        # TODO: check if the application is already installed
        workspace_info = await self.load_workspace_info(ws)
        workspace_info.applications[card.id] = card
        logger.info("Installing application %s to %s", card.id, ws)
        await self._update_workspace(workspace_info, user_info)

    async def uninstall_application(self, card_id: str, context: Optional[dict] = None):
        """Uninstall a application from the workspace."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        workspace_info = await self.load_workspace_info(ws)
        if card_id not in workspace_info.applications:
            raise KeyError("Application not found: " + card_id)
        del workspace_info.applications[card_id]
        logger.info("Uninstalling application %s from %s", card_id, ws)
        await self._update_workspace(workspace_info, user_info)

    async def register_service(self, service, context: Optional[dict] = None, **kwargs):
        """Register a service"""
        assert context is not None
        ws = context["ws"]
        source_workspace = context["ws"]
        assert source_workspace == ws, (
            f"Service must be registered in the same workspace: "
            f"{source_workspace} != {ws} (current workspace)."
        )
        logger.info("Registering service %s to %s", service.id, ws)
        sv = await self._rpc.get_remote_service(context["from"] + ":built-in")
        service["config"] = service.get("config", {})
        service["config"]["workspace"] = ws
        assert "/" not in service["id"], "Service id must not contain '/'"
        assert ":" not in service["id"], "Service id must not contain ':'"
        svc = await sv.register_service(service, **kwargs)
        assert ":" in svc["id"], "Service id info must contain ':'"
        svc["id"] = ws + "/" + svc["id"]
        return svc

    async def generate_token(
        self, config: Optional[dict] = None, context: Optional[dict] = None
    ):
        """Generate a token for the current workspace."""
        assert context is not None
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        config = config or {}
        if set(config.keys()) - {
            "expires_in",
            "workspace",
            "permission",
            "extra_scopes",
        }:
            raise ValueError(
                "Invalid keys in token config: "
                + str(
                    set(config.keys())
                    - {"expires_in", "workspace", "permission", "extra_scopes"}
                )
            )

        extra_scopes = []
        if "extra_scopes" in config:
            assert isinstance(
                config["extra_scopes"], list
            ), "Permissions must be a list"
            for scope in config["extra_scopes"]:
                assert isinstance(scope, str), "Permission must be a string"
                if ":" in scope and scope.count(":") == 1:
                    assert scope.split(":")[0] not in [
                        "ws",
                        "cid",
                    ], "Invalid scope, cannot start with ws or cid"
                extra_scopes.append(scope)

        if "workspace" in config:
            allowed_workspace = config["workspace"]
        else:
            # limit to the current workspace
            allowed_workspace = ws

        maximum_permission = user_info.get_permission(allowed_workspace)
        if not maximum_permission:
            workspace_info = await self.load_workspace_info(allowed_workspace)
            if workspace_info.owners and user_info.id in workspace_info.owners:
                maximum_permission = UserPermission.admin
            else:
                raise PermissionError(
                    f"You do not have any permission for workspace: {config['workspace']}"
                )
        if maximum_permission != UserPermission.admin:
            raise PermissionError("Only admin can generate token.")

        permission = config.get("permission", "read_write")
        assert permission in [
            "read",
            "read_write",
            "admin",
        ], f"Invalid permission: {permission} (must be one of: read, read_write, admin)"
        permission = UserPermission[permission]

        # make it a child
        user_info.id = random_id(readable=True)
        user_info.scope = create_scope(
            {allowed_workspace: permission}, extra_scopes=extra_scopes
        )
        token = generate_presigned_token(user_info)
        return token

    async def list_clients(self, workspace: str = None, context: Optional[dict] = None):
        """Return a list of clients based on the services."""
        assert context is not None
        cws = context["ws"]
        workspace = workspace or cws
        pattern = f"services:*:{workspace}/*:built-in@*"
        keys = await self._redis.keys(pattern)
        clients = set()
        for key in keys:
            # Extract the client ID from the service key
            key_parts = key.decode("utf-8").split("/")
            client_id = key_parts[1].split(":")[0]
            clients.add(workspace + "/" + client_id)
        return list(clients)

    async def ping_client(self, client_id: str, context: Optional[dict] = None):
        """Ping a client."""
        assert context is not None
        ws = context["ws"]
        if "/" not in client_id:
            client_id = ws + "/" + client_id
        try:
            svc = await self._rpc.get_remote_service(client_id + ":built-in")
            return await svc.ping("ping")  # should return "pong"
        except Exception as e:
            return f"Failed to ping client {client_id}: {e}"

    async def list_services(
        self, query: Optional[Union[dict, str]] = None, context: Optional[dict] = None
    ):
        """Return a list of services based on the query."""
        assert context is not None
        cws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if query is None:
            # list all services in the current workspace
            query = {
                "visibility": "*",
                "workspace": cws,
                "client_id": "*",
                "service_id": "*",
            }
        # Convert string query into a dictionary
        if isinstance(query, str):
            visibility = "*"
            workspace = "*"
            client_id = "*"
            service_id = "*"

            if query == "public":
                query = {
                    "visibility": visibility,
                    "workspace": workspace,
                    "client_id": client_id,
                    "service_id": service_id,
                }
            elif "/" in query and ":" in query:
                parts = query.split("/")
                workspace_part = parts[0]
                remaining = parts[1]
                client_service = remaining.split(":")
                client_id = client_service[0]
                service_id = client_service[1] if len(client_service) > 1 else "*"
                if workspace_part == "*":
                    visibility = "public"
                else:
                    workspace = workspace_part
                query = {
                    "visibility": visibility,
                    "workspace": workspace,
                    "client_id": client_id,
                    "service_id": service_id,
                }
            elif ":" in query:
                client_service = query.split(":")
                client_id = client_service[0]
                service_id = client_service[1] if len(client_service) > 1 else "*"
                query = {
                    "visibility": visibility,
                    "workspace": workspace,
                    "client_id": client_id,
                    "service_id": service_id,
                }
            elif "/" in query:
                parts = query.split("/")
                workspace_part = parts[0]
                service_id = parts[1] if len(parts) > 1 else "*"
                if workspace_part == "*":
                    visibility = "public"
                else:
                    workspace = workspace_part
                query = {
                    "visibility": visibility,
                    "workspace": workspace,
                    "client_id": client_id,
                    "service_id": service_id,
                }
            else:
                service_id = query
                query = {
                    "visibility": visibility,
                    "workspace": workspace,
                    "client_id": client_id,
                    "service_id": service_id,
                }
        else:
            if "id" in query:
                assert (
                    "service_id" not in query
                ), "Cannot specify both 'id' and 'service_id' in the query."
                query["service_id"] = query["id"]
                del query["id"]

        # Automatically limit visibility to public if workspace is "*"
        original_visibility = query.get("visibility", "*")
        workspace = query.get("workspace", "*")
        if workspace == "*":
            assert (
                original_visibility != "protected"
            ), "Cannot list protected services in all workspaces."
            query["visibility"] = "public"
        elif workspace not in ["public", cws]:
            # Check user permission for the specified workspace only once
            if user_info.check_permission(workspace, UserPermission.read):
                raise PermissionError(f"Permission denied for workspace {workspace}")

        visibility = query.get("visibility", "*")
        client_id = query.get("client_id", "*")
        service_id = query.get("service_id", "*")
        type_filter = query.get("type", None)
        app_id = "*"
        if "@" in service_id:
            service_id, app_id = service_id.split("@")
            if query.get("app_id") and query.get("app_id") != app_id:
                raise ValueError(f"App id mismatch: {query.get('app_id')} != {app_id}")
        app_id = query.get("app_id", app_id)

        # If the user does not have permission to read the workspace, only list public services
        if not user_info.check_permission(workspace, UserPermission.read):
            visibility = "public"

        # Make sure query doesn't contain other keys
        # except for any of: visibility, workspace, client_id, service_id, and type
        allowed_keys = {
            "visibility",
            "workspace",
            "client_id",
            "service_id",
            "type",
            "app_id",
        }
        if set(query.keys()) - allowed_keys:
            logger.error(
                f"Failed to list services, invalid keys: {set(query.keys()) - allowed_keys}"
            )
            raise ValueError(f"Invalid query keys: {set(query.keys()) - allowed_keys}")
        # Validate key parts
        validate_key_part(visibility)
        validate_key_part(workspace)
        validate_key_part(client_id)
        validate_key_part(service_id)
        validate_key_part(app_id)

        pattern = f"services:{visibility}:{workspace}/{client_id}:{service_id}@{app_id}"

        assert pattern.startswith(
            "services:"
        ), "Query pattern does not start with 'services:'."
        assert not any(
            char in pattern for char in "{}"
        ), "Query pattern contains invalid characters."

        keys = await self._redis.keys(pattern)

        if workspace == "*":
            # add services in the current workspace
            ws_pattern = f"services:{original_visibility}:{cws}/{client_id}:{service_id}@{app_id}"
            keys = keys + await self._redis.keys(ws_pattern)

        services = []
        for key in set(keys):
            service_data = await self._redis.hgetall(key)
            converted_service_data = {}
            for k, v in service_data.items():
                key_str = k.decode("utf-8")
                value_str = v.decode("utf-8")
                if (
                    value_str.startswith("{")
                    and value_str.endswith("}")
                    or value_str.startswith("[")
                    and value_str.endswith("]")
                ):
                    converted_service_data[key_str] = json.loads(value_str)
                else:
                    converted_service_data[key_str] = value_str
            if isinstance(query, dict) and type_filter:
                if converted_service_data.get("type") == type_filter:
                    services.append(converted_service_data)
            else:
                services.append(converted_service_data)

        return services

    async def _add_service_handler(self, message: dict):
        """Add a service to the workspace."""
        service = message["service"]
        service = ServiceInfo.model_validate(service)
        ws = message["ws"]
        client_id = message["from"]
        service.config.workspace = ws
        if "/" not in service.id:
            service.id = f"{ws}/{service.id}"
        assert ":" in service.id, "Service id info must contain ':'"
        service.app_id = service.app_id or "*"
        key = (
            f"services:{service.config.visibility.value}:{service.id}@{service.app_id}"
        )

        # Check if the service already exists
        service_exists = await self._redis.exists(key)
        await self._redis.hset(key, mapping=service.to_redis_dict())

        if service_exists:
            if ":built-in@" in key:
                self._event_bus.emit(
                    "client_updated", {"id": client_id, "workspace": ws}
                )
                logger.info(f"Updating built-in service: {service.id}")
            else:
                self._event_bus.emit("service_updated", service.model_dump())
                logger.info(f"Updating service: {service.id}")
        else:
            # Default service created by api.export({}), typically used for hypha apps
            if ":default@" in key:
                try:
                    svc = await self._rpc.get_remote_service(client_id + ":default")
                    if svc.setup:
                        await svc.setup()
                except Exception as e:
                    logger.error(
                        f"Failed to run setup for default service `{client_id}`: {e}"
                    )
            if ":built-in@" in key:
                self._event_bus.emit(
                    "client_connected", {"id": client_id, "workspace": ws}
                )
                logger.info(f"Adding built-in service: {service.id}")
            else:
                self._event_bus.emit("service_added", service.model_dump(mode="json"))
                logger.info(f"Adding service {service.id}")

    async def _remove_service_handler(self, message: dict):
        """Remove a service from the workspace."""
        service = message["service"]
        service = ServiceInfo.model_validate(service)
        ws = message["ws"]
        client_id = message["from"]
        service.config.workspace = ws
        if "/" not in service.id:
            service.id = f"{ws}/{service.id}"
        assert ":" in service.id, "Service id info must contain ':'"
        service.app_id = service.app_id or "*"
        key = (
            f"services:{service.config.visibility.value}:{service.id}@{service.app_id}"
        )
        logger.info("Removing service: %s", key)

        # Check if the service exists before removal
        service_exists = await self._redis.exists(key)

        if service_exists:
            await self._redis.delete(key)
            if ":built-in@" in key:
                self._event_bus.emit(
                    "client_disconnected", {"id": client_id, "workspace": ws}
                )
            else:
                self._event_bus.emit("service_removed", service.model_dump())
        else:
            logger.warning(f"Service {key} does not exist and cannot be removed.")

    def _create_rpc(
        self,
        client_id: str,
        default_context=None,
        user_info: UserInfo = None,
        workspace="*",
        manager_id=None,
        silent=False,
    ):
        """Create a rpc for the workspace.
        Note: Any request made through this rcp will be treated as a request from the root user.
        """
        assert "/" not in client_id
        logger.info("Creating RPC for client %s", client_id)
        assert isinstance(user_info, UserInfo) or user_info is None
        connection = RedisRPCConnection(
            self._event_bus,
            workspace,
            client_id,
            (user_info or self._root_user),
            manager_id=manager_id,
        )
        rpc = RPC(
            connection,
            workspace=workspace,
            client_id=client_id,
            default_context=default_context,
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

    def validate_context(self, context: dict, permission: UserPermission):
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(context["ws"], permission):
            raise PermissionError(f"Permission denied for workspace {context['ws']}")

    async def echo(self, data, context=None):
        """Log a app message."""
        self.validate_context(context, permission=UserPermission.read)
        return data

    async def log(self, msg, context=None):
        """Log a app message."""
        self.validate_context(context, permission=UserPermission.read)
        logger.info("%s: %s", context["from"], msg)

    async def info(self, msg, context=None):
        """Log a app message."""
        self.validate_context(context, permission=UserPermission.read)
        logger.info("%s: %s", context["from"], msg)

    async def warning(self, msg, context=None):
        """Log a app message (warning)."""
        self.validate_context(context, permission=UserPermission.read)
        logger.warning("WARNING: %s: %s", context["from"], msg)

    async def error(self, msg, context=None):
        """Log a app error message (error)."""
        self.validate_context(context, permission=UserPermission.read)
        logger.error("%s: %s", context["from"], msg)

    async def critical(self, msg, context=None):
        """Log a app error message (critical)."""
        self.validate_context(context, permission=UserPermission.read)
        logger.critical("%s: %s", context["from"], msg)

    async def load_workspace_info(self, workspace: str, load=True) -> WorkspaceInfo:
        """Load info of the current workspace from the redis store."""
        assert workspace is not None
        try:
            workspace_info = await self._redis.hget("workspaces", workspace)
            if workspace_info is None:
                raise KeyError(f"Workspace not found: {workspace}")
            workspace_info = WorkspaceInfo.model_validate(
                json.loads(workspace_info.decode())
            )
            return workspace_info
        except KeyError:
            if load and self._workspace_loader:
                workspace_info = await self._workspace_loader(
                    workspace, self._root_user
                )
                if not workspace_info:
                    raise KeyError(f"Workspace not found: {workspace}")
                await self._redis.hset(
                    "workspaces", workspace_info.name, workspace_info.model_dump_json()
                )
                self._event_bus.emit("workspace_loaded", workspace_info.model_dump())
                return workspace_info
            elif load and not self._workspace_loader:
                raise KeyError(
                    "Workspace not found and the workspace loader is not configured."
                )
            else:
                raise KeyError(f"Workspace not found: {workspace}")
        except Exception as e:
            logger.error(f"Failed to load workspace info: {e}")
            raise e

    async def get_workspace_info(self, workspace: str = None, context=None) -> dict:
        """Get the workspace info."""
        self.validate_context(context, permission=UserPermission.read)
        workspace_info = await self.load_workspace_info(workspace)
        return workspace_info.model_dump()

    async def _launch_application_for_service(
        self,
        query: dict,
        timeout: float = 60,
        context: dict = None,
    ):
        """Launch an installed application by service name."""
        self.validate_context(context, permission=UserPermission.read)
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        workspace, service_id = (
            query.get("workspace", ws),
            query.get("service_id"),
        )
        assert "app_id" in query, "App id must be provided."
        app_id = query["app_id"]
        assert app_id not in ["*"], f"Invalid app id: {app_id}"
        if workspace == "*":
            workspace = ws

        if not user_info.check_permission(workspace, UserPermission.read):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        if ":" in service_id:
            service_id = service_id.split(":")[1]
        assert service_id and service_id not in [
            "*",
            "default",
            "built-in",
        ], f"Invalid service id: {service_id}"

        # Check if the user has permission to this workspace and the one to be launched
        workspace = await self.load_workspace_info(workspace)
        if app_id not in workspace.applications:
            raise KeyError(
                f"Application id `{app_id}` not found in workspace {workspace.name}"
            )

        # Make sure the service_id is in the application services
        app_info = workspace.applications[app_id]
        assert (
            app_info.services
        ), f"No services found in application {app_id}, please make sure it's properly installed."
        # check if service_id is one of the service.id in the app
        found = False
        for svc in app_info.services:
            if svc.id.endswith(":" + service_id):
                found = True
                break
        if not found:
            raise KeyError(
                f"Service id `{service_id}` not found in application {app_id}"
            )

        async with self._get_service_api(
            "public/*:server-apps", context=context
        ) as controller:
            if not controller:
                raise Exception(
                    "Failed to launch application: server-apps service not found."
                )
            client_info = await controller.start(
                app_id,
                timeout=timeout,
                wait_for_service=service_id,
            )
            return await self.get_service(
                f"{client_info['id']}:{service_id}@{app_id}",
                context=context,
            )

    @asynccontextmanager
    async def _get_service_api(self, service_id: str, context=None):
        """Get the service api for the service."""
        self.validate_context(context, permission=UserPermission.read)
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        # Now launch the app and get the service
        svc = await self.get_service(service_id, mode="random", context=context)
        # Create a rpc client for getting the launcher service as user.
        rpc = self._create_rpc(
            "get-service-" + random_id(readable=False),
            user_info=user_info,
            workspace=ws,
            manager_id=self._client_id,
            silent=True,
        )
        api = await rpc.get_remote_service(svc["id"])
        yield api
        await rpc.disconnect()

    async def get_service(
        self,
        query: Union[dict, str],
        mode: str = "default",
        skip_timeout=False,
        timeout=5.0,
        context=None,
    ):
        """Get a service based on the query, supporting wildcard patterns and matching modes."""
        # no need to validate the context
        # self.validate_context(context, permission=UserPermission.read)
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        # Convert string query into a dictionary
        if isinstance(query, str):
            service_id = query
            query = {"id": service_id}
        else:
            if "id" not in query:
                service_id = query.get("service_id", "*")
            else:
                service_id = query["id"]

        assert isinstance(service_id, str)

        assert service_id.count("/") <= 1, "Service id must contain at most one '/'"
        assert service_id.count(":") <= 1, "Service id must contain at most one ':'"
        assert service_id.count("@") <= 1, "Service id must contain at most one '@'"

        if "/" in service_id and ":" not in service_id:
            service_id += ":default"
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
            service_id = f"{workspace}/*:{service_id}"
            query["workspace"] = workspace
            query["client_id"] = "*"
        elif "/" not in service_id and ":" in service_id:
            workspace = query.get("workspace", ws)
            query["client_id"] = service_id.split(":")[0]
            service_id = f"{workspace}/{service_id}"
            query["workspace"] = workspace
        else:
            assert "/" in service_id and ":" in service_id
            workspace = service_id.split("/")[0]
            query["client_id"] = service_id.split("/")[1].split(":")[0]
            query["workspace"] = workspace
            if "*" not in service_id:
                service_api = await self._rpc.get_remote_service(
                    service_id, timeout=timeout
                )
                if service_api:
                    return self.patch_service_config(workspace, service_api)
                else:
                    return None
        app_id = "*"
        if "@" in service_id:
            service_id, app_id = service_id.split("@")
            if query.get("app_id") and query.get("app_id") != app_id:
                raise ValueError(f"App id mismatch: {query.get('app_id')} != {app_id}")
        query["app_id"] = query.get("app_id", app_id)

        query["service_id"] = service_id.split("/")[1].split(":")[1]

        logger.info("Getting service: %s", query)

        original_visibility = query.get("visibility", "*")
        # Construct the pattern for querying Redis
        if query.get("workspace", "*") == "*":
            visibility = "public"
        else:
            # If the user does not have permission to read the workspace, only list public services
            if not user_info.check_permission(query["workspace"], UserPermission.read):
                visibility = "public"
            else:
                visibility = "*"

        pattern = f"services:{visibility}:{query['workspace']}/{query['client_id']}:{query['service_id']}@{query['app_id']}"

        assert pattern.startswith(
            "services:"
        ), "Query pattern does not start with 'services:'."
        assert not any(
            char in pattern for char in "{}"
        ), "Query pattern contains invalid characters."

        logger.debug("Query services using pattern: %s", pattern)
        keys = await self._redis.keys(pattern)

        if query["workspace"] == "*":
            # add services in the current workspace
            ws_pattern = f"services:{original_visibility}:{ws}/{query['client_id']}:{query['service_id']}@{query['app_id']}"
            keys = keys + await self._redis.keys(ws_pattern)

        logger.debug("Found service keys: %s", keys)
        within_workspace_keys = []
        outside_workspace_keys = []

        for key in set(keys):
            key_workspace = key.decode("utf-8").split("/")[1]
            if key_workspace == ws:
                within_workspace_keys.append(key)
            else:
                outside_workspace_keys.append(key)

        if mode == "random":
            random.shuffle(within_workspace_keys)
            random.shuffle(outside_workspace_keys)
        else:
            within_workspace_keys.sort()
            outside_workspace_keys.sort()

        sorted_keys = within_workspace_keys + outside_workspace_keys

        for key in sorted_keys:
            try:
                # Fetch the minimal required data to determine the correct service
                parts = key.decode("utf-8").split(":")
                service_id = parts[2] + ":" + parts[3]
                service_id, app_id = service_id.split("@")
                workspace = service_id.split("/")[0]
                # Attempt to get the remote service with a timeout
                service_api = await self._rpc.get_remote_service(
                    service_id, timeout=timeout
                )
                if service_api:
                    return self.patch_service_config(workspace, service_api)
            except asyncio.TimeoutError:
                if skip_timeout:
                    logger.warning(
                        f"Timeout while getting service {service_id}, skipping to the next one."
                    )
                    continue
                else:
                    raise TimeoutError(f"Timeout while getting service {service_id}")

        if "app_id" in query and query["app_id"] != "*":
            service_api = await self._launch_application_for_service(
                query, context=context
            )
            # No need to patch the service config because the service is already patched
            return service_api

        return None

    def patch_service_config(self, workspace, service_api):
        service_api["config"]["workspace"] = workspace
        return service_api

    async def list_workspaces(self, type=None, context=None):
        """Get all workspaces."""
        self.validate_context(context, permission=UserPermission.read)
        user_info = UserInfo.model_validate(context["user"])
        workspace_info = await self._redis.hget("workspaces", context["ws"])
        if workspace_info is None:
            logger.warning(
                "Client is operating in a non-existing workspace: %s", context["ws"]
            )
            return []
        else:
            workspace = WorkspaceInfo.model_validate(
                json.loads(workspace_info.decode())
            )
        workspaces = await self._get_bookmarked_workspaces(user_info, context=context)
        workspaces = [
            {"name": w["name"], "description": w["description"]}
            for w in workspaces
            if w["name"] != workspace.name
        ]
        workspaces = workspaces + [
            {
                "name": workspace.name,
                "description": workspace.description,
            }
        ]
        return workspaces

    async def get_workspace(self, workspace: str = None):
        """Get the service api of the workspace manager."""
        assert workspace is not None
        rpc = self._rpc
        wm = await rpc.get_remote_service(
            f"{workspace}/{self._client_id}:default", timeout=10
        )
        wm.rpc = rpc
        wm.disconnect = rpc.disconnect
        wm.register_codec = rpc.register_codec
        return wm

    async def _update_workspace(
        self, workspace: WorkspaceInfo, user_info: UserInfo, overwrite=False
    ):
        """Update the workspace config."""
        assert isinstance(workspace, WorkspaceInfo) and isinstance(user_info, UserInfo)
        if not user_info.check_permission(workspace.name, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        workspace_info = await self._redis.hget("workspaces", workspace.name)
        if not overwrite and workspace_info is None:
            raise KeyError(f"Workspace not found: {workspace.name}")
        else:
            existing_workspace = WorkspaceInfo.model_validate(
                json.loads(workspace_info.decode())
            )
            assert existing_workspace.name == workspace.name, "Workspace name mismatch."
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]
        logger.info("Updating workspace %s", workspace.name)

        await self._redis.hset(
            "workspaces", workspace.name, workspace.model_dump_json()
        )
        self._event_bus.emit("workspace_changed", workspace.model_dump())

    async def delete_client(
        self,
        client_id: str,
        workspace: str,
        user_info: UserInfo,
        context: Optional[dict] = None,
    ):
        """Delete all services associated with the given client_id in the specified workspace."""
        assert context is not None
        self.validate_context(context, permission=UserPermission.admin)
        cws = workspace
        validate_key_part(client_id)

        # Define a pattern to match all services for the given client_id in the current workspace
        pattern = f"services:*:{cws}/{client_id}:*@*"

        # Ensure the pattern is secure
        assert not any(
            char in pattern for char in "{}"
        ), "Query pattern contains invalid characters."

        keys = await self._redis.keys(pattern)
        logger.info(f"Removing {len(keys)} services for client {client_id} in {cws}")
        for key in keys:
            await self._redis.delete(key)

        if await self._redis.hexists("workspaces", cws):
            if user_info.is_anonymous and cws == user_info.id:
                logger.info(f"Unloading workspace {cws} for anonymous user.")
                # unload temporary workspace if the user exits
                await self.unload(context=context)
            else:
                # otherwise delete the workspace if it is empty
                await self.unload_if_empty(context=context)
        else:
            logger.warning(f"Workspace {cws} not found.")

    async def unload_if_empty(self, context=None):
        """Delete the workspace if it is empty."""
        self.validate_context(context, permission=UserPermission.admin)
        client_keys = await self.list_clients(context=context)
        if not client_keys:
            await self.unload(context=context)
        else:
            logger.warning(
                f"Workspace {context['ws']} is not empty, remaining clients: {client_keys[:10]}..."  # only list the first 10 maximum
            )

    async def unload(self, context=None):
        """Unload the workspace."""
        self.validate_context(context, permission=UserPermission.admin)
        ws = context["ws"]
        winfo = await self.load_workspace_info(ws)

        # list all the clients in the workspace and send a meesage to delete them
        client_keys = await self.list_clients(context=context)
        if len(client_keys) > 0:
            # if there are too many, log the first 10
            client_summary = ", ".join(client_keys[:10]) + (
                "..." if len(client_keys) > 10 else ""
            )
            logger.info(
                f"There are {len(client_keys)} clients in the workspace {ws}: "
                + client_summary
            )
            self._event_bus.emit(f"unload:{ws}", "Unloading workspace: " + ws)

        await self._redis.hdel("workspaces", ws)
        self._event_bus.emit("workspace_unloaded", winfo.model_dump())
        logger.info("Workspace %s unloaded.", ws)

    def create_service(self, service_id, service_name=None):
        interface = {
            "id": service_id,
            "name": service_name or service_id,
            "description": "Services for managing workspace.",
            # Note: We make these services public by default, and assuming we will do authorization in each function
            "config": {
                "require_context": True,
                "visibility": "public",
            },
            "echo": self.echo,
            "log": self.log,
            "info": self.info,
            "error": self.error,
            "warning": self.warning,
            "critical": self.critical,
            "list_workspaces": self.list_workspaces,
            "listWorkspaces": self.list_workspaces,
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
            "get_workspace_info": self.get_workspace_info,
            "getWorkspaceInfo": self.get_workspace_info,
            "install_application": self.install_application,
            "installApplication": self.install_application,
            "uninstall_application": self.uninstall_application,
            "uninstallApplication": self.uninstall_application,
            "get_summary": self.get_summary,
            "getSummary": self.get_summary,
            "ping": self.ping_client,
        }
        interface["config"].update(self._server_info)
        return interface
