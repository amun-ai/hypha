import re
import json
import logging
import time
import sys
from typing import Optional, Union, List, Any
from contextlib import asynccontextmanager
import random

from fakeredis import aioredis
from hypha_rpc import RPC
from hypha_rpc.utils.schema import schema_method
from pydantic import BaseModel, Field

from hypha.core import (
    Card,
    RedisRPCConnection,
    UserInfo,
    WorkspaceInfo,
    ServiceInfo,
    RemoteService,
    TokenConfig,
    UserPermission,
    ServiceTypeInfo,
)
from hypha.core.auth import generate_presigned_token, create_scope, valid_token
from hypha.utils import EventBus, random_id

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("workspace")
logger.setLevel(logging.INFO)

SERVICE_SUMMARY_FIELD = ["id", "name", "type", "description", "config"]

# Ensure the client_id is safe
_allowed_characters = re.compile(r"^[a-zA-Z0-9-_/|*]*$")


def validate_key_part(key_part: str):
    """Ensure key parts only contain safe characters."""
    if not _allowed_characters.match(key_part):
        raise ValueError(f"Invalid characters in query part: {key_part}")


class GetServiceConfig(BaseModel):
    mode: Optional[str] = Field(
        None,
        description="Mode for selecting the service, it can be 'random', 'first', 'last' or 'exact'",
    )
    timeout: Optional[float] = Field(
        10.0,
        description="The timeout duration in seconds for fetching the service. This determines how long the function will wait for a service to respond before considering it a timeout.",
    )
    case_conversion: Optional[str] = Field(
        None,
        description="The case conversion for service keys, can be 'camel', 'snake' or None, default is None.",
    )


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
        self._client_id = "ws-" + random_id(readable=False)
        rpc = self._create_rpc(self._client_id)
        self._rpc = rpc
        management_service = self.create_service(service_id, service_name)
        await rpc.register_service(
            management_service,
            {"notify": False},
        )
        self._initialized = True
        return rpc

    @schema_method
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

        if "-" not in name:
            raise ValueError(
                "Workspace name must contain at least one hyphen (e.g. my-workspace)."
            )
        # only allow numbers, letters in lower case and hyphens (no underscore)
        # use a regex to validate the workspace name
        pattern = re.compile(r"^[a-z0-9-]*$")
        if not pattern.match(name):
            raise ValueError(
                f"Invalid workspace name: {name}, only lowercase letters, numbers and hyphens are allowed."
            )
        return name

    async def _bookmark_workspace(
        self,
        workspace: WorkspaceInfo,
        user_info: UserInfo,
        context: Optional[dict] = None,
    ):
        """Bookmark the workspace for the user."""
        assert isinstance(workspace, WorkspaceInfo) and isinstance(user_info, UserInfo)
        user_workspace = await self.load_workspace_info(user_info.get_workspace())
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
            user_workspace = await self.load_workspace_info(user_info.get_workspace())
            return user_workspace.config.get("bookmarks", [])
        except KeyError:
            return []

    @schema_method
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
        try:
            if await self.load_workspace_info(ws):
                if not overwrite:
                    raise KeyError(f"Workspace already exists: {ws}")
        except KeyError:
            pass

        config["persistent"] = config.get("persistent") or False
        if user_info.is_anonymous and config["persistent"]:
            raise Exception("Only registered user can create persistent workspace.")
        workspace = WorkspaceInfo.model_validate(config)
        if user_info.id not in workspace.owners:
            workspace.owners.append(user_info.id)
        if user_info.id != "root":
            self._validate_workspace_name(workspace.name)
        # make sure we add the user's email to owners
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]

        # user workspace, let's store all the created workspaces
        if user_info.get_workspace() == workspace.name:
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
        if user_info.get_workspace() != workspace.name:
            await self._bookmark_workspace(workspace, user_info, context=context)
        return workspace.model_dump()

    @schema_method
    async def install_application(
        self,
        app_info: Card = Field(..., description="Application info"),
        force: bool = Field(False, description="Force install if already installed"),
        context: Optional[dict] = None,
    ):
        """Install an application to the workspace."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        # TODO: check if the application is already installed
        workspace_info = await self.load_workspace_info(ws)
        if app_info.id in workspace_info.applications and not force:
            raise KeyError("Application already installed: " + app_info.id)
        workspace_info.applications[app_info.id] = app_info
        logger.info("Installing application %s to %s", app_info.id, ws)
        await self._update_workspace(workspace_info, user_info)

    @schema_method
    async def uninstall_application(
        self,
        app_id: str = Field(..., description="application id"),
        context: Optional[dict] = None,
    ):
        """Uninstall a application from the workspace."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        workspace_info = await self.load_workspace_info(ws)
        if app_id not in workspace_info.applications:
            raise KeyError("Application not found: " + app_id)
        del workspace_info.applications[app_id]
        logger.info("Uninstalling application %s from %s", app_id, ws)
        await self._update_workspace(workspace_info, user_info)

    @schema_method
    async def register_service_type(
        self,
        type_info: ServiceTypeInfo = Field(..., description="Service type info"),
        context: Optional[dict] = None,
    ):
        """Register a new service type."""
        assert context is not None
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {ws}")
        workspace_info = await self.load_workspace_info(ws)
        service_type_id = type_info.id
        if "/" in service_type_id:
            _ws, service_type_id = service_type_id.split("/")
            assert (
                _ws == ws
            ), f"Workspace mismatch: {_ws} != {ws}, you can only register service type in your own workspace."
        if service_type_id in workspace_info.service_types:
            raise KeyError(f"Service type already exists: {service_type_id}")

        workspace_info.service_types[service_type_id] = type_info
        await self._update_workspace(workspace_info, user_info)
        return type_info

    @schema_method
    async def revoke_token(
        self,
        token: str = Field(..., description="token to be revoked"),
        context: dict = None,
    ):
        """Revoke a token by storing it in Redis with a prefix and expiration time."""
        self.validate_context(context, UserPermission.admin)
        try:
            payload = valid_token(token)
        except Exception as e:
            raise ValueError(f"Invalid token: {e}")
        expiration = int(payload.get("exp") - time.time())
        if expiration > 0:
            await self._redis.setex("revoked_token:" + token, expiration, "revoked")
        else:
            raise ValueError("Token has already expired")

    @schema_method
    async def get_service_type(
        self,
        type_id: str = Field(
            ...,
            description="Service type ID, it can be the type id or the full  type id with workspace: `workspace/type_iid`",
        ),
        context: Optional[dict] = None,
    ):
        """Get a service type by ID."""
        assert context is not None
        if "/" not in type_id:
            ws = context["ws"]
        else:
            ws, type_id = type_id.split("/")

        user_info = UserInfo.model_validate(context["user"])
        workspace_info = await self.load_workspace_info(ws)

        if type_id in workspace_info.service_types:
            service_type = workspace_info.service_types[type_id]
            if service_type.config.get(
                "visibility"
            ) != "public" and not user_info.check_permission(ws, UserPermission.read):
                raise PermissionError(f"Permission denied for workspace {ws}")
            service_type = service_type.model_dump()
            service_type["id"] = f"{ws}/{type_id}"
            return service_type
        full_service_type_id = f"{ws}/{type_id}"
        if full_service_type_id in workspace_info.service_types:
            return workspace_info.service_types[full_service_type_id]

        raise KeyError(f"Service type not found: {type_id}")

    @schema_method
    async def list_service_types(
        self,
        workspace: str = Field(
            None,
            description="workspace name; if not provided, the current workspace will be used",
        ),
        context: Optional[dict] = None,
    ):
        """List all service types in the workspace."""
        assert context is not None
        ws = workspace or context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read):
            raise PermissionError(f"Permission denied for workspace {ws}")
        workspace_info = await self.load_workspace_info(ws)
        return list(workspace_info.service_types.values())

    @schema_method
    async def generate_token(
        self,
        config: Optional[TokenConfig] = Field(
            None, description="config for token generation"
        ),
        context: Optional[dict] = None,
    ):
        """Generate a token for a specified workspace."""
        assert context is not None, "Context cannot be None"
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        config = config or TokenConfig()
        assert isinstance(config, TokenConfig)

        extra_scopes = config.extra_scopes or []

        allowed_workspace = config.workspace or ws

        maximum_permission = user_info.get_permission(allowed_workspace)
        if not maximum_permission:
            workspace_info = await self.load_workspace_info(allowed_workspace)
            if workspace_info.owners and user_info.id in workspace_info.owners:
                maximum_permission = UserPermission.admin
            else:
                raise PermissionError(
                    f"You do not have any permission for workspace: {allowed_workspace}"
                )
        if maximum_permission != UserPermission.admin:
            raise PermissionError("Only admin can generate token.")

        permission = config.permission
        permission = UserPermission[permission]

        # make it a child
        user_info.id = random_id(readable=True)
        user_info.scope = create_scope(
            {allowed_workspace: permission},
            current_workspace=allowed_workspace,
            extra_scopes=extra_scopes,
        )
        config.expires_in = config.expires_in or 3600
        token = generate_presigned_token(user_info, config.expires_in)
        return token

    @schema_method
    async def list_clients(
        self,
        workspace: str = Field(None, description="workspace name"),
        context: Optional[dict] = None,
    ):
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

    @schema_method
    async def ping_client(
        self,
        client_id: str = Field(..., description="client id"),
        timeout: float = Field(5, description="timeout in seconds"),
        context: Optional[dict] = None,
    ):
        """Ping a client."""
        assert context is not None
        ws = context["ws"]
        if "/" not in client_id:
            client_id = ws + "/" + client_id
        try:
            svc = await self._rpc.get_remote_service(
                client_id + ":built-in", {"timeout": timeout}
            )
            return await svc.ping("ping")  # should return "pong"
        except Exception as e:
            return f"Failed to ping client {client_id}: {e}"

    @schema_method
    async def list_services(
        self,
        query: Optional[Union[dict, str]] = Field(
            None,
            description="Query for filtering services. This can be either a dictionary specifying detailed parameters like 'visibility', 'workspace', 'client_id', 'service_id', and 'type', or a string that includes these parameters in a formatted pattern. Examples: 'workspace/client_id:service_id' or 'workspace/service_id'.",
        ),
        context: Optional[dict] = None,
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

            if "/" in query and ":" in query:
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
                workspace = query
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
            service_dict = ServiceInfo.from_redis_dict(service_data).model_dump()
            if service_dict.get("config", {}).get(
                "visibility"
            ) != "public" and not user_info.check_permission(
                workspace, UserPermission.read
            ):
                logger.warning(
                    f"Potential issue in list_services: Protected service appear in the search list: {service_id}"
                )
                continue
            if isinstance(query, dict) and type_filter:
                if service_dict.get("type") == type_filter:
                    services.append(service_dict)
            else:
                services.append(service_dict)

        return services

    @schema_method
    async def register_service(
        self,
        service: ServiceInfo = Field(..., description="Service info"),
        config: Optional[dict] = Field(
            None, description="Options for registering service"
        ),
        context: Optional[dict] = None,
    ):
        """Register a new service."""
        assert context is not None
        ws = context["ws"]
        client_id = context["from"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {ws}")
        if "/" not in client_id:
            client_id = f"{ws}/{client_id}"
        service.config.workspace = ws
        if "/" not in service.id:
            service.id = f"{ws}/{service.id}"
        assert ":" in service.id, "Service id info must contain ':'"
        service.app_id = service.app_id or "*"

        service_name = service.id.split(":")[1]
        workspace = service.id.split("/")[0]
        key = f"services:*:{workspace}/*:{service_name}@*"
        peer_keys = await self._redis.keys(key)
        if len(peer_keys) > 0:
            for peer_key in peer_keys:
                peer_service = await self._redis.hgetall(peer_key)
                peer_service = ServiceInfo.from_redis_dict(peer_service)
                if peer_service.config.singleton:
                    raise ValueError(
                        f"A singleton service with the same name ({service_name}) has already exists in the workspace ({workspace}), please remove it first or use a different name."
                    )

        # Check if the clients exists if not a built-in service
        if ":built-in" not in service.id and ws not in ["ws-user-root", "public"]:
            builtins = await self._redis.keys(f"services:*:{client_id}:built-in@*")
            if not builtins:
                logger.warning(
                    "Refuse to add service %s, client %s has been removed.",
                    service.id,
                    client_id,
                )
                return
        # Check if the service already exists
        service_exists = await self._redis.exists(
            f"services:*:{service.id}@{service.app_id}"
        )
        key = (
            f"services:{service.config.visibility.value}:{service.id}@{service.app_id}"
        )
        await self._redis.hset(key, mapping=service.to_redis_dict())

        if service_exists:
            if ":built-in@" in key:
                await self._event_bus.emit(
                    "client_updated", {"id": client_id, "workspace": ws}
                )
                logger.info(f"Updating built-in service: {service.id}")
            else:
                await self._event_bus.emit("service_updated", service.model_dump())
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
                await self._event_bus.emit(
                    "client_connected", {"id": client_id, "workspace": ws}
                )
                logger.info(f"Adding built-in service: {service.id}")
                builtins = await self._redis.keys(f"services:*:{client_id}:built-in@*")
            else:
                await self._event_bus.emit(
                    "service_added", service.model_dump(mode="json")
                )
                logger.info(f"Adding service {service.id}")

    @schema_method
    async def get_service_info(
        self,
        service_id: str = Field(
            ...,
            description="Service id, it can be the service id or the full service id with workspace: `workspace/client_id:service_id`",
        ),
        config: Optional[dict] = Field(
            None,
            description="Options for getting service, the only available config is `mode`, for selecting the service, it can be 'random', 'first', 'last' or 'exact'",
        ),
        context: Optional[dict] = None,
    ):
        """Get the service info."""
        assert isinstance(service_id, str), "Service ID must be a string."
        assert service_id.count("/") <= 1, "Service id must contain at most one '/'"
        assert service_id.count(":") <= 1, "Service id must contain at most one ':'"
        assert service_id.count("@") <= 1, "Service id must contain at most one '@'"
        if "/" not in service_id:
            service_id = f"{context['ws']}/{service_id}"
        if ":" not in service_id:
            workspace, service_id = service_id.split("/")
            service_id = f"{workspace}/*:{service_id}"

        if "@" in service_id:
            service_id, app_id = service_id.split("@")
        else:
            app_id = "*"
        assert (
            "/" in service_id and ":" in service_id
        ), f"Invalid service id: {service_id}, it must contain '/' and ':'"
        workspace = service_id.split("/")[0]
        assert (
            workspace != "*"
        ), "You must specify a workspace for the service query, otherwise please call list_services to find the service."
        logger.info("Getting service: %s", service_id)

        user_info = UserInfo.model_validate(context["user"])
        key = f"services:*:{service_id}@{app_id}"
        keys = await self._redis.keys(key)
        if not keys:
            raise KeyError(f"Service not found: {service_id}@{app_id}")
        config = config or {}
        mode = config.get("mode")
        if mode is None:
            # Set random mode for public services, since there can be many hypha servers
            if workspace == "public":
                mode = "random"
            else:
                mode = "exact"
        if mode == "exact":
            assert len(keys) == 1, f"Multiple services found for {service_id}"
            key = keys[0]
        elif mode == "random":
            key = random.choice(keys)
        elif mode == "first":
            key = keys[0]
        elif mode == "last":
            key = keys[-1]
        else:
            raise ValueError(
                f"Invalid mode: {mode}, the mode must be 'random', 'first', 'last' or 'exact'"
            )
        # if it's a public service or the user has read permission
        if not key.startswith(b"services:public:") and not user_info.check_permission(
            workspace, UserPermission.read
        ):
            raise PermissionError(
                f"Permission denied for non-public service in workspace {workspace}"
            )
        service_data = await self._redis.hgetall(key)
        return ServiceInfo.from_redis_dict(service_data)

    @schema_method
    async def unregister_service(
        self,
        service_id: str = Field(..., description="Service id"),
        context: Optional[dict] = None,
    ):
        """Unregister a new service."""
        assert context is not None
        ws = context["ws"]
        client_id = context["from"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {ws}")

        service = await self.get_service_info(
            service_id, {"mode": "exact"}, context=context
        )
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
                await self._event_bus.emit(
                    "client_disconnected", {"id": client_id, "workspace": ws}
                )
            else:
                await self._event_bus.emit("service_removed", service.model_dump())
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
            server_base_url=self._server_info["public_base_url"],
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

    @schema_method
    async def echo(
        self, data: Any = Field(..., description="echo an object"), context: Any = None
    ):
        """Log a app message."""
        self.validate_context(context, permission=UserPermission.read)
        return data

    @schema_method
    async def log(
        self, msg: str = Field(..., description="log a message"), context=None
    ):
        """Log a app message."""
        self.validate_context(context, permission=UserPermission.read)
        logger.info("%s: %s", context["from"], msg)

    @schema_method
    async def info(
        self, msg: str = Field(..., description="log a message as info"), context=None
    ):
        """Log a app message."""
        self.validate_context(context, permission=UserPermission.read)
        logger.info("%s: %s", context["from"], msg)

    @schema_method
    async def warning(
        self, msg: str = Field(..., description="log a message as info"), context=None
    ):
        """Log a app message (warning)."""
        self.validate_context(context, permission=UserPermission.read)
        logger.warning("WARNING: %s: %s", context["from"], msg)

    @schema_method
    async def error(
        self, msg: str = Field(..., description="log an error message"), context=None
    ):
        """Log a app error message (error)."""
        self.validate_context(context, permission=UserPermission.read)
        logger.error("%s: %s", context["from"], msg)

    @schema_method
    async def critical(
        self, msg: str = Field(..., description="log an critical message"), context=None
    ):
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
                await self._event_bus.emit(
                    "workspace_loaded", workspace_info.model_dump()
                )
                return workspace_info
            elif load and not self._workspace_loader:
                raise KeyError(
                    f"Workspace ({workspace}) not found and the workspace loader is not configured (requires s3 enabled)."
                )
            else:
                raise KeyError(f"Workspace not found: {workspace}")
        except Exception as e:
            logger.error(f"Failed to load workspace info: {e}")
            raise e

    @schema_method
    async def get_workspace_info(
        self, workspace: str = Field(None, description="workspace name"), context=None
    ) -> dict:
        """Get the workspace info."""
        self.validate_context(context, permission=UserPermission.read)
        workspace_info = await self.load_workspace_info(workspace)
        return workspace_info.model_dump()

    async def _launch_application_for_service(
        self,
        app_id: str,
        service_id: str,
        workspace: str = None,
        timeout: float = 10,
        case_conversion: str = None,
        context: dict = None,
    ):
        """Launch an installed application by service name."""
        self.validate_context(context, permission=UserPermission.read)
        ws = context["ws"]
        workspace = workspace or ws
        user_info = UserInfo.model_validate(context["user"])
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
            "public/server-apps", context=context
        ) as controller:
            if not controller:
                raise Exception(
                    "Failed to launch application: server-apps service not found."
                )
            client_info = await controller.start(
                app_id,
                timeout=timeout * 5,
                wait_for_service=service_id,
            )
            return await self.get_service(
                f"{client_info['id']}:{service_id}",  # should not contain @app_id
                dict(timeout=timeout, case_conversion=case_conversion),
                context=context,
            )

    @asynccontextmanager
    async def _get_service_api(self, service_id: str, context=None):
        """Get the service api for the service."""
        self.validate_context(context, permission=UserPermission.read)
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        # Check if workspace exists
        if not await self._redis.hexists("workspaces", ws):
            raise KeyError(f"Workspace {ws} does not exist")
        # Now launch the app and get the service
        svc = await self.get_service(service_id, context=context)
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

    @schema_method
    async def get_service(
        self,
        service_id: str = Field(
            ...,
            description="Service ID. This should be a service id in the format: 'workspace/service_id', 'workspace/client_id:service_id' or 'workspace/client_id:service_id@app_id'",
        ),
        config: Optional[GetServiceConfig] = Field(
            None, description="Get service config"
        ),
        context=None,
    ):
        """Get a service based on the service_id"""
        assert (
            service_id != "*"
        ), "Invalid service id: {service_id}, it cannot be a wildcard."
        # no need to validate the context
        # self.validate_context(context, permission=UserPermission.read)
        try:
            config = config or GetServiceConfig()
            # Permission check will be handled by the get_service_api function
            svc_info = await self.get_service_info(
                service_id, {"mode": config.mode}, context=context
            )
            service_api = await self._rpc.get_remote_service(
                svc_info.id,
                {"timeout": config.timeout, "case_conversion": config.case_conversion},
            )
            assert service_api, f"Failed to get service: {service_id}"
            workspace = service_id.split("/")[0]
            service_api["config"]["workspace"] = workspace
            service_api["config"][
                "url"
            ] = f"{self._server_info['public_base_url']}/{workspace}/services/{service_id}"
            return service_api
        except KeyError as exp:
            if "@" in service_id:
                app_id = service_id.split("@")[1]
                service_id = service_id.split("@")[0]
                workspace = (
                    service_id.split("/")[0] if "/" in service_id else context["ws"]
                )
                return await self._launch_application_for_service(
                    app_id,
                    service_id,
                    workspace=workspace,
                    timeout=config.timeout,
                    case_conversion=config.case_conversion,
                    context=context,
                )
            else:
                raise exp

    @schema_method
    async def list_workspaces(
        self,
        type: str = Field(None, description="Workspace type (not implemented yet)"),
        context=None,
    ):
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
        await self._event_bus.emit("workspace_changed", workspace.model_dump())

    async def delete_client(
        self,
        client_id: str,
        workspace: str,
        user_info: UserInfo,
        unload: bool = False,
        context: Optional[dict] = None,
    ):
        """Delete all services associated with the given client_id in the specified workspace."""
        assert context is not None
        self.validate_context(context, permission=UserPermission.admin)
        cws = workspace
        validate_key_part(client_id)
        if "/" in client_id:
            _cws, client_id = client_id.split("/")
            assert _cws == cws, f"Client id workspace mismatch, {_cws} != {cws}"
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

        if unload:
            if await self._redis.hexists("workspaces", cws):
                if user_info.is_anonymous and cws == user_info.get_workspace():
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
            await self._event_bus.emit(f"unload:{ws}", "Unloading workspace: " + ws)

        await self._redis.hdel("workspaces", ws)
        await self._event_bus.emit("workspace_unloaded", winfo.model_dump())
        logger.info("Workspace %s unloaded.", ws)

    @schema_method
    async def cleanup(
        self,
        workspace: str = Field(None, description="workspace name"),
        timeout: float = Field(5, description="timeout in seconds"),
        context=None,
    ):
        """Cleanup the workspace."""
        self.validate_context(context, permission=UserPermission.admin)
        ws = context["ws"]
        if workspace is None:
            workspace = ws
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(workspace, UserPermission.admin):
            raise PermissionError(f"Permission denied for workspace {workspace}")
        # list all the clients and ping them
        # If they are not responding, delete them
        clients = await self.list_clients(context=context)
        removed = []
        for client in clients:
            try:
                assert (
                    await self.ping_client(client, timeout=timeout, context=context)
                    == "pong"
                )
            except Exception as e:
                logger.error(f"Failed to ping client {client}: {e}")
                # Remove dead client
                await self.delete_client(
                    client, ws, context["user"], unload=False, context=context
                )
                removed.append(client)
        return {"removed_clients": removed}

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
            "register_service": self.register_service,
            "unregister_service": self.unregister_service,
            "list_workspaces": self.list_workspaces,
            "list_services": self.list_services,
            "list_clients": self.list_clients,
            "register_service_type": self.register_service_type,
            "get_service_type": self.get_service_type,
            "list_service_types": self.list_service_types,
            "get_service_info": self.get_service_info,
            "get_service": self.get_service,
            "generate_token": self.generate_token,
            "revoke_token": self.revoke_token,
            "create_workspace": self.create_workspace,
            "get_workspace_info": self.get_workspace_info,
            "install_application": self.install_application,
            "uninstall_application": self.uninstall_application,
            "get_summary": self.get_summary,
            "ping": self.ping_client,
            "cleanup": self.cleanup,
        }
        interface["config"].update(self._server_info)
        return interface
