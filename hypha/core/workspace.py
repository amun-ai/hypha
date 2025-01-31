import re
import json
import asyncio
import logging
import traceback
import time
import os
import sys
from typing import Optional, Union, List, Any, Dict
from contextlib import asynccontextmanager
import random
import datetime
import numpy as np

from fakeredis import aioredis
from prometheus_client import Gauge
from hypha_rpc import RPC
from hypha_rpc.utils.schema import schema_method
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    JSON,
    select,
    func,
)
from sqlmodel import SQLModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from hypha.core import (
    ApplicationManifest,
    RedisRPCConnection,
    UserInfo,
    WorkspaceInfo,
    ServiceInfo,
    TokenConfig,
    UserPermission,
    ServiceTypeInfo,
    VisibilityEnum,
)
from hypha.vectors import VectorSearchEngine
from hypha.core.auth import generate_presigned_token, create_scope, valid_token
from hypha.utils import EventBus, random_id

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("workspace")
logger.setLevel(LOGLEVEL)

SERVICE_SUMMARY_FIELD = ["id", "name", "type", "description", "config"]

# Ensure the client_id is safe
_allowed_characters = re.compile(r"^[a-zA-Z0-9-_/|*]*$")
background_tasks = set()


# Function to return a timezone-naive datetime
def naive_utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def escape_redis_syntax(value: str) -> str:
    """Escape Redis special characters in a query string, except '*'."""
    # Escape all special characters except '*'
    return re.sub(r"([{}|@$\\\-\[\]\(\)\!&~:\"])", r"\\\1", value)


def sanitize_search_value(value: str) -> str:
    """Sanitize a value to prevent injection attacks, allowing '*' for wildcard support."""
    # Allow alphanumeric characters, spaces, underscores, hyphens, dots, slashes, and '*'
    value = re.sub(
        r"[^a-zA-Z0-9 _\-./*]", "", value
    )  # Remove unwanted characters except '*'
    return escape_redis_syntax(value.strip())


# SQLModel model for storing events
class EventLog(SQLModel, table=True):
    __tablename__ = "event_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str = Field(nullable=False)
    workspace: str = Field(nullable=False)
    user_id: str = Field(nullable=False)
    timestamp: datetime.datetime = Field(default_factory=naive_utc_now, index=True)
    data: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    def to_dict(self) -> dict:
        """Convert the SQLModel model instance to a dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "workspace": self.workspace,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "data": self.data,
        }


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


class WorkspaceStatus:
    READY = "ready"
    LOADING = "loading"
    UNLOADING = "unloading"
    CLOSED = "closed"


class WorkspaceManager:
    def __init__(
        self,
        store: Any,
        redis: aioredis.FakeRedis,
        root_user: UserInfo,
        event_bus: EventBus,
        server_info: dict,
        client_id: str,
        sql_engine: Optional[str] = None,
        s3_controller: Optional[Any] = None,
        server_app_controller: Optional[Any] = None,
        artifact_manager: Optional[Any] = None,
        enable_service_search: bool = False,
        cache_dir: str = None,
    ):
        self._redis = redis
        self._store = store
        self._initialized = False
        self._rpc = None
        self._root_user = root_user
        self._event_bus = event_bus
        self._server_info = server_info
        self._client_id = client_id
        self._s3_controller = s3_controller
        self._artifact_manager = artifact_manager
        self._server_app_controller = server_app_controller
        self._sql_engine = sql_engine
        self._cache_dir = cache_dir
        if self._sql_engine:
            self.SessionLocal = async_sessionmaker(
                self._sql_engine, expire_on_commit=False, class_=AsyncSession
            )
        else:
            self.SessionLocal = None
        self._active_ws = Gauge("active_workspaces", "Number of active workspaces")
        self._active_svc = Gauge(
            "active_services", "Number of active services", ["workspace"]
        )
        self._active_clients = Gauge(
            "active_clients", "Number of active clients", ["workspace"]
        )
        self._enable_service_search = enable_service_search

    async def _get_sql_session(self):
        """Return an async session for the database."""
        return self.SessionLocal()

    def get_client_id(self):
        assert self._client_id, "client id must not be empty."
        return self._client_id

    async def setup(
        self,
        service_id="default",
        service_name="Default workspace management service",
    ):
        """Setup the workspace manager."""
        if self._initialized:
            return self._rpc
        assert self._client_id, "client id must be provided."
        rpc = self._create_rpc(self._client_id)
        self._rpc = rpc
        management_service = self.create_service(service_id, service_name)
        await rpc.register_service(
            management_service,
            {"notify": False},
        )
        if self._sql_engine:
            async with self._sql_engine.begin() as conn:
                await conn.run_sync(EventLog.metadata.create_all)
                logger.info("Database tables created successfully.")

        self._embedding_model = None
        if self._enable_service_search:
            from fastembed import TextEmbedding

            self._embedding_model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5", cache_dir=self._cache_dir
            )
            self._vector_search = VectorSearchEngine(
                self._redis,
                prefix=None,
                cache_dir=self._cache_dir,
            )
            await self._vector_search.create_collection(
                collection_name="services",
                vector_fields=[
                    {"type": "TAG", "name": "id"},
                    {"type": "TEXT", "name": "name"},
                    {"type": "TAG", "name": "type"},
                    {"type": "TEXT", "name": "description"},
                    {"type": "TEXT", "name": "docs"},
                    {"type": "TAG", "name": "app_id"},
                    {"type": "TEXT", "name": "service_schema"},
                    {
                        "type": "VECTOR",
                        "name": "service_embedding",
                        "algorithm": "FLAT",
                        "attributes": {
                            "TYPE": "FLOAT32",
                            "DIM": 384,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    },
                    {"type": "TAG", "name": "visibility"},
                    {"type": "TAG", "name": "require_context"},
                    {"type": "TAG", "name": "workspace"},
                    {"type": "TAG", "name": "flags", "separator": ","},
                    {"type": "TAG", "name": "singleton"},
                    {"type": "TEXT", "name": "created_by"},
                ],
                overwrite=True,
            )
        self._initialized = True
        return rpc

    @schema_method
    async def log_event(
        self,
        event_type: str = Field(..., description="Event type"),
        data: Optional[dict] = Field(None, description="Additional event data"),
        context: dict = None,
    ):
        """Log a new event, checking permissions."""
        assert " " not in event_type, "Event type must not contain spaces"
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(workspace, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        session = await self._get_sql_session()
        try:
            async with session.begin():
                event_log = EventLog(
                    event_type=event_type,
                    workspace=workspace,
                    user_id=user_info.id,
                    data=data,
                )
                session.add(event_log)
                await session.commit()
                logger.info(
                    f"Logged event: {event_type} by {user_info.id} in {workspace}"
                )
        except Exception as e:
            logger.error(f"Failed to log event: {event_type}, {e}")
            raise
        finally:
            await session.close()

    @schema_method
    async def get_event_stats(
        self,
        event_type: Optional[str] = Field(None, description="Event type"),
        start_time: Optional[datetime.datetime] = Field(
            None, description="Start time for filtering events"
        ),
        end_time: Optional[datetime.datetime] = Field(
            None, description="End time for filtering events"
        ),
        context: Optional[dict] = None,
    ):
        """Get statistics for specific event types, filtered by workspace, user, and time range."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(workspace, UserPermission.read):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        session = await self._get_sql_session()
        try:
            async with session.begin():
                query = select(
                    EventLog.event_type, func.count(EventLog.id).label("count")
                ).filter(
                    EventLog.workspace == workspace, EventLog.user_id == user_info.id
                )

                # Apply optional filters
                if event_type:
                    query = query.filter(EventLog.event_type == event_type)
                if start_time:
                    query = query.filter(EventLog.timestamp >= start_time)
                if end_time:
                    query = query.filter(EventLog.timestamp <= end_time)

                query = query.group_by(EventLog.event_type)
                result = await session.execute(query)
                stats = result.fetchall()
                # Convert rows to dictionaries
                stats_dicts = [dict(row._mapping) for row in stats]
                return stats_dicts
        except Exception as e:
            raise e
        finally:
            await session.close()

    @schema_method
    async def get_events(
        self,
        event_type: Optional[str] = Field(None, description="Event type"),
        start_time: Optional[datetime.datetime] = Field(
            None, description="Start time for filtering events"
        ),
        end_time: Optional[datetime.datetime] = Field(
            None, description="End time for filtering events"
        ),
        context: Optional[dict] = None,
    ):
        """Search for events with various filters, enforcing workspace and permission checks."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(workspace, UserPermission.read):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        session = await self._get_sql_session()
        try:
            async with session.begin():
                query = select(EventLog).filter(
                    EventLog.workspace == workspace, EventLog.user_id == user_info.id
                )

                # Apply optional filters
                if event_type:
                    query = query.filter(EventLog.event_type == event_type)
                if start_time:
                    query = query.filter(EventLog.timestamp >= start_time)
                if end_time:
                    query = query.filter(EventLog.timestamp <= end_time)

                result = await session.execute(query)
                # Use scalars() to get model instances, not rows
                events = result.scalars().all()

                # Convert each EventLog instance to a dictionary using to_dict()
                event_dicts = [event.to_dict() for event in events]

                return event_dicts
        except Exception as e:
            raise e
        finally:
            await session.close()

    @schema_method
    async def get_summary(self, context: Optional[dict] = None) -> dict:
        """Get a summary about the workspace."""
        assert context is not None
        ws = context["ws"]
        workspace_info = await self.load_workspace_info(ws)
        workspace_info = workspace_info.model_dump()
        workspace_summary_fields = ["name", "description"]
        clients = await self._list_client_keys(ws)
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

    def _validate_workspace_id(self, name, with_hyphen=True):
        """Validate the workspace name."""
        if not name:
            raise ValueError("Workspace name must not be empty.")

        if with_hyphen and "-" not in name:
            raise ValueError(
                "Workspace name must contain at least one hyphen (e.g. my-workspace)."
            )
        # only allow numbers, letters in lower case and hyphens (no underscore)
        # use a regex to validate the workspace name
        pattern = re.compile(r"^[a-z0-9-|]*$")
        if not pattern.match(name):
            raise ValueError(
                f"Invalid workspace name: {name}, only lowercase letters, numbers and hyphens are allowed."
            )
        return name

    @schema_method
    async def bookmark(
        self,
        item: dict = Field(..., description="bookmark item"),
        context: Optional[dict] = None,
    ):
        """Bookmark workspace or service to the user's workspace."""
        user_info = UserInfo.model_validate(context["user"])
        assert "bookmark_type" in item, "Bookmark type must be provided."
        user_workspace = await self.load_workspace_info(user_info.get_workspace())
        assert user_workspace, f"User workspace not found: {user_info.get_workspace()}"
        user_workspace.config = user_workspace.config or {}
        if "bookmarks" not in user_workspace.config:
            user_workspace.config["bookmarks"] = []
        if item["bookmark_type"] == "workspace":
            workspace = item["id"]
            workspace = await self.load_workspace_info(workspace)
            user_workspace.config["bookmarks"] = [
                b for b in user_workspace.config["bookmarks"] if b["id"] != workspace.id
            ]
            user_workspace.config["bookmarks"].append(
                {
                    "bookmark_type": "workspace",
                    "id": workspace.id,
                    "name": workspace.name,
                    "description": workspace.description,
                }
            )
        elif item["bookmark_type"] == "service":
            service_id = item["id"]

            service_info = await self.get_service_info(service_id, context=context)
            user_workspace.config["bookmarks"] = [
                b
                for b in user_workspace.config["bookmarks"]
                if b["id"] != service_info.id
            ]
            user_workspace.config["bookmarks"].append(
                {
                    "bookmark_type": "service",
                    "id": service_info.id,
                    "name": service_info.name,
                    "description": service_info.description,
                    "config": service_info.config,
                }
            )
        else:
            raise ValueError(f"Invalid bookmark type: {item['bookmark_type']}")
        await self._update_workspace(user_workspace, user_info)

    @schema_method
    async def unbookmark(
        self,
        item: dict = Field(..., description="bookmark item to be removed"),
        context: Optional[dict] = None,
    ):
        """Unbookmark workspace or service from the user's workspace."""
        user_info = UserInfo.model_validate(context["user"])
        assert "bookmark_type" in item, "Bookmark type must be provided."
        user_workspace = await self.load_workspace_info(user_info.get_workspace())
        user_workspace.config = user_workspace.config or {}
        if "bookmarks" not in user_workspace.config:
            user_workspace.config["bookmarks"] = []
        if item["bookmark_type"] == "workspace":
            workspace = item["id"]
            user_workspace.config["bookmarks"] = [
                b for b in user_workspace.config["bookmarks"] if b["id"] != workspace
            ]
        elif item["bookmark_type"] == "service":
            service_id = item["id"]
            user_workspace.config["bookmarks"] = [
                b for b in user_workspace.config["bookmarks"] if b["id"] != service_id
            ]
        else:
            raise ValueError(f"Invalid bookmark type: {item['bookmark_type']}")
        await self._update_workspace(user_workspace, user_info)

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
        assert "id" in config, "Workspace id must be provided."
        if not config.get("name"):
            config["name"] = config["id"]
        user_info = UserInfo.model_validate(context["user"])
        if user_info.is_anonymous:
            raise Exception("Only registered user can create workspace.")

        try:
            await self.load_workspace_info(config["id"])
            if not overwrite:
                raise RuntimeError(f"Workspace already exists: {config['id']}")
            exists = True
        except KeyError:
            exists = False

        config["persistent"] = config.get("persistent") or False
        if user_info.is_anonymous and config["persistent"]:
            raise Exception("Only registered user can create persistent workspace.")
        workspace = WorkspaceInfo.model_validate(config)
        if user_info.id not in workspace.owners:
            workspace.owners.append(user_info.id)
        self._validate_workspace_id(workspace.id, with_hyphen=user_info.id != "root")
        # make sure we add the user's email to owners
        _id = user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]

        # user workspace, let's store all the created workspaces
        if user_info.get_workspace() == workspace.id:
            workspace.config = workspace.config or {}
            workspace.config["bookmarks"] = [
                {
                    "bookmark_type": "workspace",
                    "id": workspace.id,
                    "name": workspace.id,
                    "description": workspace.description,
                }
            ]
        await self._redis.hset("workspaces", workspace.id, workspace.model_dump_json())
        if not exists:
            self._active_ws.inc()
        if self._s3_controller:
            await self._s3_controller.setup_workspace(workspace)
        await self._event_bus.emit("workspace_loaded", workspace.model_dump())
        if user_info.get_workspace() != workspace.id:
            await self.bookmark(
                {"bookmark_type": "workspace", "id": workspace.id}, context=context
            )
        logger.info("Created workspace %s", workspace.id)
        # preare the workspace for the user
        task = asyncio.create_task(self._prepare_workspace(workspace))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return workspace.model_dump()

    @schema_method
    async def delete_workspace(
        self,
        workspace: str = Field(..., description="workspace name"),
        context: Optional[dict] = None,
    ):
        """Remove a workspace."""
        assert context is not None
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.admin):
            raise PermissionError(f"Permission denied for workspace {ws}")
        workspace_info = await self.load_workspace_info(workspace)
        # delete all the associated keys
        keys = await self._redis.keys(f"{ws}:*")
        for key in keys:
            await self._redis.delete(key)
        if self._s3_controller:
            await self._s3_controller.cleanup_workspace(workspace_info, force=True)
        await self._redis.hdel("workspaces", workspace)
        await self._event_bus.emit("workspace_deleted", workspace_info.model_dump())
        # remove the workspace from the user's bookmarks
        user_workspace = await self.load_workspace_info(user_info.get_workspace())
        user_workspace.config = user_workspace.config or {}
        if "bookmarks" in user_workspace.config:
            user_workspace.config["bookmarks"] = [
                b for b in user_workspace.config["bookmarks"] if b["name"] != workspace
            ]
            await self._update_workspace(user_workspace, user_info)
        logger.info("Workspace %s removed by %s", workspace, user_info.id)

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
        pattern = f"services:*|*:{workspace}/*:built-in@*"
        keys = await self._redis.keys(pattern)
        clients = []
        for key in set(keys):
            service_data = await self._redis.hgetall(key)
            service = ServiceInfo.from_redis_dict(service_data)
            if "/" in service.id:
                client_id = service.id.split("/")[1].split(":")[0]
            else:
                client_id = service.id.split(":")[0]
            clients.append(
                {
                    "id": workspace + "/" + client_id,
                    "user": service.config.created_by,
                }
            )
        return clients

    async def _list_client_keys(self, workspace: str):
        """List all client keys in the workspace."""
        pattern = f"services:*|*:{workspace}/*:built-in@*"
        keys = await self._redis.keys(pattern)
        client_keys = set()
        for key in keys:
            # Extract the client ID from the service key
            key_parts = key.decode("utf-8").split("/")
            client_id = key_parts[1].split(":")[0]
            client_keys.add(workspace + "/" + client_id)
        return list(client_keys)

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

    def _convert_filters_to_hybrid_query(self, filters: dict) -> str:
        """
        Convert a filter dictionary to a Redis hybrid query string.

        Args:
            filters (dict): Dictionary of filters, e.g., {"type": "my-type", "year": [2011, 2012]}.

        Returns:
            str: Redis hybrid query string, e.g., "(@type:{my-type} @year:[2011 2012])".
        """
        from redis.commands.search.field import TextField, TagField, NumericField

        conditions = []

        for field_name, value in filters.items():
            # Find the field type in the schema
            field_type = None
            for field in self._search_fields:
                if field.name == field_name:
                    field_type = type(field)
                    break

            if not field_type:
                raise ValueError(f"Unknown field '{field_name}' in filters.")

            # Sanitize the field name
            sanitized_field_name = sanitize_search_value(field_name)

            if field_type == TagField:
                # Use `{value}` for TagField
                if not isinstance(value, str):
                    raise ValueError(
                        f"TagField '{field_name}' requires a string value."
                    )
                sanitized_value = sanitize_search_value(value)
                conditions.append(f"@{sanitized_field_name}:{{{sanitized_value}}}")

            elif field_type == NumericField:
                # Use `[min max]` for NumericField
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(
                        f"NumericField '{field_name}' requires a list or tuple with two elements."
                    )
                min_val, max_val = value
                conditions.append(f"@{sanitized_field_name}:[{min_val} {max_val}]")

            elif field_type == TextField:
                # Use `"value"` for TextField
                if not isinstance(value, str):
                    raise ValueError(
                        f"TextField '{field_name}' requires a string value."
                    )
                if "*" in value:
                    assert value.endswith("*"), "Wildcard '*' must be at the end."
                    sanitized_value = sanitize_search_value(value)
                    conditions.append(f"@{sanitized_field_name}:{sanitized_value}")
                else:
                    sanitized_value = escape_redis_syntax(value)
                    conditions.append(f'@{sanitized_field_name}:"{sanitized_value}"')

            else:
                raise ValueError(f"Unsupported field type for '{field_name}'.")

        return " ".join(conditions)

    @schema_method
    async def search_services(
        self,
        query: Optional[Union[str, Any]] = Field(
            None,
            description="Text query or precomputed embedding vector for vector search in numpy format.",
        ),
        filters: Optional[Dict[str, Any]] = Field(
            None, description="Filter dictionary for hybrid search."
        ),
        limit: Optional[int] = Field(
            5, description="Maximum number of results to return."
        ),
        offset: Optional[int] = Field(0, description="Offset for pagination."),
        return_fields: Optional[List[str]] = Field(
            None, description="Fields to return."
        ),
        order_by: Optional[str] = Field(
            None,
            description="Order by field, default is score if embedding or text_query is provided.",
        ),
        pagination: Optional[bool] = Field(
            False, description="Enable pagination, return metadata with total count."
        ),
        context: Optional[dict] = None,
    ):
        """
        Search services with support for hybrid queries and pure filter-based queries.
        """
        if not self._enable_service_search:
            raise RuntimeError("Service search is not enabled.")

        current_workspace = context["ws"]
        # Generate embedding if text_query is provided
        if isinstance(query, str):
            loop = asyncio.get_event_loop()
            embeddings = list(
                await loop.run_in_executor(None, self._embedding_model.embed, [query])
            )
            vector_query = embeddings[0]
        else:
            vector_query = query

        auth_filter = f"@visibility:{{public}} | @workspace:{{{sanitize_search_value(current_workspace)}}}"
        results = await self._vector_search.search_vectors(
            "services",
            query={"service_embedding": vector_query},
            filters=filters,
            extra_filter=auth_filter,
            limit=limit,
            offset=offset,
            return_fields=return_fields,
            order_by=order_by,
            pagination=True,
        )

        # Convert results to dictionaries and return
        results["items"] = [
            ServiceInfo.from_redis_dict(doc, in_bytes=False).model_dump()
            for doc in results["items"]
        ]
        if pagination:
            return results
        else:
            return results["items"]

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
                "type": "*",
                "visibility": "*",
                "workspace": cws,
                "client_id": "*",
                "service_id": "*",
            }
        # Convert string query into a dictionary
        if isinstance(query, str):
            visibility = "*"
            type_filter = "*"
            workspace = cws
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
                    "type": type_filter,
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
                    "type": type_filter,
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
                    "type": type_filter,
                    "visibility": visibility,
                    "workspace": workspace,
                    "client_id": client_id,
                    "service_id": service_id,
                }
            else:
                workspace = query
                query = {
                    "type": type_filter,
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
        workspace = query.get("workspace", cws)
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
        type_filter = query.get("type", "*")
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

        pattern = f"services:{visibility}|{type_filter}:{workspace}/{client_id}:{service_id}@{app_id}"

        assert pattern.startswith(
            "services:"
        ), "Query pattern does not start with 'services:'."
        assert not any(
            char in pattern for char in "{}"
        ), "Query pattern contains invalid characters."

        keys = await self._redis.keys(pattern)

        if workspace == "*":
            # add services in the current workspace
            ws_pattern = f"services:{original_visibility}|{type_filter}:{cws}/{client_id}:{service_id}@{app_id}"
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
            services.append(service_dict)

        return services

    async def _embed_service(self, redis_data):
        if "service_embedding" in redis_data:
            if isinstance(redis_data["service_embedding"], np.ndarray):
                redis_data["service_embedding"] = redis_data[
                    "service_embedding"
                ].tobytes()
            elif isinstance(redis_data["service_embedding"], bytes):
                pass
            else:
                raise ValueError(
                    f"Invalid service_embedding type: {type(redis_data['service_embedding'])}, it must be a numpy array or bytes."
                )
        elif redis_data.get("docs"):  # Only embed the service if it has docs
            assert self._embedding_model, "Embedding model is not available."
            summary = f"{redis_data.get('name', '')}\n{redis_data.get('description', '')}\n{redis_data.get('docs', '')}"
            loop = asyncio.get_event_loop()
            embeddings = list(
                await loop.run_in_executor(None, self._embedding_model.embed, [summary])
            )
            redis_data["service_embedding"] = embeddings[0].tobytes()
        return redis_data

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

        service_name = service.id.split(":")[1]
        service.name = service.name or service_name
        workspace = service.id.split("/")[0]

        # Store all the info for client's built-in services
        if service_name == "built-in" and service.type == "built-in":
            service.config.created_by = user_info.model_dump()
        else:
            service.config.created_by = {"id": user_info.id}

        # Check for existing singleton services
        if service.config.singleton:
            key = f"services:*|*:{workspace}/*:{service_name}@*"
            peer_keys = await self._redis.keys(key)
            if len(peer_keys) > 0:
                # If it's the same service being re-registered, allow it
                for peer_key in peer_keys:
                    peer_service = await self._redis.hgetall(peer_key)
                    peer_service = ServiceInfo.from_redis_dict(peer_service)
                    if (
                        peer_service.config.singleton
                        and peer_service.id == service.id
                        and peer_service.app_id == service.app_id
                    ):
                        # Same service being re-registered, allow it
                        await self._redis.delete(peer_key)
                        break
                    else:
                        raise ValueError(
                            f"A singleton service with the same name ({service_name}) already exists in the workspace ({workspace}), please remove it first or use a different name."
                        )

        key = f"services:*|*:{workspace}/*:{service_name}@*"
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
            builtins = await self._redis.keys(f"services:*|*:{client_id}:built-in@*")
            if not builtins:
                logger.warning(
                    "Refuse to add service %s, client %s has been removed.",
                    service.id,
                    client_id,
                )
                return
        # Check if the service already exists
        service_exists = await self._redis.keys(f"services:*|*:{service.id}@*")
        visibility = (
            service.config.visibility.value
            if isinstance(service.config.visibility, VisibilityEnum)
            else service.config.visibility
        )
        key = f"services:{visibility}|{service.type}:{service.id}@{service.app_id}"

        if service_exists:
            # remove all the existing services
            # This is needed because it is maybe registered with a different app_id
            for k in service_exists:
                logger.info(f"Replacing existing service: {k}")
                await self._redis.delete(k)

            if self._enable_service_search:
                redis_data = await self._embed_service(service.to_redis_dict())
            else:
                redis_data = service.to_redis_dict()
                if "service_embedding" in redis_data:
                    del redis_data["service_embedding"]
            await self._redis.hset(key, mapping=redis_data)
            if ":built-in@" in key:
                await self._event_bus.emit(
                    "client_updated", {"id": client_id, "workspace": ws}
                )
                logger.info(f"Updating built-in service: {service.id}")
            else:
                await self._event_bus.emit("service_updated", service.model_dump())
                logger.info(f"Updating service: {service.id}")
        else:
            if self._enable_service_search:
                redis_data = await self._embed_service(service.to_redis_dict())
            else:
                redis_data = service.to_redis_dict()
                if "service_embedding" in redis_data:
                    del redis_data["service_embedding"]
            await self._redis.hset(key, mapping=redis_data)
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
                self._active_clients.labels(workspace=ws).inc()
            else:
                # Remove the service embedding from the config
                if service.config and service.config.service_embedding is not None:
                    service.config.service_embedding = None
                await self._event_bus.emit(
                    "service_added", service.model_dump(mode="json")
                )
                logger.info(f"Adding service {service.id}")
                self._active_svc.labels(workspace=ws).inc()

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
        key = f"services:*|*:{service_id}@{app_id}"
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
            assert (
                len(keys) == 1
            ), f"Multiple services found for {service_id}, e.g. {keys[:5]}. You can either specify the mode as 'random', 'first' or 'last', or provide the service id in `client_id:service_id` format."
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
        if not key.startswith(b"services:public|") and not user_info.check_permission(
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
        service.type = service.type or "*"
        visibility = (
            service.config.visibility.value
            if isinstance(service.config.visibility, VisibilityEnum)
            else service.config.visibility
        )
        key = f"services:{visibility}|{service.type}:{service.id}@{service.app_id}"

        # Check if the service exists before removal
        service_keys = await self._redis.keys(key)

        if len(service_keys) > 1:
            raise ValueError(
                f"Multiple services found for {service.id}, e.g. {service_keys[:5]}. Please specify the service id in full format `workspace/client_id:service_id@app_id`."
            )
        elif len(service_keys) == 1:
            logger.info("Removing service: %s", service_keys[0])
            await self._redis.delete(service_keys[0])
            if ":built-in@" in key:
                await self._event_bus.emit(
                    "client_disconnected", {"id": client_id, "workspace": ws}
                )
                self._active_clients.labels(workspace=ws).dec()
            else:
                await self._event_bus.emit("service_removed", service.model_dump())
                self._active_svc.labels(workspace=ws).dec()
        else:
            logger.warning(f"Service {key} does not exist and cannot be removed.")
            raise KeyError(f"Service not found: {service.id}")

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
        await self.log_event("log", msg, context=context)

    async def load_workspace_info(self, workspace: str, load=True) -> WorkspaceInfo:
        """Load info of the current workspace from the redis store."""
        assert workspace is not None

        async with self._workspace_lock(workspace, "load") as prev_status:
            if prev_status == WorkspaceStatus.UNLOADING:
                raise RuntimeError(
                    f"Cannot load workspace {workspace} - unload in progress"
                )

            try:
                # Check if workspace exists in Redis
                try:
                    workspace_info = await self._redis.hget("workspaces", workspace)
                    if workspace_info is not None:
                        workspace_info = WorkspaceInfo.model_validate(
                            json.loads(workspace_info.decode())
                        )
                        # If workspace exists but isn't ready, wait for preparation
                        status_data = await self._redis.get(
                            f"workspace_status:{workspace}"
                        )
                        if status_data:
                            status = json.loads(status_data.decode())
                            if status["status"] == WorkspaceStatus.LOADING:
                                # Already being prepared, just return
                                return workspace_info

                        # Set loading status if not already loading
                        await self._set_workspace_status(
                            workspace, WorkspaceStatus.LOADING
                        )
                        return workspace_info
                except Exception as e:
                    logger.error(f"Failed to load workspace info from Redis: {e}")

                if not load:
                    raise KeyError(f"Workspace not found: {workspace}")

                logger.info(
                    f"Workspace {workspace} not found in Redis, trying to load from S3"
                )

                if not self._s3_controller:
                    raise KeyError(
                        f"Workspace ({workspace}) not found and workspace loader not configured (requires s3)."
                    )

                # Try loading from S3
                workspace_info = await self._s3_controller.load_workspace_info(
                    workspace, self._root_user
                )
                if not workspace_info:
                    raise KeyError(f"Workspace not found: {workspace}")

                # Initialize new workspace
                workspace_info.status = None
                await self._redis.hset(
                    "workspaces", workspace_info.id, workspace_info.model_dump_json()
                )

                # Set initial loading status
                await self._set_workspace_status(workspace, WorkspaceStatus.LOADING)

                # Start preparation tasks
                task = asyncio.create_task(self._prepare_workspace(workspace_info))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)

                self._active_ws.inc()
                await self._s3_controller.setup_workspace(workspace_info)
                await self._event_bus.emit(
                    "workspace_loaded", workspace_info.model_dump()
                )

                return workspace_info

            except Exception as e:
                # Make sure we set error status on failure
                await self._set_workspace_status(
                    workspace, WorkspaceStatus.CLOSED, error=str(e)
                )
                logger.error(f"Failed to load workspace info: {e}")
                raise

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

        if not self._artifact_manager:
            raise Exception(
                "Failed to launch application: artifact-manager service not found."
            )
        if ":" in service_id:
            service_id = service_id.split(":")[1]

        assert service_id and service_id not in [
            "*",
            "default",
            "built-in",
        ], f"Invalid service id: {service_id}"

        applications = await self._artifact_manager.list_children(
            f"{workspace}/applications",
            context={"ws": workspace, "user": user_info.model_dump()},
        )
        applications = {
            item["manifest"]["id"]: ApplicationManifest.model_validate(item["manifest"])
            for item in applications
        }
        if app_id not in applications:
            raise KeyError(
                f"Application id `{app_id}` not found in workspace {workspace}"
            )

        # Make sure the service_id is in the application services
        app_info = applications[app_id]
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

        if not self._server_app_controller:
            raise Exception(
                "Failed to launch application: server apps controller is not configured."
            )
        client_info = await self._server_app_controller.start(
            app_id,
            timeout=timeout * 5,
            wait_for_service=service_id,
            context=context,
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
        match: dict = Field(None, description="Match pattern for filtering workspaces"),
        offset: int = Field(0, description="Offset for pagination"),
        limit: int = Field(256, description="Maximum number of workspaces to return"),
        context=None,
    ) -> List[Dict[str, Any]]:
        """Get all workspaces with pagination."""
        self.validate_context(context, permission=UserPermission.read)
        user_info = UserInfo.model_validate(context["user"])

        # Validate page and limit
        if offset < 0:
            raise ValueError("Offset number must be greater than or equal to 0")
        if limit < 1 or limit > 256:
            raise ValueError("Limit must be greater than 0 and less than 256")

        cursor = 0
        workspaces = []
        start_index = offset
        end_index = offset + limit
        current_index = 0

        while True:
            cursor, keys = await self._redis.hscan("workspaces", cursor)
            for ws_id in keys:
                ws_data = await self._redis.hget("workspaces", ws_id)
                if ws_data:
                    ws_data = ws_data.decode()
                    workspace_info = WorkspaceInfo.model_validate(json.loads(ws_data))

                    if (
                        user_info.check_permission(
                            workspace_info.id, UserPermission.read
                        )
                        or user_info.id in workspace_info.owners
                        or user_info.email in workspace_info.owners
                    ):
                        match = match or {}
                        if not all(
                            getattr(workspace_info, k, None) == v
                            for k, v in match.items()
                        ):
                            continue
                        if start_index <= current_index < end_index:
                            workspaces.append(
                                {
                                    "id": workspace_info.id,
                                    "name": workspace_info.id,
                                    "description": workspace_info.description,
                                    "read_only": workspace_info.read_only,
                                    "persistent": workspace_info.persistent,
                                    "owners": workspace_info.owners,
                                }
                            )
                        current_index += 1
                        if current_index >= end_index:
                            return workspaces
            if cursor == 0:
                break
        return workspaces

    async def _update_workspace(
        self, workspace: WorkspaceInfo, user_info: UserInfo, overwrite=False
    ):
        """Update the workspace config."""
        assert isinstance(workspace, WorkspaceInfo) and isinstance(user_info, UserInfo)
        if not user_info.check_permission(workspace.id, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {workspace.id}")

        workspace_info = await self._redis.hget("workspaces", workspace.id)
        if not overwrite and workspace_info is None:
            raise KeyError(f"Workspace not found: {workspace.id}")
        else:
            existing_workspace = WorkspaceInfo.model_validate(
                json.loads(workspace_info.decode())
            )
            assert existing_workspace.id == workspace.id, "Workspace name mismatch."
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]
        logger.info("Updating workspace %s", workspace.id)

        await self._redis.hset("workspaces", workspace.id, workspace.model_dump_json())
        if self._s3_controller:
            await self._s3_controller.save_workspace_config(workspace)
        await self._event_bus.emit("workspace_changed", workspace.model_dump())

    @asynccontextmanager
    async def _workspace_lock(
        self, workspace_id: str, operation: str, timeout: float = 30
    ):
        """Acquire a distributed lock for workspace operations using Redis.

        Args:
            workspace_id: The workspace ID
            operation: The operation being performed (e.g. 'load', 'unload')
            timeout: Lock timeout in seconds
        """
        lock_key = f"workspace_lock:{workspace_id}"
        status_key = f"workspace_status:{workspace_id}"
        lock_value = f"{self._client_id}:{time.time()}"

        try:
            # Check if we already hold the lock
            current_lock = await self._redis.get(lock_key)
            if current_lock:
                current_lock = current_lock.decode()
                current_owner = current_lock.split(":")[0]
                if current_owner == self._client_id:
                    # We already hold the lock, just return current status
                    status_data = await self._redis.get(status_key)
                    old_status = status_data.decode() if status_data else None
                    logger.debug(
                        f"Reusing existing lock for {operation} on workspace {workspace_id}"
                    )
                    yield old_status
                    return

            # Try to acquire lock with timeout
            acquired = await self._redis.set(
                lock_key,
                lock_value,
                nx=True,  # Only set if not exists
                ex=timeout,  # Auto-expire after timeout
            )

            if not acquired:
                raise RuntimeError(
                    f"Could not acquire lock for workspace {workspace_id} - another operation in progress"
                )

            # Get current status
            old_status = await self._redis.get(status_key)
            old_status = old_status.decode() if old_status else None

            logger.info(
                f"Acquired lock for {operation} on workspace {workspace_id} (previous status: {old_status})"
            )

            yield old_status

        finally:
            # Only remove our own lock if we created it
            current = await self._redis.get(lock_key)
            if current and current.decode().split(":")[0] == self._client_id:
                await self._redis.delete(lock_key)
                logger.info(f"Released lock for workspace {workspace_id}")

    async def _set_workspace_status(
        self,
        workspace_id: str,
        status: str,
        error: Optional[str] = None,
        errors: Optional[dict] = None,
    ):
        """Set workspace status."""
        status_data = {
            "status": status,
            "timestamp": time.time(),
            "error": error,
            "errors": errors,
        }
        await self._redis.set(
            f"workspace_status:{workspace_id}", json.dumps(status_data)
        )
        await self._event_bus.emit(
            "workspace_status_changed", {"id": workspace_id, "status": status_data}
        )

    async def _prepare_workspace(self, workspace_info: WorkspaceInfo):
        """Prepare the workspace with proper locking."""
        async with self._workspace_lock(workspace_info.id, "prepare") as prev_status:
            if prev_status == WorkspaceStatus.UNLOADING:
                raise RuntimeError(
                    f"Cannot prepare workspace {workspace_info.id} - unload in progress"
                )

            await self._set_workspace_status(workspace_info.id, WorkspaceStatus.LOADING)

            errors = {}
            try:
                if workspace_info.persistent:
                    try:
                        if self._artifact_manager:
                            await self._artifact_manager.prepare_workspace(
                                workspace_info
                            )
                    except Exception as e:
                        errors["artifact_manager"] = traceback.format_exc()
                    try:
                        if self._server_app_controller:
                            await self._server_app_controller.prepare_workspace(
                                workspace_info
                            )
                    except Exception as e:
                        errors["server_app_controller"] = traceback.format_exc()

                # Update workspace status
                workspace_info.status = {"ready": True, "errors": errors}
                await self._redis.hset(
                    "workspaces", workspace_info.id, workspace_info.model_dump_json()
                )

                # Set workspace status with errors field
                await self._set_workspace_status(
                    workspace_info.id,
                    WorkspaceStatus.READY,
                    errors=errors if errors else None,
                )

                await self._event_bus.emit(
                    "workspace_ready", workspace_info.model_dump()
                )
                logger.info("Workspace %s prepared.", workspace_info.id)

            except Exception as e:
                error_msg = str(e)
                await self._set_workspace_status(
                    workspace_info.id, WorkspaceStatus.CLOSED, error=error_msg
                )
                raise

    async def unload(self, context=None):
        """Unload the workspace with proper locking."""
        self.validate_context(context, permission=UserPermission.admin)
        ws = context["ws"]

        async with self._workspace_lock(ws, "unload") as prev_status:
            if prev_status == WorkspaceStatus.LOADING:
                raise RuntimeError(
                    f"Cannot unload workspace {ws} - load/prepare in progress"
                )

            try:
                if not await self._redis.hexists("workspaces", ws):
                    logger.warning(f"Workspace {ws} has already been unloaded.")
                    return

                await self._set_workspace_status(ws, WorkspaceStatus.UNLOADING)

                # Rest of existing unload logic...
                winfo = await self.load_workspace_info(ws, load=False)
                winfo.status = None
                await self._redis.hset("workspaces", winfo.id, winfo.model_dump_json())

                client_keys = await self._list_client_keys(winfo.id)
                if client_keys:
                    client_summary = ", ".join(client_keys[:10]) + (
                        "..." if len(client_keys) > 10 else ""
                    )
                    logger.info(
                        f"There are {len(client_keys)} clients in workspace {ws}: {client_summary}"
                    )
                    await self._event_bus.emit(
                        f"unload:{ws}", "Unloading workspace: " + ws
                    )

                if not winfo.persistent:
                    keys = await self._redis.keys(f"{ws}:*")
                    for key in keys:
                        await self._redis.delete(key)
                    if self._s3_controller:
                        await self._s3_controller.cleanup_workspace(winfo)

                self._active_ws.dec()
                try:
                    self._active_clients.remove(ws)
                    self._active_svc.remove(ws)
                except KeyError:
                    pass

                await self._close_workspace(winfo)
                await self._redis.hdel("workspaces", ws)
                await self._set_workspace_status(ws, WorkspaceStatus.CLOSED)

            except Exception as e:
                error_msg = str(e)
                await self._set_workspace_status(
                    ws, WorkspaceStatus.CLOSED, error=error_msg
                )
                logger.error(f"Failed to unload workspace: {e}")
                raise

    async def _close_workspace(self, workspace_info: WorkspaceInfo):
        """Archive the workspace."""
        assert (
            workspace_info.status is None
        ), "Workspace must be unloaded before archiving."
        if workspace_info.persistent:
            if self._artifact_manager:
                try:
                    await self._artifact_manager.close_workspace(workspace_info)
                except Exception as e:
                    logger.error(f"Aritfact manager failed to close workspace: {e}")
            if self._server_app_controller:
                try:
                    await self._server_app_controller.close_workspace(workspace_info)
                except Exception as e:
                    logger.error(
                        f"Server app controller failed to close workspace: {e}"
                    )

        await self._event_bus.emit("workspace_unloaded", workspace_info.model_dump())
        logger.info("Workspace %s unloaded.", workspace_info.id)

    @schema_method
    async def wait_until_ready(self, timeout: Optional[int] = 60, context=None):
        """Wait for the workspace to be ready with status checking."""
        ws = context["ws"]
        status_key = f"workspace_status:{ws}"

        async def check_status():
            status_data = await self._redis.get(status_key)
            if not status_data:
                return None
            return json.loads(status_data.decode())

        start_time = time.time()
        while True:
            status_data = await check_status()

            if not status_data:
                raise RuntimeError(f"Workspace {ws} status not found")

            current_status = status_data["status"]

            if current_status == WorkspaceStatus.READY:
                # Ensure the status object has an errors field
                if "errors" not in status_data:
                    status_data["errors"] = None
                return {"ready": True, "status": status_data}

            if current_status == WorkspaceStatus.LOADING:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Workspace {ws} preparation timed out")
                # Continue waiting

            elif current_status == WorkspaceStatus.CLOSED:
                # For daemon apps, try to prepare the workspace
                workspace_info = await self.load_workspace_info(ws)
                if workspace_info and workspace_info.persistent:
                    await self._prepare_workspace(workspace_info)
                    continue

                if status_data.get("error"):
                    raise RuntimeError(
                        f"Workspace preparation failed: {status_data['error']}"
                    )
                raise RuntimeError(f"Workspace {ws} is closed")

            elif current_status == WorkspaceStatus.UNLOADING:
                raise RuntimeError(f"Workspace {ws} is being unloaded")

            # Wait for status change event
            try:
                await self._event_bus.wait_for(
                    "workspace_status_changed", match={"id": ws}, timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

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
        workspace_info = await self.load_workspace_info(workspace)
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(workspace, UserPermission.admin):
            raise PermissionError(
                f"Permission denied for workspace {workspace}, user: {user_info.id}"
            )
        logger.info(f"Cleaning up workspace {workspace_info.id}...")
        # list all the clients and ping them
        # If they are not responding, delete them
        client_keys = await self._list_client_keys(workspace)
        if len(client_keys) <= 0 and not workspace_info.persistent:
            await self.delete_workspace(workspace_info.id, context=context)
            return {"removed_workspace": workspace_info.id, "removed_clients": []}

        removed = []
        summary = {}
        if not workspace_info.persistent:
            unload = True
        else:
            unload = False
        for client in client_keys:
            try:
                assert (
                    await self.ping_client(client, timeout=timeout, context=context)
                    == "pong"
                )
            except Exception as e:
                logger.error(f"Failed to ping client {client}: {e}")
                user_info = UserInfo.model_validate(context["user"])
                # Remove dead client
                await self.delete_client(
                    client, workspace, user_info, unload=unload, context=context
                )
                removed.append(client)
        if removed:
            summary["removed_clients"] = removed
        return summary

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
            "log_event": self.log_event,
            "get_event_stats": self.get_event_stats,
            "get_events": self.get_events,
            "register_service": self.register_service,
            "unregister_service": self.unregister_service,
            "list_workspaces": self.list_workspaces,
            "list_services": self.list_services,
            "search_services": self.search_services,
            "list_clients": self.list_clients,
            "register_service_type": self.register_service_type,
            "get_service_type": self.get_service_type,
            "list_service_types": self.list_service_types,
            "get_service_info": self.get_service_info,
            "get_service": self.get_service,
            "generate_token": self.generate_token,
            "revoke_token": self.revoke_token,
            "create_workspace": self.create_workspace,
            "delete_workspace": self.delete_workspace,
            "bookmark": self.bookmark,
            "unbookmark": self.unbookmark,
            "get_workspace_info": self.get_workspace_info,
            "get_summary": self.get_summary,
            "ping": self.ping_client,
            "cleanup": self.cleanup,
            "wait_until_ready": self.wait_until_ready,
        }
        interface["config"].update(self._server_info)
        return interface

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
        assert isinstance(user_info, UserInfo)
        self.validate_context(context, permission=UserPermission.admin)
        cws = workspace
        validate_key_part(client_id)
        if "/" in client_id:
            _cws, client_id = client_id.split("/")
            assert _cws == cws, f"Client id workspace mismatch, {_cws} != {cws}"

        async with self._workspace_lock(cws, "delete_client") as prev_status:
            # Define a pattern to match all services for the given client_id in the current workspace
            pattern = f"services:*|*:{cws}/{client_id}:*@*"

            # Ensure the pattern is secure
            assert not any(
                char in pattern for char in "{}"
            ), "Query pattern contains invalid characters."

            keys = await self._redis.keys(pattern)
            logger.info(
                f"Removing {len(keys)} services for client {client_id} in {cws}"
            )
            for key in keys:
                await self._redis.delete(key)

            await self._event_bus.emit(
                "client_disconnected", {"id": client_id, "workspace": cws}
            )
            self._active_clients.labels(workspace=cws).dec()
            self._active_svc.labels(workspace=cws).dec(len(keys) - 1)

            if unload:
                if await self._redis.hexists("workspaces", cws):
                    if user_info.is_anonymous and cws == user_info.get_workspace():
                        logger.info(
                            f"Unloading workspace {cws} for anonymous user (while deleting client {client_id})"
                        )
                        # unload temporary workspace if the user exits
                        await self.unload(context=context)
                    else:
                        logger.info(
                            f"Unloading workspace {cws} for non-anonymous user (while deleting client {client_id})"
                        )
                        # otherwise delete the workspace if it is empty
                        await self.unload_if_empty(context=context)
                else:
                    logger.warning(
                        f"Workspace {cws} not found (while deleting client {client_id})"
                    )

    async def unload_if_empty(self, context=None):
        """Delete the workspace if it is empty."""
        self.validate_context(context, permission=UserPermission.admin)
        workspace = context["ws"]
        logger.debug(f"Attempting to unload workspace: {workspace}")
        client_keys = await self._list_client_keys(workspace)
        if not client_keys:
            logger.info(f"No active clients in workspace {workspace}. Unloading...")
            await self.unload(context=context)
        else:
            logger.warning(
                f"Skip unloading workspace {workspace} because it is not empty, remaining clients: {client_keys[:10]}..."
            )
