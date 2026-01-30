"""Provide the ImJoy core API interface."""

import asyncio
import inspect
import io
import json
import logging
import sys
import os
import gc
import uuid
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator
import msgpack
from pydantic import EmailStr, PrivateAttr, constr, SerializeAsAny, ConfigDict, AnyHttpUrl

from hypha.utils import EventBus
import jsonschema
from hypha.core.activity import ActivityTracker

from prometheus_client import Counter, Gauge


class AutoscalingConfig(BaseModel):
    """Represent autoscaling configuration for apps."""

    enabled: bool = False
    min_instances: int = 1
    max_instances: int = 10
    target_load_per_instance: float = 0.8  # Target load per instance
    target_requests_per_instance: int = 100  # Target active requests per instance
    scale_up_threshold: float = 0.9  # Scale up when load exceeds this
    scale_down_threshold: float = 0.3  # Scale down when load is below this
    scale_up_cooldown: int = 300  # Cooldown period in seconds for scale up
    scale_down_cooldown: int = 600  # Cooldown period in seconds for scale down
    metric_type: str = "load"  # Can be "load" or "custom"
    custom_metric_function: Optional[str] = None  # For custom metrics


LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("core")
logger.setLevel(LOGLEVEL)


class VisibilityEnum(str, Enum):
    """Represent the visibility of the workspace."""

    public = "public"
    protected = "protected"
    unlisted = "unlisted"


class StatusEnum(str, Enum):
    """Represent the status of a component."""

    ready = "ready"
    initializing = "initializing"
    not_initialized = "not_initialized"


class ServiceConfig(BaseModel):
    """Represent service config."""

    model_config = ConfigDict(extra="allow", use_enum_values=False)

    visibility: VisibilityEnum = VisibilityEnum.protected
    require_context: Union[Tuple[str], List[str], bool] = False
    workspace: Optional[str] = None
    flags: List[str] = []
    singleton: Optional[bool] = False
    created_by: Optional[Dict] = None
    service_embedding: Optional[Any] = None
    authorized_workspaces: Optional[List[str]] = None


class ServiceInfo(BaseModel):
    """Represent service."""

    model_config = ConfigDict(extra="allow")

    config: Optional[SerializeAsAny[ServiceConfig]] = None
    id: str
    name: Optional[str] = None
    type: Optional[str] = "generic"
    description: Optional[constr(max_length=1024)] = None  # type: ignore
    docs: Optional[str] = None
    app_id: Optional[str] = None
    service_schema: Optional[Dict[str, Any]] = None

    def is_singleton(self):
        """Check if the service is singleton."""
        return "single-instance" in self.config.flags

    def to_redis_dict(self):
        """
        Serialize the model to a Redis-compatible dictionary based on field types.
        """
        data = self.model_dump()
        redis_data = {}
        # Note: Here we only store the fields that are in the model
        # and ignore any extra fields that might be present in the data
        # Iterate over fields and encode based on their type
        for field_name, field_info in self.model_fields.items():
            value = data.get(field_name)
            if value is None or field_name == "score":
                continue
            elif field_name == "config":
                redis_data[field_name] = value
            elif field_info.annotation in {str, Optional[str]}:
                redis_data[field_name] = value
            elif field_info.annotation in {list, List[str], Optional[List[str]]}:
                redis_data[field_name] = ",".join(value)
            else:
                redis_data[field_name] = json.dumps(value)

        # Expand config fields to store as separate keys
        if "config" in redis_data and redis_data["config"]:
            # Store entire config as JSON to preserve all fields including extra fields
            redis_data["config"] = json.dumps(redis_data["config"])
            # Note: We keep the config as a single JSON field to preserve all extra fields
        return redis_data

    @classmethod
    def from_redis_dict(cls, service_data: Dict[str, Any], in_bytes=True):
        """
        Deserialize a Redis-compatible dictionary back to a model instance.
        """
        converted_data = {}
        
        # Check if config is stored as JSON (new format)
        config_key = b"config" if in_bytes else "config"
        if config_key in service_data:
            config_value = service_data[config_key]
            if config_value:
                # Deserialize JSON config
                if isinstance(config_value, bytes):
                    config_dict = json.loads(config_value.decode("utf-8"))
                elif isinstance(config_value, str):
                    config_dict = json.loads(config_value)
                else:
                    config_dict = config_value
                # Store the parsed config
                converted_data["config"] = ServiceConfig.model_validate(config_dict)
            # Remove config from service_data so it's not processed again
            del service_data[config_key]

        for field_name, field_info in cls.model_fields.items():
            if field_name == "config":
                # Config is already handled above
                continue
                
            if not in_bytes:
                value = service_data.get(field_name)
            else:
                value = service_data.get(field_name.encode("utf-8"))
            if value is None:
                converted_data[field_name] = None
            elif field_info.annotation in {str, Optional[str]}:
                converted_data[field_name] = (
                    value if isinstance(value, str) else value.decode("utf-8")
                )
            elif field_info.annotation in {list, List[str], Optional[List[str]]}:
                converted_data[field_name] = (
                    value.split(",")
                    if isinstance(value, str)
                    else value.decode("utf-8").split(",")
                )
            else:
                value_str = value if isinstance(value, str) else value.decode("utf-8")
                converted_data[field_name] = json.loads(value_str)
        return cls.model_validate(converted_data)

    @classmethod
    def model_validate(cls, data):
        data = data.copy()
        if "config" in data and data["config"] is not None:
            data["config"] = ServiceConfig.model_validate(data["config"])
        return super().model_validate(data)


class RemoteService(ServiceInfo):
    pass


class UserTokenInfo(BaseModel):
    """Represent user profile."""

    token: constr(max_length=4096)  # type: ignore
    workspace: Optional[str] = None
    expires_in: Optional[int] = None
    email: Optional[EmailStr] = None
    email_verified: Optional[bool] = None
    name: Optional[constr(max_length=64)] = None  # type: ignore
    nickname: Optional[constr(max_length=64)] = None  # type: ignore
    user_id: Optional[constr(max_length=64)] = None  # type: ignore
    picture: Optional[AnyHttpUrl] = None


class UserPermission(str, Enum):
    """Represent the permission of the workspace."""

    read = "r"
    read_write = "rw"
    admin = "a"


class ScopeInfo(BaseModel):
    """Represent scope info.

    Tokens with extra_scopes are treated as "specialized" tokens that can ONLY
    be used for their intended purpose (indicated by the scopes), NOT for general
    workspace access like WebSocket connections or service calls.

    Scope naming convention: Use format `resource:action` (e.g., `file:download`,
    `artifact:read`). These scopes must be validated by the endpoints that handle
    those specific operations.
    """

    model_config = ConfigDict(extra="allow")

    current_workspace: Optional[str] = None
    workspaces: Optional[Dict[str, UserPermission]] = {}
    client_id: Optional[str] = None
    extra_scopes: Optional[List[str]] = []  # extra scopes for specialized tokens

    def is_specialized(self) -> bool:
        """Check if this scope represents a specialized token.

        Specialized tokens have extra_scopes with our custom format (resource:action)
        and can only be used for their specific purpose (e.g., file downloads),
        not for general workspace access.

        Standard OAuth scopes (like 'openid', 'profile', 'email') don't contain
        colons and are NOT considered specialized tokens.

        Our specialized scopes follow the format `resource:action` (e.g., `file:download`,
        `artifact:read`), which always contains a colon.
        """
        return bool(self.get_specialized_scopes())

    def get_specialized_scopes(self) -> List[str]:
        """Get only the specialized scopes (our custom resource:action format).

        Returns:
            List of scopes that follow our resource:action format (contain colon)
        """
        if not self.extra_scopes:
            return []
        # Our specialized scopes use resource:action format (contain colon)
        # Standard OAuth scopes (openid, profile, email, etc.) don't contain colons
        return [s for s in self.extra_scopes if ":" in s]

    def has_scope(self, required_scope: str) -> bool:
        """Check if this scope includes the required specialized scope.

        Args:
            required_scope: The scope to check for (e.g., "file:download")

        Returns:
            True if the scope is present in extra_scopes
        """
        return required_scope in (self.extra_scopes or [])

    def has_scope_prefix(self, prefix: str) -> bool:
        """Check if any scope starts with the given prefix.

        Args:
            prefix: The scope prefix to check for (e.g., "file:download:")

        Returns:
            True if any scope in extra_scopes starts with the prefix
        """
        for scope in (self.extra_scopes or []):
            if scope.startswith(prefix):
                return True
        return False

    def get_scope_with_prefix(self, prefix: str) -> Optional[str]:
        """Get the first scope that starts with the given prefix.

        Args:
            prefix: The scope prefix to look for

        Returns:
            The matching scope string, or None if not found
        """
        for scope in (self.extra_scopes or []):
            if scope.startswith(prefix):
                return scope
        return None


class UserInfo(BaseModel):
    """Represent user info."""

    id: str
    roles: List[str]
    is_anonymous: bool
    scope: Optional[ScopeInfo] = None
    email: Optional[EmailStr] = None
    parent: Optional[str] = None
    expires_at: Optional[float] = None
    current_workspace: Optional[str] = None
    _metadata: Dict[str, Any] = PrivateAttr(
        default_factory=lambda: {}
    )  # e.g. s3 credential

    def get_workspace(self):
        return f"ws-user-{self.id}"

    def get_effective_user_id(self) -> str:
        """Get the effective user ID for tracking/attribution purposes.

        Returns the parent user ID if this is a child token (generated via
        generate_token), otherwise returns the token's own user ID.
        This ensures consistent attribution in git history and activity logs.
        """
        return self.parent if self.parent else self.id
    
    @classmethod
    def from_context(cls, context: Dict[str, Any]):
        """Create a user info from a context."""
        # context contains 'user' and 'ws'
        user_info = cls.model_validate(context["user"])
        user_info.current_workspace = context.get("ws")
        return user_info

    def get_metadata(self, key=None) -> Dict[str, Any]:
        """Return the metadata."""
        if key:
            return self._metadata.get(key)
        return self._metadata

    def set_metadata(self, key, value):
        """Set the metadata."""
        self._metadata[key] = value

    def get_permission(self, workspace: str):
        """Get the workspace permission."""
        if not self.scope:
            return None
        assert isinstance(workspace, str)
        if self.scope.workspaces.get("*"):
            return self.scope.workspaces["*"]
        return self.scope.workspaces.get(workspace, None)

    def check_permission(self, workspace: str, minimal_permission: UserPermission):
        permission = self.get_permission(workspace)
        if not permission:
            return False
        if minimal_permission == UserPermission.read:
            if permission in [
                UserPermission.read,
                UserPermission.read_write,
                UserPermission.admin,
            ]:
                return True
        elif minimal_permission == UserPermission.read_write:
            if permission in [UserPermission.read_write, UserPermission.admin]:
                return True
        elif minimal_permission == UserPermission.admin:
            if permission == UserPermission.admin:
                return True

        return False

    def is_specialized_token(self) -> bool:
        """Check if this user has a specialized token.

        Specialized tokens have extra_scopes and can only be used for their
        specific purpose (e.g., file downloads), not for general workspace access
        like WebSocket connections or service calls.
        """
        return self.scope is not None and self.scope.is_specialized()

    def has_scope(self, required_scope: str) -> bool:
        """Check if the user's token includes the required specialized scope.

        Args:
            required_scope: The scope to check for (e.g., "file:download")

        Returns:
            True if the scope is present in the user's extra_scopes
        """
        if self.scope is None:
            return False
        return self.scope.has_scope(required_scope)

    def validate_for_general_access(self) -> None:
        """Validate that this token can be used for general workspace access.

        Raises:
            PermissionError: If this is a specialized token that cannot be used
                           for general workspace operations.
        """
        if self.is_specialized_token():
            # Only show specialized scopes (not standard OAuth scopes) in the error
            specialized_scopes = self.scope.get_specialized_scopes()
            scopes = ", ".join(specialized_scopes)
            raise PermissionError(
                f"Specialized token with scopes [{scopes}] cannot be used for "
                f"general workspace access. This token can only be used for its "
                f"intended purpose."
            )

    def validate_scope(self, required_scope: str) -> None:
        """Validate that this token has the required scope for an operation.

        For specialized tokens, verifies the required scope is present.
        For non-specialized tokens, allows all operations (backwards compatible).

        Args:
            required_scope: The scope required for the operation (e.g., "file:download")

        Raises:
            PermissionError: If this is a specialized token without the required scope
        """
        if self.is_specialized_token():
            if not self.has_scope(required_scope):
                allowed_scopes = ", ".join(self.scope.extra_scopes)
                raise PermissionError(
                    f"Token with scopes [{allowed_scopes}] does not have required "
                    f"scope '{required_scope}' for this operation."
                )


class ClientInfo(BaseModel):
    """Represent service."""

    id: str
    parent: Optional[str] = None
    name: Optional[str] = None
    workspace: str
    services: List[SerializeAsAny[ServiceInfo]] = []
    user_info: UserInfo

    @classmethod
    def model_validate(cls, data):
        data = data.copy()
        data["user_info"] = UserInfo.model_validate(data["user_info"])
        data["services"] = [
            ServiceInfo.model_validate(service) for service in data["services"]
        ]
        return super().model_validate(data)


class Artifact(BaseModel):
    """Represent resource description file object."""

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = "generic"
    format_version: str = "0.2.1"
    name: Optional[str] = None
    id: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    documentation: Optional[str] = None
    covers: Optional[List[str]] = None
    badges: Optional[List[str]] = None
    authors: Optional[List[Dict[str, str]]] = None
    attachments: Optional[Dict[str, List[Any]]] = None
    files: Optional[List[Dict[str, Any]]] = None
    config: Optional[Dict[str, Any]] = None
    version: Optional[str] = "0.1.0"
    links: Optional[List[str]] = None
    maintainers: Optional[List[Dict[str, str]]] = None
    license: Optional[str] = None
    git_repo: Optional[str] = None
    source: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    @classmethod
    def model_validate(cls, data):
        data = data.copy()
        if "services" in data and data["services"] is not None:
            data["services"] = [
                ServiceInfo.model_validate(service) for service in data["services"]
            ]
        return super().model_validate(data)


class CollectionArtifact(Artifact):
    """Represent collection artifact."""

    type: Optional[str] = "collection"
    collection: Optional[List[str]] = []
    collection_schema: Optional[Dict[str, Any]] = None

    @classmethod
    def model_validate(cls, data):
        data = data.copy()
        if "collection" in data and data["collection"] is not None:
            data["collection"] = [
                CollectionArtifact.model_validate(artifact)
                for artifact in data["collection"]
            ]
        if "collection_schema" in data and data["collection_schema"] is not None:
            # make sure the schema is a valid json schema
            jsonschema.Draft7Validator.check_schema(data["collection_schema"])
        return super().model_validate(data)


class ApplicationManifest(Artifact):
    """Represent application artifact."""

    passive: Optional[bool] = False
    dependencies: Optional[List[Any]] = []
    requirements: Optional[List[str]] = []
    api_version: Optional[str] = "0.1.0"
    icon: Optional[str] = None
    env: Optional[Union[str, Dict[str, Any]]] = None
    type: Optional[str] = "application"
    entry_point: Optional[str] = None  # entry point for the application
    daemon: Optional[bool] = False  # whether the application is a daemon
    singleton: Optional[bool] = False  # whether the application is a singleton
    services: Optional[List[SerializeAsAny[ServiceInfo]]] = None  # for application
    startup_config: Optional[Dict[str, Any]] = None  # default startup configuration
    service_selection_mode: Optional[str] = (
        None  # default service selection mode for multiple instances
    )
    startup_context: Optional[Dict[str, Any]] = None  # context from installation time
    source_hash: Optional[str] = None  # hash of the source code for singleton checking
    autoscaling: Optional[AutoscalingConfig] = None  # autoscaling configuration


class ServiceTypeInfo(BaseModel):
    """Represent service type info."""

    id: str
    name: str
    definition: Dict[str, Any]
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = {}
    docs: Optional[str] = None


class WorkspaceInfo(BaseModel):
    """Represent a workspace."""

    type: str = "workspace"
    id: Optional[str] = None  # we will use name as id if not provided
    name: str
    description: Optional[str] = None
    persistent: Optional[bool] = False
    owners: Optional[List[str]] = []
    read_only: Optional[bool] = False
    icon: Optional[str] = None
    covers: Optional[List[str]] = None
    docs: Optional[str] = None
    service_types: Optional[Dict[str, ServiceTypeInfo]] = {}
    config: Optional[Dict[str, Any]] = {}
    status: Optional[Dict[str, Any]] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.id is None:
            self.id = self.name

    @classmethod
    def model_validate(cls, data):
        data = data.copy()
        return super().model_validate(data)
    
    def owned_by(self, user_info: Union[UserInfo, str]):
        if isinstance(user_info, str):
            return user_info in self.owners
        # check if user_info.id or user_info.email is in the owners list
        if user_info.id in self.owners or user_info.email in self.owners:
            return True
        return False


class TokenConfig(BaseModel):
    expires_in: Optional[int] = Field(None, description="Expiration time in seconds")
    workspace: Optional[str] = Field(None, description="Workspace name")
    permission: Optional[str] = Field(
        "read_write",
        description="Permission level",
        pattern="^(read|read_write|admin)$",
    )
    extra_scopes: Optional[List[str]] = Field(None, description="List of extra scopes")
    client_id: Optional[str] = Field(
        None,
        description="Optional client ID to restrict the token to a specific client",
    )

    @field_validator("extra_scopes")
    @classmethod
    def validate_scopes(cls, v):
        if ":" in v and v.count(":") == 1:
            prefix = v.split(":")[0]
            if prefix in ["ws", "cid"]:
                raise ValueError("Invalid scope, cannot start with ws or cid")
        return v


background_tasks = set()


class RedisRPCConnection:
    """Represent a Redis connection for handling RPC-like messaging."""
    
    _connections = {}  # Global registry of active connections: {workspace/client_id: connection}

    _counter = Counter("rpc_call_total", "Total RPC calls across all workspaces")
    _client_request_counter = Counter(
        "client_requests_total",
        "Total requests from all clients"
    )
    _client_load_gauge = Gauge(
        "client_load_current",
        "Current load (requests per minute)"
    )
    _connections_created_total = Counter(
        "redis_rpc_connections_created_total",
        "Total RPC connections created",
    )
    _connections_closed_total = Counter(
        "redis_rpc_connections_closed_total",
        "Total RPC connections closed",
    )
    _active_connections_gauge = Gauge(
        "redis_rpc_active_connections",
        "Current active RPC connections",
    )
    _created_total_int = 0
    _closed_total_int = 0

    @classmethod
    def get_metrics_snapshot(cls) -> dict:
        return {
            "rpc_connections": {
                "created_total": cls._created_total_int,
                "closed_total": cls._closed_total_int,
                "active": len(cls._connections),
            }
        }
    _tracker = None

    @classmethod
    def set_activity_tracker(cls, tracker: ActivityTracker):
        cls._tracker = tracker

    def __init__(
        self,
        event_bus: EventBus,
        workspace: str,
        client_id: str,
        user_info: UserInfo,
        manager_id: str,
        readonly: bool = False,
    ):
        """Initialize Redis RPC Connection."""
        assert workspace and "/" not in client_id, "Invalid workspace or client ID"
        self._workspace = workspace
        self._client_id = client_id
        self._user_info = user_info.model_dump()
        self._stop = False
        self._event_bus = event_bus
        self._handle_connected = None
        self._handle_disconnected = None
        self._handle_message = None
        self.manager_id = manager_id
        self._subscriptions = set()  # Local subscription state
        self._readonly = readonly  # Store readonly flag for security checks
        
        # Register this connection in the global registry
        connection_key = f"{self._workspace}/{self._client_id}"
        RedisRPCConnection._connections[connection_key] = self
        # Metrics
        RedisRPCConnection._connections_created_total.inc()
        RedisRPCConnection._created_total_int += 1
        RedisRPCConnection._active_connections_gauge.set(len(RedisRPCConnection._connections))
        
        # Register this client for targeted subscriptions
        # We'll register this asynchronously in on_message to avoid blocking
        self._registration_task = None

    def subscribe(self, event_type: str):
        """Subscribe to an event type."""
        self._subscriptions.add(event_type)
        logger.debug(f"Client {self._workspace}/{self._client_id} subscribed to: {event_type}")

    def unsubscribe(self, event_type: str):
        """Unsubscribe from an event type."""
        self._subscriptions.discard(event_type)
        logger.debug(f"Client {self._workspace}/{self._client_id} unsubscribed from: {event_type}")

    def on_disconnected(self, handler):
        """Register a disconnection event handler."""
        self._handle_disconnected = handler

    def on_connected(self, handler):
        """Register a connection open event handler."""
        self._handle_connected = handler
        assert inspect.iscoroutinefunction(
            handler
        ), "Connect handler must be a coroutine"

    def on_message(self, handler: Callable):
        """Set message handler."""
        self._handle_message = handler
        # Direct messages: always forward
        self._event_bus.on(f"{self._workspace}/{self._client_id}:msg", handler)
        
        # Broadcast messages: filter by subscription
        async def filtered_handler(message):
            # Extract message content to check event type
            message_dict = message
            if isinstance(message, bytes):
                try:
                    unpacker = msgpack.Unpacker(io.BytesIO(message))
                    message_dict = unpacker.unpack()
                except Exception:
                    # If we can't unpack, forward anyway to avoid breaking existing functionality
                    result = handler(message)
                    if asyncio.iscoroutine(result):
                        await result
                    return
            
            event_type = message_dict.get("type")
            if event_type and event_type in self._subscriptions:
                result = handler(message)  # Forward to client
                if asyncio.iscoroutine(result):
                    await result
            # else: ignore (save bandwidth)
        
        # Store filtered_handler reference for proper cleanup
        self._filtered_handler = filtered_handler
        
        self._event_bus.on(f"{self._workspace}/*:msg", filtered_handler)
        
        # Register this client for targeted event subscriptions
        async def register_client():
            await self._event_bus.register_local_client(self._workspace, self._client_id)
            await self._event_bus.subscribe_to_client_events(self._workspace, self._client_id)
            logger.debug(f"Registered and subscribed to events for {self._workspace}/{self._client_id}")
        
        self._registration_task = asyncio.create_task(register_client())
        background_tasks.add(self._registration_task)
        self._registration_task.add_done_callback(background_tasks.discard)
    
        if self._handle_connected:
            task = asyncio.create_task(self._handle_connected(self))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

    async def emit_message(self, data: Union[dict, bytes]):
        """Send message after packing additional info."""
        assert isinstance(data, bytes), "Data must be bytes"
        if self._stop:
            raise ValueError(
                f"Connection has already been closed (client: {self._workspace}/{self._client_id})"
            )
        unpacker = msgpack.Unpacker(io.BytesIO(data))
        message = unpacker.unpack()
        pos = unpacker.tell()
        target_id = message.get("to")
        
        # Security check: Block broadcast messages for readonly clients
        if self._readonly and target_id == "*":
            raise PermissionError(
                f"Read-only client {self._workspace}/{self._client_id} cannot broadcast messages. "
                f"Only point-to-point messages are allowed for read-only clients."
            )
        
        # Handle broadcast messages within workspace
        if target_id == "*":
            # Convert * to workspace/* for proper workspace isolation
            target_id = f"{self._workspace}/*"
        elif "/" not in target_id:
            if "/ws-" in target_id:
                raise ValueError(
                    f"Invalid target ID: {target_id}, it appears that the target is a workspace manager (target_id should starts with */)"
                )
            target_id = f"{self._workspace}/{target_id}"

        source_id = f"{self._workspace}/{self._client_id}"

        # Determine the workspace for the context
        # If the user has a current_workspace set, use that (preserves caller's workspace)
        # Otherwise fall back to the connection's workspace
        if self._workspace == "*":
            # For system connections, check if user has a current_workspace
            user_current_ws = self._user_info.get("current_workspace") or (
                self._user_info.get("scope", {}).get("current_workspace") if isinstance(self._user_info.get("scope"), dict) else None
            )
            if user_current_ws and user_current_ws != "*":
                # Use the user's current workspace (preserves caller's context)
                context_workspace = user_current_ws
            else:
                # Fall back to target's workspace for system connections without specific workspace
                context_workspace = target_id.split("/")[0]
        else:
            # Normal client connections use their own workspace
            context_workspace = self._workspace
        
        message.update(
            {
                "ws": context_workspace,
                "to": target_id,
                "from": source_id,
                "user": self._user_info,
            }
        )

        packed_message = msgpack.packb(message) + data[pos:]
        
        # Use Redis event bus routing (local handlers are attached there)
        await self._event_bus.emit(f"{target_id}:msg", packed_message)
            
        # Increment counters without creating per-workspace/client labels
        RedisRPCConnection._counter.inc()
        RedisRPCConnection._client_request_counter.inc()

        # Update load based on message rate (requests per minute) only for load balancing enabled clients
        if self._is_load_balancing_enabled():
            self._update_load_metric()

        if RedisRPCConnection._tracker:
            await RedisRPCConnection._tracker.reset_timer(
                self._workspace + "/" + self._client_id, "client"
            )

    def _is_load_balancing_enabled(self):
        """Check if load balancing is enabled for this client."""
        return self._client_id.endswith("__rlb")

    def _update_load_metric(self):
        """Update the load metric based on message rate (requests per minute)."""
        try:
            current_time = time.time()
            client_key = f"{self._workspace}/{self._client_id}"
            # Maintain our own per-client request counters
            if not hasattr(RedisRPCConnection, "_client_metrics"):
                RedisRPCConnection._client_metrics = {}

            metrics = RedisRPCConnection._client_metrics.get(client_key)
            if metrics is None:
                metrics = {
                    "last_time": current_time,
                    "last_requests": 0,
                }
                RedisRPCConnection._client_metrics[client_key] = metrics

            # Increment local request counter
            metrics["last_requests"] += 1

            # Compute RPM over a 60s window
            time_diff = current_time - metrics["last_time"]
            if time_diff >= 60:
                rpm = metrics["last_requests"] / (time_diff / 60.0)
                # Set gauge value without creating per-client labels
                RedisRPCConnection._client_load_gauge.set(rpm)
                metrics["last_time"] = current_time
                metrics["last_requests"] = 0
        except Exception as e:
            logger.warning("Failed to update load metric: %s", e)

    def update_load(self, load_value: float):
        """Update the current load for this client."""
        if self._is_load_balancing_enabled():
            # Set gauge value without creating per-client labels
            RedisRPCConnection._client_load_gauge.set(load_value)

    @classmethod
    def get_connection(cls, workspace: str, client_id: str):
        """Get a connection by workspace and client_id."""
        connection_key = f"{workspace}/{client_id}"
        return cls._connections.get(connection_key)

    @classmethod
    def get_client_load(cls, workspace: str, client_id: str) -> float:
        """Get the current load for a specific client (requests per minute)."""
        try:
            # Only return load for load balancing enabled clients
            if not client_id.endswith("__rlb"):
                return 0.0

            # Get load from internal tracking instead of creating labels
            client_key = f"{workspace}/{client_id}"
            if hasattr(cls, "_client_metrics") and client_key in cls._client_metrics:
                metrics = cls._client_metrics[client_key]
                current_time = time.time()
                time_diff = current_time - metrics["last_time"]
                if time_diff > 0:
                    return metrics["last_requests"] / (time_diff / 60.0)
            return 0.0
        except Exception:
            return 0.0


    async def disconnect(self, reason=None):
        """Balanced bulletproof disconnect - zero leaks without server instability."""
        # Mark as stopped FIRST
        self._stop = True

        # Save event_bus reference BEFORE any cleanup that might clear it
        # This ensures we can always unregister handlers even if an error occurs
        event_bus_ref = self._event_bus if hasattr(self, '_event_bus') else None

        # PRIORITY 1: Clean filtered_handler (the main leak source)
        # Clear reference IMMEDIATELY to prevent leaks
        if hasattr(self, '_filtered_handler') and self._filtered_handler:
            handler = self._filtered_handler
            self._filtered_handler = None  # Clear reference IMMEDIATELY
            if event_bus_ref:  # Use saved reference
                # Remove specific handler for this client's pattern
                event_bus_ref.off(f"{self._workspace}/*:msg", handler)
            logger.debug(f"Unregistered filtered_handler for {self._workspace}/{self._client_id}")
        
        # PRIORITY 2: Clear other circular references (but more gently)
        try:
            # Only clear references that are safe to clear
            if getattr(self, '_stop', False):  # Only if actually stopping
                # Don't clear _event_bus yet - we'll do it at the very end
                self._registration_task = None
                self._subscriptions = None
                self._handle_connected = None
                self._handle_disconnected = None
        except Exception as e:
            logger.debug(f"Error clearing circular references: {e}")

        
        # PRIORITY 3: Clear RPC references (but less aggressively)
        try:
            cleared_refs = 0
            # Only check a limited number of objects to avoid performance issues
            objects_checked = 0
            for obj in gc.get_objects():
                if objects_checked > 1000:  # Limit to first 1000 objects
                    break
                objects_checked += 1
                
                if obj.__class__.__name__ == 'RPC':
                    if hasattr(obj, '_connection') and obj._connection is self:
                        obj._connection = None
                        cleared_refs += 1
                    if (hasattr(obj, '_emit_message') and 
                        hasattr(obj._emit_message, '__self__') and 
                        obj._emit_message.__self__ is self):
                        async def _emit_disconnected(data):
                            raise RuntimeError("Connection has been disconnected")
                        obj._emit_message = _emit_disconnected
                        cleared_refs += 1
            
            if cleared_refs > 0:
                logger.debug(f"Cleared {cleared_refs} RPC references for {self._workspace}/{self._client_id}")
                
        except Exception as e:
            logger.debug(f"Error clearing RPC references: {e}")
        
        # Standard disconnect logic (preserved)
        try:
            # Unregister this connection from the global registry
            connection_key = f"{self._workspace}/{self._client_id}"
            RedisRPCConnection._connections.pop(connection_key, None)
            
            # Update metrics
            RedisRPCConnection._connections_closed_total.inc()
            RedisRPCConnection._closed_total_int += 1
            RedisRPCConnection._active_connections_gauge.set(len(RedisRPCConnection._connections))
            
            # Clean up direct message handler using saved event_bus reference
            if self._handle_message and event_bus_ref:
                # Remove specific handler for this client's direct messages
                event_bus_ref.off(f"{self._workspace}/{self._client_id}:msg", self._handle_message)
                self._handle_message = None
            else:
                # Just clear the reference if event_bus is not available
                self._handle_message = None
            
            # Unregister this client from local routing and unsubscribe from events
            try:
                # Cancel registration task if still running
                if self._registration_task and not self._registration_task.done():
                    self._registration_task.cancel()
                
                # Unsubscribe from client events and unregister using saved event_bus reference
                if event_bus_ref:
                    await event_bus_ref.unsubscribe_from_client_events(self._workspace, self._client_id)
                    await event_bus_ref.unregister_local_client(self._workspace, self._client_id)
                    logger.debug(f"Unsubscribed and unregistered {self._workspace}/{self._client_id}")
                else:
                    logger.debug(f"Event bus not available for {self._workspace}/{self._client_id}")
            except Exception as e:
                logger.warning(f"Error during client cleanup: {e}")
        
            logger.debug(f"Redis Connection Disconnected: {self._workspace}/{self._client_id}")
            
            # Handle disconnection callback
            if self._handle_disconnected:
                try:
                    await self._handle_disconnected(reason)
                except Exception as e:
                    logger.warning(f"Error in disconnect handler: {e}")

            # Activity tracker cleanup
            if RedisRPCConnection._tracker:
                try:
                    entity_id = self._workspace + "/" + self._client_id
                    if RedisRPCConnection._tracker.is_registered(entity_id, "client"):
                        await RedisRPCConnection._tracker.remove_entity(entity_id, "client")
                except Exception as e:
                    logger.warning(f"Error removing from tracker: {e}")

            # Clean up load gauge metrics for load balancing enabled clients
            if self._is_load_balancing_enabled():
                try:
                    client_key = f"{self._workspace}/{self._client_id}"
                    if (
                        hasattr(RedisRPCConnection, "_client_metrics")
                        and client_key in RedisRPCConnection._client_metrics
                    ):
                        del RedisRPCConnection._client_metrics[client_key]
                    RedisRPCConnection._client_load_gauge.set(0)
                except Exception as e:
                    logger.warning(f"Error cleaning load metrics: {e}")
                    
        except Exception as e:
            logger.warning(f"Standard disconnect logic error: {e}")
        
        # Gentle garbage collection (only occasionally)
        try:
            if RedisRPCConnection._closed_total_int % 20 == 0:  # Every 20 disconnects
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collected {collected} objects")
        except Exception as e:
            logger.debug(f"GC error: {e}")
        
        # FINALLY: Clear event_bus reference after all operations are done
        # This must be last to avoid NoneType errors during cleanup
        self._event_bus = None
        
        logger.debug(f"Balanced bulletproof disconnect completed for {self._workspace}/{self._client_id}")


class RedisEventBus:
    """Represent a redis event bus."""

    _counter = Counter(
        "event_bus", "Counts the events on the redis event bus", ["event", "status"]
    )
    _messages_processed_total = Counter(
        "redis_eventbus_messages_processed_total",
        "Total messages processed by RedisEventBus",
    )
    _pubsub_latency = Gauge(
        "redis_pubsub_latency_seconds",
        "Redis pubsub latency in seconds",
    )
    _active_pubsub_connections = Gauge(
        "redis_eventbus_active_pubsub_connections",
        "Current active Redis pubsub connections",
    )
    _pubsub_connections_created_total = Counter(
        "redis_eventbus_pubsub_connections_created_total",
        "Total Redis pubsub connections created",
    )
    _pubsub_connections_closed_total = Counter(
        "redis_eventbus_pubsub_connections_closed_total",
        "Total Redis pubsub connections closed",
    )
    _patterns_subscribed_total = Counter(
        "redis_eventbus_patterns_subscribed_total",
        "Total pattern subscriptions through RedisEventBus",
    )
    _patterns_unsubscribed_total = Counter(
        "redis_eventbus_patterns_unsubscribed_total",
        "Total pattern unsubscriptions through RedisEventBus",
    )
    _active_local_clients_gauge = Gauge(
        "redis_eventbus_local_clients",
        "Current number of local clients known to RedisEventBus",
    )
    _active_patterns_gauge = Gauge(
        "redis_eventbus_active_patterns",
        "Current number of active subscribed patterns",
    )
    _reconnect_attempts_total = Counter(
        "redis_eventbus_reconnect_attempts_total",
        "Total reconnect attempts for RedisEventBus pubsub",
    )
    # mirror values for admin metrics snapshot (int counters)
    _messages_processed_total_int = 0
    _pubsub_connections_created_total_int = 0
    _pubsub_connections_closed_total_int = 0
    _patterns_subscribed_total_int = 0
    _patterns_unsubscribed_total_int = 0
    _reconnect_attempts_total_int = 0

    @classmethod
    def get_metrics_snapshot(cls, instance: "RedisEventBus") -> dict:
        return {
            "eventbus": {
                "messages_processed_total": cls._messages_processed_total_int,
                "pubsub_connections": {
                    "created_total": cls._pubsub_connections_created_total_int,
                    "closed_total": cls._pubsub_connections_closed_total_int,
                    "active": 1 if instance._pubsub else 0,
                },
                "patterns": {
                    "subscribed_total": cls._patterns_subscribed_total_int,
                    "unsubscribed_total": cls._patterns_unsubscribed_total_int,
                    "active": len(instance._subscribed_patterns),
                },
                "local_clients": len(instance._local_clients),
                "reconnect_attempts_total": cls._reconnect_attempts_total_int,
            }
        }

    def __init__(self, redis) -> None:
        """Initialize the event bus."""
        self._redis = redis
        self._handle_connected = None
        self._stop = False
        self._local_event_bus = EventBus(logger)
        self._redis_event_bus = EventBus(logger)
        # Track local clients for optimized routing
        self._local_clients = set()  # Set of "workspace/client_id" strings
        self._subscribed_patterns = set()  # Track which patterns we've subscribed to
        self._pubsub = None  # Store the pubsub object for dynamic subscriptions
        # Ensure latency gauge exists in /metrics even if not updated elsewhere
        try:
            RedisEventBus._pubsub_latency.set(0)
        except Exception:
            pass
        # Health and circuit breaker state
        self._consecutive_failures = 0
        self._max_failures = int(os.environ.get("HYPHA_EVENTBUS_MAX_FAILURES", "5"))
        self._circuit_breaker_open = False
        self._last_successful_connection: Optional[float] = None

    async def register_local_client(self, workspace: str, client_id: str):
        """Register a local client for optimized event routing."""
        client_key = f"{workspace}/{client_id}"
        self._local_clients.add(client_key)
        logger.debug(f"Registered local client: {client_key}")
        # metrics
        try:
            RedisEventBus._active_local_clients_gauge.set(len(self._local_clients))
        except Exception:
            pass

    async def unregister_local_client(self, workspace: str, client_id: str):
        """Unregister a local client."""
        client_key = f"{workspace}/{client_id}"
        self._local_clients.discard(client_key)
        logger.debug(f"Unregistered local client: {client_key}")
        try:
            RedisEventBus._active_local_clients_gauge.set(len(self._local_clients))
        except Exception:
            pass

    async def subscribe_to_client_events(self, workspace: str, client_id: str):
        """Subscribe to events for a specific client using targeted prefix."""
        client_key = f"{workspace}/{client_id}"
        # Subscribe to targeted messages for this client
        pattern = f"targeted:{client_key}:*"
        if pattern not in self._subscribed_patterns:
            # Record desired subscription regardless of pubsub availability
            self._subscribed_patterns.add(pattern)
            RedisEventBus._active_patterns_gauge.set(len(self._subscribed_patterns))
            if self._pubsub and not self._circuit_breaker_open:
                try:
                    # Add timeout to prevent hanging on Redis connection issues
                    await asyncio.wait_for(self._pubsub.psubscribe(pattern), timeout=5.0)
                    logger.debug("Subscribed to client events: %s", pattern)
                    RedisEventBus._patterns_subscribed_total.inc()
                    RedisEventBus._patterns_subscribed_total_int += 1
                    # Mark successful operation
                    self._consecutive_failures = 0
                except asyncio.TimeoutError:
                    logger.warning("Timeout subscribing to client events %s", pattern)
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_failures:
                        self._circuit_breaker_open = True
                except Exception as e:
                    logger.warning("Failed to subscribe to client events %s: %s", pattern, e)
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_failures:
                        self._circuit_breaker_open = True

    async def unsubscribe_from_client_events(self, workspace: str, client_id: str):
        """Unsubscribe from events for a specific client."""
        client_key = f"{workspace}/{client_id}"
        patterns_to_remove = []
        
        # Find all patterns related to this client
        for pattern in list(self._subscribed_patterns):
            if client_key in pattern:
                patterns_to_remove.append(pattern)
        
        # Unsubscribe from all related patterns
        for pattern in patterns_to_remove:
            if self._pubsub:
                try:
                    await self._pubsub.punsubscribe(pattern)
                    RedisEventBus._patterns_unsubscribed_total.inc()
                    RedisEventBus._patterns_unsubscribed_total_int += 1
                    logger.debug(f"Unsubscribed from pattern: {pattern}")
                except Exception as e:
                    logger.warning("Failed to unsubscribe from client events %s: %s", pattern, e)
            self._subscribed_patterns.discard(pattern)
        
        RedisEventBus._active_patterns_gauge.set(len(self._subscribed_patterns))

    def is_local_client(self, workspace: str, client_id: str) -> bool:
        """Check if a client is local to this server instance."""
        client_key = f"{workspace}/{client_id}"
        return client_key in self._local_clients
    
    async def cleanup_orphaned_patterns(self):
        """Clean up orphaned patterns that may have accumulated."""
        if not self._pubsub:
            return
            
        # Get current active patterns from Redis
        try:
            # This is a best-effort cleanup - we can't easily get Redis pubsub state
            # So we'll clean up patterns that are clearly orphaned
            patterns_to_clean = []
            for pattern in list(self._subscribed_patterns):
                # Check if pattern looks like it's for a disconnected client
                if "targeted:" in pattern and ":" in pattern:
                    parts = pattern.split(":")
                    if len(parts) >= 3:
                        workspace_client = parts[1]
                        if "/" in workspace_client:
                            workspace, client_id = workspace_client.split("/", 1)
                            # Check if this client is still in our local clients
                            if not self.is_local_client(workspace, client_id):
                                patterns_to_clean.append(pattern)
            
            # Clean up orphaned patterns
            for pattern in patterns_to_clean:
                try:
                    await self._pubsub.punsubscribe(pattern)
                    self._subscribed_patterns.discard(pattern)
                    logger.debug(f"Cleaned up orphaned pattern: {pattern}")
                except Exception as e:
                    logger.warning(f"Failed to clean up orphaned pattern {pattern}: {e}")
            
            RedisEventBus._active_patterns_gauge.set(len(self._subscribed_patterns))
            
        except Exception as e:
            logger.warning(f"Error during pattern cleanup: {e}")

    async def _ensure_subscription(self, workspace: str, client_id: str):
        """Ensure we're subscribed to events for a specific client if it's not local."""
        client_key = f"{workspace}/{client_id}"
        if not self.is_local_client(workspace, client_id):
            pattern = f"event:*:{client_key}:*"
            if pattern not in self._subscribed_patterns and self._pubsub:
                try:
                    await self._pubsub.psubscribe(pattern)
                    self._subscribed_patterns.add(pattern)
                    logger.debug(f"Subscribed to pattern: {pattern}")
                except Exception as e:
                    logger.warning(f"Failed to subscribe to pattern {pattern}: {e}")

    async def init(self):
        """Setup the event bus."""
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._loop = loop

        # Start the Redis subscription task
        self._subscribe_task = loop.create_task(self._subscribe_redis())

        # Wait for readiness signal
        await self._ready

    def on(self, event_name, func):
        """Register a callback for an event from Redis."""
        self._redis_event_bus.on(event_name, func)

    def off(self, event_name, func=None):
        """Unregister a callback for an event from Redis."""
        self._redis_event_bus.off(event_name, func)

    def once(self, event_name, func):
        """Register a callback for an event and remove it once triggered."""

        def once_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self._redis_event_bus.off(event_name, once_wrapper)
            return result

        self._redis_event_bus.on(event_name, once_wrapper)

    async def wait_for(self, event_name, match=None, timeout=None):
        """Wait for an event from either local or Redis event bus."""
        local_future = asyncio.create_task(
            self._local_event_bus.wait_for(event_name, match, timeout)
        )
        redis_future = asyncio.create_task(
            self._redis_event_bus.wait_for(event_name, match, timeout)
        )
        done, pending = await asyncio.wait(
            [local_future, redis_future], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        return done.pop().result()

    def on_local(self, event_name, func):
        """Register a callback for a local event."""
        return self._local_event_bus.on(event_name, func)

    def off_local(self, event_name, func=None):
        """Unregister a callback for a local event."""
        return self._local_event_bus.off(event_name, func)

    def once_local(self, event_name, func):
        """Register a callback for a local event and remove it once triggered."""

        def once_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self._local_event_bus.off(event_name, once_wrapper)
            return result

        return self._local_event_bus.on(event_name, once_wrapper)

    async def wait_for_local(self, event_name, match=None, timeout=None):
        """Wait for local event."""
        return await self._local_event_bus.wait_for(event_name, match, timeout)

    async def emit_local(self, event_name, data=None):
        """Emit a local event."""
        if not self._ready.done():
            self._ready.set_result(True)
        local_task = self._local_event_bus.emit(event_name, data)
        if asyncio.iscoroutine(local_task):
            await local_task

    async def broadcast(self, workspace: str, event_type: str, data: dict):
        """Broadcast a system event to all clients in a workspace."""
        # First emit locally for local listeners (like apps system)
        local_emit_result = self.emit_local(event_type, data)
        if asyncio.iscoroutine(local_emit_result):
            await local_emit_result
        
        # Then emit to Redis for remote clients with wrapped message format
        message = {
            "type": event_type,
            "to": "*",
            "data": data
        }
        event_name = f"{workspace}/*:msg"
        emit_result = self.emit(event_name, message)
        if asyncio.iscoroutine(emit_result):
            await emit_result
    
    def emit(self, event_name, data):
        """Emit an event with smart routing using prefix-based system.

        For targeted messages to local clients, this skips Redis entirely
        and delivers directly via the local event bus for better performance.
        """
        if not self._ready.done():
            self._ready.set_result(True)

        # Determine the message type for routing decisions
        is_targeted_message = event_name.endswith(":msg") and "/" in event_name
        is_broadcast_message = event_name.endswith("/*:msg")

        # OPTIMIZATION: For targeted messages to local clients, skip Redis entirely
        if is_targeted_message and not is_broadcast_message:
            target_client = event_name[:-4]  # Remove ":msg" suffix
            if "/" in target_client:
                workspace, client_id = target_client.split("/", 1)
                if self.is_local_client(workspace, client_id):
                    # LOCAL ONLY - skip Redis for better performance
                    # Use _redis_event_bus since handlers are registered there
                    logger.debug(f"Local-only delivery for {target_client}")
                    return self._redis_event_bus.emit(event_name, data)

        # For remote clients or broadcast messages, use both local and Redis
        tasks = []

        # Emit locally first
        local_task = self._local_event_bus.emit(event_name, data)
        if asyncio.iscoroutine(local_task):
            tasks.append(local_task)

        # Determine the appropriate prefix for Redis
        if is_targeted_message and not is_broadcast_message:
            prefix = "targeted:"
        else:
            # Use broadcast: prefix for broadcast messages and system events
            prefix = "broadcast:"

        # Emit to Redis with appropriate prefix
        if isinstance(data, dict):
            data = json.dumps(data).encode("utf-8")
            data_type_byte = b'd'
        elif isinstance(data, str):
            data = data.encode("utf-8")
            data_type_byte = b's'
        else:
            assert data and isinstance(
                data, (str, bytes)
            ), "Data must be a string or bytes"
            if isinstance(data, str):
                data = data.encode("utf-8")
            data_type_byte = b'b'

        # Use prefix-based channel naming with data type encoded in payload
        redis_channel = prefix + event_name
        payload = data_type_byte + data
        global_task = self._loop.create_task(
            self._redis.publish(redis_channel, payload)
        )
        tasks.append(global_task)

        if tasks:
            return asyncio.gather(*tasks)
        else:
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut

    async def stop(self):
        """Stop the event bus."""
        self._stop = True

        # Cancel tasks first
        if self._subscribe_task:
            self._subscribe_task.cancel()

        # Wait for tasks to complete
        try:
            await asyncio.gather(
                self._subscribe_task, return_exceptions=True
            )
        except asyncio.CancelledError:
            pass

    async def _subscribe_redis(self):
        """Handle Redis subscription with automatic reconnection."""
        cpu_count = os.cpu_count() or 1
        concurrent_tasks = cpu_count * 10
        logger.info(
            f"Starting Redis event bus with {concurrent_tasks} concurrent task processing"
        )

        while not self._stop:
            pubsub = None
            try:
                pubsub = self._redis.pubsub()
                self._pubsub = pubsub  # Store reference for dynamic subscriptions
                self._stop = False
                # metrics
                RedisEventBus._pubsub_connections_created_total.inc()
                RedisEventBus._pubsub_connections_created_total_int += 1
                RedisEventBus._active_pubsub_connections.set(1)
                semaphore = asyncio.Semaphore(concurrent_tasks)
                
                # Subscribe to all broadcast messages (server-wide events)
                await pubsub.psubscribe("broadcast:*")
                
                # Re-subscribe to any existing targeted client patterns
                for pattern in list(self._subscribed_patterns):
                    try:
                        await pubsub.psubscribe(pattern)
                        logger.debug(f"Re-subscribed to pattern: {pattern}")
                    except Exception as e:
                        logger.warning(f"Failed to re-subscribe to pattern {pattern}: {e}")
                        self._subscribed_patterns.discard(pattern)
                
                if not self._ready.done():
                    self._ready.set_result(True)
                self._counter.labels(event="subscription", status="success").inc()
                # Mark healthy on successful subscription
                self._last_successful_connection = time.time()
                self._consecutive_failures = 0
                self._circuit_breaker_open = False

                while not self._stop:
                    try:
                        msg = await pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=0.2
                        )
                        if msg:
                            # channel = msg["channel"].decode("utf-8")  # not used
                            task = asyncio.create_task(
                                self._process_message(msg, semaphore)
                            )
                            background_tasks.add(task)
                            task.add_done_callback(background_tasks.discard)
                    except Exception as e:
                        logger.warning(f"Error getting message: {str(e)}")
                        self._counter.labels(
                            event="message_processing", status="failure"
                        ).inc()
                        # Count consecutive failures and trip breaker if necessary
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= self._max_failures:
                            self._circuit_breaker_open = True
                        await asyncio.sleep(0.1)  # Prevent tight loop on errors

            except Exception as exp:
                logger.error(f"Subscription error: {str(exp)}")
                self._counter.labels(event="subscription", status="failure").inc()
                if not self._ready.done():
                    self._ready.set_exception(exp)
                # metrics
                RedisEventBus._reconnect_attempts_total.inc()
                RedisEventBus._reconnect_attempts_total_int += 1
                # Count consecutive failures and trip breaker if necessary
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    self._circuit_breaker_open = True
                await asyncio.sleep(1)  # Prevent tight loop on connection errors
            finally:
                # Always clean up pubsub connection
                if pubsub:
                    try:
                        await pubsub.unsubscribe()
                        await pubsub.close()
                        # metrics
                        RedisEventBus._pubsub_connections_closed_total.inc()
                        RedisEventBus._pubsub_connections_closed_total_int += 1
                        RedisEventBus._active_pubsub_connections.set(0)
                    except Exception as e:
                        logger.warning(f"Error cleaning up pubsub connection: {str(e)}")
                self._pubsub = None
                # Don't clear subscribed_patterns here - we want to re-subscribe on reconnect

    async def _process_message(self, msg, semaphore):
        """Process a single message while respecting the semaphore."""
        async with semaphore:
            try:
                channel = msg["channel"].decode("utf-8") 
                RedisEventBus._counter.labels(event="*", status="processed").inc()
                RedisEventBus._messages_processed_total.inc()
                RedisEventBus._messages_processed_total_int += 1
                # Mark healthy upon receiving any message
                self._last_successful_connection = time.time()
                self._consecutive_failures = 0
                self._circuit_breaker_open = False

                # Handle prefix-based system with data type encoded in payload
                if channel.startswith("broadcast:") or channel.startswith("targeted:"):
                    # Extract event name from channel
                    if channel.startswith("broadcast:"):
                        event_type = channel[10:]  # len("broadcast:")
                    else:  # targeted:
                        event_type = channel[9:]   # len("targeted:")
                    
                    # Decode data type from first byte of payload
                    payload = msg["data"]
                    if len(payload) == 0:
                        logger.warning(f"Empty payload for channel: {channel}")
                        return
                        
                    data_type_byte = payload[:1]
                    data_payload = payload[1:]
                    
                    if data_type_byte == b'b':
                        # Binary data
                        data = data_payload
                    elif data_type_byte == b'd':
                        # JSON data
                        data = json.loads(data_payload)
                    elif data_type_byte == b's':
                        # String data
                        data = data_payload.decode("utf-8")
                    else:
                        logger.warning(f"Unknown data type byte: {data_type_byte} for channel: {channel}")
                        return
                    
                    await self._redis_event_bus.emit(event_type, data)
                    if ":" not in event_type:
                        RedisEventBus._counter.labels(
                            event=event_type, status="processed"
                        ).inc()
                else:
                    logger.info("Unknown channel: %s", channel)
            except Exception as exp:
                logger.exception(f"Error processing message: {exp}")
                RedisEventBus._counter.labels(
                    event="message_processing", status="error"
                ).inc()
                # Count consecutive failures and possibly open breaker
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    self._circuit_breaker_open = True

    async def _check_pubsub_health(self, timeout: float = 2.0) -> bool:
        """Verify Redis pubsub round-trip health.

        This publishes a unique healthcheck event through Redis and waits for it
        to arrive via the Redis-backed event bus (not the local event bus).
        Returns True if the event is observed within the timeout.
        """
        try:
            if self._pubsub is None or self._circuit_breaker_open:
                return False

            event_name = f"healthcheck:{uuid.uuid4().hex}"
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()

            def handler(_):
                if not future.done():
                    future.set_result(True)

            # Listen only on the Redis-backed event bus to avoid local short-circuiting
            self._redis_event_bus.on(event_name, handler)
            try:
                emit_result = self.emit(event_name, {"ts": time.time()})
                if asyncio.iscoroutine(emit_result):
                    await emit_result
                await asyncio.wait_for(future, timeout)
                return True
            finally:
                self._redis_event_bus.off(event_name, handler)
        except Exception:
            return False
