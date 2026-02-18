import re
import json
import hashlib
import asyncio
import logging
import traceback
import time
import weakref
import os
import sys
from typing import Optional, Union, List, Any, Dict
import random
import datetime
import numpy as np

from fakeredis import aioredis


from hypha_rpc import RPC
from hypha_rpc.utils.schema import schema_method
from hypha_rpc.rpc import RemoteException
from pydantic import BaseModel, Field
from hypha.core import RedisRPCConnection
from sqlalchemy import (
    Column,
    JSON,
    Index,
    text,
    select,
    func,
)
from sqlalchemy.exc import IntegrityError
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
from hypha.core.auth import generate_auth_token, create_scope, parse_auth_token
from hypha.utils import EventBus, random_id

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("workspace")
logger.setLevel(LOGLEVEL)

SERVICE_SUMMARY_FIELD = ["id", "name", "type", "description", "config"]
ANONYMOUS_USER_WS_PREFIX = "ws-user-anonymouz-"

# Ensure the client_id is safe
_allowed_characters = re.compile(r"^[a-zA-Z0-9-_/|*]*$")
# Use WeakSet to automatically clean up completed tasks

background_tasks = weakref.WeakSet()

EVENT_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]+$")
EVENT_CATEGORIES = {"system", "application", "billing", "audit"}
EVENT_LEVELS = {"debug", "info", "warning", "error"}
INTERCEPTOR_CONDITION_OPS = {"eq", "ne", "gt", "lt", "contains"}
INTERCEPTOR_ACTION_TYPES = {"allow", "stop", "recover"}
INTERCEPTOR_STOP_REASONS = {"policy", "security", "limit"}
INTERCEPT_STOP_CODE_BY_REASON = {
    "policy": "INTERCEPT_STOP_POLICY",
    "security": "INTERCEPT_STOP_SECURITY",
    "limit": "INTERCEPT_STOP_LIMIT",
}
BILLING_USAGE_EVENT_TYPE = "billing.usage.recorded"
ALLOWED_USAGE_SUMMARY_GROUP_BY = {"billing_point", "app_id", "workspace", "unit"}


# Function to return a timezone-naive datetime
def naive_utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def normalize_naive_utc_datetime(
    value: Optional[datetime.datetime],
) -> Optional[datetime.datetime]:
    """Normalize datetimes to timezone-naive UTC for DB compatibility."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)


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
    __table_args__ = (
        Index("ix_event_logs_workspace_timestamp", "workspace", "timestamp"),
        Index(
            "ix_event_logs_workspace_category_timestamp",
            "workspace",
            "category",
            "timestamp",
        ),
        Index(
            "ix_event_logs_workspace_event_type_timestamp",
            "workspace",
            "event_type",
            "timestamp",
        ),
        Index(
            "ux_event_logs_workspace_idempotency_key_billing",
            "workspace",
            "idempotency_key",
            unique=True,
            postgresql_where=text(
                "category = 'billing' AND idempotency_key IS NOT NULL"
            ),
            sqlite_where=text("category = 'billing' AND idempotency_key IS NOT NULL"),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str = Field(nullable=False)
    category: str = Field(default="application", nullable=False)
    level: str = Field(default="info", nullable=False)
    workspace: str = Field(nullable=False)
    user_id: str = Field(nullable=False)
    app_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    idempotency_key: Optional[str] = Field(default=None)
    billing_account_id: Optional[str] = Field(
        default=None, foreign_key="billing_accounts.id", index=True
    )
    retention_policy_id: Optional[int] = Field(
        default=None, foreign_key="billing_retention_policies.id", index=True
    )
    timestamp: datetime.datetime = Field(default_factory=naive_utc_now, index=True)
    data: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    def to_dict(self) -> dict:
        """Convert the SQLModel model instance to a dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "category": self.category,
            "level": self.level,
            "workspace": self.workspace,
            "user_id": self.user_id,
            "app_id": self.app_id,
            "session_id": self.session_id,
            "idempotency_key": self.idempotency_key,
            "billing_account_id": self.billing_account_id,
            "retention_policy_id": self.retention_policy_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "data": self.data,
        }


class BillingRetentionPolicy(SQLModel, table=True):
    __tablename__ = "billing_retention_policies"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: Optional[str] = Field(default=None)
    billable_usage_retention_days: int = Field(default=395, nullable=False)
    operational_event_retention_days: int = Field(default=90, nullable=False)
    config: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    created_at: datetime.datetime = Field(default_factory=naive_utc_now, nullable=False)
    updated_at: datetime.datetime = Field(default_factory=naive_utc_now, nullable=False)


class BillingAccount(SQLModel, table=True):
    __tablename__ = "billing_accounts"
    __table_args__ = (
        Index(
            "ux_billing_accounts_stripe_customer_id",
            "stripe_customer_id",
            unique=True,
            postgresql_where=text("stripe_customer_id IS NOT NULL"),
            sqlite_where=text("stripe_customer_id IS NOT NULL"),
        ),
    )

    id: str = Field(
        default_factory=lambda: random_id(readable=False),
        primary_key=True,
    )
    stripe_customer_id: Optional[str] = Field(default=None)
    stripe_subscription_id: Optional[str] = Field(default=None, index=True)
    stripe_subscription_status: Optional[str] = Field(default=None)
    current_period_start: Optional[datetime.datetime] = Field(default=None)
    current_period_end: Optional[datetime.datetime] = Field(default=None)
    last_webhook_event_id: Optional[str] = Field(default=None)
    last_webhook_event_ts: Optional[datetime.datetime] = Field(default=None)
    stripe_snapshot: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    plan_id: Optional[str] = Field(default=None)
    price_id: Optional[str] = Field(default=None)
    entitlement_snapshot_ref: Optional[str] = Field(default=None)
    retention_policy_id: Optional[int] = Field(
        default=None, foreign_key="billing_retention_policies.id", index=True
    )
    created_at: datetime.datetime = Field(default_factory=naive_utc_now, nullable=False)
    updated_at: datetime.datetime = Field(default_factory=naive_utc_now, nullable=False)


class WorkspaceBillingAccount(SQLModel, table=True):
    __tablename__ = "workspace_billing_accounts"
    __table_args__ = (
        Index(
            "ux_workspace_billing_accounts_workspace_active",
            "workspace",
            unique=True,
            postgresql_where=text("active = true"),
            sqlite_where=text("active = 1"),
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    workspace: str = Field(nullable=False, index=True)
    billing_account_id: str = Field(
        nullable=False, foreign_key="billing_accounts.id", index=True
    )
    active: bool = Field(default=True, nullable=False)
    created_at: datetime.datetime = Field(default_factory=naive_utc_now, nullable=False)
    updated_at: datetime.datetime = Field(default_factory=naive_utc_now, nullable=False)


class StripeWebhookEvent(SQLModel, table=True):
    __tablename__ = "stripe_webhook_events"
    __table_args__ = (
        Index(
            "ux_stripe_webhook_events_stripe_event_id",
            "stripe_event_id",
            unique=True,
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    stripe_event_id: str = Field(nullable=False)
    event_type: str = Field(nullable=False)
    created_ts: Optional[datetime.datetime] = Field(default=None)
    processed_at: datetime.datetime = Field(
        default_factory=naive_utc_now, nullable=False
    )
    payload_hash: str = Field(nullable=False)
    billing_account_id: Optional[str] = Field(
        default=None, foreign_key="billing_accounts.id", index=True
    )
    payload: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )


class EventInterceptor(SQLModel, table=True):
    __tablename__ = "event_interceptors"
    __table_args__ = (
        Index(
            "ix_event_interceptors_workspace_enabled_priority",
            "workspace",
            "enabled",
            "priority",
        ),
        Index(
            "ix_event_interceptors_workspace_created_at",
            "workspace",
            "created_at",
        ),
    )

    id: str = Field(
        default_factory=lambda: random_id(readable=False),
        primary_key=True,
    )
    name: str = Field(nullable=False)
    workspace: str = Field(nullable=False, index=True)
    app_id: Optional[str] = Field(default=None, index=True)
    session_id: Optional[str] = Field(default=None, index=True)
    enabled: bool = Field(default=True, nullable=False, index=True)
    priority: int = Field(default=100, nullable=False)
    event_types: Optional[List[str]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    categories: Optional[List[str]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    condition_field: str = Field(nullable=False)
    condition_op: str = Field(nullable=False)
    condition_value: Optional[Any] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    action_type: str = Field(nullable=False)
    action_reason: Optional[str] = Field(default=None)
    action_recovery: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    created_by: str = Field(nullable=False)
    created_at: datetime.datetime = Field(default_factory=naive_utc_now, nullable=False)
    updated_at: datetime.datetime = Field(default_factory=naive_utc_now, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "scope": {
                "workspace": self.workspace,
                "app_id": self.app_id,
                "session_id": self.session_id,
            },
            "enabled": self.enabled,
            "priority": self.priority,
            "event_selector": {
                "event_types": self.event_types or [],
                "categories": self.categories or [],
            },
            "condition": {
                "field": self.condition_field,
                "op": self.condition_op,
                "value": self.condition_value,
            },
            "action": {
                "type": self.action_type,
                "reason": self.action_reason,
                "recovery": self.action_recovery,
            },
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class InterceptorStopError(RuntimeError):
    def __init__(self, code: str, message: str, *, reason: str, interceptor: dict):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.reason = reason
        self.interceptor = interceptor


class BillingContractError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        reason: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.reason = reason
        self.details = details or {}


def validate_key_part(key_part: str):
    """Ensure key parts only contain safe characters."""
    if not _allowed_characters.match(key_part):
        raise ValueError(f"Invalid characters in query part: {key_part}")


class GetServiceConfig(BaseModel):
    mode: Optional[str] = Field(
        None,
        description="Mode for selecting the service. Can be 'random', 'first', 'last', 'exact', 'min_load', 'native:random' (prefer native/local clients), or 'select:criteria:function' format (e.g., 'select:min:get_load', 'select:max:get_cpu_usage')",
    )
    timeout: Optional[float] = Field(
        10.0,
        description="The timeout duration in seconds for fetching the service. This determines how long the function will wait for a service to respond before considering it a timeout.",
    )
    case_conversion: Optional[str] = Field(
        None,
        description="The case conversion for service keys, can be 'camel', 'snake' or None, default is None.",
    )
    select_timeout: Optional[float] = Field(
        2.0,
        description="The timeout duration in seconds for calling service functions when using select mode.",
    )


class WorkspaceStatus:
    READY = "ready"
    LOADING = "loading"
    UNLOADING = "unloading"
    CLOSED = "closed"


class WorkspaceActivityManager:
    """Manages intelligent cleanup of persistent workspaces based on activity tracking."""

    def __init__(
        self,
        activity_tracker,
        redis,
        workspace_manager,
        inactive_period: int = 300,
        check_interval: int = 60,
    ):
        """
        Initialize workspace activity manager.

        Args:
            activity_tracker: The ActivityTracker instance
            redis: Redis connection
            workspace_manager: WorkspaceManager instance (for callbacks)
            inactive_period: Seconds of inactivity before cleanup (default: 5 minutes)
            check_interval: How often to check for cleanup opportunities (default: 1 minute)
        """
        self._tracker = activity_tracker
        self._redis = redis
        self._workspace_manager = workspace_manager
        self._inactive_period = inactive_period
        self._check_interval = check_interval
        self._registrations = {}  # workspace_id -> registration_id
        self._enabled = activity_tracker is not None

        # System workspaces that should never be deleted via activity tracking
        self._protected_workspaces = {
            "public",  # Public shared workspace
            "ws-user-root",  # Root user workspace
            "ws-anonymous",  # Anonymous shared workspace
        }

        if self._enabled:
            logger.info(
                f"WorkspaceActivityManager initialized with {inactive_period}s inactive period"
            )

    def is_enabled(self) -> bool:
        """Check if activity tracking is enabled."""
        return self._enabled

    async def register_for_cleanup(self, workspace_id: str) -> bool:
        """
        Register a persistent workspace for activity-based cleanup.

        Args:
            workspace_id: The workspace ID to track

        Returns:
            True if successfully registered, False otherwise
        """
        if not self._enabled or workspace_id in self._protected_workspaces:
            return False

        try:
            # Create cleanup callback for this workspace
            async def cleanup_callback():
                await self._cleanup_inactive_workspace(workspace_id)

            # Register with activity tracker
            reg_id = self._tracker.register(
                entity_id=workspace_id,
                inactive_period=self._inactive_period,
                on_inactive=cleanup_callback,
                entity_type="workspace",
            )

            # Store registration for tracking
            self._registrations[workspace_id] = reg_id
            logger.debug(
                f"Registered workspace {workspace_id} for activity-based cleanup"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to register workspace {workspace_id} for cleanup: {e}"
            )
            return False

    async def reset_activity(self, workspace_id: str) -> bool:
        """
        Reset activity timer for a workspace when it's accessed.

        Args:
            workspace_id: The workspace ID to reset activity for

        Returns:
            True if successfully reset, False otherwise
        """
        if not self._enabled or workspace_id in self._protected_workspaces:
            return False

        try:
            await self._tracker.reset_timer(workspace_id, "workspace")
            logger.debug(f"Reset activity timer for workspace: {workspace_id}")
            return True
        except Exception as e:
            logger.debug(f"Failed to reset activity timer for {workspace_id}: {e}")
            return False

    async def unregister(self, workspace_id: str) -> bool:
        """
        Unregister a workspace from activity tracking.

        Args:
            workspace_id: The workspace ID to unregister

        Returns:
            True if successfully unregistered, False otherwise
        """
        if not self._enabled:
            return False

        try:
            if workspace_id in self._registrations:
                reg_id = self._registrations.pop(workspace_id)
                self._tracker.unregister(workspace_id, reg_id, "workspace")
                logger.debug(
                    f"Unregistered workspace {workspace_id} from activity tracking"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to unregister workspace {workspace_id}: {e}")

        return False

    async def _cleanup_inactive_workspace(self, workspace_id: str):
        """
        Internal cleanup callback for inactive user workspaces.

        Args:
            workspace_id: The workspace ID to potentially clean up
        """
        try:
            # Double-check workspace still exists and has no active clients
            if await self._redis.hexists("workspaces", workspace_id):
                client_keys = await self._workspace_manager._list_client_keys(
                    workspace_id
                )
                if not client_keys:
                    # Safe to delete - no active clients
                    await self._redis.hdel("workspaces", workspace_id)
                    logger.info(f"Cleaned up inactive workspace: {workspace_id}")
                else:
                    # Workspace has active clients, reset activity timer
                    logger.debug(
                        f"Workspace {workspace_id} still has active clients, resetting timer"
                    )
                    await self.reset_activity(workspace_id)
                    return  # Don't unregister, keep tracking

            # Always clean up registration to prevent memory leak
            if workspace_id in self._registrations:
                del self._registrations[workspace_id]

        except Exception as e:
            logger.error(f"Error cleaning up inactive workspace {workspace_id}: {e}")

    def get_tracked_workspaces(self) -> List[str]:
        """Get list of currently tracked workspace IDs."""
        return list(self._registrations.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get activity manager statistics."""
        return {
            "enabled": self._enabled,
            "inactive_period": self._inactive_period,
            "tracked_workspaces": len(self._registrations),
            "workspace_ids": list(self._registrations.keys()) if self._enabled else [],
        }


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
        enable_s3_for_anonymous_users: bool = False,
        activity_tracker: Optional[Any] = None,
    ):
        self._redis = redis
        self._store = store
        self._initialized = False
        self._rpc = None
        self._service_info = None  # Will be set during setup
        self._root_user = root_user
        self._event_bus = event_bus
        self._server_info = server_info
        self._client_id = client_id
        self._s3_controller = s3_controller
        self._artifact_manager = artifact_manager
        self._server_app_controller = server_app_controller
        self._sql_engine = sql_engine
        self._cache_dir = cache_dir
        self._enable_s3_for_anonymous_users = enable_s3_for_anonymous_users
        if self._sql_engine:
            self.SessionLocal = async_sessionmaker(
                self._sql_engine, expire_on_commit=False, class_=AsyncSession
            )
        else:
            self.SessionLocal = None
        self._enable_service_search = enable_service_search
        # Initialize elegant activity-based workspace cleanup
        self._activity_manager = WorkspaceActivityManager(
            activity_tracker=activity_tracker,
            redis=redis,
            workspace_manager=self,
            inactive_period=300,  # 5 minutes - configurable in future
            check_interval=60,  # 1 minute - configurable in future
        )

    async def _get_sql_session(self):
        """Return an async session for the database."""
        return self.SessionLocal()

    def get_client_id(self):
        assert self._client_id, "client id must not be empty."
        return self._client_id

    async def _ensure_services_collection(self):
        """Ensure the services vector collection exists."""
        if not self._enable_service_search or not self._vector_search:
            return

        try:
            # Check if collection exists by trying to get info about it
            await self._vector_search.count("services")
        except Exception as exp:
            # Collection doesn't exist, create it
            logger.info(
                f"Services vector collection doesn't exist (error: {exp}), creating it..."
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
            logger.info("Services vector collection created successfully")

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
        # Store the service info when registering
        _service_info = await rpc.register_service(
            management_service,
            {"notify": False},
        )
        self._service_info = ServiceInfo.model_validate(_service_info)
        self._service_info.id = "~"
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
            await self._ensure_services_collection()
        self._initialized = True

        return rpc

    def _validate_event_payload(
        self,
        *,
        event_type: str,
        category: str,
        level: str,
        idempotency_key: Optional[str],
    ) -> None:
        if not event_type or " " in event_type:
            raise ValueError("Event type must not contain spaces or be empty")
        if len(event_type) > 128 or not EVENT_TYPE_PATTERN.match(event_type):
            raise ValueError(
                "Invalid event type; use 1..128 chars matching [A-Za-z0-9._:/-]"
            )
        if category not in EVENT_CATEGORIES:
            raise ValueError(
                f"Invalid event category '{category}', expected one of {sorted(EVENT_CATEGORIES)}"
            )
        if level not in EVENT_LEVELS:
            raise ValueError(
                f"Invalid event level '{level}', expected one of {sorted(EVENT_LEVELS)}"
            )
        if idempotency_key is not None and not (1 <= len(idempotency_key) <= 128):
            raise ValueError("idempotency_key must be between 1 and 128 characters")

    async def _persist_event(
        self,
        *,
        event_type: str,
        data: Optional[dict],
        category: str,
        level: str,
        workspace: str,
        user_id: str,
        app_id: Optional[str],
        session_id: Optional[str],
        idempotency_key: Optional[str],
        billing_account_id: Optional[str],
        retention_policy_id: Optional[int],
        timestamp: datetime.datetime,
    ) -> EventLog:
        session = await self._get_sql_session()
        try:
            async with session.begin():
                event_log = EventLog(
                    event_type=event_type,
                    category=category,
                    level=level,
                    workspace=workspace,
                    user_id=user_id,
                    app_id=app_id,
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    billing_account_id=billing_account_id,
                    retention_policy_id=retention_policy_id,
                    timestamp=timestamp,
                    data=data,
                )
                session.add(event_log)
                await session.flush()
                await session.refresh(event_log)
                return event_log
        finally:
            await session.close()

    def _extract_event_field(self, payload: dict, field_path: str):
        value = payload
        for part in field_path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    def _evaluate_interceptor_condition(
        self,
        *,
        op: str,
        actual,
        expected,
    ) -> bool:
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "gt":
            try:
                return actual > expected
            except TypeError:
                return False
        if op == "lt":
            try:
                return actual < expected
            except TypeError:
                return False
        if op == "contains":
            if isinstance(actual, str) and isinstance(expected, str):
                return expected in actual
            if isinstance(actual, (list, tuple, set)):
                return expected in actual
            if isinstance(actual, dict):
                return expected in actual or expected in actual.values()
            return False
        return False

    def _interceptor_matches_event(
        self, interceptor: EventInterceptor, payload: dict
    ) -> bool:
        if interceptor.workspace != payload.get("workspace"):
            return False
        if interceptor.app_id and interceptor.app_id != payload.get("app_id"):
            return False
        if interceptor.session_id and interceptor.session_id != payload.get(
            "session_id"
        ):
            return False
        if (
            interceptor.event_types
            and payload.get("event_type") not in interceptor.event_types
        ):
            return False
        if (
            interceptor.categories
            and payload.get("category") not in interceptor.categories
        ):
            return False
        return True

    async def _emit_recovery_event(
        self,
        *,
        source_event: EventLog,
        interceptor: EventInterceptor,
    ) -> EventLog:
        recovery_config = interceptor.action_recovery or {}
        recovery_event_type = recovery_config.get("event_type", "interceptor.recovered")
        recovery_payload = {
            "source_event_id": source_event.id,
            "source_event_type": source_event.event_type,
            "interceptor_id": interceptor.id,
            "interceptor_name": interceptor.name,
            "reason": interceptor.action_reason,
            "recovery": recovery_config,
        }
        if source_event.data is not None:
            recovery_payload["source_event_data"] = source_event.data
        return await self._persist_event(
            event_type=recovery_event_type,
            data=recovery_payload,
            category="audit",
            level="warning",
            workspace=source_event.workspace,
            user_id=source_event.user_id,
            app_id=source_event.app_id,
            session_id=source_event.session_id,
            idempotency_key=None,
            billing_account_id=source_event.billing_account_id,
            retention_policy_id=source_event.retention_policy_id,
            timestamp=naive_utc_now(),
        )

    async def _evaluate_interceptors(self, event_log: EventLog):
        session = await self._get_sql_session()
        try:
            async with session.begin():
                query = (
                    select(EventInterceptor)
                    .filter(
                        EventInterceptor.workspace == event_log.workspace,
                        EventInterceptor.enabled.is_(True),
                    )
                    .order_by(
                        EventInterceptor.priority.asc(),
                        EventInterceptor.created_at.asc(),
                    )
                )
                result = await session.execute(query)
                interceptors = result.scalars().all()
        finally:
            await session.close()

        event_payload = event_log.to_dict()
        for interceptor in interceptors:
            if not self._interceptor_matches_event(interceptor, event_payload):
                continue
            actual = self._extract_event_field(
                event_payload, interceptor.condition_field
            )
            if not self._evaluate_interceptor_condition(
                op=interceptor.condition_op,
                actual=actual,
                expected=interceptor.condition_value,
            ):
                continue

            if interceptor.action_type == "allow":
                continue

            if interceptor.action_type == "stop":
                reason = interceptor.action_reason or "policy"
                error_code = INTERCEPT_STOP_CODE_BY_REASON.get(
                    reason, "INTERCEPT_STOP_POLICY"
                )
                raise InterceptorStopError(
                    error_code,
                    f"Event blocked by interceptor '{interceptor.name}'",
                    reason=reason,
                    interceptor=interceptor.to_dict(),
                )

            if interceptor.action_type == "recover":
                recovery_event = await self._emit_recovery_event(
                    source_event=event_log,
                    interceptor=interceptor,
                )
                return {
                    "ok": True,
                    "action": "recover",
                    "interceptor_id": interceptor.id,
                    "interceptor_name": interceptor.name,
                    "recovery_event_id": recovery_event.id,
                }
        return None

    @schema_method
    async def register_interceptor(
        self,
        interceptor: dict = Field(..., description="Interceptor configuration"),
        context: Optional[dict] = None,
    ):
        assert context is not None
        workspace = context["ws"]
        user_info = UserInfo.from_context(context)
        if not user_info.check_permission(workspace, UserPermission.admin):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        if not isinstance(interceptor, dict):
            raise ValueError("interceptor must be an object")

        name = interceptor.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValueError(
                "interceptor.name must be a non-empty string with max 128 characters"
            )

        scope = interceptor.get("scope") or {}
        if not isinstance(scope, dict):
            raise ValueError("interceptor.scope must be an object")
        scope_workspace = scope.get("workspace") or workspace
        if scope_workspace != workspace:
            raise PermissionError(
                "interceptor scope.workspace must match current workspace"
            )
        app_id = scope.get("app_id")
        session_id = scope.get("session_id")

        event_selector = interceptor.get("event_selector") or {}
        if not isinstance(event_selector, dict):
            raise ValueError("interceptor.event_selector must be an object")
        event_types = event_selector.get("event_types") or []
        categories = event_selector.get("categories") or []
        if not isinstance(event_types, list) or not all(
            isinstance(i, str) for i in event_types
        ):
            raise ValueError(
                "interceptor.event_selector.event_types must be a string array"
            )
        if not isinstance(categories, list) or not all(
            isinstance(i, str) for i in categories
        ):
            raise ValueError(
                "interceptor.event_selector.categories must be a string array"
            )
        invalid_categories = [
            item for item in categories if item not in EVENT_CATEGORIES
        ]
        if invalid_categories:
            raise ValueError(
                f"Invalid categories in interceptor selector: {sorted(set(invalid_categories))}"
            )

        condition = interceptor.get("condition") or {}
        if not isinstance(condition, dict):
            raise ValueError("interceptor.condition must be an object")
        condition_field = condition.get("field")
        condition_op = condition.get("op")
        condition_value = condition.get("value")
        if not isinstance(condition_field, str) or not condition_field.strip():
            raise ValueError("interceptor.condition.field must be a non-empty string")
        if condition_op not in INTERCEPTOR_CONDITION_OPS:
            raise ValueError(
                f"interceptor.condition.op must be one of {sorted(INTERCEPTOR_CONDITION_OPS)}"
            )

        action = interceptor.get("action") or {}
        if not isinstance(action, dict):
            raise ValueError("interceptor.action must be an object")
        action_type = action.get("type")
        if action_type not in INTERCEPTOR_ACTION_TYPES:
            raise ValueError(
                f"interceptor.action.type must be one of {sorted(INTERCEPTOR_ACTION_TYPES)}"
            )
        action_reason = action.get("reason") or "policy"
        if action_reason not in INTERCEPTOR_STOP_REASONS:
            raise ValueError(
                f"interceptor.action.reason must be one of {sorted(INTERCEPTOR_STOP_REASONS)}"
            )
        action_recovery = action.get("recovery")
        if (
            action_type == "recover"
            and action_recovery is not None
            and not isinstance(action_recovery, dict)
        ):
            raise ValueError(
                "interceptor.action.recovery must be an object when provided"
            )

        enabled = bool(interceptor.get("enabled", True))
        priority = interceptor.get("priority", 100)
        if not isinstance(priority, int):
            raise ValueError("interceptor.priority must be an integer")

        db_session = await self._get_sql_session()
        try:
            async with db_session.begin():
                db_item = EventInterceptor(
                    name=name.strip(),
                    workspace=workspace,
                    app_id=app_id,
                    session_id=session_id,
                    enabled=enabled,
                    priority=priority,
                    event_types=event_types or [],
                    categories=categories or [],
                    condition_field=condition_field,
                    condition_op=condition_op,
                    condition_value=condition_value,
                    action_type=action_type,
                    action_reason=action_reason,
                    action_recovery=(
                        action_recovery if action_type == "recover" else None
                    ),
                    created_by=user_info.id,
                    created_at=naive_utc_now(),
                    updated_at=naive_utc_now(),
                )
                db_session.add(db_item)
                await db_session.flush()
                await db_session.refresh(db_item)
                return db_item.to_dict()
        finally:
            await db_session.close()

    @schema_method
    async def list_interceptors(
        self,
        filters: Optional[dict] = Field(
            None, description="Optional interceptor filters"
        ),
        context: Optional[dict] = None,
    ):
        assert context is not None
        workspace = context["ws"]
        user_info = UserInfo.from_context(context)
        if not user_info.check_permission(workspace, UserPermission.admin):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        filters = filters or {}
        app_id = filters.get("app_id")
        session_id = filters.get("session_id")
        enabled = filters.get("enabled")

        db_session = await self._get_sql_session()
        try:
            async with db_session.begin():
                query = select(EventInterceptor).filter(
                    EventInterceptor.workspace == workspace
                )
                if app_id is not None:
                    query = query.filter(EventInterceptor.app_id == app_id)
                if session_id is not None:
                    query = query.filter(EventInterceptor.session_id == session_id)
                if enabled is not None:
                    query = query.filter(EventInterceptor.enabled.is_(bool(enabled)))
                query = query.order_by(
                    EventInterceptor.priority.asc(),
                    EventInterceptor.created_at.asc(),
                )
                result = await db_session.execute(query)
                return [item.to_dict() for item in result.scalars().all()]
        finally:
            await db_session.close()

    @schema_method
    async def remove_interceptor(
        self,
        interceptor_id: str = Field(..., description="Interceptor ID"),
        context: Optional[dict] = None,
    ):
        assert context is not None
        workspace = context["ws"]
        user_info = UserInfo.from_context(context)
        if not user_info.check_permission(workspace, UserPermission.admin):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        db_session = await self._get_sql_session()
        try:
            async with db_session.begin():
                query = select(EventInterceptor).filter(
                    EventInterceptor.id == interceptor_id,
                    EventInterceptor.workspace == workspace,
                )
                result = await db_session.execute(query)
                interceptor = result.scalars().first()
                if interceptor is None:
                    raise KeyError(f"Interceptor not found: {interceptor_id}")
                await db_session.delete(interceptor)
                return {"ok": True, "removed": True}
        finally:
            await db_session.close()

    def _parse_rfc3339_datetime(
        self,
        value: Optional[Union[str, datetime.datetime]],
        *,
        field_name: str,
        required: bool = False,
    ) -> Optional[datetime.datetime]:
        if value is None:
            if required:
                raise BillingContractError(
                    "INVALID_ARGUMENT",
                    f"{field_name} is required",
                )
            return None
        if isinstance(value, datetime.datetime):
            return normalize_naive_utc_datetime(value)
        if not isinstance(value, str):
            raise BillingContractError(
                "INVALID_ARGUMENT",
                f"{field_name} must be an RFC3339 string",
            )
        try:
            normalized = value.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            parsed = datetime.datetime.fromisoformat(normalized)
            return normalize_naive_utc_datetime(parsed)
        except Exception as exp:
            raise BillingContractError(
                "INVALID_ARGUMENT",
                f"{field_name} must be an RFC3339 datetime string",
                details={"value": value},
            ) from exp

    @staticmethod
    def _coerce_positive_number(value, *, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BillingContractError(
                "INVALID_ARGUMENT",
                f"{field_name} must be a number",
            )
        normalized = float(value)
        if normalized <= 0:
            raise BillingContractError(
                "INVALID_ARGUMENT",
                f"{field_name} must be strictly positive",
            )
        return normalized

    def _normalize_usage_payload(
        self,
        usage: dict,
        *,
        workspace: str,
        app_id: Optional[str],
        session_id: Optional[str],
        resolved_billing_account_id: Optional[str],
    ) -> tuple[dict, str]:
        canonical_payload = {
            "workspace": workspace,
            "billing_point": usage["billing_point"],
            "amount": float(usage["amount"]),
            "unit": usage["unit"],
            "idempotency_key": usage["idempotency_key"],
            "dimensions": usage.get("dimensions") or {},
            "occurred_at": usage.get("occurred_at"),
            "app_id": app_id,
            "session_id": session_id,
            "billing_account_id": resolved_billing_account_id,
            "subscription_id": usage.get("subscription_id"),
            "plan_id": usage.get("plan_id"),
            "price_id": usage.get("price_id"),
            "entitlement": usage.get("entitlement") or {},
        }
        serialized = json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":")
        )
        fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return canonical_payload, fingerprint

    async def _resolve_billing_account(
        self,
        db_session: AsyncSession,
        *,
        workspace: str,
        billing_account_id: Optional[str],
    ) -> tuple[Optional[str], Optional[int], Optional[BillingAccount]]:
        if billing_account_id:
            result = await db_session.execute(
                select(BillingAccount).filter(BillingAccount.id == billing_account_id)
            )
            account = result.scalars().first()
            if account is None:
                raise BillingContractError(
                    "NOT_FOUND",
                    f"billing_account_id not found: {billing_account_id}",
                )
            return account.id, account.retention_policy_id, account

        mapping_result = await db_session.execute(
            select(WorkspaceBillingAccount).filter(
                WorkspaceBillingAccount.workspace == workspace,
                WorkspaceBillingAccount.active.is_(True),
            )
        )
        mapping = mapping_result.scalars().first()
        if mapping is None:
            return None, None, None

        account_result = await db_session.execute(
            select(BillingAccount).filter(
                BillingAccount.id == mapping.billing_account_id
            )
        )
        account = account_result.scalars().first()
        if account is None:
            return None, None, None
        return account.id, account.retention_policy_id, account

    async def _sum_usage_amount(
        self,
        db_session: AsyncSession,
        *,
        workspace: str,
        billing_account_id: Optional[str],
        entitlement_key: str,
    ) -> float:
        query = select(EventLog).filter(
            EventLog.workspace == workspace,
            EventLog.category == "billing",
            EventLog.event_type == BILLING_USAGE_EVENT_TYPE,
        )
        if billing_account_id:
            query = query.filter(EventLog.billing_account_id == billing_account_id)
        result = await db_session.execute(query)
        events = result.scalars().all()

        used = 0.0
        for event in events:
            payload = event.data or {}
            if not isinstance(payload, dict):
                continue
            billing_point = payload.get("billing_point")
            entitlement = payload.get("entitlement") or {}
            if not isinstance(entitlement, dict):
                entitlement = {}
            row_key = entitlement.get("key") or billing_point
            if row_key != entitlement_key:
                continue
            amount = payload.get("amount")
            if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                used += float(amount)
        return used

    async def _upsert_billing_account_for_customer(
        self,
        db_session: AsyncSession,
        stripe_customer_id: str,
    ) -> BillingAccount:
        result = await db_session.execute(
            select(BillingAccount).filter(
                BillingAccount.stripe_customer_id == stripe_customer_id
            )
        )
        account = result.scalars().first()
        if account is not None:
            return account
        account = BillingAccount(
            stripe_customer_id=stripe_customer_id,
            created_at=naive_utc_now(),
            updated_at=naive_utc_now(),
        )
        db_session.add(account)
        await db_session.flush()
        await db_session.refresh(account)
        return account

    async def process_stripe_webhook_event(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise BillingContractError(
                "INVALID_ARGUMENT", "Webhook payload must be an object"
            )
        event_id = payload.get("id")
        event_type = payload.get("type")
        if not isinstance(event_id, str) or not event_id:
            raise BillingContractError(
                "INVALID_ARGUMENT", "Stripe event payload missing id"
            )
        if not isinstance(event_type, str) or not event_type:
            raise BillingContractError(
                "INVALID_ARGUMENT", "Stripe event payload missing type"
            )

        payload_raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(payload_raw.encode("utf-8")).hexdigest()
        event_created_raw = payload.get("created")
        event_created = None
        if isinstance(event_created_raw, (int, float)) and not isinstance(
            event_created_raw, bool
        ):
            event_created = normalize_naive_utc_datetime(
                datetime.datetime.fromtimestamp(
                    float(event_created_raw), tz=datetime.timezone.utc
                )
            )

        data_object = payload.get("data", {}).get("object", {})
        if not isinstance(data_object, dict):
            data_object = {}
        stripe_customer_id = data_object.get("customer")
        session = await self._get_sql_session()
        request_id = random_id(readable=False)
        try:
            async with session.begin():
                existing_result = await session.execute(
                    select(StripeWebhookEvent).filter(
                        StripeWebhookEvent.stripe_event_id == event_id
                    )
                )
                existing = existing_result.scalars().first()
                if existing is not None:
                    return {
                        "ok": True,
                        "deduped": True,
                        "event_id": event_id,
                        "event_type": existing.event_type,
                        "request_id": request_id,
                    }

                billing_account = None
                if isinstance(stripe_customer_id, str) and stripe_customer_id:
                    billing_account = await self._upsert_billing_account_for_customer(
                        session, stripe_customer_id
                    )

                snapshot = {}
                if billing_account is not None:
                    snapshot = dict(billing_account.stripe_snapshot or {})
                    last_webhook_ts = billing_account.last_webhook_event_ts
                    is_newer = (
                        event_created is None
                        or last_webhook_ts is None
                        or event_created >= last_webhook_ts
                    )
                    if is_newer:
                        snapshot["last_event"] = {
                            "id": event_id,
                            "type": event_type,
                            "created": (
                                event_created.isoformat() if event_created else None
                            ),
                        }
                        if event_type.startswith("customer.subscription."):
                            price_id = None
                            items = data_object.get("items", {}).get("data", [])
                            if isinstance(items, list) and items:
                                first_item = (
                                    items[0] if isinstance(items[0], dict) else {}
                                )
                                price = first_item.get("price", {})
                                if isinstance(price, dict):
                                    price_id = price.get("id")
                            billing_account.stripe_subscription_id = data_object.get(
                                "id"
                            )
                            billing_account.stripe_subscription_status = (
                                data_object.get("status")
                            )
                            billing_account.current_period_start = (
                                normalize_naive_utc_datetime(
                                    datetime.datetime.fromtimestamp(
                                        float(data_object.get("current_period_start")),
                                        tz=datetime.timezone.utc,
                                    )
                                )
                                if isinstance(
                                    data_object.get("current_period_start"),
                                    (int, float),
                                )
                                else billing_account.current_period_start
                            )
                            billing_account.current_period_end = (
                                normalize_naive_utc_datetime(
                                    datetime.datetime.fromtimestamp(
                                        float(data_object.get("current_period_end")),
                                        tz=datetime.timezone.utc,
                                    )
                                )
                                if isinstance(
                                    data_object.get("current_period_end"), (int, float)
                                )
                                else billing_account.current_period_end
                            )
                            if isinstance(price_id, str) and price_id:
                                billing_account.price_id = price_id
                            snapshot["subscription"] = {
                                "id": data_object.get("id"),
                                "status": data_object.get("status"),
                                "customer": stripe_customer_id,
                                "price_id": price_id,
                            }
                        elif event_type.startswith("invoice."):
                            snapshot["invoice"] = {
                                "id": data_object.get("id"),
                                "status": data_object.get("status"),
                                "subscription": data_object.get("subscription"),
                                "total": data_object.get("total"),
                                "currency": data_object.get("currency"),
                            }
                        elif event_type.startswith("payment_intent."):
                            snapshot["payment_intent"] = {
                                "id": data_object.get("id"),
                                "status": data_object.get("status"),
                                "amount": data_object.get("amount"),
                                "currency": data_object.get("currency"),
                            }
                        else:
                            snapshot["event_type"] = event_type
                        billing_account.last_webhook_event_id = event_id
                        if event_created is not None:
                            billing_account.last_webhook_event_ts = event_created
                        billing_account.stripe_snapshot = snapshot
                        billing_account.updated_at = naive_utc_now()

                webhook_event = StripeWebhookEvent(
                    stripe_event_id=event_id,
                    event_type=event_type,
                    created_ts=event_created,
                    processed_at=naive_utc_now(),
                    payload_hash=payload_hash,
                    billing_account_id=billing_account.id if billing_account else None,
                    payload=payload,
                )
                session.add(webhook_event)
                await session.flush()

                event_log = EventLog(
                    event_type=f"stripe.webhook.{event_type}",
                    category="billing",
                    level="info",
                    workspace="public",
                    user_id="stripe-webhook",
                    app_id=None,
                    session_id=None,
                    idempotency_key=f"stripe:{event_id}",
                    billing_account_id=billing_account.id if billing_account else None,
                    retention_policy_id=(
                        billing_account.retention_policy_id if billing_account else None
                    ),
                    timestamp=event_created or naive_utc_now(),
                    data={
                        "stripe_event_id": event_id,
                        "stripe_event_type": event_type,
                        "stripe_customer_id": stripe_customer_id,
                        "deduped": False,
                    },
                )
                session.add(event_log)
                await session.flush()

                return {
                    "ok": True,
                    "deduped": False,
                    "event_id": event_id,
                    "event_type": event_type,
                    "billing_account_id": (
                        billing_account.id if billing_account else None
                    ),
                    "snapshot": snapshot,
                    "request_id": request_id,
                }
        except IntegrityError:
            await session.rollback()
            existing_result = await session.execute(
                select(StripeWebhookEvent).filter(
                    StripeWebhookEvent.stripe_event_id == event_id
                )
            )
            existing = existing_result.scalars().first()
            if existing is not None:
                return {
                    "ok": True,
                    "deduped": True,
                    "event_id": event_id,
                    "event_type": existing.event_type,
                    "request_id": request_id,
                }
            raise
        finally:
            await session.close()

    @schema_method
    async def log_event(
        self,
        event_type: str = Field(..., description="Event type"),
        data: Optional[dict] = Field(None, description="Additional event data"),
        category: str = Field(
            "application",
            description="Event category: system|application|billing|audit",
        ),
        level: str = Field(
            "info",
            description="Event level: debug|info|warning|error",
        ),
        app_id: Optional[str] = Field(None, description="Application ID"),
        session_id: Optional[str] = Field(None, description="Session ID"),
        idempotency_key: Optional[str] = Field(None, description="Idempotency key"),
        billing_account_id: Optional[str] = Field(
            None, description="Billing account ID for usage-linked events"
        ),
        retention_policy_id: Optional[int] = Field(
            None, description="Retention policy ID applied to this event"
        ),
        timestamp: Optional[datetime.datetime] = Field(
            None, description="Explicit event timestamp (defaults to now)"
        ),
        context: dict = None,
    ):
        """Log a new event and run interception pipeline."""
        self._validate_event_payload(
            event_type=event_type,
            category=category,
            level=level,
            idempotency_key=idempotency_key,
        )
        self.validate_context(context, permission=UserPermission.read_write)
        workspace = context["ws"]
        user_info = UserInfo.from_context(context)
        event_timestamp = normalize_naive_utc_datetime(timestamp) or naive_utc_now()

        try:
            event_log = await self._persist_event(
                event_type=event_type,
                data=data,
                category=category,
                level=level,
                workspace=workspace,
                user_id=user_info.id,
                app_id=app_id,
                session_id=session_id,
                idempotency_key=idempotency_key,
                billing_account_id=billing_account_id,
                retention_policy_id=retention_policy_id,
                timestamp=event_timestamp,
            )
            logger.info(f"Logged event: {event_type} by {user_info.id} in {workspace}")

            pipeline_result = await self._evaluate_interceptors(event_log)
            if pipeline_result is not None:
                return pipeline_result
        except InterceptorStopError as exp:
            logger.warning(f"Event stopped by interceptor: {exp}")
            raise
        except Exception as e:
            logger.error(f"Failed to log event: {event_type}, {e}")
            raise

    @schema_method
    async def record_usage(
        self,
        usage: dict = Field(..., description="UsageRecordV1 payload"),
        context: Optional[dict] = None,
    ):
        self.validate_context(context, permission=UserPermission.read_write)
        if not isinstance(usage, dict):
            raise BillingContractError("INVALID_ARGUMENT", "usage must be an object")

        billing_point = usage.get("billing_point")
        if not isinstance(billing_point, str) or not billing_point.strip():
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "billing_point is required and must be a non-empty string",
            )
        billing_point = billing_point.strip()
        if len(billing_point) > 128 or not EVENT_TYPE_PATTERN.match(billing_point):
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "billing_point must match [A-Za-z0-9._:/-] and be <= 128 chars",
            )

        amount = self._coerce_positive_number(usage.get("amount"), field_name="amount")

        unit = usage.get("unit")
        if not isinstance(unit, str) or not unit.strip() or len(unit.strip()) > 32:
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "unit is required and must be a non-empty string with <= 32 chars",
            )
        unit = unit.strip()

        idempotency_key = usage.get("idempotency_key")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "idempotency_key is required for record_usage",
            )
        idempotency_key = idempotency_key.strip()
        if len(idempotency_key) > 128:
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "idempotency_key must be <= 128 chars",
            )

        dimensions = usage.get("dimensions") or {}
        if not isinstance(dimensions, dict):
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "dimensions must be an object when provided",
            )
        for key, value in dimensions.items():
            if not isinstance(key, str) or not key:
                raise BillingContractError(
                    "INVALID_ARGUMENT",
                    "dimensions keys must be non-empty strings",
                )
            if isinstance(value, (dict, list)):
                raise BillingContractError(
                    "INVALID_ARGUMENT",
                    "dimensions values must be scalar (string/number/boolean)",
                )

        workspace = context["ws"]
        user_info = UserInfo.from_context(context)
        app_id = usage.get("app_id")
        session_id = usage.get("session_id")
        occurred_at = self._parse_rfc3339_datetime(
            usage.get("occurred_at"),
            field_name="occurred_at",
            required=False,
        )

        entitlement = usage.get("entitlement") or {}
        if not isinstance(entitlement, dict):
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "entitlement must be an object when provided",
            )
        entitlement_key = entitlement.get("key") or billing_point
        entitlement_limit = entitlement.get("limit")
        if entitlement_limit is not None:
            entitlement_limit = self._coerce_positive_number(
                entitlement_limit,
                field_name="entitlement.limit",
            )

        request_id = random_id(readable=False)
        fingerprint = None
        session = await self._get_sql_session()
        try:
            async with session.begin():
                resolved_billing_account_id, retention_policy_id, account = (
                    await self._resolve_billing_account(
                        session,
                        workspace=workspace,
                        billing_account_id=usage.get("billing_account_id"),
                    )
                )

                usage_payload = {
                    "billing_point": billing_point,
                    "amount": amount,
                    "unit": unit,
                    "idempotency_key": idempotency_key,
                    "dimensions": dimensions,
                    "occurred_at": occurred_at.isoformat() if occurred_at else None,
                    "subscription_id": usage.get("subscription_id")
                    or (account.stripe_subscription_id if account else None),
                    "plan_id": usage.get("plan_id")
                    or (account.plan_id if account else None),
                    "price_id": usage.get("price_id")
                    or (account.price_id if account else None),
                    "entitlement": {
                        "key": entitlement_key,
                        "limit": entitlement_limit,
                    },
                }
                canonical_payload, fingerprint = self._normalize_usage_payload(
                    usage_payload,
                    workspace=workspace,
                    app_id=app_id,
                    session_id=session_id,
                    resolved_billing_account_id=resolved_billing_account_id,
                )

                query = select(EventLog).filter(
                    EventLog.workspace == workspace,
                    EventLog.category == "billing",
                    EventLog.idempotency_key == idempotency_key,
                )
                existing_result = await session.execute(query)
                existing_event = existing_result.scalars().first()
                if existing_event is not None:
                    existing_payload = existing_event.data or {}
                    existing_fingerprint = None
                    if isinstance(existing_payload, dict):
                        existing_fingerprint = existing_payload.get(
                            "idempotency_fingerprint"
                        )
                    if existing_fingerprint == fingerprint:
                        return {
                            "ok": True,
                            "deduped": True,
                            "usage_event_id": existing_event.id,
                            "request_id": request_id,
                        }
                    raise BillingContractError(
                        "IDEMPOTENCY_CONFLICT",
                        "idempotency_key already exists with different usage payload",
                    )

                if entitlement_limit is not None:
                    consumed = await self._sum_usage_amount(
                        session,
                        workspace=workspace,
                        billing_account_id=resolved_billing_account_id,
                        entitlement_key=entitlement_key,
                    )
                    if consumed + amount > entitlement_limit:
                        raise BillingContractError(
                            "TOO_MANY_REQUESTS",
                            "Usage limit exhausted",
                            reason="limit",
                            details={
                                "entitlement_key": entitlement_key,
                                "limit": entitlement_limit,
                                "consumed": consumed,
                                "requested": amount,
                            },
                        )
                    usage_payload["entitlement"]["consumed_before"] = consumed
                    usage_payload["entitlement"]["consumed_after"] = consumed + amount

                event_log = EventLog(
                    event_type=BILLING_USAGE_EVENT_TYPE,
                    data={
                        **usage_payload,
                        "workspace": workspace,
                        "idempotency_payload": canonical_payload,
                        "idempotency_fingerprint": fingerprint,
                    },
                    category="billing",
                    level="info",
                    workspace=workspace,
                    user_id=user_info.id,
                    app_id=app_id,
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                    billing_account_id=resolved_billing_account_id,
                    retention_policy_id=retention_policy_id,
                    timestamp=occurred_at or naive_utc_now(),
                )
                session.add(event_log)
                await session.flush()
                await session.refresh(event_log)
                return {
                    "ok": True,
                    "deduped": False,
                    "usage_event_id": event_log.id,
                    "request_id": request_id,
                }
        except IntegrityError:
            await session.rollback()
            async with session.begin():
                existing_result = await session.execute(
                    select(EventLog).filter(
                        EventLog.workspace == workspace,
                        EventLog.category == "billing",
                        EventLog.idempotency_key == idempotency_key,
                    )
                )
                existing_event = existing_result.scalars().first()
                if existing_event is not None:
                    existing_payload = (
                        existing_event.data
                        if isinstance(existing_event.data, dict)
                        else {}
                    )
                    existing_fingerprint = existing_payload.get(
                        "idempotency_fingerprint"
                    )
                    if fingerprint is not None and existing_fingerprint == fingerprint:
                        return {
                            "ok": True,
                            "deduped": True,
                            "usage_event_id": existing_event.id,
                            "request_id": request_id,
                        }
                    raise BillingContractError(
                        "IDEMPOTENCY_CONFLICT",
                        "idempotency_key already exists with different usage payload",
                    )
            raise
        finally:
            await session.close()

    @schema_method
    async def get_usage_summary(
        self,
        query: dict = Field(..., description="Usage summary query"),
        context: Optional[dict] = None,
    ):
        self.validate_context(context, permission=UserPermission.read)
        if not isinstance(query, dict):
            raise BillingContractError("INVALID_ARGUMENT", "query must be an object")

        start_time = self._parse_rfc3339_datetime(
            query.get("start_time"),
            field_name="start_time",
            required=True,
        )
        end_time = self._parse_rfc3339_datetime(
            query.get("end_time"),
            field_name="end_time",
            required=True,
        )
        if start_time is not None and end_time is not None and start_time > end_time:
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "start_time must be <= end_time",
            )

        group_by = query.get("group_by") or []
        if not isinstance(group_by, list) or not all(
            isinstance(item, str) for item in group_by
        ):
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "group_by must be an array of strings",
            )
        invalid_group_by = [
            item for item in group_by if item not in ALLOWED_USAGE_SUMMARY_GROUP_BY
        ]
        if invalid_group_by:
            raise BillingContractError(
                "INVALID_ARGUMENT",
                f"Invalid group_by fields: {sorted(set(invalid_group_by))}",
            )

        workspace = context["ws"]
        query_workspace = query.get("workspace")
        if query_workspace and query_workspace != workspace:
            raise BillingContractError(
                "FORBIDDEN",
                "workspace in query must match current workspace context",
            )

        filters = query.get("filters") or {}
        if not isinstance(filters, dict):
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "filters must be an object",
            )
        dimension_filters = filters.get("dimensions") or {}
        if not isinstance(dimension_filters, dict):
            raise BillingContractError(
                "INVALID_ARGUMENT",
                "filters.dimensions must be an object",
            )

        billing_account_id = query.get("billing_account_id")
        session = await self._get_sql_session()
        try:
            async with session.begin():
                sql_query = select(EventLog).filter(
                    EventLog.workspace == workspace,
                    EventLog.category == "billing",
                    EventLog.event_type == BILLING_USAGE_EVENT_TYPE,
                    EventLog.timestamp >= start_time,
                    EventLog.timestamp <= end_time,
                )
                if billing_account_id:
                    sql_query = sql_query.filter(
                        EventLog.billing_account_id == billing_account_id
                    )
                if query.get("app_id"):
                    sql_query = sql_query.filter(EventLog.app_id == query.get("app_id"))
                if query.get("session_id"):
                    sql_query = sql_query.filter(
                        EventLog.session_id == query.get("session_id")
                    )
                result = await session.execute(sql_query)
                events = result.scalars().all()

            grouped_rows = {}
            total_amount = 0.0
            total_count = 0
            query_billing_point = query.get("billing_point")
            query_unit = query.get("unit")

            for event in events:
                payload = event.data or {}
                if not isinstance(payload, dict):
                    continue
                amount = payload.get("amount")
                if not isinstance(amount, (int, float)) or isinstance(amount, bool):
                    continue
                amount = float(amount)
                billing_point = payload.get("billing_point")
                unit = payload.get("unit")
                if query_billing_point and query_billing_point != billing_point:
                    continue
                if query_unit and query_unit != unit:
                    continue

                dimensions = payload.get("dimensions") or {}
                if not isinstance(dimensions, dict):
                    dimensions = {}
                if any(
                    dimensions.get(key) != value
                    for key, value in dimension_filters.items()
                ):
                    continue

                row_values = {}
                for field in group_by:
                    if field == "workspace":
                        row_values[field] = event.workspace
                    elif field == "app_id":
                        row_values[field] = event.app_id
                    elif field == "billing_point":
                        row_values[field] = billing_point
                    elif field == "unit":
                        row_values[field] = unit

                group_key = tuple((field, row_values.get(field)) for field in group_by)
                row = grouped_rows.get(group_key)
                if row is None:
                    row = {**row_values, "amount": 0.0, "total_amount": 0.0, "count": 0}
                    grouped_rows[group_key] = row

                row["amount"] += amount
                row["total_amount"] = row["amount"]
                row["count"] += 1
                total_amount += amount
                total_count += 1

            rows = list(grouped_rows.values())
            rows.sort(
                key=lambda row: tuple(
                    "" if row.get(field) is None else str(row.get(field))
                    for field in group_by
                )
            )
            return {
                "ok": True,
                "rows": rows,
                "totals": {
                    "amount": total_amount,
                    "total_amount": total_amount,
                    "count": total_count,
                },
                "request_id": random_id(readable=False),
            }
        finally:
            await session.close()

    @schema_method
    async def get_event_stats(
        self,
        event_type: Optional[str] = Field(None, description="Event type"),
        category: Optional[str] = Field(None, description="Event category"),
        level: Optional[str] = Field(None, description="Event level"),
        app_id: Optional[str] = Field(None, description="Application ID"),
        session_id: Optional[str] = Field(None, description="Session ID"),
        idempotency_key: Optional[str] = Field(None, description="Idempotency key"),
        billing_account_id: Optional[str] = Field(
            None, description="Billing account ID"
        ),
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
        user_info = UserInfo.from_context(context)
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
                if category:
                    query = query.filter(EventLog.category == category)
                if level:
                    query = query.filter(EventLog.level == level)
                if app_id:
                    query = query.filter(EventLog.app_id == app_id)
                if session_id:
                    query = query.filter(EventLog.session_id == session_id)
                if idempotency_key:
                    query = query.filter(EventLog.idempotency_key == idempotency_key)
                if billing_account_id:
                    query = query.filter(
                        EventLog.billing_account_id == billing_account_id
                    )
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
        category: Optional[str] = Field(None, description="Event category"),
        level: Optional[str] = Field(None, description="Event level"),
        app_id: Optional[str] = Field(None, description="Application ID"),
        session_id: Optional[str] = Field(None, description="Session ID"),
        idempotency_key: Optional[str] = Field(None, description="Idempotency key"),
        billing_account_id: Optional[str] = Field(
            None, description="Billing account ID"
        ),
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
        user_info = UserInfo.from_context(context)
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
                if category:
                    query = query.filter(EventLog.category == category)
                if level:
                    query = query.filter(EventLog.level == level)
                if app_id:
                    query = query.filter(EventLog.app_id == app_id)
                if session_id:
                    query = query.filter(EventLog.session_id == session_id)
                if idempotency_key:
                    query = query.filter(EventLog.idempotency_key == idempotency_key)
                if billing_account_id:
                    query = query.filter(
                        EventLog.billing_account_id == billing_account_id
                    )
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
    async def set_env(
        self,
        key: str = Field(..., description="Environment variable key"),
        value: str = Field(..., description="Environment variable value"),
        context: Optional[dict] = None,
    ):
        """Set an environment variable for the workspace."""
        assert context is not None
        ws = context["ws"]
        user_info = UserInfo.from_context(context)
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {ws}")

        workspace_info = await self.load_workspace_info(ws)
        workspace_info.config = workspace_info.config or {}
        if "environs" not in workspace_info.config:
            workspace_info.config["environs"] = {}

        if value is None:
            if key in workspace_info.config["environs"]:
                del workspace_info.config["environs"][key]
                await self.log_event("env_removed", {"key": key}, context=context)
                logger.info(f"Environment variable '{key}' removed from workspace {ws}")
        else:
            workspace_info.config["environs"][key] = value
            await self.log_event("env_set", {"key": key}, context=context)
            logger.info(f"Environment variable '{key}' set in workspace {ws}")

        await self._update_workspace(workspace_info, user_info)

    @schema_method
    def _validate_event_subscription(self, event_type: str, workspace: str):
        """Validate that clients can only subscribe to workspace-safe event types."""
        # Allow system events and custom events within own workspace
        allowed_system_events = [
            "service_added",
            "service_removed",
            "service_updated",
            "client_connected",
            "client_disconnected",
            "client_updated",
            "workspace_loaded",
            "workspace_deleted",
            "workspace_changed",
            "workspace_ready",
            "workspace_unloaded",
            "workspace_status_changed",
        ]

        # If it's a system event, it's allowed
        if event_type in allowed_system_events:
            return True

        # SECURITY: Use exact workspace matching instead of suffix-based matching
        # to prevent cross-workspace event snooping via workspace names with similar suffixes

        # Check if event contains workspace-like patterns
        # Pattern: events that start with "ws-" or contain "/" (workspace/event format)
        if "/" in event_type:
            # Extract the workspace part from "workspace/event" format
            event_workspace = event_type.split("/")[0]
            # SECURITY: Exact match only - no suffix matching
            if event_workspace != workspace:
                raise ValueError(
                    f"Subscription to event '{event_type}' is forbidden. "
                    f"Event workspace '{event_workspace}' does not match your workspace '{workspace}'. "
                    f"Clients can only subscribe to events within their own workspace or system events."
                )
            return True

        # Check for workspace identifier patterns that could indicate cross-workspace targeting
        if event_type.startswith("ws-"):
            # SECURITY: Exact match only for workspace-prefixed events
            if not event_type.startswith(f"{workspace}/") and event_type != workspace:
                # This looks like a workspace identifier but doesn't match current workspace
                raise ValueError(
                    f"Subscription to event '{event_type}' is forbidden as it contains "
                    f"workspace identifiers. Clients can only subscribe to events "
                    f"within their own workspace ({workspace}) or system events."
                )

        # Check for colon-based namespace separators that could indicate workspace targeting
        if ":" in event_type:
            # Allow "workspace:event" format only for current workspace
            parts = event_type.split(":")
            if len(parts) >= 2 and parts[0].startswith("ws-"):
                if parts[0] != workspace:
                    raise ValueError(
                        f"Subscription to event '{event_type}' is forbidden. "
                        f"Workspace '{parts[0]}' does not match your workspace '{workspace}'."
                    )

        return True

    @schema_method
    async def subscribe(
        self,
        event_type: Union[str, List[str]] = Field(
            ..., description="Event type(s) to subscribe to"
        ),
        context: Optional[dict] = None,
    ):
        """Subscribe to event types for receiving broadcast messages."""
        assert context is not None
        workspace = context["ws"]
        client_id = context["from"].split("/")[-1]  # Extract client_id from full path

        # Get the connection for this client
        connection = RedisRPCConnection.get_connection(workspace, client_id)
        if not connection:
            raise RuntimeError(
                f"Connection not found for client {workspace}/{client_id}"
            )

        # Handle both single event type and list of event types
        event_types = event_type if isinstance(event_type, list) else [event_type]

        # Validate all event types first
        for evt in event_types:
            self._validate_event_subscription(evt, workspace)

        # If validation passes, subscribe to all events
        for evt in event_types:
            connection.subscribe(evt)

        logger.info(f"Client {workspace}/{client_id} subscribed to: {event_type}")
        return True  # Explicitly return success to ensure RPC response is sent

    @schema_method
    async def unsubscribe(
        self,
        event_type: Union[str, List[str]] = Field(
            ..., description="Event type(s) to unsubscribe from"
        ),
        context: Optional[dict] = None,
    ):
        """Unsubscribe from event types to stop receiving broadcast messages."""
        assert context is not None
        workspace = context["ws"]
        client_id = context["from"].split("/")[-1]  # Extract client_id from full path

        # Get the connection for this client
        connection = RedisRPCConnection.get_connection(workspace, client_id)
        if not connection:
            raise RuntimeError(
                f"Connection not found for client {workspace}/{client_id}"
            )

        # Handle both single event type and list of event types
        if isinstance(event_type, list):
            for evt in event_type:
                connection.unsubscribe(evt)
        else:
            connection.unsubscribe(event_type)

        logger.info(f"Client {workspace}/{client_id} unsubscribed from: {event_type}")
        return True  # Explicitly return success to ensure RPC response is sent

    @schema_method
    async def get_env(
        self,
        key: Optional[str] = Field(
            None,
            description="Environment variable key (optional, returns all if not specified)",
        ),
        context: Optional[dict] = None,
    ):
        """Get environment variable(s) from the workspace."""
        assert context is not None
        ws = context["ws"]
        user_info = UserInfo.from_context(context)
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {ws}")

        workspace_info = await self.load_workspace_info(ws)
        workspace_info.config = workspace_info.config or {}
        environs = workspace_info.config.get("environs", {})

        if key is None:
            return environs
        else:
            if key not in environs:
                raise KeyError(
                    f"Environment variable '{key}' not found in workspace {ws}"
                )
            return environs[key]

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
        user_info = UserInfo.from_context(context)
        assert "bookmark_type" in item, "Bookmark type must be provided."
        user_workspace = await self.load_workspace_info(user_info.get_workspace())
        assert user_workspace, f"User workspace not found: {user_info.get_workspace()}"
        user_workspace.config = user_workspace.config or {}
        if "bookmarks" not in user_workspace.config:
            user_workspace.config["bookmarks"] = []
        if item["bookmark_type"] == "workspace":
            workspace = item["id"]
            workspace = await self.load_workspace_info(workspace, load=True)
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
        user_info = UserInfo.from_context(context)
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
        config: Union[dict, WorkspaceInfo] = Field(
            ...,
            description="Workspace configuration dictionary or WorkspaceInfo object. Must include 'id' field. Optional fields: 'name', 'description', 'persistent', 'owners', 'config'.",
        ),
        overwrite: bool = Field(
            False,
            description="If True, overwrites an existing workspace with the same ID. If False, raises an error if workspace already exists.",
        ),
        context: Optional[dict] = None,
    ):
        """Create a new workspace for organizing services and data.

        Workspaces provide isolated environments for services, applications, and data.
        They support multi-tenancy, access control, and resource management. Persistent
        workspaces have dedicated storage and can store artifacts, while non-persistent
        workspaces are temporary and cleaned up when inactive.

        Returns:
            Dict[str, Any]: The created workspace information including:
                - id: Workspace identifier
                - name: Workspace display name
                - description: Workspace description
                - persistent: Whether workspace has persistent storage
                - owners: List of user IDs who own the workspace
                - status: Current workspace status

        Examples:
            >>> # Create a basic workspace
            >>> workspace = await workspace_manager.create_workspace(
            ...     config={"id": "research-lab", "name": "Research Lab"}
            ... )
            >>>
            >>> # Create a persistent workspace with description
            >>> workspace = await workspace_manager.create_workspace(
            ...     config={
            ...         "id": "ml-project",
            ...         "name": "ML Project",
            ...         "description": "Machine learning experiments",
            ...         "persistent": True
            ...     }
            ... )
        """
        assert context is not None
        if isinstance(config, WorkspaceInfo):
            config = config.model_dump()
        assert "id" in config, "Workspace id must be provided."
        if not config.get("name"):
            config["name"] = config["id"]
        user_info = UserInfo.from_context(context)
        if user_info.is_anonymous:
            raise Exception("Only registered user can create workspace.")

        # Check if workspace exists (only if overwrite=True)
        exists = False
        if overwrite:
            try:
                existing_workspace = await self.load_workspace_info(
                    config["id"], increment_counter=False
                )
                exists = True

                # Authorization check: verify user can overwrite this workspace
                if not self._can_overwrite_workspace(user_info, existing_workspace):
                    logger.warning(
                        f"Unauthorized overwrite attempt: user={user_info.id}, "
                        f"workspace={existing_workspace.id}"
                    )
                    raise PermissionError(
                        f"Permission denied: user {user_info.id} cannot overwrite "
                        f"workspace {existing_workspace.id}"
                    )

                logger.info(
                    f"Authorized overwrite: user={user_info.id}, "
                    f"workspace={existing_workspace.id}"
                )

                # Close the existing workspace to clean up resources (e.g., vector collections)
                if existing_workspace.persistent and self._artifact_manager:
                    logger.info(
                        f"Closing existing workspace {config['id']} before overwriting"
                    )
                    await self._artifact_manager.close_workspace(existing_workspace)
            except KeyError:
                pass
        else:
            # If not overwriting, check if workspace exists and raise error if it does
            workspace_info = await self._redis.hget("workspaces", config["id"])
            if workspace_info is not None:
                raise RuntimeError(f"Workspace already exists: {config['id']}")

        config["persistent"] = config.get("persistent") or False
        if user_info.is_anonymous and config["persistent"]:
            raise Exception("Only registered user can create persistent workspace.")
        workspace = WorkspaceInfo.model_validate(config)
        if not workspace.owned_by(user_info):
            workspace.owners.append(user_info.id)
        self._validate_workspace_id(workspace.id, with_hyphen=user_info.id != "root")
        # make sure we add the user's email to owners
        if not workspace.owned_by(user_info):
            workspace.owners.append(user_info.id)
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

        # Set initial status as loading
        workspace.status = {"ready": False, "status": WorkspaceStatus.LOADING}

        # Write to Redis
        await self._redis.hset("workspaces", workspace.id, workspace.model_dump_json())

        if not exists:
            pass

        # Skip S3 setup for anonymous users
        if (
            not workspace.id.startswith(ANONYMOUS_USER_WS_PREFIX)
            and self._s3_controller
        ):
            await self._s3_controller.setup_workspace(workspace)

        # Verify workspace exists in Redis
        try:
            workspace_info = await self._redis.hget("workspaces", workspace.id)
            if workspace_info is None:
                logger.error("Failed to verify workspace creation")
                raise KeyError(f"Failed to create workspace: {workspace.id}")

            # For non-persistent workspaces or anonymous workspaces, set to ready immediately
            # For persistent workspaces, the status will be set by _prepare_workspace
            if not workspace.persistent or workspace.id.startswith(
                ANONYMOUS_USER_WS_PREFIX
            ):
                # Update status to ready for non-persistent workspaces
                await self._set_workspace_status(workspace.id, WorkspaceStatus.READY)

        except Exception as e:
            logger.error(f"Failed to verify workspace creation: {str(e)}")
            raise

        # Broadcast to the current workspace where the command was executed
        # so that listeners in that workspace can be notified
        current_workspace = user_info.get_workspace()
        await self._event_bus.broadcast(
            current_workspace, "workspace_loaded", workspace.model_dump()
        )
        if user_info.get_workspace() != workspace.id:
            await self.bookmark(
                {"bookmark_type": "workspace", "id": workspace.id}, context=context
            )
        logger.info("Created workspace %s", workspace.id)
        # prepare the workspace for the user
        if (
            not workspace.id.startswith(ANONYMOUS_USER_WS_PREFIX)
            and workspace.persistent
        ):  # Only prepare persistent workspaces
            task = asyncio.create_task(self._prepare_workspace(workspace))
            try:
                background_tasks.add(task)
            except TypeError:
                # Fallback if task can't be added to WeakSet
                pass
        elif workspace.persistent:
            # For persistent workspaces that don't need preparation, set to ready
            await self._set_workspace_status(workspace.id, WorkspaceStatus.READY)

        return workspace.model_dump()

    @schema_method
    async def delete_workspace(
        self,
        workspace: str = Field(..., description="workspace name"),
        context: Optional[dict] = None,
    ):
        """Remove a workspace."""
        assert context is not None
        user_info = UserInfo.from_context(context)
        workspace_info = await self.load_workspace_info(workspace)
        if not user_info.check_permission(
            workspace, UserPermission.admin
        ) and not workspace_info.owned_by(user_info):
            raise PermissionError(f"Permission denied for workspace {workspace}")
        # delete all the associated keys
        keys = await self._redis.keys(f"{workspace_info.id}:*")
        for key in keys:
            await self._redis.delete(key)
        if self._s3_controller:
            await self._s3_controller.cleanup_workspace(workspace_info, force=True)
        await self._redis.hdel("workspaces", workspace_info.id)
        # Broadcast to the current workspace where the command was executed
        current_workspace = user_info.get_workspace()
        await self._event_bus.broadcast(
            current_workspace, "workspace_deleted", workspace_info.model_dump()
        )
        # remove the workspace from the user's bookmarks
        user_workspace = await self.load_workspace_info(user_info.get_workspace())
        user_workspace.config = user_workspace.config or {}
        if "bookmarks" in user_workspace.config:
            user_workspace.config["bookmarks"] = [
                b
                for b in user_workspace.config["bookmarks"]
                if b["name"] != workspace_info.id
            ]
            await self._update_workspace(user_workspace, user_info)
        logger.info("Workspace %s removed by %s", workspace_info.id, user_info.id)

    @schema_method
    async def register_service_type(
        self,
        type_info: ServiceTypeInfo = Field(..., description="Service type info"),
        context: Optional[dict] = None,
    ):
        """Register a new service type."""
        assert context is not None
        ws = context["ws"]
        user_info = UserInfo.from_context(context)
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
    async def parse_token(
        self,
        token: str = Field(..., description="token to be parsed"),
        context: dict = None,
    ):
        """Parse a token."""
        assert context is not None
        self.validate_context(context, UserPermission.read)
        user_info = await parse_auth_token(token, context["ws"])
        return user_info.model_dump(mode="json")

    @schema_method
    async def revoke_token(
        self,
        token: str = Field(..., description="token to be revoked"),
        context: dict = None,
    ):
        """Revoke a token by storing it in Redis with a prefix and expiration time."""
        self.validate_context(context, UserPermission.admin)
        try:
            user_info = await parse_auth_token(token)
        except Exception as e:
            raise ValueError(f"Invalid token: {e}")
        expiration = int(user_info.expires_at - time.time())
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

        user_info = UserInfo.from_context(context)
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
        user_info = UserInfo.from_context(context)
        if not user_info.check_permission(ws, UserPermission.read):
            raise PermissionError(f"Permission denied for workspace {ws}")
        workspace_info = await self.load_workspace_info(ws)
        return list(workspace_info.service_types.values())

    @schema_method
    async def generate_token(
        self,
        config: Optional[TokenConfig] = Field(
            None,
            description="Configuration for token generation. Includes workspace, permission level, expiration time, client ID, and extra scopes. If not provided, uses defaults.",
        ),
        context: Optional[dict] = None,
    ):
        """Generate an authentication token for workspace access.

        Creates a JWT token with specified permissions and scopes for accessing
        workspace resources. Only workspace administrators can generate tokens.
        The token can be used for programmatic access to the workspace via API
        clients or for delegating access to other users/services.

        Returns:
            str: JWT authentication token that can be used in API requests.

        Examples:
            >>> # Generate a read-only token for current workspace
            >>> token = await workspace_manager.generate_token(
            ...     config={"permission": "read", "expires_in": 3600}
            ... )
            >>>
            >>> # Generate admin token for specific workspace
            >>> token = await workspace_manager.generate_token(
            ...     config={
            ...         "workspace": "research-lab",
            ...         "permission": "admin",
            ...         "expires_in": 86400,
            ...         "client_id": "data-processor"
            ...     }
            ... )
        """
        assert context is not None, "Context cannot be None"
        ws = context["ws"]
        user_info = UserInfo.from_context(context)
        config = config or TokenConfig()
        if isinstance(config, dict):
            config = TokenConfig.model_validate(config)
        assert isinstance(config, TokenConfig)

        extra_scopes = config.extra_scopes or []

        allowed_workspace = config.workspace or ws

        maximum_permission = user_info.get_permission(allowed_workspace)
        if not maximum_permission:
            workspace_info = await self.load_workspace_info(allowed_workspace)
            if workspace_info.owned_by(user_info):
                maximum_permission = UserPermission.admin
            else:
                raise PermissionError(
                    f"You do not have any permission for workspace: {allowed_workspace}"
                )
        if maximum_permission != UserPermission.admin:
            raise PermissionError("Only admin can generate token.")

        permission = config.permission
        permission = UserPermission[permission]

        # make it a child - preserve parent chain for tracking
        # If current user already has a parent, propagate that; otherwise use current user's ID
        parent_user_id = user_info.parent if user_info.parent else user_info.id
        user_info.parent = parent_user_id
        user_info.id = random_id(readable=True)
        user_info.scope = create_scope(
            {allowed_workspace: permission},
            current_workspace=allowed_workspace,
            client_id=config.client_id,
            extra_scopes=extra_scopes,
        )
        config.expires_in = config.expires_in or 3600
        token = await generate_auth_token(user_info, config.expires_in)
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
        # SECURITY FIX (V15): Validate that user has permission on the target workspace
        user_info = UserInfo.from_context(context)
        if not user_info.check_permission(workspace, UserPermission.read):
            raise PermissionError(f"Permission denied for workspace {workspace}")
        pattern = f"services:*|*:{workspace}/*:built-in@*"
        # Use SCAN to avoid blocking Redis in large deployments
        keys = []
        cursor = 0
        while True:
            cursor, batch = await self._redis.scan(
                cursor=cursor, match=pattern, count=500
            )
            keys.extend(batch)
            if cursor == 0:
                break
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
        # First, try to find all services for this workspace (not just built-in)
        pattern = f"services:*|*:{workspace}/*:*@*"
        keys = await self._redis.keys(pattern)
        client_keys = set()
        for key in keys:
            # Extract the client ID from the service key
            # Format: services:{visibility}|{service_id}:{workspace}/{client_id}:{service_name}@{app}
            key_str = key.decode("utf-8")
            # Split by workspace to find the client part
            if f":{workspace}/" in key_str:
                parts_after_ws = key_str.split(f":{workspace}/")[1]
                # Extract client_id (everything before the next colon)
                if ":" in parts_after_ws:
                    client_id = parts_after_ws.split(":")[0]
                    client_keys.add(workspace + "/" + client_id)

        # If no services found, check for client keys directly in Redis
        if not client_keys:
            # Try alternative pattern for clients
            client_pattern = f"clients:{workspace}/*"
            client_keys_raw = await self._redis.keys(client_pattern)
            for key in client_keys_raw:
                key_str = key.decode("utf-8")
                # Extract client_id from clients:{workspace}/{client_id}
                if f"clients:{workspace}/" in key_str:
                    client_id = key_str.split(f"clients:{workspace}/")[1]
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
        ws = context.get("ws", "")
        if not ws:
            return "Failed to ping client: no workspace in context"

        if "/" not in client_id:
            client_id = ws + "/" + client_id

        # Validate that the user has permission to access the target workspace
        target_workspace = client_id.split("/")[0]

        # Only validate permissions if trying to ping a client in a different workspace
        # Skip permission check for internal system operations (e.g., check-client-exists)
        from_client = context.get("from", "")
        is_internal_check = from_client and (
            from_client.endswith("/check-client-exists")
            or from_client == "check-client-exists"
        )
        if target_workspace != ws and not is_internal_check:
            try:
                self.validate_context(context, permission=UserPermission.read)
                user_info = UserInfo.from_context(context)
                if not user_info.check_permission(
                    target_workspace, UserPermission.read
                ):
                    raise PermissionError(
                        f"Permission denied for workspace {target_workspace}"
                    )
            except Exception as e:
                logger.warning(f"Permission check failed in ping_client: {e}")
                return f"Failed to ping client {client_id}: {e}"

        # First try the built-in service
        try:
            svc = await self._rpc.get_remote_service(
                client_id + ":built-in", {"timeout": timeout}
            )
            return await svc.ping("ping")  # should return "pong"
        except Exception as e:
            # If built-in service fails, try to find any other service for this client
            workspace_id, client_name = client_id.split("/")
            pattern = f"services:*|*:{workspace_id}/{client_name}:*@*"
            keys = await self._redis.keys(pattern)

            if keys:
                # Try to ping using the first available service
                for key in keys[:3]:  # Try up to 3 services
                    try:
                        # Extract service ID from key
                        key_str = key.decode("utf-8")
                        # Format: services:{visibility}|{service_id}:{workspace}/{client_id}:{service_name}@{app}
                        if "|" in key_str:
                            service_full_id = key_str.split("|")[1].split("@")[0]
                            # Try to get this service
                            svc = await self._rpc.get_remote_service(
                                service_full_id, {"timeout": timeout}
                            )
                            # Try to call echo or ping method if available
                            if hasattr(svc, "ping"):
                                return await svc.ping("ping")
                            elif hasattr(svc, "echo"):
                                result = await svc.echo("ping")
                                return (
                                    "pong"
                                    if result == "ping"
                                    else f"Client responded: {result}"
                                )
                            else:
                                # Just check if we can get service info - if yes, client is alive
                                if hasattr(svc, "_config"):
                                    return "pong"  # Client is alive but no ping method
                    except Exception:
                        continue  # Try next service

            # All attempts failed
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
        include_unlisted: bool = Field(
            False,
            description="Whether to include unlisted services from the current workspace in search results.",
        ),
        context: Optional[dict] = None,
    ):
        """
        Search services with support for hybrid queries and pure filter-based queries.
        """
        if not self._enable_service_search:
            raise RuntimeError("Service search is not enabled.")

        # Ensure the services collection exists before searching
        await self._ensure_services_collection()

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

        # Use simple auth filter and do post-processing for unlisted services
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
        for item in results["items"]:
            item["id"] = re.sub(r"^[^|]+\|[^:]+:(.+)$", r"\1", item["id"]).split("@")[0]

        # Filter out unlisted services if not explicitly requested
        if not include_unlisted:
            filtered_items = []
            for item in results["items"]:
                service_visibility = item.get("config", {}).get("visibility")
                if service_visibility != "unlisted":
                    filtered_items.append(item)
            results["items"] = filtered_items

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
        include_unlisted: bool = Field(
            False,
            description="Whether to include unlisted services in the results. Only works if the user is from the same workspace as the unlisted services.",
        ),
        include_app_services: bool = Field(
            False,
            description="Whether to include services from installed applications that are not currently running. These services are discovered from application artifact manifests.",
        ),
        prioritize_running_services: bool = Field(
            False,
            description="When True and include_app_services is True, prioritize returning running services over non-running app services when both exist for the same client and service ID.",
        ),
        context: Optional[dict] = None,
    ):
        """Return a list of services based on the query, optionally including services from installed apps."""
        assert context is not None
        cws = context["ws"]
        user_info = UserInfo.from_context(context)
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
            assert original_visibility not in [
                "protected",
                "unlisted",
            ], "Cannot list protected or unlisted services in all workspaces."
            if include_unlisted:
                raise PermissionError(
                    "Cannot include unlisted services when searching all workspaces."
                )
            query["visibility"] = "public"
        elif workspace not in ["public", cws]:
            # Do not block listing here; later logic will restrict visibility to public
            # Check permission for including unlisted services from other workspaces
            if include_unlisted and workspace != cws:
                raise PermissionError(
                    f"Cannot include unlisted services from workspace {workspace}. Only services from your current workspace ({cws}) can include unlisted services."
                )

        # Validate include_unlisted permissions for current workspace
        if include_unlisted and workspace == cws:
            if not user_info.check_permission(cws, UserPermission.read):
                raise PermissionError(
                    f"Cannot include unlisted services without read permission for workspace {cws}."
                )

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

        # If the user does not have permission to read the workspace,
        # we need to include protected services as well (to check authorized_workspaces later)
        # but not unlisted services
        if not user_info.check_permission(workspace, UserPermission.read):
            if visibility == "*":
                # Include public and protected, but not unlisted
                visibility = "*"  # Keep as * to match both public and protected
            elif visibility == "unlisted":
                # Cannot list unlisted services without permission
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

        # Construct patterns based on whether we want to include unlisted services
        patterns = []
        if include_unlisted and workspace == cws:
            # If including unlisted in current workspace, query all visibility types
            for vis in ["public", "protected", "unlisted"]:
                patterns.append(
                    f"services:{vis}|{type_filter}:{workspace}/{client_id}:{service_id}@{app_id}"
                )
        else:
            # Normal query with specified visibility (may be "*")
            patterns.append(
                f"services:{visibility}|{type_filter}:{workspace}/{client_id}:{service_id}@{app_id}"
            )

        assert all(
            pattern.startswith("services:") for pattern in patterns
        ), "Query patterns do not start with 'services:'."
        assert all(
            not any(char in pattern for char in "{}") for pattern in patterns
        ), "Query patterns contain invalid characters."

        keys = []
        # Use SCAN for each pattern
        for pattern in patterns:
            cursor = 0
            while True:
                cursor, batch = await self._redis.scan(
                    cursor=cursor, match=pattern, count=500
                )
                keys.extend(batch)
                if cursor == 0:
                    break

        if workspace == "*":
            # add services in the current workspace (but not unlisted unless explicitly requested)
            if include_unlisted:
                # This case should already be blocked by permission check above
                pass  # Already handled above with error
            else:
                ws_pattern = f"services:{original_visibility}|{type_filter}:{cws}/{client_id}:{service_id}@{app_id}"
                # Scan additional keys in current workspace to avoid blocking Redis
                cursor = 0
                while True:
                    cursor, batch = await self._redis.scan(
                        cursor=cursor, match=ws_pattern, count=500
                    )
                    keys.extend(batch)
                    if cursor == 0:
                        break

        services = []
        for key in set(keys):
            service_data = await self._redis.hgetall(key)
            service_info = ServiceInfo.from_redis_dict(service_data)
            service_dict = service_info.model_dump()
            service_visibility = service_dict.get("config", {}).get("visibility")
            service_workspace = (
                service_dict.get("id", "").split("/")[0]
                if "/" in service_dict.get("id", "")
                else workspace
            )

            # Check if this is an unlisted service and whether we should include it
            if service_visibility == "unlisted":
                if not include_unlisted:
                    # Skip unlisted services if not explicitly requested
                    continue
                elif service_workspace != cws:
                    # Skip unlisted services from other workspaces
                    continue

            # SECURITY: Permission check for protected services
            # This is the critical security enforcement point that ensures:
            # 1. Anonymous users (with workspace="ws-anonymous") can ONLY see public services
            # 2. Authenticated users without workspace permission need authorized_workspaces
            # 3. Protected services (default visibility) are isolated by workspace
            if service_visibility not in ["public", "unlisted"]:
                has_workspace_permission = user_info.check_permission(
                    service_workspace, UserPermission.read
                )

                if not has_workspace_permission:
                    # User doesn't have permission for this workspace
                    # Check if user's workspace is in authorized_workspaces for controlled access
                    config = service_dict.get("config", {})
                    authorized_workspaces = (
                        config.get("authorized_workspaces")
                        if isinstance(config, dict)
                        else getattr(config, "authorized_workspaces", None)
                    )
                    user_workspace = cws  # Current workspace from context

                    if not (
                        authorized_workspaces
                        and user_workspace in authorized_workspaces
                    ):
                        # SECURITY: User doesn't have access - skip this protected service
                        # This ensures anonymous users (cws="ws-anonymous") cannot see protected services
                        continue

                    # User is authorized via authorized_workspaces
                    # For security, remove the authorized_workspaces list from response
                    if "authorized_workspaces" in service_dict.get("config", {}):
                        del service_dict["config"]["authorized_workspaces"]
                # else: user has workspace permission, keep authorized_workspaces for owner

            services.append(service_dict)

        # Include services from installed application artifacts if requested
        if include_app_services:
            if not self._artifact_manager:
                logger.warning("Artifact manager not available, skipping app services")
                return services
            try:
                # Build filters for querying artifacts with service_ids
                artifact_filters = {"type": "application"}

                # Add more specific filters based on query parameters
                if service_id != "*":
                    # Build the full service ID pattern to match
                    # Note: service_ids in manifest don't have @app-id suffix
                    service_id_pattern = f"{workspace}/*:{service_id}"
                    # Use $in operator to check if the service_ids array contains this specific ID
                    artifact_filters["manifest"] = {
                        "service_ids": {"$in": [service_id_pattern]}
                    }

                # Add app_id filter if specified
                if app_id != "*":
                    artifact_filters["alias"] = app_id

                # Query applications collection for the current workspace only
                # Use the workspace from the query, not from context
                app_context = dict(context)
                if workspace != "*":
                    app_context["workspace"] = workspace

                try:
                    # List children artifacts (installed apps) with filters
                    # Only query committed artifacts (stage=False)
                    all_artifacts = await self._artifact_manager.list(
                        parent_id=f"{cws}/applications",
                        filters=artifact_filters,
                        limit=1000,  # Reasonable limit for apps
                        stage=False,  # Only query committed artifacts, not staged
                        context=app_context,
                    )
                except KeyError as e:
                    logger.debug(f"KeyError checking applications collection: {e}")
                    all_artifacts = []

                # Process artifacts to extract all services at once
                existing_service_ids = {svc.get("id") for svc in services}

                # Build a map of concrete services for abstract filtering
                concrete_services_map = {}
                if prioritize_running_services:
                    # Map service patterns without specific client IDs to track concrete implementations
                    # E.g., "ws-user-github|478667/_rapp_charm-cathedral-75264096__rlb:webcontainer-compiler-worker"
                    for svc_id in existing_service_ids:
                        if svc_id and "/" in svc_id and ":" in svc_id:
                            # Extract workspace/client:service pattern
                            parts = svc_id.split("/")
                            if len(parts) == 2:
                                ws_part = parts[0]
                                client_service = parts[1].split(":")
                                if len(client_service) == 2:
                                    client_part = client_service[0]
                                    service_part = client_service[1].split("@")[
                                        0
                                    ]  # Remove @app-id if present
                                    # Check if this is a concrete service (not containing *)
                                    if "*" not in client_part:
                                        # Create a key for the abstract version
                                        abstract_key = f"{ws_part}/*:{service_part}"
                                        if abstract_key not in concrete_services_map:
                                            concrete_services_map[abstract_key] = []
                                        concrete_services_map[abstract_key].append(
                                            svc_id
                                        )

                for artifact in all_artifacts:
                    manifest = artifact.get("manifest", {})
                    app_services = manifest.get("services", [])
                    artifact_app_id = artifact.get("alias") or artifact.get("id")

                    # Process all services from this app
                    for app_service in app_services:
                        # Format service ID as <service_id>@<app-id> for lazy loading
                        original_service_id = app_service.get("id")
                        if original_service_id:
                            # Extract the service name part and append @app-id
                            parts = original_service_id.split(":")
                            if len(parts) >= 2:
                                service_name = parts[-1]
                                # Reconstruct as workspace/client:service@app-id
                                service_id_with_app = f"{':'.join(parts[:-1])}:{service_name}@{artifact_app_id}"
                            else:
                                service_id_with_app = (
                                    f"{original_service_id}@{artifact_app_id}"
                                )
                        else:
                            continue  # Skip if no service ID

                        # Check if this is an abstract service and if we should exclude it
                        if prioritize_running_services and "*" in service_id_with_app:
                            # Check if a concrete version exists
                            # Extract the base pattern without @app-id
                            base_service_id = service_id_with_app.split("@")[0]
                            if (
                                base_service_id in concrete_services_map
                                and concrete_services_map[base_service_id]
                            ):
                                # Concrete implementation exists, skip this abstract service
                                logger.debug(
                                    f"Skipping abstract service {service_id_with_app} because concrete implementations exist: {concrete_services_map[base_service_id]}"
                                )
                                continue

                        service_dict = {
                            "id": service_id_with_app,
                            "name": app_service.get("name"),
                            "type": app_service.get("type"),
                            "description": app_service.get("description"),
                            "config": app_service.get("config", {}),
                            "app_id": artifact_app_id,
                            "_source": "app_manifest",
                        }

                        # Skip if service ID is missing or already running
                        if (
                            not service_dict["id"]
                            or service_dict["id"] in existing_service_ids
                        ):
                            continue

                        # Apply type filter
                        if (
                            type_filter != "*"
                            and service_dict.get("type") != type_filter
                        ):
                            continue

                        # Apply service_id filter (double-check since we already filtered at query level)
                        if service_id != "*":
                            svc_id_parts = service_dict["id"].split(":")
                            if len(svc_id_parts) > 1:
                                svc_name = svc_id_parts[1].split("@")[0]
                                logger.debug(
                                    f"DEBUG: Filtering service_id={service_id}, service={service_dict['id']}, svc_name={svc_name}"
                                )
                                if svc_name != service_id:
                                    logger.debug(
                                        f"DEBUG: Skipping service {service_dict['id']} because {svc_name} != {service_id}"
                                    )
                                    continue

                        # Get service visibility and workspace
                        service_visibility = service_dict.get("config", {}).get(
                            "visibility", "public"
                        )
                        service_workspace = (
                            service_dict["id"].split("/")[0]
                            if "/" in service_dict["id"]
                            else workspace
                        )

                        # Apply visibility filter
                        if visibility != "*" and service_visibility != visibility:
                            if not (
                                visibility == "protected"
                                and service_visibility in ["protected", "public"]
                            ):
                                continue

                        # Check unlisted services
                        if service_visibility == "unlisted":
                            if not include_unlisted:
                                continue
                            elif service_workspace != cws:
                                continue

                        # Check protected services permissions
                        if service_visibility == "protected":
                            has_workspace_permission = user_info.check_permission(
                                service_workspace, UserPermission.read
                            )

                            if not has_workspace_permission:
                                config = service_dict.get("config", {})
                                authorized_workspaces = config.get(
                                    "authorized_workspaces", []
                                )
                                if not (
                                    authorized_workspaces
                                    and cws in authorized_workspaces
                                ):
                                    continue

                                # Remove authorized_workspaces for security
                                if "authorized_workspaces" in service_dict.get(
                                    "config", {}
                                ):
                                    del service_dict["config"]["authorized_workspaces"]

                        services.append(service_dict)

            except KeyError:
                # Applications collection doesn't exist yet
                logger.debug("Applications collection not found, skipping app services")
            except Exception as e:
                raise e

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
            None,
            description="Options for registering service, available options: skip_app_id_validation (bool, default: False) - skip validation of app_id during registration",
        ),
        context: Optional[dict] = None,
    ):
        """Register a new service."""
        assert context is not None
        ws = context["ws"]
        client_id = context["from"]
        user_info = UserInfo.from_context(context)
        config = config or {}
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(f"Permission denied for workspace {ws}")
        if "/" not in client_id:
            client_id = f"{ws}/{client_id}"
        service.config.workspace = ws
        if "/" not in service.id:
            service.id = f"{ws}/{service.id}"
        assert ":" in service.id, "Service id info must contain ':'"

        # Get workspace from service id for validation
        workspace = service.id.split("/")[0]

        # SECURITY: Validate that service ID workspace prefix matches context workspace
        # This prevents attackers from registering services that appear to belong to other workspaces
        if workspace != ws:
            raise PermissionError(
                f"Service ID workspace prefix '{workspace}' does not match the current workspace '{ws}'. "
                f"Cannot register services in other workspaces."
            )
        service_name = service.id.split(":")[1]
        service.name = service.name or service_name

        if service.app_id:
            assert "/" not in service.app_id, "App id info must not contain '/'"

        # Store all the info for client's built-in services
        if service_name == "built-in" and service.type == "built-in":
            service.config.created_by = user_info.model_dump()
        else:
            service.config.created_by = {"id": user_info.id}

        # Validate authorized_workspaces is only used with protected visibility
        if (
            hasattr(service.config, "authorized_workspaces")
            and service.config.authorized_workspaces is not None
        ):
            visibility = (
                service.config.visibility.value
                if isinstance(service.config.visibility, VisibilityEnum)
                else service.config.visibility
            )
            if visibility != "protected":
                raise ValueError(
                    f"authorized_workspaces can only be set when visibility is 'protected', got visibility='{visibility}'"
                )

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

        # Handle None app_id by defaulting to "*"
        app_id = service.app_id or "*"
        service.app_id = (
            app_id  # Store the normalized app_id back to the service object
        )
        key = f"services:{visibility}|{service.type}:{service.id}@{app_id}"

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
                await self._event_bus.broadcast(
                    ws, "client_updated", {"id": client_id, "workspace": ws}
                )
                logger.info(f"Updating built-in service: {service.id}")
            else:
                await self._event_bus.broadcast(
                    ws, "service_updated", service.model_dump()
                )
                logger.info(f"Updating service: {service.id}")
        else:
            if self._enable_service_search:
                redis_data = await self._embed_service(service.to_redis_dict())
                # Add to vector search index
                vector_data = redis_data.copy()
                vector_data["id"] = f"{visibility}|{service.type}:{service.id}@{app_id}"
                # Extract workspace and visibility for vector search filtering
                vector_data["workspace"] = service.config.workspace or ws
                vector_data["visibility"] = visibility
                # Convert bytes back to numpy array for vector search
                if "service_embedding" in vector_data and isinstance(
                    vector_data["service_embedding"], bytes
                ):
                    vector_data["service_embedding"] = np.frombuffer(
                        vector_data["service_embedding"], dtype=np.float32
                    )
                logger.info(
                    f"Adding new service to vector search: id={vector_data['id']}, workspace={vector_data['workspace']}, visibility={vector_data['visibility']}, has_embedding={'service_embedding' in vector_data}"
                )
                try:
                    # Ensure the services collection exists before adding vectors
                    await self._ensure_services_collection()
                    await self._vector_search.add_vectors(
                        "services", [vector_data], update=False
                    )
                    logger.info(
                        f"Successfully added service {vector_data['id']} to vector search"
                    )
                except Exception as e:
                    logger.error(f"Failed to add service to vector search: {e}")
                    # Re-raise to maintain original behavior
                    raise
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
                await self._event_bus.broadcast(
                    ws, "client_connected", {"id": client_id, "workspace": ws}
                )
                logger.info(f"Adding built-in service: {service.id}")

            else:
                # Remove the service embedding from the config
                if service.config and service.config.service_embedding is not None:
                    service.config.service_embedding = None
                await self._event_bus.broadcast(
                    ws, "service_added", service.model_dump(mode="json")
                )
                logger.info(f"Adding service {service.id}")

    @schema_method
    async def get_service_info(
        self,
        service_id: str = Field(
            ...,
            description="Service id, it can be the service id or the full service id with workspace: `workspace/client_id:service_id`. Special value '~' expands to the workspace manager service.",
        ),
        config: Optional[dict] = Field(
            None,
            description="Options for getting service, available configs are `mode` (for selecting the service: 'random', 'first', 'last', 'exact', 'min_load', 'native:random', or 'select:criteria:function') and `select_timeout` (timeout for function calls when using select mode)",
        ),
        context: Optional[dict] = None,
    ):
        """Get the service info."""
        # Don't validate context here - we'll check permissions after determining if service is public
        assert isinstance(service_id, str), "Service ID must be a string."
        assert service_id.count("/") <= 1, "Service id must contain at most one '/'"
        assert service_id.count(":") <= 1, "Service id must contain at most one ':'"
        assert service_id.count("@") <= 1, "Service id must contain at most one '@'"
        # Handle special "~" shortcut for workspace manager early
        # Check if the service_id is just "~" or contains "~" as the service name
        if service_id == "~" or (
            isinstance(service_id, str) and service_id.endswith("/~")
        ):
            # Return the stored service info from registration
            if not self._service_info:
                raise RuntimeError("Workspace manager service info not available")

            # Return the service info directly - no expansion needed
            return self._service_info

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
        config = config or {}
        mode = config.get("mode")
        read_app_manifest = config.get("read_app_manifest", False)
        user_info = UserInfo.from_context(context)
        key = f"services:*|*:{service_id}@{app_id}"
        keys = await self._redis.keys(key)
        if not keys:
            # If no services found in Redis, try to get from artifact manifest if app_id is provided
            if app_id != "*" and read_app_manifest and self._artifact_manager:
                try:
                    artifact = await self._artifact_manager.read(
                        app_id, context=context
                    )
                    if (
                        artifact
                        and artifact.get("manifest")
                        and artifact["manifest"].get("services")
                    ):
                        # Extract service name from service_id (format: workspace/client_id:service_name)
                        service_name = (
                            service_id.split(":")[1]
                            if ":" in service_id
                            else service_id
                        )

                        # Look for the service in the manifest services
                        for svc in artifact["manifest"]["services"]:
                            svc_info = ServiceInfo.model_validate(svc)
                            # The stored service ID is like "workspace/*:service_name"
                            if svc_info.id.endswith(":" + service_name):
                                # Check if workspace matches
                                stored_workspace = svc_info.id.split("/")[0]
                                if stored_workspace == workspace:
                                    return svc_info

                except Exception as e:
                    logger.warning(
                        f"Failed to read artifact {app_id} for service lookup: {e}"
                    )

            # If no services found in Redis and not found in artifact, raise KeyError to trigger lazy loading
            raise KeyError(f"Service not found: {service_id}@{app_id}")

        if mode is None:
            # If app_id is provided, try to get service_selection_mode directly from artifact
            if app_id != "*" and self._artifact_manager:
                try:
                    artifact = await self._artifact_manager.read(
                        app_id, context=context
                    )
                    if (
                        artifact
                        and artifact["manifest"]
                        and artifact["manifest"].get("service_selection_mode")
                    ):
                        mode = artifact["manifest"]["service_selection_mode"]
                except Exception as e:
                    logger.warning(
                        f"Failed to read artifact {app_id} for retrieving service selection mode: {e}"
                    )
            # If mode is still None, apply default logic
            if mode is None:
                # Set random mode for public services, since there can be many hypha servers
                if workspace == "public":
                    mode = "random"
                else:
                    mode = "exact"
        if mode == "exact":
            assert (
                len(keys) == 1
            ), f"Multiple services found for {service_id}, e.g. {keys[:5]}. You can either specify the mode as 'random', 'first', 'last', 'native:random', or use 'select:criteria:function' format."
            key = keys[0]
        elif mode == "random":
            key = random.choice(keys)
        elif mode == "first":
            key = keys[0]
        elif mode == "last":
            key = keys[-1]
        elif mode == "min_load":
            key = await self._select_service_by_load(keys, "min")
        elif mode == "native:random":
            key = await self._select_native_service(keys)
        elif mode and mode.startswith("select:"):
            # Parse the select syntax: select:criteria:function
            parts = mode.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid select mode format: {mode}. Expected 'select:criteria:function'"
                )

            _, criteria, function_name = parts
            select_timeout = (
                config.get("select_timeout", 2.0)
                if isinstance(config, dict)
                else getattr(config, "select_timeout", 2.0)
            )
            key = await self._select_service_by_function(
                keys, criteria, function_name, select_timeout
            )
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Mode must be 'random', 'first', 'last', 'exact', 'min_load', 'native:random', or 'select:criteria:function' format (e.g., 'select:min:get_load')"
            )
        # SECURITY: Check access permissions for protected/unlisted services
        # This ensures anonymous users can ONLY access public services via MCP/HTTP endpoints
        if not key.startswith(b"services:public|"):
            # For non-public services, validate context and permissions
            self.validate_context(context, permission=UserPermission.read)
            # First check if user has read permission in the service's workspace
            has_workspace_permission = user_info.check_permission(
                workspace, UserPermission.read
            )

            if not has_workspace_permission:
                # Get service data to check authorized_workspaces
                service_data = await self._redis.hgetall(key)
                service_info = ServiceInfo.from_redis_dict(service_data)

                # Check if the service has authorized_workspaces configured
                authorized_workspaces = getattr(
                    service_info.config, "authorized_workspaces", None
                )
                user_workspace = context.get("ws") if context else None

                # Check if user's workspace is in the authorized_workspaces list
                if not (
                    authorized_workspaces
                    and user_workspace
                    and user_workspace in authorized_workspaces
                ):
                    # SECURITY: User is not authorized - don't reveal any service information
                    # This blocks anonymous users (workspace="ws-anonymous") from accessing protected services
                    raise PermissionError(
                        f"Permission denied for non-public service in workspace {workspace}"
                    )

                # User is authorized via authorized_workspaces
                # For security, remove the authorized_workspaces list from the returned service info
                # so that authorized users from other workspaces cannot see the full list
                if hasattr(service_info.config, "authorized_workspaces"):
                    delattr(service_info.config, "authorized_workspaces")
                return service_info

        # Either it's a public service or user has permission - fetch and return service data
        service_data = await self._redis.hgetall(key)
        service_info = ServiceInfo.from_redis_dict(service_data)

        # Users with workspace permission can see authorized_workspaces
        # (Nothing to do here - just return the full service_info)

        return service_info

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
        user_info = UserInfo.from_context(context)
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
                await self._event_bus.broadcast(
                    ws, "client_disconnected", {"id": client_id, "workspace": ws}
                )

            else:
                await self._event_bus.broadcast(
                    ws, "service_removed", service.model_dump()
                )

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
                "encoder": lambda x: x.model_dump(mode="json"),
            }
        )
        return rpc

    def validate_context(self, context: dict, permission: UserPermission):
        user_info = UserInfo.from_context(context)
        if not user_info.check_permission(context["ws"], permission):
            raise PermissionError(f"Permission denied for workspace {context['ws']}")

    @schema_method
    async def echo(
        self, data: Any = Field(..., description="echo an object"), context: Any = None
    ):
        """Log a app message."""
        self.validate_context(context, permission=UserPermission.read_write)
        return data

    @schema_method
    async def log(
        self, msg: str = Field(..., description="log a message"), context=None
    ):
        """Log a app message."""
        self.validate_context(context, permission=UserPermission.read_write)
        await self.log_event("log", msg, context=context)

    async def load_workspace_info(
        self, workspace: str, load=True, increment_counter=True
    ) -> WorkspaceInfo:
        """Load info of the current workspace from the redis store.

        Args:
            workspace: The workspace ID to load
            load: Whether to attempt loading from S3 if not found in Redis
            increment_counter: Deprecated parameter, kept for compatibility
        """
        assert workspace is not None

        # First try to get workspace info from Redis
        try:
            workspace_info = await self._redis.hget("workspaces", workspace)
            if workspace_info is not None:
                workspace_info = WorkspaceInfo.model_validate(
                    json.loads(workspace_info.decode())
                )
                # Reset activity timer for user workspaces when accessed
                await self._activity_manager.reset_activity(workspace)

                # If workspace exists but isn't ready (status=None from unload), prepare it
                if workspace_info.status is None and workspace_info.persistent:
                    logger.info(
                        f"Re-preparing previously unloaded workspace: {workspace}"
                    )
                    # Set loading status first
                    workspace_info.status = {
                        "ready": False,
                        "status": WorkspaceStatus.LOADING,
                    }
                    await self._redis.hset(
                        "workspaces", workspace, workspace_info.model_dump_json()
                    )

                    # Trigger preparation in background
                    task = asyncio.create_task(self._prepare_workspace(workspace_info))
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)

                return workspace_info
        except Exception as e:
            logger.error(f"Failed to load workspace info from Redis: {e}")

        if not load:
            raise KeyError(f"Workspace not found: {workspace}")

        # Skip S3 loading for anonymous workspaces
        if workspace.startswith(ANONYMOUS_USER_WS_PREFIX):
            logger.debug(f"Anonymous workspace {workspace}, skipping S3 loading")
            workspace_info = WorkspaceInfo(
                id=workspace,
                name=workspace,
                description="Anonymous workspace",
                owners=[workspace],
                persistent=False,
                read_only=not self._enable_s3_for_anonymous_users,
                status={"ready": True},
            )
            await self._redis.hset(
                "workspaces", workspace_info.id, workspace_info.model_dump_json()
            )

            return workspace_info

        logger.info(f"Workspace {workspace} not found in Redis, trying to load from S3")

        try:
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

            # Start preparation tasks
            task = asyncio.create_task(self._prepare_workspace(workspace_info))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

            await self._s3_controller.setup_workspace(workspace_info)
            await self._event_bus.broadcast(
                workspace, "workspace_loaded", workspace_info.model_dump()
            )

            # Reset activity timer for user workspaces when loaded from S3
            await self._activity_manager.reset_activity(workspace)
            return workspace_info

        except Exception as e:
            logger.debug(f"Workspace {workspace} does not exist")
            raise

    @schema_method
    async def get_workspace_info(
        self, workspace: str = Field(None, description="workspace name"), context=None
    ) -> dict:
        """Get the workspace info."""
        self.validate_context(context, permission=UserPermission.read)
        workspace_info = await self.load_workspace_info(workspace)
        return workspace_info.model_dump()

    async def _get_application_by_service_id(
        self,
        app_id: str,
        service_id: str,
        workspace: str = None,
        context: dict = None,
    ) -> tuple:
        """Get application info and service info by service id from the artifact manager.

        Returns:
            tuple: (app_info, service_info) where app_info is ApplicationManifest and service_info is ServiceInfo
        """
        self.validate_context(context, permission=UserPermission.read)
        ws = context["ws"]
        workspace = workspace or ws
        user_info = UserInfo.from_context(context)
        assert app_id not in ["*"], f"Invalid app id: {app_id}"
        if workspace == "*":
            workspace = ws

        if not user_info.check_permission(workspace, UserPermission.read):
            raise PermissionError(f"Permission denied for workspace {workspace}")

        if not self._artifact_manager:
            raise Exception(
                "Failed to get application info: artifact-manager service not found."
            )
        assert (
            ":" not in service_id
        ), f"To get application service info, the service name should not specify the client id, i.e. please remove the client id from the service name: {service_id}"

        assert service_id and service_id not in [
            "*",
            "built-in",
        ], f"Invalid service id: {service_id}"

        # Try to find the application in both committed and staged versions
        # First try committed applications
        try:
            applications = await self._artifact_manager.list(
                f"{workspace}/applications",
                context={"ws": workspace, "user": user_info.model_dump()},
            )
        except Exception:
            logger.warning(
                f"Failed to list applications in workspace {workspace}, error: {traceback.format_exc()}"
            )
            applications = []
        # Build applications dict, using alias as fallback if manifest["id"] is missing
        # This handles legacy artifacts or partially failed installations
        applications_list = applications  # Save the list before converting to dict
        applications = {}
        for item in applications_list:
            manifest = item.get("manifest", {})
            # Use manifest["id"] if available
            app_key = manifest.get("id")
            if app_key:
                applications[app_key] = ApplicationManifest.model_validate(manifest)
            else:
                logger.warning(
                    f"Skipping artifact {item.get('id')} - missing both manifest['id'] and alias"
                )
        if app_id not in applications:
            raise KeyError(
                f"Application id `{app_id}` not found in workspace {workspace}"
            )

        # Make sure the service_id is in the application services
        app_info = applications[app_id]
        assert (
            app_info.services
        ), f"No services found in application {app_id}, please make sure it's properly installed."

        # Find the service in the application
        found_service = None
        for svc in app_info.services:
            if svc.id.endswith(":" + service_id):
                found_service = svc
                break

        if not found_service:
            raise KeyError(
                f"Service id `{service_id}` not found in application {app_id}"
            )

        return app_info, found_service

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
        # Get application and service info
        # read the info using root user since the user might not able to read it
        app_info, service_info = await self._get_application_by_service_id(
            app_id,
            service_id,
            workspace,
            context={"user": self._root_user.model_dump(), "ws": workspace},
        )

        if not self._server_app_controller:
            raise Exception(
                "Failed to launch application: server apps controller is not configured."
            )

        # Use startup_context from the app_info if available, otherwise use current context
        startup_context = getattr(app_info, "startup_context", None)
        if startup_context:
            # Create context using the stored workspace and user info from installation time
            launch_context = {
                "ws": startup_context["ws"],
                "user": startup_context["user"],
            }
        else:
            # Fallback to current context if no startup_context is available
            launch_context = context

        client_info = await self._server_app_controller.start(
            app_id,
            wait_for_service=service_id,
            context=launch_context,
        )
        return await self.get_service(
            f"{client_info['id']}:{service_id}",  # should not contain @app_id
            dict(timeout=timeout, case_conversion=case_conversion),
            context=context,
        )

    @schema_method
    async def get_service(
        self,
        service_id: str = Field(
            ...,
            description="Service ID. This should be a service id in the format: 'workspace/service_id', 'workspace/client_id:service_id' or 'workspace/client_id:service_id@app_id'. Special value '~' expands to the workspace manager service.",
        ),
        config: Optional[GetServiceConfig] = Field(
            None, description="Get service config"
        ),
        context=None,
    ):
        """Get a service based on the service_id"""
        # Don't validate workspace permission here - service visibility
        # handles access control. Public services should be accessible to anyone,
        # including anonymous users. Protected services are checked in
        # get_service_info() based on visibility.
        assert (
            service_id != "*"
        ), "Invalid service id: {service_id}, it cannot be a wildcard."
        try:
            config = config or GetServiceConfig()
            # Permission check will be handled by the get_service_api function
            svc_info = await self.get_service_info(
                service_id, {"mode": config.mode}, context=context
            )
            # Handle special "~" shortcut for workspace manager
            # Check if service_id is "~" or ends with "/~" (for HTTP endpoint calls)
            if svc_info == self._service_info:
                # Return the workspace manager service directly as a local service
                if not self._rpc:
                    raise RuntimeError("Workspace manager RPC not initialized")

                # Get the workspace manager service from our own RPC as a LOCAL service
                # Since we ARE the workspace manager, we return ourselves
                # Pass the context to get_local_service (it's synchronous, not async)
                return self._rpc.get_local_service("default", context=context)

            service_api = await self._rpc.get_remote_service(
                svc_info.id,
                {"timeout": config.timeout, "case_conversion": config.case_conversion},
            )
            assert service_api, f"Failed to get service: {svc_info.id}"

            workspace = svc_info.id.split("/")[0]
            service_api["config"]["workspace"] = workspace
            service_api["config"][
                "url"
            ] = f"{self._server_info['public_base_url']}/{workspace}/services/{svc_info.id.split('/')[-1]}"
            return service_api
        except asyncio.CancelledError:
            # Client disconnected during service retrieval
            logger.warning(
                f"Service retrieval cancelled for {service_id}, likely due to client disconnection"
            )
            raise ConnectionError(
                f"Client disconnected while retrieving service: {service_id}"
            )
        except RemoteException as e:
            # Check if this is a wrapped CancelledError (common when client disconnects)
            error_str = str(e)
            if "CancelledError" in error_str and "TimeoutError" in error_str:
                logger.warning(
                    f"Service retrieval failed for {service_id}, likely due to client disconnection: {e}"
                )
                raise ConnectionError(
                    f"Client disconnected while retrieving service: {service_id}"
                ) from e
            # Re-raise other RemoteExceptions as-is
            raise
        except KeyError as exp:
            if "@" in service_id:
                app_id = service_id.split("@")[1]
                service_id = service_id.split("@")[0]
                workspace = (
                    service_id.split("/")[0] if "/" in service_id else context["ws"]
                )
                # Extract just the service name from the full service ID
                if "/" in service_id:
                    service_name = service_id.split("/")[1]
                else:
                    service_name = service_id
                # Remove client_id part if present (e.g., "*:default" -> "default")
                if ":" in service_name:
                    service_name = service_name.split(":", 1)[1]
                assert (
                    ":" not in service_name
                ), f"To automatically launch an application, the service name should not specify the client id, i.e. please remove the client id from the service name: {service_name}"
                return await self._launch_application_for_service(
                    app_id,
                    service_name,
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
        user_info = UserInfo.from_context(context)

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

                    if user_info.check_permission(
                        workspace_info.id, UserPermission.read
                    ) or workspace_info.owned_by(user_info):
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

    def _can_overwrite_workspace(
        self, user_info: UserInfo, existing_workspace: WorkspaceInfo
    ) -> bool:
        """
        Check if user is authorized to overwrite an existing workspace.

        Authorization is granted if any of these conditions are met:
        1. User is the root user (user_info.id == "root")
        2. User owns the workspace (id or email in owners list)
        3. User has admin permission on the workspace

        Args:
            user_info: The user attempting the overwrite
            existing_workspace: The workspace being overwritten

        Returns:
            True if authorized, False otherwise
        """
        # Root user can overwrite any workspace
        if user_info.id == "root":
            return True

        # Owner can overwrite their workspace
        if existing_workspace.owned_by(user_info):
            return True

        # User with admin permission can overwrite
        if user_info.check_permission(existing_workspace.id, UserPermission.admin):
            return True

        return False

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
        if not workspace.owned_by(user_info):
            workspace.owners.append(user_info.id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]
        logger.info("Updating workspace %s", workspace.id)

        await self._redis.hset("workspaces", workspace.id, workspace.model_dump_json())
        if self._s3_controller:
            await self._s3_controller.save_workspace_config(workspace)
        await self._event_bus.broadcast(
            workspace.id, "workspace_changed", workspace.model_dump()
        )

    async def _prepare_workspace(self, workspace_info: WorkspaceInfo):
        """Prepare the workspace."""
        errors = {}
        try:
            # Skip preparation for anonymous workspaces
            if not workspace_info.id.startswith(ANONYMOUS_USER_WS_PREFIX):
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
            if errors:
                workspace_info.status = {"ready": True, "errors": errors}
                await self._redis.hset(
                    "workspaces", workspace_info.id, workspace_info.model_dump_json()
                )
            else:
                await self._set_workspace_status(
                    workspace_info.id, WorkspaceStatus.READY
                )

            await self._event_bus.broadcast(
                workspace_info.id, "workspace_ready", workspace_info.model_dump()
            )
            logger.info("Workspace %s prepared.", workspace_info.id)

        except Exception as e:
            workspace_info.status = {"ready": False, "error": str(e)}
            await self._redis.hset(
                "workspaces", workspace_info.id, workspace_info.model_dump_json()
            )
            raise

    @schema_method
    async def unload(self, context=None):
        """Unload the workspace."""
        self.validate_context(context, permission=UserPermission.admin)
        ws = context["ws"]

        try:
            if not await self._redis.hexists("workspaces", ws):
                logger.warning(f"Workspace {ws} has already been unloaded.")
                return

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
                await self._event_bus.emit(f"unload:{ws}", "Unloading workspace: " + ws)

            if not winfo.persistent:
                # For non-persistent workspaces, always clean up Redis and S3
                keys = await self._redis.keys(f"{ws}:*")
                for key in keys:
                    await self._redis.delete(key)
                if self._s3_controller:
                    await self._s3_controller.cleanup_workspace(winfo)
                await self._redis.hdel("workspaces", ws)
            else:
                # For persistent workspaces, clean up service data but preserve workspace info
                if self._s3_controller:
                    keys = await self._redis.keys(f"{ws}:*")
                    for key in keys:
                        await self._redis.delete(key)
                    await self._s3_controller.cleanup_workspace(winfo)
                    # For persistent workspaces, register with activity manager for intelligent cleanup
                    # System workspaces are protected and won't be registered
                    if await self._activity_manager.register_for_cleanup(ws):
                        logger.debug(
                            f"Registered persistent workspace {ws} for activity-based cleanup"
                        )
                    else:
                        # Either activity manager disabled or protected workspace - delete immediately
                        await self._redis.hdel("workspaces", ws)
                else:
                    logger.warning(
                        f"Skipping cleanup of persistent workspace {ws} because S3 controller is not available"
                    )

            try:
                pass
            except KeyError:
                pass

            await self._close_workspace(winfo)
        except Exception as e:
            logger.error(f"Failed to unload workspace: {e}")
            raise

    async def _close_workspace(self, workspace_info: WorkspaceInfo):
        """Archive the workspace."""
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

        await self._event_bus.broadcast(
            workspace_info.id, "workspace_unloaded", workspace_info.model_dump()
        )
        logger.info("Workspace %s unloaded.", workspace_info.id)

    @schema_method
    async def check_status(self, context=None) -> dict:
        """Check the status of the workspace."""
        assert context is not None, "Context cannot be None"
        self.validate_context(context, permission=UserPermission.read)
        ws = context["ws"]

        # Check if workspace exists
        try:
            workspace_info = await self.load_workspace_info(ws, load=False)
        except KeyError:
            return {"status": "not_found", "error": f"Workspace {ws} not found"}

        # Check workspace status
        if not workspace_info.status:
            return {"status": "loading"}

        if workspace_info.status.get("ready"):
            errors = workspace_info.status.get("errors", {})
            return {"status": "ready", "errors": errors if errors else None}
        elif workspace_info.status.get("error"):
            return {"status": "error", "error": workspace_info.status["error"]}
        else:
            return {"status": "loading"}

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
        user_info = UserInfo.from_context(context)
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
                user_info = UserInfo.from_context(context)
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
            # IMPORTANT: The workspace manager MUST be public to allow initial client connections
            # before workspace context is established. Each method implements its own permission checks.
            "config": {
                "require_context": True,
                "visibility": "public",
            },
            "echo": self.echo,
            "log": self.log,
            "log_event": self.log_event,
            "record_usage": self.record_usage,
            "get_usage_summary": self.get_usage_summary,
            "get_event_stats": self.get_event_stats,
            "get_events": self.get_events,
            "register_interceptor": self.register_interceptor,
            "list_interceptors": self.list_interceptors,
            "remove_interceptor": self.remove_interceptor,
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
            "parse_token": self.parse_token,
            "create_workspace": self.create_workspace,
            "delete_workspace": self.delete_workspace,
            "bookmark": self.bookmark,
            "unbookmark": self.unbookmark,
            "get_workspace_info": self.get_workspace_info,
            "get_summary": self.get_summary,
            "ping": self.ping_client,
            "cleanup": self.cleanup,
            "check_status": self.check_status,
            "set_env": self.set_env,
            "get_env": self.get_env,
            "subscribe": self.subscribe,
            "unsubscribe": self.unsubscribe,
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

        # First, try to disconnect the RPC connection if it exists
        try:
            connection_key = f"{cws}/{client_id}"
            if connection_key in RedisRPCConnection._connections:
                connection = RedisRPCConnection._connections[connection_key]
                await connection.disconnect("Client deleted by admin")
                logger.debug(f"Disconnected RPC connection for {connection_key}")
        except Exception as e:
            logger.warning(
                f"Failed to disconnect RPC connection for {cws}/{client_id}: {e}"
            )

        # Define a pattern to match all services for the given client_id in the current workspace
        pattern = f"services:*|*:{cws}/{client_id}:*@*"

        # Ensure the pattern is secure
        assert not any(
            char in pattern for char in "{}"
        ), "Query pattern contains invalid characters."

        keys = []
        cursor = 0
        while True:
            cursor, batch = await self._redis.scan(
                cursor=cursor, match=pattern, count=500
            )
            keys.extend(batch)
            if cursor == 0:
                break
        logger.info(f"Removing {len(keys)} services for client {client_id} in {cws}")
        for key in keys:
            await self._redis.delete(key)

        await self._event_bus.broadcast(
            cws, "client_disconnected", {"id": client_id, "workspace": cws}
        )

        # Clean up orphaned patterns
        try:
            await self._event_bus.cleanup_orphaned_patterns()
        except Exception as e:
            logger.warning(f"Failed to clean up orphaned patterns: {e}")

        if unload:
            if await self._redis.hexists("workspaces", cws):
                if user_info.is_anonymous and cws == user_info.get_workspace():
                    logger.info(
                        f"Unloading workspace {cws} for anonymous user (while deleting client {client_id})"
                    )
                    # Only unload if no other clients remain; force-unloading while
                    # other clients are still connected would disconnect them before
                    # they receive the client_disconnected event via Redis broadcast.
                    await self.unload_if_empty(context=context)
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

    @property
    def activity_manager_stats(self) -> Dict[str, Any]:
        """Get activity manager statistics for monitoring and debugging."""
        return self._activity_manager.get_stats()

    async def _set_workspace_status(self, workspace_id: str, status: str):
        """Set the status of a workspace."""
        try:
            # Try to get workspace info from Redis first
            workspace_info_data = await self._redis.hget("workspaces", workspace_id)
            if workspace_info_data is None:
                raise KeyError(f"Workspace {workspace_id} not found in Redis")

            workspace_info = WorkspaceInfo.model_validate(
                json.loads(workspace_info_data.decode())
            )

            # Update status based on the status type
            if status == WorkspaceStatus.READY:
                workspace_info.status = {"ready": True, "errors": None}
            elif status == WorkspaceStatus.LOADING:
                workspace_info.status = {"ready": False, "status": "loading"}
            elif status == WorkspaceStatus.UNLOADING:
                workspace_info.status = {"ready": False, "status": "unloading"}
            elif status == WorkspaceStatus.CLOSED:
                workspace_info.status = {"ready": False, "status": "closed"}
            else:
                workspace_info.status = {"ready": False, "status": status}

            # Save back to Redis
            await self._redis.hset(
                "workspaces", workspace_id, workspace_info.model_dump_json()
            )
            await self._event_bus.broadcast(
                workspace_id,
                "workspace_status_changed",
                {"workspace": workspace_id, "status": status},
            )
        except Exception as e:
            logger.error(f"Failed to set workspace status: {e}")
            raise

    async def _select_service_by_load(
        self, keys: List[bytes], criteria: str = "min"
    ) -> bytes:
        """Select service by client load.

        Args:
            keys: List of Redis keys for services
            criteria: Selection criteria ('min' or 'max')

        Returns:
            bytes: The Redis key of the selected service
        """
        if len(keys) == 1:
            return keys[0]

        service_values = []

        for key in keys:
            try:
                # Get service info
                service_data = await self._redis.hgetall(key)
                service_info = ServiceInfo.from_redis_dict(service_data)

                # Get client load
                from hypha.core import RedisRPCConnection

                workspace = service_info.id.split("/")[0]
                client_id = service_info.id.split("/")[1].split(":")[0]
                load = RedisRPCConnection.get_client_load(workspace, client_id)

                service_values.append((key, load, service_info))

            except Exception as e:
                logger.warning(f"Failed to get load for service {key}: {e}")
                continue

        if not service_values:
            logger.warning(
                f"No services responded for load check, falling back to random selection"
            )
            return random.choice(keys)

        # Apply selection criteria
        if criteria == "min":
            selected_key, value, _ = min(service_values, key=lambda x: float(x[1]))
            logger.debug(f"Selected service with minimum load: {value}")
        elif criteria == "max":
            selected_key, value, _ = max(service_values, key=lambda x: float(x[1]))
            logger.debug(f"Selected service with maximum load: {value}")
        else:
            raise ValueError(f"Unknown selection criteria: {criteria}")

        return selected_key

    async def _select_native_service(self, keys: List[bytes]) -> bytes:
        """Select a native service (client connected to the same Hypha server).

        Args:
            keys: List of Redis keys for services

        Returns:
            bytes: The Redis key of the selected native service, or random fallback
        """
        if len(keys) == 1:
            return keys[0]

        native_services = []
        all_services = []

        for key in keys:
            try:
                # Get service info
                service_data = await self._redis.hgetall(key)
                service_info = ServiceInfo.from_redis_dict(service_data)

                # Extract workspace and client_id from service ID
                workspace = service_info.id.split("/")[0]
                client_id = service_info.id.split("/")[1].split(":")[0]

                all_services.append((key, service_info))

                # Check if this client is local/native to this server
                if self._event_bus.is_local_client(workspace, client_id):
                    native_services.append((key, service_info))
                    logger.debug(f"Found native service: {service_info.id}")

            except Exception as e:
                logger.warning(f"Failed to check if service is native {key}: {e}")
                continue

        if native_services:
            # Prefer native services - pick randomly among them
            selected_key, service_info = random.choice(native_services)
            logger.info(f"Selected native service: {service_info.id}")
            return selected_key
        elif all_services:
            # No native services found, fall back to random selection
            selected_key, service_info = random.choice(all_services)
            logger.info(
                f"No native services found, selected random service: {service_info.id}"
            )
            return selected_key
        else:
            # Fallback to random if we couldn't process any services
            logger.warning(
                "Could not process any services, falling back to random selection"
            )
            return random.choice(keys)

    async def _select_service_by_function(
        self, keys: List[bytes], criteria: str, function_name: str, timeout: float = 2.0
    ) -> bytes:
        """Select service by calling a function on each service and applying selection criteria.

        Args:
            keys: List of Redis keys for services
            criteria: Selection criteria ('min', 'max', 'first_success', etc.)
            function_name: Name of the function to call on each service
            timeout: Timeout for function calls

        Returns:
            bytes: The Redis key of the selected service
        """
        if len(keys) == 1:
            return keys[0]

        # Get service APIs and call the specified function
        service_values = []

        for key in keys:
            try:
                # Get service info
                service_data = await self._redis.hgetall(key)
                service_info = ServiceInfo.from_redis_dict(service_data)

                # Get the service API for custom functions
                service_api = await self._rpc.get_remote_service(
                    service_info.id, {"timeout": timeout}
                )

                # Check if the service has the required function
                if not hasattr(service_api, function_name):
                    logger.warning(
                        f"Service {service_info.id} does not have function '{function_name}', skipping"
                    )
                    continue

                # Call the function
                func = getattr(service_api, function_name)
                result = await func()

                service_values.append((key, result, service_info))

            except Exception as e:
                logger.warning(f"Failed to call {function_name} on service {key}: {e}")
                continue

        if not service_values:
            logger.warning(
                f"No services responded to {function_name}, falling back to random selection"
            )
            return random.choice(keys)

        # Apply selection criteria
        if criteria == "min":
            # Select service with minimum value
            selected_key, value, _ = min(service_values, key=lambda x: float(x[1]))
            logger.debug(f"Selected service with minimum {function_name}: {value}")
        elif criteria == "max":
            # Select service with maximum value
            selected_key, value, _ = max(service_values, key=lambda x: float(x[1]))
            logger.debug(f"Selected service with maximum {function_name}: {value}")
        elif criteria == "first_success":
            # Select first service that successfully responded
            selected_key, value, _ = service_values[0]
            logger.debug(f"Selected first successful service: {value}")
        else:
            raise ValueError(f"Unknown selection criteria: {criteria}")

        return selected_key
