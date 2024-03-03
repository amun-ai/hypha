"""Provide the ImJoy core API interface."""
import asyncio
import inspect
import io
import json
import logging
import sys
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fakeredis import aioredis
import msgpack
from pydantic import (  # pylint: disable=no-name-in-module
    BaseModel,
    EmailStr,
    PrivateAttr,
    constr,
)

from hypha.utils import EventBus

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("core")
logger.setLevel(logging.INFO)


class TokenConfig(BaseModel):
    """Represent a token configuration."""

    scopes: List[str]
    expires_in: Optional[float] = None
    email: Optional[EmailStr] = None
    parent_client: Optional[str] = None


class VisibilityEnum(str, Enum):
    """Represent the visibility of the workspace."""

    public = "public"
    protected = "protected"


class StatusEnum(str, Enum):
    """Represent the status of a component."""

    ready = "ready"
    initializing = "initializing"
    not_initialized = "not_initialized"


class ServiceConfig(BaseModel):
    """Represent service config."""

    visibility: VisibilityEnum = VisibilityEnum.protected
    require_context: Union[Tuple[str], List[str], bool] = False
    workspace: str = None
    flags: List[str] = []


class ServiceInfo(BaseModel):
    """Represent service."""

    config: ServiceConfig
    id: str
    name: str
    type: str
    description: Optional[constr(max_length=256)] = ""
    docs: Optional[Dict[str, constr(max_length=1024)]] = {}

    class Config:
        """Set the config for pydantic."""

        extra='allow'

    def is_singleton(self):
        """Check if the service is singleton."""
        return "single-instance" in self.config.flags


class UserInfo(BaseModel):
    """Represent user info."""

    id: str
    roles: List[str]
    is_anonymous: bool
    email: Optional[EmailStr] = None
    parent: Optional[str] = None
    scopes: Optional[List[str]] = None  # a list of workspace
    expires_at: Optional[float] = None
    _metadata: Dict[str, Any] = PrivateAttr(
        default_factory=lambda: {}
    )  # e.g. s3 credential

    def get_metadata(self, key=None) -> Dict[str, Any]:
        """Return the metadata."""
        if key:
            return self._metadata.get(key)
        return self._metadata

    def set_metadata(self, key, value):
        """Set the metadata."""
        self._metadata[key] = value


class ClientInfo(BaseModel):
    """Represent service."""

    id: str
    parent: Optional[str] = None
    name: Optional[str] = None
    workspace: str
    services: List[ServiceInfo] = []
    user_info: UserInfo


class RDF(BaseModel):
    """Represent resource description file object."""

    name: str
    id: str
    tags: List[str]
    documentation: Optional[str] = None
    covers: Optional[List[str]] = None
    badges: Optional[List[str]] = None
    authors: Optional[List[Dict[str, str]]] = None
    attachments: Optional[Dict[str, List[Any]]] = None
    config: Optional[Dict[str, Any]] = None
    type: str
    format_version: str = "0.2.1"
    version: str = "0.1.0"
    links: Optional[List[str]] = None
    maintainers: Optional[List[Dict[str, str]]] = None
    license: Optional[str] = None
    git_repo: Optional[str] = None
    source: Optional[str] = None

    class Config:
        """Set the config for pydantic."""

        extra='allow'


class ApplicationInfo(RDF):
    """Represent an application."""

    pass


class WorkspaceInfo(BaseModel):
    """Represent a workspace."""

    name: str
    persistent: bool
    owners: List[str]
    visibility: VisibilityEnum
    read_only: bool = False
    description: Optional[str] = None
    icon: Optional[str] = None
    covers: Optional[List[str]] = None
    docs: Optional[str] = None
    allow_list: Optional[List[str]] = None
    deny_list: Optional[List[str]] = None
    applications: Optional[Dict[str, RDF]] = {}  # installed applications
    interfaces: Optional[Dict[str, List[Any]]] = {}


class RedisRPCConnection:
    """Represent a redis connection."""

    def __init__(
        self,
        redis: aioredis.FakeRedis,
        workspace: str,
        client_id: str,
        user_info: UserInfo,
    ):
        """Intialize Redis RPC Connection"""
        self._redis = redis
        self._handle_message = None
        self._subscribe_task = None
        self._workspace = workspace
        self._client_id = client_id
        assert "/" not in client_id
        self._user_info = user_info.model_dump()

    def on_message(self, handler: Callable):
        """Setting message handler."""
        self._handle_message = handler
        self._is_async = inspect.iscoroutinefunction(handler)
        self._subscribe_task = asyncio.ensure_future(self._subscribe_redis())

    async def emit_message(self, data: Union[dict, bytes]):
        """Send message."""
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
        """Disconnect."""
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
            msg = await pubsub.get_message(timeout=10)
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
        """Initialize the event bus."""
        super().__init__(logger)
        self._redis = redis
        self._local_event_bus = EventBus(logger)

    async def init(self):
        """Setup the event bus."""
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._loop = loop
        loop.create_task(self._subscribe_redis())
        await self._ready

    def on_local(self, *args, **kwargs):
        """Register a callback for a local event."""
        return self._local_event_bus.on(*args, **kwargs)

    def off_local(self, *args, **kwargs):
        """Unregister a callback for a local event."""
        return self._local_event_bus.off(*args, **kwargs)

    def once_local(self, *args, **kwargs):
        """Register a callback for a local event and remove it once triggered."""
        return self._local_event_bus.once(*args, **kwargs)

    def emit(self, event_type, data=None, target: str = "global"):
        """Emit an event to the event bus."""
        assert target in ["local", "global", "remote"]
        if target != "remote":
            self._local_event_bus.emit(event_type, data)
        if target is None or target == "global":
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
                msg = await pubsub.get_message(timeout=10)
                if msg and msg.get("type") == "message":
                    data = json.loads(msg["data"].decode())
                    super().emit(data["type"], data["data"])
        except Exception as exp:
            self._ready.set_exception(exp)
