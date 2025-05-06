"""A minimal, scalable state store based on Redis."""

# Imports
import asyncio
import json
import logging
import os
import sys
import datetime
from typing import List, Union, Any

from pydantic import BaseModel
from fastapi import Header, Cookie, FastAPI

from hypha_rpc import RPC
from hypha_rpc.utils.schema import schema_method
from aiocache.backends.redis import RedisCache
from aiocache.serializers import PickleSerializer

from hypha import __version__
from hypha.core import (
    RedisEventBus,
    RedisRPCConnection,
    ServiceInfo,
    UserInfo,
    WorkspaceInfo,
    UserPermission,
    VisibilityEnum,
    ServiceTypeInfo,
    TokenConfig
)
from hypha.core.activity import ActivityTracker
from hypha.core.auth import (
    create_scope,
    parse_token,
    generate_anonymous_user,
    generate_presigned_token,
)
from hypha.utils import random_id
from hypha.lite.workspace import WorkspaceManager # Only import manager

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("lite-store")
logger.setLevel(LOGLEVEL)

# --- RedisStore Class Definition ---
class RedisStore:
    """Represent a minimal redis store."""
    # pylint: disable=too-many-instance-attributes, too-many-public-methods

    def __init__(
        self,
        app: FastAPI,
        server_id: str | None = None,
        public_base_url: str | None = None,
        local_base_url: str | None = None,
        redis_uri: str | None = None,
        reconnection_token_life_time: float = 2 * 24 * 60 * 60,
        activity_check_interval: float = 10,
    ):
        """Initialize the minimal redis store."""
        self._app = app
        self._public_base_url = public_base_url
        self._local_base_url = local_base_url
        self._public_services: List[ServiceInfo] = []
        self._public_types: List[ServiceTypeInfo] = []
        self._ready = False
        self._workspace_manager: WorkspaceManager | None = None
        self._websocket_server = None
        self._server_id = server_id or random_id(readable=True)
        self._manager_id = "lite-manager-" + self._server_id
        self._reconnection_token_life_time = reconnection_token_life_time
        self._activity_check_interval = activity_check_interval

        self._http_anonymous_user = UserInfo(
            id="http-anonymous",
            is_anonymous=True,
            email=None,
            parent=None,
            roles=["anonymous"],
            scope=create_scope("public#r", current_workspace="public"),
            expires_at=None,
        )
        self._anonymous_workspace_info: WorkspaceInfo | None = None

        self._server_info = {
            "server_id": self._server_id,
            "hypha_version": __version__,
            "public_base_url": self._public_base_url,
            "local_base_url": self._local_base_url,
        }
        logger.info("Lite Server info: %s", self._server_info)
        self.loop = None

        if redis_uri and redis_uri.startswith("redis://"):
            from redis import asyncio as aioredis
            self._redis = aioredis.from_url(redis_uri)
            logger.info(f"Connecting to Redis at {redis_uri}")
        else:
            from fakeredis import aioredis
            logger.warning("No Redis URI provided, using fakeredis (in-memory).")
            self._redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")

        self._redis_cache = RedisCache(serializer=PickleSerializer())
        self._redis_cache.client = self._redis

        self._root_user: UserInfo | None = None
        self._event_bus = RedisEventBus(self._redis)

        self._tracker: ActivityTracker | None = None
        self._tracker_task: asyncio.Task | None = None

        self._public_workspace_interface = None
        self._root_workspace_interface = None

    def set_websocket_server(self, websocket_server):
        """Set the websocket server."""
        self._websocket_server = websocket_server

    @schema_method
    def kickout_client(self, workspace: str, client_id: str, code: int, reason: str):
        """Force disconnect a client."""
        if not self._websocket_server:
            raise RuntimeError("Websocket server not configured.")
        return self._websocket_server.force_disconnect(
            workspace, client_id, code, reason
        )

    def get_redis(self):
        """Get the underlying redis client."""
        return self._redis

    def get_event_bus(self):
        """Get the event bus."""
        return self._event_bus

    async def setup_root_user(self) -> UserInfo:
        """Setup the root user."""
        self._root_user = UserInfo(
            id="root",
            is_anonymous=False,
            email=None,
            parent=None,
            roles=["admin"],
            scope=create_scope("*#a"),
            expires_at=None,
        )
        logger.info("Root user setup.")
        return self._root_user

    async def load_or_create_workspace(self, user_info: UserInfo, workspace: str) -> WorkspaceInfo:
        """Load or create a workspace, ensuring permissions."""
        if not self._workspace_manager: # Added safety check
             raise RuntimeError("Workspace manager not initialized during load_or_create_workspace")

        if workspace is None:
            workspace = user_info.get_workspace()

        assert workspace != "*", "Dynamic workspace ('*') is not allowed here."

        if workspace == "ws-anonymous" and self._anonymous_workspace_info:
            return self._anonymous_workspace_info

        workspace_info = await self.get_workspace_info(workspace, load=True)

        if not workspace_info:
            if workspace == user_info.get_workspace():
                if not user_info.check_permission(workspace, UserPermission.read):
                    raise PermissionError(f"User {user_info.id} cannot access workspace {workspace}")
                logger.info(f"Creating workspace {workspace} for user {user_info.id}")
                workspace_info_dict = await self._workspace_manager.create_workspace(
                    config={
                        "id": workspace, "name": workspace,
                        "description": f"Workspace for user {user_info.id}",
                        "persistent": not user_info.is_anonymous,
                        "owners": [user_info.id], "read_only": user_info.is_anonymous,
                    },
                    overwrite=False,
                    context={"user": self._root_user.model_dump(), "ws": "public"}
                )
                workspace_info = WorkspaceInfo.model_validate(workspace_info_dict)
            else:
                 raise KeyError(f"Workspace {workspace} does not exist or is not accessible.")
        else:
            if not user_info.check_permission(workspace, UserPermission.read):
                 raise PermissionError(f"User {user_info.id} cannot access existing workspace {workspace}")
            logger.debug(f"Workspace {workspace} exists, access granted.")

        if workspace == "ws-anonymous":
            self._anonymous_workspace_info = workspace_info

        return workspace_info

    def get_root_user(self) -> UserInfo:
        """Get the root user."""
        if not self._root_user:
            raise RuntimeError("Root user not initialized.")
        return self._root_user

    async def check_and_cleanup_servers(self):
        """Minimal check for servers."""
        server_ids = await self.list_servers()
        logger.info("Connected lite servers: %s", server_ids)
        if self._server_id in server_ids and len(server_ids) > 1:
             logger.warning(f"Server ID conflict: {self._server_id}")

    async def _clear_client_services(self, workspace: str, client_id: str):
        """Clear services for a specific client."""
        pattern = f"services:*|*:{workspace}/{client_id}:*@*"
        keys = await self._redis.keys(pattern)
        if keys:
            logger.info(f"Removing {len(keys)} services for disconnected client {client_id} in {workspace}")
            async with self._redis.pipeline(transaction=False) as pipe:
                for key in keys: pipe.delete(key)
                await pipe.execute()

    async def init(self, reset_redis: bool):
        """Initialize the minimal store."""
        self.loop = asyncio.get_running_loop()
        if reset_redis:
            logger.warning("RESETTING ALL REDIS DATA!!!")
            await self._redis.flushall()

        await self._event_bus.init()

        self._tracker = ActivityTracker(check_interval=self._activity_check_interval)
        self._tracker_task = asyncio.create_task(self._tracker.monitor_entities())
        RedisRPCConnection.set_activity_tracker(self._tracker)
        self._tracker.register_entity_removed_callback(self._remove_tracker_entity)

        await self.setup_root_user()
        # await self.check_and_cleanup_servers()

        self._workspace_manager = WorkspaceManager(
            store=self, redis=self._redis, root_user=self._root_user,
            event_bus=self._event_bus, server_info=self._server_info,
            client_id=self._manager_id,
        )
        await self._workspace_manager.setup()

        try:
            await self.register_workspace(
                 WorkspaceInfo.model_validate({
                     "id": self._root_user.get_workspace(), "name": "Root Workspace",
                     "persistent": True, "owners": ["root"], "read_only": False,
                 }), overwrite=False, is_core=True
            )
        except RuntimeError: logger.warning("Root workspace already exists.")

        try:
            await self.register_workspace(
                WorkspaceInfo.model_validate({
                    "id": "public", "name": "Public Workspace",
                    "persistent": True, "owners": ["root"], "read_only": True,
                }), overwrite=False, is_core=True
            )
        except RuntimeError: logger.warning("Public workspace already exists.")

        try:
            self._anonymous_workspace_info = await self.register_workspace(
                WorkspaceInfo.model_validate({
                    "id": "ws-anonymous", "name": "Anonymous Workspace",
                    "description": "Shared workspace for anonymous users",
                    "persistent": True, "owners": ["root"], "read_only": True,
                }), overwrite=False, is_core=True
            )
        except RuntimeError:
            self._anonymous_workspace_info = await self.get_workspace_info("ws-anonymous")

        await self._register_root_services()

        api = await self.get_public_api()
        logger.info("Registering explicitly added public services...")
        for service in self._public_services:
            try:
                logger.info(f"Registering public service: {service.id}")
                await self._workspace_manager.register_service(
                    service=service, config={"notify": True},
                    context={"user": self._root_user.model_dump(), "ws": "public"}
                )
            except Exception:
                logger.exception(f"Failed to register public service: {service}")
                raise

        self._ready = True
        await self.get_event_bus().emit_local("startup")
        servers = await self.list_servers()

        logger.info(f"Lite Server initialized with server id: {self._server_id}")
        logger.info(f"Currently connected lite servers: {servers}")

    async def _remove_tracker_entity(self, entity_id: str, entity_type: str):
        """Callback when activity tracker removes an entity."""
        if entity_type != "client": return
        try:
            workspace, client_id = entity_id.split("/")
            logger.info(f"Activity Tracker: Removing client {client_id} from {workspace}")
            await self._workspace_manager.delete_client(
                client_id=client_id, workspace=workspace, user_info=self._root_user,
                unload=True, context={"user": self._root_user.model_dump(), "ws": workspace}
            )
        except Exception as e:
            logger.error(f"Error removing client {entity_id} via tracker: {e}", exc_info=True)

    async def _register_root_services(self):
        """Register minimal root services."""
        async with self.get_workspace_interface(
            user_info=self._root_user, workspace=self._root_user.get_workspace(),
            client_id=self._server_id, silent=False,
        ) as root_api:
            self._root_workspace_interface = root_api
            await root_api.register_service(
                {
                    "id": "admin-utils", "name": "Admin Utilities (Lite)",
                    "config": {"visibility": "protected", "require_context": False},
                    "list_servers": self.list_servers,
                    "kickout_client": self.kickout_client,
                }
            )
            logger.info("Registered admin-utils service in root workspace.")

    @schema_method
    async def list_servers(self):
        """List all connected lite servers."""
        pattern = f"services:*|*:public/*:built-in@*"
        keys = await self._redis.keys(pattern)
        clients = set()
        for key in keys:
            try:
                key_str = key.decode("utf-8")
                parts = key_str.split(":")
                client_part = parts[2].split("/")[1]
                clients.add(client_part)
            except (IndexError, UnicodeDecodeError):
                logger.warning(f"Could not parse server ID from key: {key}")
        return list(clients)

    def get_activity_tracker(self):
        """Get the activity tracker instance."""
        return self._tracker

    async def get_public_api(self):
        """Get the public API interface."""
        # This needs careful handling of the context manager lifecycle
        if self._public_workspace_interface is None:
             logger.debug("Creating public workspace interface...")
             # Store the context manager itself to manage its exit later
             self._public_cm = self.get_workspace_interface(
                 user_info=self._root_user, workspace="public",
                 client_id=self._server_id, silent=False,
             )
             self._public_workspace_interface = await self._public_cm.__aenter__()
             logger.debug("Public workspace interface created.")
        return self._public_workspace_interface

    async def client_exists(self, client_id: str, workspace: str):
        """Check if a client exists."""
        assert client_id and "/" not in client_id, f"Invalid client_id format: {client_id}"
        assert workspace, "Workspace must be provided."
        pattern = f"services:*|*:*/*:{client_id}:built-in@*"
        keys = await self._redis.keys(pattern)
        return bool(keys)

    async def remove_client(self, client_id: str, workspace: str, user_info: UserInfo, unload: bool):
        """Remove a client."""
        if not self._workspace_manager: raise RuntimeError("Workspace manager not initialized.")
        logger.debug(f"Requesting WorkspaceManager to remove client {client_id} from {workspace}")
        context = {"user": self._root_user.model_dump(), "ws": workspace}
        return await self._workspace_manager.delete_client(
            client_id, workspace, user_info, unload=unload, context=context
        )

    async def get_user_workspace(self, user_id: str):
        """Get a user's workspace info."""
        ws_id = f"ws-user-{user_id}"
        return await self.get_workspace_info(ws_id)

    async def parse_user_token(self, token: str) -> UserInfo:
        """Parse a client token."""
        try:
            user_info = parse_token(token)
            key = "revoked_token:" + token
            if await self._redis.exists(key): raise Exception("Token has been revoked")
            if "admin" in user_info.roles: user_info.scope.workspaces["*"] = UserPermission.admin
            return user_info
        except Exception as e:
            logger.error(f"Token parsing failed: {e}")
            raise ValueError(f"Invalid or expired token: {e}")

    async def login_optional(self, authorization: str = Header(None), access_token: str = Cookie(None)) -> UserInfo:
        """Return user info or anonymous user."""
        token = authorization or access_token
        if token:
            try:
                user_info = await self.parse_user_token(token)
                if user_info.scope.current_workspace is None:
                     user_info.scope.current_workspace = user_info.get_workspace()
                return user_info
            except ValueError:
                 logger.warning("Invalid token provided, falling back to anonymous user.")
                 return self._http_anonymous_user
        else:
            return self._http_anonymous_user

    async def workspace_exists(self, workspace_name: str) -> bool:
        """Check if a workspace exists."""
        return await self._redis.hexists("workspaces", workspace_name)

    async def register_workspace(self, workspace_info: Union[dict, WorkspaceInfo], overwrite=False, is_core=False) -> WorkspaceInfo:
        """Register a workspace."""
        if not self._workspace_manager: raise RuntimeError("Workspace manager not initialized.")
        context = {"user": self._root_user.model_dump(), "ws": "public"}
        info_dict = await self._workspace_manager.create_workspace(
            config=workspace_info, overwrite=overwrite, context=context, is_core=is_core,
        )
        return WorkspaceInfo.model_validate(info_dict)

    def get_manager_id(self) -> str:
        """Get the client ID of the workspace manager."""
        return self._manager_id

    def get_workspace_interface(self, user_info: UserInfo, workspace: str, client_id: str | None = None, timeout: int = 10, silent: bool = True) -> 'WorkspaceInterfaceContextManager':
        """Get a context manager for interacting with a workspace."""
        assert workspace, "Workspace name is required"
        assert user_info and isinstance(user_info, UserInfo), "User info is required"
        effective_client_id = client_id or "anon-if-" + random_id(readable=False)
        rpc = self.create_rpc(workspace, user_info, client_id=effective_client_id, silent=silent)
        # Pass 'self' as the store argument to the context manager
        return WorkspaceInterfaceContextManager(
            rpc=rpc, store=self, workspace=workspace, user_info=user_info, timeout=timeout
        )

    def create_rpc(self, workspace: str, user_info: UserInfo, client_id: str, default_context=None, silent=True) -> RPC:
        """Create an RPC object."""
        assert "/" not in client_id
        logger.debug(f"Creating RPC for client {client_id} in workspace {workspace}")
        assert user_info is not None, "User info is required"
        connection = RedisRPCConnection(
            event_bus=self._event_bus, workspace=workspace, client_id=client_id,
            user_info=user_info, manager_id=self._manager_id,
        )
        rpc = RPC(
            connection=connection, client_id=client_id, default_context=default_context,
            workspace=workspace, server_base_url=self._public_base_url, silent=silent,
        )
        rpc.register_codec({"name": "pydantic-model", "type": BaseModel, "encoder": lambda x: x.model_dump()})
        return rpc

    async def get_workspace_info(self, workspace: str, load: bool = False) -> WorkspaceInfo | None:
        """Return the workspace information."""
        if not self._workspace_manager:
            logger.warning("Workspace manager not ready during get_workspace_info call.")
            ws_data = await self._redis.hget("workspaces", workspace)
            return WorkspaceInfo.model_validate(json.loads(ws_data.decode())) if ws_data else None
        try:
            return await self._workspace_manager.load_workspace_info(workspace, load=load)
        except KeyError:
            return None

    def register_public_service(self, service: dict):
        """Register a service definition."""
        assert not self._ready, "Cannot register public service after store is initialized"
        if "id" not in service or not service.get("name"):
            raise ValueError("Public service definition requires 'id' and 'name'.")
        service["config"] = service.get("config", {})
        assert isinstance(service["config"], dict), "service.config must be a dictionary"
        service["config"]["workspace"] = "public"
        service["config"]["visibility"] = service["config"].get("visibility", "public")
        if service["config"]["visibility"] not in ["public", "protected"]:
             logger.warning(f"Public service '{service['id']}' has invalid visibility, defaulting to 'public'.")
             service["config"]["visibility"] = "public"
        try:
            formatted_service = ServiceInfo.model_validate(service)
            self._public_services.append(formatted_service)
            logger.info(f"Queued public service for registration: {formatted_service.id}")
            return {"id": formatted_service.id, "workspace": "public"}
        except Exception as e:
             logger.error(f"Failed to validate public service definition {service.get('id')}: {e}")
             raise

    def is_ready(self) -> bool:
        """Check if the store is initialized."""
        return self._ready

    async def is_alive(self):
        """Check if the workspace manager is responsive."""
        if not self._ready or not self._workspace_manager or not self._workspace_manager._rpc:
             return False
        try:
            # Use the manager's own RPC to ping its service interface
            manager_service = await self._workspace_manager._rpc.get_remote_service(
                f"public/{self._manager_id}:default", {"timeout": 1}
            )
            # Basic check: can we get the service proxy?
            return manager_service is not None
        except Exception:
            logger.warning("Liveness check failed: Workspace manager ping timed out.")
            return False

    async def teardown(self):
        """Teardown the minimal server store."""
        self._ready = False
        logger.info("Tearing down the lite redis store...")
        if self._tracker_task:
            self._tracker_task.cancel()
            try: await self._tracker_task
            except asyncio.CancelledError: logger.info("Activity tracker task successfully cancelled.")
            except Exception as e: logger.error(f"Error during activity tracker task cancellation: {e}")

        # Exit context managers for public/root interfaces if they were entered
        if hasattr(self, '_public_cm') and self._public_cm:
            try: await self._public_cm.__aexit__(None, None, None)
            except Exception as e: logger.error(f"Error exiting public CM: {e}")
        # Need similar logic for root CM if used

        if self._workspace_manager and self._workspace_manager._rpc:
            try: await self._workspace_manager._rpc.disconnect(); logger.info("Workspace manager RPC disconnected.")
            except Exception as e: logger.error(f"Error disconnecting workspace manager RPC: {e}")

        # Cleanup server clients (best effort)
        try:
            if self._root_workspace_interface:
                 # Assuming root_api object was stored, get client ID from its RPC
                 root_client_id = self._root_workspace_interface.rpc.get_client_info()["id"]
                 await self.remove_client(root_client_id, self._root_user.get_workspace(), self._root_user, unload=False)
                 logger.info(f"Removed server client {root_client_id} from root workspace.")

            if self._public_workspace_interface:
                 public_client_id = self._public_workspace_interface.rpc.get_client_info()["id"]
                 await self.remove_client(public_client_id, "public", self._root_user, unload=False)
                 logger.info(f"Removed server client {public_client_id} from public workspace.")
        except Exception as e:
            logger.error(f"Error removing server client representations during teardown: {e}")


        if self._websocket_server:
            websockets = self._websocket_server.get_websockets()
            logger.info(f"Closing {len(websockets)} websocket connections.")
            # Create tasks for closing to avoid blocking teardown
            close_tasks = [asyncio.create_task(ws.close(code=1001, reason="Server Shutting Down")) for ws in websockets.values()]
            # Wait briefly for tasks to complete, ignore errors
            await asyncio.gather(*close_tasks, return_exceptions=True)


        await self._event_bus.stop()
        logger.info("Lite store teardown complete.")


# --- WorkspaceInterfaceContextManager (Moved to end) ---
class WorkspaceInterfaceContextManager:
    """Workspace interface context manager (adapted for lite)."""

    def __init__(self, rpc: RPC, store: Any, workspace: str, user_info: UserInfo, timeout: int = 10):
        self._rpc = rpc
        self._timeout = timeout
        self._wm_proxy = None # Store the proxy here
        self._store = store # Keep reference to the Lite RedisStore if needed
        self._workspace = workspace
        self._user_info = user_info

    async def __aenter__(self):
        try:
            # Get the proxy to the remote manager service for this workspace context
            # The get_manager_service() method on the RPC object should handle finding the manager
            self._wm_proxy = await self._rpc.get_manager_service({"timeout": self._timeout})
            if self._wm_proxy is None:
                 raise RuntimeError(f"Failed to obtain workspace manager proxy for workspace '{self._workspace}'.")

            # Augment the proxy with RPC methods for convenience if needed
            # Be careful not to overwrite existing attributes on the proxy
            setattr(self._wm_proxy, 'rpc', self._rpc) # Provide access to the underlying RPC
            # setattr(self._wm_proxy, 'disconnect', self._rpc.disconnect) # Let context manager handle disconnect
            # setattr(self._wm_proxy, 'register_codec', self._rpc.register_codec)
            # setattr(self._wm_proxy, 'register_service', self._rpc.register_service) # Use proxy's own register_service

            return self._wm_proxy # Return the manager service proxy
        except Exception as e:
            # Ensure RPC is disconnected if setup fails
            await self._rpc.disconnect()
            logger.error(f"Failed to get workspace manager service for {self._workspace}: {e}")
            raise RuntimeError(f"Could not connect to workspace {self._workspace}: {e}") from e

    async def __aexit__(self, exc_type, exc, tb):
        # Disconnect the RPC connection when exiting the context
        if self._rpc:
            await self._rpc.disconnect()
            try:
                # Assuming get_client_info exists on RPC object
                client_info = self._rpc.get_client_info()
                client_id = client_info.get('id', 'unknown-client') if client_info else 'unknown-client'
                logger.debug(f"Workspace interface RPC disconnected for {self._workspace}/{client_id}")
            except Exception as e:
                 logger.error(f"Error getting client info during context exit: {e}")
