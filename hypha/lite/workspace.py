# hypha/lite/workspace.py
"""Minimal Workspace Manager for Hypha Lite."""

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

from fakeredis import aioredis # Assuming redis type hint is sufficient
from hypha_rpc import RPC
from hypha_rpc.utils.schema import schema_method
from pydantic import BaseModel, Field

# Use core definitions where applicable
from hypha.core import (
    RedisRPCConnection,
    UserInfo,
    WorkspaceInfo,
    ServiceInfo,
    UserPermission,
    VisibilityEnum,
    ServiceTypeInfo, # Keep if types are managed
    TokenConfig, # Import TokenConfig from core
)
# Auth functions might be stable enough to reuse from core
from hypha.core.auth import create_scope, parse_token, generate_presigned_token

from hypha.utils import EventBus, random_id

# Forward declaration for RedisStore type hint if needed, or use Any
# from hypha.lite.store import RedisStore

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("lite-workspace")
logger.setLevel(LOGLEVEL)

SERVICE_SUMMARY_FIELD = ["id", "name", "type", "description", "config"]

# Regex for validating keys remains useful
_allowed_characters = re.compile(r"^[a-zA-Z0-9-_/|*]*$")

def validate_key_part(key_part: str):
    """Ensure key parts only contain safe characters."""
    if not _allowed_characters.match(key_part):
        raise ValueError(f"Invalid characters in query part: {key_part}")

# --- GetServiceConfig (if needed by get_service) ---
class GetServiceConfig(BaseModel):
    mode: Optional[str] = Field(
        None, description="Mode for selecting service: 'random', 'first', 'last', 'exact'."
    )
    timeout: Optional[float] = Field(
        10.0, description="Timeout for fetching the service."
    )
    case_conversion: Optional[str] = Field(
        None, description="Case conversion for service keys ('camel', 'snake')."
    )

# --- WorkspaceManager ---
class WorkspaceManager:
    """Minimal Workspace Manager."""
    def __init__(
        self,
        store: Any, # Use 'Any' or forward ref 'RedisStore'
        redis: aioredis.FakeRedis, # Or aioredis.Redis
        root_user: UserInfo,
        event_bus: EventBus,
        server_info: dict,
        client_id: str,
        # Removed SQL, S3, Artifact, ServerApp controllers, service search
    ):
        self._redis = redis
        self._store = store
        self._initialized = False
        self._rpc: RPC | None = None
        self._root_user = root_user
        self._event_bus = event_bus
        self._server_info = server_info
        self._client_id = client_id # The manager's own client ID
        # Removed controller references

    def get_client_id(self):
        """Get the client ID of this workspace manager instance."""
        assert self._client_id, "Manager client ID not set."
        return self._client_id

    async def setup(self, service_id="default", service_name="Lite Workspace Manager"):
        """Setup the manager by registering its own service."""
        if self._initialized:
            return self._rpc

        assert self._client_id, "Manager client ID must be provided."
        # The manager connects to the event bus but listens in the 'public' workspace
        # for global requests like create_workspace, list_workspaces etc.
        # It needs its own RPC client to register itself.
        self._rpc = self._create_rpc(
             client_id=self._client_id,
             workspace="public", # Manager operates primarily in public scope
             silent=False # Manager service should be visible
        )

        management_service = self._create_management_service(service_id, service_name)
        # Register the manager's interface using its own RPC client
        await self._rpc.register_service(
            management_service,
            {"notify": False}, # Don't notify about the manager service itself
        )
        logger.info(f"Lite Workspace Manager service registered as {self._client_id}:{service_id} in public workspace.")
        self._initialized = True
        return self._rpc

    # --- Core Workspace Methods ---

    def _validate_workspace_id(self, name: str, is_core: bool = False):
        """Validate the workspace name."""
        if not name:
            raise ValueError("Workspace name must not be empty.")
        if is_core: # Allow 'public', 'root' etc.
            return name
        # Basic validation for user workspaces
        pattern = re.compile(r"^[a-z0-9-]+$") # Lowercase, numbers, hyphens only
        if not pattern.match(name):
            raise ValueError(
                f"Invalid workspace name: '{name}'. Use lowercase letters, numbers, hyphens."
            )
        # Maybe enforce length?
        return name

    @schema_method
    async def create_workspace(
        self,
        config: Union[dict, WorkspaceInfo],
        overwrite: bool = False,
        context: Optional[dict] = None,
        is_core: bool = False, # Flag for core workspaces like 'public', 'root'
    ):
        """Create a new workspace in Redis."""
        self.validate_context(context, UserPermission.admin) # Only root/admin can create via manager

        if isinstance(config, WorkspaceInfo):
            config = config.model_dump()

        ws_id = config.get("id")
        if not ws_id:
            raise ValueError("Workspace 'id' must be provided.")

        ws_name = config.get("name", ws_id)
        user_info = UserInfo.model_validate(context["user"]) # Creator info

        # Validate ID unless it's a core workspace
        self._validate_workspace_id(ws_id, is_core=is_core)

        # Check existence
        exists = await self._redis.hexists("workspaces", ws_id)
        if exists and not overwrite:
            raise RuntimeError(f"Workspace '{ws_id}' already exists.")
        if exists and overwrite:
             logger.warning(f"Overwriting existing workspace: {ws_id}")
             # Consider deleting existing related keys? Maybe too destructive.

        # Prepare WorkspaceInfo object
        workspace = WorkspaceInfo(
            id=ws_id,
            name=ws_name,
            description=config.get("description", f"Workspace {ws_name}"),
            persistent=config.get("persistent", False),
            owners=config.get("owners", [user_info.id]), # Default owner is creator
            read_only=config.get("read_only", False),
            config=config.get("config", {}),
            service_types={}, # Initialize empty
            status={"ready": True} # Mark as ready immediately in lite version
        )

        # Ensure owners list is clean
        workspace.owners = list(set(o.strip() for o in workspace.owners if o and o.strip()))
        if not workspace.owners: # Ensure at least one owner
             workspace.owners = [self._root_user.id]

        # Persist to Redis
        await self._redis.hset("workspaces", workspace.id, workspace.model_dump_json())
        logger.info(f"Workspace '{workspace.id}' created/updated in Redis.")

        # Emit event
        await self._event_bus.emit("workspace_loaded", workspace.model_dump())
        return workspace.model_dump()

    @schema_method
    async def delete_workspace(
        self,
        workspace: str = Field(..., description="Workspace name"),
        context: Optional[dict] = None,
    ):
        """Remove a workspace from Redis."""
        self.validate_context(context, UserPermission.admin)
        ws = workspace # The workspace to delete

        workspace_info = await self.load_workspace_info(ws, load=False) # Don't try S3 etc.
        if not workspace_info:
            raise KeyError(f"Workspace '{ws}' not found.")

        # Basic check: don't delete core workspaces easily
        if ws in ["public", self._root_user.get_workspace()]:
             logger.warning(f"Attempt to delete core workspace '{ws}' denied.")
             raise PermissionError(f"Cannot delete core workspace: {ws}")

        # Delete associated services (use pattern matching)
        service_pattern = f"services:*|*:{ws}/*:*@*"
        service_keys = await self._redis.keys(service_pattern)
        if service_keys:
            logger.info(f"Deleting {len(service_keys)} services associated with workspace {ws}...")
            async with self._redis.pipeline(transaction=False) as pipe:
                 for key in service_keys:
                     pipe.delete(key)
                 await pipe.execute()

        # Delete workspace entry itself
        deleted_count = await self._redis.hdel("workspaces", ws)

        if deleted_count > 0:
            logger.info(f"Workspace '{ws}' deleted successfully.")
            await self._event_bus.emit("workspace_deleted", workspace_info.model_dump())
        else:
             logger.warning(f"Workspace '{ws}' was not found in Redis hash for deletion.")


    async def load_workspace_info(
        self, workspace: str, load: bool = False, increment_counter: bool = True
    ) -> WorkspaceInfo | None:
        """Load workspace info from Redis (minimal version)."""
        ws_data = await self._redis.hget("workspaces", workspace)
        if ws_data:
            try:
                return WorkspaceInfo.model_validate(json.loads(ws_data.decode()))
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON for workspace: {workspace}")
                # Potentially delete corrupted entry?
                # await self._redis.hdel("workspaces", workspace)
                return None
            except Exception as e:
                 logger.error(f"Failed to validate workspace data for {workspace}: {e}")
                 return None
        elif load: # 'load' might imply creating if it's an expected anonymous one
             if workspace == "ws-anonymous":
                  # Auto-create anonymous workspace if not found
                  logger.info("Auto-creating missing 'ws-anonymous' workspace.")
                  anon_info = await self.create_workspace(
                       config=WorkspaceInfo(id="ws-anonymous", name="Anonymous Workspace", persistent=True, owners=["root"], read_only=True),
                       context={"user": self._root_user.model_dump(), "ws": "public"},
                       is_core=True
                  )
                  return WorkspaceInfo.model_validate(anon_info)
             else:
                  return None # Don't load from anywhere else in lite
        else:
            return None # Not found and load=False


    @schema_method
    async def get_workspace_info(
        self, workspace: str = Field(None, description="Workspace name"), context=None
    ) -> dict:
        """Get the workspace info."""
        self.validate_context(context, permission=UserPermission.read)
        ws_to_get = workspace or context["ws"] # Use context ws if not specified
        workspace_info = await self.load_workspace_info(ws_to_get, load=False) # Don't auto-load/create here
        if not workspace_info:
            raise KeyError(f"Workspace '{ws_to_get}' not found.")
        # Check permission for the target workspace
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(workspace_info.id, UserPermission.read):
             raise PermissionError(f"Permission denied for workspace {workspace_info.id}")

        return workspace_info.model_dump()

    @schema_method
    async def list_workspaces(
        self,
        match: Optional[dict] = Field(None, description="Match pattern (not implemented in lite)"),
        offset: int = Field(0, description="Offset"),
        limit: int = Field(100, description="Limit"),
        context=None,
    ) -> List[Dict[str, Any]]:
        """List workspaces user has access to."""
        self.validate_context(context, permission=UserPermission.read) # Basic permission for listing
        user_info = UserInfo.model_validate(context["user"])

        if offset < 0: offset = 0
        if limit < 1: limit = 1
        if limit > 1000: limit = 1000 # Safety limit

        # Scan Redis hash 'workspaces'
        cursor = 0
        all_workspaces = []
        while True:
            cursor, keys = await self._redis.hscan("workspaces", cursor, count=100) # Scan in batches
            for ws_id_bytes in keys:
                ws_id = ws_id_bytes.decode('utf-8')
                ws_data = await self._redis.hget("workspaces", ws_id)
                if ws_data:
                    try:
                        ws_info = WorkspaceInfo.model_validate(json.loads(ws_data.decode()))
                        # Check permission for each workspace
                        if user_info.check_permission(ws_info.id, UserPermission.read):
                             # Apply matching if needed (simplified)
                             if match:
                                 # Basic match on top-level fields
                                 if not all(getattr(ws_info, k, None) == v for k, v in match.items()):
                                     continue
                             all_workspaces.append({
                                 "id": ws_info.id,
                                 "name": ws_info.name,
                                 "description": ws_info.description,
                                 "persistent": ws_info.persistent,
                                 "read_only": ws_info.read_only,
                                 "owners": ws_info.owners,
                             })
                    except Exception as e:
                        logger.error(f"Error processing workspace '{ws_id}' during list: {e}")
            if cursor == 0:
                break

        # Apply pagination after filtering
        return all_workspaces[offset : offset + limit]

    # --- Core Service Methods ---

    @schema_method
    async def register_service(
        self,
        service: Union[ServiceInfo, dict] = Field(..., description="Service info"),
        config: Optional[dict] = Field(None, description="Registration options"),
        context: Optional[dict] = None,
    ):
        """Register a new service in Redis."""
        self.validate_context(context, UserPermission.read_write) # Need write perm to register
        ws = context["ws"]
        client_id = context["from"] # The client registering the service

        if isinstance(service, dict):
            # Basic validation before creating ServiceInfo
            if "id" not in service: raise ValueError("Service 'id' is required.")
            if "/" not in service["id"]: service["id"] = f"{ws}/{client_id}:{service['id']}" # Normalize ID format
            if ":" not in service["id"]: raise ValueError("Service 'id' must be in 'workspace/client:service' or 'client:service' format.")
            service = ServiceInfo.model_validate(service)
        elif isinstance(service, ServiceInfo):
             # Ensure ID format is correct if passed as object
             if "/" not in service.id: service.id = f"{ws}/{client_id}:{service.id}"
             if ":" not in service.id: raise ValueError("Service 'id' must be in 'workspace/client:service' format.")
        else:
            raise TypeError("Invalid type for 'service', expected dict or ServiceInfo.")

        # Validate workspace matches context
        service_ws = service.id.split("/")[0]
        if service_ws != ws:
             raise ValueError(f"Service ID workspace '{service_ws}' does not match context workspace '{ws}'.")

        # Extract parts for key construction
        service.config.workspace = ws # Ensure workspace in config
        service_client_part, service_name_part = service.id.split("/")[1].split(":", 1)
        service.app_id = service.app_id or "*" # Default app_id

        # Add creator info
        user_info = UserInfo.model_validate(context["user"])
        service.config.created_by = {"id": user_info.id} # Simplified creator info

        # Construct the Redis key
        visibility = service.config.visibility.value if isinstance(service.config.visibility, VisibilityEnum) else str(service.config.visibility)
        service_type = service.type or "*"
        key = f"services:{visibility}|{service_type}:{service.id}@{service.app_id}"

        # Check for existing service with the *exact* same key
        # In lite, we overwrite simply. More complex singleton logic removed.
        # existing_keys = await self._redis.keys(f"services:*|*:{service.id}@{service.app_id}")
        # if existing_keys:
        #      logger.warning(f"Overwriting existing service(s) for key pattern: services:*|*:{service.id}@{service.app_id}")
        #      async with self._redis.pipeline(transaction=False) as pipe:
        #           for k in existing_keys: pipe.delete(k)
        #           await pipe.execute()

        # Persist service info
        await self._redis.hset(key, mapping=service.to_redis_dict())
        logger.info(f"Service registered/updated: {key}")

        # Emit event
        is_built_in = ":built-in@" in key
        event_data = service.model_dump()
        if is_built_in:
            await self._event_bus.emit("client_connected", {"id": client_id, "workspace": ws}) # Or client_updated?
        else:
            await self._event_bus.emit("service_added", event_data) # Or service_updated?

        return event_data # Return the registered service info


    @schema_method
    async def unregister_service(
        self,
        service_id: str = Field(..., description="Full service ID (ws/client:svc@app or similar)"),
        context: Optional[dict] = None,
    ):
        """Unregister a service by deleting its key."""
        self.validate_context(context, UserPermission.read_write)
        ws = context["ws"]

        # Attempt to find the exact key - requires full ID usually
        # Example: services:public|*|public/client-abc:my-service@*
        # Need visibility, type, workspace, client, service, app_id
        # This is complex. Let's simplify: find based on service_id pattern and context ws.

        if "/" not in service_id or ":" not in service_id:
             raise ValueError("Provide full service ID like 'workspace/client:service' or 'client:service'")

        if "/" not in service_id: service_id = f"{ws}/{service_id}" # Add context ws if missing

        target_ws = service_id.split("/")[0]
        if target_ws != ws:
             # Allow root/admin to unregister from other workspaces? Check permission.
             target_perm = UserPermission.admin # Require admin for cross-workspace unregister
             user_info = UserInfo.model_validate(context["user"])
             if not user_info.check_permission(target_ws, target_perm):
                  raise PermissionError(f"Admin permission required to unregister service in workspace '{target_ws}'.")
        else:
            # Already validated write permission for context['ws']
            pass

        # Find keys matching the service ID pattern in the target workspace
        app_id = "*"
        if "@" in service_id:
            service_id_part, app_id = service_id.split("@", 1)
        else:
            service_id_part = service_id

        # Pattern uses the potentially cross-workspace service_id_part
        pattern = f"services:*|*:{service_id_part}@{app_id}"
        keys_to_delete = await self._redis.keys(pattern)

        if not keys_to_delete:
            raise KeyError(f"Service matching pattern '{pattern}' not found.")

        deleted_count = 0
        service_info_for_event = None # Store one for the event

        async with self._redis.pipeline(transaction=False) as pipe:
             for key_bytes in keys_to_delete:
                 if service_info_for_event is None: # Get info of first deleted service
                     try:
                         service_data = await self._redis.hgetall(key_bytes)
                         service_info_for_event = ServiceInfo.from_redis_dict(service_data).model_dump()
                     except Exception as e:
                         logger.error(f"Failed to get service info before deleting key {key_bytes}: {e}")
                 pipe.delete(key_bytes)
                 deleted_count += 1
             await pipe.execute()

        logger.info(f"Unregistered {deleted_count} service(s) matching pattern: {pattern}")

        # Emit event (using info from the first deleted service if available)
        if service_info_for_event:
             is_built_in = ":built-in@" in service_info_for_event["id"] # Check based on stored ID string
             if is_built_in:
                 client_id = service_info_for_event["id"].split("/")[1].split(":")[0]
                 await self._event_bus.emit("client_disconnected", {"id": client_id, "workspace": target_ws})
             else:
                 await self._event_bus.emit("service_removed", service_info_for_event)
        else:
            logger.warning(f"Could not retrieve service info for event emission during unregister.")

        return {"deleted_count": deleted_count}


    @schema_method
    async def get_service_info(
        self,
        service_id: str = Field(..., description="Service ID (e.g., ws/client:svc, client:svc@app)"),
        config: Optional[dict] = Field(None, description="Options (mode: random, first, last, exact)"),
        context: Optional[dict] = None,
    ):
        """Get service info from Redis."""
        self.validate_context(context, UserPermission.read)
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        # Normalize service ID
        if "/" not in service_id: service_id = f"{ws}/{service_id}"
        if ":" not in service_id: raise ValueError("Service ID must include client:service")

        target_ws = service_id.split("/")[0]
        app_id = "*"
        if "@" in service_id: service_id, app_id = service_id.split("@", 1)

        # Check permission for target workspace
        if not user_info.check_permission(target_ws, UserPermission.read):
             # Allow access if it's a public service even without workspace read perm
             # This requires checking the key prefix, making the query more complex.
             # Simplification: Require read permission for the workspace first.
             raise PermissionError(f"Read permission denied for workspace '{target_ws}'.")

        # Pattern to find the service key(s)
        pattern = f"services:*|*:{service_id}@{app_id}"
        keys = await self._redis.keys(pattern)

        if not keys:
            raise KeyError(f"Service not found matching pattern: {pattern}")

        # Handle multiple matches based on mode
        mode = (config or {}).get("mode", "exact") # Default to exact
        selected_key = None
        if mode == "exact":
            if len(keys) > 1:
                key_strings = [k.decode('utf-8', errors='ignore') for k in keys[:5]]
                raise ValueError(f"Multiple services found for '{service_id}@{app_id}' ({len(keys)} found, e.g., {key_strings}). Use mode 'random', 'first', or 'last'.")
            elif len(keys) == 1:
                selected_key = keys[0]
        elif mode == "random":
            selected_key = random.choice(keys)
        elif mode == "first":
            selected_key = min(keys) # Lexicographical first
        elif mode == "last":
            selected_key = max(keys) # Lexicographical last
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'random', 'first', 'last', or 'exact'.")

        if not selected_key:
             raise KeyError(f"Could not select a service key for pattern {pattern} with mode {mode}")

        # Retrieve and parse data
        service_data = await self._redis.hgetall(selected_key)
        if not service_data: # Should not happen if key exists, but check anyway
             raise KeyError(f"Service data not found for key: {selected_key.decode('utf-8', errors='ignore')}")

        service_info = ServiceInfo.from_redis_dict(service_data)

        # Final visibility check (redundant if workspace permission implies visibility access)
        # if service_info.config.visibility != VisibilityEnum.public and \
        #    not user_info.check_permission(target_ws, UserPermission.read):
        #     raise PermissionError(f"Permission denied for non-public service '{service_info.id}'.")

        return service_info # Return the Pydantic model instance

    @schema_method
    async def list_services(
        self,
        query: Optional[Union[dict, str]] = Field(None, description="Query dict or string pattern"),
        context: Optional[dict] = None,
    ):
        """List services based on query/pattern."""
        self.validate_context(context, UserPermission.read)
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        # --- Parse Query (Simplified from core) ---
        # Default to listing services in the current workspace if query is None
        if query is None:
            query = {"workspace": ws}

        if isinstance(query, str):
             # Very basic string parsing: treat as workspace ID or service name part
             if "/" in query or ":" in query or "@" in query:
                  # Treat as specific service ID pattern (needs more robust parsing)
                  logger.warning(f"Complex string query '{query}' parsing not fully supported in lite. Trying basic match.")
                  # Attempt to extract parts - this is fragile
                  parts = query.split('@')[0] # Ignore app_id for pattern search?
                  if '/' in parts:
                       workspace_part, client_service_part = parts.split('/', 1)
                  else:
                       workspace_part = ws # Assume current workspace
                       client_service_part = parts
                  if ':' in client_service_part:
                      client_part, service_part = client_service_part.split(':', 1)
                  else: # Assume it's a service name?
                      client_part = "*"
                      service_part = client_service_part
                  parsed_query = {"workspace": workspace_part, "client_id": client_part, "service_id": service_part}
             else: # Assume it's just a workspace name
                  parsed_query = {"workspace": query}
             query = parsed_query

        # Apply defaults and construct pattern
        target_ws = query.get("workspace", ws)
        visibility = query.get("visibility", "*")
        type_filter = query.get("type", "*")
        client_id = query.get("client_id", "*")
        service_id = query.get("service_id", "*")
        app_id = query.get("app_id", "*")

        # --- Permission Check ---
        can_list_protected = False
        if target_ws == "*":
            # Listing across all workspaces - only public allowed generally
            visibility = "public"
            # Need to iterate through all workspaces user *can* see for protected services? Too complex.
            # Simplification: target_ws="*" only lists public services globally.
        else:
             # Check permission for the specific target workspace
             if not user_info.check_permission(target_ws, UserPermission.read):
                  # If no read permission, can only see public services in that workspace
                  if visibility != "public":
                       logger.debug(f"User {user_info.id} lacks read permission for {target_ws}, limiting list to public services.")
                       visibility = "public"
             else:
                  # Has read permission, can see protected based on query
                  can_list_protected = (visibility == "*" or visibility == "protected")


        # --- Construct Redis Pattern ---
        # services:{visibility}|{type}:{workspace}/{client_id}:{service_id}@{app_id}
        validate_key_part(visibility)
        validate_key_part(type_filter)
        validate_key_part(target_ws)
        validate_key_part(client_id)
        validate_key_part(service_id)
        validate_key_part(app_id)
        pattern = f"services:{visibility}|{type_filter}:{target_ws}/{client_id}:{service_id}@{app_id}"

        keys = await self._redis.keys(pattern)

        # If target_ws was '*' but user has access to current ws, add those too?
        # Example: query={"workspace": "*"} should list public global + public/protected in current ws?
        # Let's keep it simple: '*' only lists public global for now.

        services = []
        for key in keys:
            service_data = await self._redis.hgetall(key)
            if not service_data: continue
            try:
                service_info = ServiceInfo.from_redis_dict(service_data)
                # Final check if we accidentally got a protected one we shouldn't see
                # (Shouldn't happen if pattern/perms are right, but as a safeguard)
                is_public = service_info.config.visibility == VisibilityEnum.public
                actual_ws = service_info.config.workspace
                if not is_public and not user_info.check_permission(actual_ws, UserPermission.read):
                     # This indicates a potential logic error or race condition
                     logger.warning(f"Filtering out service {service_info.id} due to permission mismatch.")
                     continue

                services.append(service_info.model_dump()) # Return as dict
            except Exception as e:
                 logger.error(f"Failed to parse service data for key {key.decode('utf-8', errors='ignore')}: {e}")

        return services


    # --- Core Client Methods ---

    @schema_method
    async def list_clients(
        self,
        workspace: str = Field(None, description="Workspace name"),
        context: Optional[dict] = None,
    ):
        """List clients in a workspace by finding built-in services."""
        self.validate_context(context, UserPermission.read)
        ws = workspace or context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        if not user_info.check_permission(ws, UserPermission.read):
             raise PermissionError(f"Read permission denied for workspace '{ws}'.")

        # Find all built-in services in the target workspace
        pattern = f"services:*|*:*/*:built-in@*" # Adjust pattern if type 'built-in' is used
        keys = await self._redis.keys(pattern)

        clients = {} # Use dict to store unique clients {client_id: user_info}
        for key in keys:
            key_str = key.decode('utf-8', errors='ignore')
            # Extract workspace and client ID from key: services:vis|type:ws/client:svc@app
            try:
                parts = key_str.split(':')
                ws_client_part = parts[2].split('/')
                service_ws = ws_client_part[0]
                client_id = ws_client_part[1] # This is the client ID

                if service_ws != ws: continue # Filter for the target workspace

                if client_id not in clients:
                     # Get user info from the service config if possible
                     service_data = await self._redis.hgetall(key)
                     creator_info = "unknown"
                     if service_data:
                          try:
                               service_info = ServiceInfo.from_redis_dict(service_data)
                               if service_info.config and service_info.config.created_by:
                                    creator_info = service_info.config.created_by.get("id", "unknown")
                          except Exception:
                               pass # Ignore parsing errors for creator info
                     clients[client_id] = {"id": f"{ws}/{client_id}", "user": creator_info}
            except Exception as e:
                 logger.warning(f"Could not parse client ID from key '{key_str}': {e}")

        return list(clients.values())

    @schema_method
    async def ping_client(
        self,
        client_id: str = Field(..., description="Client ID (e.g., ws/client or client)"),
        timeout: float = Field(5, description="Timeout in seconds"),
        context: Optional[dict] = None,
    ):
        """Ping a client's built-in service."""
        self.validate_context(context, UserPermission.read) # Need read to ping
        ws = context["ws"]

        # Normalize client ID
        if "/" not in client_id: client_id = f"{ws}/{client_id}"

        target_ws = client_id.split("/")[0]
        # Check permission for target workspace
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(target_ws, UserPermission.read):
             raise PermissionError(f"Read permission denied for workspace '{target_ws}'.")

        try:
            # Use the manager's RPC to send the ping request
            # Need to get the proxy for the specific client's built-in service
            svc = await self._rpc.get_remote_service(
                f"{client_id}:built-in", # Target service ID
                {"timeout": timeout}
            )
            # Assuming the built-in service implements a 'ping' method
            response = await svc.ping("ping") # Standard ping payload
            if response == "pong":
                 return "pong"
            else:
                 logger.warning(f"Client {client_id} returned unexpected ping response: {response}")
                 return f"Unexpected response: {response}"
        except asyncio.TimeoutError:
            logger.warning(f"Ping timeout for client {client_id}")
            raise TimeoutError(f"Ping timeout for client {client_id}")
        except Exception as e:
            logger.error(f"Failed to ping client {client_id}: {e}")
            raise ConnectionError(f"Failed to ping client {client_id}: {e}") from e


    async def delete_client(
        self,
        client_id: str,
        workspace: str,
        user_info: UserInfo, # Info of user initiating deletion (for checks)
        unload: bool = False,
        context: Optional[dict] = None, # Context for permission check
    ):
        """Delete all services for a client and potentially unload workspace."""
        self.validate_context(context, UserPermission.admin) # Require admin to delete clients

        # Normalize client ID
        if "/" in client_id:
            _ws, client_id_part = client_id.split("/", 1)
            if _ws != workspace:
                raise ValueError(f"Client ID workspace '{_ws}' does not match target workspace '{workspace}'.")
            client_id = client_id_part # Use only the ID part

        validate_key_part(client_id)
        validate_key_part(workspace)

        # Find and delete all service keys for this client in this workspace
        pattern = f"services:*|*:{workspace}/{client_id}:*@*"
        keys = await self._redis.keys(pattern)

        deleted_count = 0
        if keys:
            logger.info(f"Deleting {len(keys)} services for client '{client_id}' in workspace '{workspace}'...")
            async with self._redis.pipeline(transaction=False) as pipe:
                for key in keys:
                    pipe.delete(key)
                    deleted_count += 1
                await pipe.execute()

            # Emit disconnect event only if services were actually deleted
            await self._event_bus.emit("client_disconnected", {"id": f"{workspace}/{client_id}", "workspace": workspace})
        else:
             logger.warning(f"No services found for client '{client_id}' in workspace '{workspace}' to delete.")

        # Attempt to unload workspace if requested and no clients remain
        if unload:
            await self.unload_if_empty(workspace=workspace, context=context)

        return {"deleted_count": deleted_count}


    async def unload_if_empty(self, workspace: str, context=None):
        """Unload workspace if no clients remain."""
        self.validate_context(context, UserPermission.admin) # Require admin

        # Check for remaining clients (built-in services)
        pattern = f"services:*|*:*/*:built-in@*" # Search only in the target workspace
        keys = await self._redis.keys(pattern)
        clients_remain = False
        for key in keys:
             key_str = key.decode('utf-8', errors='ignore')
             try:
                 ws_part = key_str.split(':')[2].split('/')[0]
                 if ws_part == workspace:
                      clients_remain = True
                      break
             except IndexError:
                  continue

        if not clients_remain:
             logger.info(f"Workspace '{workspace}' is empty, proceeding to unload.")
             try:
                 # Call the main unload logic (which is missing in this minimal version)
                 # await self.unload(workspace=workspace, context=context) # Needs unload implementation
                 # Minimal unload: just delete from Redis hash
                 await self._redis.hdel("workspaces", workspace)
                 await self._event_bus.emit("workspace_unloaded", {"id": workspace})
                 logger.info(f"Workspace '{workspace}' marked as unloaded (removed from hash).")
             except Exception as e:
                 logger.error(f"Failed to unload empty workspace '{workspace}': {e}", exc_info=True)
        else:
             logger.debug(f"Workspace '{workspace}' not empty, skipping unload.")

    # --- Utility and RPC ---

    def _create_rpc(
        self,
        client_id: str,
        workspace: str,
        default_context=None,
        user_info: UserInfo = None, # Defaults to root if None
        manager_id: str | None = None, # ID of the manager to connect TO
        silent=True,
    ):
        """Create an RPC client instance."""
        assert "/" not in client_id
        logger.debug(f"Creating RPC for client {client_id} in workspace {workspace}")
        effective_user = user_info or self._root_user

        connection = RedisRPCConnection(
            event_bus=self._event_bus,
            workspace=workspace,
            client_id=client_id,
            user_info=effective_user,
            manager_id=manager_id or self._client_id, # Default to self if not specified
        )
        rpc = RPC(
            connection,
            workspace=workspace,
            client_id=client_id,
            default_context=default_context,
            server_base_url=self._server_info.get("public_base_url"),
            silent=silent,
        )
        rpc.register_codec(
            {"name": "pydantic-model", "type": BaseModel, "encoder": lambda x: x.model_dump()}
        )
        return rpc

    def validate_context(self, context: dict | None, permission: UserPermission):
        """Validate user permission from context."""
        if context is None:
            raise PermissionError("Context is required for this operation.")
        if "user" not in context or "ws" not in context:
            raise ValueError("Context must include 'user' and 'ws'.")

        try:
            user_info = UserInfo.model_validate(context["user"])
            workspace = context["ws"]
            if not user_info.check_permission(workspace, permission):
                raise PermissionError(
                     f"User '{user_info.id}' lacks '{permission.name}' permission for workspace '{workspace}'."
                )
        except Exception as e:
             logger.error(f"Context validation failed: {e}", exc_info=True)
             raise PermissionError(f"Context validation failed: {e}")


    # --- Management Service Definition ---
    def _create_management_service(self, service_id, service_name):
        """Define the methods exposed by the workspace manager service."""
        interface = {
            "id": service_id,
            "name": service_name,
            "description": "Lite services for managing workspaces and services.",
            "config": {
                "require_context": True, # Most methods need context
                "visibility": "public", # Manager reachable in public workspace
            },
            # Workspace methods
            "create_workspace": self.create_workspace,
            "delete_workspace": self.delete_workspace,
            "get_workspace_info": self.get_workspace_info,
            "list_workspaces": self.list_workspaces,
            # Service methods
            "register_service": self.register_service,
            "unregister_service": self.unregister_service,
            "get_service_info": self.get_service_info,
            "list_services": self.list_services,
            # Client methods
            "list_clients": self.list_clients,
            "ping_client": self.ping_client,
            # "delete_client": self.delete_client, # Maybe keep internal? Expose via admin-utils?
            # Removed: log, echo, token methods, bookmarking, service types, get_service (proxying)
        }
        # Add server info to config for potential use by clients
        interface["config"].update(self._server_info)
        return interface 