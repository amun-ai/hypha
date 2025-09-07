"""A scalable state store based on Redis."""

import asyncio
import json
import logging
import os
import sys
import datetime
from typing import List, Union
from pydantic import BaseModel
from fastapi import Header, Cookie, Request

from hypha_rpc import RPC
from hypha_rpc.utils.schema import schema_method
from starlette.routing import Mount
from aiocache.backends.redis import RedisCache
from aiocache.serializers import PickleSerializer


from hypha_rpc.rpc import RemoteException
from hypha import __version__
from hypha.core import (
    RedisEventBus,
    RedisRPCConnection,
    ServiceInfo,
    ServiceTypeInfo,
    UserInfo,
    WorkspaceInfo,
)
from hypha.core.auth import extract_token_from_scope
from hypha.core.activity import ActivityTracker
from sqlalchemy.ext.asyncio import (
    create_async_engine,
)

from hypha.core.auth import (
    create_scope,
    parse_auth_token,
    create_login_service,
    UserPermission,
    AUTH0_CLIENT_ID,
    AUTH0_DOMAIN,
    AUTH0_AUDIENCE,
    AUTH0_ISSUER,
    LOGIN_SERVICE_URL,
    set_root_token,
)
from hypha.core.workspace import WorkspaceManager
from hypha.startup import run_startup_function
from hypha.utils import random_id
from hypha.admin import AdminUtilities, setup_admin_services

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("redis-store")
logger.setLevel(LOGLEVEL)


class WorkspaceInterfaceContextManager:
    """Workspace interface context manager."""

    def __init__(self, rpc, store, workspace, user_info, timeout=10):
        self._rpc = rpc
        self._timeout = timeout
        self._wm = None
        self._store = store
        self._workspace = workspace
        self._user_info = user_info

    async def __aenter__(self):
        return await self._get_workspace_manager()

    async def __aexit__(self, exc_type, exc, tb):
        await self._wm.disconnect()

    def __await__(self):
        return self._get_workspace_manager().__await__()

    async def _get_workspace_manager(self):
        # For the special "public" workspace, do not force creation; allow lightweight access
        if self._workspace != "public":
            # Check if workspace exists or create if appropriate
            await self._store.load_or_create_workspace(self._user_info, self._workspace)
        self._wm = await self._rpc.get_manager_service({"timeout": self._timeout})
        self._wm.rpc = self._rpc
        self._wm.disconnect = self._rpc.disconnect
        self._wm.register_codec = self._rpc.register_codec
        self._wm.register_service = self._rpc.register_service
        self._wm.unregister_service = self._rpc.unregister_service
        self._wm.get_service_schema = self._rpc.get_service_schema
        self._wm.on = self._rpc.on
        self._wm.off = self._rpc.off
        self._wm.once = self._rpc.once
        self._wm.emit = self._rpc.emit
        return self._wm


class WorkspaceManagerContextManager:
    """Context manager for workspace manager operations with automatic context injection."""

    def __init__(self, store, user_info, workspace):
        self._store = store
        self._context = {"user": user_info, "ws": workspace}
        self._interface = None

    async def __aenter__(self):
        """Create and return a workspace interface with the given context."""
        self._interface = self._get_interface()
        return await self._interface.__aenter__()

    def _get_interface(self):
        user_info = UserInfo.model_validate(self._context["user"])
        workspace = self._context["ws"]
        _interface = self._store.get_workspace_interface(
            user_info,
            workspace,
            client_id=None,  # Use default anonymous client ID
            silent=True,
        )
        return _interface

    async def __aexit__(self, exc_type, exc, tb):
        """Exit the context manager."""
        if self._interface:
            await self._interface.__aexit__(exc_type, exc, tb)

    def __await__(self):
        self._interface = self._get_interface()
        return self._interface.__aenter__().__await__()


class RedisStore:
    """Represent a redis store."""

    # pylint: disable=no-self-use, too-many-instance-attributes, too-many-public-methods

    def __init__(
        self,
        app,
        server_id=None,
        public_base_url=None,
        local_base_url=None,
        redis_uri=None,
        database_uri=None,
        ollama_host=None,
        openai_config=None,
        cache_dir=None,
        enable_service_search=False,
        reconnection_token_life_time=2 * 24 * 60 * 60,
        activity_check_interval=10,
        enable_s3_for_anonymous_users=False,
        root_token=None,
        enable_admin_terminal=False,
    ):
        """Initialize the redis store."""
        self._s3_controller = None
        self._server_app_controller = None
        self._artifact_manager = None
        self._app = app
        self._codecs = {}
        self._disconnected_plugins = []
        self._public_workspace_interface = None
        self._root_workspace_interface = None
        self.public_base_url = public_base_url
        self.local_base_url = local_base_url
        self._public_services: List[ServiceInfo] = []
        self._public_types: List[ServiceTypeInfo] = []
        self._ready = False
        self._workspace_manager = None
        self._websocket_server = None
        self._cache_dir = cache_dir
        self._server_id = server_id or random_id(readable=True)
        self._manager_id = "manager-" + self._server_id
        self.reconnection_token_life_time = reconnection_token_life_time
        self._enable_service_search = enable_service_search
        self._activity_check_interval = activity_check_interval
        self._enable_s3_for_anonymous_users = enable_s3_for_anonymous_users
        # Create a fixed HTTP anonymous user. Grant read on public; set current to public
        # so we can use it to create a workspace interface safely.
        self._http_anonymous_user = UserInfo(
            id="anonymouz-http",
            is_anonymous=True,
            email=None,
            parent=None,
            roles=["anonymous"],
            scope=create_scope("ws-anonymous#r", current_workspace="ws-anonymous"),
            expires_at=None,
        )
        self._anonymous_workspace_info = None
        self._server_info = {
            "server_id": self._server_id,
            "hypha_version": __version__,
            "public_base_url": self.public_base_url,
            "local_base_url": self.local_base_url,
            "auth0_client_id": AUTH0_CLIENT_ID,
            "auth0_domain": AUTH0_DOMAIN,
            "auth0_audience": AUTH0_AUDIENCE,
            "auth0_issuer": AUTH0_ISSUER,
            "login_service_url": f"{self.public_base_url}{LOGIN_SERVICE_URL}",
        }

        logger.info("Server info: %s", self._server_info)
        self.loop = None

        self._database_uri = database_uri
        if self._database_uri is None:
            self._database_uri = (
                "sqlite+aiosqlite:///:memory:"  # In-memory SQLite for testing
            )
            logger.warning(
                "Using in-memory SQLite database for event logging, all data will be lost on restart!"
            )

        self._ollama_host = ollama_host
        if self._ollama_host is not None:
            import ollama

            self._ollama_client = ollama.AsyncClient(host=self._ollama_host)

        self._openai_config = openai_config
        if (
            self._openai_config is not None
            and self._openai_config.get("api_key") is not None
        ):
            from openai import AsyncClient

            self._openai_client = AsyncClient(**self._openai_config)
        else:
            self._openai_client = None

        self._sql_engine = create_async_engine(self._database_uri, echo=False)

        if redis_uri and redis_uri.startswith("redis://"):
            from redis import asyncio as aioredis

            # Configure connection pool for production Redis
            max_connections = int(os.environ.get("HYPHA_REDIS_MAX_CONNECTIONS", "2000"))
            health_check_interval = int(
                os.environ.get("HYPHA_REDIS_HEALTH_CHECK_INTERVAL", "30")
            )

            self._redis = aioredis.from_url(
                redis_uri,
                max_connections=max_connections,
                retry_on_timeout=True,
                retry_on_error=[],
                health_check_interval=health_check_interval,
                socket_keepalive=True,
                socket_keepalive_options={},
                socket_connect_timeout=5,
                socket_timeout=5,
                client_name=f"hypha-server:{self._server_id}:cmd",
            )

        else:  #  Create a redis server with fakeredis
            from fakeredis import aioredis

            self._redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")

        self._redis_cache = RedisCache(serializer=PickleSerializer())
        self._redis_cache.client = self._redis
        self._root_user = None
        self._root_token = root_token
        self._event_bus = RedisEventBus(self._redis)

        self._tracker = None
        self._tracker_task = None
        # self._house_keeping_task = None

        self._shared_anonymous_user = None
        self._admin_utils = None
        self._enable_admin_terminal = enable_admin_terminal
        
        # Set root token immediately if provided
        if self._root_token:
            set_root_token(self._root_token)
            logger.info(f"Root token configured in RedisStore.__init__: {self._root_token[:10]}...")

    def set_websocket_server(self, websocket_server):
        """Set the websocket server."""
        assert self._websocket_server is None, "Websocket server already set"
        self._websocket_server = websocket_server


    @schema_method
    async def kickout_client(self, workspace: str, client_id: str, code: int, reason: str):
        """Force disconnect a client."""
        return await self._websocket_server.force_disconnect(
            workspace, client_id, code, reason
        )

    def get_redis(self):
        return self._redis

    def get_sql_engine(self):
        return self._sql_engine

    def get_openai_client(self):
        return self._openai_client

    def get_ollama_client(self):
        return self._ollama_client

    def get_event_bus(self):
        """Get the event bus."""
        return self._event_bus

    def get_cache_dir(self):
        return self._cache_dir

    async def setup_root_user(self) -> UserInfo:
        """Setup the root user."""
        self._root_user = UserInfo(
            id="root",
            is_anonymous=False,
            email=None,
            parent=None,
            roles=["admin"],
            scope=create_scope("*#a", current_workspace="*"),
            expires_at=None,
        )
        return self._root_user

    def get_redis_cache(self):
        return self._redis_cache

    async def load_or_create_workspace(self, user_info: UserInfo, workspace: str):
        """Setup the workspace."""
        if workspace is None:
            workspace = user_info.get_workspace()

        assert workspace != "*", "Dynamic workspace is not allowed for this endpoint"

        # Fast path for anonymous workspace
        if workspace == "ws-anonymous" and self._anonymous_workspace_info:
            return self._anonymous_workspace_info

        # Ensure calls to store for workspace existence and permissions check
        workspace_info = await self.get_workspace_info(workspace, load=True)

        # If workspace does not exist, automatically create it
        if not workspace_info:
            if workspace == "public":
                logger.debug("Creating public workspace on demand")
                workspace_info = await self.register_workspace(
                    {
                        "id": "public",
                        "name": "public",
                        "description": "Public workspace",
                        "persistent": True,
                        "owners": [],
                        "read_only": False,
                    }
                )
            elif workspace == user_info.get_workspace():
                if not user_info.check_permission(workspace, UserPermission.read):
                    raise PermissionError(
                        f"User {user_info.id} does not have permission to access workspace {workspace}"
                    )
                logger.debug(f"Creating workspace {workspace} for user {user_info.id}")
                workspace_info = await self.register_workspace(
                    {
                        "id": workspace,
                        "name": workspace,
                        "description": f"Workspace for user {user_info.id}",
                        "persistent": not user_info.is_anonymous,
                        "owners": [user_info.id],
                        "read_only": user_info.is_anonymous
                        and not self._enable_s3_for_anonymous_users,
                    }
                )
            else:
                # Workspace doesn't exist and can't be auto-created
                raise KeyError(
                    f"Workspace {workspace} does not exist or is not accessible"
                )
        else:
            logger.debug(f"Workspace {workspace} already exists, skipping creation")
        return workspace_info

    def get_root_user(self):
        """Get the root user."""
        return self._root_user

    async def _run_startup_functions(self, startup_functions):
        """Run the startup functions in the background."""
        try:
            if startup_functions:
                for startup_function in startup_functions:
                    logger.info(f"Running startup function: {startup_function}")
                    await asyncio.sleep(0)
                    await run_startup_function(self, startup_function)
        except Exception as e:
            logger.exception(f"Error running startup function: {e}")
            # Stop the entire event loop if an error occurs
            asyncio.get_running_loop().stop()

    async def housekeeping(self):
        """Perform housekeeping tasks."""
        try:
            logger.info(f"Running housekeeping task at {datetime.datetime.now()}")
            async with self.get_workspace_interface(
                self._root_user, "ws-user-root", client_id="housekeeping"
            ) as api:
                # admin = await api.get_service("admin-utils")
                workspaces = await api.list_workspaces()
                for workspace in workspaces:
                    try:
                        logger.info(f"Cleaning up workspace {workspace.id}...")
                        summary = await api.cleanup(workspace.id)
                        if "removed_clients" in summary:
                            logger.info(
                                f"Removed {len(summary['removed_clients'])} clients from workspace {workspace.id}"
                            )
                    except Exception as e:
                        logger.exception(f"Error in housekeeping {workspace.id}: {e}")
        except Exception as exp:
            logger.error(f"Failed to run housekeeping task, error: {exp}")

    async def upgrade(self):
        """Upgrade the store."""
        current_version = await self._redis.get("hypha_version")
        if current_version == __version__:
            return

        database_change_log = []

        # Record the start of the upgrade process
        start_time = datetime.datetime.now().isoformat()
        database_change_log.append(
            {
                "time": start_time,
                "version": __version__,
                "change": "Start upgrade process",
            }
        )

        old_keys = await self._redis.keys(
            f"services:*|*:{self._root_user.get_workspace()}/workspace-client-*:*@*"
        )
        old_keys = old_keys + await self._redis.keys(
            f"services:*|*:public/workspace-client-*:*@*"
        )
        if old_keys:
            logger.info("Upgrading workspace client keys for version < 0.20.34")
            for key in old_keys:
                logger.info(
                    f"Removing workspace client service: {key}:\n{await self._redis.hgetall(key)}"
                )
                await self._redis.delete(key)

        # For versions before 0.20.34
        old_keys = await self._redis.keys("services:public:*")
        old_keys = old_keys + await self._redis.keys("services:protected:*")
        if old_keys:
            logger.info("Upgrading service keys for version < 0.20.34")
            for key in old_keys:
                key = key.decode()
                logger.info(f"Upgrading service key: {key}")
                service_data = await self._redis.hgetall(key)
                service_info = ServiceInfo.from_redis_dict(service_data)
                service_info.type = service_info.type or "*"
                new_key = key.replace(
                    "services:public:", f"services:public|{service_info.type}:"
                ).replace(
                    "services:protected:", f"services:protected|{service_info.type}:"
                )
                await self._redis.hset(new_key, mapping=service_info.to_redis_dict())
                logger.info(
                    f"Replacing service key from {key} to {new_key}:\n{await self._redis.hgetall(new_key)}"
                )
                await self._redis.delete(key)

                # Log the change
                change_time = datetime.datetime.now().isoformat()
                database_change_log.append(
                    {
                        "time": change_time,
                        "version": __version__,
                        "change": f"Upgraded service key from {key} to {new_key}",
                    }
                )

        await self._redis.set("hypha_version", __version__)

        # Record the end of the upgrade process
        end_time = datetime.datetime.now().isoformat()
        database_change_log.append(
            {"time": end_time, "version": __version__, "change": "End upgrade process"}
        )

        # Save the database change log
        await self._redis.rpush("change_log", *map(str, database_change_log))

    async def check_and_cleanup_servers(self):
        """Cleanup and check servers."""
        server_ids = await self.list_servers()
        logger.info("Connected hypha servers: %s", server_ids)
        if server_ids:
            rpc = self.create_rpc(
                "root", self._root_user, client_id="server-checker", silent=True
            )
            try:
                for server_id in server_ids:
                    try:
                        svc = await rpc.get_remote_service(
                            f"public/{server_id}:built-in", {"timeout": 2}
                        )
                        assert await svc.ping("ping") == "pong"
                    except Exception as e:
                        logger.warning(
                            f"Server {server_id} is not responding (error: {e}), cleaning up ALL its services..."
                        )
                        # Clean up ALL services from the dead server across all workspaces
                        # When a server is dead, we need to clean up everything it might have left behind
                        patterns = [
                            f"services:*|*:*/{server_id}:*@*",  # Direct server services
                            f"services:*|*:*/{server_id}-*:*@*",  # Services from server-generated clients
                            f"services:*|*:*/manager-{server_id}:*@*",  # Manager services
                        ]
                        
                        all_keys = []
                        for pattern in patterns:
                            keys = await self._redis.keys(pattern)
                            all_keys.extend(keys)
                        
                        if all_keys:
                            logger.info(f"Removing {len(all_keys)} services from dead server {server_id}")
                            # Use pipeline for efficient bulk deletion
                            pipeline = self._redis.pipeline()
                            for key in all_keys:
                                pipeline.delete(key)
                            await pipeline.execute()
                            logger.info(f"Successfully cleaned up dead server {server_id}")
                    else:
                        if server_id == self._server_id:
                            raise RuntimeError(
                                f"Server with the same id ({server_id}) is already running, please use a different server id by passing `--server-id=new-server-id` to the command line."
                            )
            except Exception as exp:
                raise exp
            finally:
                await rpc.disconnect()

    async def _clear_client_services(self, workspace: str, client_id: str):
        """Clear a workspace."""
        pattern = f"services:*|*:{workspace}/{client_id}:*@*"
        keys = await self._redis.keys(pattern)
        # Remove all services related to the server
        logger.info(
            f"Removing services for client {client_id} in workspace {workspace}"
        )
        for key in keys:
            logger.info(
                f"Removing service key: {key}:\n{await self._redis.hgetall(key)}"
            )
            await self._redis.delete(key)
    
    async def _clear_all_server_services(self):
        """Clear services registered directly by this server during graceful shutdown.
        
        Note: This only cleans up services registered by the server itself,
        NOT services registered by other clients or servers.
        """
        # Only clean up services directly registered by this server
        # We keep track of specific client IDs that belong to this server
        server_client_ids = [
            self._server_id,  # The server's main client ID
            f"manager-{self._server_id}",  # The manager service client ID
        ]
        
        # Also include the root and public workspace interface client IDs if they exist
        if self._root_workspace_interface:
            try:
                root_client_id = self._root_workspace_interface.rpc.get_client_info()["id"]
                server_client_ids.append(root_client_id)
            except:
                pass
                
        if self._public_workspace_interface:
            try:
                public_client_id = self._public_workspace_interface.rpc.get_client_info()["id"]
                server_client_ids.append(public_client_id)
            except:
                pass
        
        all_keys = []
        for client_id in server_client_ids:
            # Pattern to match services from this specific client
            pattern = f"services:*|*:*/{client_id}:*@*"
            keys = await self._redis.keys(pattern)
            all_keys.extend(keys)
        
        if all_keys:
            logger.info(f"Removing {len(all_keys)} services for server {self._server_id}")
            # Use pipeline for efficient bulk deletion
            pipeline = self._redis.pipeline()
            for key in all_keys:
                pipeline.delete(key)
            await pipeline.execute()
            logger.info(f"Successfully removed all services for server {self._server_id}")
        else:
            logger.info(f"No services found for server {self._server_id}")

    async def init(self, reset_redis, startup_functions=None):
        """Setup the store."""
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
        # Set the root token for authentication if provided
        if self._root_token:
            # Validate root token complexity
            if len(self._root_token) < 32:
                raise ValueError(
                    "Root token must be at least 32 characters long for security. "
                    "Use a strong, randomly generated token like: "
                    "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )
            # Check for simple/weak tokens
            simple_tokens = [
                "test", "root", "admin", "password", "token", "secret", 
                "123", "abc", "hypha", "default"
            ]
            token_lower = self._root_token.lower()
            for simple in simple_tokens:
                if simple in token_lower and len(self._root_token) < 64:
                    raise ValueError(
                        f"Root token appears to be too simple (contains '{simple}'). "
                        "Please use a strong, randomly generated token. "
                        "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                    )
            # Root token is already set in __init__
        await self.check_and_cleanup_servers()
        self._workspace_manager = await self.register_workspace_manager()

        try:
            await self.register_workspace(
                WorkspaceInfo.model_validate(
                    {
                        "id": self._root_user.get_workspace(),
                        "name": self._root_user.get_workspace(),
                        "description": "Root workspace",
                        "persistent": True,
                        "owners": ["root"],
                        "read_only": False,
                    }
                ),
                overwrite=False,
            )
        except RuntimeError:
            logger.warning("Root workspace already exists.")

        try:
            await self.register_workspace(
                WorkspaceInfo.model_validate(
                    {
                        "id": "public",
                        "name": "public",
                        "description": "Public workspace",
                        "persistent": True,
                        "owners": ["root"],
                        "read_only": True,
                    }
                ),
                overwrite=False,
            )
        except RuntimeError:
            logger.warning("Public workspace already exists.")

        # Setup the anonymous workspace
        try:
            self._anonymous_workspace_info = await self.register_workspace(
                WorkspaceInfo.model_validate(
                    {
                        "id": "ws-anonymous",
                        "name": "Anonymous Workspace",
                        "description": "Shared workspace for anonymous users",
                        "persistent": True,
                        "owners": ["root"],
                        "read_only": not self._enable_s3_for_anonymous_users,
                    }
                ),
                overwrite=False,
            )
        except RuntimeError:
            # Workspace already exists
            self._anonymous_workspace_info = await self.get_workspace_info(
                "ws-anonymous"
            )

        await self.upgrade()

        if self._artifact_manager:
            await self._artifact_manager.init_db()
        await self._register_root_services()
        api = await self.get_public_api()

        logger.info("Registering public services...")
        for service_type in self._public_types:
            try:
                await api.register_service_type(
                    service_type.model_dump(),
                )
            except Exception:
                logger.exception(
                    "Failed to register public service type: %s", service_type
                )
                raise
        for service in self._public_services:
            try:
                logger.info("Registering public service: %s", service.id)
                await api.register_service(
                    service.model_dump(),
                    {"notify": True},
                )
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to register public service: %s", service)
                raise

        if startup_functions:
            await self._run_startup_functions(startup_functions)
        
        # check if the login service is registered after startup functions
        # this allows startup functions to register custom login services
        try:
            await api.get_service_info("public/hypha-login")
            logger.info("Login service already registered (likely from startup function)")
        except RemoteException:
            logger.info("No custom login service found, registering default login service")
            await api.register_service(create_login_service(self))
        
        # check if the queue service is registered
        try:
            await api.get_service_info("public/queue")
        except RemoteException:
            logger.warning("Queue service is not registered, registering it now")
            # Import dynamically to avoid circular import
            from hypha.queue import create_queue_service
            await api.register_service(create_queue_service(self))
        
        self._ready = True
        await self.get_event_bus().emit_local("startup")
        servers = await self.list_servers()
        # self._house_keeping_task = asyncio.create_task(self.housekeeping())

        logger.info("Server initialized with server id: %s", self._server_id)
        logger.info("Currently connected hypha servers: %s", servers)

    async def _remove_tracker_entity(self, entity_id, entity_type):
        """Remove a client."""
        if entity_type != "client":
            return
        workspace, client_id = entity_id.split("/")
        logger.info(f"Removing client {client_id} from workspace {workspace}")
        # Remove client and potentially unload workspace if empty
        await self.remove_client(client_id, workspace, self._root_user, unload=True)

    async def _register_root_services(self):
        """Register root services."""
        self._root_workspace_interface = await self.get_workspace_interface(
            self._root_user,
            self._root_user.get_workspace(),
            client_id=self._server_id,
            silent=False,
            timeout=20,
        )
        
        # Setup admin utilities with optional terminal
        self._admin_utils = await setup_admin_services(
            self, 
            enable_terminal=self._enable_admin_terminal
        )
        
        # Register the admin service
        await self._root_workspace_interface.register_service(
            self._admin_utils.get_service_api()
        )
        
        # If terminal is enabled, setup the web interface
        if self._enable_admin_terminal and self._admin_utils.terminal:
            try:
                # Start the terminal
                await self._admin_utils.terminal.start_terminal()
                
                # Register the web interface
                from hypha.admin import register_admin_terminal_web_interface
                await register_admin_terminal_web_interface(
                    self._root_workspace_interface,
                    f"{self._root_user.get_workspace()}/admin-utils",
                    self.public_base_url,
                    self._root_user.get_workspace()
                )
                logger.info("Admin terminal web interface registered")
            except Exception as e:
                logger.error(f"Failed to register admin terminal web interface: {e}")

    @schema_method
    async def unload_workspace(self, workspace: str, wait: bool = False, timeout=10):
        """Unload a workspace."""
        context = {"user": self._root_user.model_dump(), "ws": workspace}
        await self._workspace_manager.unload(context=context)
        if wait:
            if not await self.workspace_exists(workspace):
                return
            await self._event_bus.wait_for(
                "workspace_unloaded", match={"id": workspace}, timeout=timeout
            )

    @schema_method
    async def list_servers(self):
        """List all servers."""
        pattern = f"services:*|*:public/*:built-in@*"
        keys = await self._redis.keys(pattern)
        clients = set()
        for key in keys:
            # Extract the client ID from the service key
            key_parts = key.decode("utf-8").split("/")
            client_id = key_parts[1].split(":")[0]
            clients.add(client_id)
        return list(clients)

    def get_activity_tracker(self):
        return self._tracker

    @schema_method
    async def get_metrics(self):
        """Return internal metrics snapshot for admin observability."""
        return {
            "rpc": RedisRPCConnection.get_metrics_snapshot(),
            "eventbus": RedisEventBus.get_metrics_snapshot(self._event_bus),
        }

    async def get_public_api(self):
        """Get the public API."""
        if self._public_workspace_interface is None:
            self._public_workspace_interface = await self.get_workspace_interface(
                self._root_user, "public", client_id=self._server_id, silent=False
            )
        return self._public_workspace_interface

    async def client_exists(self, client_id: str, workspace: str = None):
        """Check if a client exists."""
        assert workspace is not None, "Workspace must be provided."
        assert client_id and "/" not in client_id, "Invalid client id: " + client_id
        pattern = f"services:*|*:{workspace}/{client_id}:built-in@*"
        keys = await self._redis.keys(pattern)
        return bool(keys)

    async def remove_client(self, client_id, workspace, user_info, unload):
        """Remove a client."""
        context = {"user": self._root_user.model_dump(), "ws": workspace}
        return await self._workspace_manager.delete_client(
            client_id, workspace, user_info, unload=unload, context=context
        )

    async def get_user_workspace(self, user_id: str):
        """Get a user."""
        workspace_info = await self._redis.hget("workspaces", user_id)
        if workspace_info is None:
            return None
        workspace_info = WorkspaceInfo.model_validate(
            json.loads(workspace_info.decode())
        )
        return workspace_info

    async def get_service_type_id(self, workspace, service_id):
        """Get a service type."""
        assert (
            "/" not in service_id
        ), f"Invalid service id: {service_id}, service id cannot contain `/`"
        assert (
            "@" not in service_id
        ), f"Invalid service id: {service_id}, service id cannot contain `@`"
        assert (
            ":" not in service_id
        ), f"Invalid service id: {service_id}, service id cannot contain `:`"
        keys = await self._redis.keys(f"services:*|*:{workspace}/*:{service_id}@*")
        if keys:
            # the service key format: "services:{visibility}|{type}:{workspace}/{client_id}:{service_id}@{app_id}"
            return keys[0].decode().split(":")[1].split("|")[1]

    async def parse_user_token(self, token):
        """Parse a client token."""
        user_info = await parse_auth_token(token)
        key = "revoked_token:" + token
        if await self._redis.exists(key):
            raise Exception("Token has been revoked")
        if "admin" in user_info.roles:
            user_info.scope.workspaces["*"] = UserPermission.admin
        return user_info

    async def login_optional(
        self,
        request: Request = None,
    ):
        """Return user info or create an anonymous user.

        If authorization code is valid the user info is returned,
        If the code is invalid an anonymous user is created.
        """
        # Try to extract token using custom function if request is provided
        token = await extract_token_from_scope(request.scope)
        if token:
            user_info = await self.parse_user_token(token)
            if user_info.scope.current_workspace is None:
                user_info.scope.current_workspace = user_info.get_workspace()
            return user_info
        else:
            # Use a fixed anonymous user
            self._http_anonymous_user = UserInfo(
                id="anonymouz-http",  # Use anonymouz- prefix for consistency
                is_anonymous=True,
                email=None,
                parent=None,
                roles=["anonymous"],
                scope=create_scope("ws-anonymous#r", current_workspace="ws-anonymous"),
                expires_at=None,
            )
            return self._http_anonymous_user

    async def get_all_workspace(self):
        """Get all workspaces."""
        workspaces = await self._redis.hgetall("workspaces")
        return [
            WorkspaceInfo.model_validate(json.loads(v.decode()))
            for k, v in workspaces.items()
        ]

    async def workspace_exists(self, workspace_name: str):
        """Check if a workspace exists."""
        return await self._redis.hexists("workspaces", workspace_name)

    async def register_workspace(
        self, workspace_info: Union[dict, WorkspaceInfo], overwrite=False
    ) -> WorkspaceInfo:
        """Add a workspace."""
        info = await self._workspace_manager.create_workspace(
            workspace_info,
            overwrite=overwrite,
            context={"user": self._root_user, "ws": "public"},
        )
        return WorkspaceInfo.model_validate(info)

    def connect_to_workspace(
        self,
        workspace: str,
        client_id: str,
        user_info: UserInfo = None,
        timeout=10,
        silent=False,
    ):
        """Connect to a workspace."""
        # user get_workspace_interface
        if not client_id:
            client_id = self._server_id + "-" + random_id(readable=False)
        user_info = user_info or self._root_user
        return self.get_workspace_interface(
            user_info, workspace, client_id=client_id, timeout=timeout, silent=silent
        )

    def get_manager_id(self):
        # Return the pre-calculated manager_id directly
        # This avoids issues during startup when _workspace_manager is not yet initialized
        return self._manager_id

    async def register_workspace_manager(self):
        """Register a workspace manager."""
        manager = WorkspaceManager(
            self,
            self._redis,
            self._root_user,
            self._event_bus,
            self._server_info,
            self._manager_id,
            self._sql_engine,
            self._s3_controller,
            self._server_app_controller,
            self._artifact_manager,
            self._enable_service_search,
            self._cache_dir,
            self._enable_s3_for_anonymous_users,
            self._tracker,
        )
        await manager.setup()
        return manager

    def get_server_info(self):
        """Get the server information."""
        return self._server_info

    def get_workspace_interface(
        self,
        user_info: UserInfo,
        workspace: str,
        client_id=None,
        timeout=10,
        silent=True,
    ):
        """Get the interface of a workspace."""
        assert workspace, "Workspace name is required"
        assert user_info and isinstance(user_info, UserInfo), "User info is required"
        # the client will be hidden if client_id is None
        if silent is None:
            silent = client_id is None
        client_id = client_id or "client-" + random_id(readable=False)
        rpc = self.create_rpc(workspace, user_info, client_id=client_id, silent=silent)
        return WorkspaceInterfaceContextManager(
            rpc, self, workspace, user_info, timeout=timeout
        )

    @schema_method
    async def list_all_workspaces(self):
        """List all workspaces."""
        workspace_keys = await self._redis.hkeys("workspaces")
        workspaces = []
        for k in workspace_keys:
            workspace_info = await self._redis.hget("workspaces", k)
            workspace_info = WorkspaceInfo.model_validate(
                json.loads(workspace_info.decode())
            )
            workspaces.append(workspace_info)
        return [workspace for workspace in workspaces]

    def create_rpc(
        self,
        workspace: str,
        user_info: UserInfo,
        client_id: str = None,
        default_context=None,
        silent=True,
    ):
        """Create a rpc object for a workspace."""
        client_id = client_id or "anonymous-client-" + random_id(readable=False)
        assert "/" not in client_id
        logger.debug("Creating RPC for client %s", client_id)
        assert user_info is not None, "User info is required"
        connection = RedisRPCConnection(
            self._event_bus,
            workspace,
            client_id,
            user_info,
            manager_id=self._manager_id,
        )
        rpc = RPC(
            connection,
            client_id=client_id,
            default_context=default_context,
            workspace=workspace,
            server_base_url=self.public_base_url,
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

    async def get_workspace_info(self, workspace: str, load: bool = False):
        """Return the workspace information."""
        try:
            return await self._workspace_manager.load_workspace_info(
                workspace, load=load
            )
        except KeyError:
            return None

    async def set_workspace(
        self, workspace: WorkspaceInfo, user_info: UserInfo, overwrite: bool = False
    ):
        """Set the workspace information."""
        return await self._workspace_manager._update_workspace(
            workspace, user_info, overwrite=overwrite
        )

    def set_s3_controller(self, controller):
        """Set the s3 controller."""
        self._s3_controller = controller

    def set_server_app_controller(self, controller):
        """Set the server app controller."""
        self._server_app_controller = controller

    def get_server_app_controller(self):
        """Get the server app controller."""
        return self._server_app_controller

    def set_artifact_manager(self, controller):
        """Set the artifact controller."""
        self._artifact_manager = controller
        # Also update the workspace manager's reference if it exists
        if hasattr(self, '_workspace_manager') and self._workspace_manager:
            self._workspace_manager._artifact_manager = controller

    def register_public_service(self, service: dict):
        """Register a service."""
        assert not self._ready, "Cannot register public service after ready"

        if "name" not in service or "id" not in service:
            raise Exception("Service should at least contain `name` and `id`")

        # TODO: check if it's already exists
        service["config"] = service.get("config", {})
        assert isinstance(
            service["config"], dict
        ), "service.config must be a dictionary"
        service["config"]["workspace"] = "public"
        assert (
            "visibility" not in service
        ), "`visibility` should be placed inside `config`"
        assert (
            "require_context" not in service
        ), "`require_context` should be placed inside `config`"
        formated_service = ServiceInfo.model_validate(service)
        # Note: service can set its `visibility` to `public` or `protected`
        self._public_services.append(formated_service)
        return {
            "id": formated_service.id,
            "workspace": "public",
            "name": formated_service.name,
        }

    def register_public_type(self, service_type: dict):
        """Register a service type."""
        assert not self._ready, "Cannot register public service after ready"

        if "name" not in service_type or "id" not in service_type:
            raise Exception("Service should at least contain `name` and `id`")
        assert "definition" in service_type, "Service type should contain `definition`"
        formatted_service_type = ServiceTypeInfo.model_validate(service_type)
        self._public_types.append(formatted_service_type)

    def is_ready(self):
        """Check if the server is alive."""
        return self._ready

    async def is_alive(self):
        """Check if the server is alive."""
        return await self._public_workspace_interface.ping(
            client_id=self._server_id, timeout=1
        )

    def register_router(self, router):
        """Register a router."""
        self._app.include_router(router)

    def mount_app(self, path, app, name=None, priority=-1):
        """Mount an app to fastapi."""
        route = Mount(path, app, name=name)
        # remove existing path
        routes_remove = [route for route in self._app.routes if route.path == path]
        for rou in routes_remove:
            self._app.routes.remove(rou)
        # The default priority is -1 which assumes the last one is websocket
        self._app.routes.insert(priority, route)

    def unmount_app(self, path):
        """Unmount an app to fastapi."""
        routes_remove = [route for route in self._app.routes if route.path == path]
        for route in routes_remove:
            self._app.routes.remove(route)

    async def teardown(self):
        """Teardown the server."""
        self._ready = False
        logger.info(f"Tearing down the redis store for server {self._server_id}...")
        
        # Clean up all services registered by THIS server during graceful shutdown
        # This ensures we don't leave orphaned services in Redis
        try:
            logger.info(f"Clearing all services for server {self._server_id}")
            await self._clear_all_server_services()
        except Exception as e:
            logger.error(f"Error clearing server services: {e}")
        
        # if self._house_keeping_task:
        #     self._house_keeping_task.cancel()
        #     try:
        #         await self._house_keeping_task
        #     except asyncio.CancelledError:
        #         print("Housekeeping task successfully exited.")

        if self._tracker_task:
            self._tracker_task.cancel()
            try:
                await self._tracker_task
            except asyncio.CancelledError:
                print("Activity tracker successfully exited.")
        
        # Remove specific client interfaces
        try:
            if self._public_workspace_interface:
                client_id = self._public_workspace_interface.rpc.get_client_info()["id"]
                await self.remove_client(client_id, "public", self._root_user, unload=True)
        except Exception as e:
            logger.warning(f"Error removing public workspace client: {e}")
            
        try:
            if self._root_workspace_interface:
                client_id = self._root_workspace_interface.rpc.get_client_info()["id"]
                await self.remove_client(
                    client_id, self._root_user.get_workspace(), self._root_user, unload=True
                )
        except Exception as e:
            logger.warning(f"Error removing root workspace client: {e}")
        
        if self._websocket_server:
            websockets = self._websocket_server.get_websockets()
            for ws in websockets.values():
                try:
                    await ws.close()
                except GeneratorExit:
                    pass

        await self._event_bus.stop()
        # Close Redis client gracefully
        try:
            if hasattr(self._redis, "close"):
                await self._redis.close()
            if hasattr(self._redis, "connection_pool"):
                await self._redis.connection_pool.disconnect()
        except Exception:
            pass
        logger.info(f"Teardown complete for server {self._server_id}")

    @property
    def wm(self):
        """Get the workspace manager interface.

        Can be used in three ways:
        1. As a direct method call: await self.wm.method()
        2. As a context manager with default context: async with self.wm() as wm:
        3. As a context manager with custom context: async with self.wm(context) as wm:

        Returns:
            WorkspaceManagerWrapper: A wrapper that supports both direct usage and context manager.
        """

        class WorkspaceManagerWrapper:
            def __init__(self, store):
                self._store = store
                self._interface = None
                self._interface_cm = None
                self._default_workspace = None  # Cache the workspace to use

            def __call__(self, user_info=None, workspace=None):
                """Support both wm() and wm(context) syntax for context manager."""
                # If workspace is explicitly provided, remember it for future direct calls
                if workspace is not None:
                    self._default_workspace = workspace
                
                # Use the cached workspace or default to root user's workspace
                effective_workspace = workspace or self._default_workspace or "public"

                return WorkspaceManagerContextManager(
                    self._store,
                    user_info or self._store._root_user.model_dump(),
                    effective_workspace,
                )

            async def _ensure_interface(self):
                """Ensure we have an active interface, creating one if needed."""
                if self._interface is None:
                    # Use the cached workspace or default to root user's workspace
                    workspace = self._default_workspace or "public"
                    self._interface_cm = self._store.get_workspace_interface(
                        self._store._root_user,
                        workspace,
                        client_id=None,
                        silent=True,
                    )
                    self._interface = await self._interface_cm.__aenter__()
                return self._interface

            async def _cleanup_interface(self):
                """Clean up the interface if it exists."""
                if self._interface_cm is not None:
                    await self._interface_cm.__aexit__(None, None, None)
                    self._interface = None
                    self._interface_cm = None

            def __getattr__(self, name):
                """Support direct method access by creating a temporary interface."""

                async def wrapper(*args, **kwargs):
                    try:
                        interface = await self._ensure_interface()
                        method = getattr(interface, name)
                        result = await method(*args, **kwargs)
                        return result
                    except Exception as e:
                        await self._cleanup_interface()
                        raise e

                return wrapper

            async def __aenter__(self):
                """Support using the wrapper itself as a context manager."""
                return await self._ensure_interface()

            async def __aexit__(self, exc_type, exc, tb):
                """Clean up when used as a context manager."""
                await self._cleanup_interface()

        return WorkspaceManagerWrapper(self)
