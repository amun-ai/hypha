"""A scalable state store based on Redis."""
import asyncio
import json
import logging
import time
import sys
import datetime
from typing import List, Union
from pydantic import BaseModel
from fastapi import Header, Cookie

from hypha_rpc import RPC
from hypha_rpc.utils.schema import schema_method
from starlette.routing import Mount

from hypha import __version__
from hypha.core import (
    RedisEventBus,
    RedisRPCConnection,
    ServiceInfo,
    ServiceTypeInfo,
    UserInfo,
    WorkspaceInfo,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
)
from sqlalchemy import inspect, text
from sqlalchemy.exc import NoInspectionAvailable, NoSuchTableError

from hypha.core.auth import (
    create_scope,
    parse_token,
    generate_anonymous_user,
    UserPermission,
    AUTH0_CLIENT_ID,
    AUTH0_DOMAIN,
    AUTH0_AUDIENCE,
    AUTH0_ISSUER,
    LOGIN_SERVICE_URL,
)
from hypha.core.workspace import WorkspaceManager
from hypha.startup import run_startup_function
from hypha.utils import random_id

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("redis-store")
logger.setLevel(logging.INFO)


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
        # Check if workspace exists
        await self._store.load_or_create_workspace(self._user_info, self._workspace)
        self._wm = await self._rpc.get_manager_service({"timeout": self._timeout})
        self._wm.rpc = self._rpc
        self._wm.disconnect = self._rpc.disconnect
        self._wm.register_codec = self._rpc.register_codec
        self._wm.register_service = self._rpc.register_service
        return self._wm

    async def __aexit__(self, exc_type, exc, tb):
        await self._rpc.disconnect()


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
        reconnection_token_life_time=2 * 24 * 60 * 60,
    ):
        """Initialize the redis store."""
        self._s3_controller = None
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
        self._server_id = server_id or random_id(readable=True)
        self._manager_id = "manager-" + self._server_id
        self.reconnection_token_life_time = reconnection_token_life_time
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

        self._database_uri = database_uri
        if self._database_uri is None:
            database_uri = (
                "sqlite+aiosqlite:///:memory:"  # In-memory SQLite for testing
            )
            logger.warning(
                "Using in-memory SQLite database for event logging, all data will be lost on restart!"
            )

        self._sql_engine = create_async_engine(database_uri, echo=False)

        if redis_uri and redis_uri.startswith("redis://"):
            from redis import asyncio as aioredis

            self._redis = aioredis.from_url(redis_uri)
        else:  #  Create a redis server with fakeredis
            from fakeredis import aioredis

            self._redis = aioredis.FakeRedis.from_url("redis://localhost:9997/11")

        self._root_user = None
        self._event_bus = RedisEventBus(self._redis)

    def set_websocket_server(self, websocket_server):
        """Set the websocket server."""
        assert self._websocket_server is None, "Websocket server already set"
        self._websocket_server = websocket_server

    @schema_method
    def kickout_client(self, workspace: str, client_id: str, code: int, reason: str):
        """Force disconnect a client."""
        return self._websocket_server.force_disconnect(
            workspace, client_id, code, reason
        )

    def get_redis(self):
        return self._redis

    def get_sql_engine(self):
        return self._sql_engine

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
        return self._root_user

    async def load_or_create_workspace(self, user_info: UserInfo, workspace: str):
        """Setup the workspace."""
        if workspace is None:
            workspace = user_info.get_workspace()

        assert workspace != "*", "Dynamic workspace is not allowed for this endpoint"

        # Ensure calls to store for workspace existence and permissions check
        workspace_info = await self.get_workspace_info(workspace, load=True)

        # If workspace does not exist, automatically create it
        if not workspace_info and workspace == user_info.get_workspace():
            if not user_info.check_permission(workspace, UserPermission.read):
                raise PermissionError(
                    f"User {user_info.id} does not have permission to access workspace {workspace}"
                )
            workspace_info = await self.register_workspace(
                {
                    "id": workspace,
                    "name": workspace,
                    "description": f"Workspace for user {user_info.id}",
                    "persistent": not user_info.is_anonymous,
                    "owners": [user_info.id],
                    "read_only": user_info.is_anonymous,
                }
            )
        else:
            if not workspace_info:
                raise KeyError(
                    f"Workspace {workspace} does not exist or is not accessible"
                )
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

        workspaces = await self.get_all_workspace()
        logger.info("Upgrading workspace bookmarks for version < 0.20.34")
        for workspace in workspaces:
            workspace.config["bookmarks"] = workspace.config.get("bookmarks") or []
            for bookmark in workspace.config["bookmarks"]:
                if "type" in bookmark:
                    bookmark_type = bookmark["type"]
                    if not bookmark_type:
                        continue
                    logger.info(
                        f"Convert bookmark type: {bookmark}, workspace: {workspace.id}"
                    )
                    bookmark["bookmark_type"] = bookmark_type
                    del bookmark["type"]

                    # Log the change
                    change_time = datetime.datetime.now().isoformat()
                    database_change_log.append(
                        {
                            "time": change_time,
                            "version": __version__,
                            "change": f"Converted bookmark type in workspace {workspace.id}",
                        }
                    )
            await self.set_workspace(workspace, self._root_user, overwrite=True)

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
            for server_id in server_ids:
                try:
                    svc = await rpc.get_remote_service(
                        f"public/{server_id}:built-in", {"timeout": 2}
                    )
                    assert await svc.ping("ping") == "pong"
                except Exception as e:
                    logger.warning(
                        f"Server {server_id} is not responding (error: {e}), cleaning up..."
                    )
                    # remove root and public services
                    await self._clear_client_services("public", server_id)
                    await self._clear_client_services(
                        self._root_user.get_workspace(), server_id
                    )
                else:
                    if server_id == self._server_id:
                        raise RuntimeError(
                            f"Server with the same id ({server_id}) is already running, please use a different server id by passing `--server-id=new-server-id` to the command line."
                        )

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

    async def init(self, reset_redis, startup_functions=None):
        """Setup the store."""
        if reset_redis:
            logger.warning("RESETTING ALL REDIS DATA!!!")
            await self._redis.flushall()
        await self._event_bus.init()
        await self.setup_root_user()
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
        self._ready = True
        await self.get_event_bus().emit_local("startup")
        servers = await self.list_servers()
        logger.info("Server initialized with server id: %s", self._server_id)
        logger.info("Currently connected hypha servers: %s", servers)

    async def _register_root_services(self):
        """Register root services."""
        self._root_workspace_interface = await self.get_workspace_interface(
            self._root_user,
            self._root_user.get_workspace(),
            client_id=self._server_id,
            silent=False,
        )
        await self._root_workspace_interface.register_service(
            {
                "id": "admin-utils",
                "name": "Admin Utilities",
                "config": {
                    "visibility": "protected",
                    "require_context": False,
                },
                "list_servers": self.list_servers,
                "kickout_client": self.kickout_client,
                "list_workspaces": self.list_all_workspaces,
            }
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
        user_info = parse_token(token)
        key = "revoked_token:" + token
        if await self._redis.exists(key):
            raise Exception("Token has been revoked")
        if "admin" in user_info.roles:
            user_info.scope.workspaces["*"] = UserPermission.admin
        return user_info

    async def login_optional(
        self,
        authorization: str = Header(None),
        access_token: str = Cookie(None),
    ):
        """Return user info or create an anonymouse user.

        If authorization code is valid the user info is returned,
        If the code is invalid an an anonymouse user is created.
        """
        token = authorization or access_token
        if token:
            user_info = await self.parse_user_token(token)
            if user_info.scope.current_workspace is None:
                user_info.scope.current_workspace = user_info.get_workspace()
            return user_info
        else:
            user_info = generate_anonymous_user()
            user_workspace = user_info.get_workspace()
            user_info.scope = create_scope(
                f"{user_workspace}#a", current_workspace=user_workspace
            )
            return user_info

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
        return self._workspace_manager.get_client_id()

    async def register_workspace_manager(self):
        """Register a workspace manager."""
        manager = WorkspaceManager(
            self._redis,
            self._root_user,
            self._event_bus,
            self._server_info,
            self._manager_id,
            self._sql_engine,
            self._s3_controller,
            self._artifact_manager,
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
        logger.info("Creating RPC for client %s", client_id)
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
                "encoder": lambda x: x.model_dump(),
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

    def set_artifact_manager(self, controller):
        """Set the artifact controller."""
        self._artifact_manager = controller

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
        logger.info("Tearing down the public workspace...")
        client_id = self._public_workspace_interface.rpc.get_client_info()["id"]
        await self.remove_client(client_id, "public", self._root_user, unload=True)
        client_id = self._root_workspace_interface.rpc.get_client_info()["id"]
        await self.remove_client(
            client_id, self._root_user.get_workspace(), self._root_user, unload=True
        )
        logger.info("Teardown complete")
