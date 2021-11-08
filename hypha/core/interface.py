"""Provide interface functions for the core."""
import asyncio
import inspect
import json
import logging
import sys
from contextvars import ContextVar
from functools import partial
from typing import Dict, Optional
from contextlib import contextmanager

import pkg_resources
import shortuuid
from starlette.routing import Mount

from hypha.core import (
    ServiceInfo,
    TokenConfig,
    UserInfo,
    VisibilityEnum,
    WorkspaceInfo,
)
from hypha.core.auth import generate_presigned_token, parse_token
from hypha.core.plugin import DynamicPlugin
from hypha.utils import dotdict, EventBus

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("imjoy-core")
logger.setLevel(logging.INFO)


def parse_user(token):
    """Parse user info from a token."""
    if token:
        user_info = parse_token(token)
        uid = user_info.id
        logger.info("User connected: %s", uid)
    else:
        uid = shortuuid.uuid()
        user_info = UserInfo(
            id=uid,
            is_anonymous=True,
            email=None,
            parent=None,
            roles=[],
            scopes=[],
            expires_at=None,
        )
        logger.info("Anonymized User connected: %s", uid)

    if uid == "root":
        logger.error("Root user is not allowed to connect remotely")
        raise Exception("Root user is not allowed to connect remotely")

    return user_info


class CoreInterface:
    """Represent the interface of the ImJoy core."""

    # pylint: disable=no-self-use, too-many-instance-attributes, too-many-public-methods

    def __init__(self, app, imjoy_api=None, app_controller=None):
        """Set up instance."""
        self.event_bus = EventBus()
        self.current_user = ContextVar("current_user")
        self.current_plugin = ContextVar("current_plugin")
        self.current_workspace = ContextVar("current_workspace")
        self._all_users: Dict[str, UserInfo] = {}  # uid:user_info
        self._all_workspaces: Dict[str, WorkspaceInfo] = {}  # wid:workspace_info
        self._app = app
        self.app_controller = app_controller
        self.disconnect_delay = 1
        imjoy_api = imjoy_api or {}
        self._codecs = {}
        self._disconnected_plugins = []
        self._imjoy_api = dotdict(
            {
                "_rintf": True,
                "log": self.info,
                "info": self.info,
                "error": self.error,
                "warning": self.warning,
                "critical": self.critical,
                "registerService": self.register_service,
                "register_service": self.register_service,
                "listServices": self.list_services,
                "list_services": self.list_services,
                "getService": self.get_service,
                "get_service": self.get_service,
                "utils": {},
                "listPlugins": self.list_plugins,
                "list_plugins": self.list_plugins,
                "getPlugin": self.get_plugin,
                "get_plugin": self.get_plugin,
                "generateToken": self.generate_token,
                "generate_token": self.generate_token,
                "create_workspace": self.create_workspace,
                "createWorkspace": self.create_workspace,
                "get_workspace": self.get_workspace_interface,
                "getWorkspace": self.get_workspace_interface,
                "list_workspaces": self.list_workspaces,
                "listWorkspaces": self.list_workspaces,
                "disconnect": self.disconnect,
                # "stop_plugin": self.stop_plugin,
            }
        )
        self._imjoy_api.update(imjoy_api)

        # Add public workspace
        self.register_workspace(
            WorkspaceInfo.parse_obj(
                {
                    "name": "public",
                    "persistent": True,
                    "owners": ["root"],
                    "allow_list": [],
                    "deny_list": [],
                    "visibility": "public",
                    "read_only": True,
                }
            )
        )

        # Create root user
        self.root_user = UserInfo(
            id="root",
            is_anonymous=False,
            email=None,
            parent=None,
            roles=[],
            scopes=[],
            expires_at=None,
        )
        # Create root workspace
        self.root_workspace = WorkspaceInfo(
            name="root",
            owners=["root"],
            visibility=VisibilityEnum.protected,
            persistent=True,
        )
        self.root_workspace.set_global_event_bus(self.event_bus)

        @contextmanager
        def set_root_user():
            _user = self.current_user.get(None)
            try:
                self.current_user.set(self.root_user)
                yield self.root_user
            finally:
                if _user:
                    self.current_user.set(_user)

        self.set_root_user = set_root_user

        self.register_workspace(self.root_workspace)
        self.load_extensions()

        def remove_empty_workspace(plugin):
            # Remove the user completely if no plugins exists
            workspace = plugin.workspace
            if len(workspace.get_plugins()) <= 0 and not workspace.persistent:
                logger.info(
                    "Removing workspace (%s) completely "
                    "since there is no other plugin connected.",
                    workspace.name,
                )
                self.unregister_workspace(workspace)

        self.event_bus.on("plugin_terminated", remove_empty_workspace)

    def get_all_users(self):
        """Get all the users."""
        return list(self._all_users.values())

    def get_user_info_from_token(self, token):
        """Get user info from token."""
        user_info = parse_user(token)
        # Note here we only use the newly created user info object
        # if the same user id does not exist
        if user_info.id in self._all_users:
            user_info = self._all_users[user_info.id]
        else:
            self._all_users[user_info.id] = user_info
        return user_info

    def create_user_workspace(self, user_info, read_only: bool = False):
        """Create a workspace for the user."""
        # only registered user can have persistent workspace
        persistent = not user_info.is_anonymous
        # create the user workspace automatically
        workspace = WorkspaceInfo(
            name=user_info.id,
            owners=[user_info.id],
            visibility=VisibilityEnum.protected,
            persistent=persistent,
            read_only=read_only,
        )
        workspace.set_global_event_bus(self.event_bus)
        self.register_workspace(workspace)
        return workspace

    async def restore_plugin(self, plugin):
        """Restore the plugin."""
        if plugin in self._disconnected_plugins:
            logger.info("Plugin connection restored")
            self._disconnected_plugins.remove(plugin)
        else:
            logger.warning("Plugin connection is not in the disconnected list")

    async def remove_plugin_delayed(self, plugin):
        """Remove the plugin after a delayed period (if not cancelled)."""
        if plugin not in self._disconnected_plugins:
            self._disconnected_plugins.append(plugin)
        await asyncio.sleep(self.disconnect_delay)
        # It means the session has been reconnected
        if plugin not in self._disconnected_plugins:
            return
        await self._terminate_plugin(plugin)
        # TODO: if the plugin owner is anonymous, we should remove the workspace

    def remove_plugin_temp(self, sid):
        """Remove session temporarily."""
        plugin = DynamicPlugin.get_plugin_by_session_id(sid)
        if plugin is None:
            logger.warning(
                "Plugin (sid: %s) does not exist or has already been terminated.", sid
            )
            return
        loop = asyncio.get_running_loop()
        loop.create_task(self.remove_plugin_delayed(plugin))

    async def disconnect(
        self,
    ):
        """Disconnect from the workspace."""
        if self.current_workspace.get() == self.current_plugin.get().workspace:
            plugin = self.current_plugin.get()
            await self._terminate_plugin(plugin)
        else:
            raise Exception("Cannot disconnect from a different workspace")

    async def _terminate_plugin(self, plugin):
        """Terminate the plugin."""
        await plugin.terminate()
        user_info = plugin.user_info
        # Remove the user completely if no plugins exists
        if len(user_info.get_plugins()) <= 0:
            del self._all_users[user_info.id]
            logger.info(
                "Removing user (%s) completely since the user "
                "has no other plugin connected.",
                user_info.id,
            )

        workspace = plugin.workspace
        # check if the workspace is a not persistent worksapce
        # terminate all the child plugins
        if not workspace.persistent:
            logger.info(
                "Terminating all the child plugins in the workspace (%s).",
                workspace.name,
            )
            for plg in list(workspace.get_plugins().values()):
                if plg.user_info.parent == plugin.user_info.id:
                    await plg.terminate()

    def check_permission(self, workspace, user_info):
        """Check user permission for a workspace."""
        # pylint: disable=too-many-return-statements
        if isinstance(workspace, str):
            workspace = self.get_workspace(workspace)
            if not workspace:
                logger.error("Workspace %s not found", workspace)
                return False

        # Make exceptions for root user, the children of root and test workspace
        if (
            user_info.id == "root"
            or user_info.parent == "root"
            or workspace.name == "public"
        ):
            return True

        if workspace.name == user_info.id:
            return True

        if user_info.parent:
            parent = self._all_users.get(user_info.parent)
            if not parent:
                return False
            if not self.check_permission(workspace, parent):
                return False
            # if the parent has access
            # and the workspace is in the scopes
            # then we allow the access
            if workspace.name in user_info.scopes:
                return True

        _id = user_info.email or user_info.id

        if _id in workspace.owners:
            return True

        if workspace.visibility == VisibilityEnum.public:
            if workspace.deny_list and user_info.email not in workspace.deny_list:
                return True
        elif workspace.visibility == VisibilityEnum.protected:
            if workspace.allow_list and user_info.email in workspace.allow_list:
                return True

        if "admin" in user_info.roles:
            logger.info(
                "Allowing access to %s for admin user %s", workspace.name, user_info.id
            )
            return True

        return False

    def get_all_workspace(self):
        """Return all workspaces."""
        return list(self._all_workspaces.values())

    def is_workspace_registered(self, ws):
        """Return true if workspace is registered."""
        if ws in self._all_workspaces.values():
            return True
        return False

    def get_workspace(self, name):
        """Return the workspace."""
        if name in self._all_workspaces:
            return self._all_workspaces[name]
        return None

    def register_workspace(self, ws):
        """Register the workspace."""
        ws.set_global_event_bus(self.event_bus)
        if ws.name in self._all_workspaces:
            raise Exception(
                f"Another workspace with the same name {ws.name} already exist."
            )
        self._all_workspaces[ws.name] = ws
        self.event_bus.emit("workspace_registered", ws)

    def unregister_workspace(self, workspace: WorkspaceInfo):
        """Unregister the workspace."""
        del self._all_workspaces[workspace.name]
        self.event_bus.emit("workspace_removed", workspace)

    def load_extensions(self):
        """Load hypha extensions."""
        # Support hypha extensions
        # See how it works:
        # https://packaging.python.org/guides/creating-and-discovering-plugins/
        for entry_point in pkg_resources.iter_entry_points("hypha_extension"):
            self.current_user.set(self.root_user)
            self.current_workspace.set(self.root_workspace)
            try:
                setup_extension = entry_point.load()
                setup_extension(self)
            except Exception:
                logger.exception("Failed to setup extension: %s", entry_point.name)
                raise

    def stop_plugin(self, config):
        """Stop plugin by id."""
        if isinstance(config, str):
            config = {"id": config}
        assert "id" in config, "Please provide the plugin id"
        workspace = self.current_workspace.get()
        plugin = workspace.get_plugin_by_id(config["id"])
        loop = asyncio.get_running_loop()
        loop.create_task(self.remove_plugin_delayed(plugin))

    def register_router(self, router):
        """Register a router."""
        self._app.include_router(router)

    def register_interface(self, name, func):
        """Register a interface function."""
        assert callable(func)
        self._imjoy_api[name] = func

    def register_service(self, service: dict):
        """Register a service."""
        plugin = self.current_plugin.get()
        workspace = self.current_workspace.get()
        if "name" not in service or "type" not in service:
            raise Exception("Service should at least contain `name` and `type`")

        # TODO: check if it's already exists
        service["config"] = service.get("config", {})
        assert isinstance(
            service["config"], dict
        ), "service.config must be a dictionary"
        service["config"]["workspace"] = workspace.name
        assert (
            "visibility" not in service
        ), "`visibility` should be placed inside `config`"
        assert (
            "require_context" not in service
        ), "`require_context` should be placed inside `config`"
        formated_service = ServiceInfo.parse_obj(service)
        formated_service.set_provider(plugin)
        service_dict = formated_service.dict()
        if formated_service.config.require_context:
            for key in service_dict:
                if callable(service_dict[key]):

                    def wrap_func(func, *args, **kwargs):
                        user_info = self.current_user.get()
                        workspace = self.current_workspace.get()
                        kwargs["context"] = {
                            "user_id": user_info.id,
                            "email": user_info.email,
                            "is_anonymous": user_info.is_anonymous,
                            "workspace": workspace.name,
                        }
                        return func(*args, **kwargs)

                    setattr(
                        formated_service, key, partial(wrap_func, service_dict[key])
                    )
        # service["_rintf"] = True
        # Note: service can set its `visibility` to `public` or `protected`
        workspace.add_service(formated_service)
        return {
            "id": formated_service.get_id(),
            "workspace": workspace.name,
            "name": formated_service.name,
        }

    def unregister_service(self, service_id):
        """Unregister an service."""
        workspace = self.current_workspace.get()
        plugin = self.current_plugin.get()
        services = workspace.get_services_by_plugin(plugin)
        not_exists = True
        for service in services:
            if service.get_id() == service_id:
                workspace.remove_service(service)
                not_exists = False
        if not_exists:
            raise KeyError(
                f"The service {service_id} is not registered in the current workspace."
            )

    def list_plugins(self):
        """List all plugins in the workspace."""
        workspace = self.current_workspace.get()
        return list(workspace.get_plugins())

    async def get_plugin(self, name):
        """Return a plugin by its name."""
        workspace = self.current_workspace.get()
        plugin = workspace.get_plugin_by_name(name)
        if plugin:
            return await plugin.get_api()
        raise Exception(f"Plugin `{name}` not found (possibly because it's not ready)")

    async def get_service(self, query):
        """Return a service."""
        if isinstance(query, str):
            query = {"name": query}

        if "workspace" in query:
            workspace = self.get_workspace(query["workspace"])
            if not workspace:
                raise Exception(f"Service not found: {query} (workspace unavailable)")
        else:
            workspace = self.current_workspace.get()

        if "id" in query:
            service = workspace.get_services().get(query["id"])
            root_service = self.root_workspace.get_services().get(query["id"])
            if not service and not root_service:
                raise Exception(f"Service not found: {query['id']}")
            service = service or root_service
        elif "name" in query:
            service = workspace.get_service_by_name(query["name"])
            root_service = self.root_workspace.get_service_by_name(query["name"])
            if not service and not root_service:
                raise Exception(f"Service not found: {query['name']}")
            service = service or root_service
        else:
            raise Exception("Please specify the service id or name to get the service")

        user_info = self.current_user.get()
        if (
            not self.check_permission(workspace, user_info)
            and service.config.visibility != VisibilityEnum.public
        ):
            raise Exception(f"Permission denied: {query}")

        return dotdict(service.dict())

    def list_workspaces(
        self,
    ):
        """List the workspaces for the user."""
        user_info = self.current_user.get()
        ret = []
        for workspace in self._all_workspaces.values():
            if self.check_permission(workspace, user_info):
                ret.append({"name": workspace.name})
        return ret

    def list_services(self, query: Optional[dict] = None):
        """Return a list of services based on the query."""
        # if workspace is not set, then it means current workspace
        # if workspace = *, it means search globally
        # otherwise, it search the specified workspace
        user_info = self.current_user.get()
        if query is None:
            query = {"workspace": "*"}

        ws = query.get("workspace")
        if ws:
            del query["workspace"]
        if ws == "*":
            ret = []
            for workspace in self.get_all_workspace():
                can_access_ws = self.check_permission(workspace, user_info)
                for service in workspace.get_services().values():
                    # To access the service, it should be public or owned by the user
                    if (
                        not can_access_ws
                        and service.config.visibility != VisibilityEnum.public
                    ):
                        continue
                    match = True
                    for key in query:
                        if getattr(service, key) != query[key]:
                            match = False
                    if match:
                        ret.append(service.get_summary())
            return ret
        if ws is not None:
            workspace = self.get_workspace(ws)
        else:
            workspace = self.current_workspace.get()
        ret = []
        workspace_services = workspace.get_services()
        for service in workspace_services.values():
            match = True
            for key in query:
                if getattr(service, key) != query[key]:
                    match = False
            if match:
                ret.append(service.get_summary())

        if workspace is None:
            raise Exception("Workspace not found: {ws}")

        return ret

    def info(self, msg):
        """Log a plugin message."""
        plugin = self.current_plugin.get()
        logger.info("%s: %s", plugin.name, msg)
        workspace_logger = plugin.workspace.get_logger()
        if workspace_logger:
            workspace_logger.info("%s: %s", plugin.name, msg)

    def warning(self, msg):
        """Log a plugin message (warning)."""
        plugin = self.current_plugin.get()
        workspace_logger = plugin.workspace.get_logger()
        if workspace_logger:
            workspace_logger.warning("%s: %s", plugin.name, msg)

    def error(self, msg):
        """Log a plugin error message (error)."""
        plugin = self.current_plugin.get()
        workspace_logger = plugin.workspace.get_logger()
        if workspace_logger:
            workspace_logger.error("%s: %s", plugin.name, msg)

    def critical(self, msg):
        """Log a plugin error message (critical)."""
        plugin = self.current_plugin.get()
        workspace_logger = plugin.workspace.get_logger()
        if workspace_logger:
            workspace_logger.critical("%s: %s", plugin.name, msg)

    def generate_token(self, config: Optional[dict] = None):
        """Generate a token for the current workspace."""
        workspace = self.current_workspace.get()
        user_info = self.current_user.get()
        config = config or {}
        if "scopes" in config and config["scopes"] != [workspace.name]:
            raise Exception("Scopes must be empty or contains only the workspace name.")
        config["scopes"] = [workspace.name]
        token_config = TokenConfig.parse_obj(config)
        scopes = token_config.scopes
        for scope in scopes:
            if not self.check_permission(scope, user_info):
                raise PermissionError(f"User has no permission to scope: {scope}")
        token = generate_presigned_token(user_info, token_config)
        return token

    def create_workspace(self, config: dict):
        """Create a new workspace."""
        user_info = self.current_user.get()
        config["persistent"] = config.get("persistent") or False
        if user_info.is_anonymous and config["persistent"]:
            raise Exception("Only registered user can create persistent workspace.")
        workspace = WorkspaceInfo.parse_obj(config)
        workspace.set_global_event_bus(self.event_bus)
        if self.get_workspace(workspace.name):
            raise Exception(f"Workspace {workspace.name} already exists.")
        user_info = self.current_user.get()
        # make sure we add the user's email to owners
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]
        user_info.scopes.append(workspace.name)
        self.register_workspace(workspace)
        return self.get_workspace_interface(workspace.name)

    def _update_workspace(self, name, config: dict):
        """Bind the context to the generated workspace."""
        if not name:
            raise Exception("Workspace name is not specified.")
        if not self.get_workspace(name):
            raise Exception(f"Workspace {name} not found")
        workspace = self.get_workspace(name)
        user_info = self.current_user.get()
        if not self.check_permission(workspace, user_info):
            raise PermissionError(f"Permission denied for workspace {name}")

        if "name" in config:
            raise Exception("Changing workspace name is not allowed.")

        # make sure all the keys are valid
        # TODO: verify the type
        for key in config:
            if key.startswith("_") or not hasattr(workspace, key):
                raise KeyError(f"Invalid key: {key}")

        for key in config:
            if not key.startswith("_") and hasattr(workspace, key):
                setattr(workspace, key, config[key])
        # make sure we add the user's email to owners
        _id = user_info.email or user_info.id
        if _id not in workspace.owners:
            workspace.owners.append(_id)
        workspace.owners = [o.strip() for o in workspace.owners if o.strip()]

    def get_workspace_interface(self, name: str):
        """Bind the context to the generated workspace."""
        workspace = self.get_workspace(name)
        if not workspace:
            raise Exception(f"Workspace {name} not found")
        user_info = self.current_user.get()
        if not self.check_permission(workspace, user_info):
            raise PermissionError(f"Permission denied for workspace {name}")

        interface = self.get_interface()
        bound_interface = {}
        for key in interface:
            if callable(interface[key]):

                async def wrap_func(func, *args, **kwargs):
                    try:
                        workspace_bk = self.current_workspace.get()
                    except LookupError:
                        workspace_bk = None
                    ret = None
                    try:
                        self.current_workspace.set(workspace)
                        ret = func(*args, **kwargs)
                        if inspect.isawaitable(ret):
                            ret = await ret
                    except Exception as exp:
                        raise exp
                    finally:
                        self.current_workspace.set(workspace_bk)
                    return ret

                bound_interface[key] = partial(wrap_func, interface[key])
                bound_interface[key].__name__ = key  # required for imjoy-rpc
            else:
                bound_interface[key] = interface[key]
        bound_interface["config"] = json.loads(workspace.json())
        bound_interface["set"] = partial(self._update_workspace, name)
        bound_interface["_rintf"] = True
        event_bus = workspace.get_event_bus()
        bound_interface["on"] = event_bus.on
        bound_interface["off"] = event_bus.off
        bound_interface["once"] = event_bus.once
        bound_interface["emit"] = event_bus.emit
        # Remove disconnect, since the plugin can call disconnect()
        # from their own workspace
        del bound_interface["disconnect"]
        return dotdict(bound_interface)

    def register_service_as_root(self, service):
        """Register service as root user."""
        with self.set_root_user():
            self.current_workspace.set(self.root_workspace)
            self.current_plugin.set(None)
            return self.register_service(service)

    async def get_plugin_as_root(self, name, workspace):
        """Get a plugin api as root user."""
        with self.set_root_user():
            workspace = self.get_workspace(workspace)
            if not workspace:
                raise Exception(f"Workspace {workspace} does not exist.")
            self.current_workspace.set(workspace)
            return dotdict(await self.get_plugin(name))

    def get_interface(self):
        """Return the interface."""
        return self._imjoy_api.copy()

    def register_codec(self, config):
        """Register a codec."""
        assert "name" in config
        assert "encoder" in config or "decoder" in config
        if "type" in config:
            for codec_type, codec in list(self._codecs.items()):
                if codec.type == config["type"] or codec_type == config["name"]:
                    logger.info("Removing duplicated codec: %s", codec_type)
                    del self._codecs[codec_type]

        self._codecs[config["name"]] = dotdict(config)

    def get_codecs(self):
        """Return registered codecs for rpc."""
        return self._codecs

    def mount_app(self, path, app, name=None, priority=-1):
        """Mount an app to fastapi."""
        route = Mount(path, app, name=name)
        # remove existing path
        routes_remove = [route for route in self._app.routes if route.path == path]
        for rou in routes_remove:
            self._app.routes.remove(rou)
        # The default priority is -1 which assumes the last one is websocket
        self._app.routes.insert(priority, route)

    def umount_app(self, path):
        """Unmount an app to fastapi."""
        routes_remove = [route for route in self._app.routes if route.path == path]
        for route in routes_remove:
            self._app.routes.remove(route)
