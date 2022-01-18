"""Provide interface functions for the core."""
import asyncio
import inspect
import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import Dict, Optional, Union
from contextvars import copy_context

import pkg_resources
import shortuuid
from starlette.routing import Mount

from hypha.core import ServiceInfo, TokenConfig, UserInfo, VisibilityEnum, WorkspaceInfo
from hypha.core.auth import generate_presigned_token, parse_token
from hypha.utils import EventBus, dotdict
from hypha.core.store import RedisStore

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

    def __init__(
        self,
        app,
        app_controller=None,
        public_base_url=None,
        local_base_url=None,
    ):
        """Set up instance."""
        self.current_user = ContextVar("current_user")
        self.current_workspace = ContextVar("current_workspace")
        self.store = RedisStore.get_instance()
        self.event_bus = self.store.get_event_bus()
        self._all_users: Dict[str, UserInfo] = {}  # uid:user_info
        self._all_workspaces: Dict[str, WorkspaceInfo] = {}  # wid:workspace_info
        self._workspace_loader = None
        self._app = app
        self.app_controller = app_controller
        self.disconnect_delay = 1
        self._codecs = {}
        self._disconnected_plugins = []
        self.public_base_url = public_base_url
        self.local_base_url = local_base_url
        self._public_services: List[ServiceInfo] = []
        self._ready = False
        self.load_extensions()

        # def remove_empty_workspace(plugin):
        #     # Remove the user completely if no plugins exists
        #     user_info = plugin.user_info
        #     if len(user_info.get_plugins()) <= 0:
        #         del self._all_users[user_info.id]
        #         logger.info(
        #             "Removing user (%s) completely since the user "
        #             "has no other plugin connected.",
        #             user_info.id,
        #         )
        #     # Remove the user completely if no plugins exists
        #     workspace = plugin.workspace
        #     if len(workspace.get_plugins()) <= 0 and not workspace.persistent:
        #         logger.info(
        #             "Removing workspace (%s) completely "
        #             "since there is no other plugin connected.",
        #             workspace.name,
        #         )
        #         self.unregister_workspace(workspace)

        # self.event_bus.on("plugin_terminated", remove_empty_workspace)
        self._public_workspace = WorkspaceInfo.parse_obj(
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
        self._public_workspace_interface = None

    def get_user_info_from_token(self, token):
        """Get user info from token."""
        user_info = parse_user(token)
        return user_info

    def check_permission(self, workspace, user_info):
        """Check user permission for a workspace."""
        # pylint: disable=too-many-return-statements
        if isinstance(workspace, str):
            workspace = self._all_workspaces.get(workspace)
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

    async def get_workspace(self, name, load=True):
        """Return the workspace."""
        try:
            manager = await self.store.get_workspace_manager(name, setup=False)
            return await manager.get_workspace_info(name)
        except KeyError:
            if load and self._workspace_loader:
                try:
                    workspace = await self._workspace_loader(
                        name, await self.store.setup_root_user()
                    )
                    if workspace:
                        self._all_workspaces[workspace.name] = workspace
                except Exception:  # pylint: disable=broad-except
                    logger.exception("Failed to load workspace %s", name)
                else:
                    return workspace
        return None

    def set_workspace_loader(self, loader):
        """Set the workspace loader."""
        self._workspace_loader = loader

    def load_extensions(self):
        """Load hypha extensions."""
        # Support hypha extensions
        # See how it works:
        # https://packaging.python.org/guides/creating-and-discovering-plugins/
        for entry_point in pkg_resources.iter_entry_points("hypha_extension"):
            try:
                setup_extension = entry_point.load()
                setup_extension(self)
            except Exception:
                logger.exception("Failed to setup extension: %s", entry_point.name)
                raise

    def register_router(self, router):
        """Register a router."""
        self._app.include_router(router)

    def register_public_service(self, service: dict):
        """Register a service."""
        if "name" not in service or "type" not in service:
            raise Exception("Service should at least contain `name` and `type`")

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
        formated_service = ServiceInfo.parse_obj(service)
        # Force to require context
        formated_service.config.require_context = True
        service_dict = formated_service.dict()

        for key in service_dict:
            if callable(service_dict[key]):

                def wrap_func(func, *args, context=None, **kwargs):
                    user_info = UserInfo.parse_obj(context["user"])
                    self.current_user.set(user_info)
                    source_workspace = context["from"].split("/")[0]
                    self.current_workspace.set(source_workspace)
                    ctx = copy_context()
                    return ctx.run(func, *args, **kwargs)

                wrapped = partial(wrap_func, service_dict[key])
                wrapped.__name__ = key
                setattr(formated_service, key, wrapped)
        # service["_rintf"] = True
        # Note: service can set its `visibility` to `public` or `protected`
        self._public_services.append(ServiceInfo.parse_obj(formated_service))
        return {
            "id": formated_service.get_id(),
            "workspace": "public",
            "name": formated_service.name,
        }

    def is_ready(self):
        """Check if the server is alive."""
        return self._ready

    async def init(self):
        """Initialize the core interface."""
        await self.store.register_workspace(self._public_workspace, overwrite=True)
        manager = await self.store.get_workspace_manager("public")
        self._public_workspace_interface = await manager.get_workspace()

        await self.store.init()
        for service in self._public_services:
            await self._public_workspace_interface.register_service(service.dict())
        self._ready = True

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
