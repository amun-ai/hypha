"""Provide the ImJoy core API interface."""
import asyncio
import json
import logging
import random
import sys
from enum import Enum
from typing import Any, Dict, List, Optional

import shortuuid
from pydantic import (  # pylint: disable=no-name-in-module
    BaseModel,
    EmailStr,
    Extra,
    PrivateAttr,
)

from hypha.utils import EventBus
from hypha.core.plugin import DynamicPlugin

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("core")
logger.setLevel(logging.INFO)


class TokenConfig(BaseModel):
    """Represent a token configuration."""

    scopes: List[str]
    expires_in: Optional[int]
    email: Optional[EmailStr]


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
    require_context: bool = False
    workspace: str
    flags: List[str] = []


class ServiceInfo(BaseModel):
    """Represent service."""

    config: ServiceConfig
    name: str
    type: str

    _provider: DynamicPlugin = PrivateAttr(default_factory=lambda: None)
    _id: str = PrivateAttr(default_factory=shortuuid.uuid)

    class Config:
        """Set the config for pydantic."""

        extra = Extra.allow

    def is_singleton(self):
        """Check if the service is singleton."""
        return "single-instance" in self.config.flags

    def set_provider(self, provider: DynamicPlugin) -> None:
        """Set the provider plugin."""
        self._provider = provider

    def get_provider(self) -> DynamicPlugin:
        """Get the provider plugin."""
        return self._provider

    def get_summary(self) -> dict:
        """Get a summary about the service."""
        summary = {
            "name": self.name,
            "type": self.type,
            "id": self._id,
            "visibility": self.config.visibility.value,
            "provider": self._provider and self._provider.name,
            "provider_id": self._provider and self._provider.id,
        }
        summary.update(json.loads(self.config.json()))
        return summary

    def get_id(self) -> str:
        """Get service id."""
        return self._id


class UserInfo(BaseModel):
    """Represent user info."""

    id: str
    roles: List[str]
    is_anonymous: bool
    email: Optional[EmailStr]
    parent: Optional[str]
    scopes: Optional[List[str]]  # a list of workspace
    expires_at: Optional[int]
    _metadata: Dict[str, Any] = PrivateAttr(
        default_factory=lambda: {}
    )  # e.g. s3 credential
    _plugins: Dict[str, DynamicPlugin] = PrivateAttr(
        default_factory=lambda: {}
    )  # id:plugin

    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata."""
        return self._metadata

    def get_plugins(self) -> Dict[str, DynamicPlugin]:
        """Return the plugins."""
        return self._plugins

    def get_plugin(self, plugin_id: str) -> Optional[DynamicPlugin]:
        """Return a plugin by id."""
        return self._plugins.get(plugin_id)

    def add_plugin(self, plugin: DynamicPlugin) -> None:
        """Add a plugin."""
        self._plugins[plugin.id] = plugin

    def remove_plugin(self, plugin: DynamicPlugin) -> None:
        """Remove a plugin by id."""
        del self._plugins[plugin.id]


class WorkspaceInfo(BaseModel):
    """Represent a workspace."""

    name: str
    persistent: bool
    owners: List[str]
    visibility: VisibilityEnum
    description: Optional[str]
    icon: Optional[str]
    covers: Optional[List[str]]
    docs: Optional[str]
    allow_list: Optional[List[str]]
    deny_list: Optional[List[str]]
    read_only: bool = False
    _logger: Optional[logging.Logger] = PrivateAttr(default_factory=lambda: logger)
    _plugins: Dict[str, DynamicPlugin] = PrivateAttr(
        default_factory=lambda: {}
    )  # name: plugin
    _services: Dict[str, ServiceInfo] = PrivateAttr(default_factory=lambda: {})
    _event_bus: EventBus = PrivateAttr(default_factory=EventBus)
    _global_event_bus: EventBus = PrivateAttr(default_factory=lambda: None)

    def set_global_event_bus(self, event_bus: EventBus) -> None:
        """Set the global event bus."""
        self._global_event_bus = event_bus

    def get_logger(self) -> Optional[logging.Logger]:
        """Return the logger."""
        return self._logger

    def set_logger(self, workspace_logger: logging.Logger) -> None:
        """Return the logger."""
        self._logger = workspace_logger

    def get_plugins(self) -> Dict[str, DynamicPlugin]:
        """Return the plugins."""
        return self._plugins

    def get_plugin_by_name(self, plugin_name: str) -> Optional[DynamicPlugin]:
        """Return a plugin by its name (randomly select one if multiple exists)."""
        plugins = [
            plugin
            for plugin in self._plugins.values()
            if plugin.name == plugin_name and plugin.get_status() == "ready"
        ]
        if len(plugins) > 0:
            return random.choice(plugins)
        return None

    def get_plugin_by_id(self, plugin_id: str) -> Optional[DynamicPlugin]:
        """Return a plugin by its id."""
        plugins = [
            plugin for plugin in self._plugins.values() if plugin.id == plugin_id
        ]
        if len(plugins) > 0:
            return plugins[0]
        return None

    def add_plugin(self, plugin: DynamicPlugin) -> None:
        """Set a plugin."""
        if plugin.id in self._plugins:
            raise Exception(
                f"Plugin with the same id({plugin.id})"
                " already exists in the workspace ({self.name})"
            )

        if plugin.is_singleton():
            for plg in self._plugins.values():
                if plg.name == plugin.name:
                    logger.info(
                        "Terminating other plugins with the same name"
                        " (%s) due to single-instance flag",
                        plugin.name,
                    )
                    asyncio.ensure_future(plg.terminate())
        self._plugins[plugin.id] = plugin
        self._event_bus.emit("plugin_connected", plugin.config)

    def remove_plugin(self, plugin: DynamicPlugin) -> None:
        """Remove a plugin form the workspace."""
        plugin_id = plugin.id
        if plugin_id not in self._plugins:
            raise KeyError(f"Plugin not fould (id={plugin_id})")
        del self._plugins[plugin_id]
        self._event_bus.emit("plugin_disconnected", plugin.config)

    def get_services_by_plugin(self, plugin: DynamicPlugin) -> List[ServiceInfo]:
        """Get services by plugin."""
        return [
            self._services[k]
            for k in self._services
            if self._services[k].get_provider() == plugin
        ]

    def get_services(self) -> Dict[str, ServiceInfo]:
        """Return the services."""
        return self._services

    def add_service(self, service: ServiceInfo) -> None:
        """Add a service."""
        if service.is_singleton():
            for svc in self._plugins.values():
                if svc.name == service.name:
                    logger.info(
                        "Unregistering other services with the same name"
                        " (%s) due to single-instance flag",
                        svc.name,
                    )
                    # TODO: we need to emit unregister event here
                    self.remove_service(svc)
        self._services[service.get_id()] = service
        self._global_event_bus.emit("service_registered", service)

    def get_service_by_name(self, service_name: str) -> ServiceInfo:
        """Return a service by its name (randomly select one if multiple exists)."""
        services = [
            service
            for service in self._services.values()
            if service.name == service_name
        ]
        if len(services) > 0:
            return random.choice(services)
        return None

    def remove_service(self, service: ServiceInfo) -> None:
        """Remove a service."""
        del self._services[service.get_id()]
        self._global_event_bus.emit("service_unregistered", service)

    def get_event_bus(self):
        """Get the workspace event bus."""
        return self._event_bus

    def get_summary(self) -> dict:
        """Get a summary about the workspace."""
        summary = {
            "name": self.name,
            "plugin_count": len(self._plugins),
            "service_count": len(self._services),
            "visibility": self.visibility.value,
            "plugins": [
                {"name": plugin.name, "id": plugin.id, "type": plugin.config.type}
                for plugin in self._plugins.values()
            ],
            "services": [service.get_summary() for service in self._services.values()],
        }
        return summary
