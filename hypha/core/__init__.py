"""Provide the ImJoy core API interface."""
import asyncio
import json
import logging
import random
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple

import shortuuid
from pydantic import (  # pylint: disable=no-name-in-module
    BaseModel,
    EmailStr,
    Extra,
    PrivateAttr,
)

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
    require_context: Union[Tuple[str], List[str], bool] = False
    workspace: str = None
    flags: List[str] = []


class ServiceInfo(BaseModel):
    """Represent service."""

    config: ServiceConfig
    id: str
    name: str
    type: str

    class Config:
        """Set the config for pydantic."""

        extra = Extra.allow

    def is_singleton(self):
        """Check if the service is singleton."""
        return "single-instance" in self.config.flags


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

    def get_metadata(self) -> Dict[str, Any]:
        """Return the metadata."""
        return self._metadata


class ClientInfo(BaseModel):
    """Represent service."""

    id: str
    name: Optional[str]
    workspace: str
    services: List[ServiceInfo] = []
    user_info: UserInfo


class RDF(BaseModel):
    """Represent resource description file object."""

    name: str
    id: str
    tags: List[str]
    documentation: Optional[str]
    covers: Optional[List[str]]
    badges: Optional[List[str]]
    authors: Optional[List[Dict[str, str]]]
    attachments: Optional[Dict[str, List[Any]]]
    config: Optional[Dict[str, Any]]
    type: str
    format_version: str = "0.2.1"
    version: str = "0.1.0"
    links: Optional[List[str]]
    maintainers: Optional[List[Dict[str, str]]]
    license: Optional[str]
    git_repo: Optional[str]
    source: Optional[str]

    class Config:
        """Set the config for pydantic."""

        extra = Extra.allow


class ApplicationInfo(RDF):
    """Represent an application."""

    pass


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
    applications: Dict[str, RDF] = {}  # installed applications
    interfaces: Dict[str, List[Any]] = {}
