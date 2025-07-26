"""Base worker API for standardized worker implementation."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Protocol, Callable

from pydantic import BaseModel, Field, field_serializer

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Session status enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class WorkerConfig(BaseModel):
    """Configuration for starting a worker session."""
    id: str
    app_id: str
    workspace: str
    client_id: str
    server_url: str
    token: Optional[str] = None
    entry_point: Optional[str] = None
    artifact_id: str
    manifest: Dict[str, Any]
    timeout: Optional[int] = None
    app_files_base_url: Optional[str] = None
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    class Config:
        arbitrary_types_allowed = True


class SessionInfo(BaseModel):
    """Information about a worker session."""
    session_id: str
    app_id: str
    workspace: str
    client_id: str
    status: SessionStatus
    app_type: str
    created_at: str
    entry_point: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @field_serializer('status')
    def serialize_status(self, value):
        """Serialize enum to string for RPC compatibility."""
        return value.value if isinstance(value, SessionStatus) else value


class WorkerError(Exception):
    """Base exception for worker errors."""
    pass


class SessionNotFoundError(WorkerError):
    """Raised when a session is not found."""
    pass


class WorkerNotAvailableError(WorkerError):
    """Raised when a required worker dependency is not available."""
    pass


class WorkerProtocol(Protocol):
    """Protocol that all workers must implement."""
    
    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        ...
    
    async def start(self, config: Union[WorkerConfig, Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new worker session."""
        ...
    
    async def stop(self, session_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Stop a worker session."""
        ...
    
    async def list_sessions(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> List[SessionInfo]:
        """List all sessions for a workspace."""
        ...
    
    async def get_logs(
        self, 
        session_id: str, 
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a session."""
        ...
    
    async def get_session_info(self, session_id: str, context: Optional[Dict[str, Any]] = None) -> SessionInfo:
        """Get information about a session."""
        ...
    
    async def prepare_workspace(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Prepare workspace for worker operations."""
        ...
    
    async def close_workspace(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Close workspace and cleanup sessions."""
        ...


class BaseWorker(ABC):
    """Minimal base class for workers - provides only common utilities."""
    
    def __init__(self):
        self.instance_id = f"{self.__class__.__name__}-{id(self)}"
    
    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the worker name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the worker description."""
        pass
    
    @property
    def visibility(self) -> str:
        """Return the worker visibility."""
        return "protected"
    
    @property
    def run_in_executor(self) -> bool:
        """Return whether the worker should run in an executor."""
        return False
    
    @property
    def require_context(self) -> bool:
        """Return whether the worker requires a context."""
        return False
    
    @property
    def service_id(self) -> str:
        """Return the service id."""
        return f"{self.name.lower().replace(' ', '-')}-{self.instance_id}"
    
    # Workers must implement these methods directly
    @abstractmethod
    async def start(self, config: Union[WorkerConfig, Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new worker session."""
        pass
    
    @abstractmethod
    async def stop(self, session_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Stop a worker session."""
        pass
    
    @abstractmethod
    async def list_sessions(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> List[SessionInfo]:
        """List all sessions for a workspace."""
        pass
    
    @abstractmethod
    async def get_session_info(self, session_id: str, context: Optional[Dict[str, Any]] = None) -> SessionInfo:
        """Get information about a session."""
        pass
    
    @abstractmethod
    async def get_logs(
        self, 
        session_id: str, 
        type: Optional[str] = None,
        offset: int = 0, 
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a session."""
        pass
    
    @abstractmethod
    async def prepare_workspace(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Prepare workspace for worker operations."""
        pass
    
    @abstractmethod
    async def close_workspace(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Close workspace and cleanup sessions."""
        pass
    
    async def compile(self, manifest: Dict[str, Any], files: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Compile application manifest and files.
        
        Optional method that workers can implement to handle their own compilation logic.
        Default implementation returns manifest and files unchanged.
        """
        return manifest, files
    
    def get_worker_service(self) -> Dict[str, Any]:
        """Get the service configuration for registration."""
        return {
            "id": self.service_id,
            "name": self.name,
            "description": self.description,
            "type": "server-app-worker",
            "config": {
                "visibility": self.visibility,
                "run_in_executor": self.run_in_executor,
                "require_context": self.require_context,
            },
            "supported_types": self.supported_types,
            "start": self.start,
            "stop": self.stop,
            "list_sessions": self.list_sessions,
            "get_logs": self.get_logs,
            "get_session_info": self.get_session_info,
            "prepare_workspace": self.prepare_workspace,
            "close_workspace": self.close_workspace,
            "compile": self.compile,
            "shutdown": self.shutdown,
        }
    
    async def register_worker_service(self, server):
        """Register the worker service."""
        return await server.register_service(self.get_worker_service())
    
    async def shutdown(self) -> None:
        """Shutdown the worker - workers should override this if needed."""
        logger.info(f"Shutting down {self.name}...") 