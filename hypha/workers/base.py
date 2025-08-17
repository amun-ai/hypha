"""Base worker API for standardized worker implementation."""

import inspect
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Protocol, Callable

from pydantic import BaseModel, Field, field_serializer

logger = logging.getLogger(__name__)


async def safe_call_callback(callback: Optional[Callable], message: Dict[str, Any]) -> None:
    """Call a progress callback, handling None and sync/async functions.
    
    Args:
        callback: Optional callback function that may be sync or async
        message: Message dictionary to pass to the callback
    """
    if not callback:
        return
    
    if inspect.iscoroutinefunction(callback):
        await callback(message)
    else:
        callback(message)


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
    disable_ssl: bool = False
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

    @field_serializer("status")
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

    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new worker session."""
        ...

    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop a worker session."""
        ...

    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get logs for a session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        
        If type is specified, only items matching that type will be returned.
        """
        ...


    async def execute(
        self,
        session_id: str,
        script: str,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a script in the running session.
        
        This method allows interaction with running sessions by executing scripts.
        Different workers may implement this differently:
        - Conda worker: Execute Python code via Jupyter kernel
        - Browser worker: Execute JavaScript via Playwright
        - Other workers: May not implement this method
        
        Args:
            session_id: The session to execute in
            script: The script/code to execute
            config: Optional execution configuration
            progress_callback: Optional callback for execution progress
            context: Optional context information
            
        Returns:
            Execution results (format depends on worker implementation)
            
        Raises:
            NotImplementedError: If the worker doesn't support execution
        """
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
    def use_local_url(self) -> bool:
        """Return whether the worker should use local URLs.
        
        When True, the worker will receive local_base_url for server_url and app_files_base_url.
        This is typically True for workers running in the same cluster/host as the Hypha server
        (e.g., built-in workers, startup function workers).
        
        When False (default), the worker will receive public_base_url for external access.
        This is typically False for external workers started via CLI.
        """
        return False

    @property
    def service_id(self) -> str:
        """Return the service id."""
        return f"{self.instance_id}"

    # Workers must implement these methods directly
    @abstractmethod
    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new worker session."""
        pass

    @abstractmethod
    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop a worker session."""
        pass

    @abstractmethod
    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get logs for a session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        
        If type is specified, only items matching that type will be returned.
        """
        pass


    async def compile(
        self,
        manifest: Dict[str, Any],
        files: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Compile application manifest and files.

        Optional method that workers can implement to handle their own compilation logic.
        Default implementation returns manifest and files unchanged.
        """
        return manifest, files

    async def execute(
        self,
        session_id: str,
        script: str,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a script in the running session.
        
        Optional method that workers can implement to interact with running sessions.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"Worker {self.name} does not support the execute method"
        )

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
            "use_local_url": self.use_local_url,
            "start": self.start,
            "stop": self.stop,
            "get_logs": self.get_logs,
            "compile": self.compile,
            "execute": self.execute,
            "shutdown": self.shutdown,
        }

    async def register_worker_service(self, server):
        """Register the worker service."""
        return await server.register_service(self.get_worker_service())

    async def shutdown(self) -> None:
        """Shutdown the worker - workers should override this if needed."""
        logger.info(f"Shutting down {self.name}...")
