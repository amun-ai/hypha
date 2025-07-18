"""Base worker API for standardized worker implementation."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

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
    client_id: str
    app_id: str
    server_url: str
    public_base_url: str
    local_base_url: str
    workspace: str
    version: Optional[str] = None
    token: Optional[str] = None
    entry_point: Optional[str] = None
    app_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


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
    
    async def start(self, config: WorkerConfig) -> SessionInfo:
        """Start a new worker session."""
        ...
    
    async def stop(self, session_id: str) -> None:
        """Stop a worker session."""
        ...
    
    async def list_sessions(self, workspace: str) -> List[SessionInfo]:
        """List all sessions for a workspace."""
        ...
    
    async def get_logs(
        self, 
        session_id: str, 
        log_type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a session."""
        ...
    
    async def get_session_info(self, session_id: str) -> SessionInfo:
        """Get information about a session."""
        ...
    
    async def prepare_workspace(self, workspace: str) -> None:
        """Prepare workspace for worker operations."""
        ...
    
    async def close_workspace(self, workspace: str) -> None:
        """Close workspace and cleanup sessions."""
        ...


class BaseWorker(ABC):
    """Base class for all workers."""
    
    def __init__(self, server=None):
        self.server = server
        self.initialized = False
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}
        self.instance_id = f"{self.__class__.__name__}-{id(self)}"
    
    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        pass
    
    @property
    @abstractmethod
    def worker_name(self) -> str:
        """Return the worker name."""
        pass
    
    @property
    @abstractmethod
    def worker_description(self) -> str:
        """Return the worker description."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the worker."""
        if not self.initialized:
            if self.server:
                await self.server.register_service(self.get_service_config())
            await self._initialize_worker()
            self.initialized = True
    
    @abstractmethod
    async def _initialize_worker(self) -> None:
        """Worker-specific initialization."""
        pass
    
    async def start(self, config: Union[WorkerConfig, Dict[str, Any]]) -> SessionInfo:
        """Start a new worker session."""
        if not self.initialized:
            await self.initialize()
        
        # Handle both pydantic model and dict input for RPC compatibility
        if isinstance(config, dict):
            config = WorkerConfig(**config)
        
        session_id = self._create_session_id(config.workspace, config.client_id)
        
        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")
        
        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            app_id=config.app_id,
            workspace=config.workspace,
            client_id=config.client_id,
            status=SessionStatus.STARTING,
            app_type=config.app_type,
            entry_point=config.entry_point,
            created_at=datetime.now().isoformat(),
            metadata=config.metadata
        )
        
        self._sessions[session_id] = session_info
        
        try:
            # Worker-specific start logic
            session_data = await self._start_session(config)
            self._session_data[session_id] = session_data
            
            # Update session status
            session_info.status = SessionStatus.RUNNING
            logger.info(f"Started session {session_id}")
            
            return session_info
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to start session {session_id}: {e}")
            raise
    
    @abstractmethod
    async def _start_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Worker-specific session start logic."""
        pass
    
    async def stop(self, session_id: str) -> None:
        """Stop a worker session."""
        if session_id not in self._sessions:
            # For some workers (like A2A), sessions may auto-disconnect
            # and clean themselves up, so we shouldn't raise an error
            logger.warning(f"Session {session_id} not found for stopping, may have already been cleaned up")
            return
        
        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING
        
        try:
            await self._stop_session(session_id)
            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped session {session_id}")
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop session {session_id}: {e}")
            raise
        finally:
            # Cleanup
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)
    
    @abstractmethod
    async def _stop_session(self, session_id: str) -> None:
        """Worker-specific session stop logic."""
        pass
    
    async def list_sessions(self, workspace: str) -> List[SessionInfo]:
        """List all sessions for a workspace."""
        return [
            session_info for session_info in self._sessions.values()
            if session_info.workspace == workspace
        ]
    
    async def get_session_info(self, session_id: str) -> SessionInfo:
        """Get information about a session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return self._sessions[session_id]
    
    async def get_logs(
        self, 
        session_id: str, 
        type: Optional[str] = None,  # Legacy parameter name
        log_type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        # Handle both legacy 'type' and new 'log_type' parameter names
        actual_log_type = log_type if log_type is not None else type
        
        return await self._get_session_logs(session_id, actual_log_type, offset, limit)
    
    @abstractmethod
    async def _get_session_logs(
        self, 
        session_id: str, 
        log_type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Worker-specific log retrieval."""
        pass
    
    async def prepare_workspace(self, workspace: str) -> None:
        """Prepare workspace for worker operations."""
        logger.info(f"Preparing workspace {workspace} for {self.worker_name}")
        await self._prepare_workspace(workspace)
    
    async def _prepare_workspace(self, workspace: str) -> None:
        """Worker-specific workspace preparation."""
        pass
    
    async def close_workspace(self, workspace: str) -> None:
        """Close workspace and cleanup sessions."""
        logger.info(f"Closing workspace {workspace} for {self.worker_name}")
        
        # Stop all sessions for this workspace
        sessions_to_stop = [
            session_id for session_id, session_info in self._sessions.items()
            if session_info.workspace == workspace
        ]
        
        for session_id in sessions_to_stop:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop session {session_id}: {e}")
        
        await self._close_workspace(workspace)
    
    async def _close_workspace(self, workspace: str) -> None:
        """Worker-specific workspace cleanup."""
        pass
    
    def _create_session_id(self, workspace: str, client_id: str) -> str:
        """Create a standardized session ID."""
        return f"{workspace}/{client_id}"
    
    def get_service_config(self) -> Dict[str, Any]:
        """Get the service configuration for registration."""
        return {
            "id": f"{self.worker_name.lower().replace(' ', '-')}-{self.instance_id}",
            "name": self.worker_name,
            "description": self.worker_description,
            "type": "server-app-worker",
            "config": {
                "visibility": "protected",
                "run_in_executor": True,
            },
            "supported_types": self.supported_types,
            "start": self.start,
            "stop": self.stop,
            "list_sessions": self.list_sessions,
            "get_logs": self.get_logs,
            "get_session_info": self.get_session_info,
            "prepare_workspace": self.prepare_workspace,
            "close_workspace": self.close_workspace,
            "shutdown": self._shutdown,
        }
    
    async def _shutdown(self) -> None:
        """Shutdown the worker."""
        logger.info(f"Shutting down {self.worker_name}...")
        
        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop session {session_id}: {e}")
        
        # Worker-specific shutdown
        await self._shutdown_worker()
        
        self.initialized = False
        logger.info(f"{self.worker_name} shutdown complete")
    
    async def _shutdown_worker(self) -> None:
        """Worker-specific shutdown logic."""
        pass 