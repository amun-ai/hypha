"""Worker Manager for maintaining persistent worker connections."""

import asyncio
import logging
import time
import traceback
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager

from hypha.core import UserInfo

logger = logging.getLogger(__name__)


class WorkerConnection:
    """Represents a persistent connection to a worker."""
    
    def __init__(self, worker_id: str, worker_service: Any, workspace_interface: Any):
        self.worker_id = worker_id
        self.worker_service = worker_service
        self.workspace_interface = workspace_interface
        self.last_used = time.time()
        self.ref_count = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a reference to this worker."""
        async with self.lock:
            self.ref_count += 1
            self.last_used = time.time()
        return self.worker_service
    
    async def release(self):
        """Release a reference to this worker."""
        async with self.lock:
            self.ref_count = max(0, self.ref_count - 1)
            return self.ref_count
    
    async def close(self):
        """Close the worker connection."""
        try:
            if self.workspace_interface:
                await self.workspace_interface.disconnect()
                logger.debug("Closed connection to worker %s", self.worker_id)
        except Exception as e:
            logger.warning("Error closing worker connection %s: %s", self.worker_id, e)


class WorkerManager:
    """Manages persistent connections to workers."""
    
    def __init__(self, store, cleanup_interval: int = 300, idle_timeout: int = 600, cleanup_worker_sessions_callback=None):
        self.store = store
        self.cleanup_interval = cleanup_interval  # 5 minutes
        self.idle_timeout = idle_timeout  # 10 minutes
        self.cleanup_worker_sessions_callback = cleanup_worker_sessions_callback
        
        # worker_id -> WorkerConnection
        self._connections: Dict[str, WorkerConnection] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._shutdown = False
        
        # Track workspaces that have requested workers
        self._monitored_workspaces: Dict[str, bool] = {}  # workspace_id -> is_monitoring
        self._workspace_listeners: Dict[str, Any] = {}  # workspace_id -> workspace_api
    
    async def start(self):
        """Start the worker manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("WorkerManager started")
    
    async def shutdown(self):
        """Shutdown the worker manager."""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._lock:
            for connection in self._connections.values():
                await connection.close()
            self._connections.clear()
        
        # Close workspace listeners
        for workspace_id, workspace_interface in self._workspace_listeners.items():
            try:
                await workspace_interface.disconnect()
                logger.debug("Closed workspace listener for %s", workspace_id)
            except Exception as e:
                logger.debug("Error closing workspace listener for %s: %s", workspace_id, e)
        
        self._workspace_listeners.clear()
        self._monitored_workspaces.clear()
        
        logger.info("WorkerManager shutdown complete")
    
    async def _ensure_workspace_monitoring(self, workspace: str, context: Optional[Dict[str, Any]] = None):
        """Ensure we're monitoring a workspace for service changes."""
        if workspace in self._monitored_workspaces:
            return  # Already monitoring
        
        # Connect to workspace
        if context and workspace != "public":
            user_info = UserInfo.model_validate(context["user"])
            workspace_interface = await self.store.get_workspace_interface(
                user_info, workspace
            ).__aenter__()
            
            # Set up event listeners - this will handle ALL workers in this workspace
            await self._setup_workspace_listeners(workspace, workspace_interface)
            
            # Do initial worker discovery
            await self._discover_workspace_workers(workspace, workspace_interface)
            
            self._workspace_listeners[workspace] = workspace_interface
        
        self._monitored_workspaces[workspace] = True
    
    async def _setup_workspace_listeners(self, workspace: str, workspace_api: Any):
        """Set up event listeners for workspace changes."""
        # Listen for service added events
        async def on_service_added(data):
            await self._handle_service_added(workspace, data)
        
        # Listen for service removed events
        async def on_service_removed(data):
            await self._handle_service_removed(workspace, data)
        
        # Listen for client disconnected events
        async def on_client_disconnected(data):
            await self._handle_client_disconnected(workspace, data)
        
        # Set up the listeners first
        workspace_api.on("service_added", on_service_added)
        workspace_api.on("service_removed", on_service_removed)
        workspace_api.on("client_disconnected", on_client_disconnected)
        
        # Then subscribe to the events
        logger.info("🎧 Subscribing to events for workspace %s", workspace)
        try:
            await workspace_api.subscribe(["service_added", "service_removed", "client_disconnected"])
            logger.info("✅ Successfully subscribed to events for workspace %s", workspace)
        except Exception as e:
            logger.error("❌ Failed to subscribe to events for workspace %s: %s", workspace, e)
            raise
        
        logger.info("✅ Set up event listeners for workspace %s", workspace)

    async def _discover_workspace_workers(self, workspace: str, workspace_api: Any):
        """Discover and connect to existing workers in the workspace."""
        # List all services to find workers
        services = await workspace_api.list_services({
            "type": "server-app-worker",
            "workspace": workspace
        })
        
        logger.info("Discovered %d workers in workspace %s", len(services), workspace)

    async def _handle_service_added(self, workspace: str, data: Dict[str, Any]):
        """Handle service added event."""
        service = data.get("service", {})
        service_type = service.get("type")
        
        if service_type == "server-app-worker":
            service_id = service.get("id")
            logger.info("New worker discovered in workspace %s: %s", workspace, service_id)
            # Worker is available, but we don't pre-connect. Connection will be made on-demand.
            
    async def _handle_service_removed(self, workspace: str, data: Dict[str, Any]):
        """Handle service removed event."""
        service = data.get("service", {})
        service_type = service.get("type")
        
        if service_type == "server-app-worker":
            service_id = service.get("id")
            logger.info("Worker removed from workspace %s: %s", workspace, service_id)
            
            # Clean up any existing connections to this worker
            await self.force_cleanup_worker(service_id)
            
            # Notify app controller to clean up sessions for this worker
            if self.cleanup_worker_sessions_callback:
                try:
                    await self.cleanup_worker_sessions_callback(service_id)
                    logger.info("Notified app controller about worker death: %s", service_id)
                except Exception as e:
                    logger.error("Failed to notify app controller about worker death %s: %s", service_id, e)
                
    async def _handle_client_disconnected(self, workspace: str, data: Dict[str, Any]):
        """Handle client disconnected event."""
        client_id = data.get("client_id")
        if not client_id:
            return
            
        # Clean up all connections for services from this client
        async with self._lock:
            to_remove = []
            for worker_id in self._connections.keys():
                # Check if worker_id belongs to the disconnected client
                if client_id in worker_id:
                    to_remove.append(worker_id)
            
            for worker_id in to_remove:
                connection = self._connections.pop(worker_id, None)
                if connection:
                    asyncio.create_task(connection.close())
                
                # Notify app controller to clean up sessions for this worker
                if self.cleanup_worker_sessions_callback:
                    await self.cleanup_worker_sessions_callback(worker_id)

    def _determine_workspace(self, from_workspace: Optional[str], context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Determine which workspace to use based on parameters."""
        if from_workspace is not None:
            return from_workspace  # Use explicitly specified workspace
        elif context:
            return context.get("ws")  # Use workspace from context
        else:
            return "public"  # Default to public
    
    @asynccontextmanager
    async def get_worker(
        self, 
        worker_id: str, 
        context: Optional[Dict[str, Any]] = None,
        from_workspace: Optional[str] = None
    ):
        """Get a worker with automatic connection management.
        
        Args:
            worker_id: The ID of the worker to get
            context: Context containing user and workspace info
            from_workspace: Specific workspace to get worker from. If None, uses workspace from context.
                          If "public", uses public API.
            
        Yields:
            worker_service: The worker service proxy
        """
        connection = None
        try:
            # Determine workspace and ensure monitoring
            workspace = self._determine_workspace(from_workspace, context)
            if workspace and workspace != "public":
                await self._ensure_workspace_monitoring(workspace, context)
            
            # Get or create connection
            connection = await self._get_or_create_connection(
                worker_id, context, from_workspace
            )
            
            if connection is None:
                raise ValueError(f"Worker {worker_id} not found")
            
            # Acquire reference
            worker_service = await connection.acquire()
            
            yield worker_service
        except Exception as e:
            logger.error("Failed to get worker %s: %s", worker_id, repr(e))
            logger.error("Full traceback: %s", traceback.format_exc())
            raise
        finally:
            # Always release reference
            if connection:
                ref_count = await connection.release()
                logger.debug("Released worker %s, ref_count: %s", worker_id, ref_count)
    
    async def get_worker_ref(
        self, 
        worker_id: str, 
        context: Optional[Dict[str, Any]] = None,
        from_workspace: Optional[str] = None
    ) -> Optional[Any]:
        """Get a worker reference with persistent connection.
        
        The caller must call release_worker_ref() when done with the worker.
        
        Args:
            worker_id: The ID of the worker to get
            context: Context containing user and workspace info
            from_workspace: Specific workspace to get worker from. If None, uses workspace from context.
                          If "public", uses public API.
            
        Returns:
            worker_service: The worker service proxy, or None if not found
        """
        # Determine workspace and ensure monitoring
        workspace = self._determine_workspace(from_workspace, context)
        if workspace and workspace != "public":
            await self._ensure_workspace_monitoring(workspace, context)
        
        # Get or create connection
        connection = await self._get_or_create_connection(
            worker_id, context, from_workspace
        )
        
        if connection is None:
            return None
        
        # Acquire reference
        worker_service = await connection.acquire()
        
        # Store connection reference in the worker service for later cleanup
        if hasattr(worker_service, '_hypha_worker_id'):
            # Worker already has connection info, just return it
            pass
        else:
            # Add connection info to worker for cleanup
            worker_service._hypha_worker_id = worker_id
            worker_service._hypha_worker_manager = self
        
        return worker_service
        
    
    async def release_worker_ref(self, worker_id: str) -> None:
        """Release a worker reference obtained with get_worker_ref()."""
        async with self._lock:
            if worker_id in self._connections:
                connection = self._connections[worker_id]
                await connection.release()
    
    async def _get_or_create_connection(
        self, 
        worker_id: str, 
        context: Optional[Dict[str, Any]] = None,
        from_workspace: Optional[str] = None
    ) -> Optional[WorkerConnection]:
        """Get existing connection or create a new one."""
        async with self._lock:
            # Check if we already have a connection
            if worker_id in self._connections:
                connection = self._connections[worker_id]
                logger.debug("Reusing existing connection to worker %s", worker_id)
                return connection
            
            # Create new connection
            logger.debug("Creating new connection to worker %s", worker_id)
            connection = await self._create_connection(worker_id, context, from_workspace)
            
            if connection:
                self._connections[worker_id] = connection
            
            return connection
    
    async def _create_connection(
        self,
        worker_id: str,
        context: Optional[Dict[str, Any]] = None,
        from_workspace: Optional[str] = None
    ) -> Optional[WorkerConnection]:
        """Create a new worker connection."""
        if from_workspace == "public" or (from_workspace is None and not context):
            # Use public API
            server = await self.store.get_public_api()
            worker_service = await server.get_service(worker_id)
            return WorkerConnection(worker_id, worker_service, None)
        else:
            # Use workspace interface (persistent connection)
            # Determine which workspace to use
            if from_workspace is not None and from_workspace != "public":
                workspace = from_workspace
            else:
                workspace = context.get("ws")
            
            user_info = UserInfo.model_validate(context["user"])
            
            # Create persistent workspace interface (don't use async with!)
            workspace_interface = await self.store.get_workspace_interface(
                user_info, workspace
            ).__aenter__()
            
            worker_service = await workspace_interface.get_service(worker_id)
            return WorkerConnection(worker_id, worker_service, workspace_interface)
    
    async def _cleanup_loop(self):
        """Periodically cleanup idle connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop: %s", e)
    
    async def _cleanup_idle_connections(self):
        """Clean up idle connections."""
        current_time = time.time()
        to_remove = []
        
        async with self._lock:
            for worker_id, connection in self._connections.items():
                async with connection.lock:
                    # Remove if no references and idle for too long
                    if (connection.ref_count == 0 and 
                        current_time - connection.last_used > self.idle_timeout):
                        to_remove.append(worker_id)
            
            # Remove idle connections
            for worker_id in to_remove:
                connection = self._connections.pop(worker_id)
                asyncio.create_task(connection.close())
                logger.debug("Cleaned up idle worker connection: %s", worker_id)
    
    async def force_cleanup_worker(self, worker_id: str):
        """Force cleanup of a specific worker connection."""
        async with self._lock:
            if worker_id in self._connections:
                connection = self._connections.pop(worker_id)
                await connection.close()
                logger.info("Force cleaned up worker connection: %s", worker_id)
    
