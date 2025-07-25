"""Conda Environment Worker for executing Python code in isolated conda environments."""

import asyncio
import hashlib
import httpx
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hypha.workers.base import BaseWorker, WorkerConfig, SessionStatus, SessionInfo, SessionNotFoundError, WorkerError
from hypha.workers.conda_env_executor import CondaEnvExecutor, ExecutionResult

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("conda_env")
logger.setLevel(LOGLEVEL)

# Cache configuration
DEFAULT_CACHE_DIR = os.path.expanduser("~/.hypha_conda_cache")
MAX_CACHE_SIZE = 10  # Maximum number of cached environments
CACHE_MAX_AGE_DAYS = 30  # Maximum age for cached environments


class EnvironmentCache:
    """Manages cached conda environments with LRU eviction."""
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, max_size: int = MAX_CACHE_SIZE):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.index = {}
        else:
            self.index = {}
    
    def _save_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _compute_env_hash(self, packages: List[str], channels: List[str]) -> str:
        """Compute hash for environment specification."""
        # Sort packages and channels for consistent hashing
        # Use JSON serialization as sort key to handle mixed types (strings and dicts)
        sorted_packages = sorted(packages, key=lambda x: json.dumps(x, sort_keys=True)) if packages else []
        sorted_channels = sorted(channels) if channels else []
        
        # Create a canonical string representation
        env_spec = {
            'packages': sorted_packages,
            'channels': sorted_channels
        }
        env_str = json.dumps(env_spec, sort_keys=True)
        return hashlib.sha256(env_str.encode()).hexdigest()
    
    def get_cached_env(self, packages: List[str], channels: List[str]) -> Optional[Path]:
        """Get cached environment path if it exists and is valid."""
        env_hash = self._compute_env_hash(packages, channels)
        
        if env_hash in self.index:
            cache_entry = self.index[env_hash]
            env_path = Path(cache_entry['path'])
            
            # Check if environment still exists and is valid
            if env_path.exists() and (env_path / 'bin' / 'python').exists():
                # Update last access time for LRU
                cache_entry['last_accessed'] = time.time()
                self._save_index()
                return env_path
            else:
                # Remove invalid entry
                del self.index[env_hash]
                self._save_index()
        
        return None
    
    def add_cached_env(self, packages: List[str], channels: List[str], env_path: Path):
        """Add environment to cache."""
        env_hash = self._compute_env_hash(packages, channels)
        
        # Evict old entries if cache is full
        self._evict_if_needed()
        
        self.index[env_hash] = {
            'path': str(env_path),
            'packages': packages,
            'channels': channels,
            'created_at': time.time(),
            'last_accessed': time.time()
        }
        self._save_index()
    
    def _evict_if_needed(self):
        """Evict old entries using LRU policy."""
        # Remove entries older than max age
        current_time = time.time()
        max_age_seconds = CACHE_MAX_AGE_DAYS * 24 * 60 * 60
        
        to_remove = []
        for env_hash, entry in self.index.items():
            if current_time - entry['created_at'] > max_age_seconds:
                to_remove.append(env_hash)
        
        for env_hash in to_remove:
            self._remove_cache_entry(env_hash)
        
        # If still over limit, remove least recently used entries
        while len(self.index) >= self.max_size:
            # Find least recently accessed entry
            oldest_hash = min(
                self.index.keys(),
                key=lambda h: self.index[h]['last_accessed']
            )
            self._remove_cache_entry(oldest_hash)
    
    def _remove_cache_entry(self, env_hash: str):
        """Remove a cache entry and cleanup the environment."""
        if env_hash in self.index:
            entry = self.index[env_hash]
            env_path = Path(entry['path'])
            
            # Remove the environment directory
            if env_path.exists():
                try:
                    shutil.rmtree(env_path)
                    logger.info(f"Removed cached environment: {env_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove cached environment {env_path}: {e}")
            
            del self.index[env_hash]
            self._save_index()
    
    def cleanup_all(self):
        """Remove all cached environments."""
        for env_hash in list(self.index.keys()):
            self._remove_cache_entry(env_hash)


class CondaEnvWorker(BaseWorker):
    """Conda environment worker for executing Python code in isolated conda environments."""
    
    instance_counter: int = 0
    
    def __init__(self, server):
        """Initialize the conda environment worker."""
        super().__init__(server)
        self.controller_id = str(CondaEnvWorker.instance_counter)
        CondaEnvWorker.instance_counter += 1
        
        # Session management
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}
        
        # Environment cache
        self._env_cache = EnvironmentCache()
    
    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["python-conda"]
    
    @property
    def worker_name(self) -> str:
        """Return the worker name."""
        return "Conda Environment Worker"
    
    @property
    def worker_description(self) -> str:
        """Return the worker description."""
        return "A worker for executing Python code in isolated conda environments with package management"
    
    async def compile(self, manifest: dict, files: list, config: dict = None) -> tuple[dict, list]:
        """Compile conda environment application - validate manifest."""
        # Validate manifest has required fields
        if "packages" not in manifest and "dependencies" not in manifest:
            logger.warning("No packages or dependencies specified in manifest")
        # Set entry point is not set
        if "entry_point" not in manifest:
            manifest["entry_point"] = "main.py"
        
        # Normalize packages field
        packages = manifest.get("packages", manifest.get("dependencies", []))
        channels = manifest.get("channels", ["conda-forge"])
        
        # Ensure packages and channels are lists
        if isinstance(packages, str):
            packages = [packages]
        if isinstance(channels, str):
            channels = [channels]
        
        # Update manifest with normalized values
        manifest["packages"] = packages
        manifest["channels"] = channels
        
        return manifest, files
    
    async def start(self, config: Union[WorkerConfig, Dict[str, Any]]) -> str:
        """Start a new conda environment session."""
        # Handle both pydantic model and dict input for RPC compatibility
        if isinstance(config, dict):
            config = WorkerConfig(**config)
        
        session_id = config.id
        
        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")
        
        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            app_id=config.app_id,
            workspace=config.workspace,
            client_id=config.client_id,
            status=SessionStatus.STARTING,
            app_type=config.manifest.get("type", "conda-env"),
            entry_point=config.entry_point,
            created_at=datetime.now().isoformat(),
            metadata=config.manifest
        )
        
        self._sessions[session_id] = session_info
        
        try:
            # Get the Python script
            script_url = f"{config.app_files_base_url}/{config.manifest['entry_point']}?use_proxy=true"
            async with httpx.AsyncClient() as client:
                response = await client.get(script_url, headers={"Authorization": f"Bearer {config.token}"})
                response.raise_for_status()
                script = response.text
            
            # Start the conda environment session
            session_data = await self._start_conda_session(script, config)
            self._session_data[session_id] = session_data
            
            # Update session status
            session_info.status = SessionStatus.RUNNING
            logger.info(f"Started conda environment session {session_id}")
            
            return session_id
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to start conda environment session {session_id}: {e}")
            # Clean up failed session
            self._sessions.pop(session_id, None)
            raise
    
    async def _start_conda_session(self, script: str, config: WorkerConfig) -> Dict[str, Any]:
        """Start a conda environment session."""
        assert script is not None, "Script is not found"
        
        # Extract environment specification from manifest
        packages = config.manifest.get("packages", config.manifest.get("dependencies", []))
        channels = config.manifest.get("channels", ["conda-forge"])
        
        # Ensure lists
        if isinstance(packages, str):
            packages = [packages]
        if isinstance(channels, str):
            channels = [channels]
        
        # Check if we have a cached environment
        cached_env_path = self._env_cache.get_cached_env(packages, channels)
        
        if cached_env_path:
            logger.info(f"Using cached conda environment: {cached_env_path}")
            executor = CondaEnvExecutor({
                'name': 'cached_env',
                'channels': channels,
                'dependencies': packages
            }, env_dir=cached_env_path.parent)
            executor.env_path = cached_env_path
            executor._is_extracted = True
        else:
            # Create new temporary environment
            logger.info(f"Creating new conda environment with packages: {packages}")
            executor = CondaEnvExecutor.create_temp_env(
                packages=packages,
                channels=channels,
                name=f"hypha-session-{config.id}"
            )
            
            # Cache the environment after creation
            # We need to run the extraction first to create the environment
            await asyncio.get_event_loop().run_in_executor(
                None, executor._extract_env
            )
            
            # Add to cache
            self._env_cache.add_cached_env(packages, channels, executor.env_path)
            logger.info(f"Cached new conda environment: {executor.env_path}")
        
        # Always execute the script during startup to run initialization code
        # This ensures print statements and setup code are executed
        result = await asyncio.get_event_loop().run_in_executor(
            None, executor.execute, script, None
        )
        
        # Determine if script has execute function for later interactive calls
        needs_execute_function = "def execute(" in script or "async def execute(" in script
        
        # Store logs
        logs = {
            "stdout": [result.stdout] if result.stdout else [],
            "stderr": [result.stderr] if result.stderr else [],
            "info": []
        }
        
        if result.success:
            logs["info"].append(f"Conda environment session started successfully")
            if (result.timing and 
                hasattr(result.timing, 'env_setup_time') and 
                hasattr(result.timing, 'execution_time') and
                isinstance(result.timing.env_setup_time, (int, float)) and
                isinstance(result.timing.execution_time, (int, float))):
                logs["info"].append(f"Environment setup time: {result.timing.env_setup_time:.2f}s")
                logs["info"].append(f"Execution time: {result.timing.execution_time:.2f}s")
        else:
            logs["error"] = [result.error] if result.error else ["Unknown error occurred"]
        
        return {
            "executor": executor,
            "script": script,
            "packages": packages,
            "channels": channels,
            "needs_execute_function": needs_execute_function,
            "result": result,
            "logs": logs
        }
    
    async def execute_code(self, session_id: str, input_data: Any = None) -> ExecutionResult:
        """Execute code in a conda environment session with optional input data."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Conda environment session {session_id} not found")
        
        session_data = self._session_data.get(session_id)
        if not session_data or not session_data.get("executor"):
            raise WorkerError(f"No executor available for session {session_id}")
        
        executor = session_data["executor"]
        script = session_data["script"]
        
        try:
            # Execute the script with input data
            result = await asyncio.get_event_loop().run_in_executor(
                None, executor.execute, script, input_data
            )
            
            # Update logs
            if session_data["logs"] is None:
                session_data["logs"] = {"stdout": [], "stderr": [], "info": [], "error": []}
            
            # Ensure all log types exist
            for log_type in ["stdout", "stderr", "info", "error"]:
                if log_type not in session_data["logs"]:
                    session_data["logs"][log_type] = []
            
            if result.success:
                if result.stdout:
                    session_data["logs"]["stdout"].append(result.stdout)
                session_data["logs"]["info"].append(f"Code executed successfully at {datetime.now().isoformat()}")
            else:
                if result.stderr:
                    session_data["logs"]["stderr"].append(result.stderr)
                if result.error:
                    session_data["logs"]["error"].append(result.error)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute code in session {session_id}: {e}")
            result = ExecutionResult(success=False, error=str(e))
            if session_data.get("logs"):
                # Ensure error log type exists
                if "error" not in session_data["logs"]:
                    session_data["logs"]["error"] = []
                session_data["logs"]["error"].append(str(e))
            return result
    
    async def stop(self, session_id: str) -> None:
        """Stop a conda environment session."""
        if session_id not in self._sessions:
            logger.warning(f"Conda environment session {session_id} not found for stopping")
            return
        
        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING
        
        try:
            # Cleanup session data
            session_data = self._session_data.get(session_id)
            if session_data and session_data.get("executor"):
                executor = session_data["executor"]
                # Note: We don't cleanup the executor environment since it's cached
                # The cache manager will handle cleanup based on LRU policy
            
            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped conda environment session {session_id}")
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop conda environment session {session_id}: {e}")
            raise
        finally:
            # Cleanup
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)
    
    async def list_sessions(self, workspace: str) -> List[SessionInfo]:
        """List all conda environment sessions for a workspace."""
        return [
            session_info for session_info in self._sessions.values()
            if session_info.workspace == workspace
        ]
    
    async def get_session_info(self, session_id: str) -> SessionInfo:
        """Get information about a conda environment session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Conda environment session {session_id} not found")
        return self._sessions[session_id]
    
    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a conda environment session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Conda environment session {session_id} not found")
        
        session_data = self._session_data.get(session_id)
        if not session_data:
            return {} if type is None else []
        
        logs = session_data.get("logs", {})
        
        if type:
            target_logs = logs.get(type, [])
            end_idx = len(target_logs) if limit is None else min(offset + limit, len(target_logs))
            return target_logs[offset:end_idx]
        else:
            result = {}
            for log_type_key, log_entries in logs.items():
                end_idx = len(log_entries) if limit is None else min(offset + limit, len(log_entries))
                result[log_type_key] = log_entries[offset:end_idx]
            return result
    
    async def take_screenshot(self, session_id: str, format: str = "png") -> bytes:
        """Take a screenshot - not supported for conda environment sessions."""
        raise NotImplementedError("Screenshots not supported for conda environment sessions")
    
    async def prepare_workspace(self, workspace: str) -> None:
        """Prepare workspace for conda environment operations."""
        logger.info(f"Preparing workspace {workspace} for conda environment worker")
        pass
    
    async def close_workspace(self, workspace: str) -> None:
        """Close all conda environment sessions for a workspace."""
        logger.info(f"Closing workspace {workspace} for conda environment worker")
        
        # Stop all sessions for this workspace
        sessions_to_stop = [
            session_id for session_id, session_info in self._sessions.items()
            if session_info.workspace == workspace
        ]
        
        for session_id in sessions_to_stop:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop conda environment session {session_id}: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the conda environment worker."""
        logger.info("Shutting down conda environment worker...")
        
        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop conda environment session {session_id}: {e}")
        
        logger.info("Conda environment worker shutdown complete")
    
    def get_service(self):
        """Get the service configuration."""
        service_config = self.get_service_config()
        # Add conda environment specific methods
        service_config["execute_code"] = self.execute_code
        service_config["take_screenshot"] = self.take_screenshot
        return service_config


async def hypha_startup(server):
    """Hypha startup function to initialize conda environment worker."""
    worker = CondaEnvWorker(server)
    await server.register_service(worker.get_service_config())
    logger.info("Conda environment worker initialized and registered")


async def start_worker(server_url, workspace, token):
    """Start conda environment worker standalone."""
    from hypha_rpc import connect_to_server
    
    server = await connect_to_server(server_url=server_url, workspace=workspace, token=token)
    worker = CondaEnvWorker(server.rpc)
    logger.info(f"Conda environment worker started, server: {server_url}, workspace: {workspace}")
    
    return worker


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--token", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(start_worker(args.server_url, args.workspace, args.token))