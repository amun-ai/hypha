"""Conda Environment Worker for executing Python code in isolated conda environments."""

import asyncio
import functools
import hashlib
import httpx
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hypha.workers.base import (
    BaseWorker,
    WorkerConfig,
    SessionStatus,
    SessionInfo,
    SessionNotFoundError,
    WorkerError,
)
from hypha.workers.conda_executor import CondaEnvExecutor, ExecutionResult

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("conda")
logger.setLevel(LOGLEVEL)


def get_available_package_manager() -> str:
    """Detect available package manager, preferring mamba over conda.

    Returns:
        str: 'mamba' if available, otherwise 'conda'

    Raises:
        RuntimeError: If neither mamba nor conda is available
    """
    # Check for mamba first (faster alternative)
    try:
        result = subprocess.run(
            ["mamba", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info(f"Detected mamba package manager: {result.stdout.strip()}")
            return "mamba"
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        pass

    # Fall back to conda
    try:
        result = subprocess.run(
            ["conda", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info(f"Detected conda package manager: {result.stdout.strip()}")
            return "conda"
    except (
        subprocess.TimeoutExpired,
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        pass

    raise RuntimeError(
        "Neither mamba nor conda package manager found. Please install conda or mamba."
    )


# Cache configuration
DEFAULT_CACHE_DIR = os.path.expanduser("~/.hypha_conda_cache")
MAX_CACHE_SIZE = 10  # Maximum number of cached environments
CACHE_MAX_AGE_DAYS = 30  # Maximum age for cached environments


class EnvironmentCache:
    """Manages cached conda environments with LRU eviction."""

    def __init__(
        self, cache_dir: str = DEFAULT_CACHE_DIR, max_size: int = MAX_CACHE_SIZE
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()

    def _load_index(self):
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    self.index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache index: {e}")

    def _compute_env_hash(self, dependencies: List[str], channels: List[str]) -> str:
        """Compute hash for environment specification."""
        # Sort dependencies and channels for consistent hashing
        # Use JSON serialization as sort key to handle mixed types (strings and dicts)
        sorted_dependencies = (
            sorted(dependencies, key=lambda x: json.dumps(x, sort_keys=True))
            if dependencies
            else []
        )
        sorted_channels = sorted(channels) if channels else []

        # Create a canonical string representation
        env_spec = {"dependencies": sorted_dependencies, "channels": sorted_channels}
        env_str = json.dumps(env_spec, sort_keys=True)
        return hashlib.sha256(env_str.encode()).hexdigest()

    def get_cached_env(
        self, dependencies: List[str], channels: List[str]
    ) -> Optional[Path]:
        """Get cached environment path if it exists and is valid."""
        env_hash = self._compute_env_hash(dependencies, channels)

        if env_hash in self.index:
            cache_entry = self.index[env_hash]
            env_path = Path(cache_entry["path"])

            # Check if environment still exists and is valid
            if env_path.exists() and (env_path / "bin" / "python").exists():
                # Update last access time for LRU
                cache_entry["last_accessed"] = time.time()
                self._save_index()
                return env_path
            else:
                # Remove invalid entry
                del self.index[env_hash]
                self._save_index()

        return None

    def add_cached_env(
        self, dependencies: List[str], channels: List[str], env_path: Path
    ):
        """Add environment to cache."""
        env_hash = self._compute_env_hash(dependencies, channels)

        # Evict old entries if cache is full
        self._evict_if_needed()

        self.index[env_hash] = {
            "path": str(env_path),
            "dependencies": dependencies,
            "channels": channels,
            "created_at": time.time(),
            "last_accessed": time.time(),
        }
        self._save_index()

    def _evict_if_needed(self):
        """Evict old entries using LRU policy."""
        # Remove entries older than max age
        current_time = time.time()
        max_age_seconds = CACHE_MAX_AGE_DAYS * 24 * 60 * 60

        to_remove = []
        for env_hash, entry in self.index.items():
            if current_time - entry["created_at"] > max_age_seconds:
                to_remove.append(env_hash)

        for env_hash in to_remove:
            self._remove_cache_entry(env_hash)

        # If still over limit, remove least recently used entries
        while len(self.index) >= self.max_size:
            # Find least recently accessed entry
            oldest_hash = min(
                self.index.keys(), key=lambda h: self.index[h]["last_accessed"]
            )
            self._remove_cache_entry(oldest_hash)

    def _remove_cache_entry(self, env_hash: str):
        """Remove a cache entry and cleanup the environment."""
        if env_hash in self.index:
            entry = self.index[env_hash]
            env_path = Path(entry["path"])

            # Remove the environment directory
            if env_path.exists():
                try:
                    shutil.rmtree(env_path)
                    logger.info(f"Removed cached environment: {env_path}")
                except OSError as e:
                    logger.warning(
                        f"Failed to remove cached environment {env_path}: {e}"
                    )

            del self.index[env_hash]
            self._save_index()

    def cleanup_all(self):
        """Remove all cached environments."""
        for env_hash in list(self.index.keys()):
            self._remove_cache_entry(env_hash)


class CondaWorker(BaseWorker):
    """Conda environment worker for executing Python code in isolated conda environments."""

    instance_counter: int = 0

    def __init__(self):
        """Initialize the conda environment worker."""
        super().__init__()
        self.controller_id = str(CondaWorker.instance_counter)
        CondaWorker.instance_counter += 1

        # Detect available package manager (mamba preferred over conda)
        try:
            self.package_manager = get_available_package_manager()
        except RuntimeError as e:
            logger.error(f"Package manager detection failed: {e}")
            raise

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
    def name(self) -> str:
        """Return the worker name."""
        return f"Conda Environment Worker (using {self.package_manager})"

    @property
    def description(self) -> str:
        """Return the worker description."""
        return f"A worker for executing Python code in isolated conda environments with package management using {self.package_manager}"

    @property
    def require_context(self) -> bool:
        """Return whether the worker requires a context."""
        return True

    async def compile(
        self,
        manifest: dict,
        files: list,
        config: dict = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, list]:
        """Compile conda environment application - validate manifest."""
        # Validate manifest has required fields
        if "dependencies" not in manifest and "dependencies" not in manifest:
            logger.warning("No dependencies or dependencies specified in manifest")
        # Set entry point is not set
        if "entry_point" not in manifest:
            manifest["entry_point"] = "main.py"

        # Normalize dependencies field
        dependencies = manifest.get("dependencies", manifest.get("dependencies", []))
        channels = manifest.get("channels", ["conda-forge"])

        # Ensure dependencies and channels are lists
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        if isinstance(channels, str):
            channels = [channels]

        # Always ensure pip is available
        if "pip" not in dependencies:
            dependencies.append("pip")

        # Always ensure hypha-rpc is available via pip
        hypha_rpc_found = False

        # Check if hypha-rpc is already specified
        for dep in dependencies:
            if isinstance(dep, dict) and "pip" in dep:
                pip_packages = dep["pip"]
                if isinstance(pip_packages, list):
                    if "hypha-rpc" in pip_packages:
                        hypha_rpc_found = True
                        break
                elif isinstance(pip_packages, str) and pip_packages == "hypha-rpc":
                    hypha_rpc_found = True
                    break
            elif isinstance(dep, str) and dep == "hypha-rpc":
                hypha_rpc_found = True
                break

        # Add hypha-rpc via pip if not found
        if not hypha_rpc_found:
            dependencies.append({"pip": ["hypha-rpc"]})
            logger.info("Automatically added hypha-rpc to dependencies via pip")

        # Update manifest with normalized values
        manifest["dependencies"] = dependencies
        manifest["channels"] = channels

        return manifest, files

    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new conda environment session."""
        # Handle both pydantic model and dict input for RPC compatibility
        if isinstance(config, dict):
            config = WorkerConfig(**config)

        session_id = config.id
        progress_callback = getattr(config, "progress_callback", None)

        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")

        # Report initial progress
        if progress_callback:
            progress_callback(
                {
                    "type": "info",
                    "message": f"Starting conda environment session {session_id}",
                }
            )

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
            metadata=config.manifest,
        )

        self._sessions[session_id] = session_info

        try:
            # Phase 1: Fetch application script
            if progress_callback:
                progress_callback(
                    {"type": "info", "message": "Fetching application script..."}
                )

            script_url = f"{config.app_files_base_url}/{config.manifest['entry_point']}?use_proxy=true"
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    script_url, headers={"Authorization": f"Bearer {config.token}"}
                )
                response.raise_for_status()
                script = response.text

            if progress_callback:
                progress_callback(
                    {
                        "type": "success",
                        "message": "Application script loaded successfully",
                    }
                )

            # Phase 2: Start the conda environment session (this is the long part)
            if progress_callback:
                progress_callback(
                    {
                        "type": "info",
                        "message": f"Setting up conda environment using {self.package_manager}...",
                    }
                )

            session_data = await self._start_conda_session(
                script, config, progress_callback
            )
            self._session_data[session_id] = session_data

            # Update session status
            session_info.status = SessionStatus.RUNNING

            if progress_callback:
                progress_callback(
                    {
                        "type": "success",
                        "message": f"Conda environment session {session_id} started successfully",
                    }
                )

            logger.info(f"Started conda environment session {session_id}")
            return session_id

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)

            if progress_callback:
                progress_callback(
                    {
                        "type": "error",
                        "message": f"Failed to start conda environment session: {str(e)}",
                    }
                )

            logger.error(f"Failed to start conda environment session {session_id}: {e}")
            # Clean up failed session
            self._sessions.pop(session_id, None)
            raise

    async def _start_conda_session(
        self, script: str, config: WorkerConfig, progress_callback=None
    ) -> Dict[str, Any]:
        """Start a conda environment session."""
        assert script is not None, "Script is not found"

        # Initialize logs - they will be populated by the background task and progress callback
        logs = {
            "stdout": [],
            "stderr": [],
            "info": [f"Conda environment session started successfully"],
            "error": [],
            "progress": [],  # Store real-time progress messages
        }

        # Create a progress callback wrapper that stores messages in logs for real-time access
        def progress_callback_wrapper(message):
            # Store the progress message in logs for real-time retrieval
            logs["progress"].append(f"{message['type'].upper()}: {message['message']}")

            # Also call the original progress callback if provided
            if progress_callback:
                progress_callback(message)

        # Extract environment specification from manifest
        dependencies = config.manifest.get(
            "dependencies", config.manifest.get("dependencies", [])
        )
        channels = config.manifest.get("channels", ["conda-forge"])

        # Ensure lists
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        if isinstance(channels, str):
            channels = [channels]

        # Phase 1: Check for cached environment
        progress_callback_wrapper(
            {"type": "info", "message": "Checking for cached conda environment..."}
        )

        cached_env_path = self._env_cache.get_cached_env(dependencies, channels)

        if cached_env_path:
            progress_callback_wrapper(
                {
                    "type": "success",
                    "message": f"Found cached environment: {cached_env_path}",
                }
            )

            logger.info(f"Using cached conda environment: {cached_env_path}")
            executor = CondaEnvExecutor(
                {
                    "name": "cached_env",
                    "channels": channels,
                    "dependencies": dependencies,
                },
                env_dir=cached_env_path.parent,
            )
            executor.env_path = cached_env_path
            executor._is_extracted = True
        else:
            # Phase 2: Create new environment (this is the long part)
            # Format dependencies for display (handle both strings and dicts)
            dep_strings = []
            for dep in dependencies[:3]:
                if isinstance(dep, dict):
                    # Handle pip dependencies like {'pip': ['package1', 'package2']}
                    for key, values in dep.items():
                        if isinstance(values, list):
                            dep_strings.append(f"{key}:[{', '.join(values)}]")
                        else:
                            dep_strings.append(f"{key}:{values}")
                else:
                    dep_strings.append(str(dep))

            deps_str = ", ".join(dep_strings) + ("..." if len(dependencies) > 3 else "")
            progress_callback_wrapper(
                {
                    "type": "info",
                    "message": f"Creating new conda environment with dependencies: {deps_str}",
                }
            )

            logger.info(
                f"Creating new conda environment with dependencies: {dependencies}"
            )

            # Sanitize the config ID to make it compatible with conda/mamba
            # Replace filesystem separators and other problematic characters
            sanitized_id = (
                config.id.replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-")
                .replace(" ", "-")
            )

            executor = CondaEnvExecutor.create_temp_env(
                dependencies=dependencies,
                channels=channels,
                name=f"hypha-session-{sanitized_id}",
            )

            # This is the long-running part - environment creation
            progress_callback_wrapper(
                {
                    "type": "info",
                    "message": f"Installing packages using {self.package_manager}... (this may take several minutes)",
                }
            )

            # Run environment creation directly (now async)
            try:
                setup_time = await executor._extract_env(progress_callback_wrapper)
            except Exception as e:
                progress_callback_wrapper(
                    {
                        "type": "error",
                        "message": f"Environment creation failed: {str(e)}",
                    }
                )
                raise

            progress_callback_wrapper(
                {
                    "type": "success",
                    "message": f"Environment created successfully in {setup_time:.1f}s",
                }
            )

            # Cache the environment after creation
            self._env_cache.add_cached_env(dependencies, channels, executor.env_path)
            logger.info(f"Cached new conda environment: {executor.env_path}")

        # Phase 3: Start initialization script in background (non-blocking)
        progress_callback_wrapper(
            {
                "type": "info",
                "message": "Starting initialization script in background...",
            }
        )

        # Start the script execution in background without waiting for completion
        # This allows long-running services (like FastAPI apps) to keep running
        hypha_config = {
            "server_url": config.server_url,
            "workspace": config.workspace,
            "client_id": config.client_id,
            "token": config.token,
        }

        # Determine if script has execute function for later interactive calls

        # Create a task to run the script in background with the wrapper callback
        script_task = asyncio.create_task(
            self._run_script_in_background(
                executor, script, hypha_config, progress_callback_wrapper, config.id
            )
        )

        progress_callback_wrapper(
            {
                "type": "success",
                "message": "Conda environment session started, initialization script running in background",
            }
        )

        return {
            "executor": executor,
            "script": script,
            "dependencies": dependencies,
            "channels": channels,
            "script_task": script_task,  # Keep reference to the background task
            "logs": logs,
            "hypha_config": hypha_config,
        }

    async def _run_script_in_background(
        self,
        executor: CondaEnvExecutor,
        script: str,
        hypha_config: dict,
        progress_callback=None,
        session_id: str = None,
    ):
        """Run the initialization script in background without blocking session startup."""
        try:
            if progress_callback:
                progress_callback(
                    {"type": "info", "message": "Executing initialization script..."}
                )

            # Execute the script in a thread pool to avoid blocking
            execute_func = functools.partial(executor.execute, script, hypha_config)
            result = await asyncio.get_event_loop().run_in_executor(None, execute_func)

            # Update session logs with the background execution results
            if session_id and session_id in self._session_data:
                session_data = self._session_data[session_id]
                logs = session_data.get("logs", {})

                if result.success:
                    if result.stdout:
                        logs.setdefault("stdout", []).append(result.stdout)
                    logs.setdefault("info", []).append(
                        "Background script execution completed successfully"
                    )
                    if result.timing:
                        logs.setdefault("info", []).append(
                            f"Background script execution time: {result.timing.execution_time:.2f}s"
                        )
                else:
                    if result.error:
                        logs.setdefault("error", []).append(result.error)
                    if result.stderr:
                        logs.setdefault("stderr", []).append(result.stderr)
                    logs.setdefault("error", []).append(
                        "Background script execution failed"
                    )

            if progress_callback:
                if result.success:
                    progress_callback(
                        {
                            "type": "success",
                            "message": "Initialization script executed successfully",
                        }
                    )
                else:
                    progress_callback(
                        {
                            "type": "error",
                            "message": f"Initialization script failed: {result.error or 'Unknown error'}",
                        }
                    )

            # Log the result for debugging
            if result.success:
                logger.info("Background script execution completed successfully")
                if result.timing:
                    logger.info(
                        f"Script execution time: {result.timing.execution_time:.2f}s"
                    )
            else:
                logger.error(f"Background script execution failed: {result.error}")

            return result

        except Exception as e:
            logger.error(f"Background script execution failed with exception: {e}")

            # Update session logs with the exception
            if session_id and session_id in self._session_data:
                session_data = self._session_data[session_id]
                logs = session_data.get("logs", {})
                logs.setdefault("error", []).append(
                    f"Background script execution failed: {str(e)}"
                )

            if progress_callback:
                progress_callback(
                    {
                        "type": "error",
                        "message": f"Background script execution failed: {str(e)}",
                    }
                )
            return None

    async def execute_code(
        self,
        session_id: str,
        code: str = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute code in a conda environment session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(
                f"Conda environment session {session_id} not found"
            )

        session_data = self._session_data.get(session_id)
        if not session_data or not session_data.get("executor"):
            raise WorkerError(f"No executor available for session {session_id}")

        executor = session_data["executor"]
        hypha_config = session_data.get("hypha_config", {})

        if not code:
            raise WorkerError(f"No code provided to execute")

        try:
            # Execute the provided code directly
            execute_func = functools.partial(executor.execute, code, hypha_config)
            result = await asyncio.get_event_loop().run_in_executor(None, execute_func)

            # Update logs
            if session_data["logs"] is None:
                session_data["logs"] = {
                    "stdout": [],
                    "stderr": [],
                    "info": [],
                    "error": [],
                }

            # Ensure all log types exist
            for log_type in ["stdout", "stderr", "info", "error"]:
                if log_type not in session_data["logs"]:
                    session_data["logs"][log_type] = []

            if result.success:
                if result.stdout:
                    session_data["logs"]["stdout"].append(result.stdout)
                session_data["logs"]["info"].append(
                    f"Code executed successfully at {datetime.now().isoformat()}"
                )
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

    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop a conda environment session."""
        if session_id not in self._sessions:
            logger.warning(
                f"Conda environment session {session_id} not found for stopping"
            )
            return

        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING

        try:
            # Cleanup session data
            session_data = self._session_data.get(session_id)
            if session_data:
                # Cancel the background script task if it's still running
                script_task = session_data.get("script_task")
                if script_task and not script_task.done():
                    logger.info(
                        f"Cancelling background script task for session {session_id}"
                    )
                    script_task.cancel()
                    try:
                        await script_task
                    except asyncio.CancelledError:
                        logger.info(
                            f"Background script task cancelled for session {session_id}"
                        )

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

    async def list_sessions(
        self, workspace: str, context: Optional[Dict[str, Any]] = None
    ) -> List[SessionInfo]:
        """List all conda environment sessions for a workspace."""
        return [
            session_info
            for session_info in self._sessions.values()
            if session_info.workspace == workspace
        ]

    async def get_session_info(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """Get information about a conda environment session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(
                f"Conda environment session {session_id} not found"
            )
        return self._sessions[session_id]

    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a conda environment session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(
                f"Conda environment session {session_id} not found"
            )

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {} if type is None else []

        logs = session_data.get("logs", {})

        if type:
            target_logs = logs.get(type, [])
            end_idx = (
                len(target_logs)
                if limit is None
                else min(offset + limit, len(target_logs))
            )
            return target_logs[offset:end_idx]
        else:
            result = {}
            for log_type_key, log_entries in logs.items():
                end_idx = (
                    len(log_entries)
                    if limit is None
                    else min(offset + limit, len(log_entries))
                )
                result[log_type_key] = log_entries[offset:end_idx]
            return result

    async def take_screenshot(
        self,
        session_id: str,
        format: str = "png",
        context: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Take a screenshot - not supported for conda environment sessions."""
        raise NotImplementedError(
            "Screenshots not supported for conda environment sessions"
        )

    async def prepare_workspace(
        self, workspace: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Prepare workspace for conda environment operations."""
        logger.info(f"Preparing workspace {workspace} for conda environment worker")
        pass

    async def close_workspace(
        self, workspace: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Close all conda environment sessions for a workspace."""
        logger.info(f"Closing workspace {workspace} for conda environment worker")

        # Stop all sessions for this workspace
        sessions_to_stop = [
            session_id
            for session_id, session_info in self._sessions.items()
            if session_info.workspace == workspace
        ]

        for session_id in sessions_to_stop:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(
                    f"Failed to stop conda environment session {session_id}: {e}"
                )

    async def shutdown(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Shutdown the conda environment worker."""
        logger.info("Shutting down conda environment worker...")

        # Cancel all background script tasks first
        for session_id, session_data in self._session_data.items():
            script_task = session_data.get("script_task")
            if script_task and not script_task.done():
                logger.info(
                    f"Cancelling background script task for session {session_id}"
                )
                script_task.cancel()

        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(
                    f"Failed to stop conda environment session {session_id}: {e}"
                )

        logger.info("Conda environment worker shutdown complete")

    def get_worker_service(self) -> Dict[str, Any]:
        """Get the service configuration for registration with conda-specific methods."""
        service_config = super().get_worker_service()
        # Add conda environment specific methods
        service_config["execute_code"] = self.execute_code
        service_config["take_screenshot"] = self.take_screenshot
        return service_config


async def hypha_startup(server):
    """Hypha startup function to initialize conda environment worker."""
    worker = CondaWorker()
    await worker.register_worker_service(server)
    logger.info("Conda environment worker initialized and registered")


def main():
    """Main function for command line execution."""
    import argparse
    import sys

    def get_env_var(name: str, default: str = None) -> str:
        """Get environment variable with HYPHA_ prefix."""
        return os.environ.get(f"HYPHA_{name.upper()}", default)

    parser = argparse.ArgumentParser(
        description="Hypha Conda Environment Worker - Execute Python code in isolated conda environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (with HYPHA_ prefix):
  HYPHA_SERVER_URL     Hypha server URL (e.g., https://hypha.aicell.io)
  HYPHA_WORKSPACE      Workspace name (e.g., my-workspace)
  HYPHA_TOKEN          Authentication token
  HYPHA_SERVICE_ID     Service ID for the worker (optional)
  HYPHA_VISIBILITY     Service visibility: public or protected (default: protected)
  HYPHA_CACHE_DIR      Directory for caching conda environments (optional)

Examples:
  # Using command line arguments
  python -m hypha.workers.conda --server-url https://hypha.aicell.io --workspace my-workspace --token TOKEN

  # Using environment variables
  export HYPHA_SERVER_URL=https://hypha.aicell.io
  export HYPHA_WORKSPACE=my-workspace
  export HYPHA_TOKEN=your-token-here
  python -m hypha.workers.conda

  # Mixed usage (command line overrides environment variables)
  export HYPHA_SERVER_URL=https://hypha.aicell.io
  python -m hypha.workers.conda --workspace my-workspace --token TOKEN
        """,
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default=get_env_var("SERVER_URL"),
        help="Hypha server URL (default: from HYPHA_SERVER_URL env var)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=get_env_var("WORKSPACE"),
        help="Workspace name (default: from HYPHA_WORKSPACE env var)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=get_env_var("TOKEN"),
        help="Authentication token (default: from HYPHA_TOKEN env var)",
    )
    parser.add_argument(
        "--service-id",
        type=str,
        default=get_env_var("SERVICE_ID"),
        help="Service ID for the worker (default: from HYPHA_SERVICE_ID env var or auto-generated)",
    )
    parser.add_argument(
        "--visibility",
        type=str,
        choices=["public", "protected"],
        default=get_env_var("VISIBILITY", "protected"),
        help="Service visibility (default: protected, from HYPHA_VISIBILITY env var)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=get_env_var("CACHE_DIR"),
        help="Directory for caching conda environments (default: from HYPHA_CACHE_DIR env var or ~/.hypha_conda_cache)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.server_url:
        print(
            "Error: --server-url is required (or set HYPHA_SERVER_URL environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.workspace:
        print(
            "Error: --workspace is required (or set HYPHA_WORKSPACE environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.token:
        print(
            "Error: --token is required (or set HYPHA_TOKEN environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set up logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.setLevel(logging.INFO)

    # Detect package manager early for better error reporting
    try:
        package_manager = get_available_package_manager()
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        print(f"   Please install conda or mamba to use this worker.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Hypha Conda Environment Worker...")
    print(f"  Package Manager: {package_manager}")
    print(f"  Server URL: {args.server_url}")
    print(f"  Workspace: {args.workspace}")
    print(f"  Service ID: {args.service_id or 'auto-generated'}")
    print(f"  Visibility: {args.visibility}")
    print(f"  Cache Dir: {args.cache_dir or DEFAULT_CACHE_DIR}")

    async def run_worker():
        """Run the conda environment worker."""
        try:
            from hypha_rpc import connect_to_server

            # Override cache directory if specified
            if args.cache_dir:
                global DEFAULT_CACHE_DIR
                DEFAULT_CACHE_DIR = args.cache_dir

            # Connect to server
            server = await connect_to_server(
                server_url=args.server_url, workspace=args.workspace, token=args.token
            )

            # Create and register worker
            worker = CondaWorker()
            if args.cache_dir:
                worker._env_cache = EnvironmentCache(cache_dir=args.cache_dir)

            # Get service config and set custom properties
            service_config = await worker.get_worker_service()
            if args.service_id:
                service_config["id"] = args.service_id
            service_config["visibility"] = args.visibility

            # Register the service
            await server.rpc.register_service(service_config)

            print(f"✅ Conda Environment Worker registered successfully!")
            print(f"   Service ID: {service_config['id']}")
            print(f"   Supported types: {worker.supported_types}")
            print(f"   Visibility: {args.visibility}")
            print(f"")
            print(f"Worker is ready to process conda environment requests...")
            print(f"Press Ctrl+C to stop the worker.")

            # Keep the worker running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n🛑 Shutting down Conda Environment Worker...")
                await worker.shutdown()
                print(f"✅ Worker shutdown complete.")

        except Exception as e:
            print(f"❌ Failed to start Conda Environment Worker: {e}", file=sys.stderr)
            sys.exit(1)

    # Run the worker
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
