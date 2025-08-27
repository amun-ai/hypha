"""Conda Environment Worker for executing Python code in isolated conda environments."""

import asyncio
import hashlib
import re
import httpx
import json
import logging
import inspect
import os
import shutil
import shortuuid
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
    safe_call_callback,
)
from hypha.workers.conda_executor import CondaEnvExecutor
from hypha.workers.conda_kernel import CondaKernel

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
READY_MARKER = ".hypha_env_ready"  # Marker file indicating env was fully initialized


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
            if os.name == "nt":
                python_path = env_path / "python.exe"
            else:
                python_path = env_path / "bin" / "python"
            marker_path = env_path / READY_MARKER
            if env_path.exists() and python_path.exists() and marker_path.exists():
                # Update last access time for LRU
                cache_entry["last_accessed"] = time.time()
                self._save_index()
                return env_path
            else:
                # Remove invalid entry
                try:
                    if env_path.exists():
                        shutil.rmtree(env_path, ignore_errors=True)
                finally:
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

        # Ensure the ready marker exists to indicate a fully initialized environment
        # Only create the marker if the environment path exists
        if env_path.exists():
            marker_path = env_path / READY_MARKER
            marker_path.touch(exist_ok=True)
        self.index[env_hash] = {
            "path": str(env_path),
            "dependencies": dependencies,
            "channels": channels,
            "created_at": time.time(),
            "last_accessed": time.time(),
        }
        self._save_index()

    def invalidate_env(self, dependencies: List[str], channels: List[str]):
        """Invalidate a cached environment by spec and remove it from disk."""
        env_hash = self._compute_env_hash(dependencies, channels)
        self._remove_cache_entry(env_hash)

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

    def __init__(self, server_url: str = None, use_local_url: Union[bool, str] = False, working_dir: str = None, cache_dir: str = None):
        """Initialize the conda environment worker.
        
        Args:
            server_url: The Hypha server URL
            use_local_url: Whether to use local URLs for server communication
            working_dir: Base directory for session working directories (defaults to /tmp/hypha_sessions)
        """
        super().__init__()
        self.instance_id = f"conda-jupyter-kernel-{shortuuid.uuid()}"
        self.controller_id = str(CondaWorker.instance_counter)
        CondaWorker.instance_counter += 1
        # convert true/false string to bool, and keep the string if it's not a bool
        if isinstance(use_local_url, str):
            if use_local_url.lower() == "true":
                self._use_local_url = True
            elif use_local_url.lower() == "false":
                self._use_local_url = False
            else:
                self._use_local_url = use_local_url
        else:
            self._use_local_url = use_local_url
        self._server_url = server_url
        
        # Set up working directory base path
        if working_dir:
            self._working_dir_base = Path(working_dir)
        else:
            # Default to /tmp with random subfolder for this worker instance
            self._working_dir_base = Path(tempfile.gettempdir()) / f"hypha_sessions_{shortuuid.uuid()}"
        
        # Ensure base working directory exists
        self._working_dir_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using working directory base: {self._working_dir_base}")

        # Detect available package manager (mamba preferred over conda)
        try:
            self.package_manager = get_available_package_manager()
        except RuntimeError as e:
            logger.error(f"Package manager detection failed: {e}")
            raise

        # Session management
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}
        self._session_working_dirs: Dict[str, Path] = {}  # Track working directories per session

        # Environment cache
        # If cache_dir is None, use a default location under the working directory base
        if cache_dir is None:
            cache_dir = str(self._working_dir_base / ".hypha_conda_cache")
        self._env_cache = EnvironmentCache(cache_dir=cache_dir)

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["conda-jupyter-kernel"]

    @property
    def name(self) -> str:
        """Return the worker name."""
        return f"Conda Worker ({self.package_manager})"

    @property
    def description(self) -> str:
        """Return the worker description."""
        return f"A worker for executing Python code in isolated conda environments with package management using {self.package_manager}"

    @property
    def require_context(self) -> bool:
        """Return whether the worker requires a context."""
        return True

    @property
    def use_local_url(self) -> Union[bool, str]:
        """Return whether the worker should use local URLs."""
        return self._use_local_url

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

        # Always ensure Jupyter kernel dependencies are available  
        jupyter_deps = ["ipykernel", "jupyter_client", "pyzmq"]
        for jupyter_dep in jupyter_deps:
            dep_found = False
            for dep in dependencies:
                if isinstance(dep, str) and dep == jupyter_dep:
                    dep_found = True
                    break
            
            if not dep_found:
                dependencies.append(jupyter_dep)
                logger.info(f"Automatically added {jupyter_dep} to dependencies")

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

    async def _prepare_staged_files(
        self,
        files_to_stage: List[str],
        working_dir: Path,
        config: WorkerConfig,
        progress_callback=None,
    ) -> None:
        """Prepare and download files from artifact manager to the working directory.
        
        Args:
            files_to_stage: List of file paths from artifact manager. Can include:
                - Simple file paths: "data.csv"
                - Folder paths (ending with /): "models/"
                - Renamed files: "source.txt:target.txt"
                - Renamed folders: "source/:target/"
            working_dir: Directory where files should be placed
            config: Worker configuration with server/auth details
            progress_callback: Optional callback for progress updates
            
        Raises:
            WorkerError: If any file or directory cannot be downloaded
        """
        if not files_to_stage:
            return
        
        # Validate files_to_stage format
        if not isinstance(files_to_stage, list):
            raise WorkerError(f"files_to_stage must be a list, got {type(files_to_stage).__name__}")
        
        for item in files_to_stage:
            if not isinstance(item, str):
                raise WorkerError(f"Each item in files_to_stage must be a string, got {type(item).__name__}: {item}")
            if not item or item.strip() == "":
                raise WorkerError("Empty or whitespace-only path in files_to_stage")
            
        await safe_call_callback(progress_callback,
            {"type": "info", "message": f"Preparing {len(files_to_stage)} files/folders from artifact manager..."}
        )
        
        async def download_directory_recursive(client, source_dir, target_dir, processed_dirs=None):
            """Recursively download all files from a directory."""
            if processed_dirs is None:
                processed_dirs = set()
            
            # Avoid infinite recursion
            if source_dir in processed_dirs:
                return
            processed_dirs.add(source_dir)
            
            # Get directory listing - use the files endpoint with trailing slash
            dir_url = f"{self._server_url}/{config.workspace}/artifacts/{config.app_id}/files/{source_dir}/?use_proxy=true"
            
            try:
                response = await client.get(
                    dir_url,
                    headers={"Authorization": f"Bearer {config.token}"}
                )
                response.raise_for_status()
                items = response.json()
                
                # Process each item in the directory
                for item in items:
                    item_name = item.get("name", "")
                    item_type = item.get("type", "file")
                    
                    if item_type == "directory":
                        # Recursively download subdirectory
                        sub_source = f"{source_dir}/{item_name}".strip("/")
                        sub_target = target_dir / item_name
                        sub_target.mkdir(parents=True, exist_ok=True)
                        await download_directory_recursive(client, sub_source, sub_target, processed_dirs)
                    else:
                        # Download file
                        source_file = f"{source_dir}/{item_name}".strip("/")
                        target_file = target_dir / item_name
                        
                        file_url = f"{self._server_url}/{config.workspace}/artifacts/{config.app_id}/files/{source_file}?use_proxy=true"
                        file_response = await client.get(
                            file_url,
                            headers={"Authorization": f"Bearer {config.token}"}
                        )
                        file_response.raise_for_status()
                        
                        # Ensure parent directory exists
                        try:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            # Write file to working directory
                            target_file.write_bytes(file_response.content)
                            logger.debug(f"Downloaded {source_file} to {target_file}")
                        except OSError as write_error:
                            error_msg = f"Failed to write file {target_file}: {write_error}"
                            logger.error(error_msg)
                            raise WorkerError(error_msg) from write_error
                
                return len(items)
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    error_msg = f"Directory not found in artifact manager: {source_dir}"
                    logger.error(error_msg)
                    raise WorkerError(error_msg) from e
                else:
                    raise
        
        async with httpx.AsyncClient(verify=not config.disable_ssl, timeout=30.0) as client:
            for item in files_to_stage:
                # Parse source and target from the item
                if ":" in item:
                    source, target = item.split(":", 1)
                else:
                    source = target = item
                
                # Check if it's a folder (ends with /)
                is_folder = source.endswith("/")
                
                if is_folder:
                    # Handle folder download recursively
                    source = source.rstrip("/")
                    target = target.rstrip("/")
                    
                    await safe_call_callback(progress_callback,
                        {"type": "info", "message": f"Downloading folder recursively: {source} -> {target}"}
                    )
                    
                    # Create target folder
                    target_folder = working_dir / target
                    try:
                        target_folder.mkdir(parents=True, exist_ok=True)
                    except OSError as mkdir_error:
                        error_msg = f"Failed to create directory {target_folder}: {mkdir_error}"
                        logger.error(error_msg)
                        await safe_call_callback(progress_callback,
                            {"type": "error", "message": error_msg}
                        )
                        raise WorkerError(error_msg) from mkdir_error
                    
                    # Recursively download all files
                    file_count = await download_directory_recursive(client, source, target_folder)
                    
                    await safe_call_callback(progress_callback,
                        {"type": "success", "message": f"Downloaded folder {source} ({file_count} items)"}
                    )
                else:
                    # Handle single file download
                    await safe_call_callback(progress_callback,
                        {"type": "info", "message": f"Downloading file: {source} -> {target}"}
                    )
                    
                    file_url = f"{self._server_url}/{config.workspace}/artifacts/{config.app_id}/files/{source}?use_proxy=true"
                    
                    try:
                        response = await client.get(
                            file_url,
                            headers={"Authorization": f"Bearer {config.token}"}
                        )
                        response.raise_for_status()
                        
                        # Create target file path
                        target_file = working_dir / target
                        
                        try:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            # Write file to working directory
                            target_file.write_bytes(response.content)
                            
                            logger.info(f"Downloaded {source} to {target_file}")
                            await safe_call_callback(progress_callback,
                                {"type": "success", "message": f"Downloaded {source}"}
                            )
                        except OSError as write_error:
                            error_msg = f"Failed to write file {target_file}: {write_error}"
                            logger.error(error_msg)
                            await safe_call_callback(progress_callback,
                                {"type": "error", "message": error_msg}
                            )
                            raise WorkerError(error_msg) from write_error
                        
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            error_msg = f"File not found in artifact manager: {source}"
                            logger.error(error_msg)
                            await safe_call_callback(progress_callback,
                                {"type": "error", "message": error_msg}
                            )
                            raise WorkerError(error_msg) from e
                        else:
                            raise

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
        logger.info(f"Starting conda environment session {session_id} for {config.id}")
        async def progress_callback(message: dict):
            """Invoke optional progress callback if provided, supporting sync or async callables."""
            callback = getattr(config, "progress_callback", None)
            await safe_call_callback(callback, message)

        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")

        # Report initial progress
        await progress_callback(
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
            await progress_callback({"type": "info", "message": "Fetching application script..."})
            script_url = f"{self._server_url}/{config.workspace}/artifacts/{config.app_id}/files/{config.manifest['entry_point']}?use_proxy=true"
            async with httpx.AsyncClient(verify=not config.disable_ssl) as client:
                response = await client.get(
                    script_url, headers={"Authorization": f"Bearer {config.token}"}
                )
                response.raise_for_status()
                script = response.text
                
                # TEMPORARY PATCH: Remove any remaining <script> or <file> tags from script content
                script = re.sub(r'<script[^>]*>(.*?)</script>', r'\1', script, flags=re.DOTALL | re.IGNORECASE)
                script = re.sub(r'<file[^>]*>(.*?)</file>', r'\1', script, flags=re.DOTALL | re.IGNORECASE)
                script = script.strip()

            await progress_callback(
                {
                    "type": "success",
                    "message": "Application script loaded successfully",
                }
            )

            # Phase 2: Start the conda environment session (this is the long part)
            await progress_callback(
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

            await progress_callback(
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
            # remove cached environment
            

            await progress_callback(
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
        await progress_callback(
            {"type": "info", "message": "Checking for cached conda environment..."}
        )

        cached_env_path = self._env_cache.get_cached_env(dependencies, channels)

        is_new_env = False
        if cached_env_path:
            await progress_callback(
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
            await progress_callback(
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
            is_new_env = True

            # This is the long-running part - environment creation
            await progress_callback(
                {
                    "type": "info",
                    "message": f"Installing packages using {self.package_manager}... (this may take several minutes)",
                }
            )

            # Run environment creation directly (now async)
            try:
                setup_time = await executor._extract_env(progress_callback)
            except Exception as e:
                await progress_callback(
                    {
                        "type": "error",
                        "message": f"Environment creation failed: {str(e)}",
                    }
                )
                # Cleanup incomplete environment
                try:
                    if executor and executor.env_path and executor.env_path.exists():
                        shutil.rmtree(executor.env_path, ignore_errors=True)
                finally:
                    pass
                raise

            await progress_callback(
                {
                    "type": "success",
                    "message": f"Environment created successfully in {setup_time:.1f}s",
                }
            )

            # Do not cache yet; cache only after full readiness

        # Phase 3: Start Jupyter kernel
        await progress_callback(
            {
                "type": "info",
                "message": "Starting Jupyter kernel...",
            }
        )

        # Create session-specific working directory
        session_working_dir = self._working_dir_base / config.id
        session_working_dir.mkdir(parents=True, exist_ok=True)
        self._session_working_dirs[config.id] = session_working_dir
        logger.info(f"Created session working directory: {session_working_dir}")
        
        # Phase 3.5: Prepare staged files if specified
        files_to_stage = config.manifest.get("files_to_stage", [])
        if files_to_stage:
            try:
                await self._prepare_staged_files(
                    files_to_stage,
                    session_working_dir,
                    config,
                    progress_callback
                )
            except Exception as e:
                error_msg = f"Failed to prepare staged files: {str(e)}"
                logger.error(error_msg)
                await progress_callback(
                    {"type": "error", "message": error_msg}
                )
                # Cleanup incomplete environment if this was a newly created env
                if is_new_env:
                    try:
                        marker_path = executor.env_path / READY_MARKER
                        if not marker_path.exists() and executor.env_path.exists():
                            shutil.rmtree(executor.env_path, ignore_errors=True)
                    except Exception:
                        pass
                raise WorkerError(error_msg) from e

        # Create and start the Jupyter kernel with working directory
        kernel = CondaKernel(executor.env_path, working_dir=str(session_working_dir))
        try:
            # Honor provided startup timeout when available
            await kernel.start(timeout=float(config.timeout) if config.timeout else 30.0)
            await progress_callback(
                {
                    "type": "success",
                    "message": "Jupyter kernel started successfully",
                }
            )
        except Exception as e:
            await progress_callback(
                {
                    "type": "error",
                    "message": f"Failed to start Jupyter kernel: {str(e)}",
                }
            )
            # Cleanup incomplete environment if this was a newly created env
            if is_new_env:
                try:
                    marker_path = executor.env_path / READY_MARKER
                    if not marker_path.exists() and executor.env_path.exists():
                        shutil.rmtree(executor.env_path, ignore_errors=True)
                finally:
                    pass
            raise

        # Phase 4: Run initialization script in the kernel
        await progress_callback(
            {
                "type": "info",
                "message": "Running initialization script in kernel...",
            }
        )

        # Prepare the hypha config
        hypha_config = {
            "server_url": config.server_url,
            "workspace": config.workspace,
            "client_id": config.client_id,
            "token": config.token,
            "app_id": config.app_id,
            "disable_ssl": config.disable_ssl,
        }

        # Execute initialization script in the kernel
        init_code = f"""
import os
import sys

# Set up Hypha configuration
hypha_config = {repr(hypha_config)}
os.environ['HYPHA_SERVER_URL'] = hypha_config['server_url']
os.environ['HYPHA_WORKSPACE'] = hypha_config['workspace']
os.environ['HYPHA_CLIENT_ID'] = hypha_config['client_id']
os.environ['HYPHA_TOKEN'] = hypha_config['token']
os.environ['HYPHA_APP_ID'] = hypha_config['app_id']

# Execute the user's script
{script}
"""
        # Use provided startup timeout for initialization execution when available
        result = await kernel.execute(
            init_code, timeout=float(config.timeout) if config.timeout else 60.0,
        )
        
        # Process kernel outputs into logs
        for output in result.get("outputs", []):
            if output["type"] == "stream":
                if output["name"] == "stdout":
                    logs["stdout"].append(output["text"])
                elif output["name"] == "stderr":
                    logs["stderr"].append(output["text"])
            elif output["type"] == "error":
                logs["error"].append("\n".join(output.get("traceback", [])))
            elif output["type"] == "execute_result":
                # Convert execute results to string representation
                data = output.get("data", {})
                if "text/plain" in data:
                    logs["info"].append(f"Result: {data['text/plain']}")
        
        # Handle kernel error if present
        if result.get("error"):
            error_info = result["error"]
            logs["error"].append(f"Error: {error_info.get('ename', 'Unknown')}: {error_info.get('evalue', '')}")
            if error_info.get("traceback"):
                logs["error"].extend(error_info["traceback"])
        
        if result["success"]:
            await progress_callback(
                {
                    "type": "success",
                    "message": "Conda environment session with Jupyter kernel ready",
                }
            )
            # Mark environment as ready and cache only now for newly created envs
            try:
                if is_new_env:
                    # Create readiness marker
                    marker_path = executor.env_path / READY_MARKER
                    try:
                        marker_path.write_text("ready")
                    except Exception:
                        # If we cannot write the marker, consider env not ready
                        raise
                    # Add to cache index now that env is fully ready
                    self._env_cache.add_cached_env(dependencies, channels, executor.env_path)
                    logger.info("Cached new conda environment: %s", executor.env_path)
            except Exception:
                # If marking/cache fails, clean up to avoid half-baked cache
                try:
                    if executor.env_path.exists():
                        shutil.rmtree(executor.env_path, ignore_errors=True)
                finally:
                    pass
                raise
        else:
            await progress_callback(
                {
                    "type": "error",
                    "message": "Initialization script failed",
                }
            )
            # Cleanup incomplete environment if this was a newly created env
            if is_new_env:
                try:
                    marker_path = executor.env_path / READY_MARKER
                    if not marker_path.exists() and executor.env_path.exists():
                        shutil.rmtree(executor.env_path, ignore_errors=True)
                finally:
                    pass
            raise Exception("Initialization script failed: " + result.get("error", {}).get("evalue", "Unknown error"))


        return {
            "executor": executor,
            "kernel": kernel,
            "script": script,
            "dependencies": dependencies,
            "channels": channels,
            "logs": logs,
            "hypha_config": hypha_config,
            "env_ready": (executor.env_path / READY_MARKER).exists(),
            "is_new_env": is_new_env,
        }

    async def execute(
        self,
        session_id: str,
        script: str,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Any] = None,
        output_callback: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a script in the running Jupyter kernel session.
        
        This implements the new execute API method for interacting with running sessions.
        
        Args:
            session_id: The session to execute in
            script: The Python code to execute
            config: Optional execution configuration
            progress_callback: Optional callback for execution progress
            context: Optional context information
            
        Returns:
            Dictionary containing execution results with Jupyter-like output format
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(
                f"Conda environment session {session_id} not found"
            )

        session_data = self._session_data.get(session_id)
        if not session_data or not session_data.get("kernel"):
            raise WorkerError(f"No kernel available for session {session_id}")

        kernel = session_data["kernel"]
        
        # Check if kernel is still alive
        if not await kernel.is_alive():
            raise WorkerError(f"Kernel for session {session_id} is not alive")

        await safe_call_callback(progress_callback,
            {"type": "info", "message": "Executing code in Jupyter kernel..."}
        )

        try:
            # Configure execution options from config
            timeout = config.get("timeout", 30.0) if config else 30.0    
            # Execute the code in the kernel
            result = await kernel.execute(
                script,
                timeout=timeout,
                output_callback=output_callback
            )

            # Update session logs
            logs = session_data.get("logs", {})
            
            # Process outputs for logging
            for output in result.get("outputs", []):
                if output["type"] == "stream":
                    if output["name"] == "stdout":
                        logs.setdefault("stdout", []).append(output["text"])
                    elif output["name"] == "stderr":
                        logs.setdefault("stderr", []).append(output["text"])
                elif output["type"] == "error":
                    logs.setdefault("error", []).append("\n".join(output.get("traceback", [])))

            # Handle kernel error if present
            if result.get("error"):
                error_info = result["error"]
                logs.setdefault("error", []).append(f"Error: {error_info.get('ename', 'Unknown')}: {error_info.get('evalue', '')}")
                if error_info.get("traceback"):
                    logs.setdefault("error", []).extend(error_info["traceback"])

            # Add execution info
            status_text = "success" if result["success"] else "error"
            logs.setdefault("info", []).append(
                f"Code executed at {datetime.now().isoformat()} - Status: {status_text}"
            )

            await safe_call_callback(progress_callback,
                {"type": "success" if result["success"] else "error", 
                 "message": "Code executed successfully" if result["success"] else "Code execution failed"}
            )

            # Convert kernel format to expected API format for backward compatibility
            api_result = {
                "status": "ok" if result["success"] else "error",
                "outputs": result["outputs"]
            }
            
            if result.get("error"):
                api_result["error"] = result["error"]
                
            return api_result

        except asyncio.TimeoutError:
            error_msg = f"Code execution timed out after {timeout} seconds"
            await safe_call_callback(progress_callback, {"type": "error", "message": error_msg})
            
            logs = session_data.get("logs", {})
            logs.setdefault("error", []).append(error_msg)
            
            return {
                "status": "error",
                "outputs": [],
                "error": {
                    "ename": "TimeoutError",
                    "evalue": error_msg,
                    "traceback": [error_msg]
                }
            }
        except Exception as e:
            error_msg = f"Failed to execute code: {str(e)}"
            logger.error(f"Failed to execute code in session {session_id}: {e}")
            
            await safe_call_callback(progress_callback, {"type": "error", "message": error_msg})
            
            logs = session_data.get("logs", {})
            logs.setdefault("error", []).append(error_msg)
            
            return {
                "status": "error",
                "outputs": [],
                "error": {
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [error_msg]
                }
            }

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
                # Shutdown the Jupyter kernel
                kernel = session_data.get("kernel")
                if kernel:
                    logger.info(f"Shutting down Jupyter kernel for session {session_id}")
                    try:
                        await kernel.stop()
                    except Exception as e:
                        logger.warning(
                            f"Error shutting down kernel for session {session_id}: {e}"
                        )

                # Cleanup incomplete environment (not fully ready)
                executor = session_data.get("executor")
                env_path = executor.env_path if executor else None
                if env_path:
                    marker_path = env_path / READY_MARKER
                    # If env is not marked ready, remove it
                    if not marker_path.exists():
                        try:
                            logger.info(
                                f"Removing incomplete conda environment for session {session_id}: {env_path}"
                            )
                            shutil.rmtree(env_path, ignore_errors=True)
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove incomplete env at {env_path}: {e}"
                            )
                        # Invalidate any existing cache entry that might reference this env
                        try:
                            self._env_cache.invalidate_env(
                                session_data.get("dependencies", []),
                                session_data.get("channels", []),
                            )
                        except Exception as e:
                            logger.debug(f"Failed to invalidate cache: {e}")

            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped conda environment session {session_id}")

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop conda environment session {session_id}: {e}")
            raise
        finally:
            # Cleanup working directory
            session_working_dir = self._session_working_dirs.get(session_id)
            if session_working_dir and session_working_dir.exists():
                try:
                    shutil.rmtree(session_working_dir)
                    logger.info(f"Removed session working directory: {session_working_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove working directory {session_working_dir}: {e}")
            
            # Cleanup session data
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)
            self._session_working_dirs.pop(session_id, None)

    

    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get logs for a conda environment session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(
                f"Conda environment session {session_id} not found"
            )

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {"items": [], "total": 0, "offset": offset, "limit": limit}

        logs = session_data.get("logs", {})
        
        # Convert logs to items format
        all_items = []
        for log_type, log_entries in logs.items():
            for entry in log_entries:
                all_items.append({"type": log_type, "content": entry})
        
        # Filter by type if specified
        if type:
            filtered_items = [item for item in all_items if item["type"] == type]
        else:
            filtered_items = all_items
        
        total = len(filtered_items)
        
        # Apply pagination
        if limit is None:
            paginated_items = filtered_items[offset:]
        else:
            paginated_items = filtered_items[offset:offset + limit]
        
        return {
            "items": paginated_items,
            "total": total,
            "offset": offset,
            "limit": limit
        }

    async def shutdown(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Shutdown the conda environment worker."""
        logger.info("Shutting down conda environment worker...")

        # Stop all sessions (which will shutdown kernels)
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
        return service_config


async def hypha_startup(server):
    """Hypha startup function to initialize conda environment worker."""
    # Built-in worker should use local URLs and a specific working directory
    working_dir = os.environ.get("CONDA_WORKING_DIR")
    authorized_workspaces = [w.strip() for w in os.environ.get("CONDA_AUTHORIZED_WORKSPACES", "").strip().split(",") if w.strip()]
    worker = CondaWorker(server_url=server.config.local_base_url, use_local_url=True, working_dir=working_dir, cache_dir=os.environ.get("CONDA_CACHE_DIR", DEFAULT_CACHE_DIR))
    service = worker.get_worker_service()
    if authorized_workspaces:
        service["config"]["authorized_workspaces"] = authorized_workspaces
    await server.register_service(service)
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
  CONDA_CACHE_DIR      Directory for caching conda environments (optional)
  CONDA_WORKING_DIR    Base directory for session working directories (optional)
  CONDA_AUTHORIZED_WORKSPACES  Comma-separated list of authorized workspaces (optional)
  
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
        "--client-id",
        type=str,
        default=get_env_var("CLIENT_ID"),
        help="Client ID for the worker (default: from HYPHA_CLIENT_ID env var or auto-generated)",
    )
    parser.add_argument(
        "--visibility",
        type=str,
        choices=["public", "protected"],
        default=get_env_var("VISIBILITY", "protected"),
        help="Service visibility (default: protected, from HYPHA_VISIBILITY env var)",
    )
    parser.add_argument(
        "--disable-ssl",
        action="store_true",
        help="Disable SSL verification (default: false)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=get_env_var("CONDA_CACHE_DIR"),
        help="Directory for caching conda environments (default: from CONDA_CACHE_DIR env var or ~/.hypha_conda_cache)",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=get_env_var("CONDA_WORKING_DIR"),
        help="Base directory for session working directories (default: from CONDA_WORKING_DIR env var or /tmp/hypha_sessions_<uuid>)",
    )
    parser.add_argument(
        "--authorized-workspaces",
        type=str,
        default=get_env_var("CONDA_AUTHORIZED_WORKSPACES"),
        help="Comma-separated list of authorized workspaces (default: from CONDA_AUTHORIZED_WORKSPACES env var)",
    )
    parser.add_argument(
        "--use-local-url",
        default="false",
        help="Use local URLs for server communication (default: false for CLI workers, true for built-in workers, or specify the url for proxy etc.)",
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
    print(f"  Client ID: {args.client_id}")
    print(f"  Service ID: {args.service_id}")
    print(f"  Visibility: {args.visibility}")
    print(f"  Use Local URL: {args.use_local_url}")
    print(f"  Cache Dir: {args.cache_dir or DEFAULT_CACHE_DIR}")
    print(f"  Working Dir: {args.working_dir or 'Auto-generated in /tmp'}")
    print(f"  Authorized Workspaces: {args.authorized_workspaces}")

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
                server_url=args.server_url, workspace=args.workspace, token=args.token, client_id=args.client_id, ssl=False if args.disable_ssl else None
            )

            # Create and register worker
            worker = CondaWorker(
                server_url=args.server_url, 
                use_local_url=args.use_local_url,
                working_dir=args.working_dir,
                cache_dir=args.cache_dir
            )

            # Get service config and set custom properties
            service_config = worker.get_worker_service()
            if args.service_id:
                service_config["id"] = args.service_id
            # Set visibility in the correct location (inside config)
            service_config["config"]["visibility"] = args.visibility
            if args.authorized_workspaces:
                authorized_workspaces = [w.strip() for w in args.authorized_workspaces.split(",") if w.strip()]
                service_config["config"]["authorized_workspaces"] = authorized_workspaces

            # Register the service
            print(f"🔄 Registering conda worker with config:")
            print(f"   Service ID: {service_config['id']}")
            print(f"   Type: {service_config['type']}")
            print(f"   Supported types: {service_config['supported_types']}")
            print(f"   Visibility: {service_config.get('config', {}).get('visibility', 'N/A')}")
            print(f"   Config: {service_config.get('config', {})}")
            print(f"   Workspace: {args.workspace}")
            
            registration_result = await server.register_service(service_config)
            print(f"   Registrated service id: {registration_result.id}")

            # Verify registration by listing services
            try:
                services = await server.list_services({"type": "server-app-worker"})
                print(f"   Found {len(services)} server-app-worker services in workspace")
                conda_workers = [s for s in services if s.get('id').endswith(service_config['id'])]
                if conda_workers:
                    print(f"   ✅ Worker found in service list")
                else:
                    print(f"   ⚠️  Worker NOT found in service list!")
                    print(f"   Available workers: {[s.get('id') for s in services]}")
            except Exception as e:
                print(f"   ⚠️  Failed to verify registration: {e}")

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


async def run_from_env():
    """Run the conda worker using only environment variables.
    
    This function is useful when running the worker in containerized environments
    like Kubernetes where all configuration comes from environment variables.
    """
    try:
        from hypha_rpc import connect_to_server

        # Get configuration from environment variables
        server_url = os.environ.get("HYPHA_SERVER_URL")
        workspace = os.environ.get("HYPHA_WORKSPACE") 
        token = os.environ.get("HYPHA_TOKEN")
        client_id = os.environ.get("HYPHA_CLIENT_ID")
        service_id = os.environ.get("HYPHA_SERVICE_ID")
        visibility = os.environ.get("HYPHA_VISIBILITY", "protected")
        disable_ssl = os.environ.get("HYPHA_DISABLE_SSL", "false").lower() in ("true", "1", "yes")
        cache_dir = os.environ.get("CONDA_CACHE_DIR")
        working_dir = os.environ.get("CONDA_WORKING_DIR")
        verbose = os.environ.get("CONDA_VERBOSE", "false").lower() in ("true", "1", "yes")

        # Validate required environment variables
        if not server_url:
            raise ValueError("HYPHA_SERVER_URL environment variable is required")
        if not workspace:
            raise ValueError("HYPHA_WORKSPACE environment variable is required")
        if not token:
            raise ValueError("HYPHA_TOKEN environment variable is required")

        # Set up logging
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logger.setLevel(logging.INFO)

        # Detect package manager early for better error reporting
        try:
            package_manager = get_available_package_manager()
        except RuntimeError as e:
            logger.error(f"Package manager detection failed: {e}")
            logger.error("Please install conda or mamba to use this worker.")
            raise

        logger.info(f"Starting Hypha Conda Environment Worker from environment variables...")
        logger.info(f"  Package Manager: {package_manager}")
        logger.info(f"  Server URL: {server_url}")
        logger.info(f"  Workspace: {workspace}")
        logger.info(f"  Client ID: {client_id}")
        logger.info(f"  Service ID: {service_id}")
        logger.info(f"  Visibility: {visibility}")
        # Override cache directory if specified
        global DEFAULT_CACHE_DIR
        if cache_dir:
            DEFAULT_CACHE_DIR = cache_dir

        logger.info(f"  Cache Dir: {cache_dir or DEFAULT_CACHE_DIR}")

        # Connect to server
        server = await connect_to_server(
            server_url=server_url,
            workspace=workspace,
            token=token,
            client_id=client_id,
            ssl=False if disable_ssl else None,
        )

        # Create and register worker
        worker = CondaWorker(server_url=server_url, working_dir=working_dir, cache_dir=cache_dir)

        # Get service config and set custom properties
        service_config = worker.get_worker_service()
        if service_id:
            service_config["id"] = service_id
        # Set visibility in the correct location (inside config)
        service_config["config"]["visibility"] = visibility

        # Register the service
        logger.info(f"Registering conda worker with config:")
        logger.info(f"   Service ID: {service_config['id']}")
        logger.info(f"   Type: {service_config['type']}")
        logger.info(f"   Supported types: {service_config['supported_types']}")
        logger.info(f"   Visibility: {service_config.get('config', {}).get('visibility', 'N/A')}")
        logger.info(f"   Workspace: {workspace}")

        registration_result = await server.register_service(service_config)
        logger.info(f"   Registered service id: {registration_result.id}")

        # Verify registration by listing services
        try:
            services = await server.list_services({"type": "server-app-worker"})
            logger.info(f"   Found {len(services)} server-app-worker services in workspace")
            conda_workers = [s for s in services if s.get('id').endswith(service_config['id'])]
            if conda_workers:
                logger.info(f"   ✅ Worker found in service list")
            else:
                logger.warning(f"   ⚠️  Worker NOT found in service list!")
                logger.warning(f"   Available workers: {[s.get('id') for s in services]}")
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to verify registration: {e}")

        logger.info(f"✅ Conda Environment Worker registered successfully!")
        logger.info(f"   Service ID: {service_config['id']}")
        logger.info(f"   Supported types: {worker.supported_types}")
        logger.info(f"   Visibility: {visibility}")
        logger.info(f"Worker is ready to process conda environment requests...")

        # Keep the worker running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"Shutting down Conda Environment Worker...")
            await worker.shutdown()
            logger.info(f"Worker shutdown complete.")

    except Exception as e:
        logger.error(f"Failed to start Conda Environment Worker: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if we should run from environment variables only
    if os.environ.get("HYPHA_RUN_FROM_ENV", "false").lower() in ("true", "1", "yes"):
        asyncio.run(run_from_env())
    else:
        main()
