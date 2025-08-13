"""
Main executor module for running code in conda environments.

This module provides the CondaEnvExecutor class which handles executing Python code
in isolated conda environments with efficient data passing.
"""

import os
import sys
import json
import tempfile
import subprocess
import time
import hashlib
import shutil
import textwrap
import logging
import asyncio
import functools
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, NamedTuple
from dataclasses import dataclass
import tarfile
import traceback

import yaml
import psutil
import conda_pack


from .env_spec import EnvSpec, read_env_spec
from .shared_memory import SharedMemoryChannel

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def get_package_manager() -> str:
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
            log.info(f"Using mamba package manager: {result.stdout.strip()}")
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
            log.info(f"Using conda package manager: {result.stdout.strip()}")
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


def get_package_manager_info_base(package_manager: str) -> str:
    """Get the base directory for the package manager.

    Args:
        package_manager: Either 'mamba' or 'conda'

    Returns:
        str: Command to get package manager base directory
    """
    if package_manager == "mamba":
        # Mamba doesn't support --base flag, use different approach
        # Use cross-platform Python command with proper quoting
        if os.name == "nt":  # Windows
            return 'python -c "import os; print(os.path.dirname(os.path.dirname(os.__file__)))"'
        else:  # Unix-like (Linux, macOS)
            return "python -c 'import os; print(os.path.dirname(os.path.dirname(os.__file__)))'"
    else:
        return f"{package_manager} info --base"


@dataclass
class TimingInfo:
    """Timing information for environment setup and code execution."""

    env_setup_time: float
    execution_time: float
    total_time: float


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    timing: Optional[TimingInfo] = None


def compute_file_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class EnvCache:
    """Manages cached conda environments."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.conda_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_cache_index()

    def _load_cache_index(self) -> None:
        """Load or create the cache index file."""
        self.index_path = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}

    def _save_cache_index(self) -> None:
        """Save the cache index file."""
        with open(self.index_path, "w") as f:
            json.dump(self.cache_index, f)

    def get_env_path(self, env_pack_path: str) -> Optional[str]:
        """Get the path to a cached environment if it exists and is valid."""
        try:
            file_hash = compute_file_hash(env_pack_path)
            if file_hash in self.cache_index:
                cache_info = self.cache_index[file_hash]
                cache_path = cache_info["path"]

                # Verify the cache exists and is valid
                if os.path.exists(cache_path) and os.path.isdir(cache_path):
                    # Check if the activation script exists
                    if os.path.exists(os.path.join(cache_path, "bin", "activate")):
                        return cache_path

            return None
        except Exception:
            return None

    def add_env(self, env_pack_path: str, env_path: str) -> None:
        """Add an environment to the cache."""
        file_hash = compute_file_hash(env_pack_path)
        self.cache_index[file_hash] = {
            "path": env_path,
            "pack_path": str(Path(env_pack_path).resolve()),
            "created_at": time.time(),
        }
        self._save_cache_index()

    def cleanup_old_envs(self, max_age_days: int = 30) -> None:
        """Remove environments older than max_age_days."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        to_remove = []
        for file_hash, cache_info in self.cache_index.items():
            if current_time - cache_info["created_at"] > max_age_seconds:
                env_path = cache_info["path"]
                if os.path.exists(env_path):
                    try:
                        shutil.rmtree(env_path)
                    except Exception:
                        pass
                to_remove.append(file_hash)

        for file_hash in to_remove:
            del self.cache_index[file_hash]

        if to_remove:
            self._save_cache_index()


class CondaEnvExecutor:
    """Execute Python code in a conda environment."""

    def __init__(
        self,
        env_source: Union[str, Path, Dict, EnvSpec],
        env_dir: Optional[Path] = None,
        **kwargs,
    ):
        """Initialize the executor.

        Args:
            env_source: Source for creating the conda environment. Can be:
                - Path to a YAML file with environment specification
                - Path to a conda-pack file (.tar.gz)
                - Dictionary with environment specification
                - EnvSpec object
            env_dir: Optional directory to store the environment. If not provided,
                    a temporary directory will be created.
            **kwargs: Additional arguments:
                - cache_dir: Directory to cache environments
                - env_name: Name for the environment
        """
        self.env_source = env_source
        # Detect available package manager (mamba preferred over conda)
        self.package_manager = get_package_manager()
        self._env_dir = Path(env_dir) if env_dir else None
        self._owns_env_dir = env_dir is None
        self._env_path = None
        self._is_extracted = False
        self.env_cache = {}
        self.env_name = kwargs.get("env_name")
        self._kwargs = kwargs

    @property
    def env_path(self) -> Path:
        """Get the path to the conda environment."""
        if self._env_path is None:
            if self._env_dir is None:
                self._env_dir = Path(tempfile.mkdtemp())
                self._owns_env_dir = True
            self._env_path = self._env_dir / "env"
        return self._env_path

    @env_path.setter
    def env_path(self, value: Optional[Path]):
        """Set the path to the conda environment."""
        self._env_path = Path(value) if value else None

    @classmethod
    def create_temp_env(
        cls,
        dependencies: List[Union[str, Dict]],
        channels: Optional[List[str]] = None,
        name: str = None,
        **kwargs,
    ) -> "CondaEnvExecutor":
        """Create an executor with a temporary environment.

        Args:
            packages: List of packages to install.
            channels: Optional list of channels to use.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            CondaEnvExecutor instance.
        """
        spec = {
            "name": name or "temp_env",
            "channels": channels or ["conda-forge"],
            "dependencies": dependencies,
        }
        return cls(spec, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path], **kwargs) -> "CondaEnvExecutor":
        """Create an executor from a YAML environment specification file.

        Args:
            yaml_file: Path to the YAML file.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            CondaEnvExecutor instance.
        """
        return cls(yaml_file, **kwargs)

    async def _create_conda(self, progress_callback=None) -> float:
        """Create a conda environment from the specified source.

        Args:
            progress_callback: Optional callback to report progress

        Returns:
            float: Time taken to create the environment in seconds.
        """

        start_time = time.time()

        # If source is a conda-pack file, extract it
        if str(self.env_source).endswith((".tar.gz", ".tgz")):
            if progress_callback:
                await progress_callback(
                    {"type": "info", "message": "Extracting conda-pack file..."}
                )
            try:
                with tarfile.open(self.env_source, "r:gz") as tar:
                    tar.extractall(path=self.env_path)
                if progress_callback:
                    await progress_callback(
                        {
                            "type": "success",
                            "message": "Conda-pack file extracted successfully",
                        }
                    )
                return time.time() - start_time
            except Exception as e:
                if progress_callback:
                    await progress_callback(
                        {
                            "type": "error",
                            "message": f"Failed to extract conda-pack file: {e}",
                        }
                    )
                raise RuntimeError(f"Failed to extract conda-pack file: {e}")

        # Create environment from specification
        if progress_callback:
            await progress_callback(
                {"type": "info", "message": "Preparing environment specification..."}
            )

        spec = read_env_spec(self.env_source)
        env_file = os.path.join(self.env_path.parent, "environment.yaml")
        os.makedirs(os.path.dirname(env_file), exist_ok=True)

        # Write environment specification to file
        with open(env_file, "w") as f:
            yaml.safe_dump(spec.to_dict(), f)

        if progress_callback:
            await progress_callback(
                {
                    "type": "info",
                    "message": f"Running {self.package_manager} env create (this may take several minutes)...",
                }
            )

        # Create the environment using detected package manager (mamba or conda)
        try:
            create_cmd, executable = self._build_create_command(env_file)

            # Run subprocess with real-time output streaming
            return_code, stdout, stderr = await self._run_subprocess_with_streaming(
                create_cmd, progress_callback, executable=executable
            )

            if return_code == 0:
                if progress_callback:
                    await progress_callback(
                        {
                            "type": "success",
                            "message": f"Environment created successfully using {self.package_manager}",
                        }
                    )

                log.info(
                    f"Successfully created environment using {self.package_manager}: {self.env_path}"
                )
                return time.time() - start_time
            else:
                error_msg = f"Failed to create environment using {self.package_manager}: {stderr}"
                if progress_callback:
                    await progress_callback({"type": "error", "message": error_msg})
                raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = (
                f"Failed to create environment using {self.package_manager}: {str(e)}"
            )
            if progress_callback:
                await progress_callback({"type": "error", "message": error_msg})
            raise RuntimeError(error_msg)

    def _build_create_command(self, env_file):
        """Build the conda/mamba create command for the current platform.

        Args:
            env_file: Path to the environment.yaml file

        Returns:
            tuple: (command_string, executable_path_or_none)
        """
        # Basic command that works on all platforms
        base_cmd = (
            f"{self.package_manager} env create -p {self.env_path} -f {env_file} -y"
        )

        if os.name == "nt":  # Windows
            if self.package_manager == "mamba":
                # Mamba works directly on Windows
                return base_cmd, None
            else:
                # Conda on Windows - try to use conda directly
                # If this fails, the conda installation might need fixing
                return base_cmd, None
        else:  # Unix-like (Linux, macOS)
            if self.package_manager == "mamba":
                # Mamba usually works directly without sourcing conda.sh
                return base_cmd, None
            else:
                # Conda on Unix-like systems often needs conda.sh sourced
                # Try to find a suitable shell and conda base
                shell_executable = self._find_shell_executable()
                if shell_executable:
                    try:
                        info_base_cmd = get_package_manager_info_base(
                            self.package_manager
                        )
                        # Use proper path separator for the platform
                        conda_sh_path = "etc/profile.d/conda.sh"
                        sourced_cmd = (
                            f"source $({info_base_cmd})/{conda_sh_path} && {base_cmd}"
                        )
                        return sourced_cmd, shell_executable
                    except Exception:
                        # Fall back to direct command if sourcing fails
                        return base_cmd, None
                else:
                    # No suitable shell found, try direct command
                    return base_cmd, None

    def _find_shell_executable(self):
        """Find a suitable shell executable on Unix-like systems.

        Returns:
            str or None: Path to shell executable or None if not found
        """
        # Common shell locations on Unix-like systems
        possible_shells = [
            "/bin/bash",  # Most common
            "/usr/bin/bash",  # Alternative location
            "/bin/sh",  # POSIX shell fallback
            "/usr/bin/sh",  # Alternative sh location
        ]

        for shell_path in possible_shells:
            if os.path.isfile(shell_path) and os.access(shell_path, os.X_OK):
                return shell_path

        # Try to find bash in PATH
        bash_path = shutil.which("bash")
        if bash_path:
            return bash_path

        # Try to find sh in PATH
        sh_path = shutil.which("sh")
        if sh_path:
            return sh_path

        return None

    async def _run_subprocess_with_streaming(
        self, command, progress_callback=None, executable=None
    ):
        """Run subprocess with real-time output streaming.

        Args:
            command: Command to run
            progress_callback: Optional callback to report real-time output
            executable: Executable to use (e.g., "/bin/bash")

        Returns:
            tuple: (return_code, stdout, stderr)
        """
        # Parse command for shell execution
        if isinstance(command, str):
            if os.name == "nt":  # Windows
                cmd = ["cmd", "/c", command]
            else:  # Unix-like
                shell_executable = executable or "/bin/bash"
                cmd = [shell_executable, "-c", command]
        else:
            cmd = command

        try:
            log.info(
                f"Starting subprocess: {' '.join(cmd) if isinstance(cmd, list) else cmd}"
            )

            # Start the subprocess using standard asyncio
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            log.info(f"Subprocess started with PID: {proc.pid}")

            # Monitor both stdout and stderr for progress updates
            async def monitor_stdout() -> List[str]:
                """Monitor stdout and send to progress callback."""
                stdout_lines = []
                try:
                    while True:
                        line = await proc.stdout.readline()
                        if not line:
                            break

                        line_str = line.decode("utf-8", errors="replace").rstrip()
                        if line_str:
                            stdout_lines.append(line_str)

                            # Send to progress callback if available
                            if progress_callback:
                                try:
                                    msg_type = self._classify_message_type(line_str)
                                    await progress_callback(
                                        {
                                            "type": msg_type,
                                            "message": f"{self.package_manager}: {line_str}",
                                        }
                                    )
                                except Exception as e:
                                    log.warning(f"Progress callback error: {e}")
                except Exception as e:
                    log.error(f"Error reading stdout: {e}")
                    stdout_lines.append(f"Error reading stdout: {e}")

                return stdout_lines

            async def monitor_stderr() -> List[str]:
                """Monitor stderr and send to progress callback."""
                stderr_lines = []
                try:
                    while True:
                        line = await proc.stderr.readline()
                        if not line:
                            break

                        line_str = line.decode("utf-8", errors="replace").rstrip()
                        if line_str:
                            stderr_lines.append(line_str)

                            # Send to progress callback if available
                            if progress_callback:
                                try:
                                    msg_type = self._classify_message_type(line_str)
                                    await progress_callback(
                                        {
                                            "type": msg_type,
                                            "message": f"{self.package_manager}: {line_str}",
                                        }
                                    )
                                except Exception as e:
                                    log.warning(f"Progress callback error: {e}")
                except Exception as e:
                    log.error(f"Error reading stderr: {e}")
                    stderr_lines.append(f"Error reading stderr: {e}")

                return stderr_lines

            try:
                # Wait for both stdout, stderr monitoring and process completion
                stdout_lines, stderr_lines = await asyncio.gather(
                    monitor_stdout(), monitor_stderr()
                )

                # Wait for process to complete
                return_code = await proc.wait()

                stdout_str = "\n".join(stdout_lines) if stdout_lines else ""
                stderr_str = "\n".join(stderr_lines) if stderr_lines else ""

                if return_code == 0:
                    log.info(
                        f"Subprocess completed successfully with return code {return_code}"
                    )
                else:
                    log.error(f"Subprocess failed with return code {return_code}")

                return return_code, stdout_str, stderr_str

            except asyncio.TimeoutError:
                log.error("Subprocess timed out")
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()

                if progress_callback:
                    try:
                        await progress_callback(
                            {
                                "type": "error",
                                "message": f"{self.package_manager}: Process timed out",
                            }
                        )
                    except Exception as e:
                        log.warning(f"Progress callback error: {e}")

                raise subprocess.TimeoutExpired(cmd, 600)

        except Exception as e:
            log.error(f"Failed to start subprocess: {e}")
            if progress_callback:
                try:
                    await progress_callback(
                        {"type": "error", "message": f"Failed to start subprocess: {e}"}
                    )
                except Exception as pe:
                    log.warning(f"Progress callback error: {pe}")
            raise

    def _classify_message_type(self, message):
        """Classify a message based on its content.

        Args:
            message (str): The message to classify

        Returns:
            str: Message type ('error', 'warning', 'success', 'info')
        """
        message_lower = message.lower()

        # Error indicators
        error_keywords = ["error", "failed", "exception", "fatal", "critical", "abort"]
        if any(keyword in message_lower for keyword in error_keywords):
            return "error"

        # Warning indicators
        warning_keywords = ["warning", "warn", "deprecated", "caution"]
        if any(keyword in message_lower for keyword in warning_keywords):
            return "warning"

        # Success indicators
        success_keywords = [
            "done",
            "complete",
            "success",
            "finished",
            "transaction finished",
            "linking",
        ]
        if any(keyword in message_lower for keyword in success_keywords):
            return "success"

        # Default to info
        return "info"

    async def _extract_env(self, progress_callback=None) -> float:
        """Extract or create the conda environment.

        Args:
            progress_callback: Optional callback to report progress

        Returns:
            float: Time taken to extract/create the environment in seconds.
        """
        if self._is_extracted:
            return 0.0

        # Check if environment already exists and is valid
        python_path = os.path.join(self.env_path, "bin", "python")
        if os.name == "nt":  # Windows
            python_path = os.path.join(self.env_path, "python.exe")

        if os.path.exists(python_path):
            if progress_callback:
                await progress_callback(
                    {"type": "info", "message": "Validating existing environment..."}
                )
            try:
                result = subprocess.run(
                    [python_path, "-c", "import sys; print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.stdout.strip():
                    if progress_callback:
                        await progress_callback(
                            {
                                "type": "success",
                                "message": "Existing environment is valid and ready",
                            }
                        )
                    self._is_extracted = True
                    return 0.0  # Environment is valid and ready
            except subprocess.CalledProcessError:
                if progress_callback:
                    await progress_callback(
                        {
                            "type": "warning",
                            "message": "Existing environment is invalid, recreating...",
                        }
                    )
                pass  # Environment exists but is invalid, recreate it

        # Remove any existing directory to ensure clean slate for mamba/conda
        if os.path.exists(self.env_path):
            import shutil

            shutil.rmtree(self.env_path, ignore_errors=True)

        # Create parent directory only (let mamba/conda create the env directory)
        os.makedirs(os.path.dirname(self.env_path), exist_ok=True)

        # Create or extract the environment
        setup_time = await self._create_conda(progress_callback)
        self._is_extracted = True
        return setup_time

    def execute(self, code: str, hypha_config: dict = None) -> ExecutionResult:
        """Execute Python code in the conda environment.

        Args:
            code: The Python code to execute.
            hypha_config: Optional dict with Hypha connection info (server_url, workspace, client_id, token).

        Returns:
            ExecutionResult object containing the execution result or error.
        """
        start_time = time.time()
        # Environment should already be extracted when session was created
        env_setup_time = 0.0
        if not self._is_extracted:
            # This should not happen in normal usage - environment should be extracted during session creation
            raise RuntimeError(
                "Environment not extracted. Call _extract_env() during session initialization."
            )

        # Create temporary script file with unique name to avoid conflicts
        execution_id = str(uuid.uuid4())[:8]
        script_dir = os.path.join(self.env_path, ".scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, f"execute_{execution_id}.py")

        try:
            # Write the script directly - no templates, no complex logic
            with open(script_path, "w") as f:
                f.write(code)

            # Set environment variables
            env = os.environ.copy()

            # Add Hypha connection environment variables if provided
            if hypha_config:
                if hypha_config.get("server_url"):
                    env["HYPHA_SERVER_URL"] = hypha_config["server_url"]
                if hypha_config.get("workspace"):
                    env["HYPHA_WORKSPACE"] = hypha_config["workspace"]
                if hypha_config.get("client_id"):
                    env["HYPHA_CLIENT_ID"] = hypha_config["client_id"]
                if hypha_config.get("token"):
                    env["HYPHA_TOKEN"] = hypha_config["token"]
                if hypha_config.get("app_id"):
                    env["HYPHA_APP_ID"] = hypha_config["app_id"]

            # Execute the script
            python_path = os.path.join(self.env_path, "bin", "python")
            if os.name == "nt":  # Windows
                python_path = os.path.join(self.env_path, "python.exe")

            start_exec_time = time.time()
            try:
                result = subprocess.run(
                    [python_path, script_path],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )

                execution_time = time.time() - start_exec_time
                total_time = time.time() - start_time

                return ExecutionResult(
                    success=True,
                    result=None,  # No result data, just execution success
                    error=None,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    timing=TimingInfo(
                        env_setup_time=env_setup_time,
                        execution_time=execution_time,
                        total_time=total_time,
                    ),
                )
            except subprocess.CalledProcessError as e:
                execution_time = time.time() - start_exec_time
                total_time = time.time() - start_time

                return ExecutionResult(
                    success=False,
                    error=e.stderr or "Script execution failed",
                    stdout=e.stdout,
                    stderr=e.stderr,
                    timing=TimingInfo(
                        env_setup_time=env_setup_time,
                        execution_time=execution_time,
                        total_time=total_time,
                    ),
                )

        finally:
            # Clean up temporary script file
            if os.path.exists(script_path):
                os.remove(script_path)

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._owns_env_dir and self._env_dir and self._env_dir.exists():
            shutil.rmtree(self._env_dir)
            self._env_dir = None
            self._env_path = None
            self._is_extracted = False

    def __enter__(self) -> "CondaEnvExecutor":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.cleanup()
