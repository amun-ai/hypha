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
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, NamedTuple
from dataclasses import dataclass
import tarfile
import traceback

import yaml
import psutil
import conda_pack
import numpy as np

from .env_spec import EnvSpec, read_env_spec
from .shared_memory import SharedMemoryChannel

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

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
    stdout: str = ''
    stderr: str = ''
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
        self.cache_dir = cache_dir or os.path.expanduser("~/.conda_env_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_cache_index()
    
    def _load_cache_index(self) -> None:
        """Load or create the cache index file."""
        self.index_path = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}
    
    def _save_cache_index(self) -> None:
        """Save the cache index file."""
        with open(self.index_path, 'w') as f:
            json.dump(self.cache_index, f)
    
    def get_env_path(self, env_pack_path: str) -> Optional[str]:
        """Get the path to a cached environment if it exists and is valid."""
        try:
            file_hash = compute_file_hash(env_pack_path)
            if file_hash in self.cache_index:
                cache_info = self.cache_index[file_hash]
                cache_path = cache_info['path']
                
                # Verify the cache exists and is valid
                if os.path.exists(cache_path) and os.path.isdir(cache_path):
                    # Check if the activation script exists
                    if os.path.exists(os.path.join(cache_path, 'bin', 'activate')):
                        return cache_path
            
            return None
        except Exception:
            return None
    
    def add_env(self, env_pack_path: str, env_path: str) -> None:
        """Add an environment to the cache."""
        file_hash = compute_file_hash(env_pack_path)
        self.cache_index[file_hash] = {
            'path': env_path,
            'pack_path': str(Path(env_pack_path).resolve()),
            'created_at': time.time()
        }
        self._save_cache_index()
    
    def cleanup_old_envs(self, max_age_days: int = 30) -> None:
        """Remove environments older than max_age_days."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        to_remove = []
        for file_hash, cache_info in self.cache_index.items():
            if current_time - cache_info['created_at'] > max_age_seconds:
                env_path = cache_info['path']
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

    def __init__(self, env_source: Union[str, Path, Dict, EnvSpec], env_dir: Optional[Path] = None, **kwargs):
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
        self._env_dir = Path(env_dir) if env_dir else None
        self._owns_env_dir = env_dir is None
        self._env_path = None
        self._is_extracted = False
        self.env_cache = {}
        self.env_name = kwargs.get('env_name')
        self._kwargs = kwargs

    @property
    def env_path(self) -> Path:
        """Get the path to the conda environment."""
        if self._env_path is None:
            if self._env_dir is None:
                self._env_dir = Path(tempfile.mkdtemp())
                self._owns_env_dir = True
            self._env_path = self._env_dir / 'env'
        return self._env_path

    @env_path.setter
    def env_path(self, value: Optional[Path]):
        """Set the path to the conda environment."""
        self._env_path = Path(value) if value else None

    @classmethod
    def create_temp_env(cls, packages: List[Union[str, Dict]], channels: Optional[List[str]] = None, name: str = None, **kwargs) -> 'CondaEnvExecutor':
        """Create an executor with a temporary environment.

        Args:
            packages: List of packages to install.
            channels: Optional list of channels to use.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            CondaEnvExecutor instance.
        """
        spec = {
            'name': name or 'temp_env',
            'channels': channels or ['conda-forge'],
            'dependencies': packages
        }
        return cls(spec, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path], **kwargs) -> 'CondaEnvExecutor':
        """Create an executor from a YAML environment specification file.

        Args:
            yaml_file: Path to the YAML file.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            CondaEnvExecutor instance.
        """
        return cls(yaml_file, **kwargs)
    
    def _create_conda_env(self) -> float:
        """Create a conda environment from the specified source.

        Returns:
            float: Time taken to create the environment in seconds.
        """
        start_time = time.time()

        # If source is a conda-pack file, extract it
        if str(self.env_source).endswith(('.tar.gz', '.tgz')):
            try:
                with tarfile.open(self.env_source, 'r:gz') as tar:
                    tar.extractall(path=self.env_path)
                return time.time() - start_time
            except Exception as e:
                raise RuntimeError(f"Failed to extract conda-pack file: {e}")

        # Create environment from specification
        spec = read_env_spec(self.env_source)
        env_file = os.path.join(self.env_path.parent, "environment.yaml")
        os.makedirs(os.path.dirname(env_file), exist_ok=True)

        # Write environment specification to file
        with open(env_file, 'w') as f:
            yaml.safe_dump(spec.to_dict(), f)

        # Create the environment
        if os.name == 'nt':  # Windows
            create_cmd = f"conda env create -p {self.env_path} -f {env_file} -y"
        else:  # Unix-like
            create_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda env create -p {self.env_path} -f {env_file} -y"

        try:
            subprocess.run(create_cmd, shell=True, check=True, capture_output=True, text=True)
            return time.time() - start_time
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create conda environment: {e.stderr}")
    
    def _extract_env(self) -> float:
        """Extract or create the conda environment.

        Returns:
            float: Time taken to extract/create the environment in seconds.
        """
        if self._is_extracted:
            return 0.0

        # Create environment directory if it doesn't exist
        os.makedirs(self.env_path, exist_ok=True)

        # Check if environment already exists and is valid
        python_path = os.path.join(self.env_path, 'bin', 'python')
        if os.name == 'nt':  # Windows
            python_path = os.path.join(self.env_path, 'python.exe')

        if os.path.exists(python_path):
            try:
                result = subprocess.run(
                    [python_path, "-c", "import sys; print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if result.stdout.strip():
                    self._is_extracted = True
                    return 0.0  # Environment is valid and ready
            except subprocess.CalledProcessError:
                pass  # Environment exists but is invalid, recreate it

        # Create or extract the environment
        setup_time = self._create_conda_env()
        self._is_extracted = True
        return setup_time
    
    def _create_execution_script(self, code: str, input_data: Any = None) -> str:
        """Create a Python script that will be executed in the conda environment.

        Args:
            code: The Python code to execute.
            input_data: Optional input data to pass to the execute function.

        Returns:
            str: Path to the created execution script.
        """
        script_dir = os.path.join(self.env_path, '.scripts')
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, 'execute.py')

        script_content = '''
import os
import sys
import json
import traceback
import numpy as np
from typing import Any

def write_output(output: Any) -> None:
    """Write output to the output file."""
    # Convert numpy arrays to lists
    if isinstance(output, np.ndarray):
        output = output.tolist()
    elif isinstance(output, dict):
        output = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in output.items()}
    with open(os.environ['EXECUTOR_OUTPUT_FILE'], 'w') as f:
        json.dump({'success': True, 'result': output}, f)

def write_error(error: str) -> None:
    """Write error to the output file."""
    with open(os.environ['EXECUTOR_OUTPUT_FILE'], 'w') as f:
        json.dump({'success': False, 'error': error}, f)

try:
    # Execute the user code
{0}

    # Get input data if provided
    input_data = None
    if 'EXECUTOR_INPUT_FILE' in os.environ:
        with open(os.environ['EXECUTOR_INPUT_FILE'], 'r') as f:
            input_data = json.load(f)
            # Convert lists back to numpy arrays if needed
            if isinstance(input_data, list):
                input_data = np.array(input_data)
            elif isinstance(input_data, dict):
                input_data = {k: np.array(v) if isinstance(v, list) else v for k, v in input_data.items()}

    # Call the execute function
    if 'execute' not in locals():
        raise NameError("Code must define an 'execute' function")
    
    result = execute(input_data) if input_data is not None else execute()
    write_output(result)

except Exception as e:
    error_msg = f"{{str(e)}}\\n{{traceback.format_exc()}}"
    write_error(error_msg)
    sys.exit(1)
'''.format(textwrap.indent(code.strip(), '    '))

        with open(script_path, 'w') as f:
            f.write(script_content)

        return script_path
    
    def execute(self, code: str, input_data: Any = None) -> ExecutionResult:
        """Execute Python code in the conda environment.

        Args:
            code: The Python code to execute.
            input_data: Optional input data to pass to the execute function.

        Returns:
            ExecutionResult object containing the execution result or error.
        """
        start_time = time.time()
        env_setup_time = self._extract_env()

        # Create temporary files for input/output
        script_dir = os.path.join(self.env_path, '.scripts')
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, 'execute.py')
        output_file = os.path.join(script_dir, 'output.json')
        input_file = None

        try:
            # Create the execution script
            script_template = '''
import os
import sys
import json
import traceback
from typing import Any

def write_output(output: Any) -> None:
    """Write output to the output file."""
    with open(os.environ['EXECUTOR_OUTPUT_FILE'], 'w') as f:
        json.dump({{'success': True, 'result': output}}, f)

def write_error(error: str) -> None:
    """Write error to the output file."""
    with open(os.environ['EXECUTOR_OUTPUT_FILE'], 'w') as f:
        json.dump({{'success': False, 'error': error}}, f)

try:
{0}

    # Get input data if provided
    input_data = None
    if 'EXECUTOR_INPUT_FILE' in os.environ:
        with open(os.environ['EXECUTOR_INPUT_FILE'], 'r') as f:
            input_data = json.load(f)
            
            # Convert lists to numpy arrays if numpy is available and input_data
            # looks like it might have been a numpy array
            try:
                import numpy as np
                
                # If input_data is a list of lists with consistent dimensions, convert to np.array
                if isinstance(input_data, list):
                    if all(isinstance(x, list) for x in input_data) or all(not isinstance(x, list) for x in input_data):
                        input_data = np.array(input_data)
            except ImportError:
                # numpy not available, keep as list
                pass

    # Call the execute function
    if 'execute' not in locals() and 'execute' not in globals():
        error_msg = "NameError: Code must define an 'execute' function"
        write_error(error_msg)
        sys.exit(1)

    result = execute(input_data)
    write_output(result)

except Exception as e:
    error_msg = str(e)
    if isinstance(e, NameError) and "execute" in str(e):
        error_msg = f"NameError: Code must define an 'execute' function"
    elif isinstance(e, SyntaxError):
        error_msg = f"SyntaxError: Syntax error in code: {{str(e)}}"
    else:
        error_msg = f"{{str(e)}}\\n{{traceback.format_exc()}}"
    write_error(error_msg)
    sys.exit(1)
'''
            # Process the user code
            # First, dedent the code to remove common leading whitespace
            processed_code = textwrap.dedent(code)
            # Now indent each line by 4 spaces
            processed_code = '\n'.join('    ' + line if line.strip() else line for line in processed_code.split('\n'))
            
            # Now format it into the script template
            script_content = script_template.format(processed_code)

            # Write the script
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Write input data if provided
            if input_data is not None:
                input_file = os.path.join(script_dir, 'input.json')
                with open(input_file, 'w') as f:
                    if isinstance(input_data, np.ndarray):
                        # Convert numpy arrays to lists, preserving shape
                        input_data_json = input_data.tolist()
                    elif isinstance(input_data, dict):
                        # Convert any numpy arrays in dictionaries
                        input_data_json = {}
                        for k, v in input_data.items():
                            if isinstance(v, np.ndarray):
                                input_data_json[k] = v.tolist()
                            else:
                                input_data_json[k] = v
                    else:
                        input_data_json = input_data
                    json.dump(input_data_json, f)

            # Set environment variables
            env = os.environ.copy()
            env['EXECUTOR_OUTPUT_FILE'] = output_file
            if input_file:
                env['EXECUTOR_INPUT_FILE'] = input_file

            # Execute the script
            python_path = os.path.join(self.env_path, 'bin', 'python')
            if os.name == 'nt':  # Windows
                python_path = os.path.join(self.env_path, 'python.exe')

            start_exec_time = time.time()
            try:
                result = subprocess.run([python_path, script_path], env=env, check=True, capture_output=True, text=True)
                # Read output
                with open(output_file, 'r') as f:
                    output = json.load(f)

                execution_time = time.time() - start_exec_time
                total_time = time.time() - start_time

                return ExecutionResult(
                    success=output['success'],
                    result=output.get('result'),
                    error=output.get('error'),
                    stdout=result.stdout,
                    stderr=result.stderr,
                    timing=TimingInfo(
                        env_setup_time=env_setup_time,
                        execution_time=execution_time,
                        total_time=total_time
                    )
                )
            except subprocess.CalledProcessError as e:
                execution_time = time.time() - start_exec_time
                total_time = time.time() - start_time

                # Try to read error from output file
                error = e.stderr
                try:
                    with open(output_file, 'r') as f:
                        output = json.load(f)
                        if not output['success']:
                            error = output['error']
                except:
                    pass

                return ExecutionResult(
                    success=False,
                    error=error,
                    stdout=e.stdout,
                    stderr=e.stderr,
                    timing=TimingInfo(
                        env_setup_time=env_setup_time,
                        execution_time=execution_time,
                        total_time=total_time
                    )
                )

        finally:
            # Clean up temporary files
            if os.path.exists(script_path):
                os.remove(script_path)
            if os.path.exists(output_file):
                os.remove(output_file)
            if input_file and os.path.exists(input_file):
                os.remove(input_file)
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._owns_env_dir and self._env_dir and self._env_dir.exists():
            shutil.rmtree(self._env_dir)
            self._env_dir = None
            self._env_path = None
            self._is_extracted = False
    
    def __enter__(self) -> 'CondaEnvExecutor':
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.cleanup() 
