"""Tests for conda environment worker."""

import asyncio
import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypha.workers.conda import CondaWorker, EnvironmentCache
from hypha.workers.base import (
    WorkerConfig,
    SessionStatus,
    SessionInfo,
    SessionNotFoundError,
    WorkerError,
)
from hypha.workers.conda_executor import ExecutionResult, TimingInfo

# Mark all async functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio


class TestEnvironmentCache:
    """Test the environment cache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = EnvironmentCache(cache_dir=self.temp_dir, max_size=3)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_env_hash(self):
        """Test environment hash computation."""
        dependencies1 = ["python=3.11", "numpy", "pandas"]
        channels1 = ["conda-forge", "defaults"]

        dependencies2 = ["pandas", "numpy", "python=3.11"]  # Different order
        channels2 = ["defaults", "conda-forge"]  # Different order

        # Same dependencies and channels should produce same hash regardless of order
        hash1 = self.cache._compute_env_hash(dependencies1, channels1)
        hash2 = self.cache._compute_env_hash(dependencies2, channels2)
        assert hash1 == hash2

        # Different dependencies should produce different hash
        dependencies3 = ["python=3.10", "numpy", "pandas"]
        hash3 = self.cache._compute_env_hash(dependencies3, channels1)
        assert hash1 != hash3

    def test_cache_operations(self):
        """Test basic cache operations."""
        dependencies = ["python=3.11", "numpy"]
        channels = ["conda-forge"]
        env_path = Path(self.temp_dir) / "test_env"
        env_path.mkdir()
        (env_path / "bin").mkdir()
        (env_path / "bin" / "python").touch()

        # Initially no cached environment
        cached = self.cache.get_cached_env(dependencies, channels)
        assert cached is None

        # Add to cache
        self.cache.add_cached_env(dependencies, channels, env_path)

        # Should now be in cache
        cached = self.cache.get_cached_env(dependencies, channels)
        assert cached == env_path

        # Check cache index was updated
        assert len(self.cache.index) == 1
        env_hash = self.cache._compute_env_hash(dependencies, channels)
        assert env_hash in self.cache.index
        assert self.cache.index[env_hash]["path"] == str(env_path)

    def test_cache_validation(self):
        """Test cache entry validation."""
        dependencies = ["python=3.11"]
        channels = ["conda-forge"]
        env_path = Path(self.temp_dir) / "invalid_env"

        # Add invalid environment (doesn't exist)
        self.cache.add_cached_env(dependencies, channels, env_path)

        # Should not return invalid environment and should clean it up
        cached = self.cache.get_cached_env(dependencies, channels)
        assert cached is None
        assert len(self.cache.index) == 0

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Fill cache to capacity
        for i in range(3):
            dependencies = [f"python=3.{i+9}"]
            channels = ["conda-forge"]
            env_path = Path(self.temp_dir) / f"env_{i}"
            env_path.mkdir()
            (env_path / "bin").mkdir()
            (env_path / "bin" / "python").touch()

            self.cache.add_cached_env(dependencies, channels, env_path)

        assert len(self.cache.index) == 3

        # Access first environment to make it more recently used
        first_dependencies = ["python=3.9"]
        first_channels = ["conda-forge"]
        cached = self.cache.get_cached_env(first_dependencies, first_channels)
        assert cached is not None

        # Add one more environment (should evict least recently used)
        new_dependencies = ["python=3.12"]
        new_channels = ["conda-forge"]
        new_env_path = Path(self.temp_dir) / "env_new"
        new_env_path.mkdir()
        (new_env_path / "bin").mkdir()
        (new_env_path / "bin" / "python").touch()

        self.cache.add_cached_env(new_dependencies, new_channels, new_env_path)

        # Should still have 3 entries (max size)
        assert len(self.cache.index) == 3

        # First environment should still be there (was accessed recently)
        cached = self.cache.get_cached_env(first_dependencies, first_channels)
        assert cached is not None

        # New environment should be there
        cached = self.cache.get_cached_env(new_dependencies, new_channels)
        assert cached is not None

    def test_age_based_eviction(self):
        """Test age-based cache eviction."""
        dependencies = ["python=3.11"]
        channels = ["conda-forge"]
        env_path = Path(self.temp_dir) / "old_env"
        env_path.mkdir()
        (env_path / "bin").mkdir()
        (env_path / "bin" / "python").touch()

        # Add environment with old timestamp
        self.cache.add_cached_env(dependencies, channels, env_path)
        env_hash = self.cache._compute_env_hash(dependencies, channels)

        # Manually set old creation time (35 days ago)
        old_time = time.time() - (35 * 24 * 60 * 60)
        self.cache.index[env_hash]["created_at"] = old_time
        self.cache._save_index()

        # Trigger eviction by accessing cache
        self.cache._evict_if_needed()

        # Old environment should be removed
        assert len(self.cache.index) == 0
        cached = self.cache.get_cached_env(dependencies, channels)
        assert cached is None


class TestCondaWorkerBasic:
    """Test basic conda environment worker functionality without mocking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server = MagicMock()
        self.worker = CondaWorker()

    def test_supported_types(self):
        """Test supported application types."""
        types = self.worker.supported_types
        assert "python-conda" in types
        assert len(types) == 1

    def test_worker_properties(self):
        """Test worker properties."""
        assert "Conda Environment Worker" in self.worker.name
        assert "conda environments" in self.worker.description

    async def test_compile_manifest(self):
        """Test manifest compilation and validation."""
        # Test with dependencies field
        manifest1 = {
            "type": "python-conda",
            "dependencies": ["python=3.11", "numpy"],
            "channels": ["conda-forge"],
        }
        compiled_manifest, files = await self.worker.compile(manifest1, [])

        # Should contain original dependencies plus automatically added ones
        deps = compiled_manifest["dependencies"]
        assert "python=3.11" in deps
        assert "numpy" in deps
        assert "pip" in deps  # Automatically added
        assert {"pip": ["hypha-rpc"]} in deps  # Automatically added
        assert compiled_manifest["channels"] == ["conda-forge"]

        # Test with dependencies field (alternate name)
        manifest2 = {
            "type": "python-conda",
            "dependencies": "python=3.11",  # String should be converted to list
            "channels": "conda-forge",  # String should be converted to list
        }
        compiled_manifest, files = await self.worker.compile(manifest2, [])

        deps = compiled_manifest["dependencies"]
        assert "python=3.11" in deps
        assert "pip" in deps  # Automatically added
        assert {"pip": ["hypha-rpc"]} in deps  # Automatically added
        assert compiled_manifest["channels"] == ["conda-forge"]

        # Test with no dependencies (should add default)
        manifest3 = {"type": "python-conda"}
        compiled_manifest, files = await self.worker.compile(manifest3, [])

        deps = compiled_manifest["dependencies"]
        assert "pip" in deps  # Automatically added
        assert {"pip": ["hypha-rpc"]} in deps  # Automatically added
        assert compiled_manifest["channels"] == ["conda-forge"]

    def test_get_service(self):
        """Test service configuration."""
        service_config = self.worker.get_worker_service()

        assert "id" in service_config
        assert "name" in service_config
        assert "description" in service_config
        assert "supported_types" in service_config
        assert "start" in service_config
        assert "stop" in service_config
        assert "execute_code" in service_config

        # Check supported types
        assert "python-conda" in service_config["supported_types"]
        assert len(service_config["supported_types"]) == 1


class TestCondaWorkerIntegration:
    """Integration tests for conda environment worker using real conda environments."""

    async def test_real_conda_basic_execution(
        self, conda_integration_server, conda_test_workspace
    ):
        """Test basic conda environment creation and code execution."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache

        # Initialize worker with clean cache
        worker = CondaWorker()
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"], max_size=5
        )

        # Simple script with execute function
        script = """
def execute(input_data):
    import sys
    import platform
    
    result = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system(),
        "input_data": input_data,
        "computation": sum(range(10)) if input_data is None else input_data * 2
    }
    return result
"""

        config = WorkerConfig(
            id="real-conda-test",
            app_id="test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "python-conda",
                "dependencies": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
        )

        # Mock HTTP client for script fetching
        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            try:
                print("🚀 Starting real conda environment session...")
                session_id = await worker.start(config)
                assert session_id == "real-conda-test"

                # Verify session was created
                session_info = await worker.get_session_info(session_id)
                assert session_info.status == SessionStatus.RUNNING

                print("⚙️ Executing code in real conda environment...")
                # Execute code directly
                test_code = """
import sys
import platform

result = {
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    "platform": platform.system(),
    "input_data": 21,
    "computation": 21 * 2
}

print(f"Result: {result}")
"""
                result = await worker.execute_code(session_id, test_code)

                assert (
                    result.success
                ), f"Execution failed: {result.error}\nStderr: {result.stderr}"

                # Check that the code executed successfully by looking at stdout
                assert (
                    "Result:" in result.stdout
                ), f"Expected result output in stdout: {result.stdout}"
                assert (
                    "'computation': 42" in result.stdout
                ), f"Expected computation=42 in stdout: {result.stdout}"
                assert (
                    "'input_data': 21" in result.stdout
                ), f"Expected input_data=21 in stdout: {result.stdout}"

                print(f"✅ Execution successful: {result.stdout.strip()}")

                # Test another code execution
                test_code2 = """
import sys

result = {
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    "computation": sum(range(10))  # Should be 45
}

print(f"Result2: {result}")
"""
                result2 = await worker.execute_code(session_id, test_code2)
                assert result2.success, f"Second execution failed: {result2.error}"
                assert (
                    "'computation': 45" in result2.stdout
                ), f"Expected computation=45 in stdout: {result2.stdout}"

                # Check logs
                logs = await worker.get_logs(session_id)
                assert "info" in logs
                assert len(logs["info"]) > 0

                # Stop session
                await worker.stop(session_id)

            except Exception as e:
                print(f"❌ Integration test failed: {e}")
                # Cleanup on error
                try:
                    await worker.stop("real-conda-test")
                except:
                    pass
                raise

    async def test_real_conda_package_installation(
        self, conda_integration_server, conda_test_workspace
    ):
        """Test conda environment with additional dependencies."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache

        worker = CondaWorker()
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"], max_size=5
        )

        # Script that uses numpy (needs to be installed)
        script = """
def execute(input_data):
    import numpy as np
    import sys
    
    # Test numpy functionality
    arr = np.array([1, 2, 3, 4, 5])
    
    result = {
        "numpy_version": np.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "array_sum": int(np.sum(arr)),
        "array_mean": float(np.mean(arr)),
        "input_processed": input_data.tolist() if isinstance(input_data, np.ndarray) else input_data
    }
    
    if input_data is not None and hasattr(input_data, '__len__') and len(input_data) > 0:
        # Handle both lists and numpy arrays
        if isinstance(input_data, list):
            input_array = np.array(input_data)
        else:
            input_array = input_data  # Already a numpy array
        result["input_sum"] = int(np.sum(input_array))
        result["input_mean"] = float(np.mean(input_array))
    
    return result
"""

        config = WorkerConfig(
            id="numpy-conda-test",
            app_id="numpy-test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "python-conda",
                "dependencies": ["python=3.11", "numpy"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
        )

        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            try:
                print("🚀 Creating conda environment with numpy...")
                session_id = await worker.start(config)

                print("⚙️ Testing numpy functionality...")
                test_code = """
import numpy as np
import sys

# Test numpy functionality
arr = np.array([1, 2, 3, 4, 5])
input_data = [10, 20, 30, 40, 50]
input_array = np.array(input_data)

result = {
    "numpy_version": np.__version__,
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    "array_sum": int(np.sum(arr)),
    "array_mean": float(np.mean(arr)),
    "input_processed": input_data,
    "input_sum": int(np.sum(input_array)),
    "input_mean": float(np.mean(input_array))
}

print(f"NumPy result: {result}")
"""
                result = await worker.execute_code(session_id, test_code)

                assert (
                    result.success
                ), f"Numpy test failed: {result.error}\nStderr: {result.stderr}"
                assert (
                    "NumPy result:" in result.stdout
                ), f"Expected result in stdout: {result.stdout}"
                assert (
                    "'array_sum': 15" in result.stdout
                ), f"Expected array_sum=15 in stdout: {result.stdout}"
                assert (
                    "'input_sum': 150" in result.stdout
                ), f"Expected input_sum=150 in stdout: {result.stdout}"

                print(f"✅ NumPy test successful: {result.stdout.strip()}")

                await worker.stop(session_id)

            except Exception as e:
                print(f"❌ NumPy integration test failed: {e}")
                try:
                    await worker.stop("numpy-conda-test")
                except:
                    pass
                raise

    async def test_real_conda_caching_behavior(
        self, conda_integration_server, conda_test_workspace
    ):
        """Test that conda environments are properly cached and reused."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache

        worker = CondaWorker()
        cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"], max_size=5
        )
        worker._env_cache = cache

        script = """
def execute(input_data):
    import os
    import sys
    return {
        "python_executable": sys.executable,
        "environment_variables": dict(os.environ).get("CONDA_DEFAULT_ENV", "none"),
        "input": input_data
    }
"""

        # First session with specific dependencies
        config1 = WorkerConfig(
            id="cache-test-1",
            app_id="cache-test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "python-conda",
                "dependencies": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
        )

        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            try:
                print("🚀 Creating first conda environment...")
                # Check cache is initially empty
                initial_cache_size = len(cache.index)

                # Start first session
                session_id1 = await worker.start(config1)
                test_code1 = """
import os
import sys

result = {
    "python_executable": sys.executable,
    "environment_variables": dict(os.environ).get("CONDA_DEFAULT_ENV", "none"),
    "input": "first"
}

print(f"Cache test result 1: {result}")
"""
                result1 = await worker.execute_code(session_id1, test_code1)
                assert result1.success, f"First execution failed: {result1.error}"

                # Check cache was populated
                first_cache_size = len(cache.index)
                assert first_cache_size == initial_cache_size + 1

                # Extract environment path from stdout
                import re

                match1 = re.search(r"'python_executable': '([^']+)'", result1.stdout)
                assert (
                    match1
                ), f"Could not find python_executable in stdout: {result1.stdout}"
                env_path1 = match1.group(1)

                await worker.stop(session_id1)

                print("🔄 Creating second session with same dependencies...")
                # Start second session with same dependencies (should use cache)
                config2 = WorkerConfig(
                    id="cache-test-2",
                    app_id="cache-test-app-2",
                    workspace="test-workspace",
                    client_id="test-client",
                    server_url="http://test-server",
                    token="test-token",
                    entry_point="main.py",
                    artifact_id="test-artifact",
                    manifest={
                        "type": "python-conda",
                        "dependencies": ["python=3.11"],  # Same dependencies
                        "channels": ["conda-forge"],  # Same channels
                        "entry_point": "main.py",
                    },
                    app_files_base_url="http://test-server/files",
                )

                session_id2 = await worker.start(config2)
                test_code2 = """
import os
import sys

result = {
    "python_executable": sys.executable,
    "environment_variables": dict(os.environ).get("CONDA_DEFAULT_ENV", "none"),
    "input": "second"
}

print(f"Cache test result 2: {result}")
"""
                result2 = await worker.execute_code(session_id2, test_code2)
                assert result2.success, f"Second execution failed: {result2.error}"

                # Extract environment path from stdout
                match2 = re.search(r"'python_executable': '([^']+)'", result2.stdout)
                assert (
                    match2
                ), f"Could not find python_executable in stdout: {result2.stdout}"
                env_path2 = match2.group(1)

                # Cache size should not have increased (reused existing)
                second_cache_size = len(cache.index)
                assert second_cache_size == first_cache_size

                print(f"✅ Cache working correctly:")
                print(f"  First environment: {env_path1}")
                print(f"  Second environment: {env_path2}")
                print(f"  Environment reused: {env_path1 == env_path2}")
                print(f"  Cache entries: {second_cache_size}")

                await worker.stop(session_id2)

            except Exception as e:
                print(f"❌ Caching test failed: {e}")
                try:
                    await worker.stop("cache-test-1")
                    await worker.stop("cache-test-2")
                except:
                    pass
                raise

    async def test_real_conda_mixed_dependencies(
        self, conda_integration_server, conda_test_workspace
    ):
        """Test conda environment with both conda and pip dependencies."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache

        worker = CondaWorker()
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"], max_size=5
        )

        # Script that uses both conda (numpy) and pip dependencies
        script = """
def execute(input_data):
    import sys
    import numpy as np
    
    # Test basic functionality
    result = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy_available": True,
        "numpy_version": np.__version__,
    }
    
    # Try to import a pip package if available
    try:
        import requests
        result["requests_available"] = True
        result["requests_version"] = requests.__version__
    except ImportError:
        result["requests_available"] = False
    
    if input_data is not None and hasattr(input_data, '__len__') and len(input_data) > 0:
        arr = np.array(input_data)
        result["data_processed"] = {
            "sum": float(np.sum(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }
    
    return result
"""

        config = WorkerConfig(
            id="mixed-dependencies-test",
            app_id="mixed-test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "python-conda",
                "dependencies": ["python=3.11", "numpy", {"pip": ["requests"]}],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
        )

        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            try:
                print("🚀 Creating environment with mixed conda/pip dependencies...")
                session_id = await worker.start(config)

                print("⚙️ Testing mixed package functionality...")
                test_code = """
import sys
import numpy as np

result = {
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    "numpy_available": True,
    "numpy_version": np.__version__
}

try:
    import requests
    result["requests_available"] = True
    result["requests_version"] = requests.__version__
except ImportError:
    result["requests_available"] = False

# Test data processing
test_data = [1.5, 2.3, 3.7, 4.1, 5.9]
arr = np.array(test_data)
result["data_processed"] = {
    "sum": float(np.sum(arr)),
    "mean": float(np.mean(arr)),
    "std": float(np.std(arr))
}

print(f"Mixed dependencies result: {result}")
"""
                result = await worker.execute_code(session_id, test_code)

                assert (
                    result.success
                ), f"Mixed dependencies test failed: {result.error}\nStderr: {result.stderr}"

                # Verify conda package (numpy) works
                assert (
                    "'numpy_available': True" in result.stdout
                ), f"Expected numpy_available=True in stdout: {result.stdout}"
                assert (
                    "'numpy_version':" in result.stdout
                ), f"Expected numpy_version in stdout: {result.stdout}"

                # Verify pip package (requests) works
                assert (
                    "'requests_available': True" in result.stdout
                ), f"Expected requests_available=True in stdout: {result.stdout}"
                assert (
                    "'requests_version':" in result.stdout
                ), f"Expected requests_version in stdout: {result.stdout}"

                # Verify data processing works - check for expected sum and mean
                expected_sum = sum([1.5, 2.3, 3.7, 4.1, 5.9])  # 17.5
                expected_mean = expected_sum / 5  # 3.5
                assert (
                    f"'sum': {expected_sum}" in result.stdout
                ), f"Expected sum={expected_sum} in stdout: {result.stdout}"
                assert (
                    f"'mean': {expected_mean}" in result.stdout
                ), f"Expected mean={expected_mean} in stdout: {result.stdout}"

                print(f"✅ Mixed dependencies test successful:")
                print(f"  Output: {result.stdout.strip()}")

                await worker.stop(session_id)

            except Exception as e:
                print(f"❌ Mixed dependencies test failed: {e}")
                try:
                    await worker.stop("mixed-dependencies-test")
                except:
                    pass
                raise

    async def test_real_conda_standalone_script(
        self, conda_integration_server, conda_test_workspace
    ):
        """Test conda environment with a standalone script (no execute function)."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache

        worker = CondaWorker()
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"], max_size=5
        )

        # Standalone script without execute function
        script = """
import sys
import platform
import os

print("=== Conda Environment Info ===")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python executable: {sys.executable}")

# Test basic computation
result = sum(range(100))
print(f"Computation result: {result}")

# Test environment variables
conda = os.environ.get("CONDA_DEFAULT_ENV", "Not set")
print(f"Conda environment: {conda}")

print("=== Script completed successfully ===")
"""

        config = WorkerConfig(
            id="standalone-script-test",
            app_id="standalone-test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "python-conda",
                "dependencies": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
        )

        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            try:
                print("🚀 Running standalone script in conda environment...")
                session_id = await worker.start(config)

                # For standalone scripts, the code runs during start()
                # Check that session is running
                session_info = await worker.get_session_info(session_id)
                assert session_info.status == SessionStatus.RUNNING

                # Get logs to verify script executed
                logs = await worker.get_logs(session_id)

                # Check that stdout contains expected output
                if "stdout" in logs and logs["stdout"]:
                    stdout_content = "\n".join(logs["stdout"])
                    assert "Conda Environment Info" in stdout_content
                    assert "Python version:" in stdout_content
                    assert (
                        "Computation result: 4950" in stdout_content
                    )  # sum(range(100))
                    assert "Script completed successfully" in stdout_content

                    print("✅ Standalone script executed successfully:")
                    print("  Output captured in logs")
                    print(f"  Log types: {list(logs.keys())}")
                else:
                    # Check info logs for execution info
                    assert "info" in logs
                    assert len(logs["info"]) > 0
                    print("✅ Standalone script session created successfully")

                await worker.stop(session_id)

            except Exception as e:
                print(f"❌ Standalone script test failed: {e}")
                try:
                    await worker.stop("standalone-script-test")
                except:
                    pass
                raise


class TestCondaWorkerProgressCallback:
    """Test progress callback functionality for conda environment worker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.server = MagicMock()
        self.worker = CondaWorker()
        self.progress_messages = []

        def mock_progress_callback(info):
            self.progress_messages.append(info)

        self.progress_callback = mock_progress_callback

    async def test_progress_callback_with_new_environment(self):
        """Test progress callback during new environment creation."""
        from unittest.mock import AsyncMock

        # Mock script content
        script = """
def execute(input_data):
    return {"result": "test"}
"""

        config = WorkerConfig(
            id="progress-test",
            app_id="test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "python-conda",
                "dependencies": ["python=3.11", "numpy"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
            progress_callback=self.progress_callback,
        )

        # Mock the environment cache to return None (no cached environment)
        self.worker._env_cache = MagicMock()
        self.worker._env_cache.get_cached_env.return_value = None
        self.worker._env_cache.add_cached_env = MagicMock()

        # Mock the HTTP client
        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            # Mock the CondaEnvExecutor and its methods
            with patch("hypha.workers.conda.CondaEnvExecutor") as mock_executor_class:
                mock_executor = MagicMock()
                mock_executor_class.create_temp_env.return_value = mock_executor

                # Mock async _extract_env method
                async def mock_extract_env(progress_callback=None):
                    if progress_callback:
                        progress_callback(
                            {"type": "info", "message": "Mock environment setup"}
                        )
                    return 10.5  # Mock setup time

                mock_executor._extract_env = mock_extract_env

                # Mock execution result
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.result = {"result": "test"}
                mock_result.stdout = "Test output"
                mock_result.stderr = ""
                mock_result.error = None
                mock_result.timing = None
                mock_executor.execute.return_value = mock_result

                # Start the session
                session_id = await self.worker.start(config)

                # Verify session was created
                assert session_id == "progress-test"
                assert len(self.progress_messages) > 0

                # Check for expected progress messages
                message_types = [msg["type"] for msg in self.progress_messages]
                message_texts = [msg["message"] for msg in self.progress_messages]

                # Should have info and success messages
                assert "info" in message_types
                assert "success" in message_types

                # Check for specific expected messages
                expected_patterns = [
                    "Starting conda environment session",
                    "Fetching application script",
                    "Setting up conda environment",
                    "Checking for cached conda environment",
                    "Creating new conda environment",
                    "Installing packages",
                    "Executing initialization script",
                    "started successfully",
                ]

                message_text = " ".join(message_texts)
                found_patterns = 0
                for pattern in expected_patterns:
                    if any(pattern.lower() in msg.lower() for msg in message_texts):
                        found_patterns += 1

                # Should find most of the expected patterns
                assert (
                    found_patterns >= len(expected_patterns) // 2
                ), f"Only found {found_patterns} patterns in messages: {message_texts}"

                print(
                    f"✅ Progress callback test passed with {len(self.progress_messages)} messages"
                )
                print(
                    f"   Found {found_patterns}/{len(expected_patterns)} expected patterns"
                )

                # Clean up
                await self.worker.stop(session_id)

    async def test_progress_callback_with_cached_environment(self):
        """Test progress callback when using cached environment."""
        from pathlib import Path

        script = """
def execute(input_data):
    return {"result": "cached_test"}
"""

        config = WorkerConfig(
            id="cached-progress-test",
            app_id="cached-test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "python-conda",
                "dependencies": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
            progress_callback=self.progress_callback,
        )

        # Mock the environment cache to return a cached environment
        mock_cached_path = Path("/fake/cached/env")
        self.worker._env_cache = MagicMock()
        self.worker._env_cache.get_cached_env.return_value = mock_cached_path

        # Mock the HTTP client
        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            # Mock the CondaEnvExecutor
            with patch("hypha.workers.conda.CondaEnvExecutor") as mock_executor_class:
                mock_executor = MagicMock()
                mock_executor_class.return_value = mock_executor
                mock_executor.env_path = mock_cached_path
                mock_executor._is_extracted = True

                # Mock execution result
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.result = {"result": "cached_test"}
                mock_result.stdout = "Cached test output"
                mock_result.stderr = ""
                mock_result.error = None
                mock_result.timing = None
                mock_executor.execute.return_value = mock_result

                # Start the session
                session_id = await self.worker.start(config)

                # Verify session was created
                assert session_id == "cached-progress-test"
                assert len(self.progress_messages) > 0

                # Check for cached environment specific messages
                message_texts = [msg["message"] for msg in self.progress_messages]

                # Should mention cached environment
                cached_mentioned = any("cached" in msg.lower() for msg in message_texts)
                assert (
                    cached_mentioned
                ), f"Cached environment not mentioned in messages: {message_texts}"

                print(f"✅ Cached environment progress callback test passed")
                print(
                    f"   Messages: {[msg['message'] for msg in self.progress_messages]}"
                )

                # Clean up
                await self.worker.stop(session_id)

    async def test_progress_callback_error_handling(self):
        """Test progress callback during error scenarios."""

        config = WorkerConfig(
            id="error-progress-test",
            app_id="error-test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "python-conda",
                "dependencies": ["nonexistent-package==999.999.999"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
            progress_callback=self.progress_callback,
        )

        # Mock HTTP client to fail
        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = Exception(
                "Failed to fetch script"
            )
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            try:
                await self.worker.start(config)
                assert False, "Expected exception was not raised"
            except Exception as e:
                # Should have progress messages including error
                assert len(self.progress_messages) > 0

                message_types = [msg["type"] for msg in self.progress_messages]
                message_texts = [msg["message"] for msg in self.progress_messages]

                # Should have error message
                assert (
                    "error" in message_types
                ), f"No error message found in types: {message_types}"

                # Error message should mention the failure
                error_messages = [
                    msg["message"]
                    for msg in self.progress_messages
                    if msg["type"] == "error"
                ]
                assert any(
                    "failed" in msg.lower() for msg in error_messages
                ), f"No failure mentioned in error messages: {error_messages}"

                print(f"✅ Error handling progress callback test passed")
                print(f"   Error messages: {error_messages}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
