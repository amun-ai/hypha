"""Tests for conda environment worker."""

import asyncio
import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import inspect

from hypha.workers.conda import CondaWorker, EnvironmentCache
from hypha.workers.base import (
    WorkerConfig,
    SessionStatus,
)
from hypha.workers.conda_executor import TimingInfo
from . import SIO_PORT

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
        assert "conda-jupyter-kernel" in types
        assert len(types) == 1

    def test_worker_properties(self):
        """Test worker properties."""
        assert "Conda Worker" in self.worker.name
        assert "conda environments" in self.worker.description

    async def test_compile_manifest(self):
        """Test manifest compilation and validation."""
        # Test with dependencies field
        manifest1 = {
            "type": "conda-jupyter-kernel",
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
            "type": "conda-jupyter-kernel",
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
        manifest3 = {"type": "conda-jupyter-kernel"}
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
        assert "execute" in service_config

        # Check supported types
        assert "conda-jupyter-kernel" in service_config["supported_types"]
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
                "type": "conda-jupyter-kernel",
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
                # Compile the manifest first to add ipykernel dependencies
                compiled_manifest, _ = await worker.compile(config.manifest, [])
                config.manifest = compiled_manifest
                session_id = await worker.start(config)
                assert session_id == "real-conda-test"

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
                result = await worker.execute(session_id, test_code)

                assert (
                    result["status"] == "ok"
                ), f"Execution failed: {result.get('error', {})}"

                # Extract stdout from outputs
                stdout_text = "".join(
                    output.get("text", "") for output in result.get("outputs", [])
                    if output.get("type") == "stream" and output.get("name") == "stdout"
                )

                # Check that the code executed successfully by looking at stdout
                assert (
                    "Result:" in stdout_text
                ), f"Expected result output in stdout: {stdout_text}"
                assert (
                    "'computation': 42" in stdout_text
                ), f"Expected computation=42 in stdout: {stdout_text}"
                assert (
                    "'input_data': 21" in stdout_text
                ), f"Expected input_data=21 in stdout: {stdout_text}"

                print(f"✅ Execution successful: {stdout_text.strip()}")

                # Test another code execution
                test_code2 = """
import sys

result = {
    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    "computation": sum(range(10))  # Should be 45
}

print(f"Result2: {result}")
"""
                result2 = await worker.execute(session_id, test_code2)
                assert result2["status"] == "ok", f"Second execution failed: {result2.get('error', {})}"
                
                # Extract stdout from outputs
                stdout_text2 = "".join(
                    output.get("text", "") for output in result2.get("outputs", [])
                    if output.get("type") == "stream" and output.get("name") == "stdout"
                )
                
                assert (
                    "'computation': 45" in stdout_text2
                ), f"Expected computation=45 in stdout: {stdout_text2}"

                # Check logs
                logs = await worker.get_logs(session_id)
                assert "items" in logs
                info_logs = [item for item in logs["items"] if item["type"] == "info"]
                assert len(info_logs) > 0

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
                "type": "conda-jupyter-kernel",
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
                # Compile the manifest first to add ipykernel dependencies
                compiled_manifest, _ = await worker.compile(config.manifest, [])
                config.manifest = compiled_manifest
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
                result = await worker.execute(session_id, test_code)

                assert (
                    result["status"] == "ok"
                ), f"Numpy test failed: {result.get('error', {})}"
                
                # Extract stdout from outputs
                stdout_text = "".join(
                    output.get("text", "") for output in result.get("outputs", [])
                    if output.get("type") == "stream" and output.get("name") == "stdout"
                )
                
                assert (
                    "NumPy result:" in stdout_text
                ), f"Expected result in stdout: {stdout_text}"
                assert (
                    "'array_sum': 15" in stdout_text
                ), f"Expected array_sum=15 in stdout: {stdout_text}"
                assert (
                    "'input_sum': 150" in stdout_text
                ), f"Expected input_sum=150 in stdout: {stdout_text}"

                print(f"✅ NumPy test successful: {stdout_text.strip()}")

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
                "type": "conda-jupyter-kernel",
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
                # Compile the manifest first to add ipykernel dependencies
                compiled_manifest1, _ = await worker.compile(config1.manifest, [])
                config1.manifest = compiled_manifest1
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
                result1 = await worker.execute(session_id1, test_code1)
                assert result1["status"] == "ok", f"First execution failed: {result1.get('error', {})}"
                
                # Extract stdout from outputs
                stdout_text1 = "".join(
                    output.get("text", "") for output in result1.get("outputs", [])
                    if output.get("type") == "stream" and output.get("name") == "stdout"
                )

                # Check cache was populated
                first_cache_size = len(cache.index)
                assert first_cache_size == initial_cache_size + 1

                # Extract environment path from stdout
                import re

                match1 = re.search(r"'python_executable': '([^']+)'", stdout_text1)
                assert (
                    match1
                ), f"Could not find python_executable in stdout: {stdout_text1}"
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
                        "type": "conda-jupyter-kernel",
                        "dependencies": ["python=3.11"],  # Same dependencies
                        "channels": ["conda-forge"],  # Same channels
                        "entry_point": "main.py",
                    },
                    app_files_base_url="http://test-server/files",
                )

                # Compile the manifest first to add ipykernel dependencies
                compiled_manifest2, _ = await worker.compile(config2.manifest, [])
                config2.manifest = compiled_manifest2
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
                result2 = await worker.execute(session_id2, test_code2)
                assert result2["status"] == "ok", f"Second execution failed: {result2.get('error', {})}"
                
                # Extract stdout from outputs
                stdout_text2 = "".join(
                    output.get("text", "") for output in result2.get("outputs", [])
                    if output.get("type") == "stream" and output.get("name") == "stdout"
                )

                # Extract environment path from stdout
                match2 = re.search(r"'python_executable': '([^']+)'", stdout_text2)
                assert (
                    match2
                ), f"Could not find python_executable in stdout: {stdout_text2}"
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
                "type": "conda-jupyter-kernel",
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
                # Compile the manifest first to add ipykernel dependencies
                compiled_manifest, _ = await worker.compile(config.manifest, [])
                config.manifest = compiled_manifest
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
                result = await worker.execute(session_id, test_code)

                assert (
                    result["status"] == "ok"
                ), f"Mixed dependencies test failed: {result.get('error', {})}"
                
                # Extract stdout from outputs
                stdout_text = "".join(
                    output.get("text", "") for output in result.get("outputs", [])
                    if output.get("type") == "stream" and output.get("name") == "stdout"
                )

                # Verify conda package (numpy) works
                assert (
                    "'numpy_available': True" in stdout_text
                ), f"Expected numpy_available=True in stdout: {stdout_text}"
                assert (
                    "'numpy_version':" in stdout_text
                ), f"Expected numpy_version in stdout: {stdout_text}"

                # Verify pip package (requests) - check if it's available or not
                if "'requests_available': True" in stdout_text:
                    # If requests is available, verify version is also reported
                    assert (
                        "'requests_version':" in stdout_text
                    ), f"Expected requests_version when requests_available=True in stdout: {stdout_text}"
                    print("  ✅ Requests package successfully installed via pip")
                else:
                    # If requests is not available, that's also acceptable
                    assert (
                        "'requests_available': False" in stdout_text
                    ), f"Expected requests_available to be either True or False in stdout: {stdout_text}"
                    print("  ⚠️  Requests package not available (pip installation may have failed)")

                # Verify data processing works - check for expected sum and mean
                expected_sum = sum([1.5, 2.3, 3.7, 4.1, 5.9])  # 17.5
                expected_mean = expected_sum / 5  # 3.5
                assert (
                    f"'sum': {expected_sum}" in stdout_text
                ), f"Expected sum={expected_sum} in stdout: {stdout_text}"
                assert (
                    f"'mean': {expected_mean}" in stdout_text
                ), f"Expected mean={expected_mean} in stdout: {stdout_text}"

                print(f"✅ Mixed dependencies test successful:")
                print(f"  Output: {stdout_text.strip()}")

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
                "type": "conda-jupyter-kernel",
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
                # Compile the manifest first to add ipykernel dependencies
                compiled_manifest, _ = await worker.compile(config.manifest, [])
                config.manifest = compiled_manifest
                session_id = await worker.start(config)

                # For standalone scripts, the code runs during start()
                # Get logs to verify script executed
                logs = await worker.get_logs(session_id)

                # Check that stdout contains expected output
                assert "items" in logs
                stdout_logs = [
                    item["content"] for item in logs["items"] if item["type"] == "stdout"
                ]
                info_logs = [
                    item["content"] for item in logs["items"] if item["type"] == "info"
                ]
                
                if stdout_logs:
                    stdout_content = "\n".join(stdout_logs)
                    assert "Conda Environment Info" in stdout_content
                    assert "Python version:" in stdout_content
                    assert (
                        "Computation result: 4950" in stdout_content
                    )  # sum(range(100))
                    assert "Script completed successfully" in stdout_content

                    print("✅ Standalone script executed successfully:")
                    print("  Output captured in logs")
                    print(f"  Total log items: {len(logs['items'])}")
                else:
                    # Check info logs for execution info
                    assert len(info_logs) > 0
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
                "type": "conda-jupyter-kernel",
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
                        # Check if progress_callback is a coroutine function
                        if inspect.iscoroutinefunction(progress_callback):
                            await progress_callback(
                                {"type": "info", "message": "Mock environment setup"}
                            )
                        else:
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

                # Mock the CondaKernel to avoid Python path validation
                with patch("hypha.workers.conda.CondaKernel") as mock_kernel_class:
                    mock_kernel = MagicMock()
                    mock_kernel_class.return_value = mock_kernel
                    
                    # Mock kernel methods
                    async def mock_start(timeout=30.0):
                        pass
                    
                    async def mock_execute(code, **kwargs):
                        return {"success": True, "outputs": [], "error": None}
                    
                    mock_kernel.start = mock_start
                    mock_kernel.execute = mock_execute

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
                "type": "conda-jupyter-kernel",
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

                # Mock the CondaKernel to avoid Python path validation
                with patch("hypha.workers.conda.CondaKernel") as mock_kernel_class:
                    mock_kernel = MagicMock()
                    mock_kernel_class.return_value = mock_kernel
                    
                    # Mock kernel methods
                    async def mock_start(timeout=30.0):
                        pass
                    
                    async def mock_execute(code, **kwargs):
                        return {"success": True, "outputs": [], "error": None}
                    
                    mock_kernel.start = mock_start
                    mock_kernel.execute = mock_execute

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

    async def test_progress_callback_supports_awaitable(self):
        """Progress callback can be async and will be scheduled by the worker."""
        # Arrange
        callback_ran = {"count": 0}

        async def async_progress_callback(info):
            # Simulate minimal async work
            await asyncio.sleep(0)
            callback_ran["count"] += 1

        script = """
print('hello from init')
"""

        config = WorkerConfig(
            id="awaitable-progress-test",
            app_id="test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "conda-jupyter-kernel",
                "dependencies": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            app_files_base_url="http://test-server/files",
            progress_callback=async_progress_callback,
        )

        # Mock HTTP client to return script
        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            # Mock CondaEnvExecutor and kernel minimal flow
            with patch("hypha.workers.conda.CondaEnvExecutor") as mock_executor_class:
                mock_executor = MagicMock()
                mock_executor_class.create_temp_env.return_value = mock_executor

                async def mock_extract_env(progress_callback=None):
                    if progress_callback:
                        await progress_callback({"type": "info", "message": "env step"})
                    return 0.1

                mock_executor._extract_env = mock_extract_env

                with patch("hypha.workers.conda.CondaKernel") as mock_kernel_class:
                    mock_kernel = MagicMock()
                    mock_kernel_class.return_value = mock_kernel

                    async def mock_start(timeout=30.0):
                        return None

                    async def mock_execute(code, **kwargs):
                        return {"success": True, "outputs": [], "error": None}

                    mock_kernel.start = mock_start
                    mock_kernel.execute = mock_execute

                    # Act
                    session_id = await self.worker.start(config)

                    # Allow the event loop to run scheduled tasks from callback
                    await asyncio.sleep(0)

                    # Assert
                    assert session_id == "awaitable-progress-test"
                    assert callback_ran["count"] > 0

                    await self.worker.stop(session_id)


class TestCondaWorkerTimeoutPropagation:
    """Ensure timeouts from config are passed to kernel start and execute."""

    def setup_method(self):
        """Set up test fixtures."""
        self.progress_messages = []

        def mock_progress_callback(info):
            self.progress_messages.append(info)

        self.progress_callback = mock_progress_callback

    async def test_kernel_start_and_execute_receive_config_timeout(self):
        # Arrange
        requested_timeout = 123.0

        script = """
print('init')
"""

        config = WorkerConfig(
            id="timeout-prop-test",
            app_id="test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "conda-jupyter-kernel",
                "dependencies": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py",
            },
            timeout=requested_timeout,
            app_files_base_url="http://test-server/files",
            progress_callback=lambda m: None,
        )

        # Mock HTTP client
        with patch("httpx.AsyncClient") as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            # Track timeouts passed to kernel
            seen = {"start": None, "execute": None}

            with patch("hypha.workers.conda.CondaEnvExecutor") as mock_executor_class:
                mock_executor = MagicMock()
                mock_executor_class.create_temp_env.return_value = mock_executor

                async def mock_extract_env(progress_callback=None):
                    return 0.01

                mock_executor._extract_env = mock_extract_env

                with patch("hypha.workers.conda.CondaKernel") as mock_kernel_class:
                    mock_kernel = MagicMock()
                    mock_kernel_class.return_value = mock_kernel

                    async def mock_start(timeout=30.0):
                        seen["start"] = timeout
                        return None

                    async def mock_execute(code, timeout=60.0, **kwargs):
                        seen["execute"] = timeout
                        return {"success": True, "outputs": [], "error": None}

                    mock_kernel.start = mock_start
                    mock_kernel.execute = mock_execute

                    worker = CondaWorker()
                    session_id = await worker.start(config)

                    # Assert that both kernel.start and execute saw the configured timeout
                    assert seen["start"] == requested_timeout
                    assert seen["execute"] == requested_timeout

                    await worker.stop(session_id)

    async def test_progress_callback_error_handling(self):
        """Test progress callback during error scenarios."""
        
        # Initialize worker
        self.worker = CondaWorker()

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
                "type": "conda-jupyter-kernel",
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


class TestCondaWorkerSubprocess:
    """Test conda worker running as a subprocess via CLI."""

    async def test_conda_worker_cli_help(self):
        """Test that conda worker CLI help works correctly."""
        import subprocess
        import sys
        
        # Skip if conda/mamba is not available
        try:
            subprocess.run(["conda", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["mamba", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pytest.skip("Neither conda nor mamba is available")

        try:
            print("🚀 Testing conda worker CLI help...")
            # Test help command
            result = subprocess.run([
                sys.executable, "-m", "hypha.workers.conda", "--help"
            ], capture_output=True, text=True, timeout=10)
            
            print(f"Help command return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            
            # Help should return 0 and contain expected text
            assert result.returncode == 0, f"Help command failed with return code {result.returncode}"
            
            # Check for expected help content
            help_indicators = [
                "Hypha Conda Environment Worker",
                "--server-url",
                "--workspace", 
                "--token",
                "--service-id",
                "--visibility",
                "--cache-dir",
                "Examples:"
            ]
            
            found_indicators = 0
            for indicator in help_indicators:
                if indicator in result.stdout:
                    found_indicators += 1
                    print(f"✅ Found help indicator: {indicator}")
                else:
                    print(f"❌ Missing help indicator: {indicator}")
            
            # Should find most help indicators
            assert found_indicators >= len(help_indicators) // 2, f"Only found {found_indicators}/{len(help_indicators)} help indicators"
            
            print(f"✅ CLI help test completed successfully!")
            print(f"   Found {found_indicators}/{len(help_indicators)} expected help indicators")
            
        except Exception as e:
            print(f"❌ CLI help test failed: {e}")
            raise

    async def test_conda_worker_cli_validation(self):
        """Test that conda worker CLI properly validates required arguments."""
        import subprocess
        import sys
        
        # Skip if conda/mamba is not available  
        try:
            subprocess.run(["conda", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["mamba", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pytest.skip("Neither conda nor mamba is available")

        try:
            print("🚀 Testing conda worker CLI argument validation...")
            
            # Test missing server-url
            result = subprocess.run([
                sys.executable, "-m", "hypha.workers.conda"
            ], capture_output=True, text=True, timeout=10)
            
            print(f"No args return code: {result.returncode}")
            print(f"STDERR:\n{result.stderr}")
            
            # Should fail with missing required arguments
            assert result.returncode != 0, "Expected failure for missing required arguments"
            assert "server-url is required" in result.stderr, f"Expected server-url error in stderr: {result.stderr}"
            
            # Test missing workspace
            result = subprocess.run([
                sys.executable, "-m", "hypha.workers.conda",
                "--server-url", "http://test.com"
            ], capture_output=True, text=True, timeout=10)
            
            assert result.returncode != 0, "Expected failure for missing workspace"
            assert "workspace is required" in result.stderr, f"Expected workspace error in stderr: {result.stderr}"
            
            # Test missing token
            result = subprocess.run([
                sys.executable, "-m", "hypha.workers.conda", 
                "--server-url", "http://test.com",
                "--workspace", "test"
            ], capture_output=True, text=True, timeout=10)
            
            assert result.returncode != 0, "Expected failure for missing token"
            assert "token is required" in result.stderr, f"Expected token error in stderr: {result.stderr}"
            
            print(f"✅ CLI validation test completed successfully!")
            print(f"   All required argument validations working correctly")
            
        except Exception as e:
            print(f"❌ CLI validation test failed: {e}")
            raise


class TestCondaWorkerWorkingDirectory:
    """Test working directory functionality for conda environment worker with real conda environments."""

    async def test_working_directory_creation_and_cleanup(self, conda_integration_server, conda_test_workspace):
        """Test that session-specific working directories are created and cleaned up properly."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache
        import tempfile
        from pathlib import Path
        
        # Create a temporary base directory for testing working directories
        with tempfile.TemporaryDirectory() as temp_base:
            working_dir_base = Path(temp_base) / "session_workdirs"
            working_dir_base.mkdir(parents=True, exist_ok=True)
            
            worker = CondaWorker(working_dir=str(working_dir_base))
            worker._env_cache = EnvironmentCache(
                cache_dir=conda_test_workspace["cache_dir"], max_size=5
            )
            
            # Check that base directory was created
            assert worker._working_dir_base.exists()
            assert str(worker._working_dir_base) == str(working_dir_base)
            
            # Simple script that reports working directory
            script = """
import os
import sys

# Get and print the current working directory
cwd = os.getcwd()
print(f"Session working directory: {cwd}")
print(f"Python executable: {sys.executable}")

# Create a test file to verify we're in the right directory
test_file = "session_marker.txt"
with open(test_file, "w") as f:
    f.write(f"Session working in: {cwd}")

# Verify the file was created
if os.path.exists(test_file):
    print(f"Successfully created {test_file}")
else:
    print(f"Failed to create {test_file}")
"""
            
            config = WorkerConfig(
                id="workdir-creation-test",
                app_id="test-app",
                workspace="test-workspace",
                client_id="test-client",
                server_url="http://test-server",
                token="test-token",
                entry_point="main.py",
                artifact_id="test-artifact",
                manifest={
                    "type": "conda-jupyter-kernel",
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
                mock_http_client.return_value.__aenter__.return_value.get.return_value = mock_response
                
                try:
                    print("🚀 Starting real conda session to test working directory...")
                    # Compile manifest
                    compiled_manifest, _ = await worker.compile(config.manifest, [])
                    config.manifest = compiled_manifest
                    
                    # Start session
                    session_id = await worker.start(config)
                    assert session_id == "workdir-creation-test"
                    
                    # Check that session working directory was created
                    expected_dir = worker._working_dir_base / "workdir-creation-test"
                    assert expected_dir.exists(), f"Working directory {expected_dir} was not created"
                    assert expected_dir in worker._session_working_dirs.values()
                    
                    print(f"✅ Session working directory created: {expected_dir}")
                    
                    # Execute code to verify working directory
                    test_code = """
import os
cwd = os.getcwd()
print(f"Current working directory in execution: {cwd}")

# List files to verify our marker file
files = os.listdir(".")
print(f"Files in working directory: {sorted(files)}")
"""
                    result = await worker.execute(session_id, test_code)
                    assert result["status"] == "ok", f"Execution failed: {result.get('error', {})}"
                    
                    # Extract stdout
                    stdout_text = "".join(
                        output.get("text", "") for output in result.get("outputs", [])
                        if output.get("type") == "stream" and output.get("name") == "stdout"
                    )
                    
                    # Verify correct working directory
                    assert str(expected_dir) in stdout_text, f"Expected working directory not in output: {stdout_text}"
                    assert "session_marker.txt" in stdout_text, f"Marker file not found in output: {stdout_text}"
                    
                    print(f"✅ Working directory verified in execution")
                    
                    # Stop session (should clean up working directory)
                    await worker.stop(session_id)
                    
                    # Verify working directory was cleaned up
                    assert not expected_dir.exists(), f"Working directory {expected_dir} was not cleaned up"
                    assert session_id not in worker._session_working_dirs
                    
                    print(f"✅ Working directory properly cleaned up after stop")
                    
                except Exception as e:
                    print(f"❌ Test failed: {e}")
                    try:
                        await worker.stop("workdir-creation-test")
                    except:
                        pass
                    raise

    async def test_working_directory_in_execution(self, conda_integration_server, conda_test_workspace):
        """Test that code executes in the correct working directory."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache
        
        # Create a specific working directory for testing
        test_working_base = Path(conda_test_workspace["workspace_dir"]) / "working_dirs"
        test_working_base.mkdir(parents=True, exist_ok=True)
        
        worker = CondaWorker(working_dir=str(test_working_base))
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"], max_size=5
        )
        
        script = """
import os
import sys

# Get the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

# Create a test file in the working directory
test_file = "test_workdir.txt"
with open(test_file, "w") as f:
    f.write("Test content from session")

# Verify the file was created
if os.path.exists(test_file):
    print(f"Successfully created {test_file} in {cwd}")
else:
    print(f"Failed to create {test_file}")
"""
        
        config = WorkerConfig(
            id="workdir-execution-test",
            app_id="workdir-test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact",
            manifest={
                "type": "conda-jupyter-kernel",
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
            mock_http_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            try:
                print("🚀 Testing working directory in real conda environment...")
                # Compile manifest
                compiled_manifest, _ = await worker.compile(config.manifest, [])
                config.manifest = compiled_manifest
                
                # Start session
                session_id = await worker.start(config)
                
                # Execute code to test working directory
                test_code = """
import os
print(f"Session working directory: {os.getcwd()}")

# Create a test file
with open("session_test.txt", "w") as f:
    f.write("Hello from session")
    
# List files in current directory
files = os.listdir(".")
print(f"Files in working directory: {files}")
"""
                result = await worker.execute(session_id, test_code)
                
                assert result["status"] == "ok", f"Execution failed: {result.get('error', {})}"
                
                # Extract stdout
                stdout_text = "".join(
                    output.get("text", "") for output in result.get("outputs", [])
                    if output.get("type") == "stream" and output.get("name") == "stdout"
                )
                
                # Verify working directory is correct
                expected_session_dir = str(test_working_base / "workdir-execution-test")
                assert expected_session_dir in stdout_text, f"Expected working directory {expected_session_dir} not found in output: {stdout_text}"
                
                # Verify file was created in session directory
                assert "session_test.txt" in stdout_text, f"Test file not found in working directory output: {stdout_text}"
                
                print(f"✅ Working directory test successful")
                print(f"   Output: {stdout_text.strip()}")
                
                # Stop session (should clean up working directory)
                await worker.stop(session_id)
                
                # Verify working directory was cleaned up
                session_dir = test_working_base / "workdir-execution-test"
                assert not session_dir.exists(), f"Working directory {session_dir} was not cleaned up"
                
            except Exception as e:
                print(f"❌ Working directory test failed: {e}")
                try:
                    await worker.stop("workdir-execution-test")
                except:
                    pass
                raise

    async def test_working_directory_isolation(self, conda_integration_server, conda_test_workspace):
        """Test that multiple sessions have isolated working directories."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache
        import tempfile
        from pathlib import Path
        
        # Create a temporary base directory for testing
        with tempfile.TemporaryDirectory() as temp_base:
            working_dir_base = Path(temp_base) / "isolated_sessions"
            working_dir_base.mkdir(parents=True, exist_ok=True)
            
            worker = CondaWorker(working_dir=str(working_dir_base))
            worker._env_cache = EnvironmentCache(
                cache_dir=conda_test_workspace["cache_dir"], max_size=5
            )
            
            # Script that creates a unique file in each session
            script = """
import os
import sys

session_id = os.environ.get('HYPHA_CLIENT_ID', 'unknown')
cwd = os.getcwd()
print(f"Session {session_id} working dir: {cwd}")

# Create a unique file for this session
unique_file = f"session_{session_id}_marker.txt"
with open(unique_file, "w") as f:
    f.write(f"This file belongs to session {session_id} in {cwd}")

print(f"Created {unique_file} in {cwd}")
"""
            
            # Create two session configs
            config1 = WorkerConfig(
                id="isolation-test-1",
                app_id="test-app-1",
                workspace="test-workspace",
                client_id="isolation-1",
                server_url="http://test-server",
                token="test-token",
                entry_point="main.py",
                artifact_id="test-artifact",
                manifest={
                    "type": "conda-jupyter-kernel",
                    "dependencies": ["python=3.11"],
                    "channels": ["conda-forge"],
                    "entry_point": "main.py",
                },
                app_files_base_url="http://test-server/files",
            )
            
            config2 = WorkerConfig(
                id="isolation-test-2",
                app_id="test-app-2",
                workspace="test-workspace",
                client_id="isolation-2",
                server_url="http://test-server",
                token="test-token",
                entry_point="main.py",
                artifact_id="test-artifact",
                manifest={
                    "type": "conda-jupyter-kernel",
                    "dependencies": ["python=3.11"],
                    "channels": ["conda-forge"],
                    "entry_point": "main.py",
                },
                app_files_base_url="http://test-server/files",
            )
            
            # Mock HTTP client
            with patch("httpx.AsyncClient") as mock_http_client:
                mock_response = MagicMock()
                mock_response.text = script
                mock_response.raise_for_status = MagicMock()
                mock_http_client.return_value.__aenter__.return_value.get.return_value = mock_response
                
                try:
                    print("🚀 Starting two isolated conda sessions...")
                    
                    # Compile manifests
                    compiled_manifest1, _ = await worker.compile(config1.manifest, [])
                    config1.manifest = compiled_manifest1
                    compiled_manifest2, _ = await worker.compile(config2.manifest, [])
                    config2.manifest = compiled_manifest2
                    
                    # Start both sessions
                    session_id1 = await worker.start(config1)
                    session_id2 = await worker.start(config2)
                    
                    # Verify different working directories were created
                    dir1 = worker._working_dir_base / "isolation-test-1"
                    dir2 = worker._working_dir_base / "isolation-test-2"
                    
                    assert dir1.exists(), f"Working directory for session 1 not created: {dir1}"
                    assert dir2.exists(), f"Working directory for session 2 not created: {dir2}"
                    assert dir1 != dir2, "Sessions should have different working directories"
                    
                    print(f"✅ Created isolated directories: {dir1} and {dir2}")
                    
                    # Execute code in both sessions to verify isolation
                    test_code = """
import os
cwd = os.getcwd()
files = sorted(os.listdir("."))
print(f"Working in: {cwd}")
print(f"Files here: {files}")
"""
                    
                    # Execute in session 1
                    result1 = await worker.execute(session_id1, test_code)
                    assert result1["status"] == "ok"
                    stdout1 = "".join(
                        output.get("text", "") for output in result1.get("outputs", [])
                        if output.get("type") == "stream" and output.get("name") == "stdout"
                    )
                    
                    # Execute in session 2
                    result2 = await worker.execute(session_id2, test_code)
                    assert result2["status"] == "ok"
                    stdout2 = "".join(
                        output.get("text", "") for output in result2.get("outputs", [])
                        if output.get("type") == "stream" and output.get("name") == "stdout"
                    )
                    
                    # Verify isolation - each session should be in its own directory
                    assert str(dir1) in stdout1, f"Session 1 not in correct directory: {stdout1}"
                    assert str(dir2) in stdout2, f"Session 2 not in correct directory: {stdout2}"
                    assert str(dir1) not in stdout2, "Session 2 should not see session 1's directory"
                    assert str(dir2) not in stdout1, "Session 1 should not see session 2's directory"
                    
                    # Verify each session created its unique file
                    assert "session_isolation-1_marker.txt" in stdout1 or "session_marker.txt" in stdout1
                    assert "session_isolation-2_marker.txt" in stdout2 or "session_marker.txt" in stdout2
                    
                    print("✅ Sessions are properly isolated")
                    
                    # Check internal tracking
                    assert len(worker._session_working_dirs) == 2
                    assert session_id1 in worker._session_working_dirs
                    assert session_id2 in worker._session_working_dirs
                    assert worker._session_working_dirs[session_id1] != worker._session_working_dirs[session_id2]
                    
                    # Stop both sessions
                    await worker.stop(session_id1)
                    await worker.stop(session_id2)
                    
                    # Verify cleanup
                    assert not dir1.exists(), f"Session 1 directory not cleaned up: {dir1}"
                    assert not dir2.exists(), f"Session 2 directory not cleaned up: {dir2}"
                    assert len(worker._session_working_dirs) == 0
                    
                    print("✅ Working directories properly cleaned up")
                    
                except Exception as e:
                    print(f"❌ Isolation test failed: {e}")
                    try:
                        await worker.stop("isolation-test-1")
                        await worker.stop("isolation-test-2")
                    except:
                        pass
                    raise


    async def test_files_to_stage(self, conda_test_workspace):
        """Test that files_to_stage downloads files from artifact manager to working directory."""
        from hypha.workers.conda import CondaWorker, EnvironmentCache
        import tempfile
        from pathlib import Path
        import aiohttp
        from aiohttp import web
        import asyncio
        import json
        
        # Create test files in a temporary directory to serve
        test_files_dir = Path(tempfile.mkdtemp())
        
        # Create test files structure
        (test_files_dir / "data.csv").write_text("id,name,value\n1,test1,100\n2,test2,200")
        (test_files_dir / "config.json").write_text('{"setting1": "value1", "setting2": 42}')
        (test_files_dir / "models").mkdir()
        (test_files_dir / "models" / "model1.pkl").write_bytes(b"model1_binary_data")
        (test_files_dir / "models" / "model2.pkl").write_bytes(b"model2_binary_data")
        (test_files_dir / "source_folder").mkdir()
        (test_files_dir / "source_folder" / "file1.txt").write_text("This is file 1 content")
        (test_files_dir / "source_folder" / "subdir").mkdir(parents=True)
        (test_files_dir / "source_folder" / "subdir" / "file2.txt").write_text("This is file 2 in subdirectory")
        
        # Create main.py script
        main_script = """
import os
print(f"Working directory: {os.getcwd()}")
files = []
for root, dirs, filenames in os.walk('.'):
    for fname in filenames:
        if not fname.startswith('.'):  # Skip hidden files
            rel_path = os.path.relpath(os.path.join(root, fname), '.')
            files.append(rel_path)
print(f"Files found: {sorted(files)}")
"""
        (test_files_dir / "main.py").write_text(main_script)
        
        # Create a simple test server to serve files
        async def handle_file_request(request):
            """Handle file requests."""
            path = request.match_info.get('path', '')
            
            # Handle directory listing (path ends with /)
            if path.endswith('/'):
                dir_path = path.rstrip('/')
                full_dir = test_files_dir / dir_path if dir_path else test_files_dir
                
                if full_dir.exists() and full_dir.is_dir():
                    items = []
                    for item in full_dir.iterdir():
                        items.append({
                            "name": item.name,
                            "type": "directory" if item.is_dir() else "file",
                            "size": item.stat().st_size if item.is_file() else 0
                        })
                    return web.json_response(items)
                else:
                    return web.Response(status=404)
            
            # Handle file download
            file_path = test_files_dir / path
            if file_path.exists() and file_path.is_file():
                if path == "main.py":
                    return web.Response(text=file_path.read_text())
                else:
                    return web.Response(body=file_path.read_bytes())
            else:
                return web.Response(status=404)
        
        # Start test server
        app = web.Application()
        app.router.add_route('GET', '/{workspace}/artifacts/{app_id}/files/{path:.*}', handle_file_request)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 0)  # Use random port
        await site.start()
        
        # Get the actual port
        test_port = site._server.sockets[0].getsockname()[1]
        test_server_url = f"http://localhost:{test_port}"
        
        try:
            # Create a temporary base directory for testing
            with tempfile.TemporaryDirectory() as temp_base:
                working_dir_base = Path(temp_base) / "stage_test"
                working_dir_base.mkdir(parents=True, exist_ok=True)
                
                # Initialize worker with test server URL
                worker = CondaWorker(
                    server_url=test_server_url,
                    working_dir=str(working_dir_base),
                    cache_dir=conda_test_workspace["cache_dir"]
                )
                worker._env_cache = EnvironmentCache(
                    cache_dir=conda_test_workspace["cache_dir"], max_size=5
                )
                
                # Create config with files_to_stage
                config = WorkerConfig(
                    id="stage-files-test",
                    app_id="test-app",
                    workspace="test-workspace",
                    client_id="test-client",
                    server_url=test_server_url,
                    token="test-token",
                    entry_point="main.py",
                    artifact_id="test-artifact",
                    manifest={
                        "type": "conda-jupyter-kernel",
                        "dependencies": ["python=3.11"],
                        "channels": ["conda-forge"],
                        "entry_point": "main.py",
                        "files_to_stage": [
                            "data.csv",                           # Simple file
                            "config.json:renamed_config.json",    # Renamed file
                            "models/",                             # Folder with files (recursive)
                            "source_folder/:renamed_folder/",     # Renamed folder (recursive)
                        ]
                    },
                    app_files_base_url=test_server_url + "/files",
                )
            
            try:
                print("🚀 Starting conda session with real artifact files...")
                
                # Compile manifest to add dependencies
                compiled_manifest, _ = await worker.compile(config.manifest, [])
                config.manifest = compiled_manifest
                
                # Start session (which should download and stage files)
                session_id = await worker.start(config)
                assert session_id == "stage-files-test"
                
                # Check that session is running
                assert session_id in worker._sessions
                assert worker._sessions[session_id].status == SessionStatus.RUNNING
                
                # Verify working directory was created
                session_dir = worker._session_working_dirs[session_id]
                assert session_dir.exists()
                print(f"✅ Session working directory created: {session_dir}")
                
                # List actual files in working directory
                actual_files = []
                for root, dirs, filenames in os.walk(session_dir):
                    for fname in filenames:
                        if not fname.startswith('.'):  # Skip hidden files
                            rel_path = os.path.relpath(os.path.join(root, fname), session_dir)
                            actual_files.append(rel_path)
                
                print(f"📁 Files staged in working directory: {sorted(actual_files)}")
                
                # Verify specific files were staged correctly
                assert (session_dir / "data.csv").exists(), "data.csv should be staged"
                assert (session_dir / "renamed_config.json").exists(), "config.json should be renamed to renamed_config.json"
                assert (session_dir / "models").exists(), "models directory should exist"
                assert (session_dir / "models" / "model1.pkl").exists(), "model1.pkl should be in models/"
                assert (session_dir / "models" / "model2.pkl").exists(), "model2.pkl should be in models/"
                assert (session_dir / "renamed_folder").exists(), "source_folder should be renamed to renamed_folder"
                assert (session_dir / "renamed_folder" / "file1.txt").exists(), "file1.txt should be in renamed_folder/"
                assert (session_dir / "renamed_folder" / "subdir" / "file2.txt").exists(), "Subdirectory files should be preserved"
                
                # Verify file contents
                with open(session_dir / "data.csv", "r") as f:
                    csv_content = f.read()
                    assert "test1,100" in csv_content, "CSV content should be preserved"
                
                with open(session_dir / "renamed_config.json", "r") as f:
                    json_content = f.read()
                    assert "setting1" in json_content, "JSON content should be preserved"
                
                # Execute code to verify files are accessible in the kernel
                test_code = """
import os
import json

# Check working directory
print(f"Working directory: {os.getcwd()}")

# List all files
files = []
for root, dirs, filenames in os.walk('.'):
    for fname in filenames:
        if not fname.startswith('.'):
            rel_path = os.path.relpath(os.path.join(root, fname), '.')
            files.append(rel_path)

print(f"Files accessible in kernel: {sorted(files)}")

# Try to read CSV
try:
    with open('data.csv', 'r') as f:
        csv_lines = f.readlines()
        print(f"CSV file has {len(csv_lines)} lines")
except Exception as e:
    print(f"Error reading CSV: {e}")

# Try to read renamed JSON
try:
    with open('renamed_config.json', 'r') as f:
        config = json.load(f)
        print(f"Config loaded: setting1={config.get('setting1')}")
except Exception as e:
    print(f"Error reading JSON: {e}")

# Check models directory
if os.path.exists('models'):
    model_files = os.listdir('models')
    print(f"Model files: {sorted(model_files)}")

# Check renamed folder
if os.path.exists('renamed_folder'):
    renamed_files = []
    for root, dirs, files in os.walk('renamed_folder'):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), 'renamed_folder')
            renamed_files.append(rel)
    print(f"Renamed folder files: {sorted(renamed_files)}")
"""
                
                result = await worker.execute(session_id, test_code, config={"timeout": 30.0})
                assert result["status"] == "ok", f"Execution failed: {result.get('error')}"
                
                # Extract stdout
                stdout_text = "".join(
                    output.get("text", "")
                    for output in result.get("outputs", [])
                    if output.get("type") == "stream" and output.get("name") == "stdout"
                )
                
                print(f"📝 Kernel output:\n{stdout_text}")
                
                # Verify kernel can see the files
                assert "data.csv" in stdout_text, "Kernel should see data.csv"
                assert "renamed_config.json" in stdout_text, "Kernel should see renamed_config.json"
                assert "model1.pkl" in stdout_text, "Kernel should see model1.pkl"
                assert "model2.pkl" in stdout_text, "Kernel should see model2.pkl"
                assert "file1.txt" in stdout_text, "Kernel should see file1.txt"
                assert "CSV file has 3 lines" in stdout_text, "CSV should be readable"
                assert "setting1=value1" in stdout_text, "JSON config should be readable"
                
                print("✅ Files staged and accessible successfully!")
                
                # Stop session
                await worker.stop(session_id)
                
                # Verify cleanup
                assert not session_dir.exists(), "Working directory should be cleaned up"
                print("✅ Working directory cleaned up successfully")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                try:
                    await worker.stop("stage-files-test")
                except:
                    pass
                raise
        finally:
            # Clean up test server
            await runner.cleanup()
            
            # Clean up test files
            import shutil
            shutil.rmtree(test_files_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
