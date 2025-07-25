"""Tests for conda environment worker."""

import asyncio
import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypha.workers.conda_env import CondaEnvWorker, EnvironmentCache
from hypha.workers.base import WorkerConfig, SessionStatus, SessionInfo, SessionNotFoundError, WorkerError
from hypha.workers.conda_env_executor import ExecutionResult, TimingInfo

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
        packages1 = ["python=3.11", "numpy", "pandas"]
        channels1 = ["conda-forge", "defaults"]
        
        packages2 = ["pandas", "numpy", "python=3.11"]  # Different order
        channels2 = ["defaults", "conda-forge"]  # Different order
        
        # Same packages and channels should produce same hash regardless of order
        hash1 = self.cache._compute_env_hash(packages1, channels1)
        hash2 = self.cache._compute_env_hash(packages2, channels2)
        assert hash1 == hash2
        
        # Different packages should produce different hash
        packages3 = ["python=3.10", "numpy", "pandas"]
        hash3 = self.cache._compute_env_hash(packages3, channels1)
        assert hash1 != hash3
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        packages = ["python=3.11", "numpy"]
        channels = ["conda-forge"]
        env_path = Path(self.temp_dir) / "test_env"
        env_path.mkdir()
        (env_path / "bin").mkdir()
        (env_path / "bin" / "python").touch()
        
        # Initially no cached environment
        cached = self.cache.get_cached_env(packages, channels)
        assert cached is None
        
        # Add to cache
        self.cache.add_cached_env(packages, channels, env_path)
        
        # Should now be in cache
        cached = self.cache.get_cached_env(packages, channels)
        assert cached == env_path
        
        # Check cache index was updated
        assert len(self.cache.index) == 1
        env_hash = self.cache._compute_env_hash(packages, channels)
        assert env_hash in self.cache.index
        assert self.cache.index[env_hash]['path'] == str(env_path)
    
    def test_cache_validation(self):
        """Test cache entry validation."""
        packages = ["python=3.11"]
        channels = ["conda-forge"] 
        env_path = Path(self.temp_dir) / "invalid_env"
        
        # Add invalid environment (doesn't exist)
        self.cache.add_cached_env(packages, channels, env_path)
        
        # Should not return invalid environment and should clean it up
        cached = self.cache.get_cached_env(packages, channels)
        assert cached is None
        assert len(self.cache.index) == 0
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Fill cache to capacity
        for i in range(3):
            packages = [f"python=3.{i+9}"]
            channels = ["conda-forge"]
            env_path = Path(self.temp_dir) / f"env_{i}"
            env_path.mkdir()
            (env_path / "bin").mkdir()
            (env_path / "bin" / "python").touch()
            
            self.cache.add_cached_env(packages, channels, env_path)
        
        assert len(self.cache.index) == 3
        
        # Access first environment to make it more recently used
        first_packages = ["python=3.9"]
        first_channels = ["conda-forge"]
        cached = self.cache.get_cached_env(first_packages, first_channels)
        assert cached is not None
        
        # Add one more environment (should evict least recently used)
        new_packages = ["python=3.12"]
        new_channels = ["conda-forge"]
        new_env_path = Path(self.temp_dir) / "env_new"
        new_env_path.mkdir()
        (new_env_path / "bin").mkdir()
        (new_env_path / "bin" / "python").touch()
        
        self.cache.add_cached_env(new_packages, new_channels, new_env_path)
        
        # Should still have 3 entries (max size)
        assert len(self.cache.index) == 3
        
        # First environment should still be there (was accessed recently)
        cached = self.cache.get_cached_env(first_packages, first_channels)
        assert cached is not None
        
        # New environment should be there
        cached = self.cache.get_cached_env(new_packages, new_channels)
        assert cached is not None
    
    def test_age_based_eviction(self):
        """Test age-based cache eviction."""
        packages = ["python=3.11"]
        channels = ["conda-forge"]
        env_path = Path(self.temp_dir) / "old_env"
        env_path.mkdir()
        (env_path / "bin").mkdir()
        (env_path / "bin" / "python").touch()
        
        # Add environment with old timestamp
        self.cache.add_cached_env(packages, channels, env_path)
        env_hash = self.cache._compute_env_hash(packages, channels)
        
        # Manually set old creation time (35 days ago)
        old_time = time.time() - (35 * 24 * 60 * 60)
        self.cache.index[env_hash]['created_at'] = old_time
        self.cache._save_index()
        
        # Trigger eviction by accessing cache
        self.cache._evict_if_needed()
        
        # Old environment should be removed
        assert len(self.cache.index) == 0
        cached = self.cache.get_cached_env(packages, channels)
        assert cached is None


class TestCondaEnvWorkerBasic:
    """Test basic conda environment worker functionality without mocking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.server = MagicMock()
        self.worker = CondaEnvWorker(self.server)
    
    def test_supported_types(self):
        """Test supported application types."""
        types = self.worker.supported_types
        assert "python-conda" in types
        assert len(types) == 1
    
    def test_worker_properties(self):
        """Test worker properties."""
        assert "Conda Environment Worker" in self.worker.worker_name
        assert "conda environments" in self.worker.worker_description
    
    async def test_compile_manifest(self):
        """Test manifest compilation and validation."""
        # Test with packages field
        manifest1 = {
            "type": "python-conda",
            "packages": ["python=3.11", "numpy"],
            "channels": ["conda-forge"]
        }
        compiled_manifest, files = await self.worker.compile(manifest1, [])
        assert compiled_manifest["packages"] == ["python=3.11", "numpy"]
        assert compiled_manifest["channels"] == ["conda-forge"]
        
        # Test with dependencies field (alternate name)
        manifest2 = {
            "type": "python-conda", 
            "dependencies": "python=3.11",  # String should be converted to list
            "channels": "conda-forge"  # String should be converted to list
        }
        compiled_manifest, files = await self.worker.compile(manifest2, [])
        assert compiled_manifest["packages"] == ["python=3.11"]
        assert compiled_manifest["channels"] == ["conda-forge"]
        
        # Test with no packages (should add default)
        manifest3 = {"type": "python-conda"}
        compiled_manifest, files = await self.worker.compile(manifest3, [])
        assert compiled_manifest["packages"] == []
        assert compiled_manifest["channels"] == ["conda-forge"]
    
    def test_get_service_config(self):
        """Test service configuration."""
        service_config = self.worker.get_service()
        
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


class TestCondaEnvWorkerIntegration:
    """Integration tests for conda environment worker using real conda environments."""
    
    async def test_real_conda_basic_execution(self, conda_integration_server, conda_test_workspace):
        """Test basic conda environment creation and code execution."""
        from hypha.workers.conda_env import CondaEnvWorker, EnvironmentCache
        
        # Initialize worker with clean cache
        worker = CondaEnvWorker(conda_integration_server)
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"],
            max_size=5
        )
        
        # Simple script with execute function
        script = '''
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
'''
        
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
                "packages": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py"
            },
            app_files_base_url="http://test-server/files"
        )
        
        # Mock HTTP client for script fetching
        with patch('httpx.AsyncClient') as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            try:
                print("🚀 Starting real conda environment session...")
                session_id = await worker.start(config)
                assert session_id == "real-conda-test"
                
                # Verify session was created
                session_info = await worker.get_session_info(session_id)
                assert session_info.status == SessionStatus.RUNNING
                
                print("⚙️ Executing code in real conda environment...")
                # Execute with input data
                result = await worker.execute_code(session_id, 21)
                
                assert result.success, f"Execution failed: {result.error}\nStderr: {result.stderr}"
                assert isinstance(result.result, dict)
                assert "python_version" in result.result
                assert result.result["computation"] == 42  # 21 * 2
                assert result.result["input_data"] == 21
                
                print(f"✅ Execution successful: {result.result}")
                
                # Test without input data
                result2 = await worker.execute_code(session_id, None)
                assert result2.success
                assert result2.result["computation"] == 45  # sum(range(10))
                
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
    
    async def test_real_conda_package_installation(self, conda_integration_server, conda_test_workspace):
        """Test conda environment with additional packages."""
        from hypha.workers.conda_env import CondaEnvWorker, EnvironmentCache
        
        worker = CondaEnvWorker(conda_integration_server)
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"],
            max_size=5
        )
        
        # Script that uses numpy (needs to be installed)
        script = '''
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
'''
        
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
                "packages": ["python=3.11", "numpy"],
                "channels": ["conda-forge"],
                "entry_point": "main.py"
            },
            app_files_base_url="http://test-server/files"
        )
        
        with patch('httpx.AsyncClient') as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            try:
                print("🚀 Creating conda environment with numpy...")
                session_id = await worker.start(config)
                
                print("⚙️ Testing numpy functionality...")
                result = await worker.execute_code(session_id, [10, 20, 30, 40, 50])
                
                assert result.success, f"Numpy test failed: {result.error}\nStderr: {result.stderr}"
                assert "numpy_version" in result.result
                assert result.result["array_sum"] == 15  # 1+2+3+4+5
                assert result.result["array_mean"] == 3.0  # (1+2+3+4+5)/5
                assert result.result["input_sum"] == 150  # 10+20+30+40+50
                assert result.result["input_mean"] == 30.0
                
                print(f"✅ NumPy test successful: {result.result}")
                
                await worker.stop(session_id)
                
            except Exception as e:
                print(f"❌ NumPy integration test failed: {e}")
                try:
                    await worker.stop("numpy-conda-test")
                except:
                    pass
                raise
    
    async def test_real_conda_caching_behavior(self, conda_integration_server, conda_test_workspace):
        """Test that conda environments are properly cached and reused."""
        from hypha.workers.conda_env import CondaEnvWorker, EnvironmentCache
        
        worker = CondaEnvWorker(conda_integration_server)
        cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"],
            max_size=5
        )
        worker._env_cache = cache
        
        script = '''
def execute(input_data):
    import os
    import sys
    return {
        "python_executable": sys.executable,
        "environment_variables": dict(os.environ).get("CONDA_DEFAULT_ENV", "none"),
        "input": input_data
    }
'''
        
        # First session with specific packages
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
                "packages": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py"
            },
            app_files_base_url="http://test-server/files"
        )
        
        with patch('httpx.AsyncClient') as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            try:
                print("🚀 Creating first conda environment...")
                # Check cache is initially empty
                initial_cache_size = len(cache.index)
                
                # Start first session
                session_id1 = await worker.start(config1)
                result1 = await worker.execute_code(session_id1, "first")
                assert result1.success
                
                # Check cache was populated
                first_cache_size = len(cache.index)
                assert first_cache_size == initial_cache_size + 1
                
                # Get the environment path for comparison
                env_path1 = result1.result["python_executable"]
                
                await worker.stop(session_id1)
                
                print("🔄 Creating second session with same packages...")
                # Start second session with same packages (should use cache)
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
                        "packages": ["python=3.11"],  # Same packages
                        "channels": ["conda-forge"],  # Same channels
                        "entry_point": "main.py"
                    },
                    app_files_base_url="http://test-server/files"
                )
                
                session_id2 = await worker.start(config2)
                result2 = await worker.execute_code(session_id2, "second")
                assert result2.success
                
                # Should have same environment path (reused from cache)
                env_path2 = result2.result["python_executable"]
                
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
    
    async def test_real_conda_mixed_packages(self, conda_integration_server, conda_test_workspace):
        """Test conda environment with both conda and pip packages."""
        from hypha.workers.conda_env import CondaEnvWorker, EnvironmentCache
        
        worker = CondaEnvWorker(conda_integration_server)
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"],
            max_size=5
        )
        
        # Script that uses both conda (numpy) and pip packages
        script = '''
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
'''
        
        config = WorkerConfig(
            id="mixed-packages-test",
            app_id="mixed-test-app",
            workspace="test-workspace",
            client_id="test-client",
            server_url="http://test-server",
            token="test-token",
            entry_point="main.py",
            artifact_id="test-artifact", 
            manifest={
                "type": "python-conda",
                "packages": [
                    "python=3.11",
                    "numpy", 
                    {"pip": ["requests"]}
                ],
                "channels": ["conda-forge"],
                "entry_point": "main.py"
            },
            app_files_base_url="http://test-server/files"
        )
        
        with patch('httpx.AsyncClient') as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            try:
                print("🚀 Creating environment with mixed conda/pip packages...")
                session_id = await worker.start(config)
                
                print("⚙️ Testing mixed package functionality...")
                test_data = [1.5, 2.3, 3.7, 4.1, 5.9]
                result = await worker.execute_code(session_id, test_data)
                
                assert result.success, f"Mixed packages test failed: {result.error}\nStderr: {result.stderr}"
                
                # Verify conda package (numpy) works
                assert result.result["numpy_available"] is True
                assert "numpy_version" in result.result
                
                # Verify pip package (requests) works 
                assert result.result["requests_available"] is True
                assert "requests_version" in result.result
                
                # Verify data processing works
                assert "data_processed" in result.result
                data_processed = result.result["data_processed"]
                expected_sum = sum(test_data)
                expected_mean = sum(test_data) / len(test_data)
                
                assert abs(data_processed["sum"] - expected_sum) < 0.001
                assert abs(data_processed["mean"] - expected_mean) < 0.001
                
                print(f"✅ Mixed packages test successful:")
                print(f"  NumPy version: {result.result['numpy_version']}")
                print(f"  Requests version: {result.result['requests_version']}")
                print(f"  Data processing: {data_processed}")
                
                await worker.stop(session_id)
                
            except Exception as e:
                print(f"❌ Mixed packages test failed: {e}")
                try:
                    await worker.stop("mixed-packages-test")
                except:
                    pass
                raise
    
    async def test_real_conda_standalone_script(self, conda_integration_server, conda_test_workspace):
        """Test conda environment with a standalone script (no execute function)."""
        from hypha.workers.conda_env import CondaEnvWorker, EnvironmentCache
        
        worker = CondaEnvWorker(conda_integration_server)
        worker._env_cache = EnvironmentCache(
            cache_dir=conda_test_workspace["cache_dir"],
            max_size=5
        )
        
        # Standalone script without execute function
        script = '''
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
conda_env = os.environ.get("CONDA_DEFAULT_ENV", "Not set")
print(f"Conda environment: {conda_env}")

print("=== Script completed successfully ===")
'''
        
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
                "packages": ["python=3.11"],
                "channels": ["conda-forge"],
                "entry_point": "main.py"
            },
            app_files_base_url="http://test-server/files"
        )
        
        with patch('httpx.AsyncClient') as mock_http_client:
            mock_response = MagicMock()
            mock_response.text = script
            mock_response.raise_for_status = MagicMock()
            mock_http_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
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
                    assert "Computation result: 4950" in stdout_content  # sum(range(100))
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 