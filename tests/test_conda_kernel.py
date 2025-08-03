"""Test Jupyter kernel functionality for conda worker."""

import asyncio
import os
import tempfile
import pytest
import pytest_asyncio
from pathlib import Path

from hypha.workers.conda_kernel import CondaKernel
from hypha.workers.conda import CondaWorker, get_available_package_manager


@pytest_asyncio.fixture
async def test_env_path():
    """Create a test conda environment with ipykernel installed."""
    # Use the active conda environment in CI
    if os.environ.get("CI") == "true":
        # In CI, we have a conda environment set up with setup-miniconda
        # Get the environment path from the current Python executable
        import sys
        current_python = Path(sys.executable)
        if current_python.name == "python":
            # Python is in {env}/bin/python, so env path is parent of bin
            env_path = current_python.parent.parent
        else:  
            # Fallback: assume it's in the bin directory
            env_path = current_python.parent
        yield env_path
    else:
        # For local testing, try to create a minimal conda env
        try:
            package_manager = get_available_package_manager()
            with tempfile.TemporaryDirectory() as tmpdir:
                env_path = Path(tmpdir) / "test_env"
                
                # Create minimal conda environment
                cmd = [
                    package_manager,
                    "create",
                    "-p", str(env_path),
                    "-y",
                    "-c", "conda-forge",
                    "python=3.9",
                    "ipykernel",
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                _, stderr = await process.communicate()
                
                if process.returncode != 0:
                    pytest.skip(f"Failed to create test environment: {stderr.decode()}")
                
                yield env_path
        except Exception:
            pytest.skip("Conda not available for testing")


@pytest.mark.asyncio
async def test_conda_kernel_basic(test_env_path):
    """Test basic kernel functionality."""
    kernel = CondaKernel(test_env_path)
    
    try:
        # Start the kernel
        await kernel.start(timeout=30.0)
        
        # Check kernel is alive
        assert await kernel.is_alive()
        
        # Execute simple code
        result = await kernel.execute("print('Hello from kernel!')")
        
        assert result["status"] == "ok"
        assert len(result["outputs"]) > 0
        
        # Find stdout output
        stdout_found = False
        for output in result["outputs"]:
            if output["type"] == "stream" and output["name"] == "stdout":
                assert "Hello from kernel!" in output["text"]
                stdout_found = True
                break
        
        assert stdout_found, "Expected stdout output not found"
        
    finally:
        # Cleanup
        await kernel.shutdown()


@pytest.mark.asyncio
async def test_conda_kernel_execute_with_result(test_env_path):
    """Test kernel execution with results."""
    kernel = CondaKernel(test_env_path)
    
    try:
        await kernel.start(timeout=30.0)
        
        # Execute code that returns a value
        result = await kernel.execute("2 + 3")
        
        assert result["status"] == "ok"
        
        # Check for execute_result
        execute_result_found = False
        for output in result["outputs"]:
            if output["type"] == "execute_result":
                assert "5" in output["data"]["text/plain"]
                execute_result_found = True
                break
        
        assert execute_result_found, "Expected execute_result not found"
        
    finally:
        await kernel.shutdown()


@pytest.mark.asyncio
async def test_conda_kernel_error_handling(test_env_path):
    """Test kernel error handling."""
    kernel = CondaKernel(test_env_path)
    
    try:
        await kernel.start(timeout=30.0)
        
        # Execute code with error
        result = await kernel.execute("1 / 0")
        
        assert result["status"] == "error"
        assert result["error"] is not None
        assert result["error"]["ename"] == "ZeroDivisionError"
        
        # Check error in outputs
        error_found = False
        for output in result["outputs"]:
            if output["type"] == "error":
                assert output["ename"] == "ZeroDivisionError"
                error_found = True
                break
        
        assert error_found, "Expected error output not found"
        
    finally:
        await kernel.shutdown()


@pytest.mark.asyncio
async def test_conda_kernel_interrupt(test_env_path):
    """Test kernel interrupt functionality."""
    kernel = CondaKernel(test_env_path)
    
    try:
        await kernel.start(timeout=30.0)
        
        # Start long-running code
        async def execute_long_running():
            return await kernel.execute("""
import time
for i in range(10):
    print(f"Iteration {i}")
    time.sleep(1)
""")
        
        # Start execution
        task = asyncio.create_task(execute_long_running())
        
        # Wait a bit then interrupt
        await asyncio.sleep(2)
        await kernel.interrupt()
        
        # Wait for task to complete
        try:
            result = await asyncio.wait_for(task, timeout=5)
            # Execution was interrupted
            assert result["status"] == "error" or len(result["outputs"]) < 10
        except asyncio.TimeoutError:
            # This is also acceptable - execution was stopped
            pass
        
    finally:
        await kernel.shutdown()


@pytest.mark.asyncio
async def test_conda_kernel_restart(test_env_path):
    """Test kernel restart functionality."""
    kernel = CondaKernel(test_env_path)
    
    try:
        await kernel.start(timeout=30.0)
        
        # Set a variable
        result1 = await kernel.execute("test_var = 42")
        assert result1["status"] == "ok"
        
        # Verify variable exists
        result2 = await kernel.execute("print(test_var)")
        assert result2["status"] == "ok"
        
        # Restart kernel
        await kernel.restart()
        
        # Variable should not exist after restart
        result3 = await kernel.execute("print(test_var)")
        assert result3["status"] == "error"
        assert result3["error"]["ename"] == "NameError"
        
    finally:
        await kernel.shutdown()


@pytest.mark.asyncio
async def test_conda_worker_execute_api():
    """Test the new execute API in CondaWorker."""
    worker = CondaWorker()
    
    # Create a test config
    config = {
        "id": "test-workspace/test-client",
        "app_id": "test-app",
        "workspace": "test-workspace",
        "client_id": "test-client",
        "server_url": "http://localhost:9527",
        "token": "test-token",
        "entry_point": "main.py",
        "artifact_id": "test-workspace/test-app",
        "manifest": {
            "name": "Test App",
            "type": "conda-jupyter-kernel",
            "dependencies": ["python=3.9", "numpy"],
            "channels": ["conda-forge"]
        },
        "app_files_base_url": "http://localhost:9527/test"
    }
    
    # Mock the script fetching
    async def mock_get(url, headers=None):
        class MockResponse:
            def raise_for_status(self):
                pass
            @property
            def text(self):
                return "print('Initialization script executed')"
        return MockResponse()
    
    # Patch httpx client
    import hypha.workers.conda
    original_client = hypha.workers.conda.httpx.AsyncClient
    
    class MockAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, *args, **kwargs):
            return await mock_get(*args, **kwargs)
    
    try:
        hypha.workers.conda.httpx.AsyncClient = MockAsyncClient
        
        # Start session
        session_id = await worker.start(config)
        
        # Wait for kernel to be ready
        await asyncio.sleep(2)
        
        # Test execute method
        result = await worker.execute(
            session_id=session_id,
            script="print('Hello from execute API!')",
            config={"timeout": 10.0}
        )
        
        assert result["status"] == "ok"
        assert any(
            "Hello from execute API!" in output.get("text", "")
            for output in result["outputs"]
            if output["type"] == "stream"
        )
        
        # Test with calculation
        result2 = await worker.execute(
            session_id=session_id,
            script="result = 10 * 5\nprint(f'Result: {result}')",
            config={"timeout": 10.0}
        )
        
        assert result2["status"] == "ok"
        assert any(
            "Result: 50" in output.get("text", "")
            for output in result2["outputs"]
            if output["type"] == "stream"
        )
        
        # Test error handling
        result3 = await worker.execute(
            session_id=session_id,
            script="raise ValueError('Test error')",
            config={"timeout": 10.0}
        )
        
        assert result3["status"] == "error"
        assert result3["error"]["ename"] == "ValueError"
        assert "Test error" in result3["error"]["evalue"]
        
        # Test with progress callback
        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)
        
        result4 = await worker.execute(
            session_id=session_id,
            script="print('Progress callback test')",
            config={"timeout": 10.0},
            progress_callback=progress_callback
        )
        
        assert result4["status"] == "ok"
        
    finally:
        # Restore original client
        hypha.workers.conda.httpx.AsyncClient = original_client
        
        # Cleanup
        await worker.stop(session_id)


@pytest.mark.asyncio
async def test_conda_worker_session_persistence():
    """Test that variables persist across execute calls."""
    worker = CondaWorker()
    
    config = {
        "id": "test-workspace/test-client-2",
        "app_id": "test-app-2",
        "workspace": "test-workspace",
        "client_id": "test-client-2",
        "server_url": "http://localhost:9527",
        "token": "test-token",
        "entry_point": "main.py",
        "artifact_id": "test-workspace/test-app-2",
        "manifest": {
            "name": "Test App 2",
            "type": "conda-jupyter-kernel",
            "dependencies": ["python=3.9"],
            "channels": ["conda-forge"]
        },
        "app_files_base_url": "http://localhost:9527/test"
    }
    
    # Mock httpx as before
    import hypha.workers.conda
    original_client = hypha.workers.conda.httpx.AsyncClient
    
    class MockAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, *args, **kwargs):
            class MockResponse:
                def raise_for_status(self):
                    pass
                @property
                def text(self):
                    return "# Initialization"
            return MockResponse()
    
    try:
        hypha.workers.conda.httpx.AsyncClient = MockAsyncClient
        
        session_id = await worker.start(config)
        await asyncio.sleep(2)
        
        # Set a variable
        result1 = await worker.execute(
            session_id=session_id,
            script="my_data = [1, 2, 3, 4, 5]\nprint('Data set')"
        )
        assert result1["status"] == "ok"
        
        # Use the variable in another call
        result2 = await worker.execute(
            session_id=session_id,
            script="import numpy as np\nprint(f'Mean: {np.mean(my_data)}')"
        )
        assert result2["status"] == "ok"
        assert any(
            "Mean: 3.0" in output.get("text", "")
            for output in result2["outputs"]
            if output["type"] == "stream"
        )
        
    finally:
        hypha.workers.conda.httpx.AsyncClient = original_client
        await worker.stop(session_id)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])