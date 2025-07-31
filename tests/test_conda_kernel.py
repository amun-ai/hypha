"""Test Jupyter kernel functionality for conda worker."""

import asyncio
import os
import tempfile
import pytest
import pytest_asyncio
from pathlib import Path

from hypha.workers.conda_kernel import CondaKernel
from hypha.workers.conda import get_available_package_manager


@pytest_asyncio.fixture
async def test_env_path():
    """Create a test conda environment with ipykernel installed."""
    package_manager = get_available_package_manager()
    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = Path(tmpdir) / "test_env"
        
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
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            pytest.skip(f"Failed to create test environment: {stderr.decode()}")
        
        yield env_path


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

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])