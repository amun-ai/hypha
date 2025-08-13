"""Test Jupyter kernel functionality for conda worker."""

import asyncio
import os
import tempfile
import pytest
import pytest_asyncio
from pathlib import Path

from hypha.workers.conda_kernel import CondaKernel


@pytest_asyncio.fixture
async def test_env_path():
    """Create a test conda environment with ipykernel installed."""
    # Use the active conda environment in CI
    if os.environ.get("CI") == "true":
        # In CI, we have a conda environment set up with setup-miniconda
        # Get the environment path from the current Python executable
        import sys
        current_python = Path(sys.executable)
        # If python lives in {env}/bin/python* then env root is parent of bin
        if current_python.parent.name == "bin":
            env_path = current_python.parent.parent
        else:
            # Otherwise treat the parent directory as the environment path
            env_path = current_python.parent
        yield env_path
    else:
        # For local testing, try to create a minimal conda env or use current environment
        try:
            # Try to use current Python environment as fallback
            import sys
            current_python = Path(sys.executable)
            # If python lives in {env}/bin/python* then env root is parent of bin
            if current_python.parent.name == "bin":
                env_path = current_python.parent.parent
            else:
                env_path = current_python.parent
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
        
        # Get connection info
        connection_info = kernel.get_connection_info()
        assert connection_info is not None
        
    finally:
        # Clean up
        await kernel.stop()


@pytest.mark.asyncio
async def test_conda_kernel_execute_with_result(test_env_path):
    """Test code execution with results."""
    kernel = CondaKernel(test_env_path)
    
    try:
        await kernel.start(timeout=30.0)
        
        # Test simple execution
        result = await kernel.execute("print('Hello, World!')", timeout=10.0)
        
        assert result["success"] == True
        assert len(result["outputs"]) > 0
        
        # Check for stdout output
        stdout_found = False
        for output in result["outputs"]:
            if output["type"] == "stream" and output["name"] == "stdout":
                assert "Hello, World!" in output["text"]
                stdout_found = True
                break
        assert stdout_found, "Expected stdout output not found"
        
    finally:
        await kernel.stop()


@pytest.mark.asyncio
async def test_conda_kernel_error_handling(test_env_path):
    """Test error handling in code execution."""
    kernel = CondaKernel(test_env_path)
    
    try:
        await kernel.start(timeout=30.0)
        
        # Test error execution
        result = await kernel.execute("raise ValueError('Test error')", timeout=10.0)
        
        assert result["success"] == False
        assert "error" in result
        assert result["error"]["ename"] == "ValueError"
        assert "Test error" in result["error"]["evalue"]
        
    finally:
        await kernel.stop()


@pytest.mark.asyncio
async def test_conda_kernel_interrupt(test_env_path):
    """Test kernel interrupt functionality."""
    kernel = CondaKernel(test_env_path)
    
    try:
        await kernel.start(timeout=30.0)
        
        # Test interrupt (basic test - just verify it doesn't crash)
        await kernel.interrupt()
        
        # Verify kernel is still alive after interrupt
        assert await kernel.is_alive()
        
    finally:
        await kernel.stop()


@pytest.mark.asyncio
async def test_conda_kernel_restart(test_env_path):
    """Test kernel restart functionality."""
    kernel = CondaKernel(test_env_path)
    
    try:
        await kernel.start(timeout=30.0)
        
        # Execute some code to establish state
        result = await kernel.execute("x = 42", timeout=10.0)
        assert result["success"] == True
        
        # Restart the kernel
        await kernel.restart()
        
        # Verify kernel is still alive
        assert await kernel.is_alive()
        
        # Verify state was reset (x should not exist)
        result = await kernel.execute("print(x)", timeout=10.0)
        assert result["success"] == False
        assert "NameError" in result["error"]["ename"]
        
    finally:
        await kernel.stop()