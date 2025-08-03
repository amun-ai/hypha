"""Jupyter kernel management for conda environments."""

import asyncio
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from jupyter_client import AsyncKernelClient, KernelConnectionInfo
from jupyter_client.manager import AsyncKernelManager


class CondaKernel:
    """Manages a Jupyter kernel running in a conda environment."""

    def __init__(self, conda_env_path: Union[str, Path]):
        """Initialize the conda kernel.
        
        Args:
            conda_env_path: Path to the conda environment
        """
        self.conda_env_path = Path(conda_env_path)
        self.python_path = self.conda_env_path / "bin" / "python"
        
        # Verify the environment exists
        if not self.python_path.exists():
            # Try Windows path
            self.python_path = self.conda_env_path / "python.exe"
            if not self.python_path.exists():
                raise ValueError(f"Python not found in conda environment: {conda_env_path}")
        
        self.kernel_process = None
        self.connection_file = None
        self.kernel_manager = None
        self.kernel_client = None
        self.kernel_id = str(uuid.uuid4())

    async def start(self, timeout: float = 30.0):
        """Start the Jupyter kernel."""
        # Create a temporary connection file
        fd, self.connection_file = tempfile.mkstemp(suffix=".json", prefix="kernel_")
        os.close(fd)
        
        # Create kernel manager - use default kernel spec (no kernel_name="" to avoid None kernel_spec)
        self.kernel_manager = AsyncKernelManager(
            connection_file=self.connection_file,
        )
        
        # Override the kernel command to use our conda environment's Python
        # Set this before calling start_kernel
        self.kernel_manager.kernel_cmd = [
            str(self.python_path),
            "-m", "ipykernel_launcher",
            "-f", "{connection_file}"
        ]
        
        # Start the kernel
        await self.kernel_manager.start_kernel()
        
        # Create client
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()
        
        # Wait for kernel to be ready
        try:
            await self.kernel_client.wait_for_ready(timeout=timeout)
        except Exception as e:
            await self.stop()
            raise RuntimeError(f"Kernel failed to start within {timeout}s: {e}")

    async def stop(self):
        """Stop the kernel and clean up resources."""
        if self.kernel_client:
            self.kernel_client.stop_channels()
            self.kernel_client = None
            
        if self.kernel_manager:
            if self.kernel_manager.has_kernel:
                await self.kernel_manager.shutdown_kernel()
            self.kernel_manager = None
            
        if self.connection_file and os.path.exists(self.connection_file):
            try:
                os.unlink(self.connection_file)
            except OSError:
                pass
            self.connection_file = None

    async def execute(
        self,
        code: str,
        timeout: float = 30.0,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute code in the kernel.
        
        Args:
            code: Python code to execute
            timeout: Maximum time to wait for execution
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with execution results
        """
        if not self.kernel_client:
            raise RuntimeError("Kernel not started")
            
        # Send execution request
        msg_id = self.kernel_client.execute(code)
        
        # Collect outputs
        outputs = []
        error = None
        status = "ok"
        
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                status = "timeout"
                error = {"ename": "TimeoutError", "evalue": f"Execution timed out after {timeout}s", "traceback": []}
                break
                
            try:
                # Get message with short timeout to check for timeout
                msg = await asyncio.wait_for(
                    self.kernel_client.get_iopub_msg(timeout=1.0), 
                    timeout=1.0
                )
                
                if msg['parent_header'].get('msg_id') != msg_id:
                    continue
                    
                msg_type = msg['msg_type']
                content = msg['content']
                
                if progress_callback:
                    progress_callback({
                        "type": "message",
                        "message": f"Received {msg_type} message"
                    })
                
                if msg_type == 'stream':
                    outputs.append({
                        "type": "stream",
                        "name": content.get("name", "stdout"),
                        "text": content.get("text", "")
                    })
                elif msg_type == 'display_data':
                    outputs.append({
                        "type": "display_data", 
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {})
                    })
                elif msg_type == 'execute_result':
                    outputs.append({
                        "type": "execute_result",
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                        "execution_count": content.get("execution_count", 0)
                    })
                elif msg_type == 'error':
                    status = "error"
                    error = {
                        "ename": content.get("ename", "Unknown"),
                        "evalue": content.get("evalue", "Unknown error"),
                        "traceback": content.get("traceback", [])
                    }
                elif msg_type == 'execute_reply':
                    # This indicates the end of execution
                    if content.get('status') == 'error' and not error:
                        status = "error"
                        error = {
                            "ename": content.get("ename", "Unknown"), 
                            "evalue": content.get("evalue", "Unknown error"),
                            "traceback": content.get("traceback", [])
                        }
                    break
                    
            except asyncio.TimeoutError:
                # Short timeout hit, continue loop to check overall timeout
                continue
            except Exception as e:
                status = "error"
                error = {"ename": type(e).__name__, "evalue": str(e), "traceback": []}
                break
        
        result = {
            "status": status,
            "outputs": outputs
        }
        
        if error:
            result["error"] = error
            
        return result

    async def interrupt(self):
        """Interrupt the kernel."""
        if self.kernel_manager:
            await self.kernel_manager.interrupt_kernel()

    async def restart(self):
        """Restart the kernel."""
        if self.kernel_manager:
            await self.kernel_manager.restart_kernel()
            if self.kernel_client:
                await self.kernel_client.wait_for_ready(timeout=30.0)

    async def is_alive(self) -> bool:
        """Check if the kernel is alive."""
        if self.kernel_manager is None:
            return False
        try:
            # Handle both sync and async is_alive methods
            result = self.kernel_manager.is_alive()
            if hasattr(result, '__await__'):
                return await result
            return result
        except Exception:
            return False

    def get_connection_info(self) -> Optional[KernelConnectionInfo]:
        """Get kernel connection information."""
        if self.kernel_manager:
            return self.kernel_manager.get_connection_info()
        return None