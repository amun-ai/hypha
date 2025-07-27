"""Jupyter kernel management for conda environments."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        
        # Create kernel manager
        self.kernel_manager = AsyncKernelManager(
            kernel_name='python3',
            connection_file=self.connection_file,
        )
        
        # Set the kernel command to use our conda environment's Python
        self.kernel_manager.kernel_cmd = [
            str(self.python_path),
            "-m", "ipykernel_launcher",
            "-f", "{connection_file}"
        ]
        
        # Start the kernel
        await self.kernel_manager.start_kernel()
        
        # Create kernel client
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()
        
        # Wait for kernel to be ready
        await self.kernel_client.wait_for_ready(timeout=timeout)

    async def execute(
        self, 
        code: str, 
        silent: bool = False,
        store_history: bool = True,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute code in the kernel and return results.
        
        Args:
            code: The code to execute
            silent: If True, don't broadcast output on IOPub channel
            store_history: If True, store command in history
            timeout: Timeout in seconds for execution
            
        Returns:
            Dictionary containing execution results with keys:
            - status: 'ok' or 'error'
            - outputs: List of output dictionaries
            - execution_count: The execution count
            - error: Error information if status is 'error'
        """
        if not self.kernel_client:
            raise RuntimeError("Kernel not started. Call start() first.")
        
        # Send execute request
        msg_id = self.kernel_client.execute(
            code,
            silent=silent,
            store_history=store_history
        )
        
        # Collect outputs
        outputs = []
        execution_count = None
        status = 'ok'
        error_info = None
        
        # Set up timeout
        start_time = time.time()
        
        while True:
            try:
                # Get message with a short timeout to allow checking total timeout
                msg = await asyncio.wait_for(
                    self.kernel_client.get_iopub_msg(timeout=1),
                    timeout=1
                )
            except asyncio.TimeoutError:
                # Check if total timeout exceeded
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Execution timed out after {timeout} seconds")
                continue
            
            msg_type = msg["msg_type"]
            content = msg["content"]
            
            if msg_type == "stream":
                outputs.append({
                    "type": "stream",
                    "name": content.get("name", "stdout"),
                    "text": content.get("text", "")
                })
            elif msg_type == "display_data":
                outputs.append({
                    "type": "display_data",
                    "data": content.get("data", {}),
                    "metadata": content.get("metadata", {})
                })
            elif msg_type == "execute_result":
                outputs.append({
                    "type": "execute_result",
                    "data": content.get("data", {}),
                    "metadata": content.get("metadata", {}),
                    "execution_count": content.get("execution_count")
                })
                execution_count = content.get("execution_count")
            elif msg_type == "error":
                status = 'error'
                error_info = {
                    "ename": content.get("ename", "Unknown"),
                    "evalue": content.get("evalue", ""),
                    "traceback": content.get("traceback", [])
                }
                outputs.append({
                    "type": "error",
                    **error_info
                })
            elif msg_type == "status" and content.get("execution_state") == "idle":
                # Kernel is idle, execution complete
                break
        
        # Get the reply to check for errors
        reply = await self.kernel_client.get_shell_msg(timeout=timeout)
        if reply["content"]["status"] == "error":
            status = 'error'
            if not error_info:
                error_info = {
                    "ename": reply["content"].get("ename", "Unknown"),
                    "evalue": reply["content"].get("evalue", ""),
                    "traceback": reply["content"].get("traceback", [])
                }
        
        return {
            "status": status,
            "outputs": outputs,
            "execution_count": execution_count,
            "error": error_info
        }

    async def shutdown(self):
        """Shutdown the kernel and cleanup resources."""
        if self.kernel_client:
            self.kernel_client.stop_channels()
            
        if self.kernel_manager:
            await self.kernel_manager.shutdown_kernel(now=True)
            
        if self.connection_file and os.path.exists(self.connection_file):
            try:
                os.remove(self.connection_file)
            except OSError:
                pass

    async def interrupt(self):
        """Interrupt the currently executing cell."""
        if self.kernel_manager:
            await self.kernel_manager.interrupt_kernel()

    async def restart(self):
        """Restart the kernel."""
        if self.kernel_manager:
            await self.kernel_manager.restart_kernel()
            await self.kernel_client.wait_for_ready()

    async def is_alive(self) -> bool:
        """Check if the kernel is still alive."""
        if self.kernel_manager:
            return await self.kernel_manager.is_alive()
        return False

    async def get_kernel_info(self) -> Dict[str, Any]:
        """Get kernel information."""
        if not self.kernel_client:
            raise RuntimeError("Kernel not started")
        
        # Request kernel info
        msg_id = self.kernel_client.kernel_info()
        reply = await self.kernel_client.get_shell_msg()
        
        return reply["content"]