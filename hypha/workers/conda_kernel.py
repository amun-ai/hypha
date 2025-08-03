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
        
        # Create kernel manager with a minimal setup to avoid kernel spec issues
        self.kernel_manager = AsyncKernelManager(
            connection_file=self.connection_file,
        )
        
        # Override kernel_spec to avoid NoSuchKernel error
        from jupyter_client.kernelspec import KernelSpec
        self.kernel_manager._kernel_spec = KernelSpec(
            argv=[str(self.python_path), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name=f"Python ({self.conda_env_path.name})",
            language="python",
        )
        
        # kernel_cmd is already set via the KernelSpec above
        
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
        execution_finished = False
        
        start_time = time.time()
        
        # Create tasks for both shell and iopub channels
        async def collect_messages():
            nonlocal outputs, error, status, execution_finished
            
            # Track if we've seen the execution status (idle) message
            seen_idle = False
            
            while not execution_finished and (time.time() - start_time) < timeout:
                try:
                    messages_found = False
                    
                    # Check for iopub messages (outputs, errors, status)
                    try:
                        iopub_msg = await asyncio.wait_for(
                            self.kernel_client.get_iopub_msg(timeout=0.1), 
                            timeout=0.1
                        )
                        messages_found = True
                        
                        if iopub_msg['parent_header'].get('msg_id') == msg_id:
                            msg_type = iopub_msg['msg_type']
                            content = iopub_msg['content']
                            
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
                            elif msg_type == 'status':
                                # Track kernel status - execution is done when it goes to 'idle'
                                if content.get('execution_state') == 'idle':
                                    seen_idle = True
                                    
                    except asyncio.TimeoutError:
                        pass  # No iopub message available
                    
                    # Check for shell messages (execution completion)
                    try:
                        shell_msg = await asyncio.wait_for(
                            self.kernel_client.get_shell_msg(timeout=0.1),
                            timeout=0.1
                        )
                        messages_found = True
                        
                        if (shell_msg['parent_header'].get('msg_id') == msg_id and 
                            shell_msg['msg_type'] == 'execute_reply'):
                            content = shell_msg['content']
                            
                            if content.get('status') == 'error' and not error:
                                status = "error"
                                error = {
                                    "ename": content.get("ename", "Unknown"), 
                                    "evalue": content.get("evalue", "Unknown error"),
                                    "traceback": content.get("traceback", [])
                                }
                            
                            # Execution is finished when we get the execute_reply
                            # We'll wait a bit more for any remaining iopub messages
                            execution_finished = True
                            
                            # Give a small window for any remaining iopub messages
                            await asyncio.sleep(0.1)
                            
                            # Process any remaining iopub messages
                            try:
                                while True:
                                    iopub_msg = await asyncio.wait_for(
                                        self.kernel_client.get_iopub_msg(timeout=0.1), 
                                        timeout=0.1
                                    )
                                    
                                    if iopub_msg['parent_header'].get('msg_id') == msg_id:
                                        msg_type = iopub_msg['msg_type']
                                        content = iopub_msg['content']
                                        
                                        if msg_type == 'stream':
                                            outputs.append({
                                                "type": "stream",
                                                "name": content.get("name", "stdout"),
                                                "text": content.get("text", "")
                                            })
                                        elif msg_type == 'execute_result':
                                            outputs.append({
                                                "type": "execute_result",
                                                "data": content.get("data", {}),
                                                "metadata": content.get("metadata", {}),
                                                "execution_count": content.get("execution_count", 0)
                                            })
                            except asyncio.TimeoutError:
                                pass  # No more messages
                                
                            break
                            
                    except asyncio.TimeoutError:
                        pass  # No shell message available
                    
                    # If no messages found, sleep a bit longer to avoid busy waiting
                    if not messages_found:
                        await asyncio.sleep(0.05)
                    else:
                        await asyncio.sleep(0.01)
                    
                except Exception as e:
                    status = "error"
                    error = {"ename": type(e).__name__, "evalue": str(e), "traceback": []}
                    execution_finished = True
                    break
        
        # Run message collection with overall timeout
        try:
            await asyncio.wait_for(collect_messages(), timeout=timeout)
        except asyncio.TimeoutError:
            status = "timeout"
            error = {"ename": "TimeoutError", "evalue": f"Execution timed out after {timeout}s", "traceback": []}
        
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