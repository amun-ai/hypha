"""Provide a Python eval worker for simple Python code execution."""

import asyncio
import logging
import os
import sys
import uuid
import httpx
from typing import Any, Dict, List, Optional, Union
import json
import traceback
import io

from hypha.workers.base import BaseWorker, WorkerConfig, SessionStatus

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("python_eval")
logger.setLevel(LOGLEVEL)

MAXIMUM_LOG_ENTRIES = 2048


class PythonEvalRunner(BaseWorker):
    """Python evaluation worker for simple Python code execution."""

    instance_counter: int = 0

    def __init__(self, server):
        """Initialize the Python eval worker."""
        super().__init__(server)
        self.controller_id = str(PythonEvalRunner.instance_counter)
        PythonEvalRunner.instance_counter += 1
        self.artifact_manager = None

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["python-eval"]

    @property
    def worker_name(self) -> str:
        """Return the worker name."""
        return "Python Eval Worker"

    @property
    def worker_description(self) -> str:
        """Return the worker description."""
        return "A worker for running Python evaluation apps"

    async def _initialize_worker(self) -> None:
        """Initialize the Python eval worker."""
        self.artifact_manager = await self.server.get_service("public/artifact-manager")
    
    async def _start_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Start a Python eval session."""
        # Get the Python code from the entry point
        if not config.entry_point or not config.entry_point.endswith('.py'):
            raise Exception("Python eval worker requires a .py entry point")
        
        # Read the Python code from the artifact
        get_url = await self.artifact_manager.get_file(
            f"{config.workspace}/{config.app_id}", file_path=config.entry_point, version=config.version
        )
        
        # Fetch the Python code
        async with httpx.AsyncClient() as client:
            response = await client.get(get_url)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch Python code: {response.status_code}")
            python_code = response.text

        # Create execution environment with custom os module
        logs = []
        # Create a custom os module with injected environment variables
        import os as original_os

        # Convert all values to strings to avoid type errors
        env_vars = {
            "HYPHA_SERVER_URL": str(config.server_url),
            "HYPHA_WORKSPACE": str(config.workspace), 
            "HYPHA_CLIENT_ID": str(config.client_id),
            "HYPHA_TOKEN": str(config.token or ""),
            "HYPHA_APP_ID": str(config.app_id),
            "HYPHA_PUBLIC_BASE_URL": str(config.public_base_url),
            "HYPHA_LOCAL_BASE_URL": str(config.local_base_url),
            "HYPHA_VERSION": str(config.version or ""),
            "HYPHA_ENTRY_POINT": str(config.entry_point or "")
        }
        original_os.environ.update(env_vars)
        execution_globals = {
            "__name__": "__main__",
            "print": lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args)),
        }
        
        # Execute the Python code in a separate thread
        loop = asyncio.get_event_loop()
        error = None
        try:
            await loop.run_in_executor(None, exec, python_code, execution_globals)
            status = "completed"
        except Exception as e:
            status = "error" 
            error = traceback.format_exc()
            logs.append(f"Error: {error}")
        
        return {
            "status": status,
            "logs": {"log": logs, "error": [] if not error else [error]},
            "error": error,
            "execution_globals": execution_globals,
        }

    async def _stop_session(self, session_id: str) -> None:
        """Stop a Python eval session."""
        # Python eval sessions are stateless, so nothing to clean up
        pass

    # list_sessions method is now inherited from BaseWorker

    async def _get_session_logs(
        self, 
        session_id: str, 
        log_type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a Python eval session."""
        session_data = self._session_data.get(session_id)
        if not session_data:
            return {} if log_type is None else []

        logs = session_data.get("logs", {"log": [], "error": []})
        
        if log_type:
            target_logs = logs.get(log_type, [])
            end_idx = len(target_logs) if limit is None else min(offset + limit, len(target_logs))
            return target_logs[offset:end_idx]
        else:
            result = {}
            for log_type_key, log_entries in logs.items():
                end_idx = len(log_entries) if limit is None else min(offset + limit, len(log_entries))
                result[log_type_key] = log_entries[offset:end_idx]
            return result

    async def take_screenshot(
        self,
        session_id: str,
        format: str = "png",
    ) -> bytes:
        """Take a screenshot for a Python eval session."""
        if session_id not in self._sessions:
            raise Exception(f"Python eval session not found: {session_id}")
        
        from PIL import Image, ImageDraw, ImageFont

        # Validate format
        if format not in ["png", "jpeg"]:
            raise ValueError(f"Invalid format '{format}'. Must be 'png' or 'jpeg'")
        
        session_info = self._sessions[session_id]
        session_data = self._session_data.get(session_id, {})
        
        try:
            # Try to capture the desktop screen
            if sys.platform == "darwin":  # macOS
                import subprocess
                # Use screencapture command on macOS
                result = subprocess.run(
                    ["screencapture", "-t", format, "-"], 
                    capture_output=True, 
                    check=True
                )
                return result.stdout
            elif sys.platform.startswith("linux"):
                # Try to use scrot or gnome-screenshot on Linux
                import subprocess
                try:
                    result = subprocess.run(
                        ["scrot", "-o", "/dev/stdout"], 
                        capture_output=True, 
                        check=True
                    )
                    return result.stdout
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fall back to gnome-screenshot
                    result = subprocess.run(
                        ["gnome-screenshot", "-f", "/dev/stdout"], 
                        capture_output=True, 
                        check=True
                    )
                    return result.stdout
            elif sys.platform == "win32":
                # Use PIL ImageGrab on Windows
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                img_buffer = io.BytesIO()
                screenshot.save(img_buffer, format=format.upper())
                return img_buffer.getvalue()
            else:
                # Fallback: create a placeholder screenshot
                raise NotImplementedError(f"Screenshot not supported on {sys.platform}")
        
        except Exception as e:
            logger.warning(f"Failed to capture desktop screenshot: {str(e)}, creating placeholder")
            # Create a placeholder screenshot
            img = Image.new('RGB', (800, 600), color='#f0f0f0')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except (IOError, OSError):
                font = ImageFont.load_default()
            
            # Draw session information
            info_lines = [
                f"Python Eval Session: {session_id}",
                f"Status: {session_info.status.value}",
                f"App Type: {session_info.app_type}",
                "",
                "Logs:",
            ]
            
            # Add recent logs
            logs = session_data.get('logs', {}).get('log', [])
            recent_logs = logs[-5:] if logs else ["No logs available"]
            info_lines.extend(recent_logs)
            
            y_position = 50
            for line in info_lines:
                draw.text((50, y_position), line, fill='black', font=font)
                y_position += 30
            
            # Save to buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format=format.upper())
            return img_buffer.getvalue()

    async def _close_workspace(self, workspace: str) -> None:
        """Close all Python eval sessions for a workspace."""
        # This is now handled by the base class
        pass
    
    async def _prepare_workspace(self, workspace: str) -> None:
        """Prepare the workspace for the Python eval worker."""
        pass

    async def _shutdown_worker(self) -> None:
        """Shutdown the Python eval worker."""
        logger.info("Python eval worker shutdown complete.")
        # No specific cleanup needed for Python eval worker
        pass

    def get_service(self):
        """Get the service."""
        service_config = self.get_service_config()
        # Add Python eval specific methods
        service_config["take_screenshot"] = self.take_screenshot
        return service_config


async def hypha_startup(server):
    """Initialize the Python eval worker as a startup function."""
    python_eval_runner = PythonEvalRunner(server)
    await python_eval_runner.initialize()
    logger.info("Python eval worker registered as startup function")

async def start_worker(server_url, workspace, token):
    """Start the Python eval worker."""
    from hypha_rpc import connect_to_server
    async with connect_to_server(server_url) as server:
        python_eval_runner = PythonEvalRunner(server)
        await python_eval_runner.initialize()
        logger.info("Python eval worker registered as startup function")
        await server.serve()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--token", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(start_worker(args.server_url, args.workspace, args.token))