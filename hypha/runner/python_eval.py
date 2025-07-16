"""Provide a Python eval runner for simple Python code execution."""

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

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("python_eval")
logger.setLevel(LOGLEVEL)

MAXIMUM_LOG_ENTRIES = 2048


class PythonEvalRunner:
    """Python evaluation runner for simple Python code execution."""

    instance_counter: int = 0

    def __init__(self, server):
        """Initialize the Python eval runner."""
        self.server = server
        self.initialized = False
        self._eval_sessions: Dict[str, Dict[str, Any]] = {}
        self.controller_id = str(PythonEvalRunner.instance_counter)
        PythonEvalRunner.instance_counter += 1
        self.artifact_manager = None
    
    async def initialize(self) -> None:
        """Initialize the Python eval runner."""
        if not self.initialized:
            await self.server.register_service(self.get_service())
            self.artifact_manager = await self.server.get_service("public/artifact-manager")
            self.initialized = True

    async def start(
        self,
        client_id: str,
        app_id: str,
        server_url: str,
        public_base_url: str,
        local_base_url: str,
        workspace: str,
        version: str = None,
        token: str = None,
        entry_point: str = None,
        app_type: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        url: str = None,  # For backward compatibility
        session_id: Optional[str] = None,  # For backward compatibility
    ):
        """Start a Python eval session."""
        if not self.initialized:
            await self.initialize()

        full_session_id = f"{workspace}/{client_id}"
        
        try:
            # Get the Python code from the entry point
            if not entry_point or not entry_point.endswith('.py'):
                raise Exception("Python eval runner requires a .py entry point")
            
            # Read the Python code from the artifact
            get_url = await self.artifact_manager.get_file(
                f"{workspace}/{app_id}", file_path=entry_point, version=version
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
                "HYPHA_SERVER_URL": str(server_url),
                "HYPHA_WORKSPACE": str(workspace), 
                "HYPHA_CLIENT_ID": str(client_id),
                "HYPHA_TOKEN": str(token or ""),
                "HYPHA_APP_ID": str(app_id),
                "HYPHA_PUBLIC_BASE_URL": str(public_base_url),
                "HYPHA_LOCAL_BASE_URL": str(local_base_url),
                "HYPHA_VERSION": str(version or ""),
                "HYPHA_ENTRY_POINT": str(entry_point or "")
            }
            original_os.environ.update(env_vars)
            execution_globals = {
                "__name__": "__main__",
                "print": lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args)),
            }
            
            # Execute the Python code in a separate thread
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, exec, python_code, execution_globals)
                status = "completed"
                error = None
            except Exception as e:
                status = "error" 
                error = traceback.format_exc()
                logs.append(f"Error: {error}")
            # Store session data
            session_data = {
                "session_id": full_session_id,
                "status": status,
                "logs": logs,
                "error": error,
                "metadata": metadata,
                "app_type": app_type,
            }
            
            self._eval_sessions[full_session_id] = session_data
            
            logger.info(f"Python eval session started: {full_session_id}")
            
            return {
                "session_id": full_session_id,
                "status": status,
                "logs": logs,
                "error": error,
            }
        
        except Exception as e:
            logger.error(f"Error starting Python eval session: {str(e)}")
            if full_session_id in self._eval_sessions:
                await self.stop(full_session_id)
            raise

    async def stop(self, session_id: str) -> None:
        """Stop a Python eval session."""
        if session_id not in self._eval_sessions:
            logger.warning(f"Python eval session not found: {session_id}")
            return
        
        try:
            del self._eval_sessions[session_id]
            logger.info(f"Successfully stopped Python eval session: {session_id}")
        except Exception as e:
            logger.error(f"Error stopping Python eval session {session_id}: {str(e)}")
            raise

    async def list(self, workspace) -> List[Dict[str, Any]]:
        """List Python eval sessions for the current workspace."""
        sessions = [
            {k: v for k, v in session_info.items() if k not in ["execution_globals", "python_code"]}
            for session_id, session_info in self._eval_sessions.items()
            if session_id.startswith(workspace + "/")
        ]
        return sessions

    async def get_logs(
        self,
        session_id: str,
        type: str = None,  # pylint: disable=redefined-builtin
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a Python eval session."""
        if session_id not in self._eval_sessions:
            raise Exception(f"Python eval session not found: {session_id}")
        
        session = self._eval_sessions[session_id]
        logs = session.get("logs", [])
        
        if type is None:
            return {"log": logs}
        
        if type == "log":
            if limit is None:
                limit = MAXIMUM_LOG_ENTRIES
            return logs[offset:offset + limit]
        
        return []

    async def take_screenshot(
        self,
        session_id: str,
        format: str = "png",
    ) -> bytes:
        """Take a screenshot for a Python eval session."""
        if session_id not in self._eval_sessions:
            raise Exception(f"Python eval session not found: {session_id}")
        from PIL import Image, ImageDraw, ImageFont

        # Validate format
        if format not in ["png", "jpeg"]:
            raise ValueError(f"Invalid format '{format}'. Must be 'png' or 'jpeg'")
        
        session = self._eval_sessions[session_id]
        
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
            session_info = [
                f"Python Eval Session: {session_id}",
                f"Status: {session.get('status', 'unknown')}",
                f"App Type: {session.get('app_type', 'unknown')}",
                "",
                "Logs:",
            ]
            
            # Add recent logs
            logs = session.get('logs', [])
            recent_logs = logs[-5:] if logs else ["No logs available"]
            session_info.extend(recent_logs)
            
            y_position = 50
            for line in session_info:
                draw.text((50, y_position), line, fill='black', font=font)
                y_position += 30
            
            # Save to buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format=format.upper())
            return img_buffer.getvalue()

    async def close_workspace(self, workspace: str) -> None:
        """Close all Python eval sessions for a workspace."""
        session_ids = [
            session_id
            for session_id in self._eval_sessions.keys()
            if session_id.startswith(workspace + "/")
        ]
        for session_id in session_ids:
            await self.stop(session_id)
    
    async def prepare_workspace(self, workspace: str) -> None:
        """Prepare the workspace for the Python eval runner."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the Python eval runner."""
        logger.info("Closing Python eval runner...")
        try:
            session_ids = list(self._eval_sessions.keys())
            for session_id in session_ids:
                await self.stop(session_id)
            logger.info("Python eval runner closed successfully.")
        except Exception as e:
            logger.error("Error during Python eval runner shutdown: %s", str(e))
            raise

    def get_service(self):
        """Get the service."""
        return {
            "id": f"python-eval-worker-{self.controller_id}",
            "type": "server-app-worker",
            "name": "Python Eval Worker",
            "description": "A worker for running Python evaluation apps",
            "config": {"visibility": "protected"},
            "supported_types": ["python-eval"],
            "start": self.start,
            "stop": self.stop,
            "list": self.list,
            "get_logs": self.get_logs,
            "take_screenshot": self.take_screenshot,
            "shutdown": self.shutdown,
            "close_workspace": self.close_workspace,
            "prepare_workspace": self.prepare_workspace,
        }


async def hypha_startup(server):
    """Initialize the Python eval runner as a startup function."""
    python_eval_runner = PythonEvalRunner(server)
    await python_eval_runner.initialize()
    logger.info("Python eval runner registered as startup function")