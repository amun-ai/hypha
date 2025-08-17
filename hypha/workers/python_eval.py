"""Python Evaluation Worker for simple Python code execution."""

import asyncio
import httpx
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from hypha.workers.base import (
    BaseWorker,
    WorkerConfig,
    SessionStatus,
    SessionInfo,
    SessionNotFoundError,
    WorkerError,
)

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("python_eval")
logger.setLevel(LOGLEVEL)

MAXIMUM_LOG_ENTRIES = 2048


class PythonEvalRunner(BaseWorker):
    """Python evaluation worker for simple Python code execution."""

    instance_counter: int = 0

    def __init__(self, server):
        """Initialize the Python evaluation runner."""
        super().__init__()
        self.controller_id = str(PythonEvalRunner.instance_counter)
        PythonEvalRunner.instance_counter += 1

        # Session management
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["python-eval"]

    @property
    def name(self) -> str:
        """Return the worker name."""
        return "Python Eval Worker"

    @property
    def description(self) -> str:
        """Return the worker description."""
        return "A worker for executing Python code in isolated processes"

    @property
    def require_context(self) -> bool:
        """Return whether the worker requires a context."""
        return True

    @property
    def use_local_url(self) -> bool:
        """Return whether the worker should use local URLs."""
        return True  # Built-in worker runs in same cluster/host

    async def compile(
        self,
        manifest: dict,
        files: list,
        config: dict = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, list]:
        """Compile Python evaluation application - no compilation needed."""
        return manifest, files

    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new Python evaluation session."""
        # Handle both pydantic model and dict input for RPC compatibility
        if isinstance(config, dict):
            config = WorkerConfig(**config)

        session_id = config.id

        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")

        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            app_id=config.app_id,
            workspace=config.workspace,
            client_id=config.client_id,
            status=SessionStatus.STARTING,
            app_type=config.manifest.get("type", "unknown"),
            entry_point=config.entry_point,
            created_at=datetime.now().isoformat(),
            metadata=config.manifest,
        )

        self._sessions[session_id] = session_info

        script_url = f"{config.app_files_base_url}/{config.manifest['entry_point']}?use_proxy=true"
        # use httpx to get artifact files
        async with httpx.AsyncClient() as client:
            response = await client.get(
                script_url, headers={"Authorization": f"Bearer {config.token}"}
            )
            # raise error if response is not 200
            response.raise_for_status()
            script = response.content

        try:
            session_data = await self._start_python_session(script, config)
            self._session_data[session_id] = session_data

            # Update session status
            session_info.status = SessionStatus.RUNNING
            logger.info(f"Started Python eval session {session_id}")

            return session_id

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to start Python eval session {session_id}: {e}")
            # Clean up failed session
            self._sessions.pop(session_id, None)
            raise

    async def _start_python_session(
        self, script: str, config: WorkerConfig
    ) -> Dict[str, Any]:
        """Start a Python evaluation session."""
        assert script is not None, "Script is not found"

        # Execute Python code in subprocess
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
                env={
                    "HYPHA_SERVER_URL": config.server_url,
                    "HYPHA_WORKSPACE": config.workspace,
                    "HYPHA_CLIENT_ID": config.client_id,
                    "HYPHA_TOKEN": config.token,
                },
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise Exception("Python execution timed out")

            # Decode output
            stdout_text = stdout.decode("utf-8") if stdout else ""
            stderr_text = stderr.decode("utf-8") if stderr else ""

            # Store logs
            logs = {
                "stdout": [stdout_text] if stdout_text else [],
                "stderr": [stderr_text] if stderr_text else [],
                "info": [
                    f"Python code executed with return code: {process.returncode}"
                ],
            }

            return {
                "process_id": process.pid,
                "return_code": process.returncode,
                "logs": logs,
                "python_code": script,
            }

        except Exception as e:
            logs = {"error": [f"Failed to execute Python code: {str(e)}"], "info": []}
            return {
                "process_id": None,
                "return_code": -1,
                "logs": logs,
                "python_code": script,
            }

    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop a Python evaluation session."""
        if session_id not in self._sessions:
            logger.warning(
                f"Python eval session {session_id} not found for stopping, may have already been cleaned up"
            )
            return

        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING

        try:
            # Python processes are typically short-lived and already completed
            # Nothing special to clean up for eval sessions
            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped Python eval session {session_id}")

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop Python eval session {session_id}: {e}")
            raise
        finally:
            # Cleanup
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)

    

    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get logs for a Python eval session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Python eval session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {"items": [], "total": 0, "offset": offset, "limit": limit}

        logs = session_data.get("logs", {})
        
        # Convert logs to items format
        all_items = []
        for log_type, log_entries in logs.items():
            for entry in log_entries:
                all_items.append({"type": log_type, "content": entry})
        
        # Filter by type if specified
        if type:
            filtered_items = [item for item in all_items if item["type"] == type]
        else:
            filtered_items = all_items
        
        total = len(filtered_items)
        
        # Apply pagination
        if limit is None:
            paginated_items = filtered_items[offset:]
        else:
            paginated_items = filtered_items[offset:offset + limit]
        
        return {
            "items": paginated_items,
            "total": total,
            "offset": offset,
            "limit": limit
        }

    async def shutdown(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Shutdown the Python eval worker."""
        logger.info("Shutting down Python eval worker...")

        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop Python eval session {session_id}: {e}")

        logger.info("Python eval worker shutdown complete")

    def get_worker_service(self) -> Dict[str, Any]:
        """Get the service configuration for registration with python-eval-specific methods."""
        service_config = super().get_worker_service()
        return service_config


async def hypha_startup(server):
    """Hypha startup function to initialize Python eval worker."""
    worker = PythonEvalRunner(server)
    await worker.register_worker_service(server)
    logger.info("Python eval worker initialized and registered")


async def start_worker(server_url, workspace, token):
    """Start Python eval worker standalone."""
    from hypha_rpc import connect

    server = await connect(server_url, workspace=workspace, token=token)
    worker = PythonEvalRunner(server.rpc)
    logger.info(
        f"Python eval worker started, server: {server_url}, workspace: {workspace}"
    )

    return worker


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--token", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(start_worker(args.server_url, args.workspace, args.token))
