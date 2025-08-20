"""Kubernetes Worker for launching pods in Kubernetes clusters."""

import asyncio
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream

from hypha.workers.base import (
    BaseWorker,
    WorkerConfig,
    SessionStatus,
    SessionInfo,
    SessionNotFoundError,
    WorkerError,
    WorkerNotAvailableError,
    safe_call_callback,
)

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("k8s")
logger.setLevel(LOGLEVEL)

def to_k8s_pod_name(session_id: str) -> str:
    sanitized_id = re.sub(r'[^a-zA-Z0-9-]', '-', session_id)
    sanitized_id = re.sub(r'-+', '-', sanitized_id)
    sanitized_id = sanitized_id.strip('-')
    if sanitized_id and not sanitized_id[0].isalnum():
        sanitized_id = f"s{sanitized_id}"
    if not sanitized_id:
        sanitized_id = str(uuid.uuid4())
    if len(sanitized_id) > 50:
        sanitized_id = sanitized_id[:50].rstrip('-')
    pod_name = f"hypha-pod-{sanitized_id}".lower()
    return pod_name
class KubernetesWorker(BaseWorker):
    """Kubernetes worker for launching pods in Kubernetes clusters."""

    instance_counter: int = 0

    def __init__(
        self,
        namespace: str = "default",
        default_timeout: int = 3600,
        image_pull_policy: str = "IfNotPresent",
    ):
        """Initialize the Kubernetes worker."""
        super().__init__()
        self.namespace = namespace
        self.default_timeout = default_timeout
        self.image_pull_policy = image_pull_policy
        
        self.instance_id = f"k8s-worker-{uuid.uuid4().hex[:8]}"
        self.controller_id = str(KubernetesWorker.instance_counter)
        KubernetesWorker.instance_counter += 1

        # Session management
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}

        # Initialize Kubernetes client
        self._init_k8s_client()

    def _init_k8s_client(self):
        """Initialize Kubernetes client."""
        try:
            # Try to load in-cluster config first (when running in a pod)
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            try:
                # Fall back to local kubeconfig
                config.load_kube_config()
                logger.info("Loaded local Kubernetes config")
            except config.ConfigException:
                error_msg = "Unable to load Kubernetes config"
                logger.error(error_msg)
                raise WorkerNotAvailableError(error_msg)

        self.v1 = client.CoreV1Api()
        logger.info(f"Kubernetes client initialized for namespace: {self.namespace}")

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["k8s-pod"]

    @property
    def name(self) -> str:
        """Return the worker name."""
        return f"Kubernetes Pod Worker (namespace: {self.namespace})"

    @property
    def description(self) -> str:
        """Return the worker description."""
        return f"A worker for launching pods in Kubernetes cluster namespace '{self.namespace}'"

    @property
    def require_context(self) -> bool:
        """Return whether the worker requires a context."""
        return True

    async def compile(
        self,
        manifest: dict,
        files: list,
        config: dict = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, list]:
        """Compile Kubernetes pod application - validate manifest."""
        # Validate manifest has required fields
        required_fields = ["image"]
        for field in required_fields:
            if field not in manifest:
                raise WorkerError(f"Required field '{field}' missing from manifest")

        # Set defaults
        manifest.setdefault("image_pull_policy", self.image_pull_policy)
        manifest.setdefault("restart_policy", "Never")
        manifest.setdefault("timeout", self.default_timeout)

        # Validate image format
        image = manifest["image"]
        if not image or ":" not in image:
            raise WorkerError(f"Invalid image format: {image}")

        # Validate environment variables format
        env = manifest.get("env", {})
        if not isinstance(env, dict):
            raise WorkerError("Environment variables must be a dictionary")

        # Validate command format
        command = manifest.get("command", [])
        if command and not isinstance(command, list):
            raise WorkerError("Command must be a list")

        logger.info(f"Compiled Kubernetes pod manifest for image: {image}")
        return manifest, files

    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new Kubernetes pod session."""
        # Handle both pydantic model and dict input for RPC compatibility
        if isinstance(config, dict):
            config = WorkerConfig(**config)

        session_id = config.id
        progress_callback = getattr(config, "progress_callback", None)

        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")

        # Report initial progress
        await safe_call_callback(progress_callback,
            {
                "type": "info",
                "message": f"Starting Kubernetes pod session {session_id}",
            }
        )

        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            app_id=config.app_id,
            workspace=config.workspace,
            client_id=config.client_id,
            status=SessionStatus.STARTING,
            app_type=config.manifest.get("type", "k8s-pod"),
            entry_point=config.entry_point,
            created_at=datetime.now().isoformat(),
            metadata=config.manifest,
        )

        self._sessions[session_id] = session_info

        try:
            session_data = await self._start_k8s_pod(config, session_id, progress_callback)
            self._session_data[session_id] = session_data

            # Update session status
            session_info.status = SessionStatus.RUNNING

            await safe_call_callback(progress_callback,
                {
                    "type": "success",
                    "message": f"Kubernetes pod session {session_id} started successfully",
                }
            )

            logger.info(f"Started Kubernetes pod session {session_id}")
            return session_id

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)

            await safe_call_callback(progress_callback,
                {
                    "type": "error",
                    "message": f"Failed to start Kubernetes pod session: {str(e)}",
                }
            )

            logger.error(f"Failed to start Kubernetes pod session {session_id}: {e}")
            # Clean up failed session
            self._sessions.pop(session_id, None)
            raise

    async def _start_k8s_pod(
        self, config: WorkerConfig, session_id: str, progress_callback=None
    ) -> Dict[str, Any]:
        """Start a Kubernetes pod session."""
        # Initialize logs
        logs = {
            "stdout": [],
            "stderr": [],
            "info": [f"Kubernetes pod session started successfully"],
            "error": [],
            "progress": [],
        }

        # Create a progress callback wrapper
        def progress_callback_wrapper(message):
            logs["progress"].append(f"{message['type'].upper()}: {message['message']}")
            if progress_callback:
                progress_callback(message)

        # Extract pod specification from manifest
        manifest = config.manifest
        image = manifest["image"]
        command = manifest.get("command", [])
        env = manifest.get("env", {})
        image_pull_policy = manifest.get("image_pull_policy", self.image_pull_policy)
        restart_policy = manifest.get("restart_policy", "Never")
        timeout = manifest.get("timeout", self.default_timeout)

        progress_callback_wrapper(
            {"type": "info", "message": f"Creating Kubernetes pod with image: {image}"}
        )
        pod_name = to_k8s_pod_name(session_id)
        # Build environment variables - include Hypha configuration automatically
        env_vars = []
        
        # Add Hypha environment variables automatically
        hypha_env = {
            "HYPHA_SERVER_URL": config.server_url,
            "HYPHA_WORKSPACE": config.workspace,
            "HYPHA_CLIENT_ID": config.client_id,
            "HYPHA_TOKEN": config.token,
            "HYPHA_APP_ID": config.app_id,
        }
        
        # Merge with user-provided environment variables (user variables take precedence)
        merged_env = {**hypha_env, **env}
        
        for key, value in merged_env.items():
            env_vars.append(client.V1EnvVar(name=key, value=str(value)))

        # Create container spec with security context
        container = client.V1Container(
            name="main",
            image=image,
            command=command if command else None,
            env=env_vars,
            image_pull_policy=image_pull_policy,
            security_context=client.V1SecurityContext(
                allow_privilege_escalation=False,
                capabilities=client.V1Capabilities(drop=["ALL"]),
                run_as_non_root=True,
                seccomp_profile=client.V1SeccompProfile(type="RuntimeDefault"),
            ),
        )

        # Create pod spec with security context
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy=restart_policy,
            service_account_name="default",
            security_context=client.V1PodSecurityContext(
                run_as_user=1000,
                run_as_non_root=True,
                fs_group=1000,
                seccomp_profile=client.V1SeccompProfile(type="RuntimeDefault"),
            ),
        )

        # Create pod metadata with minimal labels and full metadata in annotations
        pod_metadata = client.V1ObjectMeta(
            name=pod_name,
            labels={
                "app": "hypha",
                "component": "worker-pod",
            },
            annotations={
                "hypha.amun.ai/app-id": config.app_id,
                "hypha.amun.ai/workspace": config.workspace,
                "hypha.amun.ai/client-id": config.client_id,
                "hypha.amun.ai/session-id": session_id,
                "hypha.amun.ai/created-at": datetime.now().isoformat(),
                "hypha.amun.ai/worker-type": "k8s",
            },
        )

        # Create pod
        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=pod_metadata,
            spec=pod_spec,
        )

        try:
            # Create the pod in Kubernetes
            self.v1.create_namespaced_pod(namespace=self.namespace, body=pod)

            progress_callback_wrapper(
                {
                    "type": "success",
                    "message": f"Pod {pod_name} created successfully",
                }
            )

            # Wait for pod to be running (with timeout)
            progress_callback_wrapper(
                {"type": "info", "message": "Waiting for pod to start..."}
            )

            start_time = time.time()
            max_wait_time = 300  # 5 minutes
            while time.time() - start_time < max_wait_time:
                try:
                    pod_status = self.v1.read_namespaced_pod(
                        name=pod_name, namespace=self.namespace
                    )
                    phase = pod_status.status.phase

                    if phase == "Running":
                        progress_callback_wrapper(
                            {
                                "type": "success",
                                "message": f"Pod {pod_name} is now running",
                            }
                        )
                        break
                    elif phase == "Failed":
                        raise WorkerError(f"Pod {pod_name} failed to start")
                    elif phase == "Succeeded":
                        # Pod completed immediately
                        progress_callback_wrapper(
                            {
                                "type": "info",
                                "message": f"Pod {pod_name} completed immediately",
                            }
                        )
                        break
                    
                    # Still pending or other state
                    await asyncio.sleep(2)
                    
                except ApiException as e:
                    if e.status == 404:
                        raise WorkerError(f"Pod {pod_name} was not found")
                    raise

            else:
                # Timeout waiting for pod to start
                raise WorkerError(f"Timeout waiting for pod {pod_name} to start")

        except ApiException as e:
            progress_callback_wrapper(
                {
                    "type": "error",
                    "message": f"Kubernetes API error: {str(e)}",
                }
            )
            raise WorkerError(f"Failed to create pod: {str(e)}")

        return {
            "pod_name": pod_name,
            "image": image,
            "command": command,
            "env": merged_env,
            "logs": logs,
            "timeout": timeout,
        }

    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop a Kubernetes pod session."""
        if session_id not in self._sessions:
            logger.warning(f"Kubernetes pod session {session_id} not found for stopping")
            return

        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING

        try:
            session_data = self._session_data.get(session_id)
            if session_data:
                pod_name = session_data["pod_name"]
                logger.info(f"Stopping Kubernetes pod {pod_name} for session {session_id}")

                try:
                    self.v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions(),
                    )
                    logger.info(f"Successfully deleted pod {pod_name}")
                except ApiException as e:
                    if e.status == 404:
                        logger.warning(f"Pod {pod_name} was already deleted")
                    else:
                        raise WorkerError(f"Failed to delete pod {pod_name}: {str(e)}")

            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped Kubernetes pod session {session_id}")

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop Kubernetes pod session {session_id}: {e}")
            raise
        finally:
            # Cleanup
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)

    async def list_sessions(
        self, workspace: str, context: Optional[Dict[str, Any]] = None
    ) -> List[SessionInfo]:
        """List all Kubernetes pod sessions for a workspace."""
        return [
            session_info
            for session_info in self._sessions.values()
            if session_info.workspace == workspace
        ]

    async def get_session_info(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """Get information about a Kubernetes pod session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Kubernetes pod session {session_id} not found")
        return self._sessions[session_id]

    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get logs for a Kubernetes pod session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Kubernetes pod session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {"items": [], "total": 0, "offset": offset, "limit": limit}

        # Try to get real-time logs from Kubernetes
        pod_name = session_data.get("pod_name")
        if pod_name:
            try:
                # Get pod logs from Kubernetes
                k8s_logs = self.v1.read_namespaced_pod_log(
                    name=pod_name, namespace=self.namespace, container="main"
                )
                
                # Update session logs with real-time data
                if k8s_logs:
                    logs = session_data.get("logs", {})
                    logs.setdefault("stdout", []).append(k8s_logs)
                    
            except ApiException as e:
                logger.warning(f"Could not fetch logs for pod {pod_name}: {e}")

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

    async def execute(
        self,
        session_id: str,
        script: str,
        config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a script in the running pod session.
        
        This method executes commands inside the running Kubernetes pod using kubectl exec.
        
        Args:
            session_id: The session to execute in
            script: The script/command to execute
            config: Optional execution configuration
            progress_callback: Optional callback for execution progress
            context: Optional context information
            
        Returns:
            Dictionary containing execution results with status and outputs
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Kubernetes pod session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            raise WorkerError(f"No pod data available for session {session_id}")

        pod_name = session_data.get("pod_name")
        if not pod_name:
            raise WorkerError(f"No pod name found for session {session_id}")

        await safe_call_callback(progress_callback,
            {"type": "info", "message": f"Executing command in pod {pod_name}..."}
        )

        try:
            # Configure execution options from config
            timeout = config.get("timeout", 30) if config else 30
            container_name = config.get("container", "main") if config else "main"
            shell = config.get("shell", "/bin/sh") if config else "/bin/sh"
            
            # Prepare the command - if script contains newlines or is complex, use shell -c
            if '\n' in script or ';' in script or '|' in script or '&&' in script:
                # Complex script - use shell with -c
                command = [shell, "-c", script]
            else:
                # Simple command - split by spaces
                command = script.strip().split()

            logger.info(f"Executing command in pod {pod_name}: {command}")

            # Execute command in the pod
            try:
                # Use the stream API to execute the command
                exec_output = stream(
                    self.v1.connect_get_namespaced_pod_exec,
                    name=pod_name,
                    namespace=self.namespace,
                    container=container_name,
                    command=command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                    _preload_content=False
                )

                # Collect output with timeout
                stdout_lines = []
                stderr_lines = []
                
                start_time = time.time()
                while exec_output.is_open():
                    exec_output.update(timeout=1)
                    
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        exec_output.close()
                        raise asyncio.TimeoutError(f"Command execution timed out after {timeout} seconds")
                    
                    # Read available output
                    if exec_output.peek_stdout():
                        stdout_lines.append(exec_output.read_stdout())
                    if exec_output.peek_stderr():
                        stderr_lines.append(exec_output.read_stderr())
                
                # Get return code
                return_code = exec_output.returncode if hasattr(exec_output, 'returncode') else None
                
                # Combine output
                stdout_text = ''.join(stdout_lines).rstrip('\n')
                stderr_text = ''.join(stderr_lines).rstrip('\n')

                # Create outputs in Jupyter-like format for consistency
                outputs = []
                
                if stdout_text:
                    outputs.append({
                        "type": "stream",
                        "name": "stdout", 
                        "text": stdout_text
                    })
                
                if stderr_text:
                    outputs.append({
                        "type": "stream",
                        "name": "stderr",
                        "text": stderr_text
                    })

                # Update session logs
                logs = session_data.get("logs", {})
                if stdout_text:
                    logs.setdefault("stdout", []).append(stdout_text)
                if stderr_text:
                    logs.setdefault("stderr", []).append(stderr_text)

                # Add execution info
                success = return_code == 0 if return_code is not None else True
                status_text = "success" if success else "error"
                logs.setdefault("info", []).append(
                    f"Command executed at {datetime.now().isoformat()} - Status: {status_text}"
                )

                await safe_call_callback(progress_callback,
                    {"type": "success" if success else "error",
                     "message": "Command executed successfully" if success else f"Command failed with return code: {return_code}"}
                )

                # Return result in consistent format
                result = {
                    "status": "ok" if success else "error",
                    "outputs": outputs
                }
                
                if not success and return_code is not None:
                    result["error"] = {
                        "ename": "CommandError",
                        "evalue": f"Command failed with return code {return_code}",
                        "traceback": [stderr_text] if stderr_text else [f"Command failed with return code {return_code}"]
                    }
                    
                return result

            except ApiException as e:
                error_msg = f"Kubernetes API error during command execution: {str(e)}"
                logger.error(f"Failed to execute command in pod {pod_name}: {e}")
                
                await safe_call_callback(progress_callback, {"type": "error", "message": error_msg})
                
                # Update logs
                logs = session_data.get("logs", {})
                logs.setdefault("error", []).append(error_msg)
                
                return {
                    "status": "error",
                    "outputs": [],
                    "error": {
                        "ename": "KubernetesApiError",
                        "evalue": str(e),
                        "traceback": [error_msg]
                    }
                }

        except asyncio.TimeoutError as e:
            error_msg = str(e)
            await safe_call_callback(progress_callback, {"type": "error", "message": error_msg})
            
            logs = session_data.get("logs", {})
            logs.setdefault("error", []).append(error_msg)
            
            return {
                "status": "error",
                "outputs": [],
                "error": {
                    "ename": "TimeoutError",
                    "evalue": error_msg,
                    "traceback": [error_msg]
                }
            }
        except Exception as e:
            error_msg = f"Failed to execute command: {str(e)}"
            logger.error(f"Failed to execute command in session {session_id}: {e}")
            
            await safe_call_callback(progress_callback, {"type": "error", "message": error_msg})
            
            logs = session_data.get("logs", {})
            logs.setdefault("error", []).append(error_msg)
            
            return {
                "status": "error", 
                "outputs": [],
                "error": {
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [error_msg]
                }
            }

    async def shutdown(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Shutdown the Kubernetes worker."""
        logger.info("Shutting down Kubernetes worker...")

        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop Kubernetes pod session {session_id}: {e}")

        logger.info("Kubernetes worker shutdown complete")


async def hypha_startup(server):
    """Hypha startup function to initialize Kubernetes worker."""
    # Get configuration from environment variables
    namespace = os.environ.get("HYPHA_K8S_NAMESPACE", "default")
    default_timeout = int(os.environ.get("HYPHA_K8S_DEFAULT_TIMEOUT", "3600"))
    image_pull_policy = os.environ.get("HYPHA_K8S_IMAGE_PULL_POLICY", "IfNotPresent")

    try:
        worker = KubernetesWorker(
            namespace=namespace,
            default_timeout=default_timeout,
            image_pull_policy=image_pull_policy,
        )
        await worker.register_worker_service(server)
        logger.info(f"Kubernetes worker initialized and registered for namespace: {namespace}")
    except WorkerNotAvailableError as e:
        logger.error(f"Kubernetes worker not available: {e}")
        logger.info("Skipping Kubernetes worker registration")
    except Exception as e:
        logger.error(f"Failed to initialize Kubernetes worker: {e}")
        raise


def main():
    """Main function for command line execution."""
    import argparse

    def get_env_var(name: str, default: str = None) -> str:
        """Get environment variable with HYPHA_ prefix."""
        return os.environ.get(f"HYPHA_{name.upper()}", default)

    parser = argparse.ArgumentParser(
        description="Hypha Kubernetes Worker - Launch pods in Kubernetes clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (with HYPHA_ prefix):
  HYPHA_SERVER_URL         Hypha server URL (e.g., https://hypha.aicell.io)
  HYPHA_WORKSPACE          Workspace name (e.g., my-workspace)
  HYPHA_TOKEN              Authentication token
  HYPHA_SERVICE_ID         Service ID for the worker (optional)
  HYPHA_VISIBILITY         Service visibility: public or protected (default: protected)
  HYPHA_K8S_NAMESPACE      Kubernetes namespace (default: default)
  HYPHA_K8S_DEFAULT_TIMEOUT Default timeout for pods in seconds (default: 3600)
  HYPHA_K8S_IMAGE_PULL_POLICY Image pull policy (default: IfNotPresent)

Examples:
  # Using command line arguments
  python -m hypha.workers.k8s --server-url https://hypha.aicell.io --workspace my-workspace --token TOKEN

  # Using environment variables
  export HYPHA_SERVER_URL=https://hypha.aicell.io
  export HYPHA_WORKSPACE=my-workspace
  export HYPHA_TOKEN=your-token-here
  export HYPHA_K8S_NAMESPACE=my-namespace
  python -m hypha.workers.k8s
        """,
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default=get_env_var("SERVER_URL"),
        help="Hypha server URL (default: from HYPHA_SERVER_URL env var)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=get_env_var("WORKSPACE"),
        help="Workspace name (default: from HYPHA_WORKSPACE env var)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=get_env_var("TOKEN"),
        help="Authentication token (default: from HYPHA_TOKEN env var)",
    )
    parser.add_argument(
        "--service-id",
        type=str,
        default=get_env_var("SERVICE_ID"),
        help="Service ID for the worker (default: from HYPHA_SERVICE_ID env var or auto-generated)",
    )
    parser.add_argument(
        "--visibility",
        type=str,
        choices=["public", "protected"],
        default=get_env_var("VISIBILITY", "protected"),
        help="Service visibility (default: protected, from HYPHA_VISIBILITY env var)",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=get_env_var("CLIENT_ID"),
        help="Client ID for the worker (default: from HYPHA_CLIENT_ID env var or auto-generated)",
    )
    parser.add_argument(
        "--disable-ssl",
        action="store_true",
        help="Disable SSL verification (default: false)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=get_env_var("K8S_NAMESPACE", "default"),
        help="Kubernetes namespace (default: default, from HYPHA_K8S_NAMESPACE env var)",
    )
    parser.add_argument(
        "--default-timeout",
        type=int,
        default=int(get_env_var("K8S_DEFAULT_TIMEOUT", "3600")),
        help="Default timeout for pods in seconds (default: 3600, from HYPHA_K8S_DEFAULT_TIMEOUT env var)",
    )
    parser.add_argument(
        "--image-pull-policy",
        type=str,
        choices=["Always", "IfNotPresent", "Never"],
        default=get_env_var("K8S_IMAGE_PULL_POLICY", "IfNotPresent"),
        help="Image pull policy (default: IfNotPresent, from HYPHA_K8S_IMAGE_PULL_POLICY env var)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.server_url:
        print(
            "Error: --server-url is required (or set HYPHA_SERVER_URL environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.workspace:
        print(
            "Error: --workspace is required (or set HYPHA_WORKSPACE environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.token:
        print(
            "Error: --token is required (or set HYPHA_TOKEN environment variable)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set up logging
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.setLevel(logging.INFO)

    print(f"Starting Hypha Kubernetes Worker...")
    print(f"  Server URL: {args.server_url}")
    print(f"  Workspace: {args.workspace}")
    print(f"  Service ID: {args.service_id}")
    print(f"  Client ID: {args.client_id}")
    print(f"  Disable SSL: {args.disable_ssl}")
    print(f"  Visibility: {args.visibility}")
    print(f"  Namespace: {args.namespace}")
    print(f"  Default Timeout: {args.default_timeout}s")
    print(f"  Image Pull Policy: {args.image_pull_policy}")

    async def run_worker():
        """Run the Kubernetes worker."""
        try:
            from hypha_rpc import connect_to_server

            # Connect to server
            server = await connect_to_server(
                server_url=args.server_url,
                workspace=args.workspace,
                token=args.token,
                client_id=args.client_id,
                ssl=False if args.disable_ssl else None,
            )

            # Create and register worker
            worker = KubernetesWorker(
                namespace=args.namespace,
                default_timeout=args.default_timeout,
                image_pull_policy=args.image_pull_policy,
            )

            # Get service config and set custom properties
            service_config = worker.get_worker_service()
            if args.service_id:
                service_config["id"] = args.service_id
            # Set visibility in the correct location (inside config)
            service_config["config"]["visibility"] = args.visibility

            # Register the service
            print(f"🔄 Registering Kubernetes worker with config:")
            print(f"   Service ID: {service_config['id']}")
            print(f"   Type: {service_config['type']}")
            print(f"   Supported types: {service_config['supported_types']}")
            print(f"   Visibility: {service_config.get('config', {}).get('visibility', 'N/A')}")
            print(f"   Namespace: {args.namespace}")

            registration_result = await server.register_service(service_config)
            print(f"   Registered service id: {registration_result.id}")

            # Verify registration by listing services
            try:
                services = await server.list_services({"type": "server-app-worker"})
                print(f"   Found {len(services)} server-app-worker services in workspace")
                k8s_workers = [s for s in services if s.get('id').endswith(service_config['id'])]
                if k8s_workers:
                    print(f"   ✅ Worker found in service list")
                else:
                    print(f"   ⚠️  Worker NOT found in service list!")
                    print(f"   Available workers: {[s.get('id') for s in services]}")
                    raise WorkerNotAvailableError("Kubernetes worker not found in service list")
            except Exception as e:
                print(f"   ⚠️  Failed to verify registration: {e}")
                raise e

            print(f"✅ Kubernetes Worker registered successfully!")
            print(f"   Service ID: {service_config['id']}")
            print(f"   Supported types: {worker.supported_types}")
            print(f"   Visibility: {args.visibility}")
            print(f"   Namespace: {args.namespace}")
            print(f"")
            print(f"Worker is ready to process Kubernetes pod requests...")
            print(f"Press Ctrl+C to stop the worker.")

            # Keep the worker running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n🛑 Shutting down Kubernetes Worker...")
                await worker.shutdown()
                print(f"✅ Worker shutdown complete.")

        except WorkerNotAvailableError as e:
            print(f"❌ Kubernetes worker not available: {e}", file=sys.stderr)
            print(f"   Make sure kubectl is configured and you have access to the cluster.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to start Kubernetes Worker: {e}", file=sys.stderr)
            sys.exit(1)

    # Run the worker
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()