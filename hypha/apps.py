import httpx
import json
import logging
import os
import sys
import multihash
import asyncio
import logging
import yaml
import sys
import time
from pathlib import Path
import base64

from hypha import hypha_rpc_version
from jinja2 import Environment, PackageLoader, select_autoescape
from typing import Any, Dict, List, Optional, Union
from hypha.core import UserInfo, UserPermission, ServiceInfo, ApplicationManifest, AutoscalingConfig, RedisRPCConnection
from hypha.utils import (
    random_id,
)
import base58
import random
from hypha.plugin_parser import extract_files_from_source
from hypha.core import WorkspaceInfo
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("apps")
logger.setLevel(LOGLEVEL)

multihash.CodecReg.register("base58", base58.b58encode, base58.b58decode)


def merge_startup_config(manifest_config: dict, **kwargs) -> dict:
    """
    Merge startup configuration with proper priority.
    
    Priority order (highest to lowest):
    1. Function kwargs (current call parameters)
    2. Manifest startup_config (stored configuration)
    
    Args:
        manifest_config: The startup_config from the manifest (can be None)
        **kwargs: Direct parameters from function calls
        
    Returns:
        Merged startup configuration dictionary
    """
    # Start with manifest config (lowest priority)
    merged_config = (manifest_config or {}).copy()
    
    # Known startup_config fields that should be processed
    # Note: progress_callback is excluded from stored config as it can't be JSON serialized
    startup_config_fields = [
        'timeout', 'wait_for_service', 'additional_kwargs', 
        'stop_after_inactive'
    ]
    
    # Override with any provided kwargs (highest priority)
    for field in startup_config_fields:
        if field in kwargs and kwargs[field] is not None:
            merged_config[field] = kwargs[field]
    
    return merged_config


def merge_startup_config_with_runtime(manifest_config: dict, **kwargs) -> dict:
    """
    Merge startup configuration including runtime-only fields like progress_callback.
    
    This version includes fields that are only used during execution and not stored.
    
    Args:
        manifest_config: The startup_config from the manifest (can be None)
        **kwargs: Direct parameters from function calls
        
    Returns:
        Merged startup configuration dictionary including runtime fields
    """
    # Start with the stored config
    merged_config = merge_startup_config(manifest_config, **kwargs)
    
    # Add runtime-only fields that don't get stored
    if 'progress_callback' in kwargs and kwargs['progress_callback'] is not None:
        merged_config['progress_callback'] = kwargs['progress_callback']
    
    return merged_config


def extract_startup_config_kwargs(**kwargs) -> dict:
    """
    Extract startup_config related kwargs from function parameters.
    
    Returns:
        Dictionary containing only startup_config related parameters (excluding runtime-only fields)
    """
    startup_config_fields = [
        'timeout', 'wait_for_service', 'additional_kwargs', 
        'stop_after_inactive'
    ]
    
    return {
        field: kwargs[field] 
        for field in startup_config_fields 
        if field in kwargs and kwargs[field] is not None
    }


def extract_all_startup_config_kwargs(**kwargs) -> dict:
    """
    Extract all startup_config related kwargs including runtime-only fields.
    
    Returns:
        Dictionary containing all startup_config related parameters including progress_callback
    """
    startup_config_fields = [
        'timeout', 'wait_for_service', 'additional_kwargs', 
        'stop_after_inactive', 'progress_callback'
    ]
    
    return {
        field: kwargs[field] 
        for field in startup_config_fields 
        if field in kwargs and kwargs[field] is not None
    }


class AutoscalingManager:
    """Manages autoscaling for applications based on client load.
    
    This class provides automatic scaling functionality for server applications.
    It monitors the load on application instances and automatically scales them
    up or down based on configurable thresholds.
    
    Key features:
    - Monitors request rate (requests per minute) for load balancing enabled clients
    - Scales up when load exceeds scale_up_threshold * target_requests_per_instance
    - Scales down when load falls below scale_down_threshold * target_requests_per_instance
    - Respects min_instances and max_instances limits
    - Uses cooldown periods to prevent rapid scaling oscillations
    - Automatically stops autoscaling when the last instance of an app is removed
    
    See docs/autoscaling.md for detailed documentation and examples.
    """
    
    def __init__(self, app_controller):
        self.app_controller = app_controller
        self._autoscaling_tasks = {}  # app_id -> asyncio.Task
        self._scaling_locks = {}  # app_id -> asyncio.Lock
        self._last_scale_time = {}  # app_id -> {scale_up: timestamp, scale_down: timestamp}
        
    async def start_autoscaling(self, app_id: str, autoscaling_config: AutoscalingConfig, context: dict):
        """Start autoscaling monitoring for an application."""
        if not autoscaling_config.enabled:
            return
            
        if app_id in self._autoscaling_tasks:
            return  # Already monitoring
            
        self._scaling_locks[app_id] = asyncio.Lock()
        self._last_scale_time[app_id] = {"scale_up": 0, "scale_down": 0}
        
        # Start monitoring task
        task = asyncio.create_task(self._monitor_app_load(app_id, autoscaling_config, context))
        self._autoscaling_tasks[app_id] = task
        logger.info(f"Started autoscaling monitoring for app {app_id}")
    
    async def stop_autoscaling(self, app_id: str):
        """Stop autoscaling monitoring for an application."""
        if app_id in self._autoscaling_tasks:
            task = self._autoscaling_tasks.pop(app_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
        self._scaling_locks.pop(app_id, None)
        self._last_scale_time.pop(app_id, None)
        logger.info(f"Stopped autoscaling monitoring for app {app_id}")
    
    async def _monitor_app_load(self, app_id: str, config: AutoscalingConfig, context: dict):
        """Monitor load for an application and scale instances as needed."""
        try:
            while True:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                async with self._scaling_locks[app_id]:
                    await self._check_and_scale(app_id, config, context)
                    
        except asyncio.CancelledError:
            logger.info(f"Autoscaling monitoring cancelled for app {app_id}")
            raise
        except Exception as e:
            logger.error(f"Error in autoscaling monitoring for app {app_id}: {e}")
    
    async def _check_and_scale(self, app_id: str, config: AutoscalingConfig, context: dict):
        """Check current load and scale instances if needed."""
        try:
            # Get current instances for this app
            current_instances = self._get_app_instances(app_id)
            current_count = len(current_instances)
            
            if current_count == 0:
                return  # No instances to monitor
            
            # Calculate average load across all instances
            total_load = 0
            active_instances = 0
            workspace = context["ws"]
            
            for session_id, session_info in current_instances.items():
                client_id = session_info.get("id", "").split("/")[-1]
                if client_id.endswith("__rlb"):  # Only consider load balancing enabled clients
                    load = RedisRPCConnection.get_client_load(workspace, client_id)
                    total_load += load
                    active_instances += 1
            
            if active_instances == 0:
                return  # No load balancing enabled instances
            
            average_load = total_load / active_instances
            current_time = time.time()
            
            # Check if we need to scale up
            if (average_load > config.scale_up_threshold * config.target_requests_per_instance and
                current_count < config.max_instances):
                
                # Check cooldown period
                if current_time - self._last_scale_time[app_id]["scale_up"] > config.scale_up_cooldown:
                    await self._scale_up(app_id, context)
                    self._last_scale_time[app_id]["scale_up"] = current_time
                    logger.info(f"Scaled up app {app_id} due to high load: {average_load:.2f}")
            
            # Check if we need to scale down
            elif (average_load < config.scale_down_threshold * config.target_requests_per_instance and
                  current_count > config.min_instances):
                
                # Check cooldown period
                if current_time - self._last_scale_time[app_id]["scale_down"] > config.scale_down_cooldown:
                    await self._scale_down(app_id, context)
                    self._last_scale_time[app_id]["scale_down"] = current_time
                    logger.info(f"Scaled down app {app_id} due to low load: {average_load:.2f}")
                    
        except Exception as e:
            logger.error(f"Error checking and scaling app {app_id}: {e}")
    
    def _get_app_instances(self, app_id: str) -> Dict[str, dict]:
        """Get all running instances for a specific app."""
        instances = {}
        for session_id, session_info in self.app_controller._sessions.items():
            if session_info.get("app_id") == app_id:
                instances[session_id] = session_info
        return instances
    
    async def _scale_up(self, app_id: str, context: dict):
        """Scale up by starting a new instance."""
        try:
            # Start a new instance with load balancing enabled
            await self.app_controller.start(
                app_id=app_id,
                context=context
            )
            logger.info(f"Successfully scaled up app {app_id}")
        except Exception as e:
            logger.error(f"Failed to scale up app {app_id}: {e}")
    
    async def _scale_down(self, app_id: str, context: dict):
        """Scale down by stopping the least loaded instance."""
        try:
            instances = self._get_app_instances(app_id)
            if len(instances) <= 1:
                return  # Don't scale down if only one instance
            
            # Find the instance with the lowest load
            min_load = float('inf')
            least_loaded_session = None
            workspace = context["ws"]
            
            for session_id, session_info in instances.items():
                client_id = session_info.get("id", "").split("/")[-1]
                if client_id.endswith("__rlb"):
                    load = RedisRPCConnection.get_client_load(workspace, client_id)
                    if load < min_load:
                        min_load = load
                        least_loaded_session = session_id
            
            if least_loaded_session:
                await self.app_controller._stop(least_loaded_session, raise_exception=False, context=None)
                logger.info(f"Successfully scaled down app {app_id} by stopping {least_loaded_session}")
                
        except Exception as e:
            logger.error(f"Failed to scale down app {app_id}: {e}")


# Add a helper function to detect raw HTML content
def is_raw_html_content(source: str) -> bool:
    """Check if source is raw HTML content (not a URL or ImJoy/Hypha template)."""
    if not source or source.startswith("http"):
        return False

    # Check for basic HTML structure
    source_lower = source.lower().strip()
    return source_lower.startswith(("<!doctype html", "<html")) 


class ServerAppController:
    """Server App Controller."""

    def __init__(
        self,
        store,
        in_docker,
        port: int,
        artifact_manager,
    ):
        """Initialize the controller."""
        self.port = int(port)
        self.store = store
        self.in_docker = in_docker
        self.artifact_manager = artifact_manager
        self._sessions = {}  # Track running sessions
        self.event_bus = store.get_event_bus()
        self.local_base_url = store.local_base_url
        self.public_base_url = store.public_base_url
        store.register_public_service(self.get_service_api())
        self.jinja_env = Environment(
            loader=PackageLoader("hypha"), autoescape=select_autoescape()
        )
        self.templates_dir = Path(__file__).parent / "templates"
        self.autoscaling_manager = AutoscalingManager(self)

        def shutdown(_) -> None:
            asyncio.ensure_future(self.shutdown())

        self.event_bus.on_local("shutdown", shutdown)

        async def client_disconnected(info: dict) -> None:
            """Handle client disconnected event."""
            # {"id": client_id, "workspace": ws}
            client_id = info["id"]
            full_client_id = info["workspace"] + "/" + client_id
            if full_client_id in self._sessions:
                app_info = self._sessions.pop(full_client_id, None)
                try:
                    # Create context for worker call
                    context = {
                        "ws": info["workspace"],
                        "user": self.store.get_root_user().model_dump()
                    }
                    await app_info["_worker"].stop(full_client_id, context=context)
                except Exception as exp:
                    logger.warning(f"Failed to stop browser tab: {exp}")

        self.event_bus.on_local("client_disconnected", client_disconnected)
        store.set_server_app_controller(self)

    async def get_server_app_workers(self, app_type: str = None, context: dict = None, random_select: bool = False):
        workspace = context.get("ws") if context else None
        
        # Get workspace service info first (fast, no get_service calls yet)
        workspace_svcs = []
        if context:
            try:
                workspace_svcs = await self.store._workspace_manager.list_services({"type": "server-app-worker"}, context=context)
                logger.info(f"Found {len(workspace_svcs)} server-app-worker services in workspace {workspace}: {[svc['id'] for svc in workspace_svcs]}")
            except Exception as e:
                logger.warning(f"Failed to get workspace workers: {e}")
        
        # Filter workspace services by app_type if specified
        if app_type and workspace_svcs:
            filtered_workspace_svcs = []
            for svc in workspace_svcs:
                try:
                    # Get the full service info to access supported_types
                    workspace_server = await self.store.get_workspace_interface(
                        context.get("user"), context.get("ws"), context.get("from")
                    )
                    full_svc = await workspace_server.get_service(svc['id'])
                    supported_types = full_svc.get("supported_types", [])
                    if app_type in supported_types:
                        filtered_workspace_svcs.append(svc)
                        logger.info(f"Workspace service {svc['id']} supports app_type {app_type}")
                except Exception as e:
                    logger.warning(f"Failed to get full workspace service info for {svc['id']}: {e}")
                    # Skip services we can't get info for
                    continue
            
            # If we found workspace services for this app type, use them
            if filtered_workspace_svcs:
                logger.info(f"Found {len(filtered_workspace_svcs)} workspace services for app_type {app_type}")
                selected_svcs = filtered_workspace_svcs
                use_workspace = True
            else:
                selected_svcs = []
                use_workspace = False
        elif workspace_svcs:
            # If no app_type specified, use all workspace services
            logger.info(f"Found {len(workspace_svcs)} workspace services (no app_type filter)")
            selected_svcs = workspace_svcs
            use_workspace = True
        else:
            selected_svcs = []
            use_workspace = False
        
        # Fallback to public workers if no workspace workers found
        if not selected_svcs:
            server = await self.store.get_public_api()
            public_svcs = await server.list_services({"type": "server-app-worker"})
            logger.info(f"Found {len(public_svcs)} server-app-worker services in public workspace: {[svc['id'] for svc in public_svcs]}")
            
            if not public_svcs:
                logger.warning("No public workers found either")
                return [] if not random_select else None
            
            # Filter public services by app_type if specified
            if app_type:
                filtered_public_svcs = []
                for svc in public_svcs:
                    try:
                        # Get the full service info to access supported_types
                        full_svc = await server.get_service(svc['id'])
                        supported_types = full_svc.get("supported_types", [])
                        if app_type in supported_types:
                            filtered_public_svcs.append(svc)
                            logger.info(f"Public service {svc['id']} supports app_type {app_type}")
                    except Exception as e:
                        logger.warning(f"Failed to get full service info for {svc['id']}: {e}")
                        # Skip services we can't get info for
                        continue
                
                logger.info(f"After filtering public services for app_type {app_type}: {len(filtered_public_svcs)} services selected")
                selected_svcs = filtered_public_svcs
            else:
                selected_svcs = public_svcs
            
            use_workspace = False
        
        if not selected_svcs:
            return [] if not random_select else None
        
        # Random selection if requested
        if random_select:
            selected_svc = random.choice(selected_svcs)
            logger.info(f"Randomly selected service: {selected_svc['id']}")
            selected_svcs = [selected_svc]
        
        # Now get the actual worker objects (slow operation, but only for selected services)
        workers = []
        for svc in selected_svcs:
            try:
                if use_workspace:
                    # For workspace services, get through workspace interface
                    user_info = UserInfo.model_validate(context["user"])
                    async with self.store.get_workspace_interface(user_info, workspace) as ws:
                        worker = await ws.get_service(svc["id"])
                else:
                    # For public services, use public API
                    worker = await server.get_service(svc["id"])
                workers.append(worker)
            except Exception as e:
                logger.warning(f"Failed to get worker service {svc['id']}: {e}")
        
        if random_select:
            return workers[0] if workers else None
        else:
            return workers


    async def setup_applications_collection(self, overwrite=True, context=None):
        """Set up the workspace."""
        ws = context["ws"]
        # Create an collection in the workspace
        manifest = {
            "id": "applications",
            "name": "Applications",
            "description": f"A collection of applications for workspace {ws}",
        }
        collection = await self.artifact_manager.create(
            type="collection",
            alias="applications",
            manifest=manifest,
            overwrite=overwrite,
            context=context,
        )
        logger.info(f"Applications collection created for workspace {ws}")
        return collection["id"]

    @schema_method
    async def install(
        self,
        source: Optional[str] = Field(
            None,
            description="The source code of the application, URL to fetch the source, or None if using config. Can be raw HTML content, ImJoy/Hypha plugin code, or a URL to download the source from. URLs must be HTTPS or localhost/127.0.0.1."
        ),
        manifest: Optional[Dict[str, Any]] = Field(
            None,
            description="Application manifest dictionary containing app metadata and settings. Can include autoscaling config. MUTUALLY EXCLUSIVE with 'config' parameter - when manifest is provided, config must be None. The manifest is stored directly without conversion and must include 'entry_point' field."
        ),
        app_id: Optional[str] = Field(
            None,
            description="The unique identifier of the application to install. This is typically the alias of the application."
        ),
        config: Optional[Dict[str, Any]] = Field(
            None,
            description="DEPRECATED: Use 'manifest' instead. Application configuration dictionary (alias for manifest for backward compatibility)."
        ),
        files: Optional[List[Dict[str, Any]]] = Field(
            None,
            description="List of files to include in the application artifact. Each file should be a dictionary with 'name' (file path), 'content' (file content), and optional 'format' ('text', 'json', or 'base64', defaults to 'text'). For 'json' format, content can be a dictionary that gets JSON serialized. For 'base64' format, content will be decoded from base64 or data URL format and stored as binary data."
        ),
        stop_after_inactive: Optional[int] = Field(
            None,
            description="Number of seconds to wait before stopping the app due to inactivity. If not provided, uses app configuration."
        ),
        wait_for_service: Optional[str] = Field(
            None,
            description="The service to wait for before installing the app. If not provided, the app will be installed without waiting for any service, set to `False` to disable it, if not set, the app will wait for the `default` service"
        ),
        workspace: Optional[str] = Field(
            None,
            description="Target workspace for installation. If not provided, uses the current workspace from context."
        ),
        overwrite: Optional[bool] = Field(
            False,
            description="Whether to overwrite existing app with same name. Set to True to replace existing installations."
        ),
        timeout: Optional[float] = Field(
            None,
            description="Maximum time to wait for installation completion in seconds. Increase for complex apps that take longer to start."
        ),
        version: Optional[str] = Field(
            None,
            description="Version identifier for the app. If not provided, uses default versioning."
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to install the app in stage mode. If True, the app will be installed as a staged artifact that can be discarded or committed later."
        ),
        progress_callback: Any = Field(
            None,
            description="Callback function to receive progress updates from the app."
        ),
        additional_kwargs: Optional[dict] = Field(
            None,
            description="Additional keyword arguments to pass to the app worker."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> str:
        """Save a server app.

        Args:
            source: The source code of the application, URL to fetch the source, or None if using manifest
            manifest: Application manifest dictionary containing app metadata and settings.
                     The manifest is stored directly without conversion and must include 'entry_point' field.
            files: List of files to include in the application artifact. Each file should be a dictionary 
                   with 'name' (file path), 'content' (file content), and optional 'format' ('text', 'json', or 'base64', defaults to 'text'). 
                   For 'json' format, content can be a dictionary. For 'base64' format, supports data URL format.
            workspace: Target workspace for installation (defaults to current workspace)
            overwrite: Whether to overwrite existing app with same name
            timeout: Maximum time to wait for installation completion
            version: Version identifier for the app
            context: Additional context information
        """
        if not workspace:
            workspace = context["ws"]
            
        progress_callback = progress_callback or (lambda x: None)
        progress_callback({
            "type": "info",
            "message": "Parsing app manifest...",
        })

        user_info = UserInfo.model_validate(context["user"])
        assert not user_info.is_anonymous, "Anonymous users cannot install apps"
        workspace_info = await self.store.get_workspace_info(workspace, load=True)
        assert workspace_info, f"Workspace {workspace} not found."
        if not user_info.check_permission(workspace_info.id, UserPermission.read_write):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to install apps in workspace {workspace_info.id}"
            )

        # Handle backward compatibility: config is an alias for manifest
        if config is not None and manifest is not None:
            raise ValueError("Cannot specify both 'config' and 'manifest' parameters. Use 'manifest' (config is deprecated).")
        if config is not None:
            manifest = config
        
        # Simplified installation: Convert source to files and use worker compilation
        
        # Initialize artifact object from manifest or create default
        if manifest:
            artifact_obj = manifest.copy()
        else:
            artifact_obj = {
                "name": "Untitled App",
                "version": "0.1.0",
                "type": "hypha"  # Default type
            }
        
        if files:
            # make sure all files have path and name
            for file in files:
                if "path" not in file and "name" not in file:
                    raise ValueError(f"File {file} must have 'name' or 'path' field")
                if "path" not in file:
                    file["path"] = file["name"]
                if "name" not in file:
                    file["name"] = file["path"].split("/")[-1]
        
        # Compute source hash if source exists
        mhash = None
        if source:
            mhash = multihash.digest(source.encode("utf-8"), "sha2-256")
            mhash = mhash.encode("base58").decode("ascii")
            # Verify the source code if hash provided
            if "source_hash" in artifact_obj:
                source_hash = artifact_obj["source_hash"]
                target_mhash = multihash.decode(source_hash.encode("ascii"), "base58")
                assert target_mhash.verify(
                    source.encode("utf-8")
                ), f"App source code verification failed (source_hash: {source_hash})."
            artifact_obj["source_hash"] = mhash
        
        # Handle URL downloads
        if source and source.startswith("http"):
            if not (
                source.startswith("https://")
                or source.startswith("http://localhost")
                or source.startswith("http://127.0.0.1")
            ):
                raise Exception("Only secured https urls are allowed: " + source)
            
            # Download source for .imjoy.html and .hypha.html files
            if (source.startswith("https://") and 
                (source.split("?")[0].endswith(".imjoy.html") or 
                 source.split("?")[0].endswith(".hypha.html"))):
                async with httpx.AsyncClient() as client:
                    response = await client.get(source)
                    assert response.status_code == 200, f"Failed to download {source}"
                    source = response.text
            else:
                # For other HTTP sources, treat as external URLs
                artifact_obj["entry_point"] = source.split("?")[0].split("/")[-1]
                source = None
        
        # Handle raw HTML content - convert type to window if needed
        elif source and is_raw_html_content(source):
            if artifact_obj.get("type") == "hypha":
                artifact_obj["type"] = "window"
            # Set default name if not provided or is default "Untitled App"
            if artifact_obj.get("name") in [None, "Untitled App"]:
                artifact_obj["name"] = "Raw HTML App"
            artifact_obj["entry_point"] = "index.html"

        # Convert source to files list with enhanced XML parsing
        app_files = []
        if source and source.strip().startswith("<"):
            # Try to extract files from XML source (ImJoy/Hypha format)
            extracted_files, remaining_source = extract_files_from_source(source)
            # let's load config.json/yaml and update the artifact_obj
            for file in extracted_files:
                if file["name"] == 'manifest.json':
                    artifact_obj.update(file["content"])
            # now remove the manifest.json from the extracted_files
            extracted_files = [file for file in extracted_files if file["name"] != 'manifest.json']
            
            # Add extracted files (config.json/yaml, script.js/py, <file> tags)
            app_files.extend(extracted_files)
            
            # If there's remaining source content, add it as source file
            if remaining_source and remaining_source.strip():
                # Use entry_point from manifest if available, otherwise default to "source"
                source_file_path = artifact_obj.get("entry_point", "source")
                app_files.append({
                    "path": source_file_path,
                    "content": remaining_source,
                    "format": "text"
                })
        elif source:
            # Use entry_point from manifest if available, otherwise default to "source"
            source_file_path = artifact_obj.get("entry_point", "source")
            app_files.append({
                "path": source_file_path,
                "content": source,
                "format": "text"
            })
        
        # Add any additional files provided
        if files:
            app_files.extend(files)
        
        # Always try to get a worker for compilation - this ensures all browser apps are handled consistently
        app_type = artifact_obj.get("type", "hypha")
        worker = None
        
        if app_type in ["application", None]:
            raise ValueError("Application type should not be application or None")
        
        if app_type == "hypha":
            assert source is not None, "Source is missing"
            # Ensure source is in top level xml format with tags such as <config> <script>
            
        if "entry_point" not in artifact_obj:
            artifact_obj["entry_point"] = "source"
        
        # Try to get worker that supports this app type
        worker = await self.get_server_app_workers(app_type, context, random_select=True)
        if not worker:
            raise Exception(f"No server app worker found for app type: {app_type}")
        
        progress_callback({
            "type": "info",
            "message": f"Compiling app using worker {worker.id}...",
        })
        # If we have a worker, let it compile the manifest and files
        if worker and hasattr(worker, 'compile'):
            try:
                # Construct compilation config with server URLs and workspace info
                compile_config = {
                    "server_url": self.local_base_url or f"http://127.0.0.1:{self.port}",
                    "public_url": self.public_base_url or f"http://127.0.0.1:{self.port}",
                    "workspace": context["ws"],
                    "user": context["user"],
                    "progress_callback": progress_callback,
                }
                
                compiled_manifest, app_files = await worker.compile(artifact_obj, app_files, config=compile_config, context=context)
                # merge the compiled manifest into the artifact_obj
                artifact_obj.update(compiled_manifest)

                logger.info(f"Worker compiled app with type {app_type}")
            except Exception as e:
                raise Exception(f"Worker compilation failed: {e}")
        elif app_type in ["window", "iframe", "web-python", "web-worker", "hypha"]:
            # All browser-based apps should have a worker available
            raise Exception(f"No worker available for browser app type: {app_type}")

        progress_callback({
            "type": "info",
            "message": "Creating application artifact...",
        })
        # Store startup_context with workspace and user info from installation time
        artifact_obj["startup_context"] = {
            "ws": context["ws"],
            "user": context["user"]
        }
        
        # Merge startup_config with proper priority: kwargs > manifest
        # Note: progress_callback is excluded from stored config as it's not serializable
        startup_config_kwargs = extract_startup_config_kwargs(
            timeout=timeout,
            wait_for_service=wait_for_service,
            additional_kwargs=additional_kwargs,
            stop_after_inactive=stop_after_inactive
        )
        
        existing_startup_config = artifact_obj.get("startup_config", {})
        merged_startup_config = merge_startup_config(existing_startup_config, **startup_config_kwargs)
        
        if merged_startup_config:
            artifact_obj["startup_config"] = merged_startup_config
        
        if mhash:
            # Store source hash for singleton checking
            artifact_obj["source_hash"] = mhash
        
        assert artifact_obj.get("type") not in ["application", None], "Application type should not be application or None"

        ApplicationManifest.model_validate(artifact_obj)

        try:
            artifact = await self.artifact_manager.read("applications", context=context)
            collection_id = artifact["id"]
        except KeyError:
            collection_id = await self.setup_applications_collection(
                overwrite=True, context=context
            )

        # Create artifact using the artifact controller - let it generate the alias
        artifact = await self.artifact_manager.create(
            type="application",
            alias=app_id,
            parent_id=collection_id,
            manifest=artifact_obj,
            overwrite=overwrite,
            version="stage",
            context=context,
        )
        
        # Now get the app_id from the artifact alias
        app_id = artifact["alias"]
        
        # Update the artifact object with the correct app_id
        artifact_obj["id"] = app_id
        
        # Update the artifact with the compiled manifest
        await self.artifact_manager.edit(
            artifact["id"],
            stage=True,
            manifest=artifact_obj,
            context=context,
        )
        
        # Upload all files  
        
        for file_info in app_files:
            if "path" not in file_info and "name" not in file_info:
                raise ValueError(f"File {file_info} must have 'name' or 'path' field")
            if "path" not in file_info:
                file_info["path"] = file_info["name"]
            if "name" not in file_info:
                file_info["name"] = file_info["path"].split("/")[-1]
            file_path = file_info["path"]
            file_content = file_info.get("content")
            file_format = file_info.get("format", "text")
            
            if not file_path or file_content is None:
                raise Exception("Each file must have 'name' and 'content' fields")
            
            # Handle different file formats
            if file_format == "base64":
                try:
                    # Check if content is in data URL format (e.g., "data:image/png;base64,...")
                    if isinstance(file_content, str) and file_content.startswith("data:"):
                        # Parse data URL format: data:[mediatype][;base64],<data>
                        if ";base64," in file_content:
                            # Extract base64 part after the comma
                            base64_content = file_content.split(";base64,", 1)[1]
                        else:
                            raise Exception(f"Data URL format not supported for file {file_path}. Expected format: data:mediatype;base64,content")
                    else:
                        # Direct base64 content
                        base64_content = file_content
                    
                    file_data = base64.b64decode(base64_content)
                except Exception as e:
                    raise Exception(f"Failed to decode base64 content for file {file_path}: {e}")
            elif file_format == "json":
                try:
                    if isinstance(file_content, (dict, list)):
                        # Serialize dictionary or list to JSON string
                        json_string = json.dumps(file_content, indent=2)
                    elif isinstance(file_content, str):
                        # Validate that it's valid JSON if it's already a string
                        json.loads(file_content)  # This will raise exception if invalid
                        json_string = file_content
                    else:
                        raise Exception(f"JSON content must be a dictionary, list, or valid JSON string")
                    
                    file_data = json_string.encode('utf-8')
                except Exception as e:
                    raise Exception(f"Failed to process JSON content for file {file_path}: {e}")
            elif file_format == "text":
                file_data = file_content.encode('utf-8') if isinstance(file_content, str) else file_content
            else:
                raise Exception(f"Unsupported file format '{file_format}' for file {file_path}. Must be 'text', 'json', or 'base64'")
            
            # Upload the file to the artifact
            put_url = await self.artifact_manager.put_file(
                artifact["id"], file_path=file_path, use_proxy=False, context=context
            )
            async with httpx.AsyncClient() as client:
                response = await client.put(put_url, data=file_data)
                assert response.status_code == 200, f"Failed to upload file {file_path}"

        if not stage:
            progress_callback({
                "type": "info",
                "message": "Committing application artifact...",
            })
            # Commit the artifact if stage is not enabled
            await self.commit_app(
                app_id,
                version=version,
                context=context,
                progress_callback=progress_callback,
                **startup_config_kwargs
            )
            # After commit, read the updated artifact to get the collected services
            updated_artifact_info = await self.artifact_manager.read(
                app_id, version=version, context=context
            )
            progress_callback({
                "type": "success",
                "message": "Installation complete!",
            })
            return updated_artifact_info.get("manifest", artifact_obj)
        progress_callback({
            "type": "success",
            "message": "Installation complete!",
        })
        return artifact_obj

    @schema_method
    async def edit_file(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to edit. This is typically the alias of the application."
        ),
        file_path: str = Field(
            ...,
            description="The path of the file to edit. Use forward slashes for path separators (e.g., 'src/main.js', 'index.html')."
        ),
        file_content: str = Field(
            ...,
            description="The new content for the file. This will completely replace the existing file content."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """Add a file to the installed application."""
        put_url = await self.artifact_manager.put_file(
            app_id, file_path=file_path, use_proxy=False, context=context
        )
        response = httpx.put(put_url, data=file_content)
        assert response.status_code == 200, f"Failed to upload {file_path} to {app_id}"

    @schema_method
    async def remove_file(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to modify. This is typically the alias of the application."
        ),
        file_path: str = Field(
            ...,
            description="The path of the file to remove from the application. Use forward slashes for path separators (e.g., 'src/main.js', 'index.html')."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """Remove a file from the installed application."""
        await self.artifact_manager.remove_file(
            app_id, file_path=file_path, context=context
        )

    @schema_method
    async def list_files(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to inspect. This is typically the alias of the application."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> List[dict]:
        """List files of an installed application."""
        return await self.artifact_manager.list_files(
            app_id, context=context
        )

    @schema_method
    async def commit_app(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to commit. This is typically the alias of the application."
        ),
        timeout: int = Field(
            None,
            description="Maximum time to wait for commit completion in seconds. If not provided, uses app configuration. Increase for complex apps that take longer to verify."
        ),
        version: str = Field(
            None,
            description="Version identifier for the committed app. If not provided, uses default versioning."
        ),
        stop_after_inactive: Optional[int] = Field(
            None,
            description="Number of seconds to wait before stopping the app due to inactivity. If not provided, uses app configuration."
        ),
        wait_for_service: Optional[str] = Field(
            None,
            description="The service to wait for before committing the app. If not provided, the app will be committed without waiting for any service."
        ),
        additional_kwargs: Optional[dict] = Field(
            None,
            description="Additional keyword arguments to pass to the app worker."
        ),
        progress_callback: Any = Field(
            None,
            description="Callback function to receive progress updates from the app."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """Finalize the edits to the application by committing the artifact."""
        try:
            # Read the manifest to check if it's a daemon app
            artifact_info = await self.artifact_manager.read(
                app_id, version="stage", context=context
            )
            manifest = artifact_info.get("manifest", {})
            manifest = ApplicationManifest.model_validate(manifest)

            # Merge startup_config with proper priority: kwargs > manifest
            startup_config_kwargs = extract_startup_config_kwargs(
                timeout=timeout,
                wait_for_service=wait_for_service,
                additional_kwargs=additional_kwargs,
                stop_after_inactive=stop_after_inactive
            )
            
            merged_startup_config = merge_startup_config(manifest.startup_config, **startup_config_kwargs)
            
            # Update the manifest with the merged startup_config
            manifest.startup_config = merged_startup_config
            
            # Save the updated manifest back to the artifact
            await self.artifact_manager.edit(
                app_id,
                version="stage",
                stage=True,
                manifest=manifest.model_dump(mode="json"),
                context=context,
            )

            # Include progress_callback with the merged config for the start call
            start_config = merged_startup_config.copy()
            if progress_callback is not None:
                start_config['progress_callback'] = progress_callback
                
            info = await self.start(
                app_id,
                version="stage",
                context=context,
                **start_config
            )
            await self.stop(info["id"], context=context)

            # After verification, read the updated manifest that includes collected services
            updated_artifact_info = await self.artifact_manager.read(
                app_id, version="stage", context=context
            )

        except Exception as exp:
            logger.error("Failed to start the app: %s during installation, error: %s", app_id, exp)
            await self.uninstall(app_id, context=context)
            # Extract core error without chaining traceback
            core_error = self._extract_core_error(exp)
            raise Exception(f"Failed to install app '{app_id}': {core_error}") from None
        await self.artifact_manager.commit(
            app_id, version=version, context=context
        )
        # After commit, read the updated artifact to get the collected services
        updated_artifact_info = await self.artifact_manager.read(
            app_id, version=version, context=context
        )
        return updated_artifact_info.get("manifest", {})

    @schema_method
    async def uninstall(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to uninstall. This is typically the alias of the application."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> None:
        """Uninstall an application by removing its artifact."""
        # Check if the artifact is in stage mode
        try:
            await self.artifact_manager.read(app_id, version="stage", context=context)
            # If we can read the stage version, it means the artifact is in stage mode
            # Use discard instead of delete to revert to the last committed state
            await self.artifact_manager.discard(app_id, context=context)
        except Exception:
            # If we can't read the stage version, the artifact is committed
            # Use delete as usual
            await self.artifact_manager.delete(app_id, context=context)

        # stop all the instances of the app
        for app in self._sessions.values():
            if app["id"] == app_id:
                await self.stop(app["id"], raise_exception=False)

    async def start_by_type(
        self,
        app_id: str,
        app_type: str,
        client_id: str,
        server_url: str,
        workspace: str,
        version: str = None,
        token: str = None,
        timeout: float = None,
        manifest: dict = None,
        progress_callback: Any = None,
        additional_kwargs: Optional[dict] = None,
        context: dict = None,
    ):
        """Start the app by type using the appropriate worker."""
        progress_callback({
            "type": "info",
            "message": f"Initializing {app_type} worker..."
        })
            
        # Get a random worker that supports this app type
        worker = await self.get_server_app_workers(app_type, context, random_select=True)
        if not worker:
            raise Exception(f"No server app worker found for type: {app_type}")
        
        progress_callback({
            "type": "info",
            "message": f"Starting {app_type} application..."
        })
        
        # Get entry point from manifest
        entry_point = manifest.entry_point
        
        # Start the app using the worker with reorganized config
        full_client_id = workspace + "/" + client_id
        additional_kwargs = additional_kwargs or {}
        session_id = await worker.start({
            "id": full_client_id,
            "app_id": app_id,
            "workspace": workspace,
            "client_id": client_id,
            "server_url": server_url,
            "app_files_base_url": f"{server_url}/{workspace}/artifacts/{app_id}/files",
            "token": token,
            "entry_point": entry_point,
            "artifact_id": f"{workspace}/{app_id}",
            "timeout": timeout,
            "manifest": manifest,
            "progress_callback": progress_callback,
            **additional_kwargs,
        }, context=context)
        
        # Store minimal session info for apps.py - worker handles detailed session management
        self._sessions[full_client_id] = {
            "id": full_client_id,
            "app_id": app_id,
            "workspace": workspace,
            "client_id": client_id,
            "app_type": app_type,
            "entry_point": entry_point,
            "name": manifest.name,
            "description": manifest.description,
            "version": version,
            "source_hash": manifest.source_hash if hasattr(manifest, 'source_hash') else None,
            "_worker": worker,
        }
        
        return session_id

    def _extract_core_error(self, error, logs=None):
        """Extract the core error message from worker logs or exception."""
        # First try to get meaningful error from worker logs
        if logs:
            try:
                if isinstance(logs, dict) and logs.get('error'):
                    error_list = logs['error']
                    if isinstance(error_list, list) and error_list:
                        return "\n".join(error_list)
                    else:
                        return str(error_list)
                elif isinstance(logs, str):
                    return logs
            except (KeyError, IndexError, TypeError, AttributeError):
                pass
        
        # Handle specific exception types
        if isinstance(error, asyncio.TimeoutError):
            return "Application startup timed out"
        
        # For other exceptions, just return the message without chaining
        return str(error)

    @schema_method
    async def start(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to start. This is typically the alias of the application."
        ),
        timeout: float = Field(
            None,
            description="Maximum time to wait for start completion in seconds. If not provided, uses default timeout from app configuration."
        ),
        version: str = Field(
            None,
            description="Version of the application to start. If not provided, uses the latest version."
        ),
        wait_for_service: Union[str, bool] = Field(
            None,
            description="Name of the service to wait for before considering the app started. If True, waits for 'default' service. If not provided, uses app configuration."
        ),
        stop_after_inactive: Optional[int] = Field(
            None,
            description="Number of seconds to wait before stopping the app due to inactivity. If not provided, uses app configuration."
        ),
        stage: bool = Field(
            False,
            description="Whether to start the app from stage mode. If True, starts from the staged version."
        ),
        progress_callback: Any = Field(
            None,
            description="Callback function to receive progress updates from the app."
        ),
        additional_kwargs: Optional[dict] = Field(
            None,
            description="Additional keyword arguments to pass to the app worker."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """Start the app and keep it alive."""
        
        progress_callback = progress_callback or (lambda x: None)

        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        # Add "__rlb" suffix to enable load balancing metrics for app clients
        # Add "_rapp_" prefix to identify app clients
        # This allows the system to track load only for clients that may have multiple instances
        client_id = "_rapp_" +random_id(readable=True) + "__rlb"

        async with self.store.get_workspace_interface(user_info, workspace) as ws:
            token = await ws.generate_token({"client_id": client_id})

        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to run app {app_id} in workspace {workspace}."
            )

        # If stage is True, use stage version, otherwise use provided version
        read_version = "stage" if stage else version
        artifact_info = await self.artifact_manager.read(
            app_id, version=read_version, context=context
        )
        manifest = artifact_info.get("manifest", {})
        manifest = ApplicationManifest.model_validate(manifest)

        if "disabled" in manifest and manifest["disabled"]:
            raise RuntimeError(f"App {app_id} is disabled")

        # Merge startup_config with proper priority: kwargs > manifest
        # Use all kwargs including runtime-only fields like progress_callback
        startup_config_kwargs = extract_all_startup_config_kwargs(
            timeout=timeout,
            wait_for_service=wait_for_service,
            additional_kwargs=additional_kwargs,
            stop_after_inactive=stop_after_inactive,
            progress_callback=progress_callback
        )
        
        final_startup_config = merge_startup_config_with_runtime(manifest.startup_config, **startup_config_kwargs)
        
        # Apply final values with defaults
        timeout = final_startup_config.get("timeout", timeout or 60)
        wait_for_service = final_startup_config.get("wait_for_service", wait_for_service)
        additional_kwargs = final_startup_config.get("additional_kwargs", additional_kwargs)
        stop_after_inactive = final_startup_config.get("stop_after_inactive", stop_after_inactive)

        # Convert True to "default" after startup_config is applied
        if wait_for_service is True or wait_for_service is None:
            wait_for_service = "default"
        if wait_for_service and ":" in wait_for_service:
            wait_for_service = wait_for_service.split(":")[1]

        detached = wait_for_service == False

        # Check if a non-singleton, non-daemon app is already running.
        # If so, return the existing session to prevent multiple instances
        # unless autoscaling is enabled.
        if not manifest.singleton and not manifest.daemon:
            if not (manifest.autoscaling and manifest.autoscaling.enabled):
                for session_info in self._sessions.values():
                    if session_info["app_id"] == app_id and session_info["workspace"] == workspace:
                        logger.info(f"Returning existing instance for app {app_id} in workspace {workspace}")
                        return session_info

        if manifest.singleton:
            # check if the app is already running
            for session_info in self._sessions.values():
                if session_info["app_id"] == app_id:
                    # For singleton apps, return the existing session instead of raising an error
                    return session_info
        if manifest.daemon and stop_after_inactive and stop_after_inactive > 0:
            raise ValueError("Daemon apps should not have stop_after_inactive set.")
        if stop_after_inactive is None:
            stop_after_inactive = (
                600
                if final_startup_config.get("stop_after_inactive") is None
                else final_startup_config.get("stop_after_inactive")
            )
        # Get app type from config, fallback to manifest type
        app_type = manifest.type if manifest else None
        if app_type in ["application", None]:
            raise ValueError("Application type should not be application or None")

        full_client_id = workspace + "/" + client_id

        # collecting services registered during the startup of the script
        collected_services: List[ServiceInfo] = []
        app_info = {
            "id": full_client_id,
            "app_id": app_id,
            "workspace": workspace,
            "client_id": client_id,
            "config": {},
            "session_id": None,  # Will be updated after start_by_type
        }

        # Only set up event waiting if not in detached mode
        event_future = None
        if not detached:
            # Create a future that will be set when the target event occurs
            event_future = asyncio.Future()
            
            def service_added(info: dict):
                logger.info(f"Service added: {info['id']}")
                if info["id"].startswith(full_client_id + ":"):
                    sinfo = ServiceInfo.model_validate(info)
                    collected_services.append(sinfo)
                    # Report service registration progress
                    service_name = info["id"].split(":")[-1]
                    progress_callback({
                        "type": "success",
                        "message": f"Service '{service_name}' registered successfully"
                    })
                        
                if info["id"] == full_client_id + ":default":
                    for key in ["config", "name", "description"]:
                        if info.get(key):
                            app_info[key] = info[key]
                    progress_callback({
                        "type": "success",
                        "message": "Default service configured"
                    })
                
                # Check if this is the target service we're waiting for
                if wait_for_service and isinstance(wait_for_service, str) and info["id"] == full_client_id + ":" + wait_for_service:
                    if not event_future.done():
                        progress_callback({
                            "type": "success",
                            "message": f"Target service '{wait_for_service}' found"
                        })
                        event_future.set_result(info)
                        logger.info(f"Target service found: {info['id']}")

            def client_connected(info: dict):
                logger.info(f"Client connected: {info}")
                progress_callback({
                    "type": "success",
                    "message": "Client connection established"
                })
                # Check if this is the target client we're waiting for
                if not wait_for_service and info["id"] == full_client_id:
                    if not event_future.done():
                        progress_callback({
                            "type": "success",
                            "message": "Application ready"
                        })
                        event_future.set_result(info)
                        logger.info(f"Target client connected: {info['id']}")

            # Set up event callbacks BEFORE starting the app to avoid timing issues
            self.event_bus.on_local("service_added", service_added)
            if not wait_for_service:
                self.event_bus.on_local("client_connected", client_connected)

        try:
            # Start the app using the new start_by_type function
            server_url = self.local_base_url or f"http://127.0.0.1:{self.port}"
            logger.debug(f"Starting app with server_url: {server_url} (local_base_url: {self.local_base_url}, port: {self.port})")
            session_id = await self.start_by_type(
                app_id=app_id,
                app_type=app_type,
                client_id=client_id,
                server_url=server_url,
                workspace=workspace,
                version=version,
                token=token,
                timeout=timeout,
                manifest=manifest,
                progress_callback=progress_callback,
                additional_kwargs=additional_kwargs,
                context=context,
            )
            
            # Update app_info with session data
            app_info["session_id"] = session_id
            
            # Progress update after worker starts but before waiting for services
            if not detached:
                if wait_for_service:
                    progress_callback({
                        "type": "info",
                        "message": f"Waiting for service '{wait_for_service}'..."
                    })
                else:
                    progress_callback({
                        "type": "info",
                        "message": "Waiting for client connection..."
                    })
            else:
                progress_callback({
                    "type": "success",
                    "message": "Application started in detached mode"
                })

            # Set up activity tracker after starting the app
            tracker = self.store.get_activity_tracker()
            if (
                not manifest.daemon
                and stop_after_inactive is not None
                and stop_after_inactive > 0
            ):

                async def _stop_after_inactive():
                    if full_client_id in self._sessions:
                        await self._stop(full_client_id, raise_exception=False, context=context)
                    logger.info(
                        f"App {full_client_id} stopped because of inactive for {stop_after_inactive}s."
                    )

                tracker.register(
                    full_client_id,
                    inactive_period=stop_after_inactive,
                    on_inactive=_stop_after_inactive,
                    entity_type="client",
                )

            # Only wait for events if not in detached mode
            if not detached and event_future is not None:
                # Now wait for the event that we set up before starting the app
                logger.info(f"Waiting for event after starting app: {full_client_id}, timeout: {timeout}, wait_for_service: {wait_for_service}")
                await asyncio.wait_for(event_future, timeout=timeout)
                logger.info(f"Successfully received event for starting app: {full_client_id}")
                await asyncio.sleep(0.1)  # Brief delay to allow service registration
            
                progress_callback({
                    "type": "info",
                    "message": "Finalizing application startup..."
                })
            elif detached:
                # In detached mode, give a brief moment for services to be registered
                # but don't wait for them
                logger.info(f"Started app in detached mode: {full_client_id}")


            # save the services
            # service_count = len(collected_services)
            progress_callback({
                "type": "info",
                "message": "Updating application manifest..."
            })
            manifest.name = manifest.name or app_info.get("name", "Untitled App")
            manifest.description = manifest.description or app_info.get(
                "description", ""
            )

            # Replace client ID with * in service IDs for manifest storage
            manifest_services = []
            for svc in collected_services:
                manifest_svc = svc.model_copy()
                service_id_parts = manifest_svc.id.split(":")
                if len(service_id_parts) >= 2:
                    # Replace client ID with * (workspace/client_id -> workspace/*)
                    workspace_client = service_id_parts[0]
                    workspace_part = workspace_client.rsplit("/", 1)[0]  # Get workspace part
                    service_name = ":".join(service_id_parts[1:])  # Get service name part
                    manifest_svc.id = f"{workspace_part}/*:{service_name}"
                manifest_services.append(manifest_svc)
            
            manifest.services = manifest_services
            manifest = ApplicationManifest.model_validate(
                manifest.model_dump(mode="json")
            )
            await self.artifact_manager.edit(
                app_id,
                version=version,
                stage=stage,
                manifest=manifest.model_dump(mode="json"),
                context=context,
            )
            
            progress_callback({
                "type": "success",
                "message": "Application manifest updated successfully"
            })

        except (asyncio.TimeoutError, Exception) as exp:
            # Get worker logs for debugging
            logs = None
            try:
                session_info = self._sessions.get(full_client_id)
                if session_info and "_worker" in session_info:
                    logs = await session_info["_worker"].get_logs(full_client_id, context=context)
            except Exception:
                pass  # Ignore log retrieval errors
            
            # Clean up session
            try:
                session_info = self._sessions.get(full_client_id)
                if session_info and "_worker" in session_info:
                    await session_info["_worker"].stop(full_client_id, context=context)
            except Exception:
                pass  # Ignore cleanup errors
            raise Exception(f"Failed to start app '{app_id}', error: {exp}, logs:\n{logs}") from None
        finally:
            # Clean up event listeners
            if not detached:
                self.event_bus.off_local("service_added", service_added)
                if not wait_for_service:
                    self.event_bus.off_local("client_connected", client_connected)

        if wait_for_service:
            app_info["service_id"] = (
                full_client_id + ":" + wait_for_service + "@" + app_id
            )
        app_info["services"] = [
            svc.model_dump(mode="json") for svc in collected_services
        ]
        if full_client_id in self._sessions:
            self._sessions[full_client_id]["services"] = app_info["services"]
            # Update session metadata with final app name and description
            self._sessions[full_client_id]["name"] = manifest.name
            self._sessions[full_client_id]["description"] = manifest.description
        
        # Start autoscaling if enabled
        if manifest.autoscaling and manifest.autoscaling.enabled:
            progress_callback({
                "type": "info",
                "message": "Configuring autoscaling..."
            })
            await self.autoscaling_manager.start_autoscaling(
                app_id, manifest.autoscaling, context
            )
            progress_callback({
                "type": "success",
                "message": "Autoscaling enabled"
            })
        
        # Final completion message
        progress_callback({
            "type": "success",
            "message": "Application startup completed successfully"
        })
        
        return app_info

    @schema_method
    async def stop(
        self,
        session_id: str = Field(
            ...,
            description="The session ID of the running application instance to stop. This is typically in the format 'workspace/client_id'."
        ),
        raise_exception: bool = Field(
            True,
            description="Whether to raise an exception if the session is not found. Set to False to ignore missing sessions."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> None:
        """Stop a server app instance."""
        user_info = UserInfo.model_validate(context["user"])
        workspace = context["ws"]
        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to stop app {session_id} in workspace {workspace}."
            )
        await self._stop(session_id, raise_exception=raise_exception, context=context)

    async def _stop(self, session_id: str, raise_exception=True, context: Optional[dict] = None):
        if session_id in self._sessions:
            app_info = self._sessions.pop(session_id, None)
            try:
                await app_info["_worker"].stop(session_id, context=context)
            except Exception as exp:
                if raise_exception:
                    raise
                else:
                    logger.warning(f"Failed to stop browser tab: {exp}")
            
            # Check if this was the last instance of an app and stop autoscaling
            app_id = app_info.get("app_id")
            if app_id:
                remaining_instances = self.autoscaling_manager._get_app_instances(app_id)
                if not remaining_instances:
                    await self.autoscaling_manager.stop_autoscaling(app_id)
                    
        elif raise_exception:
            raise Exception(f"Server app instance not found: {session_id}")

    @schema_method
    async def get_logs(
        self,
        session_id: str = Field(
            ...,
            description="The session ID of the running application instance. This is typically in the format 'workspace/client_id'."
        ),
        type: str = Field(
            None,
            description="Type of logs to retrieve: 'log', 'error', or None for all types."
        ),
        offset: int = Field(
            0,
            description="Starting offset for log entries. Use for pagination."
        ),
        limit: Optional[int] = Field(
            None,
            description="Maximum number of log entries to return. If not provided, returns all available logs."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get server app instance logs."""
        user_info = UserInfo.model_validate(context["user"])
        workspace = context["ws"]
        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to get log for app {session_id} in workspace {workspace}."
            )
        if session_id in self._sessions:
            return await self._sessions[session_id]["_worker"].get_logs(
                session_id, type=type, offset=offset, limit=limit, context=context
            )
        else:
            raise Exception(f"Server app instance not found: {session_id}")

    @schema_method
    async def take_screenshot(
        self,
        session_id: str = Field(
            ...,
            description="The session ID of the running application instance. This is typically in the format 'workspace/client_id'."
        ),
        format: str = Field(
            "png",
            description="Screenshot format: 'png' or 'jpeg'. Returns base64 encoded image data."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> str:
        """Take a screenshot of a running server app instance."""
        user_info = UserInfo.model_validate(context["user"])
        workspace = context["ws"]
        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to take screenshot of app {session_id} in workspace {workspace}."
            )
        if session_id in self._sessions:
            return await self._sessions[session_id]["_worker"].take_screenshot(
                session_id, format=format, context=context
            )
        else:
            raise Exception(f"Server app instance not found: {session_id}")


    @schema_method
    async def list_running(
        self,
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> List[str]:
        """List the running sessions for the current workspace."""
        workspace = context["ws"]
        return [
            {k: v for k, v in session_info.items() if not k.startswith("_")}
            for session_id, session_info in self._sessions.items()
            if session_id.startswith(f"{workspace}/")
        ]

    @schema_method
    async def list_apps(
        self,
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """List applications in the workspace."""
        try:
            ws = context["ws"]
            apps = await self.artifact_manager.list_children(
                f"{ws}/applications", filters={"type": "application"}, context=context
            )
            return [app["manifest"] for app in apps]
        except KeyError:
            return []
        except Exception as exp:
            raise Exception(f"Failed to list apps: {exp}") from exp

    @schema_method
    async def get_app_info(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to inspect. This is typically the alias of the application."
        ),
        version: str = Field(
            None,
            description="Version of the application to inspect. If not provided, uses the latest version."
        ),
        stage: bool = Field(
            False,
            description="Whether to inspect the staging version of the application. Set to True to inspect the staged version."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> dict:
        """Get detailed information about an installed application.

        This method returns comprehensive information about an application:
        1. Validates permissions and application existence
        2. Retrieves the application manifest and metadata
        3. Returns detailed configuration and service information

        Returns a dictionary containing:
        - manifest: The application manifest with all metadata
        - id: The application ID
        - name: The application name
        - description: The application description
        - version: The current version
        - type: The application type
        - entry_point: The main entry point file
        - services: List of services provided by the application
        - files: List of files in the application
        - config: Application configuration
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        """
        try:
            artifact_info = await self.artifact_manager.read(
                app_id, version=version, stage=stage, context=context
            )
            return artifact_info
        except Exception as exp:
            raise Exception(f"Failed to get app info for {app_id}: {exp}") from exp

    @schema_method
    async def read_file(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application containing the file."
        ),
        file_path: str = Field(
            ...,
            description="The path of the file to read. Use forward slashes for path separators (e.g., 'src/main.js', 'index.html')."
        ),
        format: str = Field(
            "text",
            description="Format to return the content in: 'text', 'json', or 'binary'. Binary returns base64 encoded content."
        ),
        version: str = Field(
            None,
            description="Version of the application to inspect. If not provided, uses the latest version."
        ),
        stage: bool = Field(
            False,
            description="Whether to read the file from the staged version of the application. Set to True to read from the staged version."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> Union[str, dict, bytes]:
        """Get the content of a specific file from an installed application.

        This method retrieves the content of a file stored in the application:
        1. Validates permissions and application existence
        2. Retrieves the file content from the artifact storage
        3. Returns the file content in the specified format

        Returns:
            - str: File content as text when format='text'
            - dict: Parsed JSON content when format='json'
            - str: Base64 encoded content when format='binary'
        """
        try:
            get_url = await self.artifact_manager.get_file(
                app_id, file_path=file_path, version=version, stage=stage, context=context
            )
            async with httpx.AsyncClient() as client:
                response = await client.get(get_url)
                if response.status_code == 200:
                    if format == "text":
                        return response.text
                    elif format == "json":
                        return response.json()
                    elif format == "binary":
                        import base64
                        return base64.b64encode(response.content).decode()
                    else:
                        raise ValueError(f"Invalid format '{format}'. Must be 'text', 'json' or 'binary'")
                else:
                    raise Exception(f"Failed to retrieve file {file_path}: HTTP {response.status_code}")
        except Exception as exp:
            raise Exception(f"Failed to get file content for {app_id}/{file_path}: {exp}") from exp

    @schema_method
    async def validate_app_manifest(
        self,
        manifest: Dict[str, Any] = Field(
            ...,
            description="Application manifest dictionary to validate."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> dict:
        """Validate an application configuration dictionary.

        This method checks if an application configuration is valid:
        1. Validates required fields are present
        2. Checks field types and formats
        3. Validates application type and entry point
        4. Checks for common configuration errors

        Returns a dictionary containing:
        - valid: Boolean indicating if the configuration is valid
        - errors: List of validation errors (if any)
        - warnings: List of validation warnings (if any)
        - suggestions: List of suggestions for improvement (if any)
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # Check required fields
            required_fields = ["name", "type", "version"]
            for field in required_fields:
                if field not in manifest:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
            
            # Check field types
            if "name" in manifest and not isinstance(manifest["name"], str):
                validation_result["errors"].append("Field 'name' must be a string")
                validation_result["valid"] = False
            
            if "type" in manifest and manifest["type"] not in ["window", "web-worker", "web-python", "mcp-server", "a2a-agent"]:
                validation_result["warnings"].append(f"Unusual application type: {manifest['type']}")
            
            # Special validation for mcp-server type
            if "type" in manifest and manifest["type"] == "mcp-server":
                if "mcpServers" not in manifest:
                    validation_result["errors"].append("MCP server applications require 'mcpServers' configuration")
                    validation_result["valid"] = False
                elif not isinstance(manifest["mcpServers"], dict):
                    validation_result["errors"].append("'mcpServers' must be a dictionary")
                    validation_result["valid"] = False
                elif len(manifest["mcpServers"]) == 0:
                    validation_result["errors"].append("'mcpServers' cannot be empty")
                    validation_result["valid"] = False
                else:
                    # Validate each MCP server configuration
                    for server_name, server_config in manifest["mcpServers"].items():
                        if not isinstance(server_config, dict):
                            validation_result["errors"].append(f"MCP server '{server_name}' configuration must be a dictionary")
                            validation_result["valid"] = False
                            continue
                        
                        if "type" not in server_config:
                            validation_result["errors"].append(f"MCP server '{server_name}' missing required field 'type'")
                            validation_result["valid"] = False
                        
                        if "url" not in server_config:
                            validation_result["errors"].append(f"MCP server '{server_name}' missing required field 'url'")
                            validation_result["valid"] = False
                        
                        if server_config.get("type") not in ["streamable-http", "stdio"]:
                            validation_result["warnings"].append(f"MCP server '{server_name}' has unusual type: {server_config.get('type')}")
            
            # Special validation for a2a-agent type
            if "type" in manifest and manifest["type"] == "a2a-agent":
                if "a2aAgents" not in manifest:
                    validation_result["errors"].append("A2A agent applications require 'a2aAgents' configuration")
                    validation_result["valid"] = False
                elif not isinstance(manifest["a2aAgents"], dict):
                    validation_result["errors"].append("'a2aAgents' must be a dictionary")
                    validation_result["valid"] = False
                elif len(manifest["a2aAgents"]) == 0:
                    validation_result["errors"].append("'a2aAgents' cannot be empty")
                    validation_result["valid"] = False
                else:
                    # Validate each A2A agent configuration
                    for agent_name, agent_config in manifest["a2aAgents"].items():
                        if not isinstance(agent_config, dict):
                            validation_result["errors"].append(f"A2A agent '{agent_name}' configuration must be a dictionary")
                            validation_result["valid"] = False
                            continue
                        
                        if "url" not in agent_config:
                            validation_result["errors"].append(f"A2A agent '{agent_name}' missing required field 'url'")
                            validation_result["valid"] = False
                        
                        # Check if URL looks like a valid A2A endpoint
                        url = agent_config.get("url", "")
                        if not url.startswith(("http://", "https://")):
                            validation_result["warnings"].append(f"A2A agent '{agent_name}' URL should start with http:// or https://")
                        
                        # Check for optional headers
                        if "headers" in agent_config and not isinstance(agent_config["headers"], dict):
                            validation_result["errors"].append(f"A2A agent '{agent_name}' headers must be a dictionary")
                            validation_result["valid"] = False
            
            if "version" in manifest and not isinstance(manifest["version"], str):
                validation_result["errors"].append("Field 'version' must be a string")
                validation_result["valid"] = False
            
            # Check entry point
            if "entry_point" in manifest:
                entry_point = manifest["entry_point"]
                if not entry_point.endswith((".html", ".js", ".py")):
                    validation_result["warnings"].append(f"Entry point '{entry_point}' has unusual extension")
            
            # Suggestions
            if "description" not in manifest:
                validation_result["suggestions"].append("Consider adding a description for better documentation")
            
            if "tags" not in manifest:
                validation_result["suggestions"].append("Consider adding tags for better discoverability")
            
            return validation_result
            
        except Exception as exp:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {exp}")
            return validation_result

    @schema_method
    async def edit_app(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to edit."
        ),
        manifest: Optional[Dict[str, Any]] = Field(
            None,
            description="Updated manifest fields to merge with the existing manifest. Only provided fields will be updated."
        ),
        disabled: Optional[bool] = Field(
            None,
            description="Whether to disable the application. Set to True to disable, False to enable. If not provided, disabled status is not changed."
        ),
        autoscaling_config: Optional[Dict[str, Any]] = Field(
            None,
            description="Autoscaling configuration dictionary. Set to None to disable autoscaling."
        ),
        autoscaling_manifest: Optional[Dict[str, Any]] = Field(
            None,
            description="Alias for autoscaling_config for backward compatibility."
        ),
        startup_config: Optional[Dict[str, Any]] = Field(
            None,
            description="Startup configuration dictionary containing default startup parameters like timeout, wait_for_service, and stop_after_inactive."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> None:
        """Edit an application's manifest, config, and/or disabled status.

        This method allows you to update various aspects of an application:
        1. Retrieves the current application manifest
        2. Merges provided manifest updates with existing manifest
        3. Merges provided config updates with existing config
        4. Updates the disabled status if provided
        5. Updates the startup_config if provided
        6. Updates the autoscaling_config if provided
        7. Saves the updated manifest

        The method performs selective updates - only the fields you provide will be changed.
        Existing fields not mentioned in the updates will remain unchanged.

        Note: This will put the application in staging mode and must be committed to take effect.
        Use commit_app() after editing to make changes permanent.
        """
        try:
            # Get current app info
            app_info = await self.get_app_info(app_id, context=context)
            current_manifest = app_info["manifest"]
            
            # Start with current manifest
            updated_manifest = current_manifest.copy() if current_manifest else {}
            
            # Update manifest fields if provided
            if manifest:
                updated_manifest.update(manifest)
            if disabled is not None:
                updated_manifest["disabled"] = disabled
                if disabled:
                    # stop all the instances of the app
                    for app in self._sessions.values():
                        if app["id"] == app_id:
                            await self.stop(app["id"], raise_exception=False)

            # Handle autoscaling configuration updates
            final_autoscaling_config = autoscaling_config
            if autoscaling_manifest is not None:
                final_autoscaling_config = autoscaling_manifest
            
            if final_autoscaling_config is not None:
                if final_autoscaling_config:
                    # Validate the autoscaling config
                    autoscaling_obj = AutoscalingConfig.model_validate(final_autoscaling_config)
                    updated_manifest["autoscaling"] = autoscaling_obj.model_dump()
                else:
                    updated_manifest["autoscaling"] = None

            # Handle startup_config updates
            if startup_config is not None:
                if startup_config:
                    # Simply update the startup_config dictionary
                    updated_manifest["startup_config"] = startup_config
                else:
                    updated_manifest["startup_config"] = None

            # Update the manifest with all changes
            await self.artifact_manager.edit(
                app_id,
                manifest=updated_manifest,
                stage=True,
                context=context
            )
            
            # If the app is currently running and autoscaling config changed, restart autoscaling
            if autoscaling_config is not None:
                app_instances = self.autoscaling_manager._get_app_instances(app_id)
                if app_instances:
                    # Stop existing autoscaling
                    await self.autoscaling_manager.stop_autoscaling(app_id)
                    
                    # Start new autoscaling if enabled
                    if autoscaling_config and autoscaling_config.get("enabled", False):
                        autoscaling_obj = AutoscalingConfig.model_validate(autoscaling_config)
                        await self.autoscaling_manager.start_autoscaling(
                            app_id, autoscaling_obj, context
                        )
            
        except Exception as exp:
            raise Exception(f"Failed to edit app '{app_id}': {exp}") from exp

    async def shutdown(self) -> None:
        """Shutdown the app controller."""
        logger.info("Closing the server app controller...")
        
        # Stop all autoscaling tasks
        for app_id in list(self.autoscaling_manager._autoscaling_tasks.keys()):
            await self.autoscaling_manager.stop_autoscaling(app_id)
        
        for app in self._sessions.values():
            await self.stop(app["id"], raise_exception=False)

    async def prepare_workspace(self, workspace_info: WorkspaceInfo):
        """Prepare the workspace."""
        context = {
            "ws": workspace_info.id,
            "user": self.store.get_root_user().model_dump(),
        }
        apps = await self.list_apps(context=context)
        # start daemon apps
        for app in apps:
            if app.get("daemon"):
                logger.info(f"Starting daemon app: {app['id']}")
                try:
                    await self.start(app["id"], context=context)
                except Exception as exp:
                    logger.error(
                        f"Failed to start daemon app: {app['id']}, error: {exp}"
                    )
        
        if workspace_info.id not in ["ws-user-root", "public", "ws-anonymous"]:
            context = {
                "ws": workspace_info.id,
                "user": self.store.get_root_user().model_dump(),
            }
            workers = await self.get_server_app_workers(context=context)
            for worker in workers:
                if worker.prepare_workspace:
                    try:
                        await worker.prepare_workspace(workspace_info.id, context=context)
                    except Exception as exp:
                        logger.warning(
                            f"Worker {worker.id} failed to prepare workspace: {workspace_info.id}, error: {exp}"
                        )

    async def close_workspace(self, workspace_info: WorkspaceInfo):
        """Archive the workspace."""
        # Define context first
        context = {
            "ws": workspace_info.id,
            "user": self.store.get_root_user().model_dump(),
        }
        # Stop all running apps
        for app in list(self._sessions.values()):
            if app["workspace"] == workspace_info.id:
                await self._stop(app["id"], raise_exception=False, context=context)
        # Send to all workers
        workers = await self.get_server_app_workers(context=context)
        if not workers:
            return
        for worker in workers:
            if worker.close_workspace:
                try:
                    await worker.close_workspace(workspace_info.id, context=context)
                except Exception as exp:
                    logger.warning(
                        f"Worker failed to close workspace: {workspace_info.id}, error: {exp}"
                    )

    async def launch(self, *args, **kwargs):
        """Launch an application."""
        # launch is deprecated, use install then start instead 
        raise NotImplementedError("Launch is deprecated, use install then start instead")

    def get_service_api(self) -> Dict[str, Any]:
        """Get a list of service API endpoints."""
        return {
            "name": "Server Apps",
            "id": "server-apps",
            "type": "server-apps",
            "config": {"visibility": "public", "require_context": True},
            "install": self.install,
            "uninstall": self.uninstall,
            "launch": self.launch,
            "start": self.start,
            "stop": self.stop,
            "list_apps": self.list_apps,
            "list_running": self.list_running,
            "get_logs": self.get_logs,
            "take_screenshot": self.take_screenshot,
            "edit_file": self.edit_file,
            "remove_file": self.remove_file,
            "list_files": self.list_files,
            "commit_app": self.commit_app,
            "get_app_info": self.get_app_info,
            "read_file": self.read_file,
            "validate_app_manifest": self.validate_app_manifest,
            "edit_app": self.edit_app,
        }
