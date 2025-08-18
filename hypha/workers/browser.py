"""Provide a browser worker."""

import os
import logging
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from urllib.parse import urlparse

import httpx
from playwright.async_api import Page, async_playwright, Browser
from jinja2 import Environment, PackageLoader, select_autoescape

from hypha.workers.base import (
    BaseWorker,
    WorkerConfig,
    SessionStatus,
    SessionInfo,
    SessionNotFoundError,
    WorkerError,
    safe_call_callback,
)
from hypha.workers.browser_cache import BrowserCache
from hypha.plugin_parser import parse_imjoy_plugin
from hypha import hypha_rpc_version
from hypha.utils import PLUGIN_CONFIG_FIELDS, safe_join

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("browser")
logger.setLevel(LOGLEVEL)

MAXIMUM_LOG_ENTRIES = 2048


def _capture_logs_from_browser_tabs(page: Page, logs: dict) -> None:
    """Capture browser tab logs."""
    logs["error"] = []

    def _app_info(message: Any) -> None:
        """Log message at info level."""
        msg_type = message.type
        logger.info("%s: %s", msg_type, message.text)
        if msg_type not in logs:
            logs[msg_type] = []
        logs[msg_type].append(message.text)
        if len(logs[msg_type]) > MAXIMUM_LOG_ENTRIES:
            logs[msg_type].pop(0)

    def _app_error(message: str) -> None:
        """Log message at error level."""
        logger.error(message)
        logs["error"].append(message)
        if len(logs["error"]) > MAXIMUM_LOG_ENTRIES:
            logs["error"].pop(0)

    page.on("console", _app_info)
    page.on("error", lambda target: _app_error(target.text))
    page.on("pageerror", lambda target: _app_error(str(target)))


class BrowserWorker(BaseWorker):
    """Browser app worker."""

    instance_counter: int = 0

    def __init__(
        self,
        in_docker: bool = False,
        use_local_url: bool = True,
    ):
        """Initialize the class."""
        super().__init__()
        self.browser: Optional[Browser] = None
        self.controller_id = str(BrowserWorker.instance_counter)
        BrowserWorker.instance_counter += 1
        self.in_docker = in_docker
        self._use_local_url = use_local_url
        self._playwright = None
        self.jinja_env = Environment(
            loader=PackageLoader("hypha"), autoescape=select_autoescape()
        )
        self._compiled_apps = {}  # Cache compiled apps

        # Session management
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}
        self.initialized = False

        # Initialize cache manager
        self.cache_manager = None

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["web-python", "web-worker", "window", "iframe", "hypha", "web-app"]

    @property
    def name(self) -> str:
        """Return the worker name."""
        return "Browser Worker"

    @property
    def description(self) -> str:
        """Return the worker description."""
        return "A worker for running web applications in browser environments"

    @property
    def require_context(self) -> bool:
        """Return whether the worker requires a context."""
        return True

    @property
    def use_local_url(self) -> bool:
        """Return whether the worker should use local URLs."""
        return self._use_local_url

    async def initialize(self) -> Browser:
        """Initialize the browser worker."""
        if self.initialized:
            return self.browser

        self._playwright = await async_playwright().start()
        args = [
            "--enable-unsafe-webgpu",
            "--use-vulkan",
            "--enable-features=Vulkan,WebAssemblyJSPI",
            "--enable-experimental-web-platform-features",
            "--site-per-process",  # Keep process isolation for security
        ]
        
        # Optionally disable web security (bypasses CORS) - controlled by environment variable
        # Note: We keep site-per-process enabled for security between different sites
        if os.environ.get("PLAYWRIGHT_DISABLE_WEB_SECURITY", "true").lower() in ("true", "1", "yes"):
            args.append("--disable-web-security")
            logger.info("Web security disabled - CORS restrictions bypassed (process isolation maintained)")
        # so it works in the docker image
        if self.in_docker:
            args.append("--no-sandbox")

        self.browser = await self._playwright.chromium.launch(
            args=args, handle_sigint=True, handle_sigterm=True, handle_sighup=True
        )
        self.initialized = True
        return self.browser

    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new browser session."""
        # Handle both pydantic model and dict input for RPC compatibility
        if isinstance(config, dict):
            config = WorkerConfig(**config)

        session_id = config.id  # Use the provided session ID

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

        try:
            # Initialize browser if needed
            if not self.initialized:
                await self.initialize()

            session_data = await self._start_browser_session(config)
            self._session_data[session_id] = session_data

            # Update session status
            session_info.status = SessionStatus.RUNNING
            logger.info(f"Started browser session {session_id}")

            return session_id

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to start browser session {session_id}: {e}")
            # Clean up failed session
            self._sessions.pop(session_id, None)
            raise

    async def _start_browser_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Start a browser app session."""
        timeout_ms = config.timeout * 1000 if config.timeout else 60000
        # Call progress callback if provided
        await safe_call_callback(config.progress_callback,
            {"type": "info", "message": "Initializing browser worker..."}
        )

        if not self.browser:
            await self.initialize()

        # Get app type from manifest
        app_type = config.manifest.get("type")
        if app_type not in self.supported_types:
            raise Exception(
                f"Browser worker only supports {self.supported_types}, got {app_type}"
            )

        await safe_call_callback(config.progress_callback,
            {"type": "info", "message": "Creating browser context..."}
        )

        # Create a new context for isolation
        # SECURITY: Each context is completely isolated from others, preventing cross-user data access
        # even when apps are served from the same origin
        context_options = {
            "viewport": {"width": 1280, "height": 720},
            "accept_downloads": False,  # Default to false for security
            # Additional security: Use unique storage state per session
            # This ensures complete isolation of cookies, localStorage, etc.
            "storage_state": None,  # Start with clean state
        }

        # Add Playwright configuration from environment variables and manifest
        self._apply_playwright_env_config(context_options, config.manifest)

        context = await self.browser.new_context(**context_options)

        # Create a new page in the context
        page = await context.new_page()

        entry_point = config.manifest.get("entry_point", "index.html")

        # Setup console and error logging
        logs = {"console": [], "error": []}

        def log_console(msg):
            logs["console"].append(
                f"[{msg.type}] {msg.text}"
                if msg.text
                else (
                    f"[{msg.type}] JSHandle@{msg.args[0]}"
                    if msg.args
                    else f"[{msg.type}]"
                )
            )

        def log_error(error):
            logs["error"].append(str(error))

        page.on("console", log_console)
        page.on("pageerror", log_error)

        # Setup authentication BEFORE navigation
        app_type = config.manifest.get("type")
        if app_type == "web-app":
            # For web-app, setup authentication before navigating to external URL
            await self._setup_page_authentication(page, config.manifest, entry_point)
        else:
            # For other types, setup authentication for local URL
            await self._setup_page_authentication(page, config.manifest)

        # Setup caching if enabled
        cache_enabled = False
        if self.cache_manager:
            # Check if caching is explicitly enabled
            enable_cache = config.manifest.get("enable_cache", True)  # Default to True
            cache_routes = config.manifest.get("cache_routes", [])

            # Add default cache routes for web-python apps if caching is enabled
            if enable_cache and app_type == "web-python" and not cache_routes:
                cache_routes = self.cache_manager.get_default_cache_routes_for_type(
                    app_type
                )

            if enable_cache and cache_routes:
                cache_enabled = True
                await self._setup_route_caching(
                    page, config.workspace, config.app_id, cache_routes
                )
                await self.cache_manager.start_recording(
                    config.workspace, config.app_id
                )

        await safe_call_callback(config.progress_callback, {"type": "info", "message": "Loading application..."})

        try:
            # Get the entry point from manifest - this should be a compiled HTML file uploaded to artifact manager
            entry_point = config.manifest.get("entry_point", "index.html")
            logger.info(
                f"Browser worker starting session with entry_point: {entry_point}, app_type: {app_type}"
            )

            # Generate URLs for the app
            local_url, public_url = self._generate_app_urls(config, entry_point)

            # For web-app type, navigate directly to the entry_point URL
            if app_type == "web-app":
                goto_url = entry_point  # Use entry_point as the external URL

                # Check if we need to append credential query parameters
                if config.manifest.get("inject_auth_params"):
                    # Parse the URL to check if it already has query parameters
                    separator = "&" if "?" in goto_url else "?"

                    # Build query parameters
                    params = []
                    if config.server_url:
                        params.append(f"server_url={config.server_url}")
                    params.append(f"client_id={config.client_id}")
                    params.append(f"workspace={config.workspace}")
                    params.append(f"app_id={config.app_id}")
                    if config.token:
                        params.append(f"token={config.token}")

                    # Append parameters to the URL
                    if params:
                        goto_url = f"{goto_url}{separator}{'&'.join(params)}"

                logger.info(
                    f"Loading web-app from external URL: {goto_url} with timeout {timeout_ms}ms"
                )
                response = await page.goto(
                    goto_url, timeout=timeout_ms, wait_until="load"
                )
            else:
                goto_url = local_url
                logger.info(
                    f"Loading browser app from URL: {local_url} with timeout {timeout_ms}ms"
                )
                response = await page.goto(goto_url, timeout=timeout_ms, wait_until="load")

            # Special handling for data: and about: URLs - they don't return a response object
            if goto_url.startswith(("data:", "about:")):
                logger.info(f"Loaded special URL successfully: {goto_url[:50]}...")
            elif not response:
                await context.close()
                raise Exception(f"Failed to load URL: {goto_url}")
            elif response.status != 200:
                await context.close()
                raise Exception(
                    f"Failed to start browser app instance, "
                    f"status: {response.status}, url: {goto_url}"
                )

            # Storage is now preloaded via add_init_script, no need for post-navigation setup

            logger.info("Browser app loaded successfully")

            # Wait a bit for JavaScript to initialize and services to be registered
            # This gives time for the api.export() call to complete
            await page.wait_for_timeout(1000)  # Wait 1 second for JS initialization
            logger.info("JavaScript initialization wait completed")
            
            # Execute startup script if specified in manifest
            startup_script_path = config.manifest.get("startup_script")
            if startup_script_path:
                await safe_call_callback(config.progress_callback,
                    {"type": "info", "message": f"Fetching and executing startup script: {startup_script_path}"}
                )
                
                try:
                    # Fetch the startup script from artifact manager using httpx
                    script_url = f"{config.server_url}/{config.workspace}/artifacts/{config.app_id}/files/{startup_script_path}?use_proxy=true"
                    
                    async with httpx.AsyncClient(verify=not getattr(config, 'disable_ssl', True)) as client:
                        response = await client.get(
                            script_url,
                            headers={"Authorization": f"Bearer {config.token}"} if config.token else {}
                        )
                        response.raise_for_status()
                        startup_script = response.text
                    
                    # Execute the startup script in the page context
                    await page.evaluate(startup_script)
                    logger.info(f"Executed startup script: {startup_script_path}")
                    
                    await safe_call_callback(config.progress_callback,
                        {"type": "success", "message": "Startup script executed successfully"}
                    )
                except httpx.HTTPStatusError as e:
                    error_msg = f"Failed to fetch startup script {startup_script_path}: HTTP {e.response.status_code}"
                    logger.error(error_msg)
                    # Log error but don't fail the session - startup script is optional enhancement
                    await safe_call_callback(config.progress_callback,
                        {"type": "warning", "message": error_msg}
                    )
                except Exception as e:
                    error_msg = f"Failed to execute startup script: {str(e)}"
                    logger.error(error_msg)
                    # Log error but don't fail the session - startup script is optional enhancement
                    await safe_call_callback(config.progress_callback,
                        {"type": "warning", "message": error_msg}
                    )

            await safe_call_callback(config.progress_callback,
                {"type": "success", "message": "Application loaded successfully"}
            )

            return {
                "local_url": local_url,
                "public_url": public_url,
                "page": page,
                "context": context,
                "logs": logs,
                "cache_enabled": cache_enabled,
            }

        except Exception as e:
            await context.close()
            raise e

    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop a browser session."""
        if session_id not in self._sessions:
            logger.warning(
                f"Browser session {session_id} not found for stopping, may have already been cleaned up"
            )
            return

        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING

        session_data = self._session_data.get(session_id)
        if session_data:
            # Stop cache recording if it was enabled
            if session_data.get("cache_enabled") and self.cache_manager:
                await self.cache_manager.stop_recording(
                    session_info.workspace, session_info.app_id
                )

            if "page" in session_data:
                await session_data["page"].close()
            if "context" in session_data:
                await session_data["context"].close()

        session_info.status = SessionStatus.STOPPED
        logger.info(f"Stopped browser session {session_id}")

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
        """Get logs for a browser session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Browser session {session_id} not found")

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
        """Shutdown the browser worker."""
        logger.info("Shutting down browser worker...")

        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.stop(session_id)

        # Shutdown browser and playwright
        if self.browser:
            await self.browser.close()

        if self._playwright:
            await self._playwright.stop()

        logger.info("Browser worker closed successfully.")

        self.initialized = False
        logger.info("Browser worker shutdown complete")

    def _build_app_url(self, config: WorkerConfig, entry_point: str) -> str:
        """Build the app URL with parameters."""
        server_url = (
            config.server_url or "http://127.0.0.1:38283"
        )  # Fallback for testing
        params = [
            f"server_url={server_url}",
            f"client_id={config.client_id}",
            f"workspace={config.workspace}",
            f"app_id={config.app_id}",
            "use_proxy=true",
        ]

        if config.token:
            params.append(f"token={config.token}")

        # Check if entry_point is already a full URL
        if entry_point.startswith("http"):
            base_url = entry_point
        else:
            # Construct URL from artifact_id and server_url
            workspace_id, app_id = config.artifact_id.split("/", 1)
            base_url = (
                f"{server_url}/{workspace_id}/artifacts/{app_id}/files/{entry_point}"
            )

        return f"{base_url}?{'&'.join(params)}"

    def _generate_app_urls(
        self, config: WorkerConfig, entry_point: str
    ) -> tuple[str, str]:
        """Generate local and public URLs for the app entry point."""
        if entry_point.startswith("http"):
            # External URL - use as-is but still add parameters
            local_url = self._build_app_url(config, entry_point)
            public_url = self._build_app_url(config, entry_point)
        else:
            # Construct URL from artifact_id and server_url
            workspace_id, app_id = config.artifact_id.split("/", 1)
            server_url = (
                config.server_url or "http://127.0.0.1:38283"
            )  # Fallback for testing
            base_url = (
                f"{server_url}/{workspace_id}/artifacts/{app_id}/files/{entry_point}"
            )

            local_url = self._build_app_url(config, base_url)
            public_url = self._build_app_url(config, base_url)

        return local_url, public_url

    async def compile(
        self,
        manifest: dict,
        files: list,
        config: dict = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, list]:
        """Compile browser app manifest and files.

        This method:
        1. Looks for 'source' file OR config/script files in the files list
        2. Compiles them to 'index.html' using appropriate template
        3. Updates manifest with correct entry_point
        4. Returns updated manifest and files
        """
        # Extract progress_callback from config; default to no-op if not provided
        def _noop(_msg):
            return None

        progress_callback = (
            config.get("progress_callback") if config and config.get("progress_callback") else _noop
        )

        await safe_call_callback(progress_callback,
            {"type": "info", "message": "Starting browser app compilation..."}
        )

        app_type = manifest.get("type")
        if app_type and app_type not in self.supported_types:
            # Not a browser app type, return as-is
            raise Exception(
                f"Browser worker only supports {self.supported_types}, got {app_type}"
            )
        if app_type is None:
            logger.warning("No app type found in manifest, using default app type")

        # Handle web-app type specially - it doesn't need compilation, just validation
        if app_type == "web-app":
            await safe_call_callback(progress_callback,
                {"type": "info", "message": "Processing web-app configuration..."}
            )

            # Validate required fields for web-app
            if not manifest.get("entry_point"):
                raise Exception("web-app type requires 'entry_point' field in manifest")

            new_manifest = manifest.copy()
            new_manifest["type"] = "web-app"
            # No entry_point needed since we navigate directly to the URL

            # No files needed for web-app type
            new_files = []

            await safe_call_callback(progress_callback,
                {
                    "type": "success",
                    "message": "Web-app configuration processed successfully",
                }
            )
            return new_manifest, new_files

        await safe_call_callback(progress_callback,
            {"type": "info", "message": f"Compiling {app_type} application..."}
        )

        # Look for different types of source files
        source_file = None
        config_file = None
        script_file = None
        source_content = ""

        files_by_name = {f.get("path"): f for f in files}

        entry_point = manifest["entry_point"]
        # Check for source file (traditional approach)
        if entry_point in files_by_name:
            source_file = files_by_name[entry_point]
            if source_file.get("content") is None:
                raise Exception(f"Source file {source_file.get('path')} is empty")
            source_content = source_file.get("content") or ""

        # Only process config/script files for compilation if there's a source file
        # This indicates they were extracted from XML, not provided directly by user
        if source_file:
            await safe_call_callback(progress_callback,
                {
                    "type": "info",
                    "message": "Processing source files and configurations...",
                }
            )
        else:
            # No source file, but we still need to ensure all browser apps have index.html as entry_point
            # This handles cases where files are provided directly (not as source)
            updated_manifest = manifest.copy()
            updated_manifest["entry_point"] = manifest.get("entry_point", "index.html")
            updated_manifest["type"] = app_type

            # Check if there are any HTML files that need to be renamed to index.html
            updated_files = []
            found_html_file = False

            for file in files:
                file_name = file.get("path", "")
                if file_name.endswith(".html") and not found_html_file:
                    # Rename the first HTML file to index.html
                    updated_file = file.copy()
                    updated_file["path"] = "index.html"
                    updated_files.append(updated_file)
                    found_html_file = True
                elif file_name != "index.html" or not found_html_file:
                    # Keep other files as-is, but avoid duplicating index.html
                    updated_files.append(file)

            return updated_manifest, updated_files

        await safe_call_callback(progress_callback,
            {"type": "info", "message": "Compiling source code to HTML template..."}
        )

        # Compile the source to HTML
        compiled_config, compiled_html = await self._compile_source_to_html(
            source_content, app_type, manifest, config
        )
        new_manifest = manifest.copy()
        new_manifest.update(compiled_config)
        app_type = new_manifest.get("type")

        await safe_call_callback(progress_callback,
            {"type": "info", "message": "Updating manifest and preparing files..."}
        )

        # Set the entry point to index.html for all browser app types
        new_manifest["entry_point"] = "index.html"
        # Ensure the manifest type is correctly set to the expected app_type
        new_manifest["type"] = new_manifest.get("type", app_type)
        entry_point = new_manifest["entry_point"]

        # Create new files list without the source/config/script files and add compiled file
        files_to_remove = set([entry_point])
        if config_file:
            files_to_remove.add(config_file["path"])
        if script_file:
            files_to_remove.add(script_file["path"])

        new_files = [f for f in files if f.get("path") not in files_to_remove]

        # Always save the compiled HTML as index.html for consistency
        new_files.append(
            {"path": entry_point, "content": compiled_html, "format": "text"}
        )
        if "scripts" in new_manifest:
            del new_manifest["scripts"]
        if "script" in new_manifest:
            del new_manifest["script"]
        if "code" in new_manifest:
            new_files.append(
                {"path": "source", "content": new_manifest["code"], "format": "text"}
            )
            del new_manifest["code"]

        await safe_call_callback(progress_callback,
            {
                "type": "success",
                "message": f"Browser app compilation completed. Generated {entry_point}",
            }
        )

        return new_manifest, new_files

    async def _compile_source_to_html(
        self, source: str, app_type: str, manifest: dict, config: dict = None
    ) -> str:
        """Compile source code to HTML using templates."""
        # Check if it's raw HTML
        if source.lower().strip().startswith(("<!doctype html", "<html")):
            # Raw HTML - return as-is
            return {}, source

        # Handle hypha app type with legacy compilation logic
        if app_type == "hypha":
            template_config = parse_imjoy_plugin(source)
            if template_config:
                # Merge template config with manifest
                final_config = manifest.copy()
                final_config.update(template_config)
                if not final_config.get("type"):
                    raise ValueError("No application type found in manifest")
                final_config["source_hash"] = manifest.get("source_hash", "")
                entry_point = template_config.get("entry_point", "index.html")
                final_config["entry_point"] = entry_point
                
                # Map hypha type to window type for template selection
                template_type = "window" if final_config["type"] == "hypha" else final_config["type"]
                
                # check if the template file exists
                template_file = safe_join("apps", template_type + "." + entry_point)
                if template_file not in self.jinja_env.loader.list_templates():
                    raise ValueError(f"Application type {final_config['type']} is not supported by any existing application worker.")
                
                try:
                    template = self.jinja_env.get_template(
                        safe_join("apps", template_type + "." + entry_point)
                    )

                    # Get server URL from config or use fallback
                    server_url = (
                        config.get("server_url", "http://127.0.0.1:38283")
                        if config
                        else "http://127.0.0.1:38283"
                    )

                    template_config = {
                        k: final_config[k]
                        for k in final_config
                        if k in PLUGIN_CONFIG_FIELDS
                    }
                    template_config["server_url"] = server_url
                    compiled_html = template.render(
                        hypha_hypha_rpc_version=hypha_rpc_version,
                        hypha_rpc_websocket_mjs=f"{server_url}/assets/hypha-rpc-websocket.mjs",
                        config=template_config,
                        script=final_config.get("script", ""),
                        requirements=final_config.get("requirements", []),
                        local_base_url=server_url,
                    )

                    return template_config, compiled_html
                except Exception as e:
                    raise ValueError(f"Failed to compile hypha plugin config: {e}") from e
            else:
                raise Exception("Failed to parse hypha plugin config")
    

        # We need to check if source has top level xml tags
        # If not, we need to wrap it as a script tag, if type=window/iframe/web-worker we use js script tag
        # If type=web-python we use python script tag

        if not source.strip().startswith("<"):
            final_config = {"script": source}
        else:
            final_config = parse_imjoy_plugin(source)

        final_config.update(manifest)

        app_type = app_type or final_config.get("type")

        assert (
            app_type in self.supported_types
        ), f"Browser worker only supports {self.supported_types}, got {app_type}"

        # Determine template file
        entry_point = final_config.get("entry_point", "index.html")
        template_name = safe_join("apps", f"{app_type}.{entry_point}")

        try:
            template = self.jinja_env.get_template(template_name)
        except Exception:
            # Fallback to generic template
            template = self.jinja_env.get_template(
                safe_join("apps", f"{app_type}.index.html")
            )

        # Get server URL from config or use fallback
        server_url = (
            config.get("server_url", "http://127.0.0.1:38283")
            if config
            else "http://127.0.0.1:38283"
        )

        # Render the template with actual URLs
        template_config = {
            k: final_config[k] for k in final_config if k in PLUGIN_CONFIG_FIELDS
        }
        template_config["server_url"] = server_url
        if "required_artifact_files" in template_config and app_type != "web-python":
            raise ValueError(
                "required_artifact_files is only supported for web-python apps"
            )
        # For web-app type, include additional fields needed by the template
        if app_type == "web-app":
            template_config.update(
                {
                    "url": final_config.get("url", ""),
                    "name": final_config.get("name", "Web App"),
                    "cookies": final_config.get("cookies", {}),
                    "local_storage": final_config.get("local_storage", {}),
                    "authorization_token": final_config.get("authorization_token", ""),
                }
            )

        compiled_html = template.render(
            hypha_hypha_rpc_version=hypha_rpc_version,
            hypha_rpc_websocket_mjs=f"{server_url}/assets/hypha-rpc-websocket.mjs",
            config=template_config,
            script=final_config.get("script", ""),
            requirements=final_config.get("requirements", []),
            local_base_url=server_url,
            source_hash=manifest.get("source_hash", ""),
        )

        return template_config, compiled_html

    async def take_screenshot(
        self,
        session_id: str,
        format: str = "png",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Take a screenshot of a browser app instance."""
        if session_id not in self._sessions:
            raise Exception(f"Browser app instance not found: {session_id}")

        session_data = self._session_data.get(session_id)
        if not session_data:
            raise Exception(f"Browser app session data not found: {session_id}")

        page = session_data["page"]

        # Validate format
        if format not in ["png", "jpeg"]:
            raise ValueError(f"Invalid format '{format}'. Must be 'png' or 'jpeg'")

        # Take screenshot
        screenshot = await page.screenshot(type=format)
        return screenshot

    async def _setup_page_authentication(self, page, manifest, target_url=None):
        """Setup authentication (cookies, local_storage, session_storage, headers) before navigation.
        
        Manifest options:
        - cookies: List of cookie objects or dict of name-value pairs
        - local_storage: Dict of key-value pairs to set in localStorage
        - session_storage: Dict of key-value pairs to set in sessionStorage
        - authorization_token: String to set as Authorization header
        """
        # Setup cookies with full configuration options
        cookies_config = manifest.get("cookies")
        if cookies_config:
            cookies = []
            
            # Determine default domain from target URL
            default_domain = None
            if target_url:
                parsed = urlparse(target_url)
                default_domain = parsed.netloc
            
            # Handle both list of cookie objects and dict of name-value pairs
            if isinstance(cookies_config, list):
                # Full cookie objects with all options
                for cookie in cookies_config:
                    cookie_obj = {
                        "name": cookie.get("name"),
                        "value": str(cookie.get("value", "")),
                        "domain": cookie.get("domain", default_domain or "localhost"),
                        "path": cookie.get("path", "/"),
                    }
                    # Add optional fields if present
                    if "expires" in cookie:
                        cookie_obj["expires"] = cookie["expires"]
                    if "httpOnly" in cookie:
                        cookie_obj["httpOnly"] = cookie["httpOnly"]
                    if "secure" in cookie:
                        cookie_obj["secure"] = cookie["secure"]
                    if "sameSite" in cookie:
                        cookie_obj["sameSite"] = cookie["sameSite"]
                    cookies.append(cookie_obj)
            elif isinstance(cookies_config, dict):
                # Simple name-value pairs (backward compatibility)
                for name, value in cookies_config.items():
                    cookies.append({
                        "name": name,
                        "value": str(value),
                        "domain": default_domain or "localhost",
                        "path": "/",
                    })
            
            if cookies:
                await page.context.add_cookies(cookies)
                logger.info(f"Setup {len(cookies)} cookies before navigation")
        
        # Setup authorization header
        auth_token = manifest.get("authorization_token")
        if auth_token:
            await page.set_extra_http_headers({"Authorization": auth_token})
            logger.info("Setup Authorization header before navigation")
        
        # Preload localStorage and sessionStorage using initialization script
        # This ensures storage is set BEFORE the page loads
        local_storage = manifest.get("local_storage", {})
        session_storage = manifest.get("session_storage", {})
        
        if local_storage or session_storage:
            # Create initialization script that runs before any page scripts
            init_script = ""
            
            if local_storage:
                init_script += "// Preload localStorage\n"
                for key, value in local_storage.items():
                    # Properly escape values for JavaScript
                    escaped_key = repr(str(key))
                    escaped_value = repr(str(value))
                    init_script += f"try {{ localStorage.setItem({escaped_key}, {escaped_value}); }} catch(e) {{ console.warn('Failed to set localStorage item:', e); }}\n"
                logger.info(f"Preloading {len(local_storage)} localStorage items")
            
            if session_storage:
                init_script += "// Preload sessionStorage\n"
                for key, value in session_storage.items():
                    # Properly escape values for JavaScript
                    escaped_key = repr(str(key))
                    escaped_value = repr(str(value))
                    init_script += f"try {{ sessionStorage.setItem({escaped_key}, {escaped_value}); }} catch(e) {{ console.warn('Failed to set sessionStorage item:', e); }}\n"
                logger.info(f"Preloading {len(session_storage)} sessionStorage items")
            
            # Add the initialization script to run on every page/frame
            await page.add_init_script(init_script)
            logger.info("Storage initialization script added")


    async def _setup_route_caching(
        self, page: Page, workspace: str, app_id: str, cache_routes: List[str]
    ) -> None:
        """Setup route interception for caching."""

        async def handle_route(route):
            """Handle intercepted routes for caching."""
            request = route.request
            url = request.url

            # Check if this URL should be cached
            if await self.cache_manager.should_cache_url(
                workspace, app_id, url, cache_routes
            ):
                # Try to get cached response first
                cached_entry = await self.cache_manager.get_cached_response(
                    workspace, app_id, url
                )

                if cached_entry:
                    # Serve from cache
                    logger.info(f"Serving cached response for {url}")
                    await route.fulfill(
                        status=cached_entry.status,
                        headers=cached_entry.headers,
                        body=cached_entry.body,
                    )
                    return

                # Not in cache, fetch and cache the response
                logger.info(f"Fetching and caching response for {url}")
                response = await route.fetch()

                # Cache the response
                body = await response.body()
                headers = dict(response.headers)

                await self.cache_manager.cache_response(
                    workspace, app_id, url, response.status, headers, body
                )

                # Fulfill the request with the fetched response
                await route.fulfill(status=response.status, headers=headers, body=body)
            else:
                # Not a cached route, continue normally
                await route.continue_()

        # Setup route interception for all requests
        await page.route("**/*", handle_route)
        logger.info(f"Setup route caching for {len(cache_routes)} patterns")

    async def clear_app_cache(
        self, workspace: str, app_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Clear cache for an app."""
        deleted_count = await self.cache_manager.clear_app_cache(workspace, app_id)
        return {"deleted_entries": deleted_count}

    async def get_app_cache_stats(
        self, workspace: str, app_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get cache statistics for an app."""
        return await self.cache_manager.get_cache_stats(workspace, app_id)

    def _apply_playwright_env_config(self, context_options: Dict[str, Any], manifest: Optional[Dict[str, Any]] = None) -> None:
        """Apply Playwright configuration from environment variables and manifest to context options.
        
        Args:
            context_options: Dictionary of context options to modify in-place
            manifest: Optional manifest configuration that can override environment variables
        """
        # Security & CSP Settings
        # Check manifest first for bypass_csp
        if manifest and 'bypass_csp' in manifest:
            context_options["bypass_csp"] = manifest['bypass_csp']
        else:
            bypass_csp = os.environ.get("PLAYWRIGHT_BYPASS_CSP")
            if bypass_csp is not None:
                context_options["bypass_csp"] = bypass_csp.lower() in ("true", "1", "yes")
            
        # Check manifest first for ignore_https_errors
        if manifest and 'ignore_https_errors' in manifest:
            context_options["ignore_https_errors"] = manifest['ignore_https_errors']
        else:
            ignore_https_errors = os.environ.get("PLAYWRIGHT_IGNORE_HTTPS_ERRORS")
            if ignore_https_errors is not None:
                context_options["ignore_https_errors"] = ignore_https_errors.lower() in ("true", "1", "yes")
            
        accept_downloads = os.environ.get("PLAYWRIGHT_ACCEPT_DOWNLOADS")
        if accept_downloads is not None:
            context_options["accept_downloads"] = accept_downloads.lower() in ("true", "1", "yes")
            
        permissions = os.environ.get("PLAYWRIGHT_PERMISSIONS")
        if permissions:
            # Split permissions by comma and strip whitespace
            context_options["permissions"] = [perm.strip() for perm in permissions.split(",") if perm.strip()]
        
        # Browser Identity & Localization
        user_agent = os.environ.get("PLAYWRIGHT_USER_AGENT")
        if user_agent:
            context_options["user_agent"] = user_agent
        
        locale = os.environ.get("PLAYWRIGHT_LOCALE")
        if locale:
            context_options["locale"] = locale
            
        timezone_id = os.environ.get("PLAYWRIGHT_TIMEZONE_ID")
        if timezone_id:
            context_options["timezone_id"] = timezone_id
        
        
        # Geolocation Configuration
        geolocation_lat = os.environ.get("PLAYWRIGHT_GEOLOCATION_LATITUDE")
        geolocation_lon = os.environ.get("PLAYWRIGHT_GEOLOCATION_LONGITUDE")
        geolocation_accuracy = os.environ.get("PLAYWRIGHT_GEOLOCATION_ACCURACY")
        if geolocation_lat and geolocation_lon:
            geolocation = {
                "latitude": float(geolocation_lat),
                "longitude": float(geolocation_lon)
            }
            if geolocation_accuracy:
                geolocation["accuracy"] = float(geolocation_accuracy)
            context_options["geolocation"] = geolocation
        
        # Viewport Configuration (override default if specified)
        viewport_width = os.environ.get("PLAYWRIGHT_VIEWPORT_WIDTH")
        viewport_height = os.environ.get("PLAYWRIGHT_VIEWPORT_HEIGHT")
        if viewport_width and viewport_height:
            context_options["viewport"] = {
                "width": int(viewport_width),
                "height": int(viewport_height)
            }
        
        # Device & Display Settings
        device_scale_factor = os.environ.get("PLAYWRIGHT_DEVICE_SCALE_FACTOR")
        if device_scale_factor:
            context_options["device_scale_factor"] = float(device_scale_factor)
        
        is_mobile = os.environ.get("PLAYWRIGHT_IS_MOBILE")
        if is_mobile is not None:
            context_options["is_mobile"] = is_mobile.lower() in ("true", "1", "yes")
            
        has_touch = os.environ.get("PLAYWRIGHT_HAS_TOUCH")
        if has_touch is not None:
            context_options["has_touch"] = has_touch.lower() in ("true", "1", "yes")
        
        # Accessibility & Preferences
        color_scheme = os.environ.get("PLAYWRIGHT_COLOR_SCHEME")
        if color_scheme and color_scheme.lower() in ["dark", "light", "no-preference", "null"]:
            context_options["color_scheme"] = color_scheme.lower()
        
        reduced_motion = os.environ.get("PLAYWRIGHT_REDUCED_MOTION")
        if reduced_motion and reduced_motion.lower() in ["no-preference", "null", "reduce"]:
            context_options["reduced_motion"] = reduced_motion.lower()
        
        # Network & JavaScript Settings
        java_script_enabled = os.environ.get("PLAYWRIGHT_JAVASCRIPT_ENABLED")
        if java_script_enabled is not None:
            context_options["java_script_enabled"] = java_script_enabled.lower() in ("true", "1", "yes")
        
        offline = os.environ.get("PLAYWRIGHT_OFFLINE")
        if offline is not None:
            context_options["offline"] = offline.lower() in ("true", "1", "yes")
        
        # HTTP Authentication
        http_username = os.environ.get("PLAYWRIGHT_HTTP_USERNAME")
        http_password = os.environ.get("PLAYWRIGHT_HTTP_PASSWORD")
        if http_username and http_password:
            context_options["http_credentials"] = {
                "username": http_username,
                "password": http_password
            }
        
        # Extra HTTP Headers - merge from manifest and environment
        if 'extra_http_headers' not in context_options:
            context_options['extra_http_headers'] = {}
            
        # First apply headers from manifest
        if manifest and 'extra_http_headers' in manifest:
            if isinstance(manifest['extra_http_headers'], dict):
                context_options['extra_http_headers'].update(manifest['extra_http_headers'])
            else:
                logger.warning(f"Invalid extra_http_headers in manifest, expected dict but got {type(manifest['extra_http_headers'])}")
        
        # Then apply headers from environment (these can override manifest headers)
        extra_headers = os.environ.get("PLAYWRIGHT_EXTRA_HTTP_HEADERS")
        if extra_headers:
            # Parse JSON format: {"Header-Name": "value", "Another-Header": "value"}
            try:
                import json
                env_headers = json.loads(extra_headers)
                context_options["extra_http_headers"].update(env_headers)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Invalid PLAYWRIGHT_EXTRA_HTTP_HEADERS format: {extra_headers}")
        
        # Clean up if no headers were added
        if not context_options['extra_http_headers']:
            del context_options['extra_http_headers']
        
        # Base URL for Relative URLs
        base_url = os.environ.get("PLAYWRIGHT_BASE_URL")
        if base_url:
            context_options["base_url"] = base_url

    async def execute(
        self,
        session_id: str,
        script: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute JavaScript code in a browser session.
        
        Args:
            session_id: The session ID to execute the script in
            script: JavaScript code to execute in the page context
            context: Optional context information
            
        Returns:
            The result of the script execution
            
        Raises:
            SessionNotFoundError: If the session doesn't exist
            WorkerError: If the script execution fails
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Browser session {session_id} not found")
        
        session_data = self._session_data.get(session_id)
        if not session_data or "page" not in session_data:
            raise WorkerError(f"No page available for session {session_id}")
        
        page = session_data["page"]
        
        try:
            # Execute the script in the page context
            result = await page.evaluate(script)
            logger.info(f"Executed script in session {session_id}")
            return result
        except Exception as e:
            error_msg = f"Failed to execute script in session {session_id}: {str(e)}"
            logger.error(error_msg)
            raise WorkerError(error_msg) from e
    
    def get_worker_service(self) -> Dict[str, Any]:
        """Get the service configuration for registration with browser-specific methods."""
        service_config = super().get_worker_service()
        # Add browser-specific methods
        service_config["take_screenshot"] = self.take_screenshot
        service_config["clear_app_cache"] = self.clear_app_cache
        service_config["get_app_cache_stats"] = self.get_app_cache_stats
        service_config["execute"] = self.execute
        return service_config


async def hypha_startup(server):
    """Hypha startup function to initialize browser worker."""
    worker = BrowserWorker(use_local_url=True)  # Built-in worker should use local URLs
    await worker.register_worker_service(server)
    logger.info("Browser worker initialized and registered")


def main():
    """Main function for command line execution."""
    import argparse
    import asyncio
    import sys

    def get_env_var(name: str, default: str = None) -> str:
        """Get environment variable with HYPHA_ prefix."""
        return os.environ.get(f"HYPHA_{name.upper()}", default)

    parser = argparse.ArgumentParser(
        description="Hypha Browser Worker - Execute web applications in isolated browser environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (with HYPHA_ prefix):
  HYPHA_SERVER_URL     Hypha server URL (e.g., https://hypha.aicell.io)
  HYPHA_WORKSPACE      Workspace name (e.g., my-workspace)
  HYPHA_TOKEN          Authentication token
  HYPHA_SERVICE_ID     Service ID for the worker (optional)
  HYPHA_VISIBILITY     Service visibility: public or protected (default: protected)
  HYPHA_IN_DOCKER      Set to 'true' if running in Docker container (default: false)

Environment Variables (with PLAYWRIGHT_ prefix):
  PLAYWRIGHT_BYPASS_CSP           Bypass Content Security Policy (true/false)
  PLAYWRIGHT_IGNORE_HTTPS_ERRORS  Ignore HTTPS certificate errors (true/false)
  PLAYWRIGHT_ACCEPT_DOWNLOADS     Allow file downloads (true/false, default: false)
  PLAYWRIGHT_PERMISSIONS          Comma-separated list of permissions to grant (e.g., "camera,microphone,geolocation")
  PLAYWRIGHT_USER_AGENT           Custom user agent string
  PLAYWRIGHT_LOCALE               Locale for the browser context (e.g., "en-US", "fr-FR")
  PLAYWRIGHT_TIMEZONE_ID          Timezone ID (e.g., "America/New_York", "Europe/London")
  PLAYWRIGHT_GEOLOCATION_LATITUDE Geolocation latitude (requires longitude)
  PLAYWRIGHT_GEOLOCATION_LONGITUDE Geolocation longitude (requires latitude)
  PLAYWRIGHT_GEOLOCATION_ACCURACY Geolocation accuracy in meters (optional)
  PLAYWRIGHT_VIEWPORT_WIDTH       Viewport width in pixels (overrides default 1280)
  PLAYWRIGHT_VIEWPORT_HEIGHT      Viewport height in pixels (overrides default 720)
  PLAYWRIGHT_DEVICE_SCALE_FACTOR  Device pixel ratio (e.g., "2" for high-DPI displays)
  PLAYWRIGHT_IS_MOBILE            Enable mobile mode (true/false)
  PLAYWRIGHT_HAS_TOUCH            Enable touch events (true/false)
  PLAYWRIGHT_COLOR_SCHEME         Color scheme preference (dark/light/no-preference/null)
  PLAYWRIGHT_REDUCED_MOTION       Reduced motion preference (no-preference/null/reduce)
  PLAYWRIGHT_JAVASCRIPT_ENABLED   Enable/disable JavaScript (true/false, default: true)
  PLAYWRIGHT_OFFLINE              Enable offline mode (true/false)
  PLAYWRIGHT_HTTP_USERNAME        HTTP basic auth username (requires password)
  PLAYWRIGHT_HTTP_PASSWORD        HTTP basic auth password (requires username)
  PLAYWRIGHT_EXTRA_HTTP_HEADERS   Extra HTTP headers as JSON (e.g., '{"X-API-Key": "value"}')
  PLAYWRIGHT_BASE_URL             Base URL for relative URLs

Examples:
  # Using command line arguments
  python -m hypha.workers.browser --server-url https://hypha.aicell.io --workspace my-workspace --token TOKEN

  # Using environment variables
  export HYPHA_SERVER_URL=https://hypha.aicell.io
  export HYPHA_WORKSPACE=my-workspace
  export HYPHA_TOKEN=your-token-here
  python -m hypha.workers.browser

  # Running in Docker
  export HYPHA_IN_DOCKER=true
  python -m hypha.workers.browser --server-url https://hypha.aicell.io --workspace my-workspace --token TOKEN
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
        "--in-docker",
        action="store_true",
        default=get_env_var("IN_DOCKER", "false").lower() == "true",
        help="Set if running in Docker container (default: from HYPHA_IN_DOCKER env var or false)",
    )
    parser.add_argument(
        "--use-local-url",
        action="store_true",
        help="Use local URLs for server communication (default: false for CLI workers)",
    )
    parser.add_argument(
        "--disable-ssl",
        action="store_true",
        help="Disable SSL verification (default: false)",
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

    print(f"Starting Hypha Browser Worker...")
    print(f"  Server URL: {args.server_url}")
    print(f"  Workspace: {args.workspace}")
    print(f"  Service ID: {args.service_id}")
    print(f"  Disable SSL: {args.disable_ssl}")
    print(f"  Visibility: {args.visibility}")
    print(f"  Use Local URL: {args.use_local_url}")
    print(f"  In Docker: {args.in_docker}")

    async def run_worker():
        """Run the browser worker."""
        try:
            from hypha_rpc import connect_to_server

            # Connect to server
            server = await connect_to_server(
                server_url=args.server_url, workspace=args.workspace, token=args.token, ssl=False if args.disable_ssl else None
            )

            # Create and register worker - use BrowserWorker directly
            worker = BrowserWorker(in_docker=args.in_docker, use_local_url=args.use_local_url)

            # Get service config and set custom properties (use get_worker_service() to include browser-specific methods)
            service_config = worker.get_worker_service()
            if args.service_id:
                service_config["id"] = args.service_id
            service_config["visibility"] = args.visibility

            # Register the service
            await server.rpc.register_service(service_config)

            print(f"✅ Browser Worker registered successfully!")
            print(f"   Service ID: {service_config['id']}")
            print(f"   Supported types: {worker.supported_types}")
            print(f"   Visibility: {args.visibility}")
            print(f"")
            print(f"Worker is ready to process browser application requests...")
            print(f"Press Ctrl+C to stop the worker.")

            # Keep the worker running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print(f"\n🛑 Shutting down Browser Worker...")
                await worker.shutdown()
                print(f"✅ Worker shutdown complete.")

        except Exception as e:
            print(f"❌ Failed to start Browser Worker: {e}", file=sys.stderr)
            sys.exit(1)

    # Run the worker
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
