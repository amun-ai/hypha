"""Provide a browser worker."""

import os
import logging
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from urllib.parse import urlparse

from playwright.async_api import Page, async_playwright, Browser
from jinja2 import Environment, PackageLoader, select_autoescape

from hypha.workers.base import BaseWorker, WorkerConfig, SessionStatus, SessionInfo, SessionNotFoundError, WorkerError
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


class BrowserAppRunner(BaseWorker):
    """Browser app worker."""

    instance_counter: int = 0

    def __init__(
        self,
        store,
        in_docker: bool = False,
    ):
        """Initialize the class."""
        super().__init__(store)
        self.browser: Optional[Browser] = None
        self.controller_id = str(BrowserAppRunner.instance_counter)
        BrowserAppRunner.instance_counter += 1
        self.in_docker = in_docker
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
        if store and hasattr(store, '_redis'):
            self.cache_manager = BrowserCache(store._redis)
        
        # Register service with store since browser worker is created directly by server
        if store:
            store.register_public_service(self.get_service())

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["web-python", "web-worker", "window", "iframe", "hypha", "web-app"]

    @property
    def worker_name(self) -> str:
        """Return the worker name."""
        return "Browser Worker"

    @property
    def worker_description(self) -> str:
        """Return the worker description."""
        return "A worker for running web applications in browser environments"

    async def initialize(self) -> Browser:
        """Initialize the browser worker."""
        if self.initialized:
            return self.browser
            
        self._playwright = await async_playwright().start()
        args = [
            "--site-per-process",
            "--enable-unsafe-webgpu",
            "--use-vulkan",
            "--enable-features=Vulkan,WebAssemblyJSPI",
            "--enable-experimental-web-platform-features",
        ]
        # so it works in the docker image
        if self.in_docker:
            args.append("--no-sandbox")

        self.browser = await self._playwright.chromium.launch(
            args=args, handle_sigint=True, handle_sigterm=True, handle_sighup=True
        )
        self.initialized = True
        return self.browser

    async def start(self, config: Union[WorkerConfig, Dict[str, Any]]) -> str:
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
            metadata=config.manifest
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
        config.progress_callback({"type": "info", "message": "Initializing browser worker..."})
            
        if not self.browser:
            await self.initialize()

        # Get app type from manifest
        app_type = config.manifest.get("type")
        if app_type not in self.supported_types:
            raise Exception(f"Browser worker only supports {self.supported_types}, got {app_type}")

        config.progress_callback({"type": "info", "message": "Creating browser context..."})

        # Create a new context for isolation
        context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}, accept_downloads=True
        )

        # Create a new page in the context
        page = await context.new_page()

        # Setup cookies, localStorage, and other authentication before loading the page
        await self._setup_page_authentication(page, config.manifest)

        logs = {}
        _capture_logs_from_browser_tabs(page, logs)
        
        # Add extra error handling for debugging
        page.on("requestfailed", lambda request: logger.error(f"Request failed: {request.url} - {request.failure}"))
        page.on("response", lambda response: logger.info(f"Response: {response.url} - {response.status}") if response.status != 200 else None)

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
                cache_routes = self.cache_manager.get_default_cache_routes_for_type(app_type)
                
            if enable_cache and cache_routes:
                cache_enabled = True
                await self._setup_route_caching(page, config.workspace, config.app_id, cache_routes)
                await self.cache_manager.start_recording(config.workspace, config.app_id)

        config.progress_callback({"type": "info", "message": "Loading application..."})

        try:
            # Get the entry point from manifest - this should be a compiled HTML file uploaded to artifact manager
            entry_point = config.manifest.get("entry_point", "index.html")
            logger.info(f"Browser worker starting session with entry_point: {entry_point}, app_type: {app_type}")
            
            # Generate URLs for the app
            local_url, public_url = self._generate_app_urls(config, entry_point)

            # For web-app type, navigate directly to the entry_point URL
            if app_type == "web-app":
                external_url = entry_point  # Use entry_point as the external URL
                logger.info(f"Loading web-app from external URL: {external_url} with timeout {timeout_ms}ms")
                response = await page.goto(external_url, timeout=timeout_ms, wait_until="load")
            else:
                logger.info(f"Loading browser app from URL: {local_url} with timeout {timeout_ms}ms")
                response = await page.goto(local_url, timeout=timeout_ms, wait_until="load")

            if not response:
                await context.close()
                raise Exception(f"Failed to load URL: {local_url}")

            if response.status != 200:
                await context.close()
                raise Exception(
                    f"Failed to start browser app instance, "
                    f"status: {response.status}, url: {local_url}"
                )

            # Apply localStorage after navigation (requires loaded page)
            if hasattr(page, '_pending_local_storage') and page._pending_local_storage:
                for key, value in page._pending_local_storage.items():
                    await page.evaluate(f"localStorage.setItem({repr(key)}, {repr(str(value))})")
                logger.info(f"Applied {len(page._pending_local_storage)} localStorage items")
                delattr(page, '_pending_local_storage')

            logger.info("Browser app loaded successfully")
            
            # Wait a bit for JavaScript to initialize and services to be registered
            # This gives time for the api.export() call to complete
            await page.wait_for_timeout(1000)  # Wait 1 second for JS initialization
            logger.info("JavaScript initialization wait completed")

            config.progress_callback({"type": "success", "message": "Application loaded successfully"})

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

    async def stop(self, session_id: str) -> None:
        """Stop a browser session."""
        if session_id not in self._sessions:
            logger.warning(f"Browser session {session_id} not found for stopping, may have already been cleaned up")
            return
        
        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING
        
        session_data = self._session_data.get(session_id)
        if session_data:
            # Stop cache recording if it was enabled
            if session_data.get("cache_enabled") and self.cache_manager:
                await self.cache_manager.stop_recording(session_info.workspace, session_info.app_id)
                
            if "page" in session_data:
                await session_data["page"].close()
            if "context" in session_data:
                await session_data["context"].close()
        
        session_info.status = SessionStatus.STOPPED
        logger.info(f"Stopped browser session {session_id}")
        
        # Cleanup
        self._sessions.pop(session_id, None)
        self._session_data.pop(session_id, None)

    async def list_sessions(self, workspace: str) -> List[SessionInfo]:
        """List all browser sessions for a workspace."""
        return [
            session_info for session_info in self._sessions.values()
            if session_info.workspace == workspace
        ]

    async def get_session_info(self, session_id: str) -> SessionInfo:
        """Get information about a browser session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Browser session {session_id} not found")
        return self._sessions[session_id]

    async def get_logs(
        self, 
        session_id: str, 
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a browser session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Browser session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {} if type is None else []

        logs = session_data.get("logs", {})
        
        if type:
            target_logs = logs.get(type, [])
            end_idx = len(target_logs) if limit is None else min(offset + limit, len(target_logs))
            return target_logs[offset:end_idx]
        else:
            result = {}
            for log_type_key, log_entries in logs.items():
                end_idx = len(log_entries) if limit is None else min(offset + limit, len(log_entries))
                result[log_type_key] = log_entries[offset:end_idx]
            return result

    async def prepare_workspace(self, workspace: str) -> None:
        """Prepare workspace for browser operations."""
        logger.info(f"Preparing workspace {workspace} for browser worker")
        pass

    async def close_workspace(self, workspace: str) -> None:
        """Close all browser sessions for a workspace."""
        logger.info(f"Closing workspace {workspace} for browser worker")
        
        # Stop all sessions for this workspace
        sessions_to_stop = [
            session_id for session_id, session_info in self._sessions.items()
            if session_info.workspace == workspace
        ]
        
        for session_id in sessions_to_stop:
            await self.stop(session_id)
        
        # Clear all cache entries for this workspace if cache manager is available
        if self.cache_manager:
            # Note: This is a simplified approach. In a production system,
            # you might want to be more selective about cache clearing
            logger.info(f"Cache cleanup for workspace {workspace} would be handled by app uninstall")

    async def shutdown(self) -> None:
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
        server_url = config.server_url or "http://127.0.0.1:38283"  # Fallback for testing
        params = [
            f"server_url={server_url}",
            f"client_id={config.client_id}",
            f"workspace={config.workspace}",
            f"app_id={config.app_id}",
            "use_proxy=true"
        ]
        
        if config.token:
            params.append(f"token={config.token}")
        
        # Check if entry_point is already a full URL
        if entry_point.startswith("http"):
            base_url = entry_point
        else:
            # Construct URL from artifact_id and server_url
            workspace_id, app_id = config.artifact_id.split('/', 1)
            base_url = f"{server_url}/{workspace_id}/artifacts/{app_id}/files/{entry_point}"
        
        return f"{base_url}?{'&'.join(params)}"
    
    def _generate_app_urls(self, config: WorkerConfig, entry_point: str) -> tuple[str, str]:
        """Generate local and public URLs for the app entry point."""
        if entry_point.startswith("http"):
            # External URL - use as-is but still add parameters
            local_url = self._build_app_url(config, entry_point)
            public_url = self._build_app_url(config, entry_point)
        else:
            # Construct URL from artifact_id and server_url
            workspace_id, app_id = config.artifact_id.split('/', 1)
            server_url = config.server_url or "http://127.0.0.1:38283"  # Fallback for testing
            base_url = f"{server_url}/{workspace_id}/artifacts/{app_id}/files/{entry_point}"
            
            local_url = self._build_app_url(config, base_url)
            public_url = self._build_app_url(config, base_url)
        
        return local_url, public_url

    async def compile(self, manifest: dict, files: list, config: dict = None) -> tuple[dict, list]:
        """Compile browser app manifest and files.
        
        This method:
        1. Looks for 'source' file OR config/script files in the files list
        2. Compiles them to 'index.html' using appropriate template
        3. Updates manifest with correct entry_point
        4. Returns updated manifest and files
        """
        # Extract progress_callback from config
        progress_callback = config.get("progress_callback") if config else None
        
        progress_callback({"type": "info", "message": "Starting browser app compilation..."})
        
        app_type = manifest.get("type")
        if app_type and app_type not in self.supported_types:
            # Not a browser app type, return as-is
            raise Exception(f"Browser worker only supports {self.supported_types}, got {app_type}")
        if app_type is None:
            logger.warning("No app type found in manifest, using default app type")
            
        # Handle web-app type specially - it doesn't need compilation, just validation
        if app_type == "web-app":
            progress_callback({"type": "info", "message": "Processing web-app configuration..."})
            
            # Validate required fields for web-app
            if not manifest.get("entry_point"):
                raise Exception("web-app type requires 'entry_point' field in manifest")
            
            new_manifest = manifest.copy()
            new_manifest["type"] = "web-app"
            # No entry_point needed since we navigate directly to the URL
            
            # No files needed for web-app type
            new_files = []
            
            progress_callback({"type": "success", "message": "Web-app configuration processed successfully"})
            return new_manifest, new_files
            
        progress_callback({"type": "info", "message": f"Compiling {app_type} application..."})
        
        # Look for different types of source files
        source_file = None
        config_file = None
        script_file = None
        source_content = ""
        
        files_by_name = {f.get("name"): f for f in files}
        
        entry_point = manifest["entry_point"]
        # Check for source file (traditional approach)
        if entry_point in files_by_name:
            source_file = files_by_name[entry_point]
            if source_file.get("content") is None:
                raise Exception(f"Source file {source_file.get('name')} is empty")
            source_content = source_file.get("content") or ""
        
        # Only process config/script files for compilation if there's a source file
        # This indicates they were extracted from XML, not provided directly by user
        if source_file:
            progress_callback({"type": "info", "message": "Processing source files and configurations..."})
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
                file_name = file.get("name", "")
                if file_name.endswith(".html") and not found_html_file:
                    # Rename the first HTML file to index.html
                    updated_file = file.copy()
                    updated_file["name"] = "index.html"
                    updated_files.append(updated_file)
                    found_html_file = True
                elif file_name != "index.html" or not found_html_file:
                    # Keep other files as-is, but avoid duplicating index.html
                    updated_files.append(file)
            
            return updated_manifest, updated_files
        
        progress_callback({"type": "info", "message": "Compiling source code to HTML template..."})
            
        # Compile the source to HTML
        compiled_config, compiled_html = await self._compile_source_to_html(source_content, app_type, manifest, config)
        new_manifest = manifest.copy()
        new_manifest.update(compiled_config)
        app_type = new_manifest.get("type")
        
        progress_callback({"type": "info", "message": "Updating manifest and preparing files..."})
        
        # All browser apps should compile to index.html as the entry point
        # This ensures consistency and that the browser worker can find the file
        compiled_entry_point = "index.html"
        
        # Set the entry point to index.html for all browser app types
        new_manifest["entry_point"] = compiled_entry_point
        # Ensure the manifest type is correctly set to the expected app_type
        new_manifest["type"] = new_manifest.get("type", app_type)
        
        
        
        # Create new files list without the source/config/script files and add compiled file
        files_to_remove = set([entry_point])
        if config_file:
            files_to_remove.add(config_file["name"])
        if script_file:
            files_to_remove.add(script_file["name"])
        
        new_files = [f for f in files if f.get("name") not in files_to_remove]
        
        # Always save the compiled HTML as index.html for consistency
        new_files.append({
            "name": compiled_entry_point,
            "content": compiled_html,
            "format": "text"
        })
        if "scripts" in new_manifest:
            del new_manifest["scripts"]
        if "script" in new_manifest:
            del new_manifest["script"]
        if "code" in new_manifest:
            new_files.append({
                "name": entry_point,
                "content": new_manifest["code"],
                "format": "text"
            })
            del new_manifest["code"]
        
        progress_callback({"type": "success", "message": f"Browser app compilation completed. Generated {compiled_entry_point}"})
            
        return new_manifest, new_files
    
    async def _compile_source_to_html(self, source: str, app_type: str, manifest: dict, config: dict = None) -> str:
        """Compile source code to HTML using templates."""
        # Check if it's raw HTML
        if source.lower().strip().startswith(("<!doctype html", "<html")):
            # Raw HTML - return as-is
            return {}, source
        
        # Handle hypha app type with legacy compilation logic
        if app_type == "hypha":
            try:
                template_config = parse_imjoy_plugin(source)
                if template_config:
                    # Merge template config with manifest
                    final_config = manifest.copy()
                    final_config.update(template_config)
                    if not final_config.get("type"):
                        raise Exception("No app type found in manifest")
                    final_config["source_hash"] = manifest.get("source_hash", "")
                    entry_point = template_config.get("entry_point", "index.html")
                    final_config["entry_point"] = entry_point

                    template = self.jinja_env.get_template(
                        safe_join("apps", final_config["type"] + "." + entry_point)
                    )

                    # Get server URL from config or use fallback
                    server_url = config.get("server_url", "http://127.0.0.1:38283") if config else "http://127.0.0.1:38283"
                    
                    template_config = {k: final_config[k] for k in final_config if k in PLUGIN_CONFIG_FIELDS}
                    template_config["server_url"] = server_url
                    compiled_html = template.render(
                        hypha_hypha_rpc_version=hypha_rpc_version,
                        hypha_rpc_websocket_mjs=f"{server_url}/assets/hypha-rpc-websocket.mjs",
                        config=template_config,
                        script=final_config["script"],
                        requirements=final_config["requirements"],
                        local_base_url=server_url,
                    )
                    
                    return template_config, compiled_html
                else:
                    raise Exception("Failed to parse hypha plugin config")
            except Exception as err:
                raise Exception(f"Failed to parse or compile the hypha app: {err}") from err
        
        # We need to check if source has top level xml tags
        # If not, we need to wrap it as a script tag, if type=window/iframe/web-worker we use js script tag
        # If type=web-python we use python script tag

        if not source.strip().startswith("<"):
            final_config = {"script": source}
        else:
            final_config = parse_imjoy_plugin(source)

        final_config.update(manifest)
        
        app_type = app_type or final_config.get("type")
        
        assert app_type in self.supported_types, f"Browser worker only supports {self.supported_types}, got {app_type}"
        
        # Determine template file
        entry_point = final_config.get("entry_point", "index.html")
        template_name = safe_join("apps", f"{app_type}.{entry_point}")
        
        try:
            template = self.jinja_env.get_template(template_name)
        except Exception:
            # Fallback to generic template
            template = self.jinja_env.get_template(safe_join("apps", f"{app_type}.index.html"))
        
        # Get server URL from config or use fallback
        server_url = config.get("server_url", "http://127.0.0.1:38283") if config else "http://127.0.0.1:38283"
        
        # Render the template with actual URLs
        template_config = {k: final_config[k] for k in final_config if k in PLUGIN_CONFIG_FIELDS}
        template_config["server_url"] = server_url
        
        # For web-app type, include additional fields needed by the template
        if app_type == "web-app":
            template_config.update({
                "url": final_config.get("url", ""),
                "name": final_config.get("name", "Web App"),
                "cookies": final_config.get("cookies", {}),
                "localStorage": final_config.get("localStorage", {}),
                "authorization_token": final_config.get("authorization_token", "")
            })
        
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
        
        # Take screenshot and return as base64
        screenshot = await page.screenshot(type=format)
        return screenshot

    async def _setup_page_authentication(self, page, manifest, target_url=None):
        """Setup authentication (cookies, localStorage, headers) before navigation."""
        # Setup cookies
        cookies_config = manifest.get("cookies", {})
        if cookies_config:
            # Convert to Playwright cookie format
            cookies = []
            if target_url:
                parsed = urlparse(target_url)
                domain = parsed.netloc
                for name, value in cookies_config.items():
                    cookies.append({
                        "name": name,
                        "value": str(value),
                        "domain": domain,
                        "path": "/"
                    })
            else:
                # For local apps, use localhost
                for name, value in cookies_config.items():
                    cookies.append({
                        "name": name,
                        "value": str(value),
                        "domain": "localhost",
                        "path": "/"
                    })
            
            await page.context.add_cookies(cookies)
            logger.info(f"Setup {len(cookies)} cookies before navigation")
        
        # Setup authorization header
        auth_token = manifest.get("authorization_token")
        if auth_token:
            await page.set_extra_http_headers({"Authorization": auth_token})
            logger.info("Setup Authorization header before navigation")
        
        # localStorage will be set after navigation since it requires the page to be loaded
        page._pending_local_storage = manifest.get("localStorage", {})

    async def _apply_pending_page_setup(self, page: Page, url: str) -> None:
        """Apply cookies and localStorage after page navigation."""
        # Set cookies with correct domain
        if hasattr(page, '_pending_cookies'):
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            for cookie in page._pending_cookies:
                cookie["domain"] = domain
            
            await page.context.add_cookies(page._pending_cookies)
            delattr(page, '_pending_cookies')
        
        # Set localStorage
        if hasattr(page, '_pending_local_storage'):
            for key, value in page._pending_local_storage.items():
                await page.evaluate(f"localStorage.setItem({repr(key)}, {repr(str(value))})")
            delattr(page, '_pending_local_storage')

    async def _setup_route_caching(self, page: Page, workspace: str, app_id: str, cache_routes: List[str]) -> None:
        """Setup route interception for caching."""
        async def handle_route(route):
            """Handle intercepted routes for caching."""
            request = route.request
            url = request.url
            
            # Check if this URL should be cached
            if await self.cache_manager.should_cache_url(workspace, app_id, url, cache_routes):
                # Try to get cached response first
                cached_entry = await self.cache_manager.get_cached_response(workspace, app_id, url)
                
                if cached_entry:
                    # Serve from cache
                    logger.info(f"Serving cached response for {url}")
                    await route.fulfill(
                        status=cached_entry.status,
                        headers=cached_entry.headers,
                        body=cached_entry.body
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
                await route.fulfill(
                    status=response.status,
                    headers=headers,
                    body=body
                )
            else:
                # Not a cached route, continue normally
                await route.continue_()
        
        # Setup route interception for all requests
        await page.route("**/*", handle_route)
        logger.info(f"Setup route caching for {len(cache_routes)} patterns")

    async def clear_app_cache(self, workspace: str, app_id: str) -> Dict[str, Any]:
        """Clear cache for an app."""
        deleted_count = await self.cache_manager.clear_app_cache(workspace, app_id)
        return {"deleted_entries": deleted_count}

    async def get_app_cache_stats(self, workspace: str, app_id: str) -> Dict[str, Any]:
        """Get cache statistics for an app."""
        return await self.cache_manager.get_cache_stats(workspace, app_id)

    def get_service(self):
        """Get the service."""
        service_config = self.get_service_config()
        # Add browser-specific methods
        service_config["take_screenshot"] = self.take_screenshot
        service_config["clear_app_cache"] = self.clear_app_cache
        service_config["get_app_cache_stats"] = self.get_app_cache_stats
        return service_config
