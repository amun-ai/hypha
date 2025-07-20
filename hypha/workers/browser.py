"""Provide a browser worker."""

import os
import logging
import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from playwright.async_api import Page, async_playwright, Browser
from jinja2 import Environment, PackageLoader, select_autoescape

from hypha.workers.base import BaseWorker, WorkerConfig, SessionStatus, SessionInfo, SessionNotFoundError, WorkerError
from hypha.plugin_parser import parse_imjoy_plugin
from hypha import main_version
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
        
        # Register service with store since browser worker is created directly by server
        if store:
            store.register_public_service(self.get_service())

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["web-python", "web-worker", "window", "iframe", "hypha"]

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
        if config.progress_callback:
            config.progress_callback({"status": "Initializing browser worker..."})
            
        if not self.browser:
            await self.initialize()

        # Get app type from manifest
        app_type = config.manifest.get("type")
        if app_type not in self.supported_types:
            raise Exception(f"Browser worker only supports {self.supported_types}, got {app_type}")

        if config.progress_callback:
            config.progress_callback({"status": "Creating browser context..."})

        # Create a new context for isolation
        context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}, accept_downloads=True
        )

        # Create a new page in the context
        page = await context.new_page()

        logs = {}
        _capture_logs_from_browser_tabs(page, logs)
        
        # Add extra error handling for debugging
        page.on("requestfailed", lambda request: logger.error(f"Request failed: {request.url} - {request.failure}"))
        page.on("response", lambda response: logger.info(f"Response: {response.url} - {response.status}") if response.status != 200 else None)

        if config.progress_callback:
            config.progress_callback({"status": "Loading application..."})

        try:
            # Get the entry point from manifest - this should be a compiled HTML file uploaded to artifact manager
            entry_point = config.manifest.get("entry_point", "index.html")
            logger.info(f"Browser worker starting session with entry_point: {entry_point}, app_type: {app_type}")
            
            # Generate URLs for the app
            local_url, public_url = self._generate_app_urls(config, entry_point)

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

            logger.info("Browser app loaded successfully")
            
            # Wait a bit for JavaScript to initialize and services to be registered
            # This gives time for the api.export() call to complete
            await page.wait_for_timeout(1000)  # Wait 1 second for JS initialization
            logger.info("JavaScript initialization wait completed")

            if config.progress_callback:
                config.progress_callback({"status": "Application loaded successfully"})

            return {
                "local_url": local_url,
                "public_url": public_url,
                "page": page,
                "context": context,
                "logs": logs,
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
        
        try:
            session_data = self._session_data.get(session_id)
            if session_data:
                if "page" in session_data:
                    await session_data["page"].close()
                if "context" in session_data:
                    await session_data["context"].close()
            
            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped browser session {session_id}")
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop browser session {session_id}: {e}")
            raise
        finally:
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
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop browser session {session_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown the browser worker."""
        logger.info("Shutting down browser worker...")
        
        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop browser session {session_id}: {e}")
        
        # Shutdown browser and playwright
        try:
            if self.browser:
                await self.browser.close()

            if self._playwright:
                await self._playwright.stop()

            logger.info("Browser worker closed successfully.")
        except Exception as e:
            logger.error("Error during browser shutdown: %s", str(e))
            raise
        
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
        
        if progress_callback:
            progress_callback({"status": "Starting browser app compilation..."})
        
        app_type = manifest.get("type")
        if app_type and app_type not in self.supported_types:
            # Not a browser app type, return as-is
            raise Exception(f"Browser worker only supports {self.supported_types}, got {app_type}")
        if app_type is None:
            logger.warning("No app type found in manifest, using default app type")
            
        if progress_callback:
            progress_callback({"status": f"Compiling {app_type} application..."})
        
        # Look for different types of source files
        source_file = None
        config_file = None
        script_file = None
        source_content = ""
        
        files_by_name = {f.get("name"): f for f in files}
        
        # Check for source file (traditional approach)
        if "source" in files_by_name:
            source_file = files_by_name["source"]
            source_content = source_file.get("content", "")
        
        # Only process config/script files for compilation if there's a source file
        # This indicates they were extracted from XML, not provided directly by user
        if source_file:
            if progress_callback:
                progress_callback({"status": "Processing source files and configurations..."})
        else:
            # No source file, but we still need to ensure all browser apps have index.html as entry_point
            # This handles cases where files are provided directly (not as source)
            updated_manifest = manifest.copy()
            updated_manifest["entry_point"] = "index.html"
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
        
        if progress_callback:
            progress_callback({"status": "Compiling source code to HTML template..."})
            
        # Compile the source to HTML
        compiled_config, compiled_html = await self._compile_source_to_html(source_content, app_type, manifest, config)
        new_manifest = manifest.copy()
        new_manifest.update(compiled_config)
        app_type = new_manifest.get("type")
        
        if progress_callback:
            progress_callback({"status": "Updating manifest and preparing files..."})
        
        # All browser apps should compile to index.html as the entry point
        # This ensures consistency and that the browser worker can find the file
        compiled_entry_point = "index.html"
        
        # Set the entry point to index.html for all browser app types
        new_manifest["entry_point"] = compiled_entry_point
        # Ensure the manifest type is correctly set to the expected app_type
        new_manifest["type"] = new_manifest.get("type", app_type)
        
        
        
        # Create new files list without the source/config/script files and add compiled file
        files_to_remove = set(["source"])
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
                "name": "source",
                "content": new_manifest["code"],
                "format": "text"
            })
            del new_manifest["code"]
        
        if progress_callback:
            progress_callback({"status": f"Browser app compilation completed. Generated {compiled_entry_point}"})
            
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
                        hypha_main_version=main_version,
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
        compiled_html = template.render(
            hypha_main_version=main_version,
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

    def get_service(self):
        """Get the service."""
        service_config = self.get_service_config()
        # Add browser-specific methods
        service_config["take_screenshot"] = self.take_screenshot
        return service_config
