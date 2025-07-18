"""Provide a browser worker."""

import uuid
import os
import logging
import sys
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import Page, async_playwright, Browser, BrowserContext

from hypha.core.store import RedisStore
from hypha.workers.base import BaseWorker, WorkerConfig, SessionStatus

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
        store.register_public_service(self.get_service())
        self.in_docker = in_docker
        self._playwright = None

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["web-python", "web-worker", "window", "iframe"]

    @property
    def worker_name(self) -> str:
        """Return the worker name."""
        return "Browser Worker"

    @property
    def worker_description(self) -> str:
        """Return the worker description."""
        return "A worker for running web applications in browser environments"

    async def _initialize_worker(self) -> None:
        """Initialize the browser worker."""
        await self.initialize()

    async def initialize(self) -> Browser:
        """Initialize the app controller."""
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
        return self.browser

    async def _shutdown_worker(self) -> None:
        """Close the browser worker."""
        logger.info("Closing the browser worker...")
        try:
            if self.browser:
                await self.browser.close()

            if self._playwright:
                await self._playwright.stop()

            logger.info("Browser worker closed successfully.")
        except Exception as e:
            logger.error("Error during browser shutdown: %s", str(e))
            raise

    async def _start_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Start a browser app session."""
        if not self.browser:
            await self.initialize()

        # Get app type from manifest
        app_type = config.manifest.get("type")
        if app_type not in self.supported_types:
            raise Exception(f"Browser worker only supports {self.supported_types}, got {app_type}")

        # Create URLs for the app
        entry_point = config.entry_point
        if not entry_point.startswith("http"):
            # Use local_url from manifest if available
            local_url = config.manifest.get('local_url', '')
            if local_url:
                entry_point = local_url
            else:
                # Fallback to constructing from artifact_id
                # artifact_id is in format "workspace/app_id", need to split it
                workspace_id, app_id = config.artifact_id.split('/', 1)
                entry_point = f"{config.server_url}/{workspace_id}/artifacts/{app_id}/files/{config.entry_point}"
        
        # Create local and public URLs
        local_url = self._build_app_url(config, entry_point)
        public_url = self._build_app_url(config, entry_point)

        # Create a new context for isolation
        context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}, accept_downloads=True
        )

        # Create a new page in the context
        page = await context.new_page()

        logs = {}
        _capture_logs_from_browser_tabs(page, logs)

        logger.info("Loading browser app: %s", local_url)
        response = await page.goto(local_url, timeout=60000, wait_until="load")

        if not response:
            await context.close()
            raise Exception(f"Failed to load URL: {local_url}")

        if response.status != 200:
            await context.close()
            raise Exception(
                f"Failed to start browser app instance, "
                f"status: {response.status}, url: {local_url}"
            )

        logger.info("Browser app loaded successfully: %s", local_url)

        return {
            "local_url": local_url,
            "public_url": public_url,
            "page": page,
            "context": context,
            "logs": logs,
        }

    def _build_app_url(self, config: WorkerConfig, entry_point: str) -> str:
        """Build the app URL with parameters."""
        params = [
            f"server_url={config.server_url}",
            f"client_id={config.client_id}",
            f"workspace={config.workspace}",
            f"app_id={config.app_id}",
            "use_proxy=true"
        ]
        
        if config.token:
            params.append(f"token={config.token}")
        
        return f"{entry_point}?{'&'.join(params)}"



    async def _stop_session(self, session_id: str) -> None:
        """Stop a browser app instance."""
        session_data = self._session_data.get(session_id)
        if not session_data:
            return

        try:
            if "page" in session_data:
                await session_data["page"].close()
            if "context" in session_data:
                await session_data["context"].close()
            logger.info(f"Successfully stopped browser session: {session_id}")
        except Exception as e:
            logger.error(f"Error stopping browser session {session_id}: {str(e)}")
            raise

    async def _get_session_logs(
        self, 
        session_id: str, 
        log_type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a browser session."""
        session_data = self._session_data.get(session_id)
        if not session_data:
            return {} if log_type is None else []

        logs = session_data.get("logs", {})
        
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

    # get_logs method is now inherited from BaseWorker

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

    async def _close_workspace(self, workspace: str) -> None:
        """Close all browser app instances for a workspace."""
        # This is now handled by the base class
        pass

    async def _prepare_workspace(self, workspace: str) -> None:
        """Prepare the workspace for the browser app."""
        pass

    def get_service(self):
        """Get the service."""
        service_config = self.get_service_config()
        # Add browser-specific methods
        service_config["take_screenshot"] = self.take_screenshot
        return service_config
