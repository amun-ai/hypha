"""Provide a browser runner."""

import uuid
import os
import logging
import sys
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import Page, async_playwright, Browser, BrowserContext

from hypha.core.store import RedisStore

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


class BrowserAppRunner:
    """Browser app runner."""

    instance_counter: int = 0

    def __init__(
        self,
        store,
        in_docker: bool = False,
    ):
        """Initialize the class."""
        self.browser: Optional[Browser] = None
        self._browser_sessions: Dict[str, Dict[str, Any]] = {}
        self.controller_id = str(BrowserAppRunner.instance_counter)
        BrowserAppRunner.instance_counter += 1
        store.register_public_service(self.get_service())
        self.in_docker = in_docker
        self._initialized = False
        self._playwright = None

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
        self._initialized = True
        return self.browser

    async def shutdown(self) -> None:
        """Close the app controller."""
        logger.info("Closing the browser app controller...")
        try:
            # Close all active sessions first
            session_ids = list(self._browser_sessions.keys())
            for session_id in session_ids:
                await self.stop(session_id)

            if self.browser:
                await self.browser.close()

            if self._playwright:
                await self._playwright.stop()

            self._initialized = False
            logger.info("Browser app controller closed successfully.")
        except Exception as e:
            logger.error("Error during browser shutdown: %s", str(e))
            raise

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
        """Start a browser app instance."""
        # Handle backward compatibility
        if url and session_id:
            return await self._start_legacy(url, session_id, metadata)
        
        # New logic for type-based starting
        full_session_id = f"{workspace}/{client_id}"
        
        if not self.browser:
            await self.initialize()

        try:
            # Create URLs for the app
            if not entry_point.startswith("http"):
                entry_point = f"{local_base_url}/{workspace}/artifacts/{app_id}/files/{entry_point}"
            
            # Create local and public URLs
            local_url = (
                f"{entry_point}?"
                + f"server_url={server_url}&client_id={client_id}&workspace={workspace}"
                + f"&app_id={app_id}"
                + f"&server_url={server_url}"
                + (f"&token={token}" if token else "")
                + (f"&version={version}" if version else "")
                + (f"&use_proxy=true")
            )
            
            public_url = (
                f"{public_base_url}/{workspace}/artifacts/{app_id}/files/{entry_point}?"
                + f"client_id={client_id}&workspace={workspace}"
                + f"&app_id={app_id}"
                + f"&server_url={public_base_url}"
                + (f"&token={token}" if token else "")
                + (f"&version={version}" if version else "")
                + (f"&use_proxy=true")
            )

            # Create a new context for isolation
            context = await self.browser.new_context(
                viewport={"width": 1280, "height": 720}, accept_downloads=True
            )

            # Create a new page in the context
            page = await context.new_page()

            logs = {}
            session_data = {
                "session_id": full_session_id,
                "url": local_url,
                "local_url": local_url,
                "public_url": public_url,
                "status": "connecting",
                "page": page,
                "context": context,
                "logs": logs,
                "metadata": metadata,
                "app_type": app_type,
            }
            
            self._browser_sessions[full_session_id] = session_data

            _capture_logs_from_browser_tabs(page, logs)

            logger.info("Loading browser app: %s", local_url)
            response = await page.goto(local_url, timeout=60000, wait_until="load")

            if not response:
                raise Exception(f"Failed to load URL: {local_url}")

            if response.status != 200:
                raise Exception(
                    f"Failed to start browser app instance, "
                    f"status: {response.status}, url: {local_url}"
                )

            logger.info("Browser app loaded successfully: %s", local_url)
            self._browser_sessions[full_session_id]["status"] = "connected"

            return {
                "session_id": full_session_id,
                "status": "connected",
                "url": local_url,
                "local_url": local_url,
                "public_url": public_url,
            }

        except Exception as e:
            logger.error("Error starting browser session: %s", str(e))
            if full_session_id in self._browser_sessions:
                await self.stop(full_session_id)
            raise

    async def _start_legacy(
        self,
        url: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Legacy start method for backward compatibility."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        if not self.browser:
            await self.initialize()

        try:
            # Create a new context for isolation
            context = await self.browser.new_context(
                viewport={"width": 1280, "height": 720}, accept_downloads=True
            )

            # Create a new page in the context
            page = await context.new_page()

            logs = {}
            self._browser_sessions[session_id] = {
                "url": url,
                "status": "connecting",
                "page": page,
                "context": context,
                "logs": logs,
                "metadata": metadata,
            }

            _capture_logs_from_browser_tabs(page, logs)

            logger.info("Loading browser app: %s", url)
            response = await page.goto(url, timeout=60000, wait_until="load")

            if not response:
                raise Exception(f"Failed to load URL: {url}")

            if response.status != 200:
                raise Exception(
                    f"Failed to start browser app instance, "
                    f"status: {response.status}, url: {url}"
                )

            logger.info("Browser app loaded successfully: %s", url)
            self._browser_sessions[session_id]["status"] = "connected"

            return {
                "session_id": session_id,
                "status": "connected",
                "url": url,
            }

        except Exception as e:
            logger.error("Error starting browser session: %s", str(e))
            if session_id in self._browser_sessions:
                await self.stop(session_id)
            raise

    async def stop(self, session_id: str) -> None:
        """Stop a browser app instance."""
        if session_id not in self._browser_sessions:
            logger.warning(f"Browser app instance not found: {session_id}")
            return

        try:
            session = self._browser_sessions[session_id]
            if "page" in session:
                await session["page"].close()
            if "context" in session:
                await session["context"].close()
            del self._browser_sessions[session_id]
            logger.info(f"Successfully stopped browser session: {session_id}")
        except Exception as e:
            logger.error(f"Error stopping browser session {session_id}: {str(e)}")
            raise

    async def list(self, workspace) -> List[Dict[str, Any]]:
        """List the browser apps for the current user."""
        sessions = [
            {k: v for k, v in page_info.items() if k not in ["page", "context"]}
            for session_id, page_info in self._browser_sessions.items()
            if session_id.startswith(workspace + "/")
        ]
        return sessions

    async def logs(
        self,
        session_id: str,
        type: str = None,  # pylint: disable=redefined-builtin
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get the logs for a browser app instance."""
        if session_id not in self._browser_sessions:
            raise Exception(f"Browser app instance not found: {session_id}")

        if type is None:
            return self._browser_sessions[session_id]["logs"]

        if type not in self._browser_sessions[session_id]["logs"]:
            return []

        if limit is None:
            limit = MAXIMUM_LOG_ENTRIES

        return self._browser_sessions[session_id]["logs"][type][offset : offset + limit]

    async def close_workspace(self, workspace: str) -> None:
        """Close all browser app instances for a workspace."""
        session_ids = [
            session_id
            for session_id in self._browser_sessions.keys()
            if session_id.startswith(workspace + "/")
        ]
        for session_id in session_ids:
            await self.stop(session_id)

    async def prepare_workspace(self, workspace: str) -> None:
        """Prepare the workspace for the browser app."""
        pass

    def get_service(self):
        """Get the service."""
        return {
            "id": f"browser-runner-{self.controller_id}",
            "type": "server-app-worker",
            "name": "Browser Server App Worker",
            "description": "A worker for running server apps",
            "config": {"visibility": "protected"},
            "supported_types": ["web-python", "web-worker", "window", "iframe"],
            "start": self.start,
            "stop": self.stop,
            "list": self.list,
            "logs": self.logs,
            "shutdown": self.shutdown,
            "prepare_workspace": self.prepare_workspace,
            "close_workspace": self.close_workspace,
        }
