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
            "--enable-features=Vulkan",
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
        url: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Start a browser app instance."""
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

    async def get_log(
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

    def get_service(self):
        """Get the service."""
        return {
            "id": "server-app-worker",
            "name": "Server App Worker",
            "description": "A worker for running server apps",
            "config": {"visibility": "protected"},
            "start": self.start,
            "stop": self.stop,
            "list": self.list,
            "get_log": self.get_log,
            "shutdown": self.shutdown,
            "close_workspace": self.close_workspace,
        }
