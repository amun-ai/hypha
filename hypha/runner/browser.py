"""Provide a browser runner."""
import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import Page, async_playwright

from hypha.core.store import RedisStore

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("browser")
logger.setLevel(logging.INFO)

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
        store: RedisStore,
        in_docker: bool = False,
    ):
        """Initialize the class."""
        self.browser = None
        self._browser_sessions = {}
        self.controller_id = str(BrowserAppRunner.instance_counter)
        BrowserAppRunner.instance_counter += 1
        self.in_docker = in_docker
        self.event_bus = store.get_event_bus()
        self.store = store

        def close(_) -> None:
            asyncio.get_running_loop().create_task(self.close())

        self.event_bus.on_local("shutdown", close)
        # asyncio.ensure_future(self.initialize())

    async def initialize(self) -> None:
        """Initialize the app controller."""
        playwright = await async_playwright().start()
        args = [
            "--site-per-process",
            "--enable-unsafe-webgpu",
            "--use-vulkan",
            "--enable-features=Vulkan",
        ]
        # so it works in the docker image
        if self.in_docker:
            args.append("--no-sandbox")
        self.browser = await playwright.chromium.launch(args=args)
        return self.browser

    async def close(self) -> None:
        """Close the app controller."""
        logger.info("Closing the browser app controller...")
        if self.browser:
            await self.browser.close()
        logger.info("Browser app controller closed.")

    async def start(
        self,
        url: str,
        session_id: str,
    ):
        """Start a browser app instance."""
        assert session_id, "client_id is required"
        if not self.browser:
            await self.initialize()
        # context = await self.browser.createIncognitoBrowserContext()
        page = await self.browser.new_page()
        logs = {}
        self._browser_sessions[session_id] = {
            "url": url,
            "status": "connecting",
            "page": page,
            "logs": logs,
        }

        _capture_logs_from_browser_tabs(page, logs)
        # TODO: dispose await context.close()

        try:
            logger.info("Loading browser app: %s", url)
            response = await page.goto(url, timeout=0, wait_until="load")
            assert response.status == 200, (
                "Failed to start browser app instance, "
                f"status: {response.status}, url: {url}"
            )
            logger.info("Browser app loaded: %s", url)
        except Exception:
            await page.close()
            del self._browser_sessions[session_id]
            raise

    async def stop(self, session_id: str) -> None:
        """Stop a browser app instance."""
        if session_id in self._browser_sessions:
            await self._browser_sessions[session_id]["page"].close()
            if session_id in self._browser_sessions:
                del self._browser_sessions[session_id]
        else:
            raise Exception(f"browser app instance not found: {session_id}")

    async def list(self, workspace) -> List[str]:
        """List the browser apps for the current user."""
        sessions = [
            {k: v for k, v in page_info.items() if k != "page"}
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
        if session_id in self._browser_sessions:
            if type is None:
                return self._browser_sessions[session_id]["logs"]
            if limit is None:
                limit = MAXIMUM_LOG_ENTRIES
            return self._browser_sessions[session_id]["logs"][type][
                offset : offset + limit
            ]
        raise Exception(f"browser app instance not found: {session_id}")
