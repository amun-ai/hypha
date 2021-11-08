"""Provide a browser runner."""
import asyncio
import logging
import sys
from typing import Any, Dict, List

from playwright.async_api import Page, async_playwright

from hypha.core.interface import CoreInterface

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("browser")
logger.setLevel(logging.INFO)


class BrowserAppRunner:
    """Browser app runner."""

    instance_counter: int = 0

    def __init__(
        self,
        core_interface: CoreInterface,
        in_docker: bool = False,
    ):
        """Initialize the class."""
        self.browser = None
        self.browser_pages = {}
        self.controller_id = str(BrowserAppRunner.instance_counter)
        BrowserAppRunner.instance_counter += 1
        self.in_docker = in_docker
        self.event_bus = core_interface.event_bus
        core_interface.register_service_as_root(self.get_service_api())
        self.core_interface = core_interface

        def close() -> None:
            asyncio.get_running_loop().create_task(self.close())

        self.event_bus.on("shutdown", close)
        # asyncio.ensure_future(self.initialize())

    @staticmethod
    def _capture_logs_from_browser_tabs(page: Page) -> None:
        """Capture browser tab logs."""

        def _app_info(message: str) -> None:
            """Log message at info level."""
            logger.info(message)

        def _app_error(message: str) -> None:
            """Log message at error level."""
            logger.error(message)

        page.on(
            "targetcreated",
            lambda target: _app_info(str(target)),
        )
        page.on("console", lambda target: _app_info(target.text))
        page.on("error", lambda target: _app_error(target.text))
        page.on("pageerror", lambda target: _app_error(str(target)))

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

    async def close(self) -> None:
        """Close the app controller."""
        logger.info("Closing the browser app controller...")
        if self.browser:
            await self.browser.close()
        logger.info("Browser app controller closed.")

    async def start(
        self,
        url: str,
        plugin_id: str,
    ):
        """Start a browser app instance."""
        user_info = self.core_interface.current_user.get()
        user_id = user_info.id

        if not self.browser:
            await self.initialize()
            # raise Exception("The app controller is not ready yet")
        # context = await self.browser.createIncognitoBrowserContext()
        page = await self.browser.new_page()
        self._capture_logs_from_browser_tabs(page)
        # TODO: dispose await context.close()

        page_id = user_id + "/" + plugin_id

        try:
            response = await page.goto(url)
            assert response.status == 200, (
                "Failed to start browser app instance, "
                f"status: {response.status}, url: {url}"
            )
            self.browser_pages[page_id] = {
                "url": url,
                "status": "connecting",
                "page": page,
            }
        except Exception:
            await page.close()
            raise

    async def stop(self, plugin_id: str) -> None:
        """Stop a browser app instance."""
        user_info = self.core_interface.current_user.get()
        user_id = user_info.id
        page_id = user_id + "/" + plugin_id
        if page_id in self.browser_pages:
            await self.browser_pages[page_id]["page"].close()
            del self.browser_pages[page_id]
        else:
            raise Exception(f"browser app instance not found: {plugin_id}")

    async def list(self) -> List[str]:
        """List the browser apps for the current user."""
        user_info = self.core_interface.current_user.get()
        user_id = user_info.id
        sessions = [
            {k: v for k, v in page_info.items() if k != "page"}
            for page_id, page_info in self.browser_pages.items()
            if page_id.startswith(user_id + "/")
        ]
        return sessions

    def get_service_api(self) -> Dict[str, Any]:
        """Get a list of service api."""
        controller = {
            "name": "browser-app-runner",
            "type": "plugin-runner",
            "config": {"visibility": "protected"},
            "start": self.start,
            "stop": self.stop,
            "list": self.list,
            "_rintf": True,
        }
        return controller
