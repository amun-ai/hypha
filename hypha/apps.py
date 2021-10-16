"""Provide an apps controller."""
import asyncio
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import shortuuid
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from jinja2 import Environment, PackageLoader, select_autoescape
from playwright.async_api import Page, async_playwright
from starlette.responses import Response

from hypha.core import StatusEnum
from hypha.core.interface import CoreInterface
from hypha.utils import dotdict, safe_join

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("apps")
logger.setLevel(logging.INFO)


def is_safe_path(basedir: str, path: str, follow_symlinks: bool = True) -> bool:
    """Check if the file path is safe."""
    # resolves symbolic links
    if follow_symlinks:
        matchpath = os.path.realpath(path)
    else:
        matchpath = os.path.abspath(path)
    return basedir == os.path.commonpath((basedir, matchpath))


class ServerAppController:
    """Server App Controller."""

    # pylint: disable=too-many-instance-attributes

    instance_counter: int = 0

    def __init__(
        self,
        core_interface: CoreInterface,
        port: int,
        in_docker: bool = False,
        apps_dir: str = "./apps",
    ):
        """Initialize the class."""
        self._status: StatusEnum = StatusEnum.not_initialized
        self.browser = None
        self.plugin_parser = None
        self.browser_pages = {}
        self.apps_dir = Path(apps_dir)
        os.makedirs(self.apps_dir, exist_ok=True)
        self.controller_id = str(ServerAppController.instance_counter)
        ServerAppController.instance_counter += 1
        self.port = int(port)
        self.in_docker = in_docker
        self.server_url = f"http://127.0.0.1:{self.port}"
        event_bus = self.event_bus = core_interface.event_bus
        self.core_interface = core_interface
        core_interface.register_interface("get_app_controller", self.get_public_api)
        core_interface.register_interface("getAppController", self.get_public_api)
        self.core_api = dotdict(core_interface.get_interface())
        self.jinja_env = Environment(
            loader=PackageLoader("hypha"), autoescape=select_autoescape()
        )
        self.templates_dir = Path(__file__).parent / "templates"
        self.builtin_apps_dir = Path(__file__).parent / "apps"
        router = APIRouter()
        self._initialize_future: Optional[asyncio.Future] = None

        @router.get("/apps/{path:path}")
        def get_app_file(path: str) -> Response:
            path = safe_join(str(self.apps_dir), path)
            if os.path.exists(path):
                return FileResponse(path)

            return JSONResponse(
                status_code=404,
                content={"success": False, "detail": f"File not found: {path}"},
            )

        # The following code is a hacky solution to call self.initialize
        # We need to find a better way to call it
        # If we try to run initialize() in the startup event callback,
        # It give connection error.
        @router.get("/initialize-apps")
        async def initialize_apps() -> JSONResponse:
            await self.initialize()
            return JSONResponse({"status": "OK"})

        def do_initialization() -> None:
            while True:
                try:
                    time.sleep(0.2)
                    response = requests.get(self.server_url + "/initialize-apps")
                    if response.ok:
                        logger.info("Server apps intialized.")
                        break
                except requests.exceptions.ConnectionError:
                    pass

        threading.Thread(target=do_initialization, daemon=True).start()

        core_interface.register_router(router)

        def close() -> None:
            asyncio.get_running_loop().create_task(self.close())

        event_bus.on("shutdown", close)

    @staticmethod
    def _capture_logs_from_browser_tabs(page: Page) -> None:
        """Capture browser tab logs."""

        def _app_info(message: str) -> None:
            """Log message at info level."""
            if page.plugin and page.plugin.workspace:
                workspace_logger = page.plugin.workspace.get_logger()
                if workspace_logger:
                    workspace_logger.info(message)
                    return
            logger.info(message)

        def _app_error(message: str) -> None:
            """Log message at error level."""
            if page.plugin and page.plugin.workspace:
                workspace_logger = page.plugin.workspace.get_logger()
                if workspace_logger:
                    workspace_logger.error(message)
                    return
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
        if self._status == StatusEnum.ready:
            return
        if self._status == StatusEnum.initializing:
            await self._initialize_future
            return

        self._status = StatusEnum.initializing
        self._initialize_future = asyncio.Future()
        try:
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
            for app_file in self.builtin_apps_dir.iterdir():
                if app_file.suffix != ".html" or app_file.name.startswith("."):
                    continue
                source = (app_file).open().read()
                pid = app_file.stem
                await self.deploy(
                    source, user_id="root", template=None, app_id=pid, overwrite=True
                )

            self.plugin_parser = await self._launch_as_root(
                "imjoy-plugin-parser", workspace="root"
            )

            # TODO: check if the plugins are marked as startup plugin
            # and if yes, we will run it directly

            self._status = StatusEnum.ready
            self._initialize_future.set_result(None)

        except Exception as err:  # pylint: disable=broad-except
            self._status = StatusEnum.not_initialized
            logger.exception("Failed to initialize the app controller")
            self._initialize_future.set_exception(err)

    async def close(self) -> None:
        """Close the app controller."""
        logger.info("Closing the browser app controller...")
        if self.browser:
            await self.browser.close()
        logger.info("Browser app controller closed.")

    async def get_public_api(self) -> Dict[str, Any]:
        """Get a list of public api."""
        if self._status != StatusEnum.ready:
            await self.initialize()

        # TODO: check permission for each function
        controller = {
            "deploy": self.deploy,
            "undeploy": self.undeploy,
            "start": self.start,
            "stop": self.stop,
            "list": self.list,
        }

        controller["_rintf"] = True
        return controller

    async def list(self, user_id: str) -> List[str]:
        """List the deployed apps."""
        return [
            f"{user_id}/{app_name}"
            for app_name in os.listdir(self.apps_dir / user_id)
            if not app_name.startswith(".")
        ]

    async def deploy(
        self,
        source: str,
        user_id: str,
        template: Optional[str] = None,
        app_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """Deploy a server app."""
        if template == "imjoy":
            if not source:
                raise Exception("Source should be provided for imjoy plugin.")
            config = await self.plugin_parser.parsePluginCode(source)
            if app_id and app_id != config.name:
                raise Exception(
                    f"You cannot specify a different id ({app_id}) "
                    f"for ImJoy plugin, it has to be `{config.name}`."
                )
            app_id = config.name
            try:
                temp = self.jinja_env.get_template(config.type + "-plugin.html")
                source = temp.render(
                    script=config.script, requirements=config.requirements
                )
            except Exception as err:
                raise Exception(
                    "Failed to compile the imjoy plugin, " f"error: {err}"
                ) from err
        elif template:
            temp = self.jinja_env.get_template(template)
            source = temp.render(script=source)
        elif not source:
            raise Exception("Source or template should be provided.")

        app_id = app_id or shortuuid.uuid()
        if (self.apps_dir / user_id / app_id).exists() and not overwrite:
            raise Exception(
                f"Another app with the same id ({app_id}) "
                f"already exists in the user's app space {user_id}."
            )

        os.makedirs(self.apps_dir / user_id / app_id, exist_ok=True)

        with open(
            self.apps_dir / user_id / app_id / "index.html", "w", encoding="utf-8"
        ) as fil:
            fil.write(source)

        return f"{user_id}/{app_id}"

    async def undeploy(self, app_id: str) -> None:
        """Deploy a server app."""
        if "/" not in app_id:
            raise Exception(
                f"Invalid app id: {app_id}, the correct format is `user-id/app-id`"
            )
        if (self.apps_dir / app_id).exists():
            shutil.rmtree(self.apps_dir / app_id, ignore_errors=True)
        else:
            raise Exception(f"Server app not found: {app_id}")

    async def start(
        self, app_id: str, workspace: str, token: Optional[str] = None
    ) -> dotdict:
        """Start a server app instance."""
        if self.browser is None:
            raise Exception("The app controller is not ready yet")
        # context = await self.browser.createIncognitoBrowserContext()
        page = await self.browser.new_page()
        page.plugin = None
        self._capture_logs_from_browser_tabs(page)
        # TODO: dispose await context.close()
        name = shortuuid.uuid()
        if "/" not in app_id:
            app_id = workspace + "/" + app_id
        url = (
            f"{self.server_url}/apps/{app_id}/index.html?"
            + f"name={name}&workspace={workspace}&server_url={self.server_url}"
            + (f"&token={token}" if token else "")
        )

        fut = asyncio.Future()

        def registered(plugin):
            if plugin.name == name:
                # return the plugin api
                page.plugin = plugin
                fut.set_result(plugin.config)
                self.event_bus.off("plugin_registered", registered)

        # TODO: Handle timeout
        self.event_bus.on("plugin_registered", registered)
        try:
            response = await page.goto(url)
            assert response.status == 200, (
                "Failed to start server app instance, "
                f"status: {response.status}, url: {url}"
            )
            self.browser_pages[name] = page
        except Exception:
            self.event_bus.off("plugin_registered", registered)
            raise

        return await fut

    async def _launch_as_root(self, app_name: str, workspace: str = "root") -> dotdict:
        """Launch an app as root user."""
        rws = self.core_interface.get_workspace_as_root(workspace)
        token = await rws.generate_token()
        config = await self.start(app_name, workspace, token=token)
        return await self.core_interface.get_plugin_as_root(
            config.name, config.workspace
        )

    async def stop(self, name: str) -> None:
        """Stop a server app instance."""
        if name in self.browser_pages:
            await self.browser_pages[name].close()
        else:
            raise Exception(f"Server app instance not found: {name}")
