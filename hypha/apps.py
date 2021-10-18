"""Provide an apps controller."""
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

import base58
import multihash
import shortuuid
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from jinja2 import Environment, PackageLoader, select_autoescape
from playwright.async_api import Page, async_playwright
from starlette.responses import Response

from hypha.core import StatusEnum
from hypha.core.interface import CoreInterface
from hypha.core.plugin import DynamicPlugin
from hypha.utils import dotdict, safe_join

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("apps")
logger.setLevel(logging.INFO)

multihash.CodecReg.register("base58", base58.b58encode, base58.b58decode)


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
        core_interface.register_service_as_root(self.get_service_api())
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

        core_interface.register_router(router)

        def close() -> None:
            asyncio.get_running_loop().create_task(self.close())

        event_bus.on("shutdown", close)
        asyncio.ensure_future(self.initialize())

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
            app_file = self.builtin_apps_dir / "imjoy-plugin-parser.html"
            source = (app_file).open(encoding="utf-8").read()
            self.plugin_parser = await self._launch_as_root(
                source, type="raw", workspace="root"
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

    def get_service_api(self) -> Dict[str, Any]:
        """Get a list of service api."""
        # TODO: check permission for each function
        controller = {
            "name": "server-apps",
            "type": "server-apps",
            "config": {"visibility": "public"},
            "install": self.install,
            "launch": self.launch,
            "stop": self.stop,
            "list": self.list,
            "_rintf": True,
        }
        return controller

    async def install(
        self,
        source: str = None,
        source_hash: str = None,
        template: Optional[str] = None,
        overwrite: bool = False,
        attachments: List[dict] = None,
    ) -> str:
        """Save a server app."""
        user_info = self.core_interface.current_user.get()
        user_id = user_info.id
        if source.startswith("http"):
            with urlopen(source) as stream:
                output = stream.read()
            source = output.decode("utf-8")
        # Compute multihash of the source code
        mhash = multihash.digest(source.encode("utf-8"), "sha2-256")
        mhash = mhash.encode("base58").decode("ascii")
        # Verify the source code, useful for downloading from the web
        if source_hash is not None:
            target_mhash = multihash.decode(source_hash.encode("ascii"), "base58")
            assert target_mhash.verify(
                source.encode("utf-8")
            ), f"App source code verification failed (source_hash: {source_hash})."
        if template == "imjoy":
            if not source:
                raise Exception("Source should be provided for imjoy plugin.")

            if self._status != StatusEnum.ready:
                await self.initialize()
            config = await self.plugin_parser.parsePluginCode(source)
            config["source_hash"] = mhash
            try:
                temp = self.jinja_env.get_template(config.type + "-plugin.html")
                source = temp.render(**config)
            except Exception as err:
                raise Exception(
                    "Failed to compile the imjoy plugin, " f"error: {err}"
                ) from err
        elif template:
            temp = self.jinja_env.get_template(template)
            source = temp.render(script=source, source_hash=mhash)
        elif not source:
            raise Exception("Source or template should be provided.")

        random_id = shortuuid.uuid()
        app_dir = self.apps_dir / user_id / random_id
        if app_dir.exists() and not overwrite:
            raise Exception(
                f"Another app with the same id ({random_id}) "
                f"already exists in the user's app space {user_id}."
            )

        os.makedirs(app_dir, exist_ok=True)

        with open(app_dir / "index.html", "w", encoding="utf-8") as fil:
            fil.write(source)

        app_id = f"{user_id}/{random_id}"

        if attachments:
            try:

                for att in attachments:
                    assert (
                        "name" in att and "source" in att
                    ), "Attachment should contain `name` and `source`"
                    if att["source"].startswith("http"):
                        with urlopen(att["source"]) as stream:
                            output = stream.read()
                        att["source"] = output
                    with open(safe_join(str(app_dir), att["name"]), "wb") as fil:
                        fil.write(source)
            except Exception:
                self.remove(app_id)
                raise

        return {"app_id": app_id, "url": f"{self.server_url}/apps/{app_id}/index.html"}

    def remove(self, app_id: str) -> None:
        """Remove a server app."""
        if "/" not in app_id:
            raise Exception(
                f"Invalid app id: {app_id}, the correct format is `user-id/app-id`"
            )
        if (self.apps_dir / app_id).exists():
            shutil.rmtree(self.apps_dir / app_id, ignore_errors=True)
        else:
            raise Exception(f"Server app not found: {app_id}")

    async def launch(
        self,
        source: str,
        workspace: str,
        token: Optional[str] = None,
        timeout: float = 60,
        attachments: List[dict] = None,
        type: str = "imjoy",  # pylint: disable=redefined-builtin
    ) -> dotdict:
        """Start a server app instance."""
        user_info = self.core_interface.current_user.get()
        user_id = user_info.id
        if type == "raw":
            template = None
        elif type != "imjoy":
            template = type + ".html"
        else:
            template = "imjoy"
        app_info = await self.install(
            source, overwrite=True, template=template, attachments=attachments
        )
        app_id = app_info["app_id"]

        if not self.browser:
            await self.initialize()
            # raise Exception("The app controller is not ready yet")
        # context = await self.browser.createIncognitoBrowserContext()
        page = await self.browser.new_page()
        page.plugin = None
        self._capture_logs_from_browser_tabs(page)
        # TODO: dispose await context.close()

        plugin_id = shortuuid.uuid()
        page_id = user_id + "/" + plugin_id
        url = (
            f"{self.server_url}/apps/{app_id}/index.html?"
            + f"id={plugin_id}&workspace={workspace}"
            + f"&server_url={self.server_url}"
            + (f"&token={token}" if token else "")
        )
        fut = asyncio.Future()

        plugin_event_bus = DynamicPlugin.create_plugin_event_bus(plugin_id)

        def cleanup(*args):
            print("cleaning up", plugin_id)
            # asyncio.create_task(self.stop(plugin_id))
            # self.remove(app_id)

        def connected(plugin):
            page.plugin = plugin
            config = dotdict(plugin.config)
            config.url = url
            config.id = plugin_id
            config.app_id = app_id
            self.browser_pages[page_id].update(config)
            self.browser_pages[page_id]["status"] = "connected"
            fut.set_result(config)

        def failed(config):
            fut.set_exception(Exception(config.detail))

        plugin_event_bus.on("connected", connected)
        plugin_event_bus.on("failed", failed)
        plugin_event_bus.on("disconnected", cleanup)

        if timeout > 0:

            async def startup_timer():
                """Killing dead app if it failed to start in time."""
                await asyncio.sleep(timeout)
                if fut.done() or fut.cancelled():
                    return
                fut.set_exception(Exception("Failed to start app: Timeout"))
                cleanup()

            timer = startup_timer()
            asyncio.ensure_future(timer)

        try:
            response = await page.goto(url)
            assert response.status == 200, (
                "Failed to start server app instance, "
                f"status: {response.status}, url: {url}"
            )
            self.browser_pages[page_id] = {
                "id": plugin_id,
                "name": app_id,
                "status": "connecting",
                "page": page,
            }
        except Exception:
            await page.close()
            raise

        return await fut

    async def _launch_as_root(
        self,
        source: str,
        type: str = "imjoy",  # pylint: disable=redefined-builtin
        workspace: str = "root",
        timeout: float = 60.0,
    ) -> dotdict:
        """Launch an app as root user."""
        rws = self.core_interface.get_workspace_interface_as_root(workspace)
        token = await rws.generate_token()
        config = await self.launch(
            source, workspace, type=type, token=token, timeout=timeout
        )
        return await self.core_interface.get_plugin_as_root(
            config.name, config.workspace
        )

    async def stop(self, plugin_id: str) -> None:
        """Stop a server app instance."""
        user_info = self.core_interface.current_user.get()
        user_id = user_info.id
        page_id = user_id + "/" + plugin_id
        if page_id in self.browser_pages:
            await self.browser_pages[page_id]["page"].close()
            del self.browser_pages[page_id]
        else:
            raise Exception(f"Server app instance not found: {plugin_id}")

    async def list(self) -> List[str]:
        """List the saved apps for the current user."""
        user_info = self.core_interface.current_user.get()
        user_id = user_info.id
        sessions = [
            {k: v for k, v in page_info.items() if k != "page"}
            for page_id, page_info in self.browser_pages.items()
            if page_id.startswith(user_id + "/")
        ]
        return sessions
