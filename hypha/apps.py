"""Provide an apps controller."""
import asyncio
import logging
import os
import shutil
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

import base58
import multihash
import shortuuid
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from jinja2 import Environment, PackageLoader, select_autoescape
from starlette.responses import Response

from hypha.core import StatusEnum
from hypha.core.interface import CoreInterface
from hypha.core.plugin import DynamicPlugin
from hypha.utils import dotdict, safe_join, PLUGIN_CONFIG_FIELDS
from hypha.runner.browser import BrowserAppRunner

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

    def __init__(
        self,
        core_interface: CoreInterface,
        port: int,
        in_docker: bool = False,
        apps_dir: str = "./apps",
    ):
        """Initialize the class."""
        self._status: StatusEnum = StatusEnum.not_initialized
        self.plugin_parser = None
        self._apps = {}
        self.apps_dir = Path(apps_dir)
        os.makedirs(self.apps_dir, exist_ok=True)
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
        os.makedirs(self.apps_dir / "built-in", exist_ok=True)
        # copy files inside the builtin apps dir to the apps dir (overwrite if exists)
        for file in self.builtin_apps_dir.glob("*"):
            shutil.copy(file, self.apps_dir / "built-in" / file.name)
        router = APIRouter()
        self._initialize_future: Optional[asyncio.Future] = None
        self._runners = {}

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
            asyncio.ensure_future(self.close())

        event_bus.on("shutdown", close)
        asyncio.ensure_future(self.initialize())

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
            # Start 2 instances of browser
            brwoser_runner = BrowserAppRunner(self.core_interface)
            await brwoser_runner.initialize()

            brwoser_runner = BrowserAppRunner(self.core_interface)
            await brwoser_runner.initialize()

            self._runners = self.core_interface.list_services({"type": "plugin-runner"})
            assert len(self._runners) > 0, "No plugin runner is available."
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
        logger.info("Closing the server app controller...")
        for app in self._apps:
            await self.stop(app["id"])

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
                source = temp.render(
                    config={k: config[k] for k in config if k in PLUGIN_CONFIG_FIELDS},
                    script=config.script,
                    requirements=config.requirements,
                )
            except Exception as err:
                raise Exception(
                    "Failed to compile the imjoy plugin, " f"error: {err}"
                ) from err
        elif template:
            temp = self.jinja_env.get_template(template)
            source = temp.render(
                script=source, source_hash=mhash, config={}, requirements=[]
            )
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

        if len(self._runners) <= 0:
            await self.initialize()

        assert len(self._runners) > 0, "No plugin runner is available"

        plugin_id = shortuuid.uuid()

        url = (
            f"{self.server_url}/apps/{app_id}/index.html?"
            + f"id={plugin_id}&workspace={workspace}"
            + f"&server_url={self.server_url}"
            + (f"&token={token}" if token else "")
        )

        return await self.start(url, plugin_id, user_id, app_id, timeout)

    # pylint: disable=too-many-statements
    async def start(self, url, plugin_id, user_id, app_id, timeout, loop_count=0):
        """Start the app and keep it alive."""
        page_id = user_id + "/" + plugin_id
        app_info = {
            "id": plugin_id,
            "name": app_id,
            "url": url,
            "status": "connecting",
            "watch": False,
            "runner": None,
        }

        def stop_plugin():
            logger.warning("Plugin %s is stopping...", plugin_id)
            asyncio.create_task(self.stop(plugin_id))

        async def check_ready(plugin, config):
            api = await plugin.get_api()
            plugin.register_exit_callback(stop_plugin)
            readiness_probe = config.get("readiness_probe", {})
            exec_func = readiness_probe.get("exec")
            if exec_func and exec_func not in api:
                fut.set_exception(
                    Exception(
                        f"readiness_probe.exec function ({exec_func})"
                        f" does not exist in plugin ({plugin.name})"
                    )
                )
                return
            if exec_func:
                exec_func = api[exec_func]
            initial_delay = readiness_probe.get("initial_delay_seconds", 0)
            period = readiness_probe.get("period_seconds", 10)
            success_threshold = readiness_probe.get("success_threshold", 1)
            failure_threshold = readiness_probe.get("failure_threshold", 3)
            timeout = readiness_probe.get("timeout", 5)
            assert timeout >= 1
            # check if it's ready
            if exec_func:
                await asyncio.sleep(initial_delay)
                success = 0
                failure = 0
                while True:
                    try:
                        logger.warning(
                            "Waiting for plugin %s to be ready...%s",
                            plugin.name,
                            failure,
                        )
                        is_ready = await asyncio.wait_for(exec_func(), timeout)
                        if is_ready:
                            success += 1
                            if success >= success_threshold:
                                break
                    except TimeoutError:
                        failure += 1
                        if failure >= failure_threshold:
                            # mark as failed
                            plugin.set_status("unready")
                            await asyncio.wait_for(api.teriminate(), timeout)
                            return
                        await asyncio.sleep(period)

            logger.warning("Plugin `%s` is ready.", plugin.name)
            fut.set_result((plugin, config))

        async def keep_alive(plugin, config, loop_count):
            api = await plugin.get_api()
            liveness_probe = config.get("liveness_probe", {})
            exec_func = liveness_probe.get("exec")
            if exec_func and exec_func not in api:
                fut.set_exception(
                    Exception(
                        f"liveness_probe.exec function ({exec_func})"
                        f" does not exist in plugin ({plugin.name})"
                    )
                )
                return
            if exec_func:
                exec_func = api[exec_func]
            initial_delay = liveness_probe.get("initial_delay_seconds", 0)
            period = liveness_probe.get("period_seconds", 10)
            failure_threshold = liveness_probe.get("failure_threshold", 3)
            timeout = liveness_probe.get("timeout", 5)
            assert timeout >= 1

            # keep-alive
            if not exec_func:
                return
            await asyncio.sleep(initial_delay)
            app_info["watch"] = True
            failure = 0
            while app_info["watch"]:
                try:
                    is_alive = await asyncio.wait_for(exec_func(), timeout)
                    # return False is the same as failed to call alive()
                    if not is_alive:
                        raise TimeoutError
                    await asyncio.sleep(period)
                except TimeoutError:
                    failure += 1
                    logger.warning("Plugin %s is failing... %s", plugin.name, failure)
                    if failure >= failure_threshold:
                        logger.warning(
                            "Plugin %s failed too" " many times, restarting now...",
                            plugin.name,
                        )

                        loop_count += 1
                        if loop_count > 10:
                            plugin.set_status("crash-loop-back-off")
                            app_info["watch"] = False
                            return
                        # Mark it as restarting
                        plugin.set_status("restarting")

                        try:
                            await asyncio.wait_for(plugin.terminate(), timeout)
                        except TimeoutError:
                            logger.error("Failed to terminate plugin %s", plugin.name)
                        # start a new one
                        await self.start(
                            url,
                            plugin_id,
                            user_id,
                            app_id,
                            timeout,
                            loop_count=loop_count,
                        )

                    else:
                        await asyncio.sleep(period)

        fut = asyncio.Future()
        plugin_event_bus = DynamicPlugin.create_plugin_event_bus(plugin_id)

        def cleanup(*args):
            app_info["watch"] = False
            print("cleaning up", plugin_id)
            # asyncio.create_task(self.stop(plugin_id))
            # self.remove(app_id)

        def connected(plugin):
            config = dotdict(plugin.config)
            config.url = url
            config.id = plugin_id
            config.app_id = app_id
            self._apps[page_id].update(config)
            self._apps[page_id]["status"] = "connected"
            asyncio.get_running_loop().create_task(check_ready(plugin, config))

        def failed(config):
            app_info["watch"] = False
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
            asyncio.get_running_loop().create_task(timer)

        runner_info = random.choice(self._runners)
        with self.core_interface.set_root_user():
            runner = await self.core_interface.get_service(runner_info)
            await runner.start(url=url, plugin_id=plugin_id)

        app_info["runner"] = runner
        self._apps[page_id] = app_info

        plugin, config = await fut
        asyncio.get_running_loop().create_task(keep_alive(plugin, config, loop_count))
        return config

    async def _launch_as_root(
        self,
        source: str,
        type: str = "imjoy",  # pylint: disable=redefined-builtin
        workspace: str = "root",
        timeout: float = 60.0,
    ) -> dotdict:
        """Launch an app as root user."""
        with self.core_interface.set_root_user():
            rws = self.core_interface.get_workspace_interface(workspace)
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
        if page_id in self._apps:
            logger.info("Stopping app: %s...", page_id)
            with self.core_interface.set_root_user():
                self._apps[page_id]["watch"] = False  # make sure we don't keep-alive
                await self._apps[page_id]["runner"].stop(plugin_id)
            del self._apps[page_id]
        else:
            raise Exception(f"Server app instance not found: {plugin_id}")

    async def list(self) -> List[str]:
        """List the saved apps for the current user."""
        user_info = self.core_interface.current_user.get()
        user_id = user_info.id
        sessions = [
            {k: v for k, v in page_info.items() if k != "runner"}
            for page_id, page_info in self._apps.items()
            if page_id.startswith(user_id + "/")
        ]
        return sessions
