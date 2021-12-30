"""Provide an apps controller."""
import asyncio
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.request import urlopen

import base58
import multihash
import shortuuid
from aiobotocore.session import get_session
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from jinja2 import Environment, PackageLoader, select_autoescape
from starlette.responses import Response

from hypha.core import StatusEnum, RDF
from hypha.core.interface import CoreInterface
from hypha.core.plugin import DynamicPlugin
from hypha.plugin_parser import parse_imjoy_plugin, convert_config_to_rdf
from hypha.runner.browser import BrowserAppRunner
from hypha.utils import (
    PLUGIN_CONFIG_FIELDS,
    dotdict,
    list_objects_async,
    remove_objects_async,
    safe_join,
)

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
        endpoint_url=None,
        access_key_id=None,
        secret_access_key=None,
        workspace_bucket="hypha-workspaces",
        user_applications_dir="applications",
    ):  # pylint: disable=too-many-arguments
        """Initialize the class."""
        self._status: StatusEnum = StatusEnum.not_initialized
        self._apps = {}
        self.apps_dir = Path(apps_dir)
        os.makedirs(self.apps_dir, exist_ok=True)
        self.port = int(port)
        self.in_docker = in_docker
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.s3_enabled = endpoint_url is not None
        self.workspace_bucket = workspace_bucket
        self.user_applications_dir = user_applications_dir
        self.local_base_url = core_interface.local_base_url
        self.public_base_url = core_interface.public_base_url

        event_bus = self.event_bus = core_interface.event_bus
        self.core_interface = core_interface
        core_interface.register_service_as_root(self.get_service_api())
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

        @router.get("/apps/{workspace}/{path:path}")
        def get_app_file(workspace: str, path: str, token: str) -> Response:
            user_info = core_interface.get_user_info_from_token(token)
            if not core_interface.check_permission(workspace, user_info):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": (
                            f"{user_info['username']} has no"
                            f" permission to access {workspace}"
                        ),
                    },
                )
            path = safe_join(str(self.apps_dir), workspace, path)
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
            brwoser_runner = BrowserAppRunner(
                self.core_interface, in_docker=self.in_docker
            )
            await brwoser_runner.initialize()

            brwoser_runner = BrowserAppRunner(
                self.core_interface, in_docker=self.in_docker
            )
            await brwoser_runner.initialize()

            self._runners = await self.core_interface.list_services(
                {"type": "plugin-runner"}
            )
            assert len(self._runners) > 0, "No plugin runner is available."
            # TODO: check if the plugins are marked as startup plugin
            # and if yes, we will run it directly

            self._status = StatusEnum.ready
            self._initialize_future.set_result(None)

        except Exception as err:  # pylint: disable=broad-except
            self._status = StatusEnum.not_initialized
            logger.exception("Failed to initialize the app controller")
            self._initialize_future.set_exception(err)

    def create_client_async(self):
        """Create client async."""
        assert self.s3_enabled, "S3 is not enabled."
        return get_session().create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="EU",
        )

    async def list_saved_workspaces(
        self,
    ):
        """List saved workspaces."""
        async with self.create_client_async() as s3_client:
            items = await list_objects_async(s3_client, self.workspace_bucket, "/")
        return [item["Key"] for item in items]

    async def list_apps(self, workspace: str = None):
        """List applications in the workspace."""
        if not workspace:
            workspace = self.core_interface.current_workspace.get()
        else:
            workspace = await self.core_interface.get_workspace(workspace)
        return [app_info.dict() for app_info in workspace.applications.values()]

    async def save_application(
        self,
        app_id: str,
        rdf: RDF,
        source: str,
        attachments: Optional[Dict[str, Any]] = None,
    ):
        """Save an application to the workspace."""
        workspace, mhash = app_id.split("/")
        async with self.create_client_async() as s3_client:
            app_dir = f"{workspace}/{self.user_applications_dir}/{mhash}"

            async def save_file(key, content):
                if isinstance(content, str):
                    content = content.encode("utf-8")
                response = await s3_client.put_object(
                    Body=content,
                    Bucket=self.workspace_bucket,
                    Key=key,
                    ContentLength=len(content),
                )

                if (
                    "ResponseMetadata" in response
                    and "HTTPStatusCode" in response["ResponseMetadata"]
                ):
                    response_code = response["ResponseMetadata"]["HTTPStatusCode"]
                    assert (
                        response_code == 200
                    ), f"Failed to save file: {key}, status code: {response_code}"
                assert "ETag" in response

            # Upload the source code
            await save_file(f"{app_dir}/index.html", source)

            if attachments:
                rdf.attachments = rdf.attachments or {}
                rdf.attachments["files"] = rdf.attachments.get("files", [])
                files = rdf.attachments["files"]
                for att in attachments:
                    assert (
                        "name" in att and "source" in att
                    ), "Attachment should contain `name` and `source`"
                    if att["source"].startswith("http") and "\n" not in att["source"]:
                        with urlopen(att["source"]) as stream:
                            output = stream.read()
                        att["source"] = output
                    await save_file(f"{app_dir}/{att['name']}", att["source"])
                    files.append(att["name"])

            content = json.dumps(rdf.dict(), indent=4)
            await save_file(f"{app_dir}/rdf.json", content)
        logger.info("Saved application (%s)to workspace: %s", mhash, workspace)

    async def prepare_application(self, app_id):
        """Download files for an application to be run."""
        local_app_dir = self.apps_dir / app_id
        if "/" not in app_id:
            raise ValueError(f"Invalid app id: {app_id}")
        workspace, mhash = app_id.split("/")
        if os.path.exists(local_app_dir):
            logger.info("Application (%s) is already prepared.", app_id)
            return

        logger.info("Preparing application (%s).", app_id)

        # Download the app to the apps dir
        async with self.create_client_async() as s3_client:
            app_dir = workspace + "/" + self.user_applications_dir + "/" + mhash

            async def download_file(key, local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                response = await s3_client.get_object(
                    Bucket=self.workspace_bucket,
                    Key=key,
                )
                if (
                    "ResponseMetadata" in response
                    and "HTTPStatusCode" in response["ResponseMetadata"]
                ):
                    response_code = response["ResponseMetadata"]["HTTPStatusCode"]
                    assert (
                        response_code == 200
                    ), f"Failed to download file: {key}, status code: {response_code}"
                assert "ETag" in response
                data = await response["Body"].read()
                with open(local_path, "wb") as fil:
                    fil.write(data)

            # Upload the source code and attachments
            await download_file(
                os.path.join(app_dir, "index.html"), local_app_dir / "index.html"
            )
            await download_file(
                os.path.join(app_dir, "rdf.json"), local_app_dir / "rdf.json"
            )
            with open(local_app_dir / "rdf.json", "r", encoding="utf-8") as fil:
                rdf = RDF.parse_obj(json.load(fil))

            if rdf.attachments:
                files = rdf.attachments.get("files")
                if files:
                    for file_name in files:
                        await download_file(
                            os.path.join(app_dir, file_name), local_app_dir / file_name
                        )
            logger.info("Application (%s) is prepared.", app_id)

    async def close(self) -> None:
        """Close the app controller."""
        logger.info("Closing the server app controller...")
        for app in self._apps.values():
            await self.stop(app["id"])

    def get_service_api(self) -> Dict[str, Any]:
        """Get a list of service api."""
        # TODO: check permission for each function
        controller = {
            "name": "server-apps",
            "type": "server-apps",
            "config": {"visibility": "public"},
            "install": self.install,
            "uninstall": self.uninstall,
            "launch": self.launch,
            "start": self.start,
            "stop": self.stop,
            "list_apps": self.list_apps,
            "list_running": self.list_running,
            "get_log": self.get_log,
            "_rintf": True,
        }
        return controller

    # pylint: disable=too-many-statements,too-many-locals
    async def install(
        self,
        source: str = None,
        source_hash: str = None,
        config: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None,
        attachments: List[dict] = None,
        workspace: Optional[str] = None,
    ) -> str:
        """Save a server app."""
        if template is None:
            if config:
                template = config.get("type") + "-plugin.html"
            else:
                template = "imjoy"
        if not workspace:
            workspace = self.core_interface.current_workspace.get()
        else:
            workspace = await self.core_interface.get_workspace(workspace)

        user_info = self.core_interface.current_user.get()
        if not self.core_interface.check_permission(workspace, user_info):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to install apps in workspace {workspace}"
            )

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
            assert config is None, "Config should not be provided for imjoy plugin."

            if self._status != StatusEnum.ready:
                await self.initialize()
            try:
                config = parse_imjoy_plugin(source)
                config["source_hash"] = mhash
                temp = self.jinja_env.get_template(config["type"] + "-plugin.html")
                source = temp.render(
                    config={k: config[k] for k in config if k in PLUGIN_CONFIG_FIELDS},
                    script=config["script"],
                    requirements=config["requirements"],
                )
            except Exception as err:
                raise Exception(
                    "Failed to parse or compile the imjoy plugin, " f"error: {err}"
                ) from err
        elif template:
            temp = self.jinja_env.get_template(template)
            default_config = {
                "name": "Untitled Plugin",
                "version": "0.1.0",
            }
            default_config.update(config or {})
            config = default_config
            source = temp.render(
                script=source,
                source_hash=mhash,
                config=config,
                requirements=config.get("requirements", []),
            )
        elif not source:
            raise Exception("Source or template should be provided.")

        app_id = f"{workspace.name}/{mhash}"

        public_url = f"{self.public_base_url}/apps/{app_id}/index.html"
        rdf_obj = convert_config_to_rdf(config, app_id, public_url)
        rdf_obj.update(
            {
                "local_url": f"{self.local_base_url}/apps/{app_id}/index.html",
                "public_url": public_url,
            }
        )
        rdf = RDF.parse_obj(rdf_obj)
        await self.save_application(app_id, rdf, source, attachments)
        workspace.install_application(rdf)
        return rdf_obj

    async def uninstall(self, app_id: str) -> None:
        """Uninstall a server app."""
        if "/" not in app_id:
            raise Exception(
                f"Invalid app id: {app_id}, the correct format is `user-id/app-id`"
            )
        workspace_name, mhash = app_id.split("/")
        workspace = await self.core_interface.get_workspace(workspace_name)

        user_info = self.core_interface.current_user.get()
        if not self.core_interface.check_permission(workspace, user_info):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to uninstall apps in workspace {workspace.name}"
            )

        async with self.create_client_async() as s3_client:
            app_dir = f"{workspace.name}/{self.user_applications_dir}/{mhash}/"
            await remove_objects_async(s3_client, self.workspace_bucket, app_dir)
        if (self.apps_dir / app_id).exists():
            shutil.rmtree(self.apps_dir / app_id, ignore_errors=True)

    async def launch(
        self,
        source: str,
        workspace: str,
        token: Optional[str] = None,
        timeout: float = 60,
        config: Optional[Dict[str, Any]] = None,
        attachments: List[dict] = None,
    ) -> dotdict:
        """Start a server app instance."""
        if token:
            user_info = self.core_interface.get_user_info_from_token(token)
        else:
            user_info = self.core_interface.current_user.get()
        if not self.core_interface.check_permission(workspace, user_info):
            raise Exception(
                f"User {user_info.id} does not have permission"
                " to run app in workspace {workspace}."
            )

        app_info = await self.install(
            source,
            attachments=attachments,
            config=config,
            workspace=workspace,
        )
        app_id = app_info["id"]

        if len(self._runners) <= 0:
            await self.initialize()

        assert len(self._runners) > 0, "No plugin runner is available"

        return await self.start(
            app_id, workspace=workspace, token=token, timeout=timeout
        )

    # pylint: disable=too-many-statements,too-many-locals
    async def start(
        self,
        app_id,
        workspace=None,
        token=None,
        plugin_id=None,
        timeout: float = 60,
        loop_count=0,
    ):
        """Start the app and keep it alive."""
        if workspace is None:
            workspace = self.core_interface.current_workspace.get().name
        if workspace != app_id.split("/")[0]:
            raise Exception("Workspace mismatch between app_id and workspace.")
        if token is None:
            ws = self.core_interface.get_workspace_interface(workspace)
            token = ws.generate_token()
        user_info = self.core_interface.get_user_info_from_token(token)
        if not self.core_interface.check_permission(workspace, user_info):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to run app {app_id} in workspace {workspace}."
            )

        if plugin_id is None:
            plugin_id = shortuuid.uuid()

        await self.prepare_application(app_id)
        local_url = (
            f"{self.local_base_url}/apps/{app_id}/index.html?"
            + f"id={plugin_id}&workspace={workspace}"
            + f"&server_url={self.local_base_url}"
            + f"&token={token}"
            if token
            else ""
        )
        public_url = (
            f"{self.public_base_url}/apps/{app_id}/index.html?"
            + f"id={plugin_id}&workspace={workspace}"
            + f"&server_url={self.public_base_url}"
            + f"&token={token}"
            if token
            else ""
        )

        page_id = workspace + "/" + plugin_id
        app_info = {
            "id": plugin_id,
            "name": app_id,
            "local_url": local_url,
            "public_url": public_url,
            "status": "connecting",
            "watch": False,
            "runner": None,
        }

        def stop_plugin():
            logger.warning("Plugin %s is stopping...", plugin_id)
            asyncio.create_task(self.stop(plugin_id, False))

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
                    # Restore status
                    if plugin.get_status() == "failing":
                        plugin.set_status("ready")
                    await asyncio.sleep(period)
                except TimeoutError:
                    failure += 1
                    # Mark as failed
                    plugin.set_status("failing")
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
                            app_id,
                            workspace=workspace,
                            token=token,
                            plugin_id=plugin_id,
                            timeout=timeout,
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

        def connected(plugin):
            config = dotdict(plugin.config)
            config.local_url = local_url
            config.id = plugin_id
            config.app_id = app_id
            config.public_url = public_url
            self._apps[page_id].update(config)
            self._apps[page_id]["status"] = "connected"
            asyncio.get_running_loop().create_task(check_ready(plugin, config))

        def failed(detail):
            app_info["watch"] = False
            fut.set_exception(Exception(detail))

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
            await runner.start(url=local_url, plugin_id=plugin_id)

        app_info["runner"] = runner
        self._apps[page_id] = app_info

        plugin, config = await fut
        asyncio.get_running_loop().create_task(keep_alive(plugin, config, loop_count))
        return config

    async def stop(self, plugin_id: str, raise_exception=True) -> None:
        """Stop a server app instance."""
        workspace = self.core_interface.current_workspace.get()
        page_id = workspace.name + "/" + plugin_id
        if page_id in self._apps:
            logger.info("Stopping app: %s...", page_id)

            app_info = self._apps[page_id]
            plugin = workspace.get_plugin_by_id(app_info["id"])
            if plugin:
                await plugin.terminate()
            with self.core_interface.set_root_user():
                app_info["watch"] = False  # make sure we don't keep-alive
                await app_info["runner"].stop(plugin_id)
            if page_id in self._apps:
                del self._apps[page_id]
        elif raise_exception:
            raise Exception(f"Server app instance not found: {plugin_id}")

    async def get_log(
        self,
        plugin_id: str,
        type: str = None,  # pylint: disable=redefined-builtin
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get server app instance log."""
        workspace = self.core_interface.current_workspace.get()
        page_id = workspace.name + "/" + plugin_id
        if page_id in self._apps:
            with self.core_interface.set_root_user():
                return await self._apps[page_id]["runner"].get_log(
                    plugin_id, type=type, offset=offset, limit=limit
                )
        else:
            raise Exception(f"Server app instance not found: {plugin_id}")

    async def list_running(self) -> List[str]:
        """List the running sessions for the current workspace."""
        workspace = self.core_interface.current_workspace.get()
        sessions = [
            {k: v for k, v in page_info.items() if k != "runner"}
            for page_id, page_info in self._apps.items()
            if page_id.startswith(workspace.name + "/")
        ]
        return sessions
