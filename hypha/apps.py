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

import aiofiles
import base58
import jose
import multihash
from aiobotocore.session import get_session
from fastapi import APIRouter
from fastapi.responses import JSONResponse, FileResponse
from jinja2 import Environment, PackageLoader, select_autoescape
from starlette.responses import Response

from hypha.core import Card, UserInfo, ServiceInfo, UserPermission
from hypha.core.auth import parse_user
from hypha.core.store import RedisStore
from hypha.plugin_parser import convert_config_to_card, parse_imjoy_plugin
from hypha.runner.browser import BrowserAppRunner
from hypha.utils import (
    PLUGIN_CONFIG_FIELDS,
    list_objects_async,
    remove_objects_async,
    safe_join,
    random_id,
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
        store: RedisStore,
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
        self._sessions = {}
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
        self.local_base_url = store.local_base_url
        self.public_base_url = store.public_base_url
        self._rpc_lib_script = "https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.12/dist/hypha-rpc-websocket.min.js"
        # self._rpc_lib_script = "http://localhost:9099/hypha-rpc-websocket.js"
        self.event_bus = store.get_event_bus()
        self.store = store
        # self.event_bus.on("workspace_unloaded", self._on_workspace_unloaded)
        store.register_public_service(self.get_service_api())
        self.jinja_env = Environment(
            loader=PackageLoader("hypha"), autoescape=select_autoescape()
        )
        self.templates_dir = Path(__file__).parent / "templates"
        self.builtin_apps_dir = Path(__file__).parent / "built-in"
        shutil.rmtree(self.apps_dir / "built-in", ignore_errors=True)
        # copy files inside the builtin apps dir to the apps dir (overwrite if exists)
        shutil.copytree(self.builtin_apps_dir, self.apps_dir / "built-in")
        router = APIRouter()
        # start the browser runner
        self._runner = [
            BrowserAppRunner(self.store, in_docker=self.in_docker),
            BrowserAppRunner(self.store, in_docker=self.in_docker),
        ]

        @router.get("/{workspace}/a/{path:path}")
        async def get_app_file(
            workspace: str, path: str, token: str = None
        ) -> Response:
            if workspace == "built-in":
                # get the jinja template from the built-in apps dir
                path = safe_join(str(self.builtin_apps_dir), path)
                if not is_safe_path(str(self.builtin_apps_dir), path):
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": f"Unsafe path: {path}",
                        },
                    )
                if not os.path.exists(path):
                    return JSONResponse(
                        status_code=404,
                        content={"success": False, "detail": f"File not found: {path}"},
                    )
                # compile the jinja template
                template = self.jinja_env.get_template(path)
                return Response(template.render(rpc_lib_script=self._rpc_lib_script))
            else:
                if token is None:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": (f"Token not provided for {workspace}/{path}"),
                        },
                    )
                try:
                    user_info = parse_user(token)
                except jose.exceptions.JWTError:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": (
                                f"Invalid token not provided for {workspace}/{path}"
                            ),
                        },
                    )
                if not user_info.check_permission(workspace, UserPermission.read):
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

        store.register_router(router)

        def close(_) -> None:
            asyncio.ensure_future(self.close())

        self.event_bus.on_local("shutdown", close)

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

    async def _on_workspace_unloaded(self, workspace: dict):
        # Shutdown the apps in the workspace
        for app in list(self._sessions.values()):
            if app["workspace"] == workspace["name"]:
                logger.info(
                    "Shutting down app %s (since workspace %s has been removed)",
                    app["id"],
                    workspace["name"],
                )
                await self.stop(app["id"])

    async def list_saved_workspaces(
        self,
    ):
        """List saved workspaces."""
        async with self.create_client_async() as s3_client:
            items = await list_objects_async(s3_client, self.workspace_bucket, "/")
        return [item["Key"] for item in items]

    async def list_apps(
        self,
        workspace: str = None,
        context: Optional[dict] = None,
    ):
        """List applications in the workspace."""
        if not workspace:
            workspace = context["ws"]

        workspace = await self.store.get_workspace(workspace, load=True)
        assert workspace, f"Workspace {workspace} not found."
        return [app_info.model_dump() for app_info in workspace.applications.values()]

    async def save_application(
        self,
        workspace: str,
        app_id: str,
        card: Card,
        source: str,
        attachments: Optional[Dict[str, Any]] = None,
    ):
        """Save an application to the workspace."""
        mhash = app_id
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
                card.attachments = card.attachments or {}
                card.attachments["files"] = card.attachments.get("files", [])
                files = card.attachments["files"]
                for att in attachments:
                    assert (
                        "name" in att and "source" in att
                    ), "Attachment should contain `name` and `source`"
                    if att["source"].startswith("http") and "\n" not in att["source"]:
                        if not att["source"].startswith("https://"):
                            raise Exception(
                                "Only https sources are allowed: " + att["source"]
                            )
                        with urlopen(att["source"]) as stream:
                            output = stream.read()
                        att["source"] = output
                    await save_file(f"{app_dir}/{att['name']}", att["source"])
                    files.append(att["name"])

            content = json.dumps(card.model_dump(), indent=4)
            await save_file(f"{app_dir}/manifest.json", content)
        logger.info("Saved application (%s)to workspace: %s", mhash, workspace)

    async def prepare_application(self, workspace, app_id):
        """Download files for an application to be run."""
        assert "/" not in app_id, (
            "Invalid app id: " + app_id + ", should not contain '/'"
        )
        local_app_dir = self.apps_dir / workspace / app_id
        mhash = app_id
        # if os.path.exists(local_app_dir):
        #     logger.info("Application (%s) is already prepared.", app_id)
        #     return

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
                async with aiofiles.open(local_path, "wb") as fil:
                    await fil.write(data)

            # Upload the source code and attachments
            await download_file(
                os.path.join(app_dir, "index.html"), local_app_dir / "index.html"
            )
            await download_file(
                os.path.join(app_dir, "manifest.json"), local_app_dir / "manifest.json"
            )

            async with aiofiles.open(
                local_app_dir / "manifest.json", "r", encoding="utf-8"
            ) as fil:
                card = Card.model_validate(json.loads(await fil.read()))

            if card.attachments:
                files = card.attachments.get("files")
                if files:
                    for file_name in files:
                        await download_file(
                            os.path.join(app_dir, file_name), local_app_dir / file_name
                        )
            logger.info("Application (%s) is prepared.", app_id)

    async def close(self) -> None:
        """Close the app controller."""
        logger.info("Closing the server app controller...")
        for app in self._sessions.values():
            await self.stop(app["id"])

    def get_service_api(self) -> Dict[str, Any]:
        """Get a list of service api."""
        # TODO: check permission for each function
        controller = {
            "name": "Server Apps",
            "id": "server-apps",
            "type": "server-apps",
            "config": {"visibility": "public", "require_context": True},
            "install": self.install,
            "uninstall": self.uninstall,
            "launch": self.launch,
            "start": self.start,
            "stop": self.stop,
            "list_apps": self.list_apps,
            "list_running": self.list_running,
            "get_log": self.get_log,
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
        timeout: float = 60,
        force: bool = False,
        context: Optional[dict] = None,
    ) -> str:
        """Save a server app."""
        if template is None:
            if config:
                template = config.get("type") + "-app.html"
            else:
                template = "hypha"
        if not workspace:
            workspace = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        workspace_info = await self.store.get_workspace(workspace, load=True)
        assert workspace_info, f"Workspace {workspace} not found."
        if not user_info.check_permission(
            workspace_info.name, UserPermission.read_write
        ):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to install apps in workspace {workspace_info.name}"
            )

        if source.startswith("http"):
            if not source.startswith("https://"):
                raise Exception("Only https sources are allowed: " + source)
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
        if template == "hypha":
            if not source:
                raise Exception("Source should be provided for hypha app.")
            assert config is None, "Config should not be provided for hypha app."

            try:
                config = parse_imjoy_plugin(source)
                config["source_hash"] = mhash
                temp = self.jinja_env.get_template(config["type"] + "-app.html")
                source = temp.render(
                    config={k: config[k] for k in config if k in PLUGIN_CONFIG_FIELDS},
                    script=config["script"],
                    requirements=config["requirements"],
                    local_base_url=self.local_base_url,
                    rpc_lib_script=self._rpc_lib_script,
                )
            except Exception as err:
                raise Exception(
                    f"Failed to parse or compile the hypha app {mhash}: {source[:100]}...",
                ) from err
        elif template:
            temp = self.jinja_env.get_template(template)
            default_config = {
                "name": "Untitled Plugin",
                "version": "0.1.0",
                "local_base_url": self.local_base_url,
            }
            default_config.update(config or {})
            config = default_config
            source = temp.render(
                script=source,
                source_hash=mhash,
                config=config,
                requirements=config.get("requirements", []),
                rpc_lib_script=self._rpc_lib_script,
            )
        elif not source:
            raise Exception("Source or template should be provided.")

        app_id = f"{mhash}"

        public_url = (
            f"{self.public_base_url}/{workspace_info.name}/a/{app_id}/index.html"
        )
        card_obj = convert_config_to_card(config, app_id, public_url)
        card_obj.update(
            {
                "local_url": f"{self.local_base_url}/{workspace_info.name}/a/{app_id}/index.html",
                "public_url": public_url,
            }
        )
        card = Card.model_validate(card_obj)
        await self.save_application(
            workspace_info.name, app_id, card, source, attachments
        )
        async with self.store.get_workspace_interface(
            workspace_info.name, user_info
        ) as ws:
            await ws.install_application(card.model_dump(), force=force)
        try:
            info = await self.start(
                app_id,
                timeout=timeout,
                wait_for_service="default",
                context=context,
            )
            await self.stop(info["id"], context=context)
        except asyncio.TimeoutError:
            logger.error("Failed to start the app: %s during installation", app_id)
            await self.uninstall(app_id, context=context)
            raise TimeoutError(
                "Failed to start the app: %s during installation" % app_id
            )
        except Exception as exp:
            logger.exception("Failed to start the app: %s during installation", app_id)
            await self.uninstall(app_id, context=context)
            raise Exception(
                f"Failed to start the app: {app_id} during installation, error: {exp}"
            )

        return card_obj

    async def uninstall(self, app_id: str, context: Optional[dict] = None) -> None:
        """Uninstall a server app."""
        assert "/" not in app_id, (
            "Invalid app id: " + app_id + ", should not contain '/'"
        )
        workspace_name = context["ws"]
        mhash = app_id
        workspace = await self.store.get_workspace(workspace_name, load=True)
        assert workspace, f"Workspace {workspace} not found."
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(workspace.name, UserPermission.read_write):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to uninstall apps in workspace {workspace.name}"
            )
        async with self.store.get_workspace_interface(workspace.name, user_info) as ws:
            await ws.uninstall_application(app_id)

        async with self.create_client_async() as s3_client:
            app_dir = f"{workspace.name}/{self.user_applications_dir}/{mhash}/"
            await remove_objects_async(s3_client, self.workspace_bucket, app_dir)
        if (self.apps_dir / workspace.name / app_id).exists():
            shutil.rmtree(self.apps_dir / workspace.name / app_id, ignore_errors=True)

    async def launch(
        self,
        source: str,
        timeout: float = 60,
        config: Optional[Dict[str, Any]] = None,
        attachments: List[dict] = None,
        wait_for_service: str = None,
        context: Optional[dict] = None,
    ) -> dict:
        """Start a server app instance."""
        workspace = context["ws"]
        app_info = await self.install(
            source,
            attachments=attachments,
            config=config,
            workspace=workspace,
            context=context,
        )
        app_id = app_info["id"]
        return await self.start(
            app_id,
            timeout=timeout,
            wait_for_service=wait_for_service,
            context=context,
        )

    # pylint: disable=too-many-statements,too-many-locals
    async def start(
        self,
        app_id,
        client_id=None,
        timeout: float = 60,
        wait_for_service: Union[str, bool] = None,
        context: Optional[dict] = None,
    ):
        """Start the app and keep it alive."""
        if wait_for_service is True:
            wait_for_service = "default"
        if wait_for_service and ":" in wait_for_service:
            wait_for_service = wait_for_service.split(":")[1]

        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        async with self.store.get_workspace_interface(workspace, user_info) as ws:
            token = await ws.generate_token()

        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to run app {app_id} in workspace {workspace}."
            )

        if client_id is None:
            client_id = random_id(readable=True)

        workspace_info = await self.store.get_workspace(workspace, load=True)
        assert workspace, f"Workspace {workspace} not found."
        assert (
            app_id in workspace_info.applications
        ), f"App {app_id} not found in workspace {workspace}, please install it first."

        await self.prepare_application(workspace, app_id)
        server_url = self.local_base_url.replace("http://", "ws://")
        server_url = server_url.replace("https://", "wss://")
        local_url = (
            f"{self.local_base_url}/{workspace}/a/{app_id}/index.html?"
            + f"client_id={client_id}&workspace={workspace}"
            + f"&app_id={app_id}"
            + f"&server_url={server_url}/ws"
            + f"&token={token}"
            if token
            else ""
        )
        server_url = self.public_base_url.replace("http://", "ws://")
        server_url = server_url.replace("https://", "wss://")
        public_url = (
            f"{self.public_base_url}/{workspace}/a/{app_id}/index.html?"
            + f"client_id={client_id}&workspace={workspace}"
            + f"&app_id={app_id}"
            + f"&server_url={server_url}/ws"
            + f"&token={token}"
            if token
            else ""
        )

        runner = random.choice(self._runner)

        full_client_id = workspace + "/" + client_id
        await runner.start(url=local_url, session_id=full_client_id)
        self._sessions[full_client_id] = {
            "id": full_client_id,
            "app_id": app_id,
            "workspace": workspace,
            "local_url": local_url,
            "public_url": public_url,
            "_runner": runner,
        }
        # collecting services registered during the startup of the script
        collected_services: List[ServiceInfo] = []
        app_info = {
            "id": full_client_id,
            "app_id": app_id,
            "workspace": workspace,
            "config": {},
        }

        def service_added(info: dict):
            if info["id"].startswith(full_client_id + ":"):
                collected_services.append(ServiceInfo.model_validate(info))
            if info["id"] == full_client_id + ":default":
                for key in ["config", "name", "description"]:
                    if info.get(key):
                        app_info[key] = info[key]

        self.event_bus.on_local("service_added", service_added)

        try:
            if wait_for_service:
                print(f"Waiting for service: {full_client_id}:{wait_for_service}")
                await self.event_bus.wait_for_local(
                    "service_added",
                    match={"id": full_client_id + ":" + wait_for_service},
                    timeout=timeout,
                )
            else:
                await self.event_bus.wait_for_local(
                    "client_connected", match={"id": full_client_id}, timeout=timeout
                )

            # save the services
            workspace_info.applications[app_id].services = collected_services
            Card.model_validate(workspace_info.applications[app_id])
            await self.store.set_workspace(workspace_info, user_info)
        except asyncio.TimeoutError:
            raise Exception(
                f"Failed to start the app: {workspace}/{app_id}, timeout reached ({timeout}s)."
            )
        except Exception as exp:
            raise Exception(
                f"Failed to start the app: {workspace}/{app_id}, error: {exp}"
            )
        finally:
            self.event_bus.off_local("service_added", service_added)

        return app_info

    async def stop(
        self,
        session_id: str,
        raise_exception=True,
        context: Optional[dict] = None,
    ) -> None:
        """Stop a server app instance."""
        if session_id in self._sessions:
            logger.info("Stopping app: %s...", session_id)

            app_info = self._sessions[session_id]
            if session_id in self._sessions:
                del self._sessions[session_id]
            try:
                await app_info["_runner"].stop(session_id)
            except Exception as exp:
                if raise_exception:
                    raise
                else:
                    logger.warning("Failed to stop browser tab: %s", exp)
        elif raise_exception:
            raise Exception(f"Server app instance not found: {session_id}")

    async def get_log(
        self,
        session_id: str,
        type: str = None,  # pylint: disable=redefined-builtin
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[dict] = None,
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get server app instance log."""
        if session_id in self._sessions:
            return await self._sessions[session_id]["_runner"].get_log(
                session_id, type=type, offset=offset, limit=limit
            )
        else:
            raise Exception(f"Server app instance not found: {session_id}")

    async def list_running(self, context: Optional[dict] = None) -> List[str]:
        """List the running sessions for the current workspace."""
        workspace = context["ws"]
        sessions = [
            {k: v for k, v in session_info.items() if not k.startswith("_")}
            for session_id, session_info in self._sessions.items()
            if session_id.startswith(workspace + "/")
        ]
        return sessions
