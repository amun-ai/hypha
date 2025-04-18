import httpx
import logging
import os
import sys
import multihash
import asyncio
import logging
import sys
from pathlib import Path

from hypha import main_version
from jinja2 import Environment, PackageLoader, select_autoescape
from typing import Any, Dict, List, Optional, Union
from hypha.core import UserInfo, UserPermission, ServiceInfo, ApplicationManifest
from hypha.utils import (
    random_id,
    PLUGIN_CONFIG_FIELDS,
    safe_join,
)
import base58
import random
from hypha.plugin_parser import convert_config_to_artifact, parse_imjoy_plugin
from hypha.core import WorkspaceInfo

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("apps")
logger.setLevel(LOGLEVEL)

multihash.CodecReg.register("base58", base58.b58encode, base58.b58decode)


class ServerAppController:
    """Server App Controller."""

    def __init__(
        self,
        store,
        in_docker,
        port: int,
        artifact_manager,
    ):
        """Initialize the controller."""
        self.port = int(port)
        self.store = store
        self.in_docker = in_docker
        self.artifact_manager = artifact_manager
        self._sessions = {}  # Track running sessions
        self.event_bus = store.get_event_bus()
        self.local_base_url = store.local_base_url
        self.public_base_url = store.public_base_url
        store.register_public_service(self.get_service_api())
        self.jinja_env = Environment(
            loader=PackageLoader("hypha"), autoescape=select_autoescape()
        )
        self.templates_dir = Path(__file__).parent / "templates"

        def shutdown(_) -> None:
            asyncio.ensure_future(self.shutdown())

        self.event_bus.on_local("shutdown", shutdown)

        async def client_disconnected(info: dict) -> None:
            """Handle client disconnected event."""
            # {"id": client_id, "workspace": ws}
            client_id = info["id"]
            full_client_id = info["workspace"] + "/" + client_id
            if full_client_id in self._sessions:
                app_info = self._sessions.pop(full_client_id, None)
                try:
                    await app_info["_runner"].stop(full_client_id)
                except Exception as exp:
                    logger.warning(f"Failed to stop browser tab: {exp}")

        self.event_bus.on_local("client_disconnected", client_disconnected)
        store.set_server_app_controller(self)

    async def get_runners(self):
        # start the browser runner
        server = await self.store.get_public_api()
        svcs = await server.list_services("public/server-app-worker")
        if not svcs:
            return []
        runners = [await server.get_service(svc["id"]) for svc in svcs]
        if runners:
            return runners
        else:
            []

    async def setup_applications_collection(self, overwrite=True, context=None):
        """Set up the workspace."""
        ws = context["ws"]
        # Create an collection in the workspace
        manifest = {
            "id": "applications",
            "name": "Applications",
            "description": f"A collection of applications for workspace {ws}",
        }
        collection = await self.artifact_manager.create(
            type="collection",
            alias="applications",
            manifest=manifest,
            overwrite=overwrite,
            context=context,
        )
        logger.info(f"Applications collection created for workspace {ws}")
        return collection["id"]

    async def install(
        self,
        source: str = None,
        source_hash: str = None,
        config: Optional[Dict[str, Any]] = None,
        workspace: Optional[str] = None,
        overwrite: bool = False,
        timeout: float = 60,
        version: str = None,
        context: Optional[dict] = None,
    ) -> str:
        """Save a server app."""
        if not workspace:
            workspace = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        workspace_info = await self.store.get_workspace_info(workspace, load=True)
        assert workspace_info, f"Workspace {workspace} not found."
        if not user_info.check_permission(workspace_info.id, UserPermission.read_write):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to install apps in workspace {workspace_info.id}"
            )

        if config:
            config["entry_point"] = config.get("entry_point", "index.html")
            template = config.get("type") + "." + config["entry_point"]
        else:
            template = "hypha"

        if source.startswith("http"):
            if not (
                source.startswith("https://")
                or source.startswith("http://localhost")
                or source.startswith("http://127.0.0.1")
            ):
                raise Exception("Only secured https urls are allowed: " + source)
            if source.startswith("https://") and (
                source.split("?")[0].endswith(".imjoy.html")
                or source.split("?")[0].endswith(".hypha.html")
            ):
                # download source with httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(source)
                    assert response.status_code == 200, f"Failed to download {source}"
                    source = response.text
            else:
                template = None

        # Compute multihash of the source code
        mhash = multihash.digest(source.encode("utf-8"), "sha2-256")
        mhash = mhash.encode("base58").decode("ascii")
        # Verify the source code, useful for downloading from the web
        if source_hash is not None:
            target_mhash = multihash.decode(source_hash.encode("ascii"), "base58")
            assert target_mhash.verify(
                source.encode("utf-8")
            ), f"App source code verification failed (source_hash: {source_hash})."

        if template is None:
            config = config or {}
            config["entry_point"] = config.get("entry_point", source)
            entry_point = config["entry_point"]
        elif template == "hypha":
            if not source:
                raise Exception("Source should be provided for hypha app.")
            assert config is None, "Config should not be provided for hypha app."

            try:
                config = parse_imjoy_plugin(source)
                config["source_hash"] = mhash
                entry_point = config.get("entry_point", "index.html")
                config["entry_point"] = entry_point
                temp = self.jinja_env.get_template(
                    safe_join("apps", config["type"] + "." + entry_point)
                )

                source = temp.render(
                    hypha_main_version=main_version,
                    hypha_rpc_websocket_mjs=self.public_base_url
                    + "/assets/hypha-rpc-websocket.mjs",
                    config={k: config[k] for k in config if k in PLUGIN_CONFIG_FIELDS},
                    script=config["script"],
                    requirements=config["requirements"],
                    local_base_url=self.local_base_url,
                )
            except Exception as err:
                raise Exception(
                    f"Failed to parse or compile the hypha app {mhash}: {source[:100]}...",
                ) from err
        elif template:
            assert (
                "." in template
            ), f"Invalid template name: {template}, should be a file name with extension."
            # extract the last dash separated part as the file name
            temp = self.jinja_env.get_template(safe_join("apps", template))
            default_config = {
                "name": "Untitled App",
                "version": "0.1.0",
                "local_base_url": self.local_base_url,
            }
            default_config.update(config or {})
            config = default_config
            entry_point = config.get("entry_point", template)
            config["entry_point"] = entry_point
            source = temp.render(
                hypha_main_version=main_version,
                hypha_rpc_websocket_mjs=self.public_base_url
                + "/assets/hypha-rpc-websocket.mjs",
                script=source,
                source_hash=mhash,
                config=config,
                requirements=config.get("requirements", []),
            )
        elif not source:
            raise Exception("Source or template should be provided.")

        app_id = f"{mhash}"

        if template:
            public_url = f"{self.public_base_url}/{workspace_info.id}/artifacts/applications:{app_id}/files/{entry_point}"
            artifact_obj = convert_config_to_artifact(config, app_id, public_url)
            artifact_obj.update(
                {
                    "local_url": f"{self.local_base_url}/{workspace_info.id}/artifacts/applications:{app_id}/files/{entry_point}",
                    "public_url": public_url,
                }
            )
        else:
            artifact_obj = convert_config_to_artifact(config, app_id, source)
        ApplicationManifest.model_validate(artifact_obj)

        try:
            artifact = await self.artifact_manager.read("applications", context=context)
            collection_id = artifact["id"]
        except KeyError:
            collection_id = await self.setup_applications_collection(
                overwrite=True, context=context
            )

        # Create artifact using the artifact controller
        artifact = await self.artifact_manager.create(
            parent_id=collection_id,
            alias=f"applications:{app_id}",
            manifest=artifact_obj,
            overwrite=overwrite,
            version="stage",
            context=context,
        )

        if template:
            # Upload the main source file
            put_url = await self.artifact_manager.put_file(
                artifact["id"], file_path=config["entry_point"], context=context
            )
            async with httpx.AsyncClient() as client:
                response = await client.put(put_url, data=source)
                assert (
                    response.status_code == 200
                ), f"Failed to upload {config['entry_point']}"

        if version != "stage":
            # Commit the artifact if stage is not enabled
            await self.commit(
                app_id,
                timeout=timeout,
                version=version,
                context=context,
            )
        return artifact_obj

    async def add_file(
        self,
        app_id: str,
        file_path: str,
        file_content: str,
        context: Optional[dict] = None,
    ):
        """Add a file to the installed application."""
        put_url = await self.artifact_manager.put_file(
            f"applications:{app_id}", file_path=file_path, context=context
        )
        response = httpx.put(put_url, data=file_content)
        assert response.status_code == 200, f"Failed to upload {file_path} to {app_id}"

    async def remove_file(
        self,
        app_id: str,
        file_path: str,
        context: Optional[dict] = None,
    ):
        """Remove a file from the installed application."""
        await self.artifact_manager.remove_file(
            f"applications:{app_id}", file_path=file_path, context=context
        )

    async def list_files(
        self, app_id: str, context: Optional[dict] = None
    ) -> List[dict]:
        """List files of an installed application."""
        return await self.artifact_manager.list_files(
            f"applications:{app_id}", context=context
        )

    async def edit(self, app_id: str, context: Optional[dict] = None):
        """Edit an application by re-opening its artifact."""
        await self.artifact_manager.edit(
            f"applications:{app_id}", version="stage", context=context
        )

    async def commit(
        self,
        app_id: str,
        timeout: int = 30,
        version: str = None,
        context: Optional[dict] = None,
    ):
        """Finalize the edits to the application by committing the artifact."""
        try:
            info = await self.start(
                app_id,
                timeout=timeout,
                wait_for_service="default",
                version="stage",
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
        await self.artifact_manager.commit(
            f"applications:{app_id}", version=version, context=context
        )

    async def uninstall(self, app_id: str, context: Optional[dict] = None) -> None:
        """Uninstall an application by removing its artifact."""
        await self.artifact_manager.delete(f"applications:{app_id}", context=context)

    async def launch(
        self,
        source: str,
        timeout: float = 60,
        config: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        wait_for_service: str = None,
        context: Optional[dict] = None,
    ) -> dict:
        """Start a server app instance."""
        app_info = await self.install(
            source,
            config=config,
            overwrite=overwrite,
            context=context,
        )
        app_id = app_info["id"]
        return await self.start(
            app_id,
            timeout=timeout,
            wait_for_service=wait_for_service,
            context=context,
        )

    async def start(
        self,
        app_id,
        client_id=None,
        timeout: float = 60,
        version: str = None,
        wait_for_service: Union[str, bool] = None,
        stop_after_inactive: Optional[int] = None,
        context: Optional[dict] = None,
    ):
        """Start the app and keep it alive."""
        if wait_for_service is True:
            wait_for_service = "default"
        if wait_for_service and ":" in wait_for_service:
            wait_for_service = wait_for_service.split(":")[1]

        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        async with self.store.get_workspace_interface(user_info, workspace) as ws:
            token = await ws.generate_token()

        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to run app {app_id} in workspace {workspace}."
            )

        if client_id is None:
            client_id = random_id(readable=True)

        artifact_info = await self.artifact_manager.read(
            f"applications:{app_id}", version=version, context=context
        )
        manifest = artifact_info.get("manifest", {})
        manifest = ApplicationManifest.model_validate(manifest)
        if manifest.singleton:
            # check if the app is already running
            for session_info in self._sessions.values():
                if session_info["app_id"] == app_id:
                    raise RuntimeError(
                        f"App {app_id} is a singleton app and already running (id: {session_info['id']})"
                    )
        if manifest.daemon and stop_after_inactive and stop_after_inactive > 0:
            raise ValueError("Daemon apps should not have stop_after_inactive set.")
        if stop_after_inactive is None:
            stop_after_inactive = (
                600
                if manifest.stop_after_inactive is None
                else manifest.stop_after_inactive
            )
        entry_point = manifest.entry_point
        assert entry_point, f"Entry point not found for app {app_id}."
        if not entry_point.startswith("http"):
            entry_point = f"{self.local_base_url}/{workspace}/artifacts/applications:{app_id}/files/{entry_point}"
        server_url = self.local_base_url
        local_url = (
            f"{entry_point}?"
            + f"server_url={server_url}&client_id={client_id}&workspace={workspace}"
            + f"&app_id={app_id}"
            + f"&server_url={server_url}"
            + (f"&token={token}" if token else "")
            + (f"&version={version}" if version else "")
            + (f"&use_proxy=true")
        )
        server_url = self.public_base_url
        public_url = (
            f"{self.public_base_url}/{workspace}/artifacts/applications:{app_id}/files/{entry_point}?"
            + f"client_id={client_id}&workspace={workspace}"
            + f"&app_id={app_id}"
            + f"&server_url={server_url}"
            + (f"&token={token}" if token else "")
            + (f"&version={version}" if version else "")
            + (f"&use_proxy=true")
        )
        runners = await self.get_runners()
        if not runners:
            raise Exception("No server app worker found")
        runner = random.choice(runners)

        full_client_id = workspace + "/" + client_id
        metadata = {
            "id": full_client_id,
            "app_id": app_id,
            "workspace": workspace,
            "local_url": local_url,
            "public_url": public_url,
            "_runner": runner,
        }

        await runner.start(
            url=local_url,
            session_id=full_client_id,
            metadata=metadata,
        )
        self._sessions[full_client_id] = metadata

        # test activity tracker
        tracker = self.store.get_activity_tracker()
        if not manifest.daemon and stop_after_inactive and stop_after_inactive > 0:

            async def _stop_after_inactive():
                if full_client_id in self._sessions:
                    await runner.stop(full_client_id)
                logger.info(
                    f"App {full_client_id} stopped because of inactive for {stop_after_inactive}s."
                )

            tracker.register(
                full_client_id,
                inactive_period=stop_after_inactive,
                on_inactive=_stop_after_inactive,
                entity_type="client",
            )

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
                logger.info(f"Waiting for service: {full_client_id}:{wait_for_service}")
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
            manifest.name = manifest.name or app_info.get("name", "Untitled App")
            manifest.description = manifest.description or app_info.get(
                "description", ""
            )
            manifest.services = collected_services
            manifest = ApplicationManifest.model_validate(
                manifest.model_dump(mode="json")
            )
            await self.artifact_manager.edit(
                f"applications:{app_id}",
                version=version,
                manifest=manifest.model_dump(mode="json"),
                context=context,
            )

        except asyncio.TimeoutError:
            logs = await runner.get_log(full_client_id)
            await runner.stop(full_client_id)
            raise Exception(
                f"Failed to start the app: {workspace}/{app_id}, timeout reached ({timeout}s), browser logs:\n{logs}"
            )
        except Exception as exp:
            logs = await runner.get_log(full_client_id)
            await runner.stop(full_client_id)
            raise Exception(
                f"Failed to start the app: {workspace}/{app_id}, error: {exp}, browser logs:\n{logs}"
            ) from exp
        finally:
            self.event_bus.off_local("service_added", service_added)

        if wait_for_service:
            app_info["service_id"] = (
                full_client_id + ":" + wait_for_service + "@" + app_id
            )
        return app_info

    async def stop(
        self, session_id: str, raise_exception=True, context: Optional[dict] = None
    ) -> None:
        """Stop a server app instance."""
        user_info = UserInfo.model_validate(context["user"])
        workspace = context["ws"]
        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to stop app {session_id} in workspace {workspace}."
            )
        await self._stop(session_id, raise_exception=raise_exception)

    async def _stop(self, session_id: str, raise_exception=True):
        if session_id in self._sessions:
            app_info = self._sessions.pop(session_id, None)
            try:
                await app_info["_runner"].stop(session_id)
            except Exception as exp:
                if raise_exception:
                    raise
                else:
                    logger.warning(f"Failed to stop browser tab: {exp}")
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
        user_info = UserInfo.model_validate(context["user"])
        workspace = context["ws"]
        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to get log for app {session_id} in workspace {workspace}."
            )
        if session_id in self._sessions:
            return await self._sessions[session_id]["_runner"].get_log(
                session_id, type=type, offset=offset, limit=limit
            )
        else:
            raise Exception(f"Server app instance not found: {session_id}")

    async def list_running(self, context: Optional[dict] = None) -> List[str]:
        """List the running sessions for the current workspace."""
        workspace = context["ws"]
        return [
            {k: v for k, v in session_info.items() if not k.startswith("_")}
            for session_id, session_info in self._sessions.items()
            if session_id.startswith(f"{workspace}/")
        ]

    async def list_apps(self, context: Optional[dict] = None):
        """List applications in the workspace."""
        try:
            ws = context["ws"]
            apps = await self.artifact_manager.list_children(
                f"{ws}/applications", context=context
            )
            return [app["manifest"] for app in apps]
        except KeyError:
            return []
        except Exception as exp:
            raise Exception(f"Failed to list apps: {exp}") from exp

    async def shutdown(self) -> None:
        """Shutdown the app controller."""
        logger.info("Closing the server app controller...")
        for app in self._sessions.values():
            await self.stop(app["id"], raise_exception=False)

    async def prepare_workspace(self, workspace_info: WorkspaceInfo):
        """Prepare the workspace."""
        context = {
            "ws": workspace_info.id,
            "user": self.store.get_root_user().model_dump(),
        }
        apps = await self.list_apps(context=context)
        # start daemon apps
        for app in apps:
            if app.get("daemon"):
                logger.info(f"Starting daemon app: {app['id']}")
                try:
                    await self.start(app["id"], context=context)
                except Exception as exp:
                    logger.error(
                        f"Failed to start daemon app: {app['id']}, error: {exp}"
                    )

    async def close_workspace(self, workspace_info: WorkspaceInfo):
        """Archive the workspace."""
        # Stop all running apps
        for app in list(self._sessions.values()):
            if app["workspace"] == workspace_info.id:
                await self._stop(app["id"], raise_exception=False)
        # Send to all runners
        runners = await self.get_runners()
        if not runners:
            return
        for runner in runners:
            try:
                await runner.close_workspace(workspace_info.id)
            except Exception as exp:
                logger.warning(
                    f"Worker failed to close workspace: {workspace_info.id}, error: {exp}"
                )

    def get_service_api(self) -> Dict[str, Any]:
        """Get a list of service API endpoints."""
        return {
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
            "add_file": self.add_file,
            "remove_file": self.remove_file,
            "list_files": self.list_files,
            "edit": self.edit,
            "commit": self.commit,
        }
