import httpx
import logging
import sys
import multihash
import asyncio
import logging
import sys
from pathlib import Path

from hypha import main_version
from jinja2 import Environment, PackageLoader, select_autoescape
from typing import Any, Dict, List, Optional, Union
from hypha.core import UserInfo, UserPermission, ServiceInfo, ApplicationArtifact
from hypha.runner.browser import BrowserAppRunner
from hypha.utils import (
    random_id,
    PLUGIN_CONFIG_FIELDS,
    safe_join,
)
import base58
import random
from hypha.plugin_parser import convert_config_to_artifact, parse_imjoy_plugin

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("apps")
logger.setLevel(logging.INFO)

multihash.CodecReg.register("base58", base58.b58encode, base58.b58decode)


class ServerAppController:
    """Server App Controller."""

    def __init__(
        self,
        store,
        in_docker,
        port: int,
        artifact_manager,
        workspace_bucket="hypha-workspaces",
    ):
        """Initialize the controller."""
        self.port = int(port)
        self.store = store
        self.in_docker = in_docker
        self.artifact_manager = artifact_manager
        self.workspace_bucket = workspace_bucket
        self._sessions = {}  # Track running sessions
        self.event_bus = store.get_event_bus()
        self.local_base_url = store.local_base_url
        self.public_base_url = store.public_base_url
        store.register_public_service(self.get_service_api())
        self.jinja_env = Environment(
            loader=PackageLoader("hypha"), autoescape=select_autoescape()
        )
        self.templates_dir = Path(__file__).parent / "templates"
        # start the browser runner
        self._runner = [
            BrowserAppRunner(self.store, in_docker=self.in_docker),
            BrowserAppRunner(self.store, in_docker=self.in_docker),
        ]

        def close(_) -> None:
            asyncio.ensure_future(self.close())

        self.event_bus.on_local("shutdown", close)

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
        if config:
            config["entry_point"] = config.get("entry_point", "index.html")
            template = config.get("type") + "." + config["entry_point"]
        else:
            template = "hypha"
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

        if source.startswith("http"):
            if not source.startswith("https://"):
                raise Exception("Only https sources are allowed: " + source)
            # download source with httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(source)
                assert response.status_code == 200, f"Failed to download {source}"
                source = response.text
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
                + "//assets/hypha-rpc-websocket.mjs",
                script=source,
                source_hash=mhash,
                config=config,
                requirements=config.get("requirements", []),
            )
        elif not source:
            raise Exception("Source or template should be provided.")

        app_id = f"{mhash}"

        public_url = f"{self.public_base_url}/{workspace_info.id}/artifacts/applications:{app_id}/files/{entry_point}"
        artifact_obj = convert_config_to_artifact(config, app_id, public_url)
        artifact_obj.update(
            {
                "local_url": f"{self.local_base_url}/{workspace_info.id}/artifacts/applications:{app_id}/files/{entry_point}",
                "public_url": public_url,
            }
        )
        ApplicationArtifact.model_validate(artifact_obj)

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
        artifact = artifact_info.get("manifest", {})
        artifact = ApplicationArtifact.model_validate(artifact)

        entry_point = artifact.entry_point
        assert entry_point, f"Entry point not found for app {app_id}."
        server_url = self.local_base_url
        local_url = (
            f"{self.local_base_url}/{workspace}/artifacts/applications:{app_id}/files/{entry_point}?"
            + f"client_id={client_id}&workspace={workspace}"
            + f"&app_id={app_id}"
            + f"&server_url={server_url}"
            + (f"&token={token}" if token else "")
            + (f"&version={version}" if version else "")
        )
        server_url = self.public_base_url
        public_url = (
            f"{self.public_base_url}/{workspace}/artifacts/applications:{app_id}/files/{entry_point}?"
            + f"client_id={client_id}&workspace={workspace}"
            + f"&app_id={app_id}"
            + f"&server_url={server_url}"
            + (f"&token={token}" if token else "")
            + (f"&version={version}" if version else "")
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
            artifact.services = collected_services
            artifact = ApplicationArtifact.model_validate(
                artifact.model_dump(mode="json")
            )
            await self.artifact_manager.edit(
                f"applications:{app_id}",
                version=version,
                manifest=artifact.model_dump(mode="json"),
                context=context,
            )

        except asyncio.TimeoutError:
            raise Exception(
                f"Failed to start the app: {workspace}/{app_id}, timeout reached ({timeout}s)."
            )
        except Exception as exp:
            raise Exception(
                f"Failed to start the app: {workspace}/{app_id}, error: {exp}"
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
            apps = await self.artifact_manager.list_children(
                "applications", context=context
            )
            return [app["manifest"] for app in apps]
        except KeyError:
            return []
        except Exception as exp:
            raise Exception(f"Failed to list apps: {exp}") from exp

    async def close(self) -> None:
        """Close the app controller."""
        logger.info("Closing the server app controller...")
        for app in self._sessions.values():
            await self.stop(app["id"])

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
