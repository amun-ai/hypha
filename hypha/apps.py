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
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("apps")
logger.setLevel(LOGLEVEL)

multihash.CodecReg.register("base58", base58.b58encode, base58.b58decode)


# Add a helper function to detect raw HTML content
def is_raw_html_content(source: str) -> bool:
    """Check if source is raw HTML content (not a URL or ImJoy/Hypha template)."""
    if not source or source.startswith("http"):
        return False

    # Check for basic HTML structure
    source_lower = source.lower().strip()
    return source_lower.startswith(("<!doctype html", "<html", "<head", "<body")) or (
        "<html" in source_lower and "</html>" in source_lower
    )


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

    @schema_method
    async def install(
        self,
        source: str = Field(
            None,
            description="The source code of the application, URL to fetch the source, or None if using config. Can be raw HTML content, ImJoy/Hypha plugin code, or a URL to download the source from. URLs must be HTTPS or localhost/127.0.0.1."
        ),
        source_hash: str = Field(
            None,
            description="Optional hash of the source for verification. Use this to ensure the downloaded source matches the expected content."
        ),
        config: Optional[Dict[str, Any]] = Field(
            None,
            description="Application configuration dictionary containing app metadata and settings. Can include startup_config with default startup parameters like timeout, wait_for_service, and stop_after_inactive. Required fields: name, type, version. Optional fields: description, entry_point, requirements, etc."
        ),
        workspace: Optional[str] = Field(
            None,
            description="Target workspace for installation. If not provided, uses the current workspace from context."
        ),
        overwrite: bool = Field(
            False,
            description="Whether to overwrite existing app with same name. Set to True to replace existing installations."
        ),
        timeout: float = Field(
            60,
            description="Maximum time to wait for installation completion in seconds. Increase for complex apps that take longer to start."
        ),
        version: str = Field(
            None,
            description="Version identifier for the app. If not provided, uses default versioning."
        ),
        stage: bool = Field(
            False,
            description="Whether to install the app in stage mode. If True, the app will be installed as a staged artifact that can be discarded or committed later."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> str:
        """Save a server app.

        Args:
            source: The source code of the application, URL to fetch the source, or None if using config
            source_hash: Optional hash of the source for verification
            config: Application configuration dictionary containing app metadata and settings.
                   Can include startup_config with default startup parameters like timeout,
                   wait_for_service, and stop_after_inactive.
            workspace: Target workspace for installation (defaults to current workspace)
            overwrite: Whether to overwrite existing app with same name
            timeout: Maximum time to wait for installation completion
            version: Version identifier for the app
            context: Additional context information
        """
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

        # Determine template type
        if config and config.get("type"):
            config["entry_point"] = config.get("entry_point", "index.html")
            template = config.get("type") + "." + config["entry_point"]
        else:
            template = "hypha"

        # Handle different source types
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
        elif is_raw_html_content(source):
            # Handle raw HTML content
            if not config:
                config = {
                    "name": "Raw HTML App",
                    "version": "0.1.0",
                    "type": "window",
                    "entry_point": "index.html",
                }
            else:
                config["entry_point"] = config.get("entry_point", "index.html")
                config["type"] = config.get("type", "window")
            template = "raw_html"

        # Compute multihash of the source code
        mhash = multihash.digest(source.encode("utf-8"), "sha2-256")
        mhash = mhash.encode("base58").decode("ascii")
        # Verify the source code, useful for downloading from the web
        if source_hash is not None:
            target_mhash = multihash.decode(source_hash.encode("ascii"), "base58")
            assert target_mhash.verify(
                source.encode("utf-8")
            ), f"App source code verification failed (source_hash: {source_hash})."

        # Process based on template type
        if template is None:
            config = config or {}
            config["entry_point"] = config.get("entry_point", source)
            entry_point = config["entry_point"]
        elif template == "raw_html":
            # For raw HTML, we'll upload the content as-is
            entry_point = config["entry_point"]
        elif template == "hypha":
            if not source:
                raise Exception("Source should be provided for hypha app.")

            try:
                # Parse the template config
                template_config = parse_imjoy_plugin(source)
                template_config["source_hash"] = mhash
                entry_point = template_config.get("entry_point", "index.html")
                template_config["entry_point"] = entry_point

                # Merge with provided config, giving priority to template config for app metadata
                # but allowing additional config like startup_config to be provided
                if config:
                    merged_config = config.copy()
                    merged_config.update(
                        template_config
                    )  # Template config overrides provided config
                    config = merged_config
                else:
                    config = template_config

                temp = self.jinja_env.get_template(
                    safe_join("apps", config["type"] + "." + entry_point)
                )

                source = temp.render(
                    hypha_main_version=main_version,
                    hypha_rpc_websocket_mjs=self.local_base_url
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
                hypha_rpc_websocket_mjs=self.local_base_url
                + "/assets/hypha-rpc-websocket.mjs",
                script=source,
                source_hash=mhash,
                config=config,
                requirements=config.get("requirements", []),
            )
        elif not source:
            raise Exception("Source or template should be provided.")

        # Create artifact object first (with placeholder app_id)
        placeholder_app_id = "temp_app_id"
        if template and template != "raw_html":
            # Use placeholder URLs that will be updated after we get the actual app_id
            placeholder_url = f"{self.public_base_url}/{workspace_info.id}/artifacts/{placeholder_app_id}/files/{entry_point}"
            artifact_obj = convert_config_to_artifact(config, placeholder_app_id, placeholder_url)
        elif template == "raw_html":
            # Use placeholder URLs that will be updated after we get the actual app_id
            placeholder_url = f"{self.public_base_url}/{workspace_info.id}/artifacts/{placeholder_app_id}/files/{entry_point}"
            artifact_obj = convert_config_to_artifact(config, placeholder_app_id, placeholder_url)
        else:
            artifact_obj = convert_config_to_artifact(config, placeholder_app_id, source)

        # startup_config is now handled automatically by convert_config_to_artifact
        # if it exists in the config

        # Store startup_context with workspace and user info from installation time
        artifact_obj["startup_context"] = {
            "ws": context["ws"],
            "user": context["user"]
        }
        
        # Store source hash for singleton checking
        artifact_obj["source_hash"] = mhash

        ApplicationManifest.model_validate(artifact_obj)

        try:
            artifact = await self.artifact_manager.read("applications", context=context)
            collection_id = artifact["id"]
        except KeyError:
            collection_id = await self.setup_applications_collection(
                overwrite=True, context=context
            )

        # Create artifact using the artifact controller - let it generate the alias
        artifact = await self.artifact_manager.create(
            type="application",
            parent_id=collection_id,
            manifest=artifact_obj,
            overwrite=overwrite,
            version="stage",
            context=context,
        )
        
        # Now get the app_id from the artifact alias
        app_id = artifact["alias"]
        
        # Update the artifact object with the correct app_id and URLs
        artifact_obj["id"] = app_id
        if template and template != "raw_html":
            public_url = f"{self.public_base_url}/{workspace_info.id}/artifacts/{app_id}/files/{entry_point}"
            artifact_obj.update(
                {
                    "local_url": f"{self.local_base_url}/{workspace_info.id}/artifacts/{app_id}/files/{entry_point}",
                    "public_url": public_url,
                }
            )
        elif template == "raw_html":
            public_url = f"{self.public_base_url}/{workspace_info.id}/artifacts/{app_id}/files/{entry_point}"
            artifact_obj.update(
                {
                    "local_url": f"{self.local_base_url}/{workspace_info.id}/artifacts/{app_id}/files/{entry_point}",
                    "public_url": public_url,
                }
            )
        
        # Update the artifact with the correct app_id and URLs
        await self.artifact_manager.edit(
            artifact["id"],
            stage=True,
            manifest=artifact_obj,
            context=context,
        )

        # Upload files for template-based apps and raw HTML
        if template and template != "raw_html":
            # Upload the main source file
            put_url = await self.artifact_manager.put_file(
                artifact["id"], file_path=config["entry_point"], use_proxy=False, context=context
            )
            async with httpx.AsyncClient() as client:
                response = await client.put(put_url, data=source)
                assert (
                    response.status_code == 200
                ), f"Failed to upload {config['entry_point']}"
        elif template == "raw_html":
            # Upload the raw HTML content
            put_url = await self.artifact_manager.put_file(
                artifact["id"], file_path=entry_point, use_proxy=False, context=context
            )
            async with httpx.AsyncClient() as client:
                response = await client.put(put_url, data=source)
                assert response.status_code == 200, f"Failed to upload {entry_point}"

        if not stage:
            # Commit the artifact if stage is not enabled
            commit_version = version if version else "v1"
            await self.commit_app(
                app_id,
                timeout=timeout,
                version=commit_version,
                context=context,
            )
            # After commit, read the updated artifact to get the collected services
            updated_artifact_info = await self.artifact_manager.read(
                app_id, version=commit_version, context=context
            )
            return updated_artifact_info.get("manifest", artifact_obj)
        return artifact_obj

    @schema_method
    async def edit_file(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to edit. This is typically the alias of the application."
        ),
        file_path: str = Field(
            ...,
            description="The path of the file to edit. Use forward slashes for path separators (e.g., 'src/main.js', 'index.html')."
        ),
        file_content: str = Field(
            ...,
            description="The new content for the file. This will completely replace the existing file content."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """Add a file to the installed application."""
        put_url = await self.artifact_manager.put_file(
            app_id, file_path=file_path, use_proxy=False, context=context
        )
        response = httpx.put(put_url, data=file_content)
        assert response.status_code == 200, f"Failed to upload {file_path} to {app_id}"

    @schema_method
    async def remove_file(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to modify. This is typically the alias of the application."
        ),
        file_path: str = Field(
            ...,
            description="The path of the file to remove from the application. Use forward slashes for path separators (e.g., 'src/main.js', 'index.html')."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """Remove a file from the installed application."""
        await self.artifact_manager.remove_file(
            app_id, file_path=file_path, context=context
        )

    @schema_method
    async def list_files(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to inspect. This is typically the alias of the application."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> List[dict]:
        """List files of an installed application."""
        return await self.artifact_manager.list_files(
            app_id, context=context
        )

    @schema_method
    async def commit_app(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to commit. This is typically the alias of the application."
        ),
        timeout: int = Field(
            30,
            description="Maximum time to wait for commit completion in seconds. Increase for complex apps that take longer to verify."
        ),
        version: str = Field(
            None,
            description="Version identifier for the committed app. If not provided, uses default versioning."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """Finalize the edits to the application by committing the artifact."""
        try:
            # Read the manifest to check if it's a daemon app
            artifact_info = await self.artifact_manager.read(
                app_id, version="stage", context=context
            )
            manifest = artifact_info.get("manifest", {})
            manifest = ApplicationManifest.model_validate(manifest)

            # Don't use timeout during verification, except for daemon apps
            # which have special handling
            startup_config = manifest.startup_config or {}
            verification_timeout = (
                None
                if not manifest.daemon
                else startup_config.get("stop_after_inactive")
            )

            info = await self.start(
                app_id,
                timeout=timeout,
                wait_for_service="default",
                version="stage",
                stop_after_inactive=verification_timeout,
                context=context,
            )
            await self.stop(info["id"], context=context)

            # After verification, read the updated manifest that includes collected services
            updated_artifact_info = await self.artifact_manager.read(
                app_id, version="stage", context=context
            )

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
            app_id, version=version, context=context
        )
        # After commit, read the updated artifact to get the collected services
        updated_artifact_info = await self.artifact_manager.read(
            app_id, version=version, context=context
        )
        return updated_artifact_info.get("manifest", {})



    @schema_method
    async def uninstall(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to uninstall. This is typically the alias of the application."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> None:
        """Uninstall an application by removing its artifact."""
        # Check if the artifact is in stage mode
        try:
            artifact_info = await self.artifact_manager.read(app_id, version="stage", context=context)
            # If we can read the stage version, it means the artifact is in stage mode
            # Use discard instead of delete to revert to the last committed state
            await self.artifact_manager.discard(app_id, context=context)
        except Exception:
            # If we can't read the stage version, the artifact is committed
            # Use delete as usual
            await self.artifact_manager.delete(app_id, context=context)

    @schema_method
    async def launch(
        self,
        source: str = Field(
            ...,
            description="The source code of the application, URL to fetch the source, or application configuration. Can be raw HTML content, ImJoy/Hypha plugin code, or a URL to download the source from."
        ),
        timeout: float = Field(
            60,
            description="Maximum time to wait for launch completion in seconds. Increase for complex apps that take longer to start."
        ),
        config: Optional[Dict[str, Any]] = Field(
            None,
            description="Application configuration dictionary containing app metadata and settings. Can include startup_config with default startup parameters."
        ),
        overwrite: bool = Field(
            False,
            description="Whether to overwrite existing app with same name. Set to True to replace existing installations."
        ),
        wait_for_service: str = Field(
            None,
            description="Name of the service to wait for before considering the app started. If not provided, waits for default service."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> dict:
        """Install and start a server app instance in stage mode."""
        # Install app in stage mode
        app_info = await self.install(
            source,
            config=config,
            overwrite=overwrite,
            stage=True,
            context=context,
        )
        app_id = app_info["id"]
        # Start the app in stage mode
        return await self.start(
            app_id,
            timeout=timeout,
            wait_for_service=wait_for_service,
            stage=True,
            context=context,
        )

    @schema_method
    async def start(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to start. This is typically the alias of the application."
        ),
        client_id: str = Field(
            None,
            description="Optional client ID for the session. If not provided, a random ID will be generated."
        ),
        timeout: float = Field(
            None,
            description="Maximum time to wait for start completion in seconds. If not provided, uses default timeout from app configuration."
        ),
        version: str = Field(
            None,
            description="Version of the application to start. If not provided, uses the latest version."
        ),
        wait_for_service: Union[str, bool] = Field(
            None,
            description="Name of the service to wait for before considering the app started. If True, waits for 'default' service. If not provided, uses app configuration."
        ),
        stop_after_inactive: Optional[int] = Field(
            None,
            description="Number of seconds to wait before stopping the app due to inactivity. If not provided, uses app configuration."
        ),
        stage: bool = Field(
            False,
            description="Whether to start the app from stage mode. If True, starts from the staged version."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
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
            # Add "_rlb" suffix to enable load balancing metrics for app clients
            # This allows the system to track load only for clients that may have multiple instances
            client_id = random_id(readable=True) + "_rlb"

        # If stage is True, use stage version, otherwise use provided version
        read_version = "stage" if stage else version
        artifact_info = await self.artifact_manager.read(
            app_id, version=read_version, context=context
        )
        manifest = artifact_info.get("manifest", {})
        manifest = ApplicationManifest.model_validate(manifest)

        # Apply default startup config from manifest if not explicitly provided
        startup_config = manifest.startup_config or {}
        if timeout is None and "timeout" in startup_config:
            timeout = startup_config["timeout"]
        else:
            timeout = 120
        if wait_for_service is None and "wait_for_service" in startup_config:
            wait_for_service = startup_config["wait_for_service"]
        # Only apply startup_config if stop_after_inactive is None (not explicitly set)
        # Do not override explicitly passed values (including verification timeout)
        if stop_after_inactive is None and "stop_after_inactive" in startup_config:
            stop_after_inactive = startup_config["stop_after_inactive"]

        if manifest.singleton:
            # check if the app is already running
            for session_info in self._sessions.values():
                if session_info["app_id"] == app_id:
                    # For singleton apps, return the existing session instead of raising an error
                    return session_info
        if manifest.daemon and stop_after_inactive and stop_after_inactive > 0:
            raise ValueError("Daemon apps should not have stop_after_inactive set.")
        if stop_after_inactive is None:
            stop_after_inactive = (
                600
                if startup_config.get("stop_after_inactive") is None
                else startup_config.get("stop_after_inactive")
            )
        entry_point = manifest.entry_point
        assert entry_point, f"Entry point not found for app {app_id}."
        if not entry_point.startswith("http"):
            entry_point = f"{self.local_base_url}/{workspace}/artifacts/{app_id}/files/{entry_point}"
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
            f"{self.public_base_url}/{workspace}/artifacts/{app_id}/files/{entry_point}?"
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
            "source_hash": manifest.source_hash,  # Store source hash for singleton checking
            "name": manifest.name,  # Add app name for UI display
            "description": manifest.description,  # Add app description for UI display
        }

        await runner.start(
            url=local_url,
            session_id=full_client_id,
            metadata=metadata,
        )
        self._sessions[full_client_id] = metadata

        # test activity tracker
        tracker = self.store.get_activity_tracker()
        if (
            not manifest.daemon
            and stop_after_inactive is not None
            and stop_after_inactive > 0
        ):

            async def _stop_after_inactive():
                if full_client_id in self._sessions:
                    await self._stop(full_client_id, raise_exception=False)
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
            "client_id": client_id,
            "config": {},
        }

        def service_added(info: dict):
            logger.info(f"Service added: {info}")
            if info["id"].startswith(full_client_id + ":"):
                sinfo = ServiceInfo.model_validate(info)
                collected_services.append(sinfo)
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

            # Give some time for additional services to be registered (e.g., from setup() method)
            # This is particularly important during verification/installation to collect all services
            await asyncio.sleep(2.0)

            # save the services
            manifest.name = manifest.name or app_info.get("name", "Untitled App")
            manifest.description = manifest.description or app_info.get(
                "description", ""
            )
            
            # Replace client ID with * in service IDs for manifest storage
            manifest_services = []
            for svc in collected_services:
                manifest_svc = svc.model_copy()
                service_id_parts = manifest_svc.id.split(":")
                if len(service_id_parts) >= 2:
                    # Replace client ID with * (workspace/client_id -> workspace/*)
                    workspace_client = service_id_parts[0]
                    workspace_part = workspace_client.rsplit("/", 1)[0]  # Get workspace part
                    service_name = ":".join(service_id_parts[1:])  # Get service name part
                    manifest_svc.id = f"{workspace_part}/*:{service_name}"
                manifest_services.append(manifest_svc)
            
            manifest.services = manifest_services
            manifest = ApplicationManifest.model_validate(
                manifest.model_dump(mode="json")
            )
            await self.artifact_manager.edit(
                app_id,
                version=version,
                stage=stage,
                manifest=manifest.model_dump(mode="json"),
                context=context,
            )

        except asyncio.TimeoutError:
            try:
                logs = await runner.logs(full_client_id)
            except Exception:
                logs = "No logs available (browser session not found)"
            try:
                await runner.stop(full_client_id)
            except Exception:
                pass  # Session might not exist or already stopped
            raise Exception(
                f"Failed to start the app: {workspace}/{app_id}, timeout reached ({timeout}s), browser logs:\n{logs}"
            )
        except Exception as exp:
            try:
                logs = await runner.logs(full_client_id)
            except Exception:
                logs = "No logs available (browser session not found)"
            try:
                await runner.stop(full_client_id)
            except Exception:
                pass  # Session might not exist or already stopped
            raise Exception(
                f"Failed to start the app: {workspace}/{app_id}, error: {exp}, browser logs:\n{logs}"
            ) from exp
        finally:
            self.event_bus.off_local("service_added", service_added)

        if wait_for_service:
            app_info["service_id"] = (
                full_client_id + ":" + wait_for_service + "@" + app_id
            )
        app_info["services"] = [
            svc.model_dump(mode="json") for svc in collected_services
        ]
        if full_client_id in self._sessions:
            self._sessions[full_client_id]["services"] = app_info["services"]
            # Update session metadata with final app name and description
            self._sessions[full_client_id]["name"] = manifest.name
            self._sessions[full_client_id]["description"] = manifest.description
        return app_info

    @schema_method
    async def stop(
        self,
        session_id: str = Field(
            ...,
            description="The session ID of the running application instance to stop. This is typically in the format 'workspace/client_id'."
        ),
        raise_exception: bool = Field(
            True,
            description="Whether to raise an exception if the session is not found. Set to False to ignore missing sessions."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
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

    @schema_method
    async def logs(
        self,
        session_id: str = Field(
            ...,
            description="The session ID of the running application instance. This is typically in the format 'workspace/client_id'."
        ),
        type: str = Field(
            None,
            description="Type of logs to retrieve: 'log', 'error', or None for all types."
        ),
        offset: int = Field(
            0,
            description="Starting offset for log entries. Use for pagination."
        ),
        limit: Optional[int] = Field(
            None,
            description="Maximum number of log entries to return. If not provided, returns all available logs."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get server app instance logs."""
        user_info = UserInfo.model_validate(context["user"])
        workspace = context["ws"]
        if not user_info.check_permission(workspace, UserPermission.read):
            raise Exception(
                f"User {user_info.id} does not have permission"
                f" to get log for app {session_id} in workspace {workspace}."
            )
        if session_id in self._sessions:
            return await self._sessions[session_id]["_runner"].logs(
                session_id, type=type, offset=offset, limit=limit
            )
        else:
            raise Exception(f"Server app instance not found: {session_id}")


    @schema_method
    async def list_running(
        self,
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> List[str]:
        """List the running sessions for the current workspace."""
        workspace = context["ws"]
        return [
            {k: v for k, v in session_info.items() if not k.startswith("_")}
            for session_id, session_info in self._sessions.items()
            if session_id.startswith(f"{workspace}/")
        ]

    @schema_method
    async def list_apps(
        self,
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ):
        """List applications in the workspace."""
        try:
            ws = context["ws"]
            apps = await self.artifact_manager.list_children(
                f"{ws}/applications", filters={"type": "application"}, context=context
            )
            return [app["manifest"] for app in apps]
        except KeyError:
            return []
        except Exception as exp:
            raise Exception(f"Failed to list apps: {exp}") from exp

    @schema_method
    async def get_app_info(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to inspect. This is typically the alias of the application."
        ),
        version: str = Field(
            None,
            description="Version of the application to inspect. If not provided, uses the latest version."
        ),
        stage: bool = Field(
            False,
            description="Whether to inspect the staging version of the application. Set to True to inspect the staged version."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> dict:
        """Get detailed information about an installed application.

        This method returns comprehensive information about an application:
        1. Validates permissions and application existence
        2. Retrieves the application manifest and metadata
        3. Returns detailed configuration and service information

        Returns a dictionary containing:
        - manifest: The application manifest with all metadata
        - id: The application ID
        - name: The application name
        - description: The application description
        - version: The current version
        - type: The application type
        - entry_point: The main entry point file
        - services: List of services provided by the application
        - files: List of files in the application
        - config: Application configuration
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
        """
        try:
            artifact_info = await self.artifact_manager.read(
                app_id, version=version, stage=stage, context=context
            )
            return artifact_info
        except Exception as exp:
            raise Exception(f"Failed to get app info for {app_id}: {exp}") from exp

    @schema_method
    async def get_file_content(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application containing the file."
        ),
        file_path: str = Field(
            ...,
            description="The path of the file to read. Use forward slashes for path separators (e.g., 'src/main.js', 'index.html')."
        ),
        format: str = Field(
            "text",
            description="Format to return the content in: 'text', 'json', or 'binary'. Binary returns base64 encoded content."
        ),
        version: str = Field(
            None,
            description="Version of the application to inspect. If not provided, uses the latest version."
        ),
        stage: bool = Field(
            False,
            description="Whether to read the file from the staged version of the application. Set to True to read from the staged version."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> Union[str, dict, bytes]:
        """Get the content of a specific file from an installed application.

        This method retrieves the content of a file stored in the application:
        1. Validates permissions and application existence
        2. Retrieves the file content from the artifact storage
        3. Returns the file content in the specified format

        Returns:
            - str: File content as text when format='text'
            - dict: Parsed JSON content when format='json'
            - str: Base64 encoded content when format='binary'
        """
        try:
            get_url = await self.artifact_manager.get_file(
                app_id, file_path=file_path, version=version, stage=stage, context=context
            )
            async with httpx.AsyncClient() as client:
                response = await client.get(get_url)
                if response.status_code == 200:
                    if format == "text":
                        return response.text
                    elif format == "json":
                        return response.json()
                    elif format == "binary":
                        import base64
                        return base64.b64encode(response.content).decode()
                    else:
                        raise ValueError(f"Invalid format '{format}'. Must be 'text', 'json' or 'binary'")
                else:
                    raise Exception(f"Failed to retrieve file {file_path}: HTTP {response.status_code}")
        except Exception as exp:
            raise Exception(f"Failed to get file content for {app_id}/{file_path}: {exp}") from exp

    @schema_method
    async def validate_app_config(
        self,
        config: Dict[str, Any] = Field(
            ...,
            description="Application configuration dictionary to validate."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> dict:
        """Validate an application configuration dictionary.

        This method checks if an application configuration is valid:
        1. Validates required fields are present
        2. Checks field types and formats
        3. Validates application type and entry point
        4. Checks for common configuration errors

        Returns a dictionary containing:
        - valid: Boolean indicating if the configuration is valid
        - errors: List of validation errors (if any)
        - warnings: List of validation warnings (if any)
        - suggestions: List of suggestions for improvement (if any)
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # Check required fields
            required_fields = ["name", "type", "version"]
            for field in required_fields:
                if field not in config:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
            
            # Check field types
            if "name" in config and not isinstance(config["name"], str):
                validation_result["errors"].append("Field 'name' must be a string")
                validation_result["valid"] = False
            
            if "type" in config and config["type"] not in ["window", "web-worker", "web-python"]:
                validation_result["warnings"].append(f"Unusual application type: {config['type']}")
            
            if "version" in config and not isinstance(config["version"], str):
                validation_result["errors"].append("Field 'version' must be a string")
                validation_result["valid"] = False
            
            # Check entry point
            if "entry_point" in config:
                entry_point = config["entry_point"]
                if not entry_point.endswith((".html", ".js", ".py")):
                    validation_result["warnings"].append(f"Entry point '{entry_point}' has unusual extension")
            
            # Suggestions
            if "description" not in config:
                validation_result["suggestions"].append("Consider adding a description for better documentation")
            
            if "tags" not in config:
                validation_result["suggestions"].append("Consider adding tags for better discoverability")
            
            return validation_result
            
        except Exception as exp:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {exp}")
            return validation_result

    @schema_method
    async def edit_app(
        self,
        app_id: str = Field(
            ...,
            description="The unique identifier of the application to edit."
        ),
        manifest: Optional[Dict[str, Any]] = Field(
            None,
            description="Updated manifest fields to merge with the existing manifest. Only provided fields will be updated."
        ),
        config: Optional[Dict[str, Any]] = Field(
            None,
            description="Updated config fields to merge with the existing config. Only provided fields will be updated."
        ),
        disabled: Optional[bool] = Field(
            None,
            description="Whether to disable the application. Set to True to disable, False to enable. If not provided, disabled status is not changed."
        ),
        context: Optional[dict] = Field(
            None,
            description="Additional context information including user and workspace details. Usually provided automatically by the system."
        ),
    ) -> None:
        """Edit an application's manifest, config, and/or disabled status.

        This method allows you to update various aspects of an application:
        1. Retrieves the current application manifest
        2. Merges provided manifest updates with existing manifest
        3. Merges provided config updates with existing config
        4. Updates the disabled status if provided
        5. Saves the updated manifest

        The method performs selective updates - only the fields you provide will be changed.
        Existing fields not mentioned in the updates will remain unchanged.

        Note: This will put the application in staging mode and must be committed to take effect.
        Use commit_app() after editing to make changes permanent.
        """
        try:
            # Get current app info
            app_info = await self.get_app_info(app_id, context=context)
            current_manifest = app_info["manifest"]
            
            # Start with current manifest
            updated_manifest = current_manifest.copy() if current_manifest else {}
            
            # Update manifest fields if provided
            if manifest:
                updated_manifest.update(manifest)
            
            # Handle config updates
            if config is not None or disabled is not None:
                # Ensure config exists in manifest
                if "config" not in updated_manifest or updated_manifest["config"] is None:
                    updated_manifest["config"] = {}
                
                # Update config fields if provided
                if config:
                    updated_manifest["config"].update(config)
                
                # Update disabled status if provided
                if disabled is not None:
                    updated_manifest["config"]["disabled"] = disabled

            # Update the manifest with all changes
            await self.artifact_manager.edit(
                app_id,
                manifest=updated_manifest,
                stage=True,
                context=context
            )
            
        except Exception as exp:
            raise Exception(f"Failed to edit app '{app_id}': {exp}") from exp

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
            "logs": self.logs,
            "edit_file": self.edit_file,
            "remove_file": self.remove_file,
            "list_files": self.list_files,
            "commit_app": self.commit_app,
            "get_app_info": self.get_app_info,
            "get_file_content": self.get_file_content,
            "validate_app_config": self.validate_app_config,
            "edit_app": self.edit_app,
        }
