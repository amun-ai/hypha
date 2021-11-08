"""Provide a plugin interface."""
import asyncio
import logging
import sys
import traceback

import shortuuid
from imjoy_rpc.rpc import RPC
from imjoy_rpc.utils import ContextLocal, dotdict

from hypha.utils import EventBus

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("dynamic-plugin")
logger.setLevel(logging.INFO)


class DynamicPlugin:
    """Represent a dynamic plugin."""

    # pylint: disable=too-many-instance-attributes
    _all_plugins = {}
    _event_busses = {}

    @staticmethod
    def get_plugin_by_session_id(session_id: str) -> "DynamicPlugin":
        """Get a plugin by its session id."""
        return DynamicPlugin._all_plugins.get(session_id)

    @staticmethod
    def get_plugin_by_id(plugin_id: str) -> "DynamicPlugin":
        """Get a plugin by its session id."""
        filtered = [p for p in DynamicPlugin._all_plugins.values() if p.id == plugin_id]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            logger.warning("Found multiple plugins with the same id: %s", plugin_id)
            return filtered[0]
        return None

    @staticmethod
    def remove_plugin(plugin):
        """Remove a plugin."""
        DynamicPlugin._all_plugins = {
            key: val for key, val in DynamicPlugin._all_plugins.items() if val != plugin
        }
        if plugin.id in DynamicPlugin._event_busses:
            DynamicPlugin._event_busses[plugin.id].emit("disconnected", plugin)
            del DynamicPlugin._event_busses[plugin.id]

    @staticmethod
    def create_plugin_event_bus(plugin_id) -> EventBus:
        """Remove a plugin."""
        bus = EventBus()
        DynamicPlugin._event_busses[plugin_id] = bus
        return bus

    @staticmethod
    def add_plugin(plugin):
        """Add plugin."""
        DynamicPlugin._all_plugins[plugin.session_id] = plugin
        if plugin.id in DynamicPlugin._event_busses:
            DynamicPlugin._event_busses[plugin.id].emit("connected", plugin)

    @staticmethod
    def plugin_failed(config):
        """Mark plugin as failed."""
        if "id" in config and config["id"] in DynamicPlugin._event_busses:
            DynamicPlugin._event_busses[config["id"]].emit(
                "failed", config.get("detail")
            )

    def __init__(
        self,
        config: dict,
        interface: dict,
        codecs: dict,
        connection,
        workspace,
        user_info,
        event_bus,
    ):
        """Set up instance."""
        self.loop = asyncio.get_event_loop()
        self.config = dotdict(config)
        self._codecs = codecs
        assert self.config.workspace == workspace.name
        self.workspace = workspace
        self.user_info = user_info
        self.event_bus = event_bus
        self.id = self.config.id or shortuuid.uuid()  # pylint: disable=invalid-name
        self.name = self.config.name
        self.source_hash = self.config.source_hash
        self.initializing = False
        self._disconnected = True
        self._log_history = []
        self.connection = connection
        self.api = None
        self.running = False
        self.terminating = False
        self._api_fut = asyncio.Future()
        self._rpc = None
        self._status = "not-initialized"
        self._terminate_callbacks = []
        self.session_id = self.connection.get_session_id()
        DynamicPlugin.add_plugin(self)
        # Note: we don't need to bind the interface
        # to the plugin as we do in the js version
        # We will use context variables `current_plugin`
        # to obtain the current plugin
        self._initial_interface = dotdict(interface)
        self._initial_interface._intf = True
        self._initial_interface.config = self.config.copy()
        self._initial_interface.exit = self.terminate
        if "token" in self._initial_interface.config:
            del self._initial_interface.config["token"]
        self.initialize_if_needed(self.connection, self.config)

        def initialized(data):
            """Handle initialized message."""
            if "error" in data:
                self.error(data["error"])
                logger.error("Plugin failed to initialize: %s", data["error"])
                raise Exception(data["error"])

            asyncio.ensure_future(self._setup_rpc(connection, data["config"]))

        self.connection.on("initialized", initialized)
        self.connection.connect()

    def is_singleton(self):
        """Check if the plugin is singleton."""
        return "single-instance" in self.config.get("flags", [])

    def get_status(self):
        """Check if the plugin is singleton."""
        return self._status

    def set_status(self, status: str):
        """Set the readiness of the plugin."""
        if not self.api and status == "ready":
            raise ValueError("Failed to mark plugin as ready")
        self._status = status

    def register_exit_callback(self, exit_callback):
        """Set the exit callback function."""
        self._terminate_callbacks.append(exit_callback)

    def dispose_object(self, obj):
        """Dispose object in RPC store."""
        store = self._rpc._object_store  # pylint: disable=protected-access
        found = False
        for object_id, object_instance in list(store.items()):
            if object_instance == obj:
                del store[object_id]
                found = True
        if not found:
            raise KeyError("Object not found in the store")

    async def get_api(self):
        """Get the plugin api."""
        return await self._api_fut

    async def _setup_rpc(self, connection, plugin_config):
        """Set up rpc."""
        self.initializing = True
        logger.info("Setting up imjoy-rpc for %s", plugin_config["name"])
        _rpc_context = ContextLocal()
        _rpc_context.api = self._initial_interface
        _rpc_context.default_config = {}
        self._rpc = RPC(connection, _rpc_context, codecs=self._codecs)

        self._register_rpc_events()
        self._rpc.set_interface(self._initial_interface)
        await self._send_interface()
        self._allow_execution = plugin_config.get("allow_execution")
        if self._allow_execution:
            await self._execute_plugin()

        self.config.passive = self.config.passive or plugin_config.get("passive")
        if self.config.passive:

            async def func(*args):
                pass

            self.api = dotdict(
                passive=True, _rintf=True, setup=func, on=func, off=func, emit=func
            )
        else:
            self.api = await self._request_remote()

        self.api["config"] = dotdict(
            id=self.id,
            name=self.config.name,
            namespace=self.config.namespace,
            type=self.config.type,
            workspace=self.config.workspace,
            tag=self.config.tag,
        )

        self._disconnected = False
        self.initializing = False
        logger.info(
            "Plugin loaded successfully (workspace=%s, "
            "name=%s, description=%s, api=%s)",
            self.config.workspace,
            self.name,
            self.config.description,
            list(self.api),
        )
        self._status = "ready"
        if "exit" not in self.api:
            self.api.exit = self.terminate
        self._api_fut.set_result(self.api)

    def error(self, *args):
        """Log an error."""
        self._log_history.append({"type": "error", "value": args})
        logger.error("Error in Plugin %s: $%s", self.id, args)

    def log(self, *args):
        """Log."""
        if isinstance(args[0], dict):
            self._log_history.append(args[0])
            logger.info("Plugin $%s:%s", self.id, args[0])
        else:
            msg = " ".join(map(str, args))
            self._log_history.append({"type": "info", "value": msg})
            logger.info("Plugin $%s: $%s", self.id, msg)

    def _set_disconnected(self):
        """Set disconnected state."""
        self._disconnected = True
        self.running = False
        self.initializing = False
        self.terminating = False
        self._status = "disconnected"

    def is_disconnected(self):
        """Check if plugin is disconnected."""
        return self._disconnected

    def _register_rpc_events(self):
        """Register rpc events."""

        def disconnected(details):
            """Handle disconnected."""
            if details:
                if "error" in details:
                    self.error(details["message"])
                if "info" in details:
                    self.log(details.info)
            self._set_disconnected()

        self._rpc.on("disconnected", disconnected)

        def remote_ready(_):
            """Handle remote ready."""
            api = self._rpc.get_remote()
            # this make sure if reconnect, setup will be called again
            if api and "setup" in api:
                asyncio.ensure_future(api.setup())

        self._rpc.on("remoteReady", remote_ready)

        def remote_idle():
            """Handle remote idle."""
            self.running = False

        self._rpc.on("remoteIdle", remote_idle)

        def remote_busy():
            """Handle remote busy."""
            self.running = True

        self._rpc.on("remoteBusy", remote_busy)

    async def _execute_plugin(self):
        """Execute plugin."""
        # pylint: disable=no-self-use
        logger.warning("Skipping plugin execution.")

    def _send_interface(self):
        """Send the interface."""
        fut = self.loop.create_future()

        def interface_set_as_remote(result):
            """Set interface as remote."""
            fut.set_result(result)

        # pylint: disable=protected-access
        self._rpc._connection.once("interfaceSetAsRemote", interface_set_as_remote)
        self._rpc.send_interface()
        return fut

    def _request_remote(self):
        """Request remote."""
        fut = self.loop.create_future()

        def remote_ready(result):
            """Set remote ready."""
            try:
                fut.set_result(self._rpc.get_remote())
            # TODO: this happens when the the plugin is reconnected
            except asyncio.InvalidStateError:
                pass

        self._rpc.once("remoteReady", remote_ready)
        self._rpc.request_remote()
        return fut

    @staticmethod
    def initialize_if_needed(connection, default_config):
        """Initialize if needed."""

        def imjoy_rpc_ready(data):
            """Handle rpc ready message."""
            config = data["config"] or {}
            forwarding_functions = ["close", "on", "off", "emit"]
            type_ = config.get("type") or default_config.get("type")
            if type_ in ["rpc-window", "window"]:
                forwarding_functions = forwarding_functions + [
                    "resize",
                    "show",
                    "hide",
                    "refresh",
                ]

            credential = None
            if config.get("credential_required"):
                if isinstance(config.credential_fields, list):
                    raise Exception(
                        "Please specify the `config.credential_fields` "
                        "as an array of object."
                    )

                if default_config["credential_handler"]:
                    credential = default_config["credential_handler"](
                        config["credential_fields"]
                    )

                else:
                    credential = {}
                    # for k in config['credential_fields']:
                    #     credential[k.id] = prompt(k.label, k.value)
            connection.emit(
                {
                    "type": "initialize",
                    "config": {
                        "name": default_config.get("name"),
                        "type": default_config.get("type"),
                        "allow_execution": True,
                        "enable_service_worker": True,
                        "forwarding_functions": forwarding_functions,
                        "expose_api_globally": True,
                        "credential": credential,
                    },
                    "peer_id": data["peer_id"],
                }
            )

        connection.once("imjoyRPCReady", imjoy_rpc_ready)

    async def terminate(self):
        """Terminate."""
        try:
            if self._rpc:
                if (
                    self.api
                    and self.api.exit
                    and callable(self.api.exit)
                    # pylint: disable=comparison-with-callable
                    and self.api.exit != self.terminate
                ):
                    logger.info(
                        "Terminating plugin %s/%s", self.config.workspace, self.name
                    )
                    self.api.exit()

                for exit_callback in self._terminate_callbacks:
                    try:
                        exit_callback()
                    except Exception:  # pylint: disable=broad-except
                        logger.exception("Error in exit callback")
                self._rpc.disconnect()
        except Exception:  # pylint: disable=broad-except
            logger.error(
                "Error while terminating plugin %s: %s", self.id, traceback.format_exc()
            )
        finally:
            self._set_disconnected()
            # The following needed to be done when terminating a plugin
            # 1. remove the plugin from the user_info,
            #   if user_info becomes empty, remove user_info from
            #   core_interface.all_users
            # 2. unregister services associated with the plugin,
            #   e.g. for ASGI service, we need to make sure the
            #   route is removed from the server
            # 3. remove plugin from its workspace,
            #   if workspace contains no plugin, depending on
            #   whether its a persistent workspace, we need to
            #   clean it up

            # clean up for 1.
            self.user_info.remove_plugin(self)
            # clean up for 2
            services = self.workspace.get_services_by_plugin(self)
            for service in services:
                self.workspace.remove_service(service)
            # clean up for 3
            self.workspace.remove_plugin(self)
            DynamicPlugin.remove_plugin(self)
            self.event_bus.emit("plugin_terminated", self)
            # finally done
            logger.info("Plugin %s terminated.", self.config.name)
