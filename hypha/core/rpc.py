"""Provide the RPC."""
import asyncio
import inspect
import io
import logging
import os
import sys
import threading
import traceback
import uuid
import weakref
from collections import OrderedDict
from functools import reduce

from imjoy_rpc.utils import (
    FuturePromise,
    MessageEmitter,
    ReferenceStore,
    dotdict,
    format_traceback,
)

API_VERSION = "0.2.3"
ALLOWED_MAGIC_METHODS = ["__enter__", "__exit__"]
IO_METHODS = [
    "fileno",
    "seek",
    "truncate",
    "detach",
    "write",
    "read",
    "read1",
    "readall",
    "close",
    "closed",
    "__enter__",
    "__exit__",
    "flush",
    "isatty",
    "__iter__",
    "__next__",
    "readable",
    "readline",
    "readlines",
    "seekable",
    "tell",
    "writable",
    "writelines",
]

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("RPC")


def index_object(obj, ids):
    """Index an object."""
    if isinstance(ids, str):
        return index_object(obj, ids.split("."))
    elif len(ids) == 0:
        return obj
    else:
        if isinstance(obj, dict):
            _obj = obj[ids[0]]
        elif isinstance(obj, (list, tuple)):
            _obj = obj[int(ids[0])]
        else:
            _obj = getattr(obj, ids[0])
        return index_object(_obj, ids[1:])


class InterfaceList(list):
    """A disposible list."""

    __slots__ = "__rid__"

    def __init__(self, *args, rid=None, **kwargs):
        if rid:
            self.__rid__ = rid
        super().__init__(*args, **kwargs)


class RPC(MessageEmitter):
    """Represent the RPC."""

    def __init__(
        self,
        connection,
        client_id=None,
        default_target_id=None,
        config=None,
        codecs=None,
    ):
        """Set up instance."""
        self.manager_api = {}
        self.services = {}
        self._object_store = {}
        self._method_weakmap = weakref.WeakKeyDictionary()
        self._local_api = None
        self._store = ReferenceStore()
        self._remote_interface = None
        self._codecs = codecs or {}
        self.work_dir = os.getcwd()
        self.abort = threading.Event()
        self.id = client_id
        self.default_target_id = default_target_id

        if config is None:
            config = {}
        self.set_config(config)

        self._remote_logger = dotdict({"info": self._log, "error": self._error})
        super().__init__(self._remote_logger)

        try:
            # FIXME: What exception do we expect?
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        if connection is not None:
            self._connection = connection
            self._setup_handlers(connection)

        self.check_modules()

    def init(self):
        """Initialize the RPC."""
        logger.info("%s initialized", self.config.name)
        self._connection.emit(
            {
                "type": "initialized",
                "config": dict(self.config),
                "peer_id": self._connection.peer_id,
            }
        )

    def reset(self):
        """Reset."""
        self._event_handlers = {}
        self.services = {}
        self._object_store = {}
        self._method_weakmap = weakref.WeakKeyDictionary()
        self._local_api = None
        self._store = ReferenceStore()
        self._remote_interface = None

    def disconnect(self):
        """Disconnect."""
        self._connection.emit({"type": "disconnect"})
        self.reset()
        self._connection.disconnect()

    def default_exit(self):
        """Exit default."""
        logger.info("Terminating plugin: %s", self.id)
        self.abort.set()

    def set_config(self, config):
        """Set config."""
        if config is not None:
            config = dotdict(config)
        else:
            config = dotdict()
        self.id = config.id or self.id or str(uuid.uuid4())
        self.allow_execution = config.allow_execution or False
        self.config = dotdict(
            {
                "allow_execution": self.allow_execution,
                "api_version": API_VERSION,
                "dedicated_thread": True,
                "description": config.description or "[TODO]",
                "id": self.id,
                "lang": "python",
                "name": config.name or "ImJoy RPC Python",
                "type": "rpc-worker",
                "work_dir": self.work_dir,
                "version": config.version or "0.1.0",
            }
        )

    async def register_service(self, service):
        """Register a service."""
        encoded = self.encode_service(service, target_id=self.id)
        if self._remote_interface is None:
            await self.get_service()
        await self._remote_interface.register_service(encoded)

    async def get_service(self, service_id=None, target_id=None):
        """Get a service."""
        if service_id is None:
            self.request_remote(target_id=target_id)
            await self.wait_for("remoteReady", 5.0)
            assert self._remote_interface is not None
            return self._remote_interface

        if self._remote_interface is None:
            await self.get_service()
        interface = await self._remote_interface.get_service(service_id)
        service = self.decode_service(interface, target_id=self.id)
        return service

    def export(self, api, config=None):
        """Export the api interface."""
        # TODO: setup forwarding_functions
        if config is None:
            config = api.get("config")

        if config:
            self.set_config(config)

        # convert and store it in a docdict
        # such that the methods are hashable
        if isinstance(api, dict):
            api = dotdict(
                {
                    a: api[a]
                    for a in api.keys()
                    if not a.startswith("_")
                    or a in ALLOWED_MAGIC_METHODS
                    or a == "_rintf"
                }
            )
        elif inspect.isclass(type(api)):
            api = dotdict(
                {
                    a: getattr(api, a)
                    for a in dir(api)
                    if not a.startswith("_")
                    or a in ALLOWED_MAGIC_METHODS
                    or a == "_rintf"
                }
            )
        else:
            raise Exception("Invalid api export")

        self._local_api = self._encode(api, as_interface=True, target_id=self.id)
        self._fire("interfaceAvailable")
        return self._local_api

    def check_modules(self):
        """Check if all the modules exists."""
        try:
            import numpy as np

            self.NUMPY_MODULE = np
        except ImportError:
            self.NUMPY_MODULE = False
            logger.warning(
                "Failed to import numpy, ndarray encoding/decoding will not work"
            )

    def request_remote(self, target_id=None):
        """Request remote interface."""
        target_id = target_id or self.default_target_id
        assert (
            target_id is not None
        ), "No target_id specified for requesting remote interface."
        self._connection.emit(
            {"type": "getInterface", "target": target_id, "source": self.id}
        )

    def send_interface(self, target_id):
        """Send interface."""
        if self._local_api is None:
            raise Exception("interface is not set.")
        self._connection.emit(
            {
                "type": "setInterface",
                "api": self._local_api,
                "target": target_id,
                "source": self.id,
            }
        )

    def _dispose_object(self, object_id):
        if object_id in self._object_store:
            del self._object_store[object_id]
        else:
            raise Exception("Object (id={}) not found.".format(object_id))

    def dispose_object(self, obj):
        """Dispose object."""
        if not hasattr(obj, "__rid__"):
            raise Exception(
                "Invalid object, it must be a disposable"
                " object with __rid__ attribute."
            )

        def pfunc(resolve, reject):
            """Handle plugin function."""

            def handle_disposed(data):
                """Handle disposed."""
                if "error" in data:
                    reject(data["error"])
                else:
                    resolve(None)

            self._connection.once("disposed", handle_disposed)
            self._connection.emit({"type": "disposeObject", "object_id": obj.__rid__})

        return FuturePromise(pfunc, self._remote_logger)

    def _gen_remote_method(self, source, name, plugin_id=None, target_id=None):
        """Return remote method."""

        def remote_method(*arguments, **kwargs):
            """Run remote method."""
            arguments = list(arguments)
            # encode keywords to a dictionary and pass to the last argument
            if kwargs:
                arguments = arguments + [kwargs]

            def pfunc(resolve, reject):
                encoded_promise = self._encode([resolve, reject], target_id=target_id)
                # store the key id for removing them from the reference store together
                resolve.__promise_pair = encoded_promise[0]["_rid"]
                reject.__promise_pair = encoded_promise[1]["_rid"]

                if name in [
                    "register",
                    "registerService",
                    "register_service",
                    "export",
                    "on",
                ]:
                    args = self._encode(
                        arguments, as_interface=True, target_id=target_id
                    )
                else:
                    args = self._encode(arguments, target_id=target_id)

                call_func = {
                    "type": "method",
                    "source": target_id,
                    "target": source,
                    "name": name,
                    "object_id": plugin_id,
                    "args": args,
                    "with_kwargs": bool(kwargs),
                    "promise": encoded_promise,
                }
                self._connection.emit(call_func)

            return FuturePromise(pfunc, self._remote_logger, self.dispose_object)

        remote_method.__remote_method = True  # pylint: disable=protected-access
        return remote_method

    def _gen_remote_callback(self, source, cid, with_promise, target_id=None):
        """Return remote callback."""
        if with_promise:

            def remote_callback(*arguments, **kwargs):
                # encode keywords to a dictionary and pass to the last argument
                arguments = list(arguments)
                if kwargs:
                    arguments = arguments + [kwargs]

                def pfunc(resolve, reject):
                    encoded_promise = self._encode(
                        [resolve, reject], target_id=target_id
                    )
                    # store the key id
                    # for removing them from the reference store together
                    resolve.__promise_pair = encoded_promise[0]["_rid"]
                    reject.__promise_pair = encoded_promise[1]["_rid"]
                    self._connection.emit(
                        {
                            "type": "callback",
                            "id": cid,
                            "source": target_id,
                            "target": source,
                            # 'object_id'  : self.id,
                            "args": self._encode(arguments, target_id=target_id),
                            "with_kwargs": bool(kwargs),
                            "promise": encoded_promise,
                        }
                    )

                return FuturePromise(pfunc, self._remote_logger, self.dispose_object)

        else:

            def remote_callback(*arguments, **kwargs):
                # encode keywords to a dictionary and pass to the last argument
                arguments = list(arguments)
                if kwargs:
                    arguments = arguments + [kwargs]
                self._connection.emit(
                    {
                        "type": "callback",
                        "id": cid,
                        "source": target_id,
                        "target": source,
                        # 'object_id'  : self.id,
                        "args": self._encode(arguments, target_id=target_id),
                        "with_kwargs": bool(kwargs),
                    }
                )

        return remote_callback

    async def wait_for(self, event, timeout=None):
        """Wait for event."""
        fut = self.loop.create_future()
        self.once(event, fut.set_result)
        return await asyncio.wait_for(fut, timeout)

    def set_remote_interface(self, api):
        """Set remote interface."""
        _remote = self._decode(api, False, target_id=self.id)
        # update existing interface instead of recreating it
        # this will preserve the object reference
        if self._remote_interface:
            self._remote_interface.clear()
            for k in _remote:
                self._remote_interface[k] = _remote[k]
        else:
            self._remote_interface = _remote
        self._fire("remoteReady")

    def _log(self, info):
        self._connection.emit({"type": "log", "message": info})

    def _error(self, error):
        self._connection.emit({"type": "error", "message": error})

    def _call_method(
        self, method, args, kwargs, resolve=None, reject=None, method_name=None
    ):
        try:
            result = method(*args, **kwargs)
            if result is not None and inspect.isawaitable(result):

                async def _wait(result):
                    try:
                        result = await result
                        if resolve is not None:
                            resolve(result)
                        elif result is not None:
                            logger.debug("returned value %s", result)
                    except Exception as err:
                        traceback_error = traceback.format_exc()
                        logger.exception("Error in method %s", err)
                        self._connection.emit(
                            {"type": "error", "message": traceback_error}
                        )
                        if reject is not None:
                            reject(Exception(format_traceback(traceback_error)))

                asyncio.ensure_future(_wait(result))
            else:
                if resolve is not None:
                    resolve(result)
        except Exception as err:
            traceback_error = traceback.format_exc()
            logger.error("Error in method %s: %s", method_name, err)
            self._connection.emit({"type": "error", "message": traceback_error})
            if reject is not None:
                reject(Exception(format_traceback(traceback_error)))

    def _setup_handlers(self, connection):
        connection.on("init", self.init)
        connection.on("execute", self._handle_execute)
        connection.on("method", self._handle_method)
        connection.on("callback", self._handle_callback)
        connection.on("error", self._handle_error)
        connection.on("disconnected", self._disconnected_hanlder)
        connection.on("getInterface", self._get_interface_handler)
        connection.on("setInterface", self._set_interface_handler)
        connection.on("interfaceSetAsRemote", self._remote_set_handler)
        connection.on("disposeObject", self._dispose_object_handler)

    def _dispose_object_handler(self, data):
        try:
            self._dispose_object(data["object_id"])
            self._connection.emit({"type": "disposed"})
        except Exception as e:
            logger.error("failed to dispose object: %s", e)
            self._connection.emit({"type": "disposed", "error": str(e)})

    def _disconnected_hanlder(self, data):
        self._fire("beforeDisconnect")
        self._connection.disconnect()
        self._fire("disconnected", data)

    def _get_interface_handler(self, data):
        if self._local_api is not None:
            self.send_interface(data["source"])
        else:

            def _send_interface():
                self.send_interface(data["source"])

            self.once("interfaceAvailable", _send_interface)

    def _set_interface_handler(self, data):
        self.set_remote_interface(data["api"])
        self._connection.emit(
            {
                "type": "interfaceSetAsRemote",
                "target": data["source"],
                "source": self.id,
            }
        )

    def _remote_set_handler(self, data):
        self._fire("interfaceSetAsRemote", data)

    def _handle_execute(self, data):
        if self.allow_execution:
            try:
                t = data["code"]["type"]
                if t == "script":
                    content = data["code"]["content"]
                    # TODO: fix the imjoy module such that it will
                    # stick to the current context api
                    exec(content)
                elif t == "requirements":
                    pass
                else:
                    raise Exception("unsupported type")
                self._connection.emit({"type": "executed"})
            except Exception as err:
                traceback_error = traceback.format_exc()
                logger.exception("Error during execution: %s", err)
                self._connection.emit({"type": "executed", "error": traceback_error})
        else:
            self._connection.emit(
                {"type": "executed", "error": "execution is not allowed"}
            )
            logger.warning("execution is blocked due to allow_execution=False")

    def _handle_method(self, data):
        reject = None
        try:
            if "promise" in data:
                resolve, reject = self._decode(
                    data["promise"], False, target_id=data["source"]
                )
            _interface = self._object_store[data["object_id"]]
            method = index_object(_interface, data["name"])
            if "promise" in data:
                args = self._decode(data["args"], True, target_id=data["source"])
                if data.get("with_kwargs"):
                    kwargs = args.pop()
                else:
                    kwargs = {}
                # args.append({'id': self.id})

                self._call_method(
                    method, args, kwargs, resolve, reject, method_name=data["name"]
                )
            else:
                args = self._decode(data["args"], True, target_id=data["source"])
                if data.get("with_kwargs"):
                    kwargs = args.pop()
                else:
                    kwargs = {}
                # args.append({'id': self.id})
                self._call_method(method, args, kwargs, method_name=data["name"])
        except Exception as err:
            traceback_error = traceback.format_exc()
            logger.exception("Error during calling method: %s", err)
            self._connection.emit({"type": "error", "message": traceback_error})
            if callable(reject):
                reject(traceback_error)

    def _handle_callback(self, data):
        reject = None
        try:
            if "promise" in data:
                resolve, reject = self._decode(
                    data["promise"], False, target_id=data["source"]
                )
                method = self._store.fetch(data["id"])
                if method is None:
                    raise Exception(
                        "Callback function can only called once, "
                        "if you want to call a function for multiple times, "
                        "please make it as a plugin api function. "
                        "See https://imjoy.io/docs for more details."
                    )
                args = self._decode(data["args"], True, target_id=data["source"])
                if data.get("with_kwargs"):
                    kwargs = args.pop()
                else:
                    kwargs = {}
                self._call_method(
                    method, args, kwargs, resolve, reject, method_name=data["id"]
                )

            else:
                method = self._store.fetch(data["id"])
                if method is None:
                    raise Exception(
                        "Callback function can only called once, "
                        "if you want to call a function for multiple times, "
                        "please make it as a plugin api function. "
                        "See https://imjoy.io/docs for more details."
                    )
                args = self._decode(data["args"], True, target_id=data["source"])
                if data.get("with_kwargs"):
                    kwargs = args.pop()
                else:
                    kwargs = {}
                self._call_method(method, args, kwargs, method_name=data["id"])

        except Exception as err:
            traceback_error = traceback.format_exc()
            logger.exception("error when calling callback function: %s", err)
            self._connection.emit({"type": "error", "message": traceback_error})
            if callable(reject):
                reject(traceback_error)

    def _handle_error(self, detail):
        self._fire("error", detail)

    def encode_service(self, service, target_id):
        """Encode service."""
        encoded = self._encode(service, as_interface=True, target_id=target_id)
        encoded["_rtype"] = "service"
        return encoded

    def decode_service(self, interface, target_id):
        """Decode service."""
        assert interface["_rtype"] == "service"
        del interface["_rtype"]
        decoded = self._decode(interface, with_promise=True, target_id=target_id)
        return decoded
    
    def encode(self, a_object, as_interface=False, object_id=None, target_id=None):
        """Encode object."""
        return self._encode(a_object, as_interface=as_interface, object_id=object_id, target_id=target_id)

    def _encode(self, a_object, as_interface=False, object_id=None, target_id=None):
        """Encode object."""
        if isinstance(a_object, (int, float, bool, str, bytes)) or a_object is None:
            return a_object

        if not as_interface and isinstance(a_object, dict):
            as_interface = a_object.get("_rintf", False)
        as_interface = bool(as_interface)

        if callable(a_object):
            if as_interface:
                if not object_id:
                    raise Exception("object_id is not specified.")
                _intf, _rid = object_id.split(":")
                b_object = {
                    "_rtype": "interface",
                    "_rtarget_id": target_id,
                    "_rintf": _intf,
                    "_rid": _rid,
                }
                self._method_weakmap[a_object] = b_object
            elif a_object in self._method_weakmap:
                b_object = self._method_weakmap[a_object]
            else:
                cid = self._store.put(a_object)
                b_object = {
                    "_rtype": "callback",
                    # Some functions do not have the __name__ attribute
                    # for example when we use functools.partial to create functions
                    "_rname": getattr(a_object, "__name__", cid),
                    "_rtarget_id": target_id,
                    "_rid": cid,
                }
            return b_object

        if isinstance(a_object, tuple):
            a_object = list(a_object)

        if isinstance(a_object, dict):
            a_object = dict(a_object)

        # skip if already encoded
        if isinstance(a_object, dict) and "_rtype" in a_object:
            # escape service object
            if a_object["_rtype"] == "service":
                return a_object
            # make sure the interface functions are encoded
            temp = a_object["_rtype"]
            del a_object["_rtype"]
            b_object = self._encode(
                a_object, as_interface, object_id, target_id=target_id
            )
            b_object["_rtype"] = temp
            return b_object

        isarray = isinstance(a_object, list)
        b_object = None

        encoded_obj = None
        for tp in self._codecs:
            codec = self._codecs[tp]
            if codec.encoder and isinstance(a_object, codec.type):
                # TODO: what if multiple encoders found
                encoded_obj = codec.encoder(a_object)
                if isinstance(encoded_obj, dict) and "_rtype" not in encoded_obj:
                    encoded_obj["_rtype"] = codec.name
                # encode the functions in the interface object
                if isinstance(encoded_obj, dict) and "_rintf" in encoded_obj:
                    temp = encoded_obj["_rtype"]
                    del encoded_obj["_rtype"]
                    encoded_obj = self._encode(encoded_obj, True, target_id=target_id)
                    encoded_obj["_rtype"] = temp
                b_object = encoded_obj
                return b_object

        if self.NUMPY_MODULE and isinstance(
            a_object, (self.NUMPY_MODULE.ndarray, self.NUMPY_MODULE.generic)
        ):
            v_bytes = a_object.tobytes()
            b_object = {
                "_rtype": "ndarray",
                "_rvalue": v_bytes,
                "_rshape": a_object.shape,
                "_rdtype": str(a_object.dtype),
            }

        elif isinstance(a_object, Exception):
            b_object = {"_rtype": "error", "_rvalue": str(a_object)}
        elif isinstance(a_object, memoryview):
            b_object = {"_rtype": "memoryview", "_rvalue": a_object.tobytes()}
        elif isinstance(
            a_object, (io.IOBase, io.TextIOBase, io.BufferedIOBase, io.RawIOBase)
        ):
            b_object = {
                m: getattr(a_object, m) for m in IO_METHODS if hasattr(a_object, m)
            }
            b_object["_rintf"] = True
            b_object = self._encode(b_object, target_id=target_id)

        # NOTE: "typedarray" is not used
        elif isinstance(a_object, OrderedDict):
            b_object = {
                "_rtype": "orderedmap",
                "_rvalue": self._encode(
                    list(a_object), as_interface, target_id=target_id
                ),
            }
        elif isinstance(a_object, set):
            b_object = {
                "_rtype": "set",
                "_rvalue": self._encode(
                    list(a_object), as_interface, target_id=target_id
                ),
            }
        elif isinstance(a_object, (list, dict)):

            keys = range(len(a_object)) if isarray else a_object.keys()
            # encode interfaces
            if as_interface:
                b_object = [] if isarray else {}
                if object_id is None:
                    object_id = str(uuid.uuid4())
                    # Note: object with the same id will be overwritten
                    if object_id in self._object_store:
                        logger.warning(
                            "Overwritting interface object with the same id: %s",
                            object_id,
                        )
                    self._object_store[object_id] = a_object

                has_function = False
                for key in keys:
                    if isinstance(key, str) and (
                        key.startswith("_") and key not in ALLOWED_MAGIC_METHODS
                    ):
                        continue
                    _obj_id = (
                        object_id + "." + str(key)
                        if ":" in object_id
                        else object_id + ":" + str(key)
                    )
                    encoded = self._encode(
                        a_object[key],
                        as_interface,
                        # We need to convert to a string here,
                        # otherwise 0 will not be truthy.
                        _obj_id,
                        target_id=target_id,
                    )
                    if callable(a_object[key]):
                        has_function = True
                    if isarray:
                        b_object.append(encoded)
                    else:
                        b_object[key] = encoded
                # convert list to dict
                if isarray and has_function:
                    b_object = {k: b_object[k] for k in range(len(b_object))}
                    b_object["_rarray"] = True
                    b_object["_rlength"] = len(b_object)
                if not isarray and has_function:
                    b_object["_rintf"] = object_id

                # remove interface when closed
                if "on" in a_object and callable(a_object["on"]):

                    def remove_interface(_):
                        if object_id in self._object_store:
                            del self._object_store[object_id]

                    a_object["on"]("close", remove_interface)
            else:
                b_object = [] if isarray else {}
                for key in keys:
                    if isarray:
                        b_object.append(self._encode(a_object[key], target_id=target_id))
                    else:
                        b_object[key] = self._encode(a_object[key], target_id=target_id)
        else:
            raise Exception(
                "imjoy-rpc: Unsupported data type:"
                f" {type(a_object)}, you can register a custom"
                " codec to encode/decode the object."
            )
        return b_object

    def decode(self, a_object, with_promise, target_id=None):
        """Decode object."""
        return self._decode(a_object, with_promise, target_id=target_id)

    def _decode(self, a_object, with_promise, target_id=None):
        """Decode object."""
        if a_object is None:
            return a_object
        if isinstance(a_object, dict) and "_rtype" in a_object:
            b_object = None
            if a_object["_rtype"] == "service":
                return a_object
            if (
                self._codecs.get(a_object["_rtype"])
                and self._codecs[a_object["_rtype"]].decoder
            ):
                if "_rintf" in a_object:
                    temp = a_object["_rtype"]
                    del a_object["_rtype"]
                    a_object = self._decode(a_object, with_promise, target_id=target_id)
                    a_object["_rtype"] = temp
                b_object = self._codecs[a_object["_rtype"]].decoder(a_object)
            elif a_object["_rtype"] == "callback":
                b_object = self._gen_remote_callback(
                    a_object.get("_rtarget_id"),
                    a_object["_rid"],
                    with_promise,
                    target_id=target_id,
                )
            elif a_object["_rtype"] == "interface":
                b_object = self._gen_remote_method(
                    a_object.get("_rtarget_id"),
                    a_object["_rid"],
                    a_object["_rintf"],
                    target_id=target_id,
                )
            elif a_object["_rtype"] == "ndarray":
                # create build array/tensor if used in the plugin
                try:
                    if isinstance(a_object["_rvalue"], (list, tuple)):
                        a_object["_rvalue"] = reduce(
                            (lambda x, y: x + y), a_object["_rvalue"]
                        )
                    # make sure we have bytes instead of memoryview, e.g. for Pyodide
                    elif isinstance(a_object["_rvalue"], memoryview):
                        a_object["_rvalue"] = a_object["_rvalue"].tobytes()
                    elif not isinstance(a_object["_rvalue"], bytes):
                        raise Exception(
                            "Unsupported data type: " + str(type(a_object["_rvalue"]))
                        )
                    if self.NUMPY_MODULE:
                        b_object = self.NUMPY_MODULE.frombuffer(
                            a_object["_rvalue"], dtype=a_object["_rdtype"]
                        ).reshape(tuple(a_object["_rshape"]))

                    else:
                        b_object = a_object
                        logger.warning(
                            "numpy is not available, failed to decode ndarray"
                        )

                except Exception as exc:
                    logger.debug("Error in converting: %s", exc)
                    b_object = a_object
                    raise exc
            elif a_object["_rtype"] == "memoryview":
                b_object = memoryview(a_object["_rvalue"])
            elif a_object["_rtype"] == "blob":
                if isinstance(a_object["_rvalue"], str):
                    b_object = io.StringIO(a_object["_rvalue"])
                elif isinstance(a_object["_rvalue"], bytes):
                    b_object = io.BytesIO(a_object["_rvalue"])
                else:
                    raise Exception(
                        "Unsupported blob value type: " + str(type(a_object["_rvalue"]))
                    )
            elif a_object["_rtype"] == "typedarray":
                if self.NUMPY_MODULE:
                    b_object = self.NUMPY_MODULE.frombuffer(
                        a_object["_rvalue"], dtype=a_object["_rdtype"]
                    )
                else:
                    b_object = a_object["_rvalue"]
            elif a_object["_rtype"] == "orderedmap":
                b_object = OrderedDict(
                    self._decode(a_object["_rvalue"], with_promise, target_id=target_id)
                )
            elif a_object["_rtype"] == "set":
                b_object = set(
                    self._decode(a_object["_rvalue"], with_promise, target_id=target_id)
                )
            elif a_object["_rtype"] == "error":
                b_object = Exception(a_object["_rvalue"])
            else:
                # make sure all the interface functions are decoded
                if "_rintf" in a_object:
                    temp = a_object["_rtype"]
                    del a_object["_rtype"]
                    a_object = self._decode(a_object, with_promise, target_id=target_id)
                    a_object["_rtype"] = temp
                b_object = a_object
        elif isinstance(a_object, (dict, list, tuple)):
            if isinstance(a_object, tuple):
                a_object = list(a_object)
            isarray = isinstance(a_object, list)
            b_object = [] if isarray else dotdict()
            keys = range(len(a_object)) if isarray else a_object.keys()
            for key in keys:
                val = a_object[key]
                if isarray:
                    b_object.append(self._decode(val, with_promise, target_id=target_id))
                else:
                    b_object[key] = self._decode(val, with_promise, target_id=target_id)
        # make sure we have bytes instead of memoryview, e.g. for Pyodide
        elif isinstance(a_object, memoryview):
            b_object = a_object.tobytes()
        elif isinstance(a_object, bytearray):
            b_object = bytes(a_object)
        else:
            b_object = a_object

        # object id, used for dispose the object
        if isinstance(a_object, dict) and a_object.get("_rintf"):
            if a_object.get("_rarray"):
                b_object = InterfaceList(
                    [b_object[i] for i in range(a_object["_rlength"])]
                )
                b_object.__rid__ = a_object.get("_rintf")
            # make the dict hashable
            if isinstance(b_object, dict):
                if not isinstance(b_object, dotdict):
                    b_object = dotdict(b_object)
                # __rid__ is used for hashing the object for removing it afterwards
                b_object.__rid__ = a_object.get("_rintf")

        return b_object
