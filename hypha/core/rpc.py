"""Provide the RPC."""
import asyncio
import inspect
import io
import logging
import os
import sys
import threading
import traceback
import shortuuid
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


class Timer:
    def __init__(self, timeout, callback, *args, label="timer", **kwargs):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())
        self._args = args
        self._kwrags = kwargs
        self._label = label

    async def _job(self):
        await asyncio.sleep(self._timeout)
        ret = self._callback(*self._args, **self._kwrags)
        if ret is not None and inspect.isawaitable(ret):
            await ret

    def clear(self):
        self._task.cancel()

    def reset(self):
        self._task.cancel()
        self._task = asyncio.ensure_future(self._job())


class RPC(MessageEmitter):
    """Represent the RPC."""

    def __init__(
        self,
        connection,
        client_id=None,
        root_target_id=None,
        default_context=None,
        codecs=None,
        method_timeout=10,
    ):
        """Set up instance."""
        self.manager_api = {}
        self._store = ReferenceStore()
        self._codecs = codecs or {}
        self.work_dir = os.getcwd()
        self.abort = threading.Event()
        self.client_id = client_id
        assert client_id and isinstance(client_id, str)
        self.root_target_id = root_target_id
        self.default_context = default_context or {}
        self._method_annotations = weakref.WeakKeyDictionary()
        self._remote_root_service = None
        self._method_timeout = method_timeout
        self._remote_logger = dotdict({"info": self._log, "error": self._error})
        super().__init__(self._remote_logger)
        self._services = {}
        self._object_store = {
            "services": self._services,
        }
        self.add_service(
            {
                "id": "built-in",
                "type": "built-in",
                "name": "RPC built-in services",
                "config": {"require_context": True},
                "get_service": self.get_local_service,
            }
        )
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

    def reset(self):
        """Reset."""
        self._event_handlers = {}
        self._services = {}
        self._store = ReferenceStore()

    def disconnect(self):
        """Disconnect."""
        self._fire("disconnect")
        self._connection.disconnect()

    async def get_remote_root_service(self, timeout=None):
        if self.root_target_id:
            self._remote_root_service = await self.get_remote_service(
                service_uri=self.root_target_id + ":/", timeout=timeout
            )
            return self._remote_root_service

    def get_all_local_services(self):
        """Get all the local services."""
        return self._services

    def get_local_service(self, service_id):
        assert service_id is not None
        return self._services.get(service_id)

    async def get_remote_service(self, service_uri=None, timeout=None):
        """Get a remote service."""
        if service_uri is None and self.root_target_id:
            service_uri = self.root_target_id
        elif ":" not in service_uri:
            service_uri = self.client_id + ":" + service_uri
        provider, service_id = service_uri.split(":")
        assert provider
        try:
            method = self._generate_remote_method(
                {
                    "_rtarget": provider,
                    "_rmethod": "services.built-in.get_service",
                    "_rpromise": True,
                }
            )
            return await asyncio.wait_for(method(service_id), timeout=timeout)
        except Exception:
            logger.exception("Failed to get remote service")
            raise

    def add_service(self, api, overwrite=False):
        """Add a service (silently without triggering notifications)."""
        # convert and store it in a docdict
        # such that the methods are hashable
        if isinstance(api, dict):
            api = dotdict(
                {
                    a: api[a]
                    for a in api.keys()
                    if not a.startswith("_") or a in ALLOWED_MAGIC_METHODS
                }
            )
        elif inspect.isclass(type(api)):
            api = dotdict(
                {
                    a: getattr(api, a)
                    for a in dir(api)
                    if not a.startswith("_") or a in ALLOWED_MAGIC_METHODS
                }
            )
        else:
            raise Exception("Invalid service object type: {}".format(type(api)))

        if "id" not in api:
            api["id"] = "default"

        if "name" not in api:
            api["name"] = api["id"]

        if "config" not in api:
            api["config"] = {}

        if "type" not in api:
            api["type"] = "generic"

        if not overwrite and api["id"] in self._services:
            raise Exception(
                f"Service already exists: {api['id']}, please specify"
                f" a different id (not `{api['id']}`) or overwrite=True"
            )
        self._services[api["id"]] = api
        return api

    async def register_service(self, api, overwrite=False, notify=True):
        """Register a service."""
        service = self.add_service(api, overwrite=overwrite)
        if notify:
            self._fire("serviceUpdated", {"service_id": service["id"], "api": service})
            await self._notify_service_update()
        return service

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

    def _encode_callback(
        self, name, callback, cid, session_id, clear_after_called=False, timer=None
    ):
        encoded = {
            "_rtype": "method",
            "_rtarget": self.client_id,
            "_rmethod": f"{session_id}.__callbacks__.{cid}.{name}",
            "_rpromise": False,
        }

        def wrapped_callback(*args, **kwargs):
            try:
                callback(*args, **kwargs)
            finally:
                if clear_after_called and session_id in self._object_store:
                    logger.info(
                        "Deleting session %s from %s", session_id, self.client_id
                    )
                    del self._object_store[session_id]
                    # if hasattr(self._connection, "close"):
                    #     self._connection.close()
                if timer:
                    timer.clear()

        return encoded, wrapped_callback

    def _encode_promise(
        self,
        resolve,
        reject,
        session_id,
        clear_after_called=False,
        with_timer=True,
        method_name=None,
    ):
        """Encode a group of callbacks without promise."""
        if session_id not in self._object_store:
            self._object_store[session_id] = {}
        callback_store = self._object_store[session_id]
        if "__callbacks__" not in callback_store:
            callback_store["__callbacks__"] = {}
        store = callback_store["__callbacks__"]
        cid = shortuuid.uuid()
        store[cid] = {}
        encoded = {}

        if with_timer and reject and self._method_timeout:
            timer = Timer(
                self._method_timeout,
                reject,
                f"Method call time out: {method_name}",
                label=method_name,
            )
            encoded["clear_timer"], store[cid]["clear_timer"] = self._encode_callback(
                "clear_timer", timer.clear, cid, session_id, clear_after_called=False
            )
            encoded["reset_timer"], store[cid]["reset_timer"] = self._encode_callback(
                "reset_timer", timer.reset, cid, session_id, clear_after_called=False
            )
            encoded["heatbeat_interval"] = self._method_timeout / 2
        else:
            timer = None

        encoded["resolve"], store[cid]["resolve"] = self._encode_callback(
            "resolve",
            resolve,
            cid,
            session_id,
            clear_after_called=clear_after_called,
            timer=timer,
        )
        encoded["reject"], store[cid]["reject"] = self._encode_callback(
            "reject",
            reject,
            cid,
            session_id,
            clear_after_called=clear_after_called,
            timer=timer,
        )

        return encoded

    def _generate_remote_method(self, encoded_method):
        """Return remote method."""

        target_id = encoded_method["_rtarget"]
        method_id = encoded_method["_rmethod"]
        with_promise = encoded_method.get("_rpromise", False)
        # do not clear if scope == "session"
        def remote_method(*arguments, **kwargs):
            """Run remote method."""
            arguments = list(arguments)
            # encode keywords to a dictionary and pass to the last argument
            if kwargs:
                arguments = arguments + [kwargs]

            def pfunc(resolve, reject):
                local_session_id = shortuuid.uuid()
                args = self._encode(
                    arguments,
                    session_id=local_session_id,
                )

                call_func = {
                    "type": "method",
                    "from": self.client_id,
                    "to": target_id,
                    "method": method_id,
                    "params": args,
                    "with_kwargs": bool(kwargs),
                }
                if with_promise:
                    call_func["promise"] = self._encode_promise(
                        resolve=resolve,
                        reject=reject,
                        session_id=local_session_id,
                        clear_after_called=True,
                        with_timer=True,
                        method_name=f"{target_id}:{method_id}",
                    )
                self._connection.emit(call_func)

            return FuturePromise(pfunc, self._remote_logger)

        # Generate debugging information for the method
        remote_method.__remote_method__ = (
            encoded_method.copy()
        )  # pylint: disable=protected-access
        return remote_method

    async def wait_for(self, event, query=None, timeout=None):
        """Wait for event."""
        fut = self.loop.create_future()

        def on_event(data):
            if not query:
                fut.set_result(data)
                self.off(event, on_event)
            elif isinstance(query, dict):
                matched = True
                for key in query:
                    if data.get(key) != query[key]:
                        matched = False
                        break
                if matched:
                    fut.set_result(data)
                    self.off(event, on_event)
            elif query == data:
                fut.set_result(data)
                self.off(event, on_event)

        self.on(event, on_event)
        try:
            ret = await asyncio.wait_for(fut, timeout)
        # except timeout error
        except asyncio.exceptions.TimeoutError:
            self.off(event, on_event)
            raise
        return ret

    def set_remote_interface(self, data):
        """Set remote interface."""
        data["api"] = self._decode(data["api"])
        self._fire("remoteReady", data)

    def _log(self, info):
        logger.info("RPC Info: %s", info)

    def _error(self, error):
        logger.error("RPC Error: %s", error)

    def _call_method(
        self,
        method,
        args,
        kwargs,
        resolve=None,
        reject=None,
        method_name=None,
        heatbeat_task=None,
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
                            logger.debug("returned value (%s): %s", method_name, result)
                    except Exception as err:
                        traceback_error = traceback.format_exc()
                        logger.exception("Error in method (%s): %s", method_name, err)
                        if reject is not None:
                            reject(Exception(format_traceback(traceback_error)))
                    finally:
                        if heatbeat_task:
                            heatbeat_task.cancel()

                return asyncio.ensure_future(_wait(result))
            else:
                if resolve is not None:
                    resolve(result)
                if heatbeat_task:
                    heatbeat_task.cancel()
        except Exception as err:
            traceback_error = traceback.format_exc()
            logger.exception("Error in method %s: %s", method_name, err)
            if reject is not None:
                reject(Exception(format_traceback(traceback_error)))
            if heatbeat_task:
                heatbeat_task.cancel()

    def _setup_handlers(self, connection):
        connection.on("method", self._handle_method)
        connection.on("disconnected", self._disconnected_hanlder)

    async def _notify_service_update(self):
        if not self._remote_root_service:
            # try to get the root service
            await self.get_remote_root_service(timeout=5.0)

        if self._remote_root_service:
            await self._remote_root_service.update_client_info(self.get_client_info())

    def get_client_info(self):
        return {
            "id": self.client_id,
            "services": [
                {
                    "id": service["id"],
                    "type": service["type"],
                    "name": service["name"],
                    "config": service["config"],
                }
                for service in self._services.values()
            ],
        }

    def _disconnected_hanlder(self, data):
        self._fire("beforeDisconnect")
        self._connection.disconnect()
        self._fire("disconnected", data)

    def _handle_method(self, data):
        reject = None
        method_task = None
        heatbeat_task = None
        try:
            assert "method" in data and "params" in data
            if "promise" in data:
                promise = self._decode(data["promise"])
                resolve, reject = promise["resolve"], promise["reject"]
                if "reset_timer" in promise and "heatbeat_interval" in promise:

                    async def heatbeat(interval):
                        while True:
                            try:
                                await promise["reset_timer"]()
                            except Exception:  # pylint: disable=broad-except
                                if method_task:
                                    method_task.cancel()
                                break
                            await asyncio.sleep(interval)

                    heatbeat_task = asyncio.ensure_future(
                        heatbeat(promise["heatbeat_interval"])
                    )
                elif "clear_timer" in promise:
                    asyncio.ensure_future(promise["clear_timer"]())
            else:
                resolve, reject = None, None
                # TODO: add dispose handler to the result object
            args = self._decode(data["params"])
            if data.get("with_kwargs"):
                kwargs = args.pop()
            else:
                kwargs = {}
            method_name = f'{data["from"]}:{data["method"]}'
            try:
                method = index_object(self._object_store, data["method"])
            except Exception:
                logger.exception("Failed to find method %s", method_name)
                raise Exception(f"Method not found: {method_name}")
            assert callable(method), f"Invalid method: {method_name}"
            if method in self._method_annotations and self._method_annotations[
                method
            ].get("require_context"):
                self.default_context.update({"client_id": data["from"]})
                kwargs["context"] = self.default_context
            method_task = self._call_method(
                method,
                args,
                kwargs,
                resolve,
                reject,
                method_name=method_name,
                heatbeat_task=heatbeat_task,
            )

        except Exception as err:
            traceback_error = traceback.format_exc()
            logger.exception("Error during calling method: %s", err)
            if callable(reject):
                reject(traceback_error)

    def encode(self, a_object, session_id=None):
        """Encode object."""
        return self._encode(
            a_object,
            session_id=session_id,
        )

    def _encode(
        self,
        a_object,
        service_idx=None,
        object_id=None,
        session_id=None,
    ):
        """Encode object."""
        if isinstance(a_object, (int, float, bool, str, bytes)) or a_object is None:
            return a_object

        if service_idx:
            assert isinstance(service_idx, str)

        if isinstance(a_object, tuple):
            a_object = list(a_object)

        if isinstance(a_object, dict):
            a_object = dict(a_object)

        # skip if already encoded
        if isinstance(a_object, dict) and "_rtype" in a_object:
            # make sure the interface functions are encoded
            temp = a_object["_rtype"]
            del a_object["_rtype"]
            b_object = self._encode(
                a_object,
                service_idx,
                object_id,
                session_id=session_id,
            )
            b_object["_rtype"] = temp
            return b_object

        if callable(a_object):
            # Reuse the remote method
            if hasattr(a_object, "__remote_method__"):
                return a_object.__remote_method__

            if service_idx:
                b_object = {
                    "_rtype": "method",
                    "_rtarget": self.client_id,
                    "_rmethod": "services." + service_idx,
                    "_rpromise": True,
                }
            else:
                if isinstance(object_id, str):
                    object_id = f"{object_id}-{shortuuid.uuid()}"
                else:
                    object_id = f"{shortuuid.uuid()}"
                b_object = {
                    "_rtype": "method",
                    "_rtarget": self.client_id,
                    "_rmethod": f"{session_id}.{object_id}",
                    "_rpromise": True,
                }
                if session_id not in self._object_store:
                    self._object_store[session_id] = {}
                self._object_store[session_id][object_id] = a_object
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
                    encoded_obj = self._encode(
                        encoded_obj,
                        True,
                        session_id=session_id,
                    )
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
            b_object = self._encode(b_object, session_id=session_id)

        # NOTE: "typedarray" is not used
        elif isinstance(a_object, OrderedDict):
            b_object = {
                "_rtype": "orderedmap",
                "_rvalue": self._encode(
                    list(a_object),
                    service_idx,
                    session_id=session_id,
                ),
            }
        elif isinstance(a_object, set):
            b_object = {
                "_rtype": "set",
                "_rvalue": self._encode(
                    list(a_object),
                    service_idx,
                    session_id=session_id,
                ),
            }
        elif isinstance(a_object, (list, dict)):
            is_service = a_object in self._services.values()
            # require_context only applies to the top-level functions
            if is_service and "config" in a_object:
                require_context = bool(a_object["config"].get("require_context"))
            else:
                require_context = False
            keys = range(len(a_object)) if isarray else a_object.keys()
            b_object = [] if isarray else {}
            if is_service:
                service_idx = None
                for sk, sv in self._services.items():
                    if a_object == sv:
                        service_idx = sk
                        break
                assert isinstance(service_idx, str)
            for key in keys:
                if callable(a_object[key]) and require_context:
                    # mark the method as a remote method that requires context
                    self._method_annotations[a_object[key]] = {
                        "require_context": require_context
                    }

                encoded = self._encode(
                    a_object[key],
                    session_id=session_id,
                    service_idx=service_idx + "." + str(key) if service_idx else None,
                    object_id=key,
                )
                if isarray:
                    b_object.append(encoded)
                else:
                    b_object[key] = encoded
        else:
            raise Exception(
                "imjoy-rpc: Unsupported data type:"
                f" {type(a_object)}, you can register a custom"
                " codec to encode/decode the object."
            )
        return b_object

    def decode(self, a_object):
        """Decode object."""
        return self._decode(a_object)

    def _decode(self, a_object):
        """Decode object."""
        if a_object is None:
            return a_object
        if isinstance(a_object, dict) and "_rtype" in a_object:
            b_object = None
            if (
                self._codecs.get(a_object["_rtype"])
                and self._codecs[a_object["_rtype"]].decoder
            ):
                if "_rintf" in a_object:
                    temp = a_object["_rtype"]
                    del a_object["_rtype"]
                    a_object = self._decode(a_object)
                    a_object["_rtype"] = temp
                b_object = self._codecs[a_object["_rtype"]].decoder(a_object)
            elif a_object["_rtype"] == "method":
                b_object = self._generate_remote_method(a_object)
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
                b_object = OrderedDict(self._decode(a_object["_rvalue"]))
            elif a_object["_rtype"] == "set":
                b_object = set(self._decode(a_object["_rvalue"]))
            elif a_object["_rtype"] == "error":
                b_object = Exception(a_object["_rvalue"])
            else:
                # make sure all the interface functions are decoded
                if "_rintf" in a_object:
                    temp = a_object["_rtype"]
                    del a_object["_rtype"]
                    a_object = self._decode(a_object)
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
                    b_object.append(self._decode(val))
                else:
                    b_object[key] = self._decode(val)
        # make sure we have bytes instead of memoryview, e.g. for Pyodide
        elif isinstance(a_object, memoryview):
            b_object = a_object.tobytes()
        elif isinstance(a_object, bytearray):
            b_object = bytes(a_object)
        else:
            b_object = a_object
        return b_object
