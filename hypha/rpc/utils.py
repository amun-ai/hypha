"""Provide utility functions for RPC."""
import asyncio
import copy
import secrets
import string
import traceback


def generate_password(length=50):
    """Generate a password."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for i in range(length))


_hash_id = generate_password()


class dotdict(dict):  # pylint: disable=invalid-name
    """Access dictionary attributes with dot.notation."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __setattr__(self, name, value):
        """Set the attribute."""
        # Make an exception for __rid__
        if name == "__rid__":
            super().__setattr__("__rid__", value)
        else:
            super().__setitem__(name, value)

    def __hash__(self):
        """Return the hash."""
        if self.__rid__ and type(self.__rid__) is str:
            return hash(self.__rid__ + _hash_id)

        # FIXME: This does not address the issue of inner list
        return hash(tuple(sorted(self.items())))

    def __deepcopy__(self, memo=None):
        """Make a deep copy."""
        return dotdict(copy.deepcopy(dict(self), memo=memo))


def format_traceback(traceback_string):
    """Format traceback."""
    formatted_lines = traceback_string.splitlines()
    # remove the second and third line
    formatted_lines.pop(1)
    formatted_lines.pop(1)
    formatted_error_string = "\n".join(formatted_lines)
    formatted_error_string = formatted_error_string.replace(
        'File "<string>"', "Plugin script"
    )
    return formatted_error_string


class Promise(object):  # pylint: disable=useless-object-inheritance
    """Represent a promise."""

    def __init__(self, pfunc, logger=None):
        """Set up promise."""
        self._resolve_handler = None
        self._finally_handler = None
        self._catch_handler = None
        self._logger = logger

        def resolve(*args, **kwargs):
            return self.resolve(*args, **kwargs)

        def reject(*args, **kwargs):
            return self.reject(*args, **kwargs)

        try:
            pfunc(resolve, reject)
        except Exception as exp:
            logger.error("Uncaught Exception: {}".format(exp))
            reject(exp)

    def resolve(self, result):
        """Resolve promise."""
        try:
            if self._resolve_handler:
                return self._resolve_handler(result)
        except Exception as exc:  # pylint: disable=broad-except
            if self._catch_handler:
                self._catch_handler(exc)
            elif not self._finally_handler:
                if self._logger:
                    self._logger.error("Uncaught Exception: {}".format(exc))
        finally:
            if self._finally_handler:
                self._finally_handler()

    def reject(self, error):
        """Reject promise."""
        try:
            if self._catch_handler:
                return self._catch_handler(error)
            elif not self._finally_handler:
                if self._logger:
                    self._logger.error("Uncaught Exception: {}".format(error))
        finally:
            if self._finally_handler:
                self._finally_handler()

    def then(self, handler):
        """Implement then callback.

        Set handler and return the promise.
        """
        self._resolve_handler = handler
        return self

    def finally_(self, handler):
        """Implement finally callback.

        Set handler and return the promise.
        """
        self._finally_handler = handler
        return self

    def catch(self, handler):
        """Implement catch callback.

        Set handler and return the promise.
        """
        self._catch_handler = handler
        return self


class FuturePromise(Promise, asyncio.Future):
    """Represent a promise as a future."""

    def __init__(self, pfunc, logger=None, dispose=None):
        """Set up promise."""
        self.__dispose = dispose
        self.__obj = None
        asyncio.Future.__init__(self)
        Promise.__init__(self, pfunc, logger)

    async def __aenter__(self):
        """Enter context for async."""
        ret = await self
        if isinstance(ret, dict):
            if "__enter__" in ret:
                ret = await ret["__enter__"]()
            self.__obj = ret
        return ret

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context for async."""
        if self.__obj:
            if "__exit__" in self.__obj:
                await self.__obj["__exit__"]()
            if self.__dispose:
                await self.__dispose(self.__obj)
            del self.__obj

    def resolve(self, result):
        """Resolve promise."""
        if self._resolve_handler or self._finally_handler:
            super().resolve(result)
        else:
            self.set_result(result)

    def reject(self, error):
        """Reject promise."""
        if self._catch_handler or self._finally_handler:
            super().reject(error)
        else:
            if error:
                self.set_exception(Exception(str(error)))
            else:
                self.set_exception(Exception())


class MessageEmitter:
    """Represent a message emitter."""

    def __init__(self, logger=None):
        """Set up instance."""
        self._event_handlers = {}
        self._logger = logger

    def on(self, event, handler):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def once(self, event, handler):
        """Register an event handler that should only run once."""
        # wrap the handler function,
        # this is needed because setting property
        # won't work for member function of a class instance
        def wrap_func(*args, **kwargs):
            return handler(*args, **kwargs)

        wrap_func.___event_run_once = True
        self.on(event, wrap_func)

    def off(self, event=None, handler=None):
        """Reset one or all event handlers."""
        if event is None and handler is None:
            self._event_handlers = {}
        elif event is not None and handler is None:
            if event in self._event_handlers:
                self._event_handlers[event] = []
        else:
            if event in self._event_handlers:
                self._event_handlers[event].remove(handler)

    def emit(self, msg):
        """Emit a message."""
        raise NotImplementedError

    def _fire(self, event, data=None):
        """Fire an event handler."""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(data)
                except Exception as err:
                    traceback_error = traceback.format_exc()
                    if self._logger:
                        self._logger.exception(err)
                    self.emit({"type": "error", "message": traceback_error})
                finally:
                    if hasattr(handler, "___event_run_once"):
                        self._event_handlers[event].remove(handler)
        else:
            if self._logger and self._logger.debug:
                self._logger.debug("Unhandled event: {}, data: {}".format(event, data))


def encode_zarr_store(zobj):
    """Encode the zarr store."""
    import zarr

    path_prefix = f"{zobj.path}/" if zobj.path else ""

    def getItem(key, options=None):
        return zobj.store[path_prefix + key]

    def setItem(key, value):
        zobj.store[path_prefix + key] = value

    def containsItem(key, options=None):
        if path_prefix + key in zobj.store:
            return True

    return {
        "_rintf": True,
        "_rtype": "zarr-array" if isinstance(zobj, zarr.Array) else "zarr-group",
        "getItem": getItem,
        "setItem": setItem,
        "containsItem": containsItem,
    }


def register_default_codecs(options=None):
    """Register default codecs."""
    from imjoy_rpc import api

    if options is None or "zarr-array" in options:
        import zarr

        api.registerCodec(
            {"name": "zarr-array", "type": zarr.Array, "encoder": encode_zarr_store}
        )

    if options is None or "zarr-group" in options:
        import zarr

        api.registerCodec(
            {"name": "zarr-group", "type": zarr.Group, "encoder": encode_zarr_store}
        )
