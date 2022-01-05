import msgpack
import numpy as np


def encode(obj):
    if callable(obj):
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
    if isinstance(obj, np.ndarray):
        return msgpack.ExtType(
            57, msgpack.packb({"_rtype": "ndarray", "data": obj.tobytes()})
        )
    raise TypeError("Unknown type: %r" % (obj,))


def decode(code, data):
    if code == 57:
        obj = msgpack.unpackb(data)
        if obj["_rtype"] == "ndarray":
            obj = np.random.random(size=(50000, 100))
    return ExtType(code, data)


data = np.random.random(size=(50000, 100))
encoded = msgpack.packb(data, default=encode)

decoded = msgpack.unpackb(encoded, ext_hook=decode)
print(decoded)
