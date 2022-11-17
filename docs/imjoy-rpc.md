# ImJoy RPC V2

Symmetrical Transparent Remote Procedure Calls

The core library that powers [ImJoy](https://imjoy.io).

The imjoy-rpc v2 is wrapped under a submodule `imjoy_rpc.hypha`:

```python
from imjoy_rpc.hypha import connect_to_server

server = await connect_to_server({"server_url": server_url})
```


## Data type representation

ImJoy RPC is built on top of two-way transport layer. The data representation is a JSON object (but can contain binary data, e.g. `ArrayBuffer` in JS or `bytes` in Python). The goal is to represent more complex data types with primitive types that are commonly supported by many programming language, including strings, numbers, boolean, bytes, list/array and dictionary/object.


| Javascript | Python | imjoy-rpc representation |
|------------|--------- | ---- |
| String   | str        | v |
| Number   | int/float | v |
| Boolean  |  bool     | v |
| null/undefined  | None    | v |
| ArrayBuffer | bytes  | v |
| Array([])   | list/tuple |[_encode(v)] |
| Object({})  | dict  | {_encode(v)} |
| Set | Set | {_rtype: "set", _rvalue: _encode(Array.from(v))} |
| Map | OrderedDict  |{_rtype: "orderedmap", _rvalue: _encode(Array.from(v))} |
| Error | Exception | { _rtype: "error", _rvalue: v.toString() } |
| DataView | memoryview  |  { _rtype: "memoryview", _rvalue: v.buffer }|
| TypedArray | 1-D numpy array*  |{_rtype: "typedarray", _rvalue: v.buffer, _rdtype: dtype} |
| tf.Tensor/nj.array | numpy array  |{_rtype: "ndarray", _rvalue: v.buffer, _rshape: shape, _rdtype: _dtype} |
| Function* | function/callable* | {_rtype: "method", _rtarget: _rtarget, _rmethod: name, _rpromise: true/false } |
| Class | class/dotdict()* | {...} |
| custom | custom | encoder(v) (default `_rtype` = encoder name) |

Notes:
 - `_encode(...)` in the imjoy-rpc representation means the type will be recursively encoded (decoded).
 - For any JS object or Python dictonary, if it has a key named `_rintf` and the value is set to true, then this object will be treated as an interface. All functions contained in the object/dictionary, including all the children dictionary at any level, will be treated as `interface` function. Otherwise, it will be encoded as `callback`. The difference is `interface` function can be called many times, but the `callback` function can only be called once (Also see "Remote Objects" below).
 - For n-D numpy array, there is no established n-D array library in javascript, the current behavior is, if there is `tf`(Tensorflow.js) detected, then it will be decoded into `tf.Tensor`. If `nj`(numjs) is detected, then it will be decoded into `nj.array`.
 - Typed array will be represented as numpy array if available, otherwise it will be converted to raw bytes.    
    Type Conversion
    | Javascript | Numpy  | _rdtype |
    | -- | -- | -- |
    | Int8Array | int8 | int8 |
    | Int16Array| int16 |int16 |
    |  Int32Array| int32 | int32 |
    |  Uint8Array| uint8 | uint8 |
    |  Uint16Array| uint16 | uint16 |
    |  Uint32Array| uint32 | uint32 |
    |  Float32Array| float32 | float32 |
    |  Float64Array| float64 | float64 |
    |  Array| array | array |
    |note: 64-bit integers (signed or unsigned) are not supported|

 - `dotdict` in Python is a simple wrapper over `dict` that support using the dot notation to get item, similar to what you can do with Javascript object.
 - In Python, file instances (inherit from `io.IOBase`) will be automatically encoded.

 ## Encoding and decoding custom objects
 For the data or object types that are not in the table above, for example, a custom class, you can support them by register your own `codec`(i.e. encoder and decoder) with `server.register_codec()`.

 You need to provide a `name`, a `type`, `encoder` and `decoder` function. For example: in javascript, you can call `server.register_codec({"name": "my_custom_codec", "type": MyClass, "encoder": (obj)=>{ ... return encoded;}, "decoder": (obj)=>{... return decoded;})`, or in Python you can do `server.register_codec(name="my_custom_codec", type=MyClass, encoder=my_encoder_func, decoder=my_decoder_func)`.
 

 The basic idea of using a custom codec is to use the `encoder` to represent your custom data type into array/dictionary of primitive types (string, number etc.) such that they can be send via the transport layer of imjoy-rpc. Then use the `decoder` to reconstruct the object remotely based on the representation.

For the `name`, it will be assigned as `_rtype` for the data representation, therefore please be aware that you should not use a name that already used internally (see the table above), unless you want to overried the default encoding. Also note that you cannot overried the encoding of primitive types and functions.

The `encoder` function take an object as input and you need to return the represented object/dictionary. You can only use primitive types plus array/list and object/dict in the represented object. If you want to include function in the encoding, please also include a key `_rintf` and set the value to true. By default, if your returned object does not contain a key `_rtype`, the codec `name` will be used as `_rtype`. You can also assign a different `_rtype` name, that allows the conversion between different types.

The `decoder` function converts the encoded object into the actual object. It will only be called when the `_rtype` of an object matches the `name` of the codec.

### Remote function calls and arguments
Remote function call is almost the same as calling a local function. The arguments are mapped directly, for example, you can call a Python function `foo(a, b, c)` from javascript or vise versa. However, since Javascript does not support named arguments as Python does, ImJoy does the following conversion:
 * For functions defined in Javascript, there is no difference when calling from Python
 * For functions defined in Python, when calling from Javascript, if the last argument is an object and its `_rkwargs` is set to true, then it will be converted into keyword arguments when calling the Python function. For example, if you have a Python function defined as `def foo(a, b, c=None):`, in Javascript, you should call it as `foo(9, 10, {c: 33, _rkwargs: true})`.


### Example 1: Encode zarr store

```python
import asyncio
from imjoy_rpc.hypha import connect_to_server

import zarr
import numpy as np

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


async def start_server(server_url):
    server = await connect_to_server({"server_url": server_url})

    # Register the codecs
    server.register_codec(
        {"name": "zarr-group", "type": zarr.Group, "encoder": encode_zarr_store}
    )

    z = zarr.array(np.arange(100))
  
    # Use the echo function to do a round-trip with the zarr object
    # Note: Since we didn't create a decoder, so we won't get the zarr object, but a zarr store interface
    z2 = await server.echo(z)
    print(z2)

if __name__ == "__main__":
    server_url = "https://ai.imjoy.io"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()
```
