"""Test the triton server proxy."""
import gzip

import msgpack
import numpy as np
import pytest
import requests

from . import SERVER_URL


def execute(inputs, server_url, model_name, **kwargs):
    """Execute a model on the trition server."""
    # Represent the numpy array with imjoy_rpc encoding
    # See: https://github.com/imjoy-team/imjoy-rpc#data-type-representation
    for idx, input_data in enumerate(inputs):
        if isinstance(input_data, (np.ndarray, np.generic)):
            inputs[idx] = {
                "_rtype": "ndarray",
                "_rvalue": input_data.tobytes(),
                "_rshape": input_data.shape,
                "_rdtype": str(input_data.dtype),
            }

    kwargs.update(
        {
            "inputs": inputs,
            "model_name": model_name,
        }
    )
    # Encode the arguments as msgpack
    data = msgpack.dumps(kwargs)

    # Compress the data and send it via a post request to the server
    compressed_data = gzip.compress(data)
    response = requests.post(
        f"{server_url}/public/services/triton-client/execute",
        data=compressed_data,
        headers={
            "Content-Type": "application/msgpack",
            "Content-Encoding": "gzip",
        },
    )

    if response.ok:
        # Decode the results form the response
        results = msgpack.loads(response.content)
        # Convert the ndarray objects into numpy arrays
        for key in results:
            result = results[key]
            if (
                isinstance(result, dict)
                and result.get("_rtype") == "ndarray"
                and result["_rdtype"] != "object"
            ):
                results[key] = np.frombuffer(
                    result["_rvalue"], dtype=result["_rdtype"]
                ).reshape(result["_rshape"])
        return results
    raise Exception(f"Failed to execute {model_name}: {response.text}")


@pytest.mark.skip(reason="requires a triton server")
def test_trition_execute(fastapi_server):
    """Test trition execute."""
    image_array = np.random.randint(0, 255, [3, 256, 256]).astype("float32")
    params = {"diameter": 30}
    results = execute(
        inputs=[image_array, params],
        server_url=SERVER_URL,
        model_name="cellpose-python",
        decode_json=True,
    )
    mask = results["mask"]
    assert mask.shape == (1, 256, 256)
