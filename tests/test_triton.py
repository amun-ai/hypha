"""Test the triton server proxy."""
import msgpack
import numpy as np
import pytest
import requests

from . import SIO_SERVER_URL


@pytest.mark.skip(reason="requires a triton server")
def test_trition_execute():
    """Test trition execute."""
    image_array = np.random.randint(0, 255, [3, 256, 256]).astype("float32")
    # Represent the image array as imjoy_rpc encoding
    # See: https://github.com/imjoy-team/imjoy-rpc#data-type-representation
    encoded_image = {
        "_rtype": "ndarray",
        "_rvalue": image_array.tobytes(),
        "_rshape": image_array.shape,
        "_rdtype": str(image_array.dtype),
    }
    params = {"diameter": 30}

    # Encode the arguments as msgpack
    encoded_args = msgpack.dumps(
        {
            "inputs": [encoded_image, params],
            "model_name": "cellpose-python",
            "model_version": "1",
        }
    )

    # Send a post request to the server
    response = requests.post(
        f"{SIO_SERVER_URL}/public/services/triton-client/execute",
        data=encoded_args,
        headers={"Content-type": "application/msgpack"},
    )

    if response.ok:
        # Decode the results form the response
        results = msgpack.loads(response.content)
        mask_obj = results["mask"]
        # Convert the mask object into a numpy array
        mask = np.frombuffer(mask_obj["_rvalue"], dtype=mask_obj["_rdtype"]).reshape(
            mask_obj["_rshape"]
        )
        assert mask.shape == (1, 256, 256)
    else:
        raise Exception("Failed to execute")
