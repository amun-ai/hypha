import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


@batch
def infer_func(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    return [input1_batch + 1.0]


def start_triton_server(http_port=9930, grpc_port=9920, model_repository=None):
    triton_config = TritonConfig(
        http_address="127.0.0.1",
        http_port=str(http_port),
        grpc_address="127.0.0.1",
        grpc_port=str(grpc_port),
        metrics_port="9921",
        model_repository=model_repository,
    )

    # Connecting inference callback with Triton Inference Server
    with Triton(config=triton_config) as triton:
        # Load model into Triton Inference Server
        triton.bind(
            model_name="AddOne",
            infer_func=infer_func,
            inputs=[
                Tensor(dtype=np.float32, shape=(-1,)),
            ],
            outputs=[
                Tensor(dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        # Serve model through Triton Inference Server
        triton.serve()
