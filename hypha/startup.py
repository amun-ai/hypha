import asyncio
import logging
import os
import sys


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("services")
logger.setLevel(logging.INFO)
import importlib.util


def _load_function(module_path, function_name):
    spec = importlib.util.spec_from_file_location(
        f"config_function_{function_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    return function


async def run_start_function(store: any, startup_function_uri: str):
    parts = startup_function_uri.split(":")
    module_path = parts[0]
    assert os.path.exists(module_path), f"Module {module_path} does not exist"
    entrypoint = parts[1] if len(parts) > 1 else None
    assert (
        entrypoint
    ), f"Entrypoint is required for {startup_function_uri}, please use {startup_function_uri}:entrypoint_function"

    server = store.get_public_workspace_interface()
    # load the python module and get the entrypoint
    load_func = _load_function(module_path, entrypoint)
    # make sure the load_func is a coroutine
    assert asyncio.iscoroutinefunction(
        load_func
    ), f"Entrypoint {entrypoint} is not a coroutine"
    await load_func(server)
    logger.info(f"Services loaded from {startup_function_uri}")
