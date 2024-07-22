import asyncio
import logging
import os
import sys


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("services")
logger.setLevel(logging.INFO)
import importlib.util
from importlib import import_module


def _load_function(module_path, entrypoint):
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("user_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = import_module(module_path)

    return getattr(module, entrypoint)


async def run_startup_function(store: any, startup_function_uri: str):
    parts = startup_function_uri.split(":")
    module_path = parts[0]
    assert not module_path.endswith(".py") or os.path.exists(
        module_path
    ), f"Module {module_path} does not exist"
    entrypoint = parts[1] if len(parts) > 1 else None
    assert (
        entrypoint
    ), f"Entrypoint is required for {startup_function_uri}, please use {startup_function_uri}:entrypoint_function"

    # load the python module and get the entrypoint
    load_func = _load_function(module_path, entrypoint)
    # make sure the load_func is a coroutine
    assert asyncio.iscoroutinefunction(
        load_func
    ), f"Entrypoint {entrypoint} is not a coroutine"
    server = await store.get_public_api()
    await load_func(server)
    logger.info(f"Successfully executed the startup function: {startup_function_uri}")
