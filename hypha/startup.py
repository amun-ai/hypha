import asyncio
import logging
import os
import sys
from functools import partial


LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("services")
logger.setLevel(LOGLEVEL)

import importlib.util
from importlib import import_module
from hypha.core.auth import set_parse_token_function, set_generate_token_function, set_login_service, create_custom_login_service


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
    
    # Add auth-related functions to the server interface
    server["set_parse_token_function"] = set_parse_token_function
    server["set_generate_token_function"] = set_generate_token_function
    server["set_login_service"] = partial(set_login_service, server)
    server["create_custom_login_service"] = create_custom_login_service
    
    # Also provide access to the store for more advanced use cases
    server["store"] = store
    server["get_workspace_interface"] = store.get_workspace_interface
    server["get_root_user"] = store.get_root_user
    
    logger.info("Executing startup function: %s", startup_function_uri)
    await load_func(server)
    logger.info("Successfully executed the startup function: %s", startup_function_uri)
