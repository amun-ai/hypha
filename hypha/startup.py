import asyncio
import logging
import os
import sys
from functools import partial
from typing import Callable, Optional, Dict, Any

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("services")
logger.setLevel(LOGLEVEL)

import importlib.util
from importlib import import_module
from hypha.core.auth import set_parse_token_function, set_generate_token_function, set_get_token_function


async def register_auth_service(
    server,
    parse_token: Optional[Callable] = None,
    generate_token: Optional[Callable] = None,
    get_token: Optional[Callable] = None,
    login_service: Optional[Dict[str, Any]] = None,
    index_handler: Optional[Callable] = None,
    start_handler: Optional[Callable] = None,
    check_handler: Optional[Callable] = None,
    report_handler: Optional[Callable] = None,
    **extra_handlers
):
    """Register a custom authentication service with Hypha.
    
    This function provides a unified interface for customizing authentication in Hypha.
    You can customize token parsing, token generation, token extraction, and the login service.
    
    Args:
        server: The Hypha server instance
        parse_token: Optional custom token parsing function (async or sync)
        generate_token: Optional custom token generation function (async or sync)
        get_token: Optional custom token extraction function (async or sync) that receives
                   a scope object and returns the token string
        login_service: Optional complete login service dictionary
        index_handler: Optional handler for serving login page (overrides login_service)
        start_handler: Optional handler for starting login session (overrides login_service)
        check_handler: Optional handler for checking login status (overrides login_service)
        report_handler: Optional handler for reporting login completion (overrides login_service)
        **extra_handlers: Additional handlers for the login service
    
    Example:
        async def hypha_startup(server):
            await server["register_auth_service"](
                parse_token=my_parse_token,
                generate_token=my_generate_token,
                start_handler=my_start_login,
                check_handler=my_check_login,
                report_handler=my_report_login,
                index_handler=my_login_page
            )
    """
    # Set custom token functions if provided
    if parse_token:
        await set_parse_token_function(parse_token)
        logger.info("Custom parse_token function registered via register_auth_service")
    
    if generate_token:
        await set_generate_token_function(generate_token)
        logger.info("Custom generate_token function registered via register_auth_service")
    
    if get_token:
        await set_get_token_function(get_token)
        logger.info("Custom get_token function registered via register_auth_service")
    
    # Handle login service registration
    if login_service or any([index_handler, start_handler, check_handler, report_handler]):
        if login_service:
            # Use provided login service
            service = login_service.copy()
            
            # Override with specific handlers if provided
            if index_handler:
                service["index"] = index_handler
            if start_handler:
                service["start"] = start_handler
            if check_handler:
                service["check"] = check_handler
            if report_handler:
                service["report"] = report_handler
                
            # Add any extra handlers
            for key, value in extra_handlers.items():
                if key not in ["parse_token", "generate_token"]:
                    service[key] = value
        else:
            # Build login service from handlers
            # We need at least the required handlers
            if not all([index_handler, start_handler, check_handler, report_handler]):
                # If not all handlers provided, create a minimal default service
                from hypha.core.auth import create_login_service
                default_service = create_login_service(server.get("store"))
                
                service = {
                    "id": "hypha-login",
                    "name": "Hypha Login",
                    "type": "functions",
                    "description": "Custom authentication service for Hypha",
                    "config": {"visibility": "public", "workspace": "public"},
                    "index": index_handler or default_service["index"],
                    "start": start_handler or default_service["start"],
                    "check": check_handler or default_service["check"],
                    "report": report_handler or default_service["report"],
                }
            else:
                service = {
                    "id": "hypha-login",
                    "name": "Hypha Login",
                    "type": "functions",
                    "description": "Custom authentication service for Hypha",
                    "config": {"visibility": "public", "workspace": "public"},
                    "index": index_handler,
                    "start": start_handler,
                    "check": check_handler,
                    "report": report_handler,
                }
            
            # Add extra handlers
            for key, value in extra_handlers.items():
                if key not in ["parse_token", "generate_token"]:
                    service[key] = value
        
        # Ensure service has required fields
        if "id" not in service:
            service["id"] = "hypha-login"
        if "name" not in service:
            service["name"] = "Hypha Login"
        if "type" not in service:
            service["type"] = "functions"
        if "config" not in service:
            service["config"] = {}
        service["config"]["visibility"] = "public"
        service["config"]["workspace"] = "public"
        
        # Register the login service
        try:
            # Check if there's already a hypha-login service and unregister it
            try:
                existing_service = await server.get_service("public/hypha-login")
                if existing_service:
                    logger.info("Replacing existing hypha-login service")
            except:
                pass  # Service doesn't exist yet
            
            await server.register_service(service, overwrite=True)
            logger.info(f"Successfully registered custom login service: {service['id']}")
        except Exception as e:
            logger.error(f"Failed to register login service: {e}")
            raise

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
    
    # Add the simplified auth registration function
    server["register_auth_service"] = partial(register_auth_service, server)
    
    # Keep legacy functions for backward compatibility
    server["set_parse_token_function"] = set_parse_token_function
    server["set_generate_token_function"] = set_generate_token_function
    server["set_get_token_function"] = set_get_token_function
    
    # Also provide access to the store for more advanced use cases
    server["store"] = store
    server["get_workspace_interface"] = store.get_workspace_interface
    server["get_root_user"] = store.get_root_user
    
    logger.info("Executing startup function: %s", startup_function_uri)
    await load_func(server)
    logger.info("Successfully executed the startup function: %s", startup_function_uri)
