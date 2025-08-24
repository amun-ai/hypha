"""Example startup file showing how to use internal apps.

This file demonstrates how to use internal apps with Hypha's startup system.
You can use this as a template or reference.

To use this startup file:
    hypha --startup-functions=hypha.apps.example_startup:hypha_startup

Or set the environment variable:
    export HYPHA_STARTUP_FUNCTIONS="hypha.apps.example_startup:hypha_startup"
"""

import os
import logging

logger = logging.getLogger(__name__)


async def hypha_startup(server):
    """Example startup function that installs internal apps based on configuration.
    
    This function demonstrates how to:
    1. Check for environment variables
    2. Install internal apps conditionally
    3. Configure apps based on environment
    
    Args:
        server: The Hypha server instance
    """
    logger.info("Running example startup function")
    
    # Example 1: Install LLM proxy if enabled
    if os.environ.get("HYPHA_ENABLE_LLM_PROXY", "").lower() == "true":
        try:
            from hypha.apps.llm_proxy import hypha_startup as llm_startup
            await llm_startup(server)
            logger.info("LLM proxy installed via startup")
        except Exception as e:
            logger.error(f"Failed to install LLM proxy: {e}")
    
    # Example 2: Install multiple apps based on a list
    enabled_apps = os.environ.get("HYPHA_ENABLED_APPS", "").split(",")
    for app_name in enabled_apps:
        app_name = app_name.strip()
        if not app_name:
            continue
            
        try:
            # Dynamically import and run the app's startup function
            module = __import__(f"hypha.apps.{app_name}", fromlist=["hypha_startup"])
            if hasattr(module, "hypha_startup"):
                await module.hypha_startup(server)
                logger.info(f"App '{app_name}' installed via startup")
            else:
                logger.warning(f"App '{app_name}' has no hypha_startup function")
        except ImportError as e:
            logger.warning(f"App '{app_name}' not found: {e}")
        except Exception as e:
            logger.error(f"Failed to install app '{app_name}': {e}")
    
    # Example 3: Register a custom service
    if os.environ.get("HYPHA_REGISTER_CUSTOM_SERVICE", "").lower() == "true":
        await server.register_service({
            "id": "example-service",
            "name": "Example Service",
            "description": "An example service registered at startup",
            "hello": lambda name="World": f"Hello, {name}!",
        })
        logger.info("Example service registered")
    
    logger.info("Example startup completed")