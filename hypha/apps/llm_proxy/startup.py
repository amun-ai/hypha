"""Startup function for LLM Proxy app.

This module provides the startup function that can be used with
Hypha's startup system to automatically install and configure the
LLM proxy application.
"""

import logging
import asyncio
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


async def hypha_startup(server: Dict[str, Any]) -> None:
    """Startup function to install and configure LLM proxy app.
    
    This function is designed to be compatible with Hypha's startup system.
    It can be called from the main startup.py or registered as a startup module.
    
    Args:
        server: The Hypha server dictionary containing services and configuration
    
    Environment variables:
        HYPHA_ENABLE_LLM_PROXY: Set to "true" to enable LLM proxy installation
        HYPHA_LLM_PROXY_CONFIG: Optional JSON string with proxy configuration
        OPENAI_API_KEY: OpenAI API key for the proxy
        ANTHROPIC_API_KEY: Anthropic API key for the proxy
        Other LLM provider API keys as needed
    """
    # Check if LLM proxy should be enabled
    enable_llm_proxy = os.environ.get("HYPHA_ENABLE_LLM_PROXY", "").lower() == "true"
    
    if not enable_llm_proxy:
        logger.debug("LLM proxy is not enabled (set HYPHA_ENABLE_LLM_PROXY=true to enable)")
        return
    
    logger.info("LLM Proxy startup initiated")
    
    try:
        # Get the app controller service
        app_controller = await server.get_service("server-apps")
        if not app_controller:
            logger.warning("Server apps controller not found, skipping LLM proxy installation")
            return
        
        # Check if conda worker is available
        try:
            conda_worker = await server.get_service("conda-worker")
            if not conda_worker:
                # Try to register conda worker if not available
                logger.info("Conda worker not found, attempting to register...")
                from hypha.workers.conda import CondaWorker
                conda_worker_instance = CondaWorker()
                await server.register_service(conda_worker_instance.get_worker_service())
                logger.info("Conda worker registered successfully")
        except Exception as e:
            logger.error(f"Failed to setup conda worker: {e}")
            return
        
        # Import the install function
        from .llm_proxy_app import install_llm_proxy
        
        # Get configuration from environment
        config = {}
        
        # Check for API keys in environment
        api_keys = {}
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", 
                   "GEMINI_API_KEY", "COHERE_API_KEY", "HUGGINGFACE_API_KEY"]:
            if os.environ.get(key):
                api_keys[key] = os.environ[key]
        
        if api_keys:
            config["api_keys"] = api_keys
            logger.info(f"Found API keys for: {', '.join(api_keys.keys())}")
        
        # Get custom configuration from environment
        import json
        custom_config = os.environ.get("HYPHA_LLM_PROXY_CONFIG")
        if custom_config:
            try:
                config.update(json.loads(custom_config))
                logger.info("Applied custom LLM proxy configuration")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse HYPHA_LLM_PROXY_CONFIG: {e}")
        
        # Install the LLM proxy application
        logger.info("Installing LLM proxy application...")
        app_id = await install_llm_proxy(
            app_controller,
            workspace="public",
            config=config
        )
        logger.info(f"LLM proxy application installed successfully with ID: {app_id}")
        
        # Optionally start the app automatically
        auto_start = os.environ.get("HYPHA_LLM_PROXY_AUTO_START", "true").lower() == "true"
        if auto_start:
            logger.info("Auto-starting LLM proxy application...")
            await app_controller.start(app_id=app_id)
            logger.info("LLM proxy application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to install LLM proxy: {e}", exc_info=True)


async def register_llm_proxy_app(
    server: Dict[str, Any],
    app_controller: Any,
    conda_worker: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    auto_start: bool = True
) -> Optional[str]:
    """Helper function to register LLM proxy app programmatically.
    
    This can be called directly from other parts of the Hypha server
    to install the LLM proxy without using environment variables.
    
    Args:
        server: The Hypha server instance
        app_controller: The ServerAppController instance
        conda_worker: Optional CondaWorker instance (will create if not provided)
        config: Optional configuration for the LLM proxy
        auto_start: Whether to auto-start the app after installation
        
    Returns:
        The app ID if successful, None otherwise
    """
    try:
        # Ensure conda worker is available
        if not conda_worker:
            from hypha.workers.conda import CondaWorker
            conda_worker = CondaWorker()
            await server.register_service(conda_worker.get_worker_service())
        
        # Import and install the app
        from .llm_proxy_app import install_llm_proxy
        
        app_id = await install_llm_proxy(
            app_controller,
            workspace="public", 
            config=config
        )
        
        if auto_start:
            await app_controller.start(app_id=app_id)
        
        return app_id
        
    except Exception as e:
        logger.error(f"Failed to register LLM proxy app: {e}")
        return None