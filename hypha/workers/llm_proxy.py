"""LLM Proxy Worker for Hypha - integrates litellm for multi-provider LLM support."""

import asyncio
import logging
import traceback
import uuid
from typing import Any, Dict, List, Optional


from fastapi import FastAPI
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

from litellm.proxy import proxy_server
import litellm
from litellm.router import Router

from hypha.core import UserInfo
from hypha.workers.base import BaseWorker
from hypha_rpc import connect_to_server

logger = logging.getLogger(__name__)


class LLMProxyWorker(BaseWorker):
    """Worker for LLM proxy using litellm for multi-provider support."""

    def __init__(self, store, workspace_manager, worker_id):
        """Initialize the LLM proxy worker."""
        super().__init__()
        self._store = store
        self._workspace_manager = workspace_manager
        self._worker_id = worker_id
        self._sessions = {}
        self._cleanup_task = None

    def _get_litellm_endpoints(self, service_id: str) -> Dict[str, str]:
        """Programmatically get all available litellm endpoints.
        
        Returns a dictionary mapping endpoint names to their URLs.
        """
        import re
        
        # Get all routes from the litellm proxy server app
        app = proxy_server.app
        endpoints = {}
        
        # Track seen paths to avoid duplicates
        seen_paths = set()
        
        for route in app.routes:
            if hasattr(route, 'path') and '/v1/' in route.path:
                # Get the original path
                path = route.path
                
                # Skip provider-prefixed routes (they're duplicates)
                if path.startswith('/{provider}/'):
                    continue
                
                # Create a simplified name from the path
                # Remove /v1/ prefix and parameters
                name_parts = path.replace('/v1/', '').split('/')
                
                # Filter out path parameters and create a name
                clean_parts = []
                for part in name_parts:
                    if part and not part.startswith('{'):
                        clean_parts.append(part)
                
                if clean_parts:
                    # Create a descriptive name
                    endpoint_name = '_'.join(clean_parts)
                    
                    # Replace path parameters with wildcards for display
                    display_path = re.sub(r'\{[^}]+\}', '*', path)
                    
                    # Only add unique paths
                    if display_path not in seen_paths:
                        seen_paths.add(display_path)
                        # Construct the actual URL with service_id
                        url = f"/apps/{service_id}{display_path}"
                        endpoints[endpoint_name] = url
        
        # Add some common aliases for important endpoints
        common_endpoints = {
            "openai_chat": f"/apps/{service_id}/v1/chat/completions",
            "claude_messages": f"/apps/{service_id}/v1/messages", 
            "completions": f"/apps/{service_id}/v1/completions",
            "embeddings": f"/apps/{service_id}/v1/embeddings",
            "models": f"/apps/{service_id}/v1/models",
            "health": f"/apps/{service_id}/health",
        }
        
        # Merge common endpoints (they override auto-generated ones)
        endpoints.update(common_endpoints)
        
        return endpoints

    async def _resolve_secrets_in_model_list(self, model_list: List[Dict], server) -> List[Dict]:
        """Resolve HYPHA_SECRET: prefixed values in model_list by fetching from workspace env.
        
        Args:
            model_list: List of model configurations
            server: The connected server instance to fetch env variables from
            
        Returns:
            Updated model_list with resolved secrets
        """
        resolved_list = []
        
        for model_config in model_list:
            # Deep copy to avoid modifying the original
            resolved_model = model_config.copy()
            
            # Check litellm_params for secrets
            if "litellm_params" in resolved_model:
                litellm_params = resolved_model["litellm_params"].copy()
                
                # Check each parameter for HYPHA_SECRET: prefix
                for param_key, param_value in litellm_params.items():
                    if isinstance(param_value, str) and param_value.startswith("HYPHA_SECRET:"):
                        # Extract the env variable name
                        env_key = param_value[len("HYPHA_SECRET:"):]
                        
                        try:
                            # Use the server directly to access workspace env
                            # The server has get_env method exposed directly
                            actual_value = await server.get_env(env_key)
                            litellm_params[param_key] = actual_value
                            logger.info(f"Resolved secret for model {resolved_model.get('model_name')}: {param_key} from workspace env key: {env_key}")
                        except Exception as e:
                            logger.error(f"Failed to resolve secret for {param_key} from workspace env key '{env_key}': {e}")
                            # Keep the original value or raise an error
                            raise ValueError(f"Could not resolve workspace secret '{env_key}' for parameter '{param_key}': {e}")
                
                resolved_model["litellm_params"] = litellm_params
            
            resolved_list.append(resolved_model)
        
        return resolved_list
    
    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["llm-proxy"]
    
    @property
    def name(self) -> str:
        """Return the worker name."""
        return "LLM Proxy Worker"
    
    @property
    def description(self) -> str:
        """Return the worker description."""
        return "Worker for proxying LLM requests via litellm with support for OpenAI, Claude, Gemini, and more"

    async def start_worker(self):
        """Start the worker itself (called during initialization)."""
        logger.info(f"Starting LLM proxy worker: {self._worker_id}")
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

    async def stop(self):
        """Stop the worker."""
        logger.info(f"Stopping LLM proxy worker: {self._worker_id}")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all sessions
        for session_id in list(self._sessions.keys()):
            await self._cleanup_session(session_id)

    async def _cleanup_sessions(self):
        """Periodically clean up inactive sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                current_time = asyncio.get_event_loop().time()
                for session_id, session in list(self._sessions.items()):
                    if current_time - session.get("last_access", 0) > 300:  # 5 minutes
                        await self._cleanup_session(session_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def _cleanup_session(self, session_id: str):
        """Clean up a specific session."""
        session = self._sessions.pop(session_id, None)
        if session:
            # Disconnect the client if it exists
            client = session.get("client")
            if client:
                try:
                    # Unregister the LLM service before disconnecting
                    if "registered_service_id" in session:
                        try:
                            service_id_to_unregister = session["registered_service_id"]
                            logger.info(f"Attempting to unregister service: {service_id_to_unregister}")
                            await client.unregister_service(service_id_to_unregister)
                            logger.info(f"Successfully unregistered service: {service_id_to_unregister}")
                        except Exception as e:
                            logger.warning(f"Failed to unregister service {session.get('registered_service_id')}: {e}")
                    
                    # Disconnect the client
                    await client.disconnect()
                    logger.info(f"Disconnected client for session {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to disconnect client: {e}")
            
            # Clean up router if exists
            router = session.get("router")
            if router and hasattr(router, 'close'):
                try:
                    await router.close()
                except Exception:
                    pass
            
            logger.info(f"Cleaned up session {session_id}")

    @schema_method
    async def get_logs(
        self,
        session_id: str = Field(..., description="Session ID to get logs for"),
        type: str = Field(None, description="Type of logs to get (log/error/None for all)"),
        offset: int = Field(0, description="Starting offset for log entries"),
        limit: Optional[int] = Field(None, description="Maximum number of log entries to return"),
        context: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Get logs for a session.
        
        Returns a dictionary with:
        - items: List of log events
        - total: Total number of log items
        - offset: The offset used
        - limit: The limit used
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"items": [], "total": 0, "offset": offset, "limit": limit}
        
        logs = session.get("logs", [])
        
        # Convert logs to items format
        items = []
        for log in logs:
            # Determine log type based on content
            log_type = "log"
            if "error" in log.lower() or "failed" in log.lower():
                log_type = "error"
            
            items.append({
                "type": log_type,
                "content": log
            })
        
        # Filter by type if specified
        if type:
            items = [item for item in items if item["type"] == type]
        
        total = len(items)
        
        # Apply offset and limit
        if limit is not None:
            items = items[offset:offset + limit]
        else:
            items = items[offset:]
        
        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit
        }

    @schema_method
    async def compile(
        self,
        manifest: dict,
        files: list,
        config: dict = None,
        context: Optional[dict] = None,
    ) -> tuple[dict, list]:
        """Compile/prepare an LLM proxy session."""
        assert context is not None
        user_info = UserInfo.from_context(context)
        
        session_id = str(uuid.uuid4())
        
        # Use the manifest directly - it contains the app configuration
        if not manifest:
            raise ValueError("No manifest found")
        
        # Extract model configuration from the app's config section
        app_config = manifest.get("config", {})
        model_list = app_config.get("model_list", [])
        litellm_settings = app_config.get("litellm_settings", {})
        
        if not model_list:
            raise ValueError("No model_list found in app manifest config")
        
        # Store session info as dict
        session_info = {
            "session_id": session_id,
            "status": "compiled",
            "logs": "LLM proxy session prepared",
            "outputs": {}
        }
        
        self._sessions[session_id] = {
            "info": session_info,
            "model_list": model_list,
            "litellm_settings": litellm_settings,
            "workspace": context["ws"],
            "user": user_info.id,
            "last_access": asyncio.get_event_loop().time(),
            "logs": ["LLM proxy session initialized"],
        }
        
        # Add session_id to manifest for the execute phase
        manifest["session_id"] = session_id
        
        # Return the updated manifest and files (unchanged)
        return manifest, files

    async def _create_litellm_app(self, session_id: str) -> FastAPI:
        """Create or configure the litellm FastAPI app.
        
        This configures litellm's global proxy_server app with our model list
        and settings, then returns the app for ASGI serving.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        model_list = session["model_list"]
        litellm_settings = session.get("litellm_settings", {})
        
        # Create a Router with our model configuration
        # The Router handles load balancing and model selection
        router = Router(
            model_list=model_list,
            routing_strategy=litellm_settings.get("routing_strategy", "simple-shuffle"),
            set_verbose=litellm_settings.get("debug", False),
            debug_level=litellm_settings.get("debug_level", "INFO"),
        )
        
        # Set the global router that the proxy_server endpoints will use
        proxy_server.llm_router = router
        proxy_server.llm_model_list = model_list
        proxy_server.general_settings = litellm_settings
        
        # Configure other global settings
        proxy_server.user_debug = litellm_settings.get("debug", False)
        proxy_server.user_detailed_debug = litellm_settings.get("detailed_debug", False)
        proxy_server.user_telemetry = litellm_settings.get("telemetry", False)
        
        # Set litellm global settings
        litellm.drop_params = litellm_settings.get("drop_params", True)
        litellm.set_verbose = litellm_settings.get("debug", False)
        
        # Get the app - this is the FastAPI app with all routes already configured
        app = proxy_server.app
        
        logger.info(f"Configured litellm proxy app for session {session_id} with {len(model_list)} models")
        
        return app

    @schema_method
    async def start(
        self,
        config: dict,
        context: Optional[dict] = None,
    ) -> str:
        """Start an LLM proxy session with multi-provider support."""
        assert context is not None
        
        # Extract session_id from the config (it was added during compile)
        session_id = config.get("manifest", {}).get("session_id")
        if not session_id:
            # If no session_id in manifest, this might be a direct start without compile
            # Generate a new session_id
            session_id = str(uuid.uuid4())
            
        # Check if we have a compiled session
        session = self._sessions.get(session_id)
        if not session:
            # No compiled session, create one from the manifest
            manifest = config.get("manifest", {})
            if not manifest:
                raise ValueError("No manifest found in config")
            
            app_config = manifest.get("config", {})
            model_list = app_config.get("model_list", [])
            litellm_settings = app_config.get("litellm_settings", {})
            
            if not model_list:
                raise ValueError("No model_list found in app manifest config")
            
            # Create the session
            user_info = UserInfo.from_context(context)
            session_info = {
                "session_id": session_id,
                "status": "starting",
                "logs": "LLM proxy session starting",
                "outputs": {}
            }
            
            self._sessions[session_id] = {
                "info": session_info,
                "model_list": model_list,
                "litellm_settings": litellm_settings,
                "workspace": context["ws"],
                "user": user_info.id,
                "last_access": asyncio.get_event_loop().time(),
                "logs": ["LLM proxy session created"],
            }
            session = self._sessions[session_id]
        
        session["last_access"] = asyncio.get_event_loop().time()
        
        # Now start the LLM proxy service
        try:
            # Get service_id from manifest config, or use default pattern
            manifest = config.get("manifest", {})
            service_id = manifest.get("config", {}).get("service_id", f"llm-{session_id}")
            session["service_id"] = service_id
            
            # Determine which providers are configured
            configured_providers = set()
            for model_config in session["model_list"]:
                litellm_params = model_config.get("litellm_params", {})
                model = litellm_params.get("model", "")
                
                # Detect provider from model string
                if "gpt" in model.lower() or "openai" in model.lower():
                    configured_providers.add("OpenAI")
                elif "claude" in model.lower() or "anthropic" in model.lower():
                    configured_providers.add("Claude/Anthropic")
                elif "gemini" in model.lower() or "vertex" in model.lower():
                    configured_providers.add("Gemini/Vertex AI")
                elif "bedrock" in model.lower():
                    configured_providers.add("AWS Bedrock")
                elif "llama" in model.lower() or "replicate" in model.lower():
                    configured_providers.add("Replicate/Llama")
            
            # Create ASGI serve function for this session
            async def serve_llm(args, context=None):
                """ASGI handler for the LLM proxy service."""
                if context:
                    logger.debug(f'LLM proxy: {context["user"]["id"]} - {args["scope"]["method"]} - {args["scope"]["path"]}')
                await app(args["scope"], args["receive"], args["send"])
            
            # Get the connection info from config
            client_id = config.get("client_id", "")
            app_id = config.get("app_id", "")
            workspace = config.get("workspace", context["ws"])
            server_url = config.get("server_url", "")
            token = config.get("token", "")
            
            # Connect as a client to register services
            client = await connect_to_server({
                "server_url": server_url,
                "client_id": client_id,
                "workspace": workspace,
                "token": token,
                "app_id": app_id,
            })
            
            # Store client in session for cleanup
            session["client"] = client
            
            # Resolve any HYPHA_SECRET: prefixed values in model_list
            # This must be done after connecting to the server
            resolved_model_list = await self._resolve_secrets_in_model_list(
                session["model_list"], 
                client
            )
            session["model_list"] = resolved_model_list
            
            # Re-create the litellm app with resolved secrets
            app = await self._create_litellm_app(session_id)
            session["app"] = app
            
            # Register the LLM service as an ASGI service (type="asgi")
            service_info = await client.register_service({
                "id": service_id,
                "name": f"LLM Proxy - {session_id}",
                "type": "asgi",  # Changed from "llm-proxy" to "asgi" to use existing ASGI middleware
                "description": f"LLM proxy service with {len(session['model_list'])} models",
                "serve": serve_llm,  # ASGI handler function
                "config": {
                    "visibility": "protected",  # Only accessible within workspace
                    "require_context": True,
                },
                "app_id": app_id,  # Associate with the app
            })
            
            # Store the full service ID including the client ID path
            full_service_id = service_info["id"]
            logger.info(f"Registered LLM proxy service with full ID: {full_service_id}")
            session["registered_service_id"] = full_service_id
            
            # Update session info - using /apps/ path now instead of /llm/
            session["info"]["status"] = "running"
            session["info"]["logs"] = f"LLM proxy service ready with ID: {service_id}"
            session["info"]["outputs"] = {
                "service_id": service_id,
                "models": [m.get("model_name") for m in session["model_list"]],
                "providers": list(configured_providers),
                "endpoints": self._get_litellm_endpoints(service_id),
                "base_url": f"/apps/{service_id}",
            }
            
            session["logs"].append(f"LLM proxy started with service ID: {service_id}")
            session["logs"].append(f"Supporting providers: {', '.join(configured_providers)}") 
            
            # Return the session_id string as expected by the apps controller
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start LLM proxy: {e}")
            traceback.print_exc()
            
            session["info"]["status"] = "failed"
            session["info"]["logs"] = f"Failed to start LLM proxy: {str(e)}"
            session["logs"].append(f"Failed: {str(e)}")
            raise

    @schema_method
    async def stop(
        self,
        session_id: str,
        context: Optional[dict] = None
    ) -> None:
        """Stop an LLM proxy session."""
        if session_id not in self._sessions:
            logger.warning(f"LLM proxy session {session_id} not found for stopping")
            return
        
        logger.info(f"Stopping LLM proxy session {session_id}")
        await self._cleanup_session(session_id)
    
    def get_app_for_service(self, service_id: str):
        """Get the FastAPI app for a given service ID."""
        # Look for the session that has this service_id
        for session_id, session in self._sessions.items():
            if session.get("service_id") == service_id:
                return session.get("app")
        return None
    
    def get_service_api(self):
        """Get the service API definition."""
        return {
            "id": self._worker_id,
            "name": "LLM Proxy Worker",
            "type": "server-app-worker",  # Required for ServerAppController to find us
            "description": "Worker for proxying LLM requests via litellm with multi-provider support",
            "config": {
                "visibility": "public",
                "require_context": True,
            },
            "supported_types": self.supported_types,  # Add this so ServerAppController can find us
            "compile": self.compile,
            "start": self.start,
            "stop": self.stop,
            "get_logs": self.get_logs,
        }


async def hypha_startup(server):
    """Register the LLM proxy worker as a startup function."""
    import sys
    print("LLM PROXY STARTUP CALLED", file=sys.stderr)
    logger.info("LLM proxy startup function called")
    
    import traceback
    try:
        logger.info("Starting LLM proxy worker registration...")
        from hypha.workers.llm_proxy import LLMProxyWorker
        
        # The server object has a store attribute
        store = server.store if hasattr(server, 'store') else server
        
        # Get workspace manager from store
        workspace_manager = None
        if hasattr(store, 'get_workspace_manager'):
            workspace_manager = store.get_workspace_manager()
        
        # Create the LLM proxy worker
        worker = LLMProxyWorker(store, workspace_manager, "llm-proxy")
        
        # Start the worker itself (background tasks)
        await worker.start_worker()
        
        # Get and register the service
        service = worker.get_service_api()
        result = await server.register_service(service)
        
        logger.info(f"LLM proxy worker registered successfully with id: {result.id}, supported_types: {service.get('supported_types')}")
        print(f"LLM PROXY REGISTERED: {result.id}", file=sys.stderr)
        return service
    except Exception as e:
        logger.error(f"Failed to register LLM proxy worker: {e}")
        print(f"LLM PROXY ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        raise