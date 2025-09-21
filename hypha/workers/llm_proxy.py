"""LLM Proxy Worker for Hypha - integrates litellm for multi-provider LLM support."""

import asyncio
import logging
import traceback
import uuid
from typing import Any, Dict, List, Optional


from fastapi import FastAPI
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

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
        from hypha.litellm.proxy import proxy_server

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
        # Try to get the session by the provided ID (could be either session_id or full_client_id)
        session = self._sessions.get(session_id)

        # Handle redirect references
        if session and "redirect_to" in session:
            # This is a redirect, get the actual session
            actual_session_id = session["redirect_to"]
            session = self._sessions.get(actual_session_id)
            session_id = actual_session_id

        if not session:
            logger.warning(f"_cleanup_session: Session {session_id} not found for cleanup")
            return

        # Find all keys that reference this same session object or redirect to it
        keys_to_remove = []
        for key, sess in list(self._sessions.items()):
            if sess is session:  # Same object
                keys_to_remove.append(key)
            elif isinstance(sess, dict) and sess.get("redirect_to") == session_id:
                keys_to_remove.append(key)

        # Remove all references to this session
        for key in keys_to_remove:
            del self._sessions[key]
            logger.info(f"Removed session reference with key: {key}")
        
        # Disconnect the client if it exists
        client = session.get("client")
        if client:
            try:
                # Unregister the LLM service before disconnecting
                if "registered_service_id" in session:
                    try:
                        service_id_to_unregister = session["registered_service_id"]
                        logger.info(f"_cleanup_session: Attempting to unregister service: {service_id_to_unregister}")
                        await client.unregister_service(service_id_to_unregister)
                        logger.info(f"_cleanup_session: Successfully unregistered service: {service_id_to_unregister}")
                    except Exception as e:
                        # Service may already be gone if client disconnected
                        # This is expected behavior, not an error
                        if "not found" in str(e).lower() or "Service not found" in str(e):
                            logger.debug(f"_cleanup_session: Service {session.get('registered_service_id')} already cleaned up (expected if client disconnected)")
                        else:
                            # Only log as warning if it's not a "not found" issue
                            logger.warning(f"_cleanup_session: Failed to unregister service {session.get('registered_service_id')}: {e}")
                
                # Disconnect the client - this should also trigger cleanup
                logger.info(f"_cleanup_session: Disconnecting client for session {session_id}")
                await client.disconnect()
                logger.info(f"_cleanup_session: Disconnected client for session {session_id}")
            except Exception as e:
                logger.error(f"_cleanup_session: Failed to disconnect client: {e}")
                import traceback
                logger.error(f"_cleanup_session: Disconnect traceback: {traceback.format_exc()}")
        else:
            logger.warning(f"_cleanup_session: Session {session_id} has no client for cleanup")
        
        # Clean up router if exists
        router = session.get("router")
        if router:
            try:
                # Call reset to clean up callbacks and other resources
                if hasattr(router, 'reset'):
                    router.reset()
                    logger.debug(f"_cleanup_session: Reset router for session {session_id}")
                # If router has a close method, call it
                if hasattr(router, 'close'):
                    await router.close()
                    logger.debug(f"_cleanup_session: Closed router for session {session_id}")
                # Clear the router's internal caches if they exist
                if hasattr(router, 'cache'):
                    if hasattr(router.cache, 'clear'):
                        router.cache.clear()
                if hasattr(router, 'deployment_latency_map'):
                    router.deployment_latency_map.clear()
                if hasattr(router, 'model_list'):
                    router.model_list = []
            except Exception as e:
                logger.warning(f"_cleanup_session: Failed to cleanup router: {e}")

        # Clean up the app object if it exists
        app = session.get("app")
        if app:
            # Clear any app-specific state
            session["app"] = None

        # Clear session data to help with garbage collection
        # First set large objects to None to break references
        session["router"] = None
        session["client"] = None
        session["app"] = None
        session["model_list"] = None
        session["litellm_settings"] = None

        # Then clear the entire session
        session.clear()

        logger.info(f"_cleanup_session: Cleaned up session {session_id}")

    def _get_actual_session(self, session_id: str):
        """Get the actual session, following redirects if necessary."""
        session = self._sessions.get(session_id)
        if session and "redirect_to" in session:
            # This is a redirect, get the actual session
            return self._sessions.get(session["redirect_to"])
        return session

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
        session = self._get_actual_session(session_id)
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
        session = self._get_actual_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        model_list = session["model_list"]
        litellm_settings = session.get("litellm_settings", {})
        
        # Ensure each model has an API key to prevent initialization errors
        # For models with mock_response, use a dummy key
        for model in model_list:
            if "litellm_params" in model:
                params = model["litellm_params"]
                # If there's a mock_response and no api_key, add a dummy one
                if "mock_response" in params and not params.get("api_key"):
                    params["api_key"] = "dummy-key-for-mock"
        
        # Create a Router with our model configuration
        # The Router handles load balancing and model selection
        from hypha.litellm.router import Router
        router = Router(
            model_list=model_list,
            routing_strategy=litellm_settings.get("routing_strategy", "simple-shuffle"),
            set_verbose=litellm_settings.get("debug", False),
            debug_level=litellm_settings.get("debug_level", "INFO"),
        )
        
        # Store session-specific configuration instead of setting global variables
        session["router"] = router
        session["litellm_settings"] = litellm_settings
        
        # Get the app - this is the FastAPI app with all routes already configured
        # We'll handle session-specific routing in the serve function
        from hypha.litellm.proxy import proxy_server
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
        manifest_session_id = config.get("manifest", {}).get("session_id")
        logger.info(f"Manifest session_id: {manifest_session_id}")
        
        # Get the full client ID for indexing
        full_client_id = config.get("id", f"{context['ws']}/{config.get('client_id', 'unknown')}")
        
        # Check if we have a compiled session with this ID
        session = None
        session_id = manifest_session_id
        
        if manifest_session_id:
            session = self._sessions.get(manifest_session_id)
            logger.info(f"Session lookup for {manifest_session_id}: {'found' if session else 'not found'}")
            
            # If we found a session, also index it by full_client_id for cleanup
            if session:
                session["full_client_id"] = full_client_id
                # Store a mapping instead of duplicating the session
                if full_client_id != manifest_session_id:
                    self._sessions[full_client_id] = {"redirect_to": manifest_session_id}
                logger.info(f"Added full_client_id index for existing session: {full_client_id}")
            
        if not session:
            # No compiled session found (or no session_id in manifest)
            # Generate a new session_id for this start
            session_id = str(uuid.uuid4())
            logger.info(f"Creating new session {session_id} (manifest had: {manifest_session_id})")
            
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
            
            # Get the full client ID from config for cleanup purposes
            full_client_id = config.get("id", f"{context['ws']}/{config.get('client_id', session_id)}")
            
            self._sessions[session_id] = {
                "info": session_info,
                "model_list": model_list,
                "litellm_settings": litellm_settings,
                "workspace": context["ws"],
                "user": user_info.id,
                "last_access": asyncio.get_event_loop().time(),
                "logs": [f"LLM proxy session created with ID {session_id} from manifest"],
                "full_client_id": full_client_id,  # Store for cleanup
            }
            session = self._sessions[session_id]
            
            # Store a mapping from full_client_id to session_id instead of duplicating the session
            # This avoids creating multiple references to the same object
            if full_client_id != session_id:
                # Create a lightweight reference that points to the actual session_id
                self._sessions[full_client_id] = {"redirect_to": session_id}
            
            logger.info(f"Created session {session_id} from manifest with {len(model_list)} models, indexed as {full_client_id}")
        
        session["last_access"] = asyncio.get_event_loop().time()
        
        # Now start the LLM proxy service
        logger.info(f"Starting LLM proxy service for session {session_id}")
        try:
            # Get service_id from manifest config, or use default pattern
            manifest = config.get("manifest", {})
            service_id = manifest.get("config", {}).get("service_id", f"llm-{session_id}")
            session["service_id"] = service_id
            logger.info(f"Will register service with ID: {service_id}")
            
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
            
            # Get the connection info from config
            client_id = config.get("client_id", "")
            app_id = config.get("app_id", "")
            workspace = config.get("workspace", context["ws"])
            server_url = config.get("server_url", "")
            token = config.get("token", "")
            
            # Store the full client ID for later cleanup
            full_client_id = config.get("id", f"{workspace}/{client_id}")
            session["full_client_id"] = full_client_id
            
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
            
            # Create session-specific ASGI serve function that uses this session's router
            def create_session_serve_function(session_id: str):
                async def serve_llm(args, context=None):
                    """ASGI handler for the LLM proxy service with session-specific routing."""
                    if context:
                        logger.debug(f'LLM proxy: {context["user"]["id"]} - {args["scope"]["method"]} - {args["scope"]["path"]}')
                    
                    # Get session-specific configuration
                    current_session = self._get_actual_session(session_id)
                    if not current_session:
                        # Session not found - return 404
                        scope = args["scope"]
                        receive = args["receive"]
                        send = args["send"]
                        
                        await send({
                            'type': 'http.response.start',
                            'status': 404,
                            'headers': [(b'content-type', b'application/json')],
                        })
                        await send({
                            'type': 'http.response.body',
                            'body': b'{"error": "Session not found"}',
                        })
                        return
                    
                    # Temporarily set global variables for this request
                    # This is a workaround since litellm uses global state
                    from hypha.litellm.proxy import proxy_server
                    original_router = getattr(proxy_server, 'llm_router', None)
                    original_model_list = getattr(proxy_server, 'llm_model_list', None)
                    original_settings = getattr(proxy_server, 'general_settings', None)
                    
                    try:
                        # Set session-specific configuration
                        proxy_server.llm_router = current_session.get("router")
                        proxy_server.llm_model_list = current_session.get("model_list", [])
                        proxy_server.general_settings = current_session.get("litellm_settings", {})
                        
                        # Call the actual app
                        await app(args["scope"], args["receive"], args["send"])
                        
                    finally:
                        # Restore original global state
                        if original_router is not None:
                            proxy_server.llm_router = original_router
                        if original_model_list is not None:
                            proxy_server.llm_model_list = original_model_list  
                        if original_settings is not None:
                            proxy_server.general_settings = original_settings
                
                return serve_llm
            
            # Create the session-specific serve function
            serve_llm = create_session_serve_function(session_id)
            
            # Check if we already registered this service in a previous start
            if "registered_service_id" in session and session.get("client"):
                # Service already registered, just return the session_id
                logger.info(f"Service {service_id} already registered as {session['registered_service_id']}, skipping registration")
                return session_id
            
            # Register the LLM service as an ASGI service (type="asgi")
            logger.info(f"Registering service {service_id} for app {app_id}")
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
            logger.info(f"Successfully registered LLM proxy service with full ID: {full_service_id}")
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
            
            # Return the actual session_id we're using (not the manifest one)
            # This is important for restart scenarios where we create a new session
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
        logger.info(f"Stopping LLM proxy session {session_id}")
        
        # Check if session exists in memory
        session = self._get_actual_session(session_id)
        if session:
            
            # Try to unregister the service first if we have a client
            client = session.get("client")
            if client and "registered_service_id" in session:
                try:
                    service_id_to_unregister = session["registered_service_id"]
                    logger.info(f"Unregistering service before cleanup: {service_id_to_unregister}")
                    await client.unregister_service(service_id_to_unregister)
                    logger.info(f"Successfully unregistered service: {service_id_to_unregister}")
                except Exception as e:
                    # Service may already be gone if client disconnected
                    # This is expected behavior, not an error
                    if "not found" in str(e).lower() or "Service not found" in str(e):
                        logger.debug(f"Service {session.get('registered_service_id')} already cleaned up (expected if client disconnected)")
                    else:
                        # Only log as error if it's not a "not found" issue
                        logger.warning(f"Failed to unregister service {session.get('registered_service_id')}: {e}")
            else:
                logger.warning(f"Session {session_id} has no client or registered_service_id for cleanup")
                logger.warning(f"Session keys: {list(session.keys())}")
            
            # Now cleanup the session
            await self._cleanup_session(session_id)
        else:
            # Session not in memory - still try to clean up any registered services
            logger.warning(f"LLM proxy session {session_id} not found in memory, attempting service cleanup")
            
            # Try to connect and clean up any services registered by this session
            if context and context.get("ws"):
                try:
                    from hypha_rpc import connect_to_server
                    # Extract workspace and client_id from session_id
                    if "/" in session_id:
                        workspace, client_id = session_id.split("/", 1)
                    else:
                        workspace = context.get("ws")
                        client_id = session_id
                    
                    # We need to clean up any services that might have been registered
                    # Since we don't have the session data, we can't unregister specific services
                    # But the workspace manager should clean them up when the client disconnects
                    logger.info(f"Session {session_id} not in memory, relying on client disconnection for cleanup")
                except Exception as e:
                    logger.warning(f"Error during cleanup attempt for {session_id}: {e}")
    
    def get_app_for_service(self, service_id: str):
        """Get the FastAPI app for a given service ID."""
        # Look for the session that has this service_id
        for session_id, session in self._sessions.items():
            # Skip redirect entries
            if "redirect_to" in session:
                continue
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
        # Check if litellm dependencies are installed
        try:
            import openai
            import tiktoken
        except ImportError as e:
            logger.warning(f"LiteLLM dependencies not installed: {e}. Skipping LLM proxy worker registration.")
            logger.warning("To use LLM proxy, install with: pip install hypha[llm-proxy]")
            print(f"LLM PROXY SKIPPED: Missing dependencies - {e}", file=sys.stderr)
            return None

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