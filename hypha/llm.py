"""LLM Routing Middleware for Hypha - routes requests to LLM proxy services using ASGI."""

import logging
from typing import Optional
from hypha.core import UserInfo, UserPermission, ScopeInfo
from hypha.core.auth import parse_auth_token

logger = logging.getLogger(__name__)


class LLMRoutingMiddleware:
    """
    Middleware that routes LLM requests to the appropriate LLM proxy service using ASGI.
    
    This middleware intercepts requests matching the pattern:
    /<workspace>/llm/<service_id>/...
    
    And dynamically routes them to the LLM proxy service's ASGI handler.
    Similar to MCP proxy pattern, this enables horizontal scaling without mounting.
    """

    def __init__(self, app, base_path: str = "", store=None):
        """
        Initialize the LLM routing middleware.
        
        Args:
            app: The ASGI application to wrap
            base_path: Base path prefix (if any)
            store: The Hypha store for accessing services
        """
        self.app = app
        self.base_path = base_path
        self.store = store
        logger.info("LLM routing middleware initialized for dynamic ASGI routing")

    async def __call__(self, scope, receive, send):
        """
        Process incoming requests and dynamically route LLM requests via ASGI.
        
        Routes requests matching /<workspace>/llm/<service_id>/... to LLM proxy services.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        original_path = path
        
        # Debug logging - log ALL HTTP requests to see what the middleware sees
        if "/llm/" in path:
            logger.warning(f"LLM Middleware sees request: {path}")
        
        # Remove base path if present
        if self.base_path and path.startswith(self.base_path):
            path = path[len(self.base_path):]
        
        # Check if this is an LLM proxy request
        # Pattern: /<workspace>/llm/<service_id>/...
        path_parts = path.strip("/").split("/")
        
        # Log LLM requests for debugging
        if len(path_parts) >= 2 and path_parts[1] == "llm":
            logger.info(f"LLM Middleware: Detected LLM request - original: {original_path}, processed: {path}, parts: {path_parts}")
        
        if len(path_parts) >= 3 and path_parts[1] == "llm":
            workspace = path_parts[0]
            # URL decode the service_id since it may contain : and @ characters
            import urllib.parse
            service_id = urllib.parse.unquote(path_parts[2])
            
            # Get the remaining path after the service_id
            if len(path_parts) > 3:
                remaining_path = "/" + "/".join(path_parts[3:])
            else:
                remaining_path = "/"
            
            logger.debug(f"LLM request detected: workspace={workspace}, service_id={service_id}, path={remaining_path}")
            
            # Extract user token from authorization header for workspace access control
            user_token = None
            for header_name, header_value in scope.get("headers", []):
                if header_name == b"authorization":
                    auth_header = header_value.decode()
                    if auth_header.startswith("Bearer "):
                        user_token = auth_header[7:]  # Remove "Bearer " prefix
                    break
            
            try:
                # Parse user token to get user info for context
                parsed_user_info = None
                if user_token:
                    try:
                        parsed_user_info = await parse_auth_token(user_token, expected_workspace=workspace)
                        parsed_user_info.current_workspace = workspace
                    except Exception as e:
                        logger.warning(f"Failed to parse user token: {e}")
                        # Continue without user context for now
                        pass
                
                # Modify the scope to remove the prefix and route to the LLM service
                modified_scope = scope.copy()
                modified_scope["path"] = remaining_path
                modified_scope["raw_path"] = remaining_path.encode()
                
                # Add custom headers for the LLM proxy to identify the workspace and user
                if "headers" not in modified_scope:
                    modified_scope["headers"] = []
                
                # Add workspace header
                modified_scope["headers"].append((b"x-hypha-workspace", workspace.encode()))
                
                # Dynamically forward the request to the LLM service's ASGI handler
                logger.debug(f"Dynamically routing LLM request to service {service_id} via ASGI")
                
                # The LLM proxy serve method expects args dict with scope, receive, send
                args = {
                    "scope": modified_scope,
                    "receive": receive,
                    "send": send
                }
                
                # Create context for the service call if we have user info
                service_context = None
                if parsed_user_info:
                    service_context = {
                        "user": parsed_user_info.model_dump(),
                        "ws": workspace,
                        "from": f"{workspace}/middleware"
                    }
                
                # Get the ASGI handler and call it within the same context to prevent destruction
                await self._call_llm_asgi_handler(workspace, service_id, user_token, args, service_context)
                return
                
            except PermissionError as e:
                # User doesn't have access to this workspace/service
                logger.warning(f"Access denied for LLM service {service_id} in workspace {workspace}: {e}")
                await send({
                    'type': 'http.response.start',
                    'status': 403,
                    'headers': [(b'content-type', b'application/json')],
                })
                await send({
                    'type': 'http.response.body',
                    'body': b'{"error": "Access denied"}',
                })
                return
                
            except ValueError as e:
                # Service not found
                logger.warning(f"LLM service {service_id} not found in workspace {workspace}: {e}")
                await send({
                    'type': 'http.response.start',
                    'status': 404,
                    'headers': [(b'content-type', b'application/json')],
                })
                await send({
                    'type': 'http.response.body',
                    'body': b'{"error": "LLM service not found"}',
                })
                return
                
            except Exception as connection_error:
                # Check if it's a connection error that we should handle specially
                error_msg = str(connection_error)
                if ("Connection has already been closed" in error_msg or 
                    "Failed to send the request" in error_msg):
                    
                    logger.warning(f"Connection error when calling LLM service {service_id}: {connection_error}")
                    
                    # Return a 503 Service Unavailable for connection issues
                    await send({
                        'type': 'http.response.start',
                        'status': 503,
                        'headers': [(b'content-type', b'application/json')],
                    })
                    await send({
                        'type': 'http.response.body',
                        'body': b'{"error": "LLM service temporarily unavailable"}',
                    })
                    return
                
                # Re-raise other exceptions to be handled by the existing error handlers
                raise
        
        # Not an LLM request, pass through to the main app
        await self.app(scope, receive, send)
    
    async def _call_llm_asgi_handler(self, workspace: str, service_id: str, user_token: str, args: dict, service_context: dict):
        """
        Call the LLM service's ASGI handler within the workspace context to prevent destruction.
        
        This ensures the service remains valid during the entire ASGI call.
        """
        if not self.store:
            raise RuntimeError("Store is not configured in LLMRoutingMiddleware")

        # If user token is provided, validate and extract user info
        user_info = None
        if user_token:
            try:
                # Parse the JWT token to get user info (validates the workspace too)
                user_info = await parse_auth_token(user_token, expected_workspace=workspace)
                user_info.current_workspace = workspace
            except Exception as e:
                logger.warning(f"Failed to parse user token: {e}")
                raise PermissionError(f"Invalid authentication token")
        
        # If no valid user info, create a system user (for backward compatibility)
        if not user_info:
            # Create scope with admin permissions for the target workspace
            scope_info = ScopeInfo(
                current_workspace=workspace,
                workspaces={workspace: UserPermission.admin},  # Admin permissions for the workspace
                client_id="system-llm-middleware"
            )
            
            user_info = UserInfo(
                id="system",
                roles=["admin"],
                is_anonymous=False,
                email="system@hypha.ai",
                parent=None,
                expires_at=None,
                current_workspace=workspace,
                scope=scope_info
            )

        # Get workspace interface and call the service within the same context
        async with self.store.get_workspace_interface(
            user_info, workspace, silent=True
        ) as workspace_manager:
            # Look for the LLM service in the workspace
            services = await workspace_manager.list_services()

            for svc in services:
                svc_id = svc.get("id", "")

                # Check if this service matches our pattern
                # The service could be registered with full session path like:
                # "ws-user-user-1/_rapp_xyz:llm-abc123@app_id"
                if ":" in svc_id:
                    # Extract the service name part after the colon
                    parts = svc_id.split(":", 1)
                    if len(parts) >= 2:
                        service_name = parts[1]
                        # Remove app_id suffix if present (after @)
                        if "@" in service_name:
                            service_name = service_name.split("@")[0]

                        # Check if this matches our service_id
                        if service_name == service_id:
                            # Get the full service object and call it immediately
                            service = await workspace_manager.get_service(svc_id)
                            if service and hasattr(service, "serve"):
                                await service.serve(args, context=service_context)
                                return
                            elif service:
                                raise ValueError(f"LLM service '{svc_id}' found but has no serve method")
                            else:
                                raise ValueError(f"LLM service '{svc_id}' found in listing but could not be retrieved")

                # Also check direct match for backward compatibility
                if svc_id == service_id:
                    service = await workspace_manager.get_service(svc_id)
                    if service and hasattr(service, "serve"):
                        await service.serve(args, context=service_context)
                        return
                    elif service:
                        raise ValueError(f"LLM service '{svc_id}' found but has no serve method")
                    else:
                        raise ValueError(f"LLM service '{svc_id}' found in listing but could not be retrieved")

            # If we get here, no matching service was found
            available_llm_services = [
                svc.get("id", "") for svc in services 
                if "llm-" in svc.get("id", "") or ":llm-" in svc.get("id", "")
            ]
            
            # If the user has permissions but no services are found, it's a 404
            # If the user doesn't have permissions, it would have failed earlier with PermissionError
            if len(available_llm_services) == 0:
                # No LLM services visible to this user in this workspace - likely access denied
                raise PermissionError(f"Access denied to LLM services in workspace '{workspace}'")
            else:
                # Service exists but specific service not found
                raise ValueError(
                    f"LLM service '{service_id}' not found in workspace '{workspace}'. "
                    f"Available LLM services: {available_llm_services}"
                )
    
    async def _get_llm_asgi_handler(self, workspace: str, service_id: str, user_token: str = None):
        """
        Dynamically get the ASGI handler for the LLM service.
        
        This retrieves the serve method from the registered LLM service,
        which wraps the litellm FastAPI app for ASGI handling.

        Args:
            workspace: The workspace name
            service_id: The LLM service ID (format: llm-<session_id>)
            user_token: Optional JWT token for user authentication and permission checking

        Returns:
            The ASGI handler function

        Raises:
            RuntimeError: If store is not configured
            PermissionError: If user doesn't have access to the workspace/service
            ValueError: If LLM service is not found or doesn't have serve method
        """
        if not self.store:
            raise RuntimeError("Store is not configured in LLMRoutingMiddleware")

        # If user token is provided, validate and extract user info
        user_info = None
        if user_token:
            try:
                # Parse the JWT token to get user info (validates the workspace too)
                user_info = await parse_auth_token(user_token, expected_workspace=workspace)
                user_info.current_workspace = workspace
            except Exception as e:
                logger.warning(f"Failed to parse user token: {e}")
                raise PermissionError(f"Invalid authentication token")
        
        # If no valid user info, create a system user (for backward compatibility)
        if not user_info:
            
            
            # Create scope with admin permissions for the target workspace
            scope_info = ScopeInfo(
                current_workspace=workspace,
                workspaces={workspace: UserPermission.admin},  # Admin permissions for the workspace
                client_id="system-llm-middleware"
            )
            
            user_info = UserInfo(
                id="system",
                roles=["admin"],
                is_anonymous=False,
                email="system@hypha.ai",
                parent=None,
                expires_at=None,
                current_workspace=workspace,
                scope=scope_info
            )

        # Get workspace interface to access services
        # Note that `serve` method will be destroyed when exiting this context
        async with self.store.get_workspace_interface(
            user_info, workspace, silent=True
        ) as workspace_manager:
            # Look for the LLM service in the workspace
            services = await workspace_manager.list_services()

            for svc in services:
                svc_id = svc.get("id", "")

                # Check if this service matches our pattern
                # The service could be registered with full session path like:
                # "ws-user-user-1/_rapp_xyz:llm-abc123@app_id"
                if ":" in svc_id:
                    # Extract the service name part after the colon
                    parts = svc_id.split(":", 1)
                    if len(parts) >= 2:
                        service_name = parts[1]
                        # Remove app_id suffix if present (after @)
                        if "@" in service_name:
                            service_name = service_name.split("@")[0]

                        # Check if this matches our service_id
                        if service_name == service_id:
                            # Get the full service object
                            service = await workspace_manager.get_service(svc_id)
                            if service and hasattr(service, "serve"):
                                return service.serve
                            elif service:
                                raise ValueError(f"LLM service '{svc_id}' found but has no serve method")
                            else:
                                raise ValueError(f"LLM service '{svc_id}' found in listing but could not be retrieved")

                # Also check direct match for backward compatibility
                if svc_id == service_id:
                    service = await workspace_manager.get_service(svc_id)
                    if service and hasattr(service, "serve"):
                        return service.serve
                    elif service:
                        raise ValueError(f"LLM service '{svc_id}' found but has no serve method")
                    else:
                        raise ValueError(f"LLM service '{svc_id}' found in listing but could not be retrieved")

            # If we get here, no matching service was found
            available_llm_services = [
                svc.get("id", "") for svc in services 
                if "llm-" in svc.get("id", "") or ":llm-" in svc.get("id", "")
            ]
            
            # If the user has permissions but no services are found, it's a 404
            # If the user doesn't have permissions, it would have failed earlier with PermissionError
            if len(available_llm_services) == 0:
                # No LLM services visible to this user in this workspace - likely access denied
                raise PermissionError(f"Access denied to LLM services in workspace '{workspace}'")
            else:
                # Service exists but specific service not found
                raise ValueError(
                    f"LLM service '{service_id}' not found in workspace '{workspace}'. "
                    f"Available LLM services: {available_llm_services}"
                )
