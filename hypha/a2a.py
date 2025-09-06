"""A2A (Agent-to-Agent) service support for Hypha.

This module provides integration with the A2A protocol using the official a2a-sdk.
It allows registering A2A agents as Hypha services and automatically exposes them
via HTTP endpoints with the proper A2A protocol handling.
"""

import inspect
import logging
import asyncio
from typing import Any, Dict, Optional, Callable
from starlette.routing import Route
from starlette.types import ASGIApp
from starlette.routing import Match
from starlette.requests import Request
import json

logger = logging.getLogger(__name__)


from a2a.types import AgentCard
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
    AgentExecutor,
    RequestContext,
    EventQueue,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import Message
from a2a.utils import new_agent_text_message

class HyphaAgentExecutor(AgentExecutor):
    """AgentExecutor that adapts Hypha services to A2A SDK interface."""

    def __init__(self, run_function: Callable):
        """Initialize with Hypha service's run function.

        Args:
            run_function: The service's run function that will process messages
        """
        self.run_function = run_function
        logger.info(
            f"HyphaAgentExecutor initialized with run_function: {type(run_function)}"
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the agent following A2A SDK pattern.

        Args:
            context: RequestContext containing the user's message
            event_queue: EventQueue to send responses back to the client
        """
        try:
            logger.info(
                f"HyphaAgentExecutor.execute called with context: {type(context)}"
            )

            # Extract message from context
            message = context.message
            logger.info(f"Extracted message from context: {type(message)}")

            # Convert A2A message to Hypha format
            message_dict = self._convert_a2a_message_to_hypha_format(message)
            logger.info(f"Converted message to Hypha format: {message_dict}")

            # Filter context to only serializable items (avoid EventQueue issues)
            filtered_context = self._filter_context_for_hypha(context)

            # Call the Hypha service's run function
            logger.info(f"Calling run function: {type(self.run_function)}")

            if inspect.iscoroutinefunction(self.run_function):
                result = await self.run_function(message_dict, filtered_context)
            else:
                result = self.run_function(message_dict, filtered_context)

                # Handle case where sync function returns a Future (Hypha wrapped async functions)
                if asyncio.isfuture(result) or inspect.iscoroutine(result):
                    result = await result

            logger.info(f"Run function returned: {result}")

            # Convert result to A2A message and send back
            response_message = new_agent_text_message(str(result))
            await event_queue.enqueue_event(response_message)

            logger.info("Response sent successfully via event queue")

        except Exception as e:
            logger.exception(f"Error in HyphaAgentExecutor.execute: {e}")
            # Send error message back to client
            error_message = new_agent_text_message(f"Error: {str(e)}")
            await event_queue.enqueue_event(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel handler - not implemented for Hypha services."""
        logger.warning("Cancel not supported for Hypha services")
        error_message = new_agent_text_message("Cancel operation not supported")
        await event_queue.enqueue_event(error_message)

    def _convert_a2a_message_to_hypha_format(self, message: Message) -> Dict[str, Any]:
        """Convert A2A SDK Message to Hypha format."""
        try:
            # Extract text from A2A message parts
            parts = []
            for part in message.parts:
                if hasattr(part, "root"):
                    # A2A SDK Part object with root containing the actual part data
                    root = part.root
                    if hasattr(root, "kind") and hasattr(root, "text"):
                        parts.append({"kind": root.kind, "text": root.text})
                    elif hasattr(root, "text"):
                        # Fallback for simpler text parts
                        parts.append({"kind": "text", "text": root.text})
                    elif hasattr(root, "kind") and hasattr(root, "data"):
                        parts.append({"kind": root.kind, "data": root.data})
                    else:
                        # Fallback: convert to text
                        parts.append({"kind": "text", "text": str(root)})
                elif hasattr(part, "text"):
                    # Direct text attribute
                    parts.append({"kind": "text", "text": part.text})
                elif hasattr(part, "kind") and hasattr(part, "text"):
                    # Part with kind and text attributes
                    parts.append({"kind": part.kind, "text": part.text})
                else:
                    # Fallback: convert to string
                    parts.append({"kind": "text", "text": str(part)})

            return {
                "parts": parts,
                "role": message.role,
                "messageId": message.message_id,
            }
        except Exception as e:
            logger.exception(f"Error converting A2A message: {e}")
            # Fallback: create basic message
            return {
                "parts": [{"kind": "text", "text": "Error processing message"}],
                "role": "user",
                "messageId": "error",
            }

    def _filter_context_for_hypha(self, context: RequestContext) -> Optional[Dict]:
        """Filter RequestContext to only include serializable data for Hypha."""
        try:
            # For now, return None to avoid serialization issues
            # In the future, we could extract specific fields from context
            return None
        except Exception as e:
            logger.exception(f"Error filtering context: {e}")
            return None


async def create_a2a_app_from_service(service, service_info):
    """Create A2AStarletteApplication from a Hypha service.

    This function follows the A2A samples repository pattern:
    1. Extract agent_card from service
    2. Create HyphaAgentExecutor from service's run function
    3. Create DefaultRequestHandler with InMemoryTaskStore
    4. Create A2AStarletteApplication with agent card and handler
    5. Return server.build() as ASGI app

    Args:
        service: Hypha service object with agent_card and run function
        service_info: Service information from Hypha

    Returns:
        ASGI application that can be called with (scope, receive, send)
    """

    logger.info(
        f"create_a2a_app_from_service called with service type: {type(service)}"
    )

    # Extract run function
    if "run" in service:
        run_function = service["run"]
        logger.info("Extracted 'run' function from service")
    else:
        raise ValueError("Service does not have a 'run' function")

    # Extract or create agent card
    if "agent_card" in service:
        agent_card_data = service["agent_card"]
        logger.info("Found agent_card in service")

        # Handle callable agent_card (function that returns card data)
        if callable(agent_card_data):
            logger.info("agent_card is callable, calling it...")
            if inspect.iscoroutinefunction(agent_card_data):
                agent_card_data = await agent_card_data()
            else:
                agent_card_data = agent_card_data()
                # Handle case where sync function returns a Future (Hypha wrapped async functions)
                if asyncio.isfuture(agent_card_data) or inspect.iscoroutine(
                    agent_card_data
                ):
                    agent_card_data = await agent_card_data
            logger.info("agent_card function called successfully")

    elif "extended_agent_card" in service:
        agent_card_data = service["extended_agent_card"]
        logger.info("Found extended_agent_card in service, using as agent_card")

        # Handle callable extended_agent_card
        if callable(agent_card_data):
            logger.info("extended_agent_card is callable, calling it...")
            if inspect.iscoroutinefunction(agent_card_data):
                agent_card_data = await agent_card_data()
            else:
                agent_card_data = agent_card_data()
                # Handle case where sync function returns a Future (Hypha wrapped async functions)
                if asyncio.isfuture(agent_card_data) or inspect.iscoroutine(
                    agent_card_data
                ):
                    agent_card_data = await agent_card_data
            logger.info("extended_agent_card function called successfully")
    else:
        # Create default agent card from service info
        agent_card_data = {
            "protocol_version": "0.3.0",
            "name": service_info.id,
            "description": f"A2A agent for Hypha service {service_info.id}",
            "url": f"http://localhost:9876/a2a/{service_info.id}",  # Will be updated by middleware
            "version": "1.0.0",
            "capabilities": {
                "streaming": True,
                "push_notifications": False,
                "state_transition_history": False,
            },
                            "default_input_modes": ["text"],
            "default_output_modes": ["text"],
            "skills": [
                {
                    "id": "chat",
                    "name": "Chat",
                    "description": "Chat with the agent",
                    "tags": ["chat"],
                    "examples": ["hello", "help"],
                }
            ],
        }
        logger.info("Created default agent_card from service info")

    # Ensure required fields are present
    if "capabilities" not in agent_card_data:
        agent_card_data["capabilities"] = {
            "streaming": True,
            "push_notifications": False,
            "state_transition_history": False,
        }

        if "default_input_modes" not in agent_card_data:
            agent_card_data["default_input_modes"] = ["text"]

        if "default_output_modes" not in agent_card_data:
            agent_card_data["default_output_modes"] = ["text"]

    if "skills" not in agent_card_data:
        agent_card_data["skills"] = [
            {
                "id": "chat",
                "name": "Chat",
                "description": "Chat with the agent",
                "tags": ["chat"],
                "examples": ["hello", "help"],
            }
        ]

    # Ensure skills have required tags field
    for skill in agent_card_data["skills"]:
        if "tags" not in skill:
            skill["tags"] = ["chat"]

    logger.info("Creating HyphaAgentExecutor...")
    try:
        agent_executor = HyphaAgentExecutor(run_function)
        logger.info("HyphaAgentExecutor created successfully")
    except Exception as e:
        logger.error(f"Failed to create HyphaAgentExecutor: {e}")
        raise

    logger.info("Creating A2A AgentCard...")
    try:
        agent_card = AgentCard(**agent_card_data)
        logger.info("AgentCard created successfully")
    except Exception as e:
        logger.error(f"Failed to create AgentCard: {e}")
        logger.error(f"Agent card data: {agent_card_data}")
        raise

    logger.info("Creating DefaultRequestHandler...")
    try:
        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor, task_store=task_store
        )
        logger.info("DefaultRequestHandler created successfully")
    except Exception as e:
        logger.error(f"Failed to create DefaultRequestHandler: {e}")
        raise

    logger.info("Creating A2AStarletteApplication...")
    try:
        server_app_builder = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )
        logger.info("A2AStarletteApplication created successfully")
    except Exception as e:
        logger.error(f"Failed to create A2AStarletteApplication: {e}")
        raise

    logger.info("Building final ASGI app...")
    try:
        asgi_app = server_app_builder.build()
        logger.info("ASGI app built successfully")
        return asgi_app
    except Exception as e:
        logger.error(f"Failed to build ASGI app: {e}")
        raise


class A2ARoutingMiddleware:
    """Middleware specifically for routing A2A services."""

    def __init__(self, app: ASGIApp, base_path=None, store=None):
        self.app = app
        self.base_path = base_path.rstrip("/") if base_path else ""
        self.store = store

        # A2A route patterns
        # Main route: /{workspace}/a2a/{service_id}/{path:path}
        route = self.base_path.rstrip("/") + "/{workspace}/a2a/{service_id:path}"
        self.route = Route(route, endpoint=None)
        
        # Info route for service information: /{workspace}/a2a-info/{service_id}
        # Using a2a-info to avoid conflicts with the main route
        info_route = self.base_path.rstrip("/") + "/{workspace}/a2a-info/{service_id}"
        self.info_route = Route(info_route, endpoint=None)
        
        logger.info(f"A2A middleware initialized with route patterns:")
        logger.info(f"  - Main: {self.route.path}")
        logger.info(f"  - Info: {self.info_route.path}")

    def _get_authorization_header(self, scope):
        """Extract the Authorization header from the scope."""
        for key, value in scope.get("headers", []):
            if key.decode("utf-8").lower() == "authorization":
                return value.decode("utf-8")
        return None

    def _get_access_token_from_cookies(self, scope):
        """Extract the access_token cookie from the scope."""
        try:
            cookies = None
            for key, value in scope.get("headers", []):
                if key.decode("utf-8").lower() == "cookie":
                    cookies = value.decode("utf-8")
                    break
            if not cookies:
                return None
            # Split cookies and find the access_token
            cookie_dict = {
                k.strip(): v.strip()
                for k, v in (cookie.split("=") for cookie in cookies.split(";"))
            }
            return cookie_dict.get("access_token")
        except Exception as exp:
            return None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "N/A")
            
            # Check if the current request path matches the info route first
            info_match, info_params = self.info_route.matches(scope)
            if info_match == Match.FULL:
                path_params = info_params.get("path_params", {})
                workspace = path_params["workspace"]
                service_id = path_params["service_id"]
                
                logger.debug(
                    f"A2A Middleware: Info route match - workspace='{workspace}', service_id='{service_id}'"
                )
                
                # Return comprehensive agent information
                await self._send_agent_info(send, workspace, service_id)
                return
            
            # Check if the current request path matches the main A2A route
            match, params = self.route.matches(scope)
            path_params = params.get("path_params", {})
            if match == Match.FULL:
                # Extract workspace and service_id from the matched path
                workspace = path_params["workspace"]
                service_id_path = path_params[
                    "service_id"
                ]  # This contains service_id and optional path
                # Parse service_id and path from service_id_path
                # service_id_path could be just "service_id" or "service_id/additional/path"
                service_id_parts = service_id_path.split("/", 1)
                service_id = service_id_parts[0]
                path = "/" + service_id_parts[1] if len(service_id_parts) > 1 else "/"

                logger.info(
                    f"Processing A2A request - workspace='{workspace}', service_id='{service_id}', path='{path}'"
                )

                try:
                    # Prepare scope for the A2A service
                    scope = {
                        k: scope[k]
                        for k in scope
                        if isinstance(
                            scope[k],
                            (str, int, float, bool, tuple, list, dict, bytes),
                        )
                    }
                    if not path.startswith("/"):
                        path = "/" + path
                    scope["path"] = path
                    scope["raw_path"] = path.encode("latin-1")

                    # Get _mode from query string
                    query = scope.get("query_string", b"").decode("utf-8")
                    if query:
                        _mode = dict([q.split("=") for q in query.split("&")]).get(
                            "_mode"
                        )
                    else:
                        _mode = None

                    # Get authentication info
                    access_token = self._get_access_token_from_cookies(scope)
                    authorization = self._get_authorization_header(scope)

                    # Create a Request object with the scope
                    request = Request(scope, receive=receive)
                    # Login and get user info
                    user_info = await self.store.login_optional(request)

                    # Get the A2A service
                    # For public services, we need to create a workspace interface
                    # that allows accessing services in that workspace
                    # We use the user's current workspace as the base for the interface
                    # but query services from the target workspace
                    async with self.store.get_workspace_interface(
                        user_info, user_info.scope.current_workspace
                    ) as api:
                        try:
                            logger.info(
                                f"A2A Middleware: Looking up service '{service_id}' in workspace '{workspace}' (current workspace: '{user_info.scope.current_workspace}')"
                            )
                            # For cross-workspace service access, use the full service ID
                            full_service_id = f"{workspace}/{service_id}"
                            service_info = await api.get_service_info(
                                full_service_id, {"mode": _mode}
                            )
                            logger.info(
                                f"A2A Middleware: Found service '{service_id}' of type '{service_info.type}'"
                            )
                        except (KeyError, Exception) as e:
                            logger.error(f"A2A Middleware: Service lookup failed: {e}")
                            # Handle both KeyError and RemoteException
                            if "Service not found" in str(e) or "Permission denied" in str(e):
                                # List all services for debugging
                                try:
                                    all_services = await api.list_services()
                                    logger.info(
                                        f"A2A Middleware: Available services: {[s.id for s in all_services]}"
                                    )
                                except Exception as list_e:
                                    logger.error(
                                        f"A2A Middleware: Failed to list services: {list_e}"
                                    )

                                await self._send_error_response(
                                    send, 404, f"Service {service_id} not found or not accessible"
                                )
                                return
                            else:
                                raise  # Re-raise if it's another type of error

                        # Verify it's an A2A service
                        if service_info.type != "a2a":
                            await self._send_error_response(
                                send, 400, f"Service {service_id} is not an A2A service"
                            )
                            return

                        service = await api.get_service(service_info.id)

                        # Create the A2A ASGI app and handle the request
                        await self.handle_a2a_service(
                            service, service_info, scope, receive, send
                        )
                        return

                except Exception as exp:
                    logger.exception(f"Error in A2A service: {exp}")
                    await self._send_error_response(
                        send, 500, f"Internal Server Error: {exp}"
                    )
                    return

        # Continue to next middleware if not an A2A route
        await self.app(scope, receive, send)

    async def handle_a2a_service(self, service, service_info, scope, receive, send):
        """Handle A2A service requests by creating an A2A application."""

        try:

            # Create A2A application from the service
            a2a_app = await create_a2a_app_from_service(service, service_info)

            # Call the A2A application with the request
            await a2a_app(scope, receive, send)

        except Exception as e:
            logger.exception(f"Error handling A2A service: {e}")
            await self._send_error_response(
                send, 500, f"Internal server error: {str(e)}"
            )

    async def _send_agent_info(self, send, workspace, service_id):
        """Send comprehensive A2A agent information including agent card, skills, and capabilities."""
        # Handle empty service_id case
        if not service_id:
            message = {
                "error": "A2A service ID required",
                "message": "No service ID was provided in the URL.",
                "help": "Please specify a service ID. URL format: /{workspace}/a2a/{service_id}",
                "example": f"/{workspace}/a2a/your-service-id"
            }
            status_code = 404
        else:
            try:
                # Get authentication info
                from starlette.requests import Request
                
                # Create a minimal scope for auth check
                scope = {
                    "type": "http",
                    "method": "GET",
                    "headers": [],
                    "query_string": b"",
                }
                
                # Try to get the service and extract A2A capabilities
                try:
                    # Create a mock request for auth
                    request = Request(scope, receive=None, send=None)
                    user_info = await self.store.login_optional(request)
                    
                    # Get service through workspace interface
                    async with self.store.get_workspace_interface(
                        user_info, user_info.scope.current_workspace
                    ) as api:
                        # Build full service ID
                        full_service_id = f"{workspace}/{service_id}"
                        
                        # Try to get service info
                        try:
                            service_info = await api.get_service_info(full_service_id, {})
                            
                            # Verify it's an A2A service
                            if service_info.type != "a2a":
                                message = {
                                    "error": "Not an A2A service",
                                    "message": f"Service '{service_id}' is not an A2A service (type: '{service_info.type}')",
                                    "help": "This endpoint is for A2A services only.",
                                }
                                status_code = 400
                            else:
                                service = await api.get_service(full_service_id)
                                
                                # Extract agent card information
                                agent_card_data = None
                                
                                # Try to get agent_card
                                if "agent_card" in service:
                                    agent_card_data = service["agent_card"]
                                    # Handle callable agent_card
                                    if callable(agent_card_data):
                                        if inspect.iscoroutinefunction(agent_card_data):
                                            agent_card_data = await agent_card_data()
                                        else:
                                            agent_card_data = agent_card_data()
                                            if asyncio.isfuture(agent_card_data) or inspect.iscoroutine(agent_card_data):
                                                agent_card_data = await agent_card_data
                                elif "extended_agent_card" in service:
                                    agent_card_data = service["extended_agent_card"]
                                    # Handle callable extended_agent_card
                                    if callable(agent_card_data):
                                        if inspect.iscoroutinefunction(agent_card_data):
                                            agent_card_data = await agent_card_data()
                                        else:
                                            agent_card_data = agent_card_data()
                                            if asyncio.isfuture(agent_card_data) or inspect.iscoroutine(agent_card_data):
                                                agent_card_data = await agent_card_data
                                
                                # If no agent card found, create default
                                if not agent_card_data:
                                    agent_card_data = {
                                        "protocol_version": "0.3.0",
                                        "name": service_info.id,
                                        "description": f"A2A agent for Hypha service {service_info.id}",
                                        "url": f"/{workspace}/a2a/{service_id}",
                                        "version": "1.0.0",
                                        "capabilities": {
                                            "streaming": True,
                                            "push_notifications": False,
                                            "state_transition_history": False,
                                        },
                                        "default_input_modes": ["text"],
                                        "default_output_modes": ["text"],
                                        "skills": [
                                            {
                                                "id": "chat",
                                                "name": "Chat",
                                                "description": "Chat with the agent",
                                                "tags": ["chat"],
                                                "examples": ["hello", "help"],
                                            }
                                        ],
                                    }
                                
                                # Ensure required fields are present
                                if "capabilities" not in agent_card_data:
                                    agent_card_data["capabilities"] = {
                                        "streaming": True,
                                        "push_notifications": False,
                                        "state_transition_history": False,
                                    }
                                
                                if "default_input_modes" not in agent_card_data:
                                    agent_card_data["default_input_modes"] = ["text"]
                                
                                if "default_output_modes" not in agent_card_data:
                                    agent_card_data["default_output_modes"] = ["text"]
                                
                                if "skills" not in agent_card_data:
                                    agent_card_data["skills"] = [
                                        {
                                            "id": "chat",
                                            "name": "Chat",
                                            "description": "Chat with the agent",
                                            "tags": ["chat"],
                                            "examples": ["hello", "help"],
                                        }
                                    ]
                                
                                # Extract additional service information
                                service_functions = []
                                if "run" in service:
                                    service_functions.append("run")
                                if "agent_card" in service:
                                    service_functions.append("agent_card")
                                if "extended_agent_card" in service:
                                    service_functions.append("extended_agent_card")
                                
                                # Build comprehensive response
                                message = {
                                    "service": {
                                        "id": service_info.id,
                                        "name": getattr(service_info, 'name', service_info.id),
                                        "type": "a2a",
                                        "description": getattr(service_info, 'description', 'A2A agent service'),
                                    },
                                    "endpoints": {
                                        "a2a": f"/{workspace}/a2a/{service_id}",
                                        "info": f"/{workspace}/a2a-info/{service_id}",
                                    },
                                    "agent_card": agent_card_data,
                                    "service_functions": service_functions,
                                    "help": "Use the A2A endpoint for agent communication following the A2A protocol.",
                                }
                                status_code = 200
                            
                        except Exception as e:
                            if "not found" in str(e).lower() or "permission denied" in str(e).lower():
                                # Service not found or not accessible
                                message = {
                                    "error": "Service not found or not accessible",
                                    "message": f"The service '{service_id}' was not found in workspace '{workspace}' or you don't have permission to access it.",
                                    "help": "Ensure the service exists and you have permission to access it.",
                                }
                                status_code = 404
                            else:
                                raise
                            
                except Exception as e:
                    logger.warning(f"Failed to get service info for {service_id}: {e}")
                    # Fallback to basic info without service details
                    message = {
                        "service": {
                            "id": service_id,
                            "workspace": workspace,
                            "status": "unknown",
                        },
                        "endpoints": {
                            "a2a": f"/{workspace}/a2a/{service_id}",
                            "info": f"/{workspace}/a2a-info/{service_id}",
                        },
                        "help": "Use the A2A endpoint for agent communication.",
                        "note": "Could not retrieve detailed service information. The service may not be running or accessible.",
                    }
                    status_code = 200
                    
            except Exception as e:
                logger.error(f"Error getting A2A service info: {e}")
                message = {
                    "error": "Failed to retrieve service information",
                    "message": str(e),
                    "endpoints": {
                        "a2a": f"/{workspace}/a2a/{service_id}",
                        "info": f"/{workspace}/a2a-info/{service_id}",
                    },
                }
                status_code = 500
        
        response_body = json.dumps(message, indent=2).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(response_body)).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_body,
            }
        )

    async def _send_error_response(self, send, status_code, message):
        """Send an error response."""
        response_body = json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(response_body)).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_body,
            }
        )
