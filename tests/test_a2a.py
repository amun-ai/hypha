"""Test A2A (Agent-to-Agent) services."""

import asyncio
import json
import time

import pytest
import httpx
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SERVER_URL

# A2A client imports
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, Message, TextPart, Role
import uuid

def create_text_message(text: str) -> SendMessageRequest:
    """Helper function to create a properly formatted A2A text message request."""
    message = Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[TextPart(kind="text", text=text)],
    )

    params = MessageSendParams(message=message)

    return SendMessageRequest(id=str(uuid.uuid4()), params=params)


# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_a2a_agent_registration(fastapi_server, test_user_token):
    """Test basic A2A agent registration."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define a simple agent card
    agent_card = {
        "protocol_version": "0.3.0",
        "name": "Test Agent",
        "description": "A simple test agent for unit testing",
        "url": f"http://localhost:9527/{workspace}/a2a/test-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "push_notifications": False},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": [
            {
                "id": "echo",
                "name": "Echo Skill",
                "description": "Echoes back the input message",
                "tags": ["test", "echo"],
                "examples": ["echo hello", "repeat this message"],
            }
        ],
    }

    # Define a simple run function
    async def simple_run(message, context=None):
        """Simple echo agent function."""
        if isinstance(message, dict):
            # Handle A2A Message format
            text_parts = [
                part.get("text", "")
                for part in message.get("parts", [])
                if part.get("kind") == "text"
            ]
            text = " ".join(text_parts)
        else:
            text = str(message)

        return f"Echo: {text}"

    # Register A2A service
    service = await api.register_service(
        {
            "id": "test-agent",
            "type": "a2a",
            "config": {
                "visibility": "public",
            },
            "agent_card": agent_card,
            "run": simple_run,
        }
    )

    assert service["type"] == "a2a"
    assert "test-agent" in service["id"]

    await api.disconnect()


async def test_a2a_agent_card_endpoint(fastapi_server, test_user_token):
    """Test A2A agent card HTTP endpoint."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define agent card
    agent_card = {
        "protocol_version": "0.3.0",
        "name": "Card Test Agent",
        "description": "Agent for testing card endpoint",
        "url": f"http://localhost:9527/{workspace}/a2a/card-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": True, "push_notifications": False},
        "default_input_modes": ["text/plain", "application/json"],
        "default_output_modes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "test-skill",
                "name": "Test Skill",
                "description": "A test skill",
                "tags": ["test"],
            }
        ],
    }

    async def agent_run(message, context=None):
        return "Hello from card test agent"

    # Register A2A service
    await api.register_service(
        {
            "id": "card-agent",
            "type": "a2a",
            "config": {"visibility": "public"},
            "agent_card": agent_card,
            "run": agent_run,
        }
    )

    # Test agent card endpoint
    async with httpx.AsyncClient() as client:
        # Test /.well-known/agent-card.json endpoint (A2A standard)
        response = await client.get(
            f"{SERVER_URL}/{workspace}/a2a/card-agent/.well-known/agent-card.json"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        card = response.json()
        assert card["name"] == "Card Test Agent"
        assert card["protocolVersion"] == "0.3.0"
        assert card["url"] == f"http://localhost:9527/{workspace}/a2a/card-agent"
        assert card["capabilities"]["streaming"] is True
        assert len(card["skills"]) == 1
        assert card["skills"][0]["id"] == "test-skill"

    await api.disconnect()


async def test_a2a_message_send(fastapi_server, test_user_token):
    """Test A2A message/send JSON-RPC method."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define agent card
    agent_card = {
        "protocol_version": "0.3.0",
        "name": "Message Test Agent",
        "description": "Agent for testing message sending",
        "url": f"http://localhost:9527/{workspace}/a2a/msg-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "push_notifications": False},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": [
            {
                "id": "respond",
                "name": "Respond Skill",
                "description": "Responds to messages",
                "tags": ["test", "respond"],
            }
        ],
    }

    async def message_agent_run(message, context=None):
        """Agent that processes A2A messages."""
        # Handle A2A Message format
        if isinstance(message, dict) and "parts" in message:
            text_parts = []
            for part in message["parts"]:
                if part.get("kind") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("kind") == "data":
                    text_parts.append(f"Data: {part.get('data')}")

            user_text = " ".join(text_parts)
            return f"Agent received: {user_text}"
        else:
            return f"Agent received: {message}"

    # Register A2A service
    await api.register_service(
        {
            "id": "msg-agent",
            "type": "a2a",
            "config": {"visibility": "public"},
            "agent_card": agent_card,
            "run": message_agent_run,
        }
    )

    # Test message/send JSON-RPC call
    async with httpx.AsyncClient() as client:
        # Prepare A2A message/send request
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello, test agent!"}],
                    "messageId": "test-message-1",
                    "kind": "message",
                }
            },
        }

        response = await client.post(
            f"{SERVER_URL}/{workspace}/a2a/msg-agent",
            json=rpc_request,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        rpc_response = response.json()
        assert rpc_response["jsonrpc"] == "2.0"
        assert rpc_response["id"] == 1
        assert "result" in rpc_response

        # Check if result is a Task or Message
        result = rpc_response["result"]
        if result.get("kind") == "task":
            # It's a Task object
            assert "status" in result
            assert "id" in result
            assert "contextId" in result
        elif result.get("kind") == "message":
            # It's a Message object
            assert "parts" in result
            assert "messageId" in result

    await api.disconnect()


async def test_a2a_task_management(fastapi_server, test_user_token):
    """Test A2A task creation and management."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define agent card for task-based agent
    agent_card = {
        "protocol_version": "0.3.0",
        "name": "Task Test Agent",
        "description": "Agent for testing task management",
        "url": f"http://localhost:9527/{workspace}/a2a/task-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "push_notifications": False},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": [
            {
                "id": "process",
                "name": "Process Skill",
                "description": "Processes requests as tasks",
                "tags": ["test", "task"],
            }
        ],
    }

    async def task_agent_run(message, context=None):
        """Agent that creates tasks for processing."""
        # Simulate some processing
        await asyncio.sleep(0.1)

        if isinstance(message, dict) and "parts" in message:
            text_parts = [
                part.get("text", "")
                for part in message.get("parts", [])
                if part.get("kind") == "text"
            ]
            user_text = " ".join(text_parts)
            return f"Task completed: {user_text}"
        else:
            return f"Task completed: {message}"

    # Register A2A service
    await api.register_service(
        {
            "id": "task-agent",
            "type": "a2a",
            "config": {"visibility": "public"},
            "agent_card": agent_card,
            "run": task_agent_run,
        }
    )

    # Test message/send that creates a task
    async with httpx.AsyncClient() as client:
        rpc_request = {
            "jsonrpc": "2.0",
            "id": "task-test-1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Process this request"}],
                    "messageId": "task-message-1",
                    "kind": "message",
                }
            },
        }

        response = await client.post(
            f"{SERVER_URL}/{workspace}/a2a/task-agent",
            json=rpc_request,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        rpc_response = response.json()
        assert rpc_response["jsonrpc"] == "2.0"
        assert rpc_response["id"] == "task-test-1"

        result = rpc_response["result"]

        # If result is a task, test tasks/get
        if result.get("kind") == "task":
            task_id = result["id"]

            # Test tasks/get method
            get_task_request = {
                "jsonrpc": "2.0",
                "id": "get-task-1",
                "method": "tasks/get",
                "params": {"id": task_id, "historyLength": 5},
            }

            response = await client.post(
                f"{SERVER_URL}/{workspace}/a2a/task-agent",
                json=get_task_request,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == 200
            get_response = response.json()
            assert get_response["jsonrpc"] == "2.0"
            assert get_response["id"] == "get-task-1"

            task = get_response["result"]
            assert task["id"] == task_id
            assert task["kind"] == "task"
            assert "status" in task

    await api.disconnect()


async def test_a2a_dynamic_agent_card(fastapi_server, test_user_token):
    """Test A2A agent with dynamic agent card function."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define a function that returns agent card
    def create_dynamic_agent_card():
        return {
            "protocol_version": "0.3.0",
            "name": "Dynamic Agent",
            "description": f"Agent with dynamic card created at {time.time()}",
            "url": f"http://localhost:9527/{workspace}/a2a/dynamic-agent",
            "version": "1.0.0",
            "capabilities": {"streaming": False, "push_notifications": False},
            "default_input_modes": ["text/plain"],
            "default_output_modes": ["text/plain"],
            "skills": [
                {
                    "id": "dynamic",
                    "name": "Dynamic Skill",
                    "description": "Dynamically generated skill",
                    "tags": ["dynamic", "test"],
                }
            ],
        }

    async def dynamic_agent_run(message, context=None):
        return "Hello from dynamic agent"

    # Register A2A service with function-based agent card
    await api.register_service(
        {
            "id": "dynamic-agent",
            "type": "a2a",
            "config": {"visibility": "public"},
            "agent_card": create_dynamic_agent_card,  # Function instead of dict
            "run": dynamic_agent_run,
        }
    )

    # Test that agent card is generated correctly
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SERVER_URL}/{workspace}/a2a/dynamic-agent/.well-known/agent-card.json"
        )
        assert response.status_code == 200

        card = response.json()
        assert card["name"] == "Dynamic Agent"
        assert "dynamic card created at" in card["description"]
        assert card["skills"][0]["id"] == "dynamic"

    await api.disconnect()


async def test_a2a_error_handling(fastapi_server, test_user_token):
    """Test A2A error handling for invalid requests."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Register a simple agent
    agent_card = {
        "protocol_version": "0.3.0",
        "name": "Error Test Agent",
        "description": "Agent for testing error handling",
        "url": f"http://localhost:9527/{workspace}/a2a/error-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "push_notifications": False},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/plain"],
        "skills": [
            {
                "id": "error-test",
                "name": "Error Test",
                "description": "Tests error conditions",
                "tags": ["test", "error"],
            }
        ],
    }

    async def error_agent_run(message, context=None):
        # Check if message requests an error
        if isinstance(message, dict) and "parts" in message:
            text_parts = [
                part.get("text", "")
                for part in message.get("parts", [])
                if part.get("kind") == "text"
            ]
            text = " ".join(text_parts).lower()
            if "error" in text:
                raise ValueError("Requested error for testing")

        return "No error occurred"

    await api.register_service(
        {
            "id": "error-agent",
            "type": "a2a",
            "config": {"visibility": "public"},
            "agent_card": agent_card,
            "run": error_agent_run,
        }
    )

    async with httpx.AsyncClient() as client:
        # Test 1: Invalid JSON-RPC request
        response = await client.post(
            f"{SERVER_URL}/{workspace}/a2a/error-agent",
            json={"invalid": "request"},
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200  # JSON-RPC errors are returned with 200
        rpc_response = response.json()
        assert "error" in rpc_response
        assert rpc_response["error"]["code"] == -32600  # Invalid Request

        # Test 2: Method not found (A2A SDK treats this as validation error)
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "unknown/method",
            "params": {},
        }
        response = await client.post(
            f"{SERVER_URL}/{workspace}/a2a/error-agent",
            json=rpc_request,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200
        rpc_response = response.json()
        assert "error" in rpc_response
        assert (
            rpc_response["error"]["code"] == -32600
        )  # Request payload validation error

        # Test 3: Agent runtime error
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "please trigger an error"}],
                    "messageId": "error-message-1",
                    "kind": "message",
                }
            },
        }
        response = await client.post(
            f"{SERVER_URL}/{workspace}/a2a/error-agent",
            json=rpc_request,
            headers={"Content-Type": "application/json"},
        )

        # Should return either a task with failed status or an error response
        assert response.status_code == 200
        rpc_response = response.json()

        if "error" in rpc_response:
            # Direct error response
            assert rpc_response["error"]["code"] == -32603  # Internal error
        else:
            # Task with failed status
            result = rpc_response["result"]
            if result.get("kind") == "task":
                # Task might be in failed state or completed with error message
                pass

    await api.disconnect()


async def test_a2a_nonexistent_service(fastapi_server, test_user_token):
    """Test accessing nonexistent A2A service."""
    async with httpx.AsyncClient() as client:
        # Try to access agent card for nonexistent service
        response = await client.get(
            f"{SERVER_URL}/ws-user-user-1/a2a/nonexistent/.well-known/agent-card.json"
        )
        assert response.status_code == 404

        # Try to send message to nonexistent service
        rpc_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}],
                    "messageId": "test",
                    "kind": "message",
                }
            },
        }
        response = await client.post(
            f"{SERVER_URL}/ws-user-user-1/a2a/nonexistent",
            json=rpc_request,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 404


async def test_a2a_client_sdk(fastapi_server, test_user_token):
    """Test A2A interaction using the official A2A client SDK.

    This test demonstrates end-to-end A2A (Agent-to-Agent) communication using the
    official a2a-sdk client library. It tests:

    1. Agent card resolution - fetching agent metadata via HTTP
    2. A2A client creation from resolved agent card
    3. Message sending with proper A2A protocol formatting
    4. Response handling for different message types (greeting, echo, structured data)
    5. Validation that agent responses are properly resolved from Hypha RPC Futures

    The test creates a Hypha service with type="a2a" that handles A2A protocol
    messages and responds appropriately. It then uses the A2A SDK client to interact
    with this service, similar to how external A2A agents would communicate with
    Hypha-hosted agents.
    """
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define agent card for our test agent
    agent_card = {
        "protocol_version": "0.3.0",
        "name": "SDK Test Agent",
        "description": "Agent for testing A2A SDK client integration",
        "url": f"{SERVER_URL}/{workspace}/a2a/sdk-agent",
        "version": "1.0.0",
        "capabilities": {"streaming": True, "push_notifications": False},
        "default_input_modes": ["text/plain", "application/json"],
        "default_output_modes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "greet",
                "name": "Greeting Skill",
                "description": "Provides friendly greetings and responses",
                "tags": ["greeting", "conversation"],
                "examples": ["Hello there!", "How are you doing?", "Nice to meet you"],
            },
            {
                "id": "echo",
                "name": "Echo Skill",
                "description": "Echoes back user input with formatting",
                "tags": ["echo", "utility"],
                "examples": ["echo hello world", "repeat this message"],
            },
        ],
    }

    async def sdk_agent_run(message, context=None):
        """Agent function that handles different types of requests."""
        # Handle A2A Message format
        if isinstance(message, dict) and "parts" in message:
            text_parts = []
            for part in message["parts"]:
                if part.get("kind") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("kind") == "data":
                    data = part.get("data", {})
                    text_parts.append(f"Data received: {json.dumps(data)}")

            user_text = " ".join(text_parts).strip()

            # Simulate different responses based on input
            if "hello" in user_text.lower() or "hi" in user_text.lower():
                return "Hello! I'm the SDK Test Agent. How can I help you today?"
            elif "echo" in user_text.lower():
                # Extract text after "echo"
                echo_text = user_text.lower().replace("echo", "").strip()
                return f"Echo: {echo_text}"
            elif "json" in user_text.lower():
                # Return structured data
                return {
                    "type": "response",
                    "agent": "SDK Test Agent",
                    "message": "Here's some structured data",
                    "timestamp": time.time(),
                    "data": {"key": "value", "number": 42},
                }
            else:
                return f"SDK Agent processed: {user_text}"
        else:
            return f"SDK Agent received: {message}"

    # Register A2A service
    await api.register_service(
        {
            "id": "sdk-agent",
            "type": "a2a",
            "config": {"visibility": "public"},
            "agent_card": agent_card,
            "run": sdk_agent_run,
        }
    )

    # Test 1: Resolve agent card using A2ACardResolver
    async with httpx.AsyncClient() as httpx_client:
        base_url = f"{SERVER_URL}/{workspace}/a2a/sdk-agent"
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
            agent_card_path="/.well-known/agent-card.json",
        )

        resolved_card = await resolver.get_agent_card()
        assert resolved_card is not None
        assert resolved_card.name == "SDK Test Agent"
        assert resolved_card.protocol_version == "0.3.0"
        assert len(resolved_card.skills) == 2
        assert resolved_card.capabilities.streaming is True

        # Test 2: Create A2A client and send message
        client = A2AClient(httpx_client=httpx_client, agent_card=resolved_card)

        # Send a greeting message
        response = await client.send_message(create_text_message("Hello SDK Agent!"))
        assert response is not None

        # Response could be either a Message or Task
        if hasattr(response, "kind") and response.kind == "task":
            # It's a Task object
            assert response.status.state in ["completed", "working", "submitted"]
            if response.status.state == "completed" and response.artifacts:
                # Check if we got the expected greeting response
                text_parts = []
                for artifact in response.artifacts:
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
                response_text = " ".join(text_parts)
                assert "Hello!" in response_text or "SDK Test Agent" in response_text
        elif hasattr(response, "parts"):
            # It's a Message object
            text_parts = []
            for part in response.parts:
                if hasattr(part, "text"):
                    text_parts.append(part.text)
            response_text = " ".join(text_parts)
            assert "Hello!" in response_text or "SDK Test Agent" in response_text

        # Test 3: Send echo message
        echo_response = await client.send_message(
            create_text_message("echo test message")
        )
        assert echo_response is not None

        # Extract response text (handling both Task and Message responses)
        response_text = ""

        # Handle SendMessageResponse structure
        if hasattr(echo_response, "root") and hasattr(echo_response.root, "result"):
            result = echo_response.root.result

            if hasattr(result, "kind") and result.kind == "task":
                if result.status.state == "completed" and result.artifacts:
                    for artifact in result.artifacts:
                        for part in artifact.parts:
                            if hasattr(part, "text"):
                                response_text += part.text
                            elif hasattr(part, "root") and hasattr(part.root, "text"):
                                response_text += part.root.text
            elif hasattr(result, "parts"):
                for part in result.parts:
                    if hasattr(part, "text"):
                        response_text += part.text
                    elif hasattr(part, "root") and hasattr(part.root, "text"):
                        response_text += part.root.text

        assert "Echo:" in response_text and "test message" in response_text

        # Test 4: Send structured data
        json_response = await client.send_message(
            create_text_message("send me some json data")
        )
        assert json_response is not None

        # Should contain structured response
        response_found = False
        if hasattr(json_response, "root") and hasattr(json_response.root, "result"):
            result = json_response.root.result
            if hasattr(result, "parts"):
                for part in result.parts:
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        if "structured data" in part.root.text:
                            response_found = True
                    elif hasattr(part, "root") and hasattr(part.root, "data"):
                        if isinstance(part.root.data, dict):
                            response_found = True

        assert response_found, "Expected structured data response not found"

    await api.disconnect()
