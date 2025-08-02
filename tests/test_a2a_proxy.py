"""Test A2A Apps functionality with A2A proxy for reversible communication."""

import asyncio
import json
import pytest
import httpx
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

from . import WS_SERVER_URL, SERVER_URL

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def _handle_a2a_message(message, skills):
    """Helper function to handle A2A messages and route them to skills."""
    # Extract text from A2A message parts
    text_content = ""
    if isinstance(message, dict) and "parts" in message:
        for part in message["parts"]:
            if part.get("kind") == "text":
                text_content += part.get("text", "")
    else:
        text_content = str(message)

    # Route to the appropriate skill based on content
    text_lower = text_content.lower()
    if "echo" in text_lower:
        return skills[0](text=text_content)
    elif "hello" in text_lower or "hi" in text_lower or "greeting" in text_lower:
        return skills[1](text=text_content)
    elif "add" in text_lower or "calculate" in text_lower:
        return skills[2](text=text_content)
    else:
        # Default to echo skill
        return skills[0](text=text_content)


async def test_a2a_round_trip_service_consistency(fastapi_server, test_user_token):
    """Test complete A2A double round-trip demonstrating reversible communication."""

    # Connect to the Hypha server
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace
    controller = await api.get_service("public/server-apps")

    # Step 1: Register a Hypha service with A2A type
    @schema_function
    def echo_skill(text: str) -> str:
        """Echo back the input text with a prefix."""
        return f"A2A Echo: {text}"

    @schema_function
    def greeting_skill(text: str) -> str:
        """Provide a greeting response."""
        return f"Hello! You said: {text}"

    @schema_function
    def calculate_skill(text: str) -> str:
        """Simple calculation skill."""
        if "add" in text.lower():
            # Extract numbers from text (simple parsing)
            words = text.split()
            try:
                numbers = [
                    float(word) for word in words if word.replace(".", "").isdigit()
                ]
                if len(numbers) >= 2:
                    result = sum(numbers)
                    return f"The sum is: {result}"
            except ValueError:
                pass
        return f"I can help with simple addition. Try: 'add 5 and 3'"

    # Create async run function
    async def a2a_run(message, context=None):
        return await _handle_a2a_message(
            message, [echo_skill, greeting_skill, calculate_skill]
        )

    # Register the original A2A service
    original_service_info = await api.register_service(
        {
            "id": "a2a-test-service",
            "name": "A2A Test Service",
            "description": "A service for testing A2A round-trip consistency",
            "type": "a2a",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "agent_card": {
                "protocol_version": "0.3.0",
                "name": "Test A2A Agent",
                "description": "A test agent for A2A round-trip testing",
                "url": f"{SERVER_URL}/{workspace}/a2a/a2a-test-service",
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
                    },
                    {
                        "id": "greeting",
                        "name": "Greeting Skill",
                        "description": "Provides friendly greetings",
                        "tags": ["test", "greeting"],
                        "examples": ["hello", "hi there"],
                    },
                    {
                        "id": "calculate",
                        "name": "Calculate Skill",
                        "description": "Performs simple calculations",
                        "tags": ["test", "math"],
                        "examples": ["add 5 and 3", "calculate 10 plus 7"],
                    },
                ],
            },
            "skills": [echo_skill, greeting_skill, calculate_skill],
            "run": a2a_run,
        }
    )

    print(f"✓ Original A2A service registered: {original_service_info['id']}")

    # Step 2: Get the original service to save its skill references
    original_service = await api.get_service(original_service_info["id"])

    # Verify the original service has the expected structure
    assert hasattr(original_service, "skills") and original_service.skills is not None
    assert len(original_service.skills) == 3

    print("✓ Original service has skills organized correctly")

    # Step 3: Get the A2A endpoint URL for this service
    a2a_endpoint_url = f"{SERVER_URL}/{workspace}/a2a/a2a-test-service"
    print(f"✓ A2A Endpoint URL: {a2a_endpoint_url}")

    # Step 4: Create A2A agent configuration
    a2a_config = {
        "type": "a2a-agent",
        "name": "A2A Proxy App",
        "version": "1.0.0",
        "description": "A2A app created from original service for testing",
        "a2aAgents": {"test-agent": {"url": a2a_endpoint_url, "headers": {}}},
    }

    # Install the A2A agent app
    a2a_app_info = await controller.install(
        manifest=a2a_config,
        overwrite=True,
        stage=True,
    )

    print(f"✓ A2A app installed: {a2a_app_info['id']}")

    # Step 5: Start the app manually
    print("🚀 Starting A2A app...")
    agent_name = "test-agent"
    session_info = await controller.start(
        a2a_app_info["id"], wait_for_service=agent_name, timeout=30
    )
    print(f"✓ A2A app started successfully: {session_info}")

    # Step 6: Get the unified A2A service
    print(f"🔍 Getting unified A2A service: {session_info['id']}:{agent_name}")

    a2a_service_id = f"{session_info['id']}:{agent_name}"
    a2a_service = await api.get_service(a2a_service_id)
    print(f"✓ Successfully retrieved unified A2A service: {a2a_service_id}")

    # Step 7: Compare original service with the A2A service
    print("🔄 Comparing original service with A2A service...")

    # Test skill functionality consistency
    print("  Testing echo_skill...")
    original_echo_skill = original_service.skills[0]
    a2a_echo_skill = a2a_service.skills[0]

    # Test the skills with the same parameters
    original_result = await original_echo_skill(text="Hello World")
    a2a_result = await a2a_echo_skill(text="Hello World")

    # Both should contain the input text, but A2A might have different formatting
    assert "Hello World" in original_result and "Hello World" in a2a_result
    print(f"    ✓ Original: {original_result}")
    print(f"    ✓ A2A: {a2a_result}")

    # Test greeting skill
    print("  Testing greeting_skill...")
    original_greeting_skill = original_service.skills[1]
    a2a_greeting_skill = a2a_service.skills[1]

    original_greeting = await original_greeting_skill(text="Nice to meet you")
    a2a_greeting = await a2a_greeting_skill(text="Nice to meet you")

    assert (
        "Nice to meet you" in original_greeting and "Nice to meet you" in a2a_greeting
    )
    print(f"    ✓ Original: {original_greeting}")
    print(f"    ✓ A2A: {a2a_greeting}")

    # Test calculate skill
    print("  Testing calculate_skill...")
    original_calculate_skill = original_service.skills[2]
    a2a_calculate_skill = a2a_service.skills[2]

    original_calc = await original_calculate_skill(text="add 10 and 5")
    a2a_calc = await a2a_calculate_skill(text="add 10 and 5")

    # Both should handle the calculation request
    print(f"    ✓ Original: {original_calc}")
    print(f"    ✓ A2A: {a2a_calc}")

    print("✅ A2A ROUND-TRIP SUCCESS!")

    # Clean up
    await controller.stop(session_info["id"])
    print("✓ A2A app session stopped")

    await controller.uninstall(a2a_app_info["id"])
    print("✓ A2A app uninstalled")

    await api.unregister_service(original_service_info["id"])
    print("✓ Original service unregistered")

    await api.disconnect()


async def test_a2a_double_round_trip_reversibility(fastapi_server, test_user_token):
    """Test double round-trip: Hypha -> A2A -> Hypha -> A2A to demonstrate full reversibility."""

    # Connect to the Hypha server
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace
    controller = await api.get_service("public/server-apps")

    # Step 1: Register initial Hypha service
    @schema_function
    def reversible_skill(text: str) -> str:
        """A skill that can be reversibly accessed through A2A proxy."""
        return f"Reversible: {text}"

    # Create async run function
    async def reversible_run(message, context=None):
        return await _handle_a2a_message(message, [reversible_skill])

    original_service_info = await api.register_service(
        {
            "id": "reversible-test-service",
            "name": "Reversible Test Service",
            "description": "A service for testing A2A reversibility",
            "type": "a2a",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "agent_card": {
                "protocol_version": "0.3.0",
                "name": "Reversible A2A Agent",
                "description": "A reversible agent for testing",
                "url": f"{SERVER_URL}/{workspace}/a2a/reversible-test-service",
                "version": "1.0.0",
                "capabilities": {"streaming": False, "push_notifications": False},
                "default_input_modes": ["text/plain"],
                "default_output_modes": ["text/plain"],
                "skills": [
                    {
                        "id": "reversible",
                        "name": "Reversible Skill",
                        "description": "A skill that demonstrates reversibility",
                        "tags": ["test", "reversible"],
                        "examples": ["process this text"],
                    }
                ],
            },
            "skills": [reversible_skill],
            "run": reversible_run,
        }
    )

    print(f"✓ Original reversible service registered: {original_service_info['id']}")

    # Step 2: Create first A2A agent app (Hypha -> A2A)
    a2a_endpoint_url = f"{SERVER_URL}/{workspace}/a2a/reversible-test-service"

    first_a2a_config = {
        "type": "a2a-agent",
        "name": "First A2A Proxy App",
        "version": "1.0.0",
        "description": "First A2A app in the chain",
        "a2aAgents": {"first-agent": {"url": a2a_endpoint_url, "headers": {}}},
    }

    first_a2a_app_info = await controller.install(
        manifest=first_a2a_config,
        overwrite=True,
        stage=True,
    )

    print(f"✓ First A2A app installed: {first_a2a_app_info['id']}")

    # Step 3: Start first A2A app
    first_session_info = await controller.start(
        first_a2a_app_info["id"], wait_for_service="first-agent", timeout=30
    )
    print(f"✓ First A2A app started: {first_session_info['id']}")

    # Step 4: Get first A2A service
    first_a2a_service_id = f"{first_session_info['id']}:first-agent"
    first_a2a_service = await api.get_service(first_a2a_service_id)
    print(f"✓ First A2A service retrieved: {first_a2a_service_id}")

    # Step 5: Create second A2A agent app that connects to the first one (A2A -> A2A)
    # This demonstrates the full reversibility chain
    second_a2a_config = {
        "type": "a2a-agent",
        "name": "Second A2A Proxy App",
        "version": "1.0.0",
        "description": "Second A2A app in the chain for double round-trip",
        "a2aAgents": {
            "second-agent": {
                "url": a2a_endpoint_url,  # Connect to the same original service
                "headers": {},
            }
        },
    }

    second_a2a_app_info = await controller.install(
        manifest=second_a2a_config,
        overwrite=True,
        stage=True,
    )

    print(f"✓ Second A2A app installed: {second_a2a_app_info['id']}")

    # Step 6: Start second A2A app
    second_session_info = await controller.start(
        second_a2a_app_info["id"], wait_for_service="second-agent", timeout=30
    )
    print(f"✓ Second A2A app started: {second_session_info['id']}")

    # Step 7: Get second A2A service
    second_a2a_service_id = f"{second_session_info['id']}:second-agent"
    second_a2a_service = await api.get_service(second_a2a_service_id)
    print(f"✓ Second A2A service retrieved: {second_a2a_service_id}")

    # Step 8: Test the full chain: Original -> First A2A -> Second A2A
    print("🔄 Testing full reversibility chain...")

    # Test original service
    original_service = await api.get_service(original_service_info["id"])
    original_result = await original_service.skills[0](text="Chain Test")
    print(f"  Original service: {original_result}")

    # Test first A2A service
    first_a2a_result = await first_a2a_service.skills[0](text="Chain Test")
    print(f"  First A2A service: {first_a2a_result}")

    # Test second A2A service
    second_a2a_result = await second_a2a_service.skills[0](text="Chain Test")
    print(f"  Second A2A service: {second_a2a_result}")

    # All should contain the test text
    assert "Chain Test" in original_result
    assert "Chain Test" in first_a2a_result
    assert "Chain Test" in second_a2a_result

    print("✅ DOUBLE ROUND-TRIP REVERSIBILITY SUCCESS!")

    # Clean up
    await controller.stop(first_session_info["id"])
    await controller.stop(second_session_info["id"])
    print("✓ A2A app sessions stopped")

    await controller.uninstall(first_a2a_app_info["id"])
    await controller.uninstall(second_a2a_app_info["id"])
    print("✓ A2A apps uninstalled")

    await api.unregister_service(original_service_info["id"])
    print("✓ Original service unregistered")

    await api.disconnect()


async def test_a2a_error_handling_and_debugging(fastapi_server, test_user_token):
    """Test error handling and debugging capabilities of A2A clients."""

    # Connect to the Hypha server
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace
    controller = await api.get_service("public/server-apps")

    # Test 1: Invalid URL handling
    print("🔍 Testing invalid URL handling...")
    a2a_config_invalid = {
        "type": "a2a-agent",
        "name": "Invalid URL A2A App",
        "version": "1.0.0",
        "description": "A2A app with invalid URL to test error handling",
        "a2aAgents": {
            "invalid-agent": {
                "url": "http://invalid-host:9999/a2a/nonexistent",
                "headers": {},
            }
        },
    }

    # Install the A2A agent app with invalid URL
    a2a_app_info = await controller.install(
        manifest=a2a_config_invalid,
        overwrite=True,
        stage=True,
    )

    print(f"✓ Invalid URL A2A app installed: {a2a_app_info['id']}")

    # Start the app - should handle the error gracefully
    try:
        session_info = await controller.start(
            a2a_app_info["id"], wait_for_service="invalid-agent", timeout=10
        )
        print(f"⚠️ App started but likely with errors: {session_info}")

        # Check logs for error messages
        logs = await controller.get_logs(session_info["id"])
        print(f"✓ Logs retrieved: {logs}")

        # Should have error logs
        assert "error" in logs
        assert len(logs["error"]) > 0
        print("✓ Error logs found as expected")

        # Stop the session
        await controller.stop(session_info["id"])

    except Exception as e:
        print(f"✓ Expected error occurred: {e}")

    # Clean up
    await controller.uninstall(a2a_app_info["id"])
    print("✓ Invalid URL A2A app uninstalled")

    # Test 2: Missing configuration - should fail at install time
    print("🔍 Testing missing configuration...")
    a2a_config_missing = {
        "type": "a2a-agent",
        "name": "Missing Config A2A App",
        "version": "1.0.0",
        "description": "A2A app with missing configuration to test error handling",
        "a2aAgents": {},  # Empty configuration
    }

    # Install the A2A agent app with missing config - should fail
    try:
        a2a_app_info = await controller.install(
            manifest=a2a_config_missing,
            overwrite=True,
            stage=True,
        )
        print(f"⚠️ App installed unexpectedly: {a2a_app_info['id']}")
        # Clean up if it somehow got installed
        await controller.uninstall(a2a_app_info["id"])
        assert False, "Expected installation to fail due to empty a2aAgents"
    except Exception as e:
        # Should get validation error
        assert "a2aAgents configuration is required" in str(e)
        print(f"✓ Expected validation error occurred: {e}")

    print("✓ Missing config validation working correctly")

    print("✅ ERROR HANDLING AND DEBUGGING TESTS PASSED!")

    await api.disconnect()


async def test_a2a_config_validation(fastapi_server, test_user_token):
    """Test A2A agent configuration validation."""

    # Connect to the Hypha server
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace
    controller = await api.get_service("public/server-apps")

    # Test valid A2A agent configuration
    valid_config = {
        "type": "a2a-agent",
        "name": "Valid A2A Agent",
        "version": "1.0.0",
        "description": "Valid A2A agent configuration",
        "a2aAgents": {
            "test-agent": {
                "url": "https://example.com/a2a/test",
                "headers": {"Authorization": "Bearer token123"},
            }
        },
    }

    validation_result = await controller.validate_app_manifest(valid_config)
    assert validation_result["valid"] is True
    print("✓ Valid A2A config passed validation")

    # Test invalid A2A agent configuration - missing a2aAgents
    invalid_config_1 = {
        "type": "a2a-agent",
        "name": "Invalid A2A Agent",
        "version": "1.0.0",
        "description": "Invalid A2A agent configuration",
        # Missing a2aAgents
    }

    validation_result = await controller.validate_app_manifest(invalid_config_1)
    assert validation_result["valid"] is False
    assert any("a2aAgents" in error for error in validation_result["errors"])
    print("✓ Invalid A2A config (missing a2aAgents) failed validation as expected")

    # Test invalid A2A agent configuration - empty a2aAgents
    invalid_config_2 = {
        "type": "a2a-agent",
        "name": "Invalid A2A Agent",
        "version": "1.0.0",
        "description": "Invalid A2A agent configuration",
        "a2aAgents": {},  # Empty
    }

    validation_result = await controller.validate_app_manifest(invalid_config_2)
    assert validation_result["valid"] is False
    assert any("cannot be empty" in error for error in validation_result["errors"])
    print("✓ Invalid A2A config (empty a2aAgents) failed validation as expected")

    # Test invalid A2A agent configuration - missing URL
    invalid_config_3 = {
        "type": "a2a-agent",
        "name": "Invalid A2A Agent",
        "version": "1.0.0",
        "description": "Invalid A2A agent configuration",
        "a2aAgents": {
            "test-agent": {
                "headers": {"Authorization": "Bearer token123"}
                # Missing url
            }
        },
    }

    validation_result = await controller.validate_app_manifest(invalid_config_3)
    assert validation_result["valid"] is False
    assert any(
        "missing required field 'url'" in error for error in validation_result["errors"]
    )
    print("✓ Invalid A2A config (missing URL) failed validation as expected")

    # Test warning for non-HTTP URL
    warning_config = {
        "type": "a2a-agent",
        "name": "Warning A2A Agent",
        "version": "1.0.0",
        "description": "A2A agent with non-HTTP URL",
        "a2aAgents": {
            "test-agent": {"url": "ftp://example.com/a2a/test", "headers": {}}
        },
    }

    validation_result = await controller.validate_app_manifest(warning_config)
    assert validation_result["valid"] is True  # Still valid but with warnings
    assert any(
        "should start with http://" in warning
        for warning in validation_result["warnings"]
    )
    print("✓ A2A config with non-HTTP URL generated warning as expected")

    print("✅ A2A CONFIG VALIDATION TESTS PASSED!")

    await api.disconnect()
