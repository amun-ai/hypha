"""Test the Server Apps progress monitoring and early disconnect features."""

import asyncio
import time
import pytest
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SERVER_URL

# All test coroutines will be treated as marked
pytestmark = pytest.mark.asyncio


async def test_app_progress_logging(minio_server, fastapi_server, test_user_token):
    """Test that apps can send progress messages via api.log()."""
    api = await connect_to_server({
        "name": "test-progress-client", 
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    controller = await api.get_service("public/server-apps")
    
    progress_messages = []
    
    def progress_callback(msg):
        progress_messages.append(msg)
    
    # Install an app that uses api.log() for progress
    app_source = '''
    api.export({
        async setup() {
            await api.log("Starting application setup...");
            await api.log({type: "progress", content: "Loading configuration..."});
            await new Promise(resolve => setTimeout(resolve, 100));
            await api.log({type: "success", content: "Configuration loaded"});
            console.log("App setup completed");
        },
        async test() {
            return "test function works";
        }
    })
    '''
    
    # Install the app
    app_info = await controller.install(
        source=app_source,
        app_id="progress-test-app",
        manifest={
            "name": "Progress Test App",
            "type": "web-worker",
            "version": "1.0.0"
        },
        overwrite=True,
        progress_callback=progress_callback
    )
    
    # Start the app to see progress messages
    progress_messages.clear()
    started_app = await controller.start(
        app_info["id"],
        wait_for_service="default",
        timeout=10,
        progress_callback=progress_callback
    )
    
    # Check that we received progress messages
    app_messages = [msg for msg in progress_messages if "[App]" in msg.get("message", "")]
    assert len(app_messages) >= 2, f"Expected at least 2 app messages, got {len(app_messages)}: {app_messages}"
    
    # Check message types
    message_types = [msg["type"] for msg in app_messages]
    assert "info" in message_types or "progress" in message_types
    assert "success" in message_types
    
    # Test the app still works
    app_service = await api.get_service(f"default@{app_info['id']}")
    result = await app_service.test()
    assert result == "test function works"
    
    # Stop the app
    await controller.stop(started_app["id"])
    
    # Cleanup
    await controller.uninstall(app_info["id"])
    await api.disconnect()


async def test_window_app_syntax_error_early_failure(minio_server, fastapi_server, test_user_token):
    """Test that window apps with syntax errors fail immediately without waiting for timeout."""
    api = await connect_to_server({
        "name": "test-window-syntax-error-client", 
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    controller = await api.get_service("public/server-apps")
    
    progress_messages = []
    
    def progress_callback(msg):
        progress_messages.append(msg)
        print(f"Progress: {msg}")  # Debug output
    
    # Install a window app with a syntax error
    app_source = '''
    // This window app has a syntax error
    console.log("Starting window app...");
    // Syntax error: missing closing bracket
    if (true {
        console.log("This won't run");
    }
    console.log("Never reached");
    '''
    
    # Install the app without testing since it has a syntax error
    app_info = await controller.install(
        source=app_source,
        app_id="window-syntax-error-app",
        manifest={
            "name": "Window Syntax Error App",
            "type": "window",
            "version": "1.0.0"
        },
        overwrite=True,
        stage=True,  # Don't test the app during installation
        progress_callback=progress_callback
    )
    
    # Try to start the app - it should fail quickly due to syntax error
    progress_messages.clear()
    
    # Record start time
    start_time = time.time()
    timeout_seconds = 60  # Use a long timeout to ensure we're failing early
    
    with pytest.raises(Exception) as exc_info:
        await controller.start(
            app_info["id"],
            wait_for_service="default",
            timeout=timeout_seconds,
            stage=True,  # Start from the staged version
            progress_callback=progress_callback
        )
    
    # Record end time
    elapsed_time = time.time() - start_time
    
    # CRITICAL TEST: Syntax errors should fail within 5 seconds, not wait for full timeout
    assert elapsed_time < 5, f"Window app with syntax error should fail quickly (< 5s), but took {elapsed_time:.2f}s. This indicates early failure detection is not working for window apps!"
    
    # Check that the error message mentions syntax error or parse error
    error_str = str(exc_info.value).lower()
    assert any(keyword in error_str for keyword in ["syntax", "parse", "unexpected", "disconnect", "error"]), \
        f"Error message should indicate syntax/parse error or disconnect, got: {exc_info.value}"
    
    # Check if we got any error messages in progress
    error_messages = [msg for msg in progress_messages if msg["type"] == "error"]
    print(f"Error messages received: {error_messages}")
    
    # Cleanup
    await controller.uninstall(app_info["id"])
    await api.disconnect()

async def test_app_early_disconnect_on_error(minio_server, fastapi_server, test_user_token):
    """Test that apps can disconnect early to signal failure."""
    api = await connect_to_server({
        "name": "test-disconnect-client", 
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    controller = await api.get_service("public/server-apps")
    
    progress_messages = []
    
    def progress_callback(msg):
        progress_messages.append(msg)
    
    # Install an app that fails during setup
    app_source = '''
    api.export({
        async setup() {
            await api.log("Starting application...");
            await api.log({type: "progress", content: "Checking requirements..."});
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Simulate an error condition
            const errorMsg = "Missing required configuration!";
            await api.log({type: "error", content: errorMsg});
            console.error(errorMsg); // Also log to console.error so browser worker can capture it
            await api.disconnect(); // Disconnect to signal failure
        }
    })
    '''
    
    # Install the app without testing since it's designed to fail
    app_info = await controller.install(
        source=app_source,
        app_id="failure-test-app",
        manifest={
            "name": "Failure Test App",
            "type": "web-worker",
            "version": "1.0.0"
        },
        overwrite=True,
        stage=True,  # Don't test the app during installation
        progress_callback=progress_callback
    )
    
    # Don't commit - we'll test directly from stage since the app is designed to fail
    
    # Try to start the app - it should fail quickly
    progress_messages.clear()
    
    # Record start time
    start_time = time.time()
    timeout_seconds = 10
    
    with pytest.raises(Exception) as exc_info:
        await controller.start(
            app_info["id"],
            wait_for_service="default",
            timeout=timeout_seconds,
            stage=True,  # Start from the staged version
            progress_callback=progress_callback
        )
    
    # Record end time
    elapsed_time = time.time() - start_time
    
    # IMPORTANT: Check that we failed early, not after timeout
    # The app disconnects after ~100ms, so we should fail within 2 seconds
    assert elapsed_time < 3, f"App should have failed quickly (< 3s), but took {elapsed_time:.2f}s. This suggests early disconnect is not working!"
    
    # Check that the actual error message "Missing required configuration!" is captured
    error_str = str(exc_info.value)
    
    # The error should contain our specific error message
    assert "Missing required configuration!" in error_str, \
        f"Error message should contain 'Missing required configuration!', got: {error_str[:500]}"
    
    # Also check that we received the error message in progress callbacks
    error_messages = [msg for msg in progress_messages if msg["type"] == "error"]
    assert len(error_messages) >= 1, "Should have received at least one error message in progress"
    
    # The progress error should also contain our specific message
    progress_error_str = " ".join([str(msg.get("message", "")) for msg in error_messages])
    assert "Missing required configuration!" in progress_error_str, \
        f"Progress error messages should contain 'Missing required configuration!', got: {progress_error_str}"
    
    # Cleanup
    await controller.uninstall(app_info["id"])
    await api.disconnect()
