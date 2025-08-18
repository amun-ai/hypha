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
            await api.log({type: "error", content: "Missing required configuration!"});
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
    
    # Check that the error message contains our error log
    assert "Missing required configuration" in str(exc_info.value) or "disconnected" in str(exc_info.value).lower()
    
    # Check that we received the error message in progress
    error_messages = [msg for msg in progress_messages if msg["type"] == "error"]
    assert len(error_messages) >= 1, "Should have received at least one error message"
    
    # Cleanup
    await controller.uninstall(app_info["id"])
    await api.disconnect()


async def test_app_intermediate_progress_messages(minio_server, fastapi_server, test_user_token):
    """Test that apps can send intermediate progress messages during execution."""
    api = await connect_to_server({
        "name": "test-intermediate-client", 
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    controller = await api.get_service("public/server-apps")
    
    progress_messages = []
    
    def progress_callback(msg):
        progress_messages.append(msg)
    
    # Install an app with multiple progress stages
    app_source = '''
    api.export({
        async setup() {
            await api.log("Phase 1: Initialization");
            await api.log({type: "progress", content: "Step 1/3: Loading modules..."});
            await new Promise(resolve => setTimeout(resolve, 50));
            
            await api.log({type: "progress", content: "Step 2/3: Connecting to services..."});
            await new Promise(resolve => setTimeout(resolve, 50));
            
            await api.log({type: "progress", content: "Step 3/3: Finalizing setup..."});
            await new Promise(resolve => setTimeout(resolve, 50));
            
            await api.log({type: "success", content: "All systems operational!"});
        },
        async processData(data) {
            await api.log({type: "info", content: `Processing ${data.length} items...`});
            // Simulate processing
            await new Promise(resolve => setTimeout(resolve, 100));
            await api.log({type: "success", content: "Processing complete!"});
            return data.map(x => x * 2);
        }
    })
    '''
    
    # Install and start the app
    app_info = await controller.install(
        source=app_source,
        app_id="multi-progress-app",
        manifest={
            "name": "Multi Progress App",
            "type": "web-worker",
            "version": "1.0.0"
        },
        overwrite=True,
        progress_callback=progress_callback
    )
    
    progress_messages.clear()
    started_app = await controller.start(
        app_info["id"],
        wait_for_service="default",
        timeout=10,
        progress_callback=progress_callback
    )
    
    # Check we got multiple progress messages
    app_messages = [msg for msg in progress_messages if "[App]" in msg.get("message", "")]
    assert len(app_messages) >= 4, f"Expected at least 4 progress messages, got {len(app_messages)}"
    
    # Check that we have different progress stages
    progress_contents = [msg["message"] for msg in app_messages if "Step" in msg.get("message", "")]
    assert len(progress_contents) == 3, "Should have 3 step messages"
    
    # Stop and cleanup
    await controller.stop(started_app["id"])
    await controller.uninstall(app_info["id"])
    await api.disconnect()


async def test_app_no_wait_for_service_with_disconnect(minio_server, fastapi_server, test_user_token):
    """Test that when wait_for_service is False, the app starts in detached mode regardless of failures."""
    api = await connect_to_server({
        "name": "test-nowait-client", 
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    controller = await api.get_service("public/server-apps")
    
    progress_messages = []
    
    def progress_callback(msg):
        progress_messages.append(msg)
    
    # Install an app that disconnects immediately without registering services
    app_source = '''
    // App that fails immediately without registering any service
    (async () => {
        await api.log({type: "error", content: "Critical initialization error"});
        await api.disconnect();
    })();
    '''
    
    # Install the app without testing since it's designed to fail
    app_info = await controller.install(
        source=app_source,
        app_id="no-service-failure-app",
        manifest={
            "name": "No Service Failure App",
            "type": "web-worker",
            "version": "1.0.0"
        },
        overwrite=True,
        stage=True,  # Don't test the app during installation
        progress_callback=progress_callback
    )
    
    # Don't commit - we'll test directly from stage since the app is designed to fail
    
    # Start the app with wait_for_service=False - it should start in detached mode
    # and return immediately without waiting, even though the app will fail
    progress_messages.clear()
    
    # Record start time
    start_time = time.time()
    
    # This should NOT raise an exception because we're not waiting
    result = await controller.start(
        app_info["id"],
        wait_for_service=False,  # Don't wait for service - detached mode
        timeout=5,
        stage=True,  # Start from the staged version
        progress_callback=progress_callback
    )
    
    # Record end time
    elapsed_time = time.time() - start_time
    
    # IMPORTANT: Check that we returned quickly since we're in detached mode
    assert elapsed_time < 2, f"App should have started immediately in detached mode (< 2s), but took {elapsed_time:.2f}s"
    
    # The result should indicate the app started (even though it will fail later)
    assert result is not None, "Should have returned a result for detached start"
    assert "id" in result, "Result should contain app ID"
    
    # Check that we got the success message for detached mode
    success_messages = [msg for msg in progress_messages if msg["type"] == "success" and "detached" in msg.get("message", "").lower()]
    assert len(success_messages) > 0, "Should have received success message for detached mode"
    
    # Cleanup
    await controller.uninstall(app_info["id"])
    await api.disconnect()


async def test_early_failure_vs_timeout(minio_server, fastapi_server, test_user_token):
    """Test that early disconnect is significantly faster than timeout."""
    api = await connect_to_server({
        "name": "test-timing-client", 
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    controller = await api.get_service("public/server-apps")
    
    # Test 1: App that disconnects early after error
    app_source_early_fail = '''
    api.export({
        async setup() {
            await api.log("Starting...");
            await new Promise(resolve => setTimeout(resolve, 500)); // Wait 0.5 seconds
            await api.log({type: "error", content: "Early failure detected!"});
            await api.disconnect(); // Disconnect immediately after error
        }
    })
    '''
    
    app_info_early = await controller.install(
        source=app_source_early_fail,
        app_id="early-fail-timing-app",
        manifest={
            "name": "Early Fail Timing App",
            "type": "web-worker",
            "version": "1.0.0"
        },
        overwrite=True,
        stage=True  # Don't test the app during installation
    )
    # Don't commit - test directly from stage
    
    # Test 2: App that just hangs (will timeout)
    app_source_hang = '''
    api.export({
        async setup() {
            await api.log("Starting...");
            // Hang forever - will cause timeout
            await new Promise(() => {});
        }
    })
    '''
    
    app_info_hang = await controller.install(
        source=app_source_hang,
        app_id="hang-timing-app",
        manifest={
            "name": "Hang Timing App",
            "type": "web-worker",
            "version": "1.0.0"
        },
        overwrite=True,
        stage=True  # Don't test the app during installation
    )
    # Don't commit - test directly from stage
    
    # Test early failure timing
    start_time_early = time.time()
    with pytest.raises(Exception):
        await controller.start(
            app_info_early["id"],
            wait_for_service="default",
            timeout=10,  # 10 second timeout
            stage=True  # Start from staged version
        )
    elapsed_early = time.time() - start_time_early
    
    # Test timeout timing  
    start_time_timeout = time.time()
    with pytest.raises(Exception):
        await controller.start(
            app_info_hang["id"],
            wait_for_service="default",
            timeout=3,  # 3 second timeout (shorter for test speed)
            stage=True  # Start from staged version
        )
    elapsed_timeout = time.time() - start_time_timeout
    
    # Verify timings
    assert elapsed_early < 2, f"Early disconnect should be fast (< 2s), but took {elapsed_early:.2f}s"
    assert elapsed_timeout >= 3, f"Timeout should take at least 3s, but took {elapsed_timeout:.2f}s"
    assert elapsed_early < elapsed_timeout / 2, f"Early disconnect ({elapsed_early:.2f}s) should be much faster than timeout ({elapsed_timeout:.2f}s)"
    
    # Cleanup
    await controller.uninstall(app_info_early["id"])
    await controller.uninstall(app_info_hang["id"])
    await api.disconnect()


async def test_progress_with_worker_selection(minio_server, fastapi_server, test_user_token):
    """Test that progress monitoring works with custom worker selection."""
    api = await connect_to_server({
        "name": "test-worker-client", 
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    controller = await api.get_service("public/server-apps")
    
    progress_messages = []
    
    def progress_callback(msg):
        progress_messages.append(msg)
    
    # Install a simple app
    app_source = '''
    api.export({
        async setup() {
            await api.log("App starting with custom worker selection...");
            await api.log({type: "success", content: "Worker selected successfully"});
        },
        async test() {
            return "works";
        }
    })
    '''
    
    app_info = await controller.install(
        source=app_source,
        app_id="worker-selection-app",
        manifest={
            "name": "Worker Selection App",
            "type": "web-worker",
            "version": "1.0.0"
        },
        overwrite=True,
        progress_callback=progress_callback
    )
    
    # Start with worker selection mode
    progress_messages.clear()
    started_app = await controller.start(
        app_info["id"],
        wait_for_service="default",
        worker_selection_mode="first",  # Use first available worker
        timeout=10,
        progress_callback=progress_callback
    )
    
    # Check we got progress messages
    app_messages = [msg for msg in progress_messages if "[App]" in msg.get("message", "")]
    assert len(app_messages) >= 2, "Should have received app progress messages"
    
    # Test the app
    app_service = await api.get_service(f"default@{app_info['id']}")
    result = await app_service.test()
    assert result == "works"
    
    # Cleanup
    await controller.stop(started_app["id"])
    await controller.uninstall(app_info["id"])
    await api.disconnect()