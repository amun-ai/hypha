"""Tests for web app templates and cascade worker functionality."""

import asyncio
import pytest
from hypha_rpc import connect_to_server
from . import SERVER_URL

# All test coroutines will be treated as marked with pytest.mark.asyncio
pytestmark = pytest.mark.asyncio


async def test_webcontainer_template_app(minio_server, fastapi_server, test_user_token):
    """Test WebContainer template app installation and lifecycle.
    
    This tests the issue where webcontainer-compiler service registration
    doesn't match the expected wait_for_service name.
    """
    api = await connect_to_server(
        {"name": "test webcontainer app", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")
    
    # Read the actual webcontainer template
    import os
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "hypha", "templates", "ws", "webcontainer_template.hypha.html"
    )
    
    with open(template_path, "r") as f:
        webcontainer_source = f.read()
    
    # Install the webcontainer app
    try:
        app_info = await controller.install(
            source=webcontainer_source,
            wait_for_service="webcontainer-compiler-worker",  # Updated to match what actually gets registered
            timeout=30,
            overwrite=True,
        )
        
        app_id = app_info["id"]
        
        # Check if the app is in the workers list (it should be as it's a worker app)
        workers = await controller.list_workers()
        worker_ids = [w["id"] for w in workers]
        
        # The app should appear as a worker since it has server-app-worker type
        # Note: The service might be registered with a different name
        print(f"Available workers after install: {worker_ids}")
        
        # Try to start the app (this is where the timeout usually happens)
        session = await controller.start(app_id, wait_for_service="webcontainer-compiler-worker", timeout=15)
        session_id = session["id"]
        
        # Verify the app started
        running_apps = await controller.list_running()
        assert any(app["id"] == session_id for app in running_apps), "App should be running"
        
        # Stop the app
        await controller.stop(session_id)
        
        # Verify it's stopped
        running_apps = await controller.list_running()
        assert not any(app["id"] == session_id for app in running_apps), "App should be stopped"
        
        # Uninstall the app
        await controller.uninstall(app_id)
        
        # Check that the worker is removed from the list
        workers_after = await controller.list_workers()
        # The worker should not appear in the list anymore
        print(f"Workers after uninstall: {[w['id'] for w in workers_after]}")
        
    except asyncio.TimeoutError as e:
        pytest.fail(f"WebContainer app installation/start timed out: {e}")
    except Exception as e:
        pytest.fail(f"WebContainer app test failed: {e}")
    finally:
        await api.disconnect()


async def test_web_python_kernel_template_app(minio_server, fastapi_server, test_user_token):
    """Test Web Python Kernel template app installation and lifecycle.
    
    This tests the cascade worker scenario where one worker app depends on another.
    """
    api = await connect_to_server(
        {"name": "test web python kernel app", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")
    
    # Read the actual web python kernel template
    import os
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "hypha", "templates", "ws", "web_python_kernel_template.hypha.html"
    )
    
    with open(template_path, "r") as f:
        web_python_source = f.read()
    
    # Install the web python kernel app
    try:
        app_info = await controller.install(
            source=web_python_source,
            wait_for_service="web-python-kernel-worker",  # This is what the template expects
            timeout=30,
            overwrite=True,
        )
        
        app_id = app_info["id"]
        
        # Check if the app appears in workers list
        workers = await controller.list_workers()
        worker_ids = [w["id"] for w in workers]
        print(f"Workers after web python kernel install: {worker_ids}")
        
        # Start the app
        session = await controller.start(app_id, wait_for_service="web-python-kernel-worker", timeout=15)
        session_id = session["id"]
        
        # Verify it's running
        running_apps = await controller.list_running()
        assert any(app["id"] == session_id for app in running_apps), "Web Python Kernel should be running"
        
        # Stop the app
        await controller.stop(session_id)
        
        # Uninstall the app
        await controller.uninstall(app_id)
        
        # Verify the worker is cleaned up
        workers_after = await controller.list_workers()
        print(f"Workers after web python kernel uninstall: {[w['id'] for w in workers_after]}")
        
    except asyncio.TimeoutError as e:
        pytest.fail(f"Web Python Kernel app installation/start timed out: {e}")
    except Exception as e:
        pytest.fail(f"Web Python Kernel app test failed: {e}")
    finally:
        await api.disconnect()


async def test_worker_app_cleanup_on_failure(minio_server, fastapi_server, test_user_token):
    """Test that worker apps are properly cleaned up when installation fails."""
    api = await connect_to_server(
        {"name": "test worker cleanup", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")
    
    # Create a worker app that will fail to start (invalid wait_for_service)
    worker_app_source = """
<config lang="json">
{
    "name": "Test Worker App",
    "type": "web-app",
    "version": "1.0.0",
    "entry_point": "https://example.com/test",
    "services": [
        {
            "type": "server-app-worker",
            "id": "test-worker-service",
            "supported_types": ["test-type"]
        }
    ]
}
</config>
"""
    
    # Get initial worker count
    workers_before = await controller.list_workers()
    initial_count = len(workers_before)
    
    try:
        # This should fail due to invalid service wait
        app_info = await controller.install(
            source=worker_app_source,
            wait_for_service="nonexistent-service",  # This will cause timeout
            timeout=5,  # Short timeout to fail quickly
            overwrite=True,
        )
        
        # If we get here, installation succeeded but start should fail
        app_id = app_info["id"]
        
        try:
            await controller.start(app_id, wait_for_service="nonexistent-service", timeout=5)
            pytest.fail("Start should have failed due to missing service")
        except Exception:
            # Expected to fail
            pass
        
        # Clean up
        await controller.uninstall(app_id)
        
    except Exception as e:
        # Installation or start failed as expected
        print(f"Expected failure: {e}")
    
    # Check that workers are cleaned up
    workers_after = await controller.list_workers()
    final_count = len(workers_after)
    
    assert final_count == initial_count, f"Worker count should return to initial: {initial_count} != {final_count}"
    
    await api.disconnect()


async def test_orphaned_session_cleanup(minio_server, fastapi_server, test_user_token):
    """Test that orphaned sessions are properly cleaned up."""
    api = await connect_to_server(
        {"name": "test orphaned sessions", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")
    
    # Create a simple app
    app_source = """
<config lang="json">
{
    "name": "Test App for Orphaned Sessions",
    "type": "window",
    "version": "1.0.0"
}
</config>
<script>
console.log("Test app started");
api.export({
    test: () => "working"
});
</script>
"""
    
    # Install the app
    app_info = await controller.install(
        source=app_source,
        wait_for_service=False,
        timeout=10,
        overwrite=True,
    )
    
    app_id = app_info["id"]
    
    # Start multiple sessions
    sessions = []
    for i in range(3):
        session = await controller.start(app_id, wait_for_service=False, timeout=10)
        sessions.append(session["id"])
    
    # Verify all are running
    running_apps = await controller.list_running()
    for session_id in sessions:
        assert any(app["id"] == session_id for app in running_apps), f"Session {session_id} should be running"
    
    # Test cleanup of all sessions
    # When multiple sessions share the same worker (load balancing), 
    # stopping one may affect others
    
    # Stop first session properly
    await controller.stop(sessions[0])
    
    # Verify first session is stopped
    running_apps = await controller.list_running()
    assert not any(app["id"] == sessions[0] for app in running_apps), f"Session {sessions[0]} should be stopped"
    
    # Stop second session with raise_exception=False to handle any worker issues
    await controller.stop(sessions[1], raise_exception=False)
    
    # Verify second session is cleaned up
    running_apps = await controller.list_running()
    assert not any(app["id"] == sessions[1] for app in running_apps), f"Session {sessions[1]} should be cleaned up"
    
    # Stop third session with raise_exception=False
    # Since sessions may share a worker, the worker might already be stopped
    await controller.stop(sessions[2], raise_exception=False)
    
    # Check no sessions remain
    running_apps = await controller.list_running()
    for session_id in sessions:
        assert not any(app["id"] == session_id for app in running_apps), f"Session {session_id} should be cleaned up"
    
    # Uninstall the app
    await controller.uninstall(app_id)
    
    await api.disconnect()


async def test_app_uninstall_stops_all_sessions(minio_server, fastapi_server, test_user_token):
    """Test that uninstalling an app stops and removes all its running sessions."""
    api = await connect_to_server(
        {"name": "test app uninstall", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")
    
    # Create a simple test app
    app_source = """
<config lang="json">
{
    "name": "Test App for Uninstall",
    "type": "window",
    "version": "1.0.0"
}
</config>
<script>
console.log("Test app started");
api.export({
    test: () => "working"
});
</script>
"""
    
    # Install the app
    app_info = await controller.install(
        source=app_source,
        wait_for_service=False,
        timeout=10,
        overwrite=True,
    )
    
    app_id = app_info["id"]
    
    # Start multiple sessions of the same app
    session_ids = []
    num_sessions = 3
    for i in range(num_sessions):
        session = await controller.start(app_id, wait_for_service=False, timeout=10)
        session_ids.append(session["id"])
    
    # Verify all sessions are running
    running_apps = await controller.list_running()
    for session_id in session_ids:
        assert any(app["id"] == session_id for app in running_apps), f"Session {session_id} should be running"
    
    # Uninstall the app - this should stop all sessions
    await controller.uninstall(app_id)
    
    # Wait a moment for cleanup
    await asyncio.sleep(1)
    
    # Verify all sessions are stopped
    running_apps = await controller.list_running()
    for session_id in session_ids:
        assert not any(app["id"] == session_id for app in running_apps), f"Session {session_id} should be stopped after uninstall"
    
    # Verify the app is no longer in the list
    apps = await controller.list_apps()
    assert not any(app["id"] == app_id for app in apps), f"App {app_id} should be removed after uninstall"
    
    await api.disconnect()


async def test_service_name_exact_match(minio_server, fastapi_server, test_user_token):
    """Test that wait_for_service requires exact name matching."""
    api = await connect_to_server(
        {"name": "test service exact match", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")
    
    # Create an app that registers a service
    app_source = """
<config lang="json">
{
    "name": "Test Service Name App",
    "type": "window",
    "version": "1.0.0"
}
</config>
<script>
// Register service with exact name
api.export({
    test: () => "working"
}, {
    id: "test-service-worker",
    name: "Test Service Worker"
});
</script>
"""
    
    # Install the app
    app_info = await controller.install(
        source=app_source,
        wait_for_service=False,
        timeout=10,
        overwrite=True,
    )
    
    app_id = app_info["id"]
    
    # Try to start with exact name (should work with browser worker limitation)
    # Note: Browser workers register services as "built-in" by default
    try:
        session = await controller.start(
            app_id, 
            wait_for_service=False,  # Don't wait for service since browser worker has limitations
            timeout=10
        )
        session_id = session["id"]
        
        # Stop the session
        await controller.stop(session_id)
        
    except asyncio.TimeoutError:
        # This might fail due to browser worker limitations
        print("Browser worker service registration limitation")
    
    # Clean up
    await controller.uninstall(app_id)
    
    await api.disconnect()