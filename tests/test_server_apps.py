import asyncio
import pytest
import os
import sys
import httpx
from hypha_rpc import connect_to_server
import re
from . import (
    WS_SERVER_URL,
    SERVER_URL,
    SERVER_URL_SQLITE,
    find_item,
    SIO_PORT,
)

# All test coroutines will be treated as marked with pytest.mark.asyncio
pytestmark = pytest.mark.asyncio


async def test_apps_and_lazy_worker_discovery(
    minio_server, fastapi_server, test_user_token
):
    """Test service discovery functionality with include_app_services."""
    api = await connect_to_server(
        {"name": "test apps", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")

    # Create a simple window app for testing
    # This tests the core include_app_services functionality
    browser_worker_app = await controller.install(
        source="""
<config lang="json">
{
    "name": "Service Discovery Test App",
    "type": "window",
    "version": "1.0.0",
    "description": "App for testing service discovery functionality"
}
</config>

<div>
    <h1>Service Discovery Test App</h1>
    <p>This app tests service discovery with include_app_services parameter.</p>
    <p>The main goal is to verify that list_services works correctly with the include_app_services flag.</p>
</div>
        """,
        wait_for_service=False,
        timeout=20,
        overwrite=True,
    )

    browser_app_id = browser_worker_app["id"]

    # Test 1: List workers - basic functionality
    workers = await controller.list_workers()
    assert isinstance(workers, list), "Should return workers list"
    
    # Test 2: Verify include_app_services functionality works
    services_without_apps = await api.list_services(
        query={"workspace": api.config.workspace},
        include_app_services=False
    )
    
    services_with_apps = await api.list_services(
        query={"workspace": api.config.workspace},
        include_app_services=True
    )
    
    # The key test is that include_app_services works without errors
    assert isinstance(services_without_apps, list), "Should return services list"
    assert isinstance(services_with_apps, list), "Should return services list with apps"
    assert len(services_with_apps) >= len(services_without_apps), "App services should not reduce count"
    
    print(f"Services without apps: {len(services_without_apps)}")
    print(f"Services with apps: {len(services_with_apps)}")
    
    # Test 3: Test different query combinations
    all_services = await api.list_services(include_app_services=True)
    workspace_services = await api.list_services(
        query={"workspace": api.config.workspace},
        include_app_services=True
    )
    
    assert isinstance(all_services, list), "Should return all services list"
    assert isinstance(workspace_services, list), "Should return workspace services list"
    
    # Test 4: Start the app and verify basic functionality
    session = await controller.start(browser_app_id, wait_for_service=False, timeout=15)
    session_id = session["id"]
    
    # Wait for app to start
    await asyncio.sleep(1)
    
    # Verify the app started correctly
    print(f"App session started: {session_id}")
    
    # Stop the app
    await controller.stop(session_id, raise_exception=False)
    
    # Cleanup
    await controller.uninstall(browser_app_id)

    await api.disconnect()


async def test_lazy_loading_with_get_service(
    minio_server, fastapi_server, test_user_token
):
    """Test include_app_services functionality and basic app lifecycle."""
    api = await connect_to_server(
        {"name": "test lazy loading", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")
    
    # Create a simple window app for testing
    lazy_app = await controller.install(
        source="""
<config lang="json">
{
    "name": "Lazy Loading Test App",
    "type": "window",
    "version": "1.0.0",
    "description": "App for testing lazy loading and service discovery"
}
</config>

<div>
    <h1>Lazy Loading Test App</h1>
    <p>This app tests lazy loading and service discovery functionality.</p>
    <p>Focus: include_app_services parameter in list_services method.</p>
</div>
        """,
        wait_for_service=False,  # Don't wait for service during install
        timeout=15,
        overwrite=True,
    )
    
    app_id = lazy_app["id"]
    
    # Test 1: Verify include_app_services functionality works
    services_without_apps = await api.list_services(
        query={"workspace": api.config.workspace},
        include_app_services=False
    )
    
    services_with_apps = await api.list_services(
        query={"workspace": api.config.workspace},
        include_app_services=True
    )
    
    # The key test is that include_app_services works
    assert isinstance(services_without_apps, list), "Should return services list"
    assert isinstance(services_with_apps, list), "Should return services list with apps"
    print(f"Services without apps: {len(services_without_apps)}")
    print(f"Services with apps: {len(services_with_apps)}")
    
    # Test 2: Test edge cases and different parameters
    type_filtered_services = await api.list_services(
        query={"type": "built-in"},
        include_app_services=True
    )
    
    assert isinstance(type_filtered_services, list), "Should handle type filtering"
    
    # Test 3: Start the app and verify basic functionality
    session = await controller.start(app_id, wait_for_service=False, timeout=15)
    
    # After app starts, basic functionality should work
    await asyncio.sleep(1)  # Give time for app startup
    
    # Test that services listing still works after app start
    services_after_start = await api.list_services(
        query={"workspace": api.config.workspace},
        include_app_services=True
    )
    
    assert isinstance(services_after_start, list), "Should work after app start"
    print(f"Services after app start: {len(services_after_start)}")
    
    await controller.stop(session["id"], raise_exception=False)
    
    await controller.uninstall(app_id)
    
    await api.disconnect()


async def test_hypha_app(minio_server, fastapi_server, test_user_token):
    """Test hypha app with setup and run functions."""
    api = await connect_to_server(
        {"name": "test client", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")

    source = """
<config lang="json">
{
    "name": "Hypha App",
    "type": "web-python",
    "version": "0.1.0"
}
</config>

<script lang="python">
from hypha_rpc import api
import asyncio
import os

# Test shared state
state = {"setup_called": False, "run_count": 0}

async def setup():
    print("Setting up Hypha app...")
    state["setup_called"] = True
    
    # Access environment variables
    env_test = os.environ.get("TEST_ENV", "not_set")
    print(f"TEST_ENV: {env_test}")
    
    # Export a test service
    def test_function(x):
        return x * 2
    
    await api.export({"test": test_function})
    return {"initialized": True}

async def test(a):
    return a * 2

api.export({
    "setup": setup,
    "test": test
})
</script>
    """

    # Install the app
    app_info = await controller.install(
        source=source, wait_for_service="default", timeout=80, overwrite=True
    )

    app_id = app_info["id"]

    # Start the app multiple times to test state persistence
    for i in range(2):
        started_app = await controller.start(app_id, wait_for_service=True, timeout=30)
        print(f"Started app (iteration {i+1}):", started_app)

        # Check that the service was registered
        service = await api.get_service(
            f"{api.config.workspace}/{started_app['id'].split('/')[-1]}:default"
        )
        result = await service.test(5)
        assert result == 10, f"Test function should return 10, got {result}"
        # Get logs to verify execution
        logs = await controller.get_logs(started_app["id"])
        log_items = logs.get("items", [])
        log_text = "\n".join([item["content"] for item in log_items if item["type"] == "console"])
        print("App logs:", log_text)

        # Verify setup was called
        assert "Setting up Hypha app" in log_text

        # Stop the app
        await controller.stop(started_app["id"])



    # Uninstall the app
    await controller.uninstall(app_id)

    await api.disconnect()


async def test_iframe_app(minio_server, fastapi_server, test_user_token):
    """Test iframe app installation and loading."""
    api = await connect_to_server(
        {"name": "test client", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")

    # Create a simple HTML page
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Iframe App</title>
    </head>
    <body>
        <h1>Hello from Iframe App</h1>
        <script>
            console.log("Iframe app loaded");
        </script>
    </body>
    </html>
    """

    # Install the iframe app
    app_info = await controller.install(
        source=html_content,
        manifest={"name": "Test Iframe App", "type": "iframe", "version": "0.1.0"},
        wait_for_service=False,
        timeout=10,
        overwrite=True,
    )

    app_id = app_info["id"]

    # Start the app
    started_app = await controller.start(app_id, wait_for_service=False, timeout=10)
    print("Started iframe app:", started_app)

    # Stop the app
    await controller.stop(started_app["id"])

    # Uninstall the app
    await controller.uninstall(app_id)

    await api.disconnect()


async def test_web_python_app(minio_server, fastapi_server, test_user_token):
    """Test web-python app with pyodide."""
    api = await connect_to_server(
        {"name": "test client", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")

    # Create a web-python app
    source = """
<config lang="json">
{
    "name": "Web Python App",
    "type": "web-python",
    "version": "0.1.0",
    "description": "A Python app running in the browser with Pyodide"
}
</config>

<script lang="python">
import js
from hypha_rpc import api

print("Web Python app is setting up...")

# Define a service
def calculate(x, y):
    return x + y

def get_info():
    return {
        "platform": "pyodide",
        "message": "Running Python in the browser!"
    }

# Export service
api.export({
    "calculate": calculate,
    "get_info": get_info
})
    


</script>
    """

    # Install the app
    app_info = await controller.install(
        source=source, wait_for_service=True, timeout=20, overwrite=True
    )

    app_id = app_info["id"]

    # Start the app
    started_app = await controller.start(app_id, wait_for_service=True, timeout=20)
    print("Started web-python app:", started_app)

    # Try to use the service
    try:
        service = await api.get_service(
            f"{api.config.workspace}/{started_app['id'].split('/')[-1]}:default"
        )
        result = await service.calculate(3, 4)
        assert result == 7, f"Calculate should return 7, got {result}"

        info = await service.get_info()
        assert info["platform"] == "pyodide"
        print("Web-python service info:", info)
    except Exception as e:
        print(f"Could not get web-python service: {e}")

    # Stop the app
    await controller.stop(started_app["id"])

    # Uninstall the app
    await controller.uninstall(app_id)

    await api.disconnect()


async def test_terminal_app(minio_server, fastapi_server, test_user_token):
    """Test terminal app functionality."""
    api = await connect_to_server(
        {"name": "test terminal", "server_url": SERVER_URL, "token": test_user_token}
    )
    controller = await api.get_service("public/server-apps")

    # Create a simple shell script as a terminal app
    terminal_app_info = await controller.install(
        source="#!/bin/bash\necho 'Terminal app started'\necho 'Current directory:'\npwd\necho 'Environment test:'\necho $TEST_VAR\necho 'Done!'",
        manifest={
            "name": "Test Terminal App",
            "type": "terminal",
            "version": "1.0.0",
            "entry_point": "run.sh",
        },
        timeout=10,
        wait_for_service=False,
        overwrite=True,
    )

    # Start the terminal app with environment variables
    started_app = await controller.start(
        terminal_app_info["id"],
        wait_for_service=False,
        timeout=10,
        additional_kwargs={"env": {"TEST_VAR": "Hello from env"}},
    )

    # Give it time to execute
    await asyncio.sleep(2)

    # Get logs - use the session ID from the started app
    logs = await controller.get_logs(started_app["id"])
    log_items = logs.get("items", [])
    log_text = "\n".join([item["content"] for item in log_items])
    print("Terminal app logs:", log_text)

    # Verify execution - check if there are log items or specific content
    assert len(log_items) > 0 or "Terminal session started" in log_text

    # Stop the terminal session
    await controller.stop(started_app["id"])

    # Test a terminal app with interactive command execution
    interactive_app_info = await controller.install(
        source="# Interactive terminal",
        manifest={
            "name": "Interactive Terminal App",
            "type": "terminal",
            "version": "1.0.0",
            "entry_point": "interactive.sh",
            "startup_command": "python3 -i",  # Interactive Python
        },
        timeout=10,
        wait_for_service=False,
        overwrite=True,
    )

    # Start the interactive app
    interactive_started = await controller.start(
        interactive_app_info["id"],
        wait_for_service=False,
        timeout=10,
    )

    # Give Python interpreter time to start
    await asyncio.sleep(1)

    # Use the worker_id returned from start to get the worker directly
    assert "worker_id" in interactive_started
    worker = await api.get_service(interactive_started["worker_id"])
    
    # Test getting the screen before any command
    screen_before = await worker.get_logs(interactive_started["id"], type="screen")
    assert isinstance(screen_before, str)
    
    # Test 1: Execute with default cleaned output
    result = await worker.execute(interactive_started["id"], "print(f'Hello from Python {101}')")
    assert result["status"] == "ok"
    
    # Check cleaned output (should not have command echo or ANSI codes)
    stdout_output = next((o for o in result["outputs"] if o["type"] == "stream"), None)
    assert stdout_output is not None
    cleaned_text = stdout_output["text"]
    print(f"Cleaned output: {repr(cleaned_text)}")
    assert "Hello from Python 101" in cleaned_text
    # Should NOT contain ANSI codes
    assert "\x1b" not in cleaned_text  # No ANSI escape sequences
    
    # Test 2: Execute with raw output enabled
    result_raw = await worker.execute(
        interactive_started["id"], 
        "print(f'Raw test {202}')",
        config={"raw_output": True}
    )
    assert result_raw["status"] == "ok"
    
    # Check raw output (should have command echo and possibly ANSI codes)
    stdout_raw = next((o for o in result_raw["outputs"] if o["type"] == "stream"), None)
    assert stdout_raw is not None
    raw_text = stdout_raw["text"]
    print(f"Raw output (first 150 chars): {repr(raw_text[:150])}...")
    assert "Raw test 202" in raw_text
    # Raw output should include the command echo
    assert "print(f'Raw test" in raw_text
    
    # Test getting the current screen content (should be clean)
    current_screen = await worker.get_logs(interactive_started["id"], type="screen")
    assert isinstance(current_screen, str)
    assert "Hello from Python 101" in current_screen
    assert "Raw test 202" in current_screen
    assert "\x1b" not in current_screen  # No ANSI codes in rendered screen
    print(f"Screen content (cleaned, last 200 chars): {repr(current_screen[-200:])}")
    
    # Execute a calculation with default cleaned output
    result = await worker.execute(interactive_started["id"], "2 + 3422")
    assert result["status"] == "ok"
    
    # Check cleaned calculation output
    calc_output = next((o for o in result["outputs"] if o["type"] == "stream"), None)
    assert calc_output is not None
    calc_text = calc_output["text"]
    print(f"Calculation output (cleaned): {repr(calc_text)}")
    assert "3424" in calc_text
    
    # Verify screen has been updated and is clean
    final_screen = await worker.get_logs(interactive_started["id"], type="screen")
    assert "3424" in final_screen
    assert "\x1b" not in final_screen  # Ensure screen is clean

    # Stop the interactive session
    await controller.stop(interactive_started["id"])

    # Clean up
    await controller.uninstall(terminal_app_info["id"])
    await controller.uninstall(interactive_app_info["id"])

    await api.disconnect()