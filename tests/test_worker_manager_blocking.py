"""Test to reproduce the WorkerManager blocking issue during app installation."""

import asyncio
import time
import pytest
from hypha_rpc import connect_to_server
from tests import WS_SERVER_URL


@pytest.mark.asyncio
async def test_worker_manager_blocking_issue(fastapi_server, test_user_token):
    """Test to reproduce the WorkerManager blocking issue during app installation.
    
    This test reproduces the issue where WorkerManager's workspace monitoring setup
    causes blocking during app installation, leading to "waiting for service 'default'"
    timeouts. The issue occurs when multiple workspace connections are established
    concurrently and the subscription system blocks worker access.
    """
    # Test app that registers a "default" service
    TEST_MCP_PROXY_APP = """
<config lang="json">
{
    "name": "Test MCP Proxy",
    "type": "mcp-proxy",
    "version": "0.1.0",
    "description": "Test MCP proxy app to reproduce WorkerManager blocking issue",
    "tags": ["mcp", "test"],
    "covers": ["*"],
    "icon": "🔧",
    "inputs": null,
    "outputs": null,
    "flags": [],
    "env": "",
    "permissions": [],
    "requirements": [],
    "dependencies": [],
    "timeout": 30,
    "servers": {
        "test-server": {
            "command": "echo",
            "args": ["Hello from test MCP server"],
            "env": {}
        }
    }
}
</config>

<script lang="python">
import asyncio
from hypha_rpc import api

async def setup():
    # Register a default service that the installation process waits for
    api.export({
        "name": "Test MCP Service",
        "id": "default",
        "type": "mcp-service",
        "config": {
            "visibility": "public",
            "run_in_executor": False
        }
    })
    await api.log("MCP proxy service registered")

if __name__ == "__main__":
    asyncio.run(setup())
</script>
"""

    # Connect to server
    api = await connect_to_server(
        {
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,  # Increased timeout to catch blocking
            "token": test_user_token,
        }
    )
    
    try:
        controller = await api.get_service("public/server-apps")
        
        # Test 1: Single app installation that should trigger WorkerManager monitoring
        print("🧪 Testing single MCP proxy app installation...")
        start_time = time.time()
        
        try:
            app_info = await controller.install(
                source=TEST_MCP_PROXY_APP,
                overwrite=True,
                wait_for_service="default",  # This is where the blocking occurs
                timeout=15,  # Shorter timeout to catch the issue faster
            )
            
            installation_time = time.time() - start_time
            print(f"✅ App installed successfully in {installation_time:.2f}s: {app_info.id}")
            
            # Try to start the app
            start_time = time.time()
            started_app = await controller.start(app_info.id, wait_for_service="default")
            start_time_elapsed = time.time() - start_time
            print(f"✅ App started successfully in {start_time_elapsed:.2f}s")
            
            # Clean up this app
            await controller.uninstall(app_info.id)
            
        except asyncio.TimeoutError as e:
            installation_time = time.time() - start_time
            print(f"❌ REPRODUCED ISSUE: App installation timed out after {installation_time:.2f}s")
            print(f"   Error: {e}")
            raise AssertionError("Successfully reproduced the WorkerManager blocking issue!")
        
        except Exception as e:
            installation_time = time.time() - start_time
            print(f"❌ REPRODUCED ISSUE: App installation failed after {installation_time:.2f}s")
            print(f"   Error: {e}")
            if "waiting for service" in str(e).lower() or "timeout" in str(e).lower():
                raise AssertionError("Successfully reproduced the WorkerManager blocking issue!")
            raise
        
        # Test 2: Concurrent app installations to stress test WorkerManager
        print("🧪 Testing concurrent MCP proxy app installations...")
        
        async def install_concurrent_app(app_suffix):
            """Install an app concurrently to stress test the WorkerManager"""
            try:
                concurrent_app_code = TEST_MCP_PROXY_APP.replace(
                    '"Test MCP Proxy"', f'"Test MCP Proxy {app_suffix}"'
                ).replace(
                    'test-server', f'test-server-{app_suffix}'
                )
                
                start_time = time.time()
                app_info = await controller.install(
                    source=concurrent_app_code,
                    overwrite=True,
                    wait_for_service="default",
                    timeout=20,
                )
                installation_time = time.time() - start_time
                print(f"✅ Concurrent app {app_suffix} installed in {installation_time:.2f}s")
                return app_info
                
            except Exception as e:
                installation_time = time.time() - start_time
                print(f"❌ Concurrent app {app_suffix} failed after {installation_time:.2f}s: {e}")
                if "waiting for service" in str(e).lower() or "timeout" in str(e).lower():
                    return f"BLOCKING_ISSUE_{app_suffix}"
                raise
        
        # Launch 3 concurrent installations
        start_time = time.time()
        concurrent_tasks = [
            install_concurrent_app(f"concurrent_{i}")
            for i in range(3)
        ]
        
        try:
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            total_time = time.time() - start_time
            print(f"🏁 Concurrent installations completed in {total_time:.2f}s")
            
            # Check for blocking issues
            blocking_issues = [r for r in results if isinstance(r, str) and "BLOCKING_ISSUE" in r]
            if blocking_issues:
                print(f"❌ REPRODUCED ISSUE: {len(blocking_issues)} concurrent installations had blocking issues")
                raise AssertionError("Successfully reproduced the WorkerManager blocking issue in concurrent scenario!")
            
            # Clean up successful installations
            successful_apps = [r for r in results if hasattr(r, 'id')]
            for app_info in successful_apps:
                try:
                    await controller.uninstall(app_info.id)
                except Exception as e:
                    print(f"⚠️  Failed to clean up {app_info.id}: {e}")
                    
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ REPRODUCED ISSUE: Concurrent installations failed after {total_time:.2f}s")
            print(f"   Error: {e}")
            if "waiting for service" in str(e).lower() or "timeout" in str(e).lower():
                raise AssertionError("Successfully reproduced the WorkerManager blocking issue in concurrent scenario!")
            raise
        
        print("✅ All tests passed - WorkerManager blocking issue was NOT reproduced")
        print("   This suggests the issue may be environment-specific or require different conditions")
        
    finally:
        await api.disconnect()


# Additional test with a simpler approach to trigger the issue
@pytest.mark.asyncio 
async def test_simple_worker_manager_race_condition(fastapi_server, test_user_token):
    """Simple test to trigger the WorkerManager race condition."""
    
    # Simple Python app that should register quickly
    SIMPLE_PYTHON_APP = """
<config lang="json">
{
    "name": "Simple Test App",
    "type": "web-python",
    "version": "0.1.0",
    "description": "Simple app to test WorkerManager race condition"
}
</config>

<script lang="python">
from hypha_rpc import api

# Register the default service immediately
api.export({
    "name": "Simple Service",
    "id": "default",
    "test": lambda: "Hello World"
})

api.log("Simple service registered")
</script>
"""

    api = await connect_to_server({
        "server_url": WS_SERVER_URL,
        "method_timeout": 20,
        "token": test_user_token,
    })
    
    try:
        controller = await api.get_service("public/server-apps")
        
        # Try to install multiple apps quickly to trigger race conditions
        print("🧪 Testing rapid successive app installations...")
        
        for i in range(3):
            start_time = time.time()
            try:
                app_code = SIMPLE_PYTHON_APP.replace(
                    "Simple Test App", f"Simple Test App {i}"
                )
                
                app_info = await controller.install(
                    source=app_code,
                    overwrite=True,
                    wait_for_service="default",
                    timeout=10,  # Short timeout to catch issues
                )
                
                installation_time = time.time() - start_time
                print(f"✅ App {i} installed in {installation_time:.2f}s: {app_info.id}")
                
                # Clean up immediately
                await controller.uninstall(app_info.id)
                
            except Exception as e:
                installation_time = time.time() - start_time
                print(f"❌ App {i} failed after {installation_time:.2f}s: {e}")
                
                if ("waiting for service" in str(e).lower() or 
                    "timeout" in str(e).lower() or
                    "connection" in str(e).lower()):
                    print(f"🎯 REPRODUCED ISSUE: WorkerManager blocking detected on app {i}!")
                    raise AssertionError(f"Successfully reproduced WorkerManager blocking on app {i}: {e}")
                
                raise
        
        print("✅ All rapid installations succeeded - issue not reproduced in this scenario")
        
    finally:
        await api.disconnect()