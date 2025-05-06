# tests/test_hypha_lite.py
import asyncio
import sys
import pytest
import pytest_asyncio
import hypha_rpc as hypha
from hypha.core import UserPermission, UserInfo, WorkspaceInfo
from hypha.utils import random_id
import time
import os
import signal

# Define a module-scoped event_loop fixture.
@pytest.fixture(scope='module')
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Define a port for the test server
TEST_PORT = 9529 # Choose a port unlikely to be in use
SERVER_URL = f"http://127.0.0.1:{TEST_PORT}"

@pytest_asyncio.fixture(scope="module")
async def lite_server_process(event_loop):
    """Fixture to start and stop the hypha.lite server."""
    cmd = [
        sys.executable, # Use the same python interpreter
        "run_lite_server.py",      # New way
        "--host=127.0.0.1",
        f"--port={TEST_PORT}",
        "--reset-redis", # Ensure clean state (uses fakeredis by default)
        "--base-path=/", # Explicitly set base path if needed
        f"--public-base-url={SERVER_URL}", # Set public URL for client connection
    ]
    print(f"\nStarting server with command: {' '.join(cmd)}")

    # Use environment variable to increase log level for server if desired
    env = os.environ.copy()
    env["HYPHA_LOGLEVEL"] = "DEBUG" # Optional: for more server logs during test

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE, # Capture stdout
        stderr=asyncio.subprocess.PIPE, # Capture stderr
        env=env,
        preexec_fn=os.setsid if sys.platform != "win32" else None # Ensure termination kills group
    )

    # Wait a bit for the server to start
    # A more robust check would ping a health endpoint if we add one
    await asyncio.sleep(3) # Give server time to initialize

    # Check if process started successfully
    if process.returncode is not None:
        stdout, stderr = await process.communicate()
        print("Server stdout:\n", stdout.decode() if stdout else "N/A")
        print("Server stderr:\n", stderr.decode() if stderr else "N/A")
        pytest.fail(f"Server process failed to start with code {process.returncode}.")

    print(f"Server process started (PID: {process.pid})")

    yield SERVER_URL # Provide the URL to the tests

    # --- Teardown ---
    print(f"\nStopping server process (PID: {process.pid})...")
    if process.returncode is None: # Check if process is still running
        try:
            # Send SIGTERM to the process group
            if sys.platform != "win32":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()

            # Wait for the process to terminate
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
            print(f"Server process terminated with code {process.returncode}.")
            # Optionally print captured output
            # print("Server stdout:\n", stdout.decode() if stdout else "N/A")
            # print("Server stderr:\n", stderr.decode() if stderr else "N/A")

        except asyncio.TimeoutError:
            print(f"Server process {process.pid} did not terminate gracefully, sending SIGKILL.")
            if sys.platform != "win32":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill() # Force kill on Windows
            await process.wait() # Ensure it's cleaned up
        except ProcessLookupError:
             print(f"Server process {process.pid} already terminated.")
        except Exception as e:
            print(f"Error during server teardown: {e}")
    else:
         print(f"Server process {process.pid} already exited with code {process.returncode}.")


@pytest.mark.asyncio
async def test_lite_server_connection(lite_server_process):
    """Test if we can connect to the lite server."""
    server_url = lite_server_process
    try:
        server = await hypha.connect_to_server({"server_url": server_url, "client_id": "test-conn-client"})
        assert server is not None
        # Check basic server info if available via a public service later
        print(f"Successfully connected to lite server at {server_url}")
        # Perform a basic operation, like listing public workspaces (should include 'public')
        wm = await server.get_workspace_manager()
        assert wm is not None, "Failed to get workspace manager service proxy"
        public_workspaces = await wm.list_workspaces(context={"user": server.get_user_info(), "ws": "public"})
        print(f"Public workspaces: {public_workspaces}")
        assert any(ws['id'] == 'public' for ws in public_workspaces), "'public' workspace not found"
        await server.disconnect()
    except Exception as e:
        pytest.fail(f"Failed to connect or interact with the lite server: {e}")

@pytest.mark.asyncio
async def test_lite_workspace_management(lite_server_process):
    """Test basic workspace creation, listing, and deletion."""
    server_url = lite_server_process
    server = await hypha.connect_to_server({"server_url": server_url, "client_id": "test-ws-client"})
    assert server is not None

    wm = await server.get_workspace_manager() # Get proxy to manager service in public ws
    assert wm is not None

    test_ws_id = "lite-test-ws-" + random_id(readable=True)
    user_info = server.get_user_info() # Use the anonymous user from connection for context
    root_context = {"user": server.get_user_info(roles=["admin"]), "ws": "public"} # Need admin context for create/delete

    try:
        # 1. List initial workspaces (expecting public, maybe root)
        initial_ws = await wm.list_workspaces(context=root_context)
        print(f"Initial workspaces: {[ws['id'] for ws in initial_ws]}")
        assert isinstance(initial_ws, list)
        assert not any(ws['id'] == test_ws_id for ws in initial_ws)

        # 2. Create a new workspace
        print(f"Creating workspace: {test_ws_id}")
        ws_config = {
            "id": test_ws_id,
            "name": "Lite Test Workspace",
            "description": "A temporary workspace for testing",
            "owners": [user_info.id], # Add current user as owner
            "persistent": False
        }
        created_ws_info_dict = await wm.create_workspace(config=ws_config, context=root_context)
        created_ws_info = WorkspaceInfo.model_validate(created_ws_info_dict)
        assert created_ws_info.id == test_ws_id
        print(f"Workspace '{test_ws_id}' created.")
        # Give redis a moment
        await asyncio.sleep(0.1)

        # 3. List workspaces again, check for the new one
        current_ws = await wm.list_workspaces(context=root_context)
        print(f"Current workspaces: {[ws['id'] for ws in current_ws]}")
        assert any(ws['id'] == test_ws_id for ws in current_ws)

        # 4. Get workspace info
        fetched_info_dict = await wm.get_workspace_info(workspace=test_ws_id, context=root_context)
        fetched_info = WorkspaceInfo.model_validate(fetched_info_dict)
        assert fetched_info.id == test_ws_id
        assert fetched_info.name == "Lite Test Workspace"
        print(f"Successfully fetched info for '{test_ws_id}'.")

        # 5. Delete the workspace
        print(f"Deleting workspace: {test_ws_id}")
        await wm.delete_workspace(workspace=test_ws_id, context=root_context)
        print(f"Workspace '{test_ws_id}' deleted.")
        # Give redis a moment
        await asyncio.sleep(0.1)

        # 6. List workspaces again, verify deletion
        final_ws = await wm.list_workspaces(context=root_context)
        print(f"Final workspaces: {[ws['id'] for ws in final_ws]}")
        assert not any(ws['id'] == test_ws_id for ws in final_ws)

    finally:
        # Ensure cleanup if assertion fails mid-test
        try:
             await wm.delete_workspace(workspace=test_ws_id, context=root_context)
             print(f"Ensured cleanup of workspace '{test_ws_id}'.")
        except Exception:
             pass # Ignore if already deleted or failed
        await server.disconnect()


@pytest.mark.asyncio
async def test_lite_service_management(lite_server_process):
    """Test basic service registration, listing, and unregistration."""
    server_url = lite_server_process
    server = await hypha.connect_to_server({"server_url": server_url, "client_id": "test-svc-client"})
    assert server is not None

    wm = await server.get_workspace_manager() # Manager API
    assert wm is not None

    # Use the public workspace for this simple test
    ws_id = "public"
    test_service_id = "test-lite-service-" + random_id(readable=True)
    # The full service ID includes client ID: public/test-svc-client:test-lite-service-....
    full_service_id_pattern = f"{ws_id}/{server.client_id}:{test_service_id}"

    # Context for operations within the public workspace
    public_context = {"user": server.get_user_info(), "ws": ws_id}
    # Admin context for potentially deleting if needed (though register/unregister might not need admin in public?)
    # Let's assume the connected (anonymous) user can register/unregister in public for now
    # If permission fails, we'll need to adjust context or permissions.

    try:
        # 1. List services initially, ensure test service is not present
        initial_services = await wm.list_services(context=public_context)
        print(f"Initial public services (showing ID): {[s.get('id') for s in initial_services[:5]]}...")
        assert not any(s['id'] == test_service_id for s in initial_services) # Check just the base ID part

        # 2. Register the test service
        service_def = {
            "id": test_service_id, # Manager will prepend ws/client
            "name": "Lite Test Service",
            "type": "test",
            "description": "A dummy service for testing",
            "config": {
                "visibility": "public",
                "require_context": False,
                # Add some methods for potential future get_service test
                "my_method": lambda x: f"called with {x}",
            },
            # Actual methods are defined on the client side, this is just the definition
        }
        print(f"Registering service with base ID: {test_service_id}")
        registered_info = await wm.register_service(service=service_def, context=public_context)
        print(f"Service registered with full ID: {registered_info.get('id')}")
        assert test_service_id in registered_info['id']
        assert server.client_id in registered_info['id']
        assert registered_info['name'] == "Lite Test Service"
        # Give redis a moment
        await asyncio.sleep(0.1)

        # 3. List services again, check for the new one
        current_services = await wm.list_services(context=public_context)
        found_service = None
        for s in current_services:
            # Check if the service ID *ends* with the base ID we registered
             if s['id'].endswith(":"+test_service_id):
                  found_service = s
                  break
        print(f"Current public services (showing ID): {[s.get('id') for s in current_services[:5]]}...")
        assert found_service is not None, f"Service '{test_service_id}' not found after registration."
        assert found_service['name'] == "Lite Test Service"

        # 4. Get service info for the specific service
        # Use the full ID returned by register_service
        fetched_info = await wm.get_service_info(service_id=registered_info['id'], context=public_context)
        assert fetched_info is not None
        # fetched_info is ServiceInfo object, access attributes
        assert fetched_info.name == "Lite Test Service"
        assert fetched_info.id == registered_info['id']
        print(f"Successfully fetched info for service: {fetched_info.id}")

        # 5. Unregister the service
        print(f"Unregistering service: {registered_info['id']}")
        unregister_result = await wm.unregister_service(service_id=registered_info['id'], context=public_context)
        assert unregister_result['deleted_count'] >= 1
        print(f"Service '{registered_info['id']}' unregistered.")
        # Give redis a moment
        await asyncio.sleep(0.1)

        # 6. List services again, verify removal
        final_services = await wm.list_services(context=public_context)
        found_service_after_delete = None
        for s in final_services:
             if s['id'].endswith(":"+test_service_id):
                  found_service_after_delete = s
                  break
        print(f"Final public services (showing ID): {[s.get('id') for s in final_services[:5]]}...")
        assert found_service_after_delete is None, f"Service '{test_service_id}' still found after unregistration."

    finally:
        # Ensure cleanup
        try:
             # Use the full ID if we know it
             service_to_cleanup = f"{ws_id}/{server.client_id}:{test_service_id}"
             await wm.unregister_service(service_id=service_to_cleanup, context=public_context)
             print(f"Ensured cleanup of service '{service_to_cleanup}'.")
        except Exception:
             pass # Ignore if already deleted or failed
        await server.disconnect() 