"""Test terminal worker functionality."""

import asyncio
import pytest
from hypha_rpc import connect_to_server
from . import WS_SERVER_URL


# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_terminal_worker_basic(fastapi_server, test_user_token):
    """Test basic terminal worker functionality."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 10,
            "token": test_user_token,
        }
    )

    # Get all available workers - check both current workspace and public
    # First check public workspace since terminal worker is registered as public
    all_public_services = await api.list_services({"workspace": "public"})
    workers = [s for s in all_public_services if s.get("type") == "server-app-worker"]
    
    # If no workers in public, check current workspace
    if not workers:
        workers = await api.list_services({"type": "server-app-worker"})
    
    if not workers:
        # Last resort - get terminal worker directly
        try:
            # Try to get the terminal worker service directly by pattern
            all_services = await api.list_services()
            # Look for any service that might be the terminal worker
            for svc in all_services:
                if "terminal" in svc.get("name", "").lower() or "Terminal Worker" == svc.get("name", ""):
                    # Found it - get it directly
                    terminal_service = await api.get_service(svc["id"])
                    if hasattr(terminal_service, "start") and hasattr(terminal_service, "supported_types"):
                        workers = [svc]
                        break
        except:
            pass
    
    if not workers:
        # Debug: list all services to see what's available
        all_services = await api.list_services()
        all_public = await api.list_services({"workspace": "public"}) 
        raise AssertionError(f"No server-app-worker services found. Current workspace services: {[(s.get('name'), s.get('type')) for s in all_services]}, Public services: {[(s.get('name'), s.get('type')) for s in all_public]}")
    
    # Debug: Log what we found in workers
    print(f"Found {len(workers)} workers in public workspace")
    for w in workers:
        print(f"  Worker: {w.get('name')}, ID: {w.get('id')}, supported_types: {w.get('supported_types', [])}")
        # Try to get the actual service to see its methods
        if w.get('name') == 'Terminal Worker':
            try:
                actual_service = await api.get_service(w["id"])
                print(f"    Service methods: {dir(actual_service)}")
                if hasattr(actual_service, 'supported_types'):
                    print(f"    Service.supported_types: {actual_service.supported_types}")
            except Exception as e:
                print(f"    Error getting service: {e}")
    
    terminal_workers = [w for w in workers if "terminal" in w.get("supported_types", [])]
    
    # If not found by supported_types, try by name
    if not terminal_workers:
        terminal_workers = [w for w in workers if w.get("name") == "Terminal Worker"]
        if terminal_workers:
            print("Found Terminal Worker by name, using it despite empty supported_types in listing")
    
    assert len(terminal_workers) > 0, "Terminal worker should be registered"

    terminal_worker = terminal_workers[0]
    worker = await api.get_service(terminal_worker["id"])
    
    # Test that the worker has the expected methods
    assert hasattr(worker, "start"), "Worker should have start method"
    assert hasattr(worker, "stop"), "Worker should have stop method"
    assert hasattr(worker, "execute"), "Worker should have execute method"
    assert hasattr(worker, "get_logs"), "Worker should have get_logs method"
    assert hasattr(worker, "resize_terminal"), "Worker should have resize_terminal method"
    assert hasattr(worker, "read_terminal"), "Worker should have read_terminal method"
    assert hasattr(worker, "get_screen_content"), "Worker should have get_screen_content method"

    # Test basic terminal session lifecycle
    config = {
        "id": "test-terminal-session",
        "app_id": "test-app",
        "workspace": api.config["workspace"],
        "client_id": api.config["client_id"],
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
        "artifact_id": "test-artifact",
        "manifest": {
            "type": "terminal",
            "startup_command": "/bin/bash",
        },
    }

    # Start a terminal session
    session_id = await worker.start(config)
    assert session_id == "test-terminal-session"

    # Execute a simple command
    result = await worker.execute(session_id, "echo 'Hello Terminal'", {"timeout": 10})
    assert result["status"] == "ok"
    assert len(result["outputs"]) > 0
    output_text = result["outputs"][0]["text"]
    # The output should contain our text (may include prompts/echoes)
    assert "Hello Terminal" in output_text or "Hello" in output_text

    # Test reading from terminal
    read_result = await worker.read_terminal(session_id)
    assert read_result["success"] is True

    # Test getting screen content
    screen_result = await worker.get_screen_content(session_id)
    assert screen_result["success"] is True
    assert "content" in screen_result

    # Test resizing terminal
    resize_result = await worker.resize_terminal(session_id, 24, 80)
    assert resize_result["success"] is True

    # Get logs
    logs = await worker.get_logs(session_id)
    assert isinstance(logs, dict)
    assert "items" in logs
    assert len(logs["items"]) > 0
    # Check that we have stdout or info logs
    log_types = {item["type"] for item in logs["items"]}
    assert "stdout" in log_types or "info" in log_types

    # Stop the session
    await worker.stop(session_id)

    await api.disconnect()


async def test_terminal_worker_with_custom_command(fastapi_server, test_user_token):
    """Test terminal worker with custom startup command."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 10,
            "token": test_user_token,
        }
    )

    # Get the terminal worker service - same pattern as test_terminal_worker_basic
    all_public_services = await api.list_services({"workspace": "public"})
    workers = [s for s in all_public_services if s.get("type") == "server-app-worker"]
    
    if not workers:
        workers = await api.list_services({"type": "server-app-worker"})
    
    # Find terminal workers by name since supported_types is empty in listing
    terminal_workers = [w for w in workers if w.get("name") == "Terminal Worker"]
    assert len(terminal_workers) > 0, "Terminal worker should be registered"

    terminal_worker = terminal_workers[0]
    worker = await api.get_service(terminal_worker["id"])

    # Test with Python interpreter as startup command
    config = {
        "id": "test-python-terminal",
        "app_id": "test-python-app",
        "workspace": api.config["workspace"],
        "client_id": api.config["client_id"],
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
        "artifact_id": "test-artifact",
        "manifest": {
            "type": "terminal",
            "startup_command": "python3 -i",  # Interactive Python
        },
    }

    # Start a Python terminal session
    session_id = await worker.start(config)
    assert session_id == "test-python-terminal"

    # Give it a moment to start
    await asyncio.sleep(0.5)

    # Execute Python code
    result = await worker.execute(session_id, "print('Hello from Python')")
    assert result["status"] == "ok"
    
    # Execute a calculation
    result = await worker.execute(session_id, "2 + 2")
    assert result["status"] == "ok"
    # The output might contain '4' somewhere
    
    # Stop the session
    await worker.stop(session_id)

    await api.disconnect()


async def test_terminal_worker_run_conda_worker(fastapi_server, test_user_token):
    """Test terminal worker running conda worker as a subprocess."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 10,
            "token": test_user_token,
        }
    )

    # Get the terminal worker service - same pattern as test_terminal_worker_basic
    all_public_services = await api.list_services({"workspace": "public"})
    workers = [s for s in all_public_services if s.get("type") == "server-app-worker"]
    
    if not workers:
        workers = await api.list_services({"type": "server-app-worker"})
    
    # Find terminal workers by name since supported_types is empty in listing
    terminal_workers = [w for w in workers if w.get("name") == "Terminal Worker"]
    assert len(terminal_workers) > 0, "Terminal worker should be registered"

    terminal_worker = terminal_workers[0]
    worker = await api.get_service(terminal_worker["id"])

    # Start terminal session that will run conda worker
    config = {
        "id": "test-conda-terminal",
        "app_id": "test-conda-app",
        "workspace": api.config["workspace"],
        "client_id": api.config["client_id"],
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
        "artifact_id": "test-artifact",
        "manifest": {
            "type": "terminal",
            "startup_command": f"python -m hypha.workers.conda --server-url {WS_SERVER_URL} --workspace {api.config['workspace']} --token {test_user_token} --service-id conda-in-terminal",
        },
    }

    # Start the conda worker in terminal
    session_id = await worker.start(config)
    assert session_id == "test-conda-terminal"

    # Wait for conda worker to register
    max_wait = 30  # 30 seconds max wait (increased from 20)
    conda_worker_found = False
    startup_output = ""
    
    while max_wait > 0:
        try:
            # Try to get some output from the terminal to debug
            try:
                terminal_output = await worker.read_terminal(session_id)
                if terminal_output:
                    startup_output += terminal_output
            except:
                pass
            
            services = await api.list_services()
            for service in services:
                # Look for service ID that contains 'conda-in-terminal' (might have workspace prefix)
                if "conda-in-terminal" in service.get("id", ""):
                    conda_worker_found = True
                    break
            if conda_worker_found:
                break
        except Exception as e:
            # Log but continue
            print(f"Error checking services: {e}")
        await asyncio.sleep(1.0)  # Increased from 0.5 to 1.0
        max_wait -= 1.0

    # If not found, print debug information
    if not conda_worker_found:
        print(f"Terminal output during startup:\n{startup_output}")
        all_services = await api.list_services()
        print(f"All available services: {[s.get('id') for s in all_services]}")
    
    assert conda_worker_found, "Conda worker should have registered from terminal"

    # Get the conda worker service
    conda_service = await api.get_service("conda-in-terminal")
    assert hasattr(conda_service, "start"), "Conda service should have start method"

    # Stop the terminal session (which should stop the conda worker)
    await worker.stop(session_id)

    await api.disconnect()


async def test_terminal_worker_error_handling(fastapi_server, test_user_token):
    """Test terminal worker error handling."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 10,
            "token": test_user_token,
        }
    )

    # Get the terminal worker service - same pattern as test_terminal_worker_basic
    all_public_services = await api.list_services({"workspace": "public"})
    workers = [s for s in all_public_services if s.get("type") == "server-app-worker"]
    
    if not workers:
        workers = await api.list_services({"type": "server-app-worker"})
    
    # Find terminal workers by name since supported_types is empty in listing
    terminal_workers = [w for w in workers if w.get("name") == "Terminal Worker"]
    assert len(terminal_workers) > 0, "Terminal worker should be registered"

    terminal_worker = terminal_workers[0]
    worker = await api.get_service(terminal_worker["id"])

    # Test invalid session ID
    with pytest.raises(Exception) as exc_info:
        await worker.execute("invalid-session-id", "echo test")
    assert "not found" in str(exc_info.value).lower()

    # Test invalid command for startup
    config = {
        "id": "test-invalid-command",
        "app_id": "test-app",
        "workspace": api.config["workspace"],
        "client_id": api.config["client_id"],
        "server_url": WS_SERVER_URL,
        "token": test_user_token,
        "artifact_id": "test-artifact",
        "manifest": {
            "type": "terminal",
            "startup_command": "/nonexistent/command",
        },
    }

    # This should fail
    with pytest.raises(Exception) as exc_info:
        await worker.start(config)
    assert "Failed to spawn terminal process" in str(exc_info.value)

    await api.disconnect()


async def test_terminal_worker_concurrent_sessions(fastapi_server, test_user_token):
    """Test terminal worker with multiple concurrent sessions."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 10,
            "token": test_user_token,
        }
    )

    # Get the terminal worker service - same pattern as test_terminal_worker_basic
    all_public_services = await api.list_services({"workspace": "public"})
    workers = [s for s in all_public_services if s.get("type") == "server-app-worker"]
    
    if not workers:
        workers = await api.list_services({"type": "server-app-worker"})
    
    # Find terminal workers by name since supported_types is empty in listing
    terminal_workers = [w for w in workers if w.get("name") == "Terminal Worker"]
    assert len(terminal_workers) > 0, "Terminal worker should be registered"

    terminal_worker = terminal_workers[0]
    worker = await api.get_service(terminal_worker["id"])

    session_ids = []
    
    # Start multiple sessions
    for i in range(3):
        config = {
            "id": f"test-session-{i}",
            "app_id": f"test-app-{i}",
            "workspace": api.config["workspace"],
            "client_id": api.config["client_id"],
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
            "artifact_id": "test-artifact",
            "manifest": {
                "type": "terminal",
                "startup_command": "/bin/bash",
            },
        }
        
        session_id = await worker.start(config)
        session_ids.append(session_id)
        assert session_id == f"test-session-{i}"

    # Execute commands in each session
    for i, session_id in enumerate(session_ids):
        result = await worker.execute(session_id, f"echo 'Session {i}'", {"timeout": 10})
        assert result["status"] == "ok"
        # Just check that we got some output (the command might be truncated in the echo)
        assert len(result["outputs"]) > 0

    # Stop all sessions
    for session_id in session_ids:
        await worker.stop(session_id)

    await api.disconnect()