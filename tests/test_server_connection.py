import os
import pytest
import pytest_asyncio
import asyncio
import subprocess
import time
from hypha_rpc import connect_to_server

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def parent_server(tmp_path):
    """Start a parent hypha server."""
    port = 9527
    server_id = "parent-server"
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "hypha.server",
            "--host=127.0.0.1",
            "--port=" + str(port),
            "--server-id=" + server_id,
            "--cache-dir=" + str(tmp_path),
        ]
    )
    # Wait for server to start
    for _ in range(50):
        try:
            server = await connect_to_server({"server_url": f"http://127.0.0.1:{port}"})
            await server.disconnect()
            break
        except Exception:
            await asyncio.sleep(0.1)
    else:
        raise Exception("Failed to start parent server")
    
    server_info = {"url": f"http://127.0.0.1:{port}", "process": proc, "port": port}
    try:
        yield server_info
    finally:
        proc.terminate()
        proc.wait()

@pytest_asyncio.fixture
async def child_server(parent_server, tmp_path):
    """Start a child hypha server."""
    port = parent_server["port"] + 1
    server_id = "child-server"
    # Connect to parent server and get a token
    parent = await connect_to_server({"server_url": parent_server["url"]})
    
    # Get the workspace manager service
    wm = await parent.get_service("public/manager-parent-server:default")
    
    # Create test workspace
    await wm.create_workspace({
        "id": "test-workspace",
        "name": "Test Workspace",
        "description": "Test workspace for server connection",
        "persistent": False,
        "owners": ["test-user"],
    }, overwrite=True, context={"user": {"id": "root", "roles": ["admin"], "is_anonymous": False}, "ws": "public"})
    
    token = await wm.generate_token({
        "workspace": "test-workspace",
        "permission": "admin",
        "expires_in": 3600
    }, context={"user": {"id": "root", "roles": ["admin"], "is_anonymous": False}, "ws": "test-workspace"})
    await parent.disconnect()
    
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "hypha.server",
            "--host=127.0.0.1",
            "--port=" + str(port),
            "--server-id=" + server_id,
            "--cache-dir=" + str(tmp_path),
            "--parent-server=" + parent_server["url"],
            "--parent-token=" + token,
            "--parent-workspace=test-workspace",
            "--parent-client-id=test-client",
        ]
    )
    # Wait for server to start
    for _ in range(50):
        try:
            server = await connect_to_server({"server_url": f"http://127.0.0.1:{port}"})
            await server.disconnect()
            break
        except Exception:
            await asyncio.sleep(0.1)
    else:
        raise Exception("Failed to start child server")
    
    server_info = {"url": f"http://127.0.0.1:{port}", "process": proc, "port": port}
    try:
        yield server_info
    finally:
        proc.terminate()
        proc.wait()

@pytest.mark.asyncio
async def test_server_connection(parent_server, child_server):
    """Test connecting parent and child servers."""
    # Connect to parent server
    parent = await connect_to_server({"server_url": parent_server["url"]})
    
    # Connect to child server
    child = await connect_to_server({"server_url": child_server["url"]})
    
    # Register a service in child server
    def hello(name):
        return f"Hello {name} from child server"
    
    child_service = await child.register_service({
        "id": "hello-service",
        "name": "Hello Service",
        "config": {"visibility": "public"},
        "hello": hello
    })
    
    # Try to access the child service from parent server
    # The service should be available at test-ws>test-workspace>test-client/hello-service
    parent_service = await parent.get_service("test-ws>test-workspace>test-client/hello-service")
    
    # Call the service
    result = await parent_service.hello("World")
    assert result == "Hello World from child server"
    
    # Clean up
    await parent.disconnect()
    await child.disconnect()

@pytest.mark.asyncio
async def test_invalid_token(parent_server, tmp_path):
    """Test connecting with an invalid token."""
    port = parent_server["port"] + 1
    server_id = "child-server"
    
    # Try to start child server with invalid token
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "hypha.server",
            "--host=127.0.0.1",
            "--port=" + str(port),
            "--server-id=" + server_id,
            "--cache-dir=" + str(tmp_path),
            "--parent-server=" + parent_server["url"],
            "--parent-token=invalid-token",
            "--parent-workspace=test-workspace",
            "--parent-client-id=test-client",
        ]
    )
    
    try:
        # Server should fail to start or fail to connect
        with pytest.raises(Exception):
            for _ in range(50):
                try:
                    server = await connect_to_server({"server_url": f"http://127.0.0.1:{port}"})
                    await server.disconnect()
                    break
                except Exception:
                    await asyncio.sleep(0.1)
            else:
                raise Exception("Server should not start with invalid token")
    finally:
        proc.terminate()
        proc.wait()

@pytest_asyncio.fixture
async def grandchild_server(child_server, tmp_path):
    """Start a grandchild hypha server (connected to child server)."""
    port = child_server["port"] + 1
    server_id = "grandchild-server"
    
    # Connect to child server and get a token
    child = await connect_to_server({"server_url": child_server["url"]})
    
    # Get the workspace manager service
    wm = await child.get_service("public/manager-child-server:default")
    
    # Create test workspace
    await wm.create_workspace({
        "id": "test-workspace-2",
        "name": "Test Workspace 2",
        "description": "Test workspace for server connection",
        "persistent": False,
        "owners": ["test-user"],
    }, overwrite=True, context={"user": {"id": "root", "roles": ["admin"], "is_anonymous": False}, "ws": "public"})
    
    token = await wm.generate_token({
        "workspace": "test-workspace-2",
        "permission": "admin",
        "expires_in": 3600
    }, context={"user": {"id": "root", "roles": ["admin"], "is_anonymous": False}, "ws": "test-workspace-2"})
    await child.disconnect()
    
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "hypha.server",
            "--host=127.0.0.1",
            "--port=" + str(port),
            "--server-id=" + server_id,
            "--cache-dir=" + str(tmp_path),
            "--parent-server=" + child_server["url"],
            "--parent-token=" + token,
            "--parent-workspace=test-workspace-2",
            "--parent-client-id=test-client-2",
        ]
    )
    
    # Wait for server to start
    for _ in range(50):
        try:
            server = await connect_to_server({"server_url": f"http://127.0.0.1:{port}"})
            await server.disconnect()
            break
        except Exception:
            await asyncio.sleep(0.1)
    else:
        raise Exception("Failed to start grandchild server")
    
    server_info = {"url": f"http://127.0.0.1:{port}", "process": proc, "port": port}
    try:
        yield server_info
    finally:
        proc.terminate()
        proc.wait()

@pytest.mark.asyncio
async def test_multi_server_chain(parent_server, child_server, grandchild_server):
    """Test chaining multiple servers together."""
    # Connect to all servers
    parent = await connect_to_server({"server_url": parent_server["url"]})
    child = await connect_to_server({"server_url": child_server["url"]})
    grandchild = await connect_to_server({"server_url": grandchild_server["url"]})
    
    # Register a service in grandchild server
    def hello(name):
        return f"Hello {name} from grandchild server"
    
    grandchild_service = await grandchild.register_service({
        "id": "hello-service",
        "name": "Hello Service",
        "config": {"visibility": "public"},
        "hello": hello
    })
    
    # Try to access the grandchild service from parent server
    # The service should be available at test-ws>test-workspace>test-client>test-workspace-2>test-client-2/hello-service
    parent_service = await parent.get_service("test-ws>test-workspace>test-client>test-workspace-2>test-client-2/hello-service")
    
    # Call the service
    result = await parent_service.hello("World")
    assert result == "Hello World from grandchild server"
    
    # Clean up
    await parent.disconnect()
    await child.disconnect()
    await grandchild.disconnect()

@pytest.mark.asyncio
async def test_service_discovery(parent_server, child_server):
    """Test service discovery across servers."""
    # Connect to servers
    parent = await connect_to_server({"server_url": parent_server["url"]})
    child = await connect_to_server({"server_url": child_server["url"]})
    
    # Register multiple services in child server
    services = []
    for i in range(3):
        def make_hello(i):
            def hello(name):
                return f"Hello {name} from service {i}"
            return hello
            
        service = await child.register_service({
            "id": f"hello-service-{i}",
            "name": f"Hello Service {i}",
            "config": {"visibility": "public"},
            "hello": make_hello(i)
        })
        services.append(service)
    
    # Get the service discovery service from parent
    discovery = await parent.get_service("public/manager-parent-server:default")
    
    # List services in the child workspace
    services = await discovery.list_services(
        context={"user": {"id": "root", "roles": ["admin"], "is_anonymous": False}, "ws": "test-ws>test-workspace>test-client"}
    )
    
    # Should find all 3 services
    assert len([s for s in services if s["id"].startswith("hello-service-")]) == 3
    
    # Clean up
    await parent.disconnect()
    await child.disconnect() 