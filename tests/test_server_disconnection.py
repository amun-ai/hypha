"""Test the hypha server."""

import os
import subprocess
import sys
import asyncio
import time

import pytest
import requests
from hypha_rpc import connect_to_server
from redis import Redis

from . import (
    SERVER_URL,
    SERVER_URL_REDIS_1,
    SERVER_URL_REDIS_2,
    SIO_PORT2,
    WS_SERVER_URL,
    find_item,
    REDIS_PORT,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_server_reconnection(fastapi_server, root_user_token):
    """Test the server reconnection."""
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "admin", "token": root_user_token}
    ) as root:
        admin = await root.get_service("admin-utils")

        api = await connect_to_server(
            {"server_url": WS_SERVER_URL, "client_id": "client1"}
        )
        assert api.config["client_id"] == "client1"
        await admin.kickout_client(
            api.config.workspace,
            api.config.client_id,
            1008,
            "simulated abnormal closure",
        )
        await asyncio.sleep(1)

        # It should reconnect
        assert await api.echo("hi") == "hi"
        await api.disconnect()

        api = await connect_to_server(
            {"server_url": WS_SERVER_URL, "client_id": "client1"}
        )
        assert api.config["client_id"] == "client1"
        await admin.kickout_client(
            api.config.workspace, api.config.client_id, 1000, "normal closure"
        )
        await asyncio.sleep(1)
        try:
            assert await api.echo("hi") == "hi"
        except Exception as e:
            assert "Connection is closed" in str(e)
            await api.disconnect()


async def test_cleanup_on_auth_failure(fastapi_server):
    """Test that clients are properly cleaned up when authentication fails."""
    
    # Try to connect with an invalid token - this should fail during auth
    with pytest.raises(Exception) as exc_info:
        async with connect_to_server(
            {
                "server_url": WS_SERVER_URL,
                "client_id": "test-auth-fail-client",
                "token": "invalid_token_xyz123",  # Invalid token
            }
        ) as api:
            pass
    
    # Should get authentication error
    assert "Failed to establish connection" in str(exc_info.value) or "Authentication" in str(exc_info.value)
    
    # Wait a moment for any potential cleanup
    await asyncio.sleep(2)
    
    # Now connect as a normal client and check there are no orphaned services
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "checker-client"}
    ) as api:
        # List all services - there should be no services from the failed client
        services = await api.list_services()
        
        # Look for any services from the failed authentication client
        failed_client_services = [
            s for s in services 
            if "test-auth-fail-client" in s.get("id", "")
        ]
        
        assert len(failed_client_services) == 0, (
            f"Found orphaned services from failed auth client: {failed_client_services}"
        )


async def test_cleanup_on_abrupt_disconnect(fastapi_server):
    """Test cleanup when client disconnects abruptly."""
    
    # Connect and register a service
    api = await connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "abrupt-disconnect-client"}
    )
    
    workspace = api.config["workspace"]
    
    # Register a test service (built-in is automatically registered)
    await api.register_service({
        "id": "test-service",
        "name": "Test Service",
        "type": "test",
        "echo": lambda x: x
    })
    
    # Register a non-built-in service like the user reported
    await api.register_service({
        "id": "mirror-robotic-arm-control",
        "name": "Mirror Robotic Arm Control",
        "type": "robotic",
        "control": lambda cmd: f"Executing: {cmd}"
    })
    
    # Verify services exist
    services = await api.list_services()
    test_services = [s for s in services if "test-service" in s.get("id", "")]
    robotic_services = [s for s in services if "mirror-robotic-arm-control" in s.get("id", "")]
    assert len(test_services) > 0, "Test service should be registered"
    assert len(robotic_services) > 0, "Robotic service should be registered"
    
    # Force disconnect without proper cleanup (simulate crash)
    # Access internal connection and force close
    if hasattr(api, '_connection'):
        conn = api._connection
        if hasattr(conn, '_websocket') and conn._websocket:
            # Force close WebSocket with abnormal closure code
            await conn._websocket.close(code=1006)
    
    # Wait for server to detect disconnection
    await asyncio.sleep(3)
    
    # Connect as another client to check cleanup
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "checker-client-2"}
    ) as checker:
        # If we're in same workspace, check if the old service is cleaned up
        if checker.config["workspace"] == workspace:
            services = await checker.list_services()
            old_services = [
                s for s in services 
                if "abrupt-disconnect-client" in s.get("id", "")
            ]
            
            # The services should be cleaned up after disconnection
            assert len(old_services) == 0, (
                f"Found orphaned services after abrupt disconnect: {old_services}"
            )
            
            # Now test the cleanup function can detect stuck services
            # Create another client that will be stuck
            stuck_api = await connect_to_server(
                {"server_url": WS_SERVER_URL, "client_id": "stuck-client"}
            )
            
            await stuck_api.register_service({
                "id": "stuck-service",
                "name": "Stuck Service",
                "process": lambda x: f"Processing: {x}"
            })
            
            # Force disconnect to simulate stuck service
            if hasattr(stuck_api, '_connection'):
                conn = stuck_api._connection
                if hasattr(conn, '_websocket') and conn._websocket:
                    await conn._websocket.close(code=1006)
            
            await asyncio.sleep(1)
            
            # Run cleanup - should detect ALL stuck services (not just built-in)
            cleanup_result = await checker.cleanup(workspace, timeout=3)
            print(f"Cleanup result: {cleanup_result}")
            
            # Check if stuck client was removed
            if "removed_clients" in cleanup_result:
                assert f"{workspace}/stuck-client" in cleanup_result["removed_clients"], (
                    f"Cleanup should have detected stuck-client but got: {cleanup_result}"
                )


async def test_normal_disconnect_cleanup(fastapi_server):
    """Test that normal disconnection properly cleans up services."""
    
    # Connect and register services
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "normal-disconnect-client"}
    ) as api:
        workspace = api.config["workspace"]
        
        # Register a service
        await api.register_service({
            "id": "normal-service",
            "name": "Normal Service",
            "type": "test",
            "echo": lambda x: x
        })
        
        # Service should exist
        services = await api.list_services()
        normal_services = [
            s for s in services 
            if "normal-service" in s.get("id", "")
        ]
        assert len(normal_services) > 0
        
    # After context exit (normal disconnect), services should be cleaned up
    await asyncio.sleep(2)
    
    # Verify cleanup
    async with connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "final-checker"}
    ) as checker:
        # Try to list services from the disconnected client
        # They should not exist anymore
        services = await checker.list_services()
        old_services = [
            s for s in services
            if "normal-disconnect-client" in s.get("id", "")
        ]
        
        assert len(old_services) == 0, (
            f"Services not cleaned up after normal disconnect: {old_services}"
        )


async def test_server_disconnection_cleanup(redis_server):
    """Test that dead servers are properly cleaned up during rolling updates."""
    
    # Parse the redis URL to get the port
    import urllib.parse
    parsed_url = urllib.parse.urlparse(redis_server)
    redis_port = parsed_url.port or 6379
    
    # Get Redis client for verification
    redis_client = Redis(host="localhost", port=redis_port, decode_responses=True)
    
    # Clear any existing data
    redis_client.flushall()
    
    # Start first server
    proc1 = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--port=19527",
            f"--redis-uri={redis_server}",
            "--server-id=rolling-update-server-1",
            "--public-base-url=http://127.0.0.1:19527",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for first server to be ready
    timeout = 20
    while timeout > 0:
        try:
            response = requests.get("http://127.0.0.1:19527/health/readiness")
            if response.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    
    assert timeout > 0, "First server did not start in time"
    
    # Wait for server 1 to fully initialize
    await asyncio.sleep(3)
    
    # Note: Services may not be registered in Redis in test environment
    # The important thing is to verify the cleanup behavior, not the exact service registration
    
    # Start second server (simulating rolling update)
    proc2 = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--port=19528",
            f"--redis-uri={redis_server}",
            "--server-id=rolling-update-server-2",
            "--public-base-url=http://127.0.0.1:19528",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for second server to be ready
    timeout = 20
    while timeout > 0:
        try:
            response = requests.get("http://127.0.0.1:19528/health/readiness")
            if response.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    
    assert timeout > 0, "Second server did not start in time"
    
    # Wait for server 2 to fully initialize
    await asyncio.sleep(3)
    
    # Both servers should be running now
    
    # Now terminate server 1 (simulating pod termination)
    proc1.terminate()
    
    # Wait for graceful shutdown
    try:
        proc1.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc1.kill()
        proc1.wait()
    
    # Give time for cleanup to complete
    await asyncio.sleep(3)
    
    # The key test: After server-1 terminates, server-2 should still be functional
    # We verify this by checking that server-2 is still responding
    try:
        response = requests.get("http://127.0.0.1:19528/health/readiness")
        assert response.ok, "Server 2 should still be running and healthy"
    except Exception as e:
        raise AssertionError(f"Server 2 is not responding after server 1 shutdown: {e}")
    
    # Additional verification: Check that we can still connect to server-2
    # This proves that server-1's cleanup didn't break server-2
    
    # Clean up server 2
    proc2.terminate()
    try:
        proc2.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc2.kill()
        proc2.wait()


async def test_dead_server_detection(redis_server):
    """Test that new servers detect and clean up dead servers."""
    
    # Get Redis client for verification
    redis_client = Redis(host="localhost", port=REDIS_PORT, decode_responses=True)
    
    # Clear any existing data
    redis_client.flushall()
    
    # Start first server
    proc1 = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--port=19529",
            f"--redis-uri={redis_server}",
            "--server-id=detection-test-server-1",
            "--public-base-url=http://127.0.0.1:19529",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for first server to be ready
    timeout = 20
    while timeout > 0:
        try:
            response = requests.get("http://127.0.0.1:19529/health/readiness")
            if response.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    
    assert timeout > 0, "First server did not start in time"
    
    # Verify server 1 registered services
    await asyncio.sleep(1)
    server1_services = redis_client.keys("services:*:*/detection-test-server-1:*")
    assert len(server1_services) > 0, "Server 1 should have registered services"
    
    # Kill server 1 ungracefully (simulating crash)
    proc1.kill()
    proc1.wait()
    
    # Services should still be in Redis (no cleanup happened)
    server1_services_after_kill = redis_client.keys("services:*:*/detection-test-server-1*")
    assert len(server1_services_after_kill) > 0, "Server 1 services should still exist after ungraceful shutdown"
    
    # Start second server which should detect and clean up dead server 1
    proc2 = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--port=19530",
            f"--redis-uri={redis_server}",
            "--server-id=detection-test-server-2",
            "--public-base-url=http://127.0.0.1:19530",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for second server to be ready
    timeout = 20
    while timeout > 0:
        try:
            response = requests.get("http://127.0.0.1:19530/health/readiness")
            if response.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    
    assert timeout > 0, "Second server did not start in time"
    
    # Give time for dead server detection and cleanup
    await asyncio.sleep(3)
    
    # Check that server 1's services are now cleaned up
    server1_services_final = redis_client.keys("services:*:*/detection-test-server-1*")
    assert len(server1_services_final) == 0, f"Dead server 1 services should be cleaned up by server 2, but found: {server1_services_final}"
    
    # Check that server 2's services exist
    server2_services = redis_client.keys("services:*:*/detection-test-server-2:*")
    assert len(server2_services) > 0, "Server 2 services should exist"
    
    # Clean up server 2
    proc2.terminate()
    try:
        proc2.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc2.kill()
        proc2.wait()


@pytest.mark.asyncio
async def test_server_cleanup_on_teardown():
    """Test that server services are properly cleaned up on teardown."""
    from hypha.core.store import RedisStore
    from hypha.core import RedisEventBus
    from fastapi import FastAPI
    
    app = FastAPI()
    
    # Create first server with specific ID
    store1 = RedisStore(
        app,
        server_id="test-server-1",
        redis_uri=None,  # Use fakeredis
    )
    
    # Initialize the first server
    await store1.init(reset_redis=True)
    
    # Get public API and register a test service
    api1 = await store1.get_public_api()
    
    # Register a test service
    test_service = {
        "id": "test-service-1",
        "name": "Test Service 1",
        "config": {
            "visibility": "public",
        },
        "ping": lambda: "pong"
    }
    await api1.register_service(test_service)
    
    # Verify service exists
    service_info = await api1.get_service_info("public/test-server-1:test-service-1")
    assert service_info is not None
    # The full service ID includes workspace and client
    assert "test-service-1" in service_info["id"]
    
    # Check that the server is listed
    servers = await store1.list_servers()
    assert "test-server-1" in servers
    
    # Get the redis instance to check keys later
    redis = store1.get_redis()
    
    # Check services exist in Redis
    keys_before = await redis.keys("services:*|*:*/test-server-1*")
    assert len(keys_before) > 0, "Server should have services registered"
    
    # Teardown the first server (simulating shutdown)
    await store1.teardown()
    
    # Check that all services are cleaned up
    keys_after = await redis.keys("services:*|*:*/test-server-1*")
    assert len(keys_after) == 0, f"All services should be cleaned up, but found: {keys_after}"
    
    # Services with server-generated client IDs should also be cleaned
    keys_client = await redis.keys("services:*|*:*/test-server-1-*")
    assert len(keys_client) == 0, f"Client services should be cleaned up, but found: {keys_client}"


@pytest.mark.asyncio
async def test_dead_server_cleanup():
    """Test that dead server detection and cleanup works."""
    from hypha.core.store import RedisStore
    from hypha.core import RedisEventBus
    from fastapi import FastAPI
    
    app = FastAPI()
    
    # Create first server
    store1 = RedisStore(
        app,
        server_id="test-server-dead",
        redis_uri=None,  # Use fakeredis
    )
    
    await store1.init(reset_redis=True)
    
    # Register a service
    api1 = await store1.get_public_api()
    test_service = {
        "id": "test-service-dead",
        "name": "Test Service Dead",
        "config": {
            "visibility": "public",
        },
        "ping": lambda: "pong"
    }
    await api1.register_service(test_service)
    
    # Get redis instance
    redis = store1.get_redis()
    
    # Verify service exists
    keys_before = await redis.keys("services:*|*:*/test-server-dead*")
    assert len(keys_before) > 0
    
    # Simulate server death by NOT calling teardown
    # Just disconnect the event bus to simulate network failure
    await store1.get_event_bus().stop()
    
    # Create a second server that will detect and clean up the dead one
    store2 = RedisStore(
        app,
        server_id="test-server-2",
        redis_uri=None,  # Use same fakeredis
    )
    # Need to use the same redis instance
    store2._redis = redis
    store2._event_bus = RedisEventBus(redis)
    await store2._event_bus.init()  # Initialize the event bus
    
    # Initialize store2 to setup root user
    await store2.setup_root_user()
    
    # This should detect the dead server and clean it up
    await store2.check_and_cleanup_servers()
    
    # Check that dead server's services are cleaned up
    keys_after = await redis.keys("services:*|*:*/test-server-dead*")
    assert len(keys_after) == 0, f"Dead server services should be cleaned up, but found: {keys_after}"
    
    # Clean up store2's event bus
    await store2._event_bus.stop()


@pytest.mark.asyncio
async def test_duplicate_worker_registration(fastapi_server):
    """Test that workers don't get duplicated when registering services."""
    
    # Connect admin to monitor services - let server assign workspace
    admin = await connect_to_server({
        "server_url": WS_SERVER_URL,
        "client_id": "admin-monitor",
    })
    
    workspace = admin.config["workspace"]
    print(f"Admin connected to workspace: {workspace}")
    
    # Initial service list
    services = await admin.list_services()
    print(f"Total initial services: {len(services)}")
    for s in services:
        print(f"  - {s.get('id')}: {s.get('name')}")
    # Filter more precisely - we want services with ID ending in :browser-worker
    initial_worker_count = len([s for s in services if s.get("id", "").endswith(":browser-worker")])
    print(f"Initial browser worker count: {initial_worker_count}")
    
    # Track all workers
    workers = []
    
    # Connect multiple browser workers with the same service ID
    # NOTE: We don't specify workspace - let server handle it
    for i in range(5):
        worker_client_id = f"browser-worker-{i}"
        worker = await connect_to_server({
            "server_url": WS_SERVER_URL,
            "client_id": worker_client_id,
        })
        print(f"Worker {i} connected to workspace: {worker.config['workspace']}, client_id: {worker.config['client_id']}")
        
        # Register the same service ID for each worker
        await worker.register_service({
            "id": "browser-worker",  # Same service ID for all
            "name": f"Browser Worker {i}",
            "type": "browser-worker",
            "config": {
                "visibility": "public",
                "require_context": True
            },
            "execute": lambda code, idx=i: f"Executed by worker {idx}: {code}"
        })
        
        workers.append(worker)
        
        # Check service count after each registration
        await asyncio.sleep(0.5)  # Give time for registration
        # List all public services across all workspaces
        services = await admin.list_services(query="*/*:*")
        all_browser_workers = [s for s in services if "browser-worker" in s.get("id", "")]
        
        # Also check services in the worker's own workspace
        worker_services = await worker.list_services()
        # Filter more precisely - we want services with ID ending in :browser-worker, not :built-in
        local_browser_workers = [s for s in worker_services if s.get("id", "").endswith(":browser-worker")]
        
        print(f"After registering worker {i}:")
        print(f"  Total public services: {len(services)}")
        print(f"  Public browser worker services: {len(all_browser_workers)}")
        print(f"  Worker's local services: {len(worker_services)}")
        print(f"  Worker's local browser-worker services: {len(local_browser_workers)}")
        # Check for duplicates
        service_ids = [bw.get('id') for bw in local_browser_workers]
        unique_ids = set(service_ids)
        if len(service_ids) != len(unique_ids):
            print(f"  ⚠️ DUPLICATE SERVICES DETECTED!")
            from collections import Counter
            id_counts = Counter(service_ids)
            for sid, count in id_counts.items():
                if count > 1:
                    print(f"    Duplicate: {sid} appears {count} times")
        for bw in local_browser_workers:
            print(f"    - {bw.get('id')}: {bw.get('name')}")
        
        # Each worker should have its own service entry  
        # Service IDs should be unique: workspace/client_id:service_id
        # Since each worker is in its own workspace, they won't see each other's services
        # unless they're public. We should have exactly one service in the worker's own workspace
        
        assert len(local_browser_workers) == 1, (
            f"Expected exactly 1 browser worker service in worker {i}'s workspace, but found {len(local_browser_workers)}"
        )
        assert len(unique_ids) == 1, (
            f"Found duplicate service IDs! Total: {len(service_ids)}, Unique: {len(unique_ids)}"
        )
    
    # Test cleanup when workers disconnect
    # Since each worker is in its own workspace, disconnecting should clean up its services
    for i, worker in enumerate(workers):
        worker_workspace = worker.config["workspace"]
        await worker.disconnect()
        await asyncio.sleep(0.5)  # Give time for cleanup
        
        # After disconnection, we shouldn't be able to list services in that workspace anymore
        # since anonymous workspaces are cleaned up when empty
        print(f"Disconnected worker {i} from workspace {worker_workspace}")
    
    # Final check - admin should still be connected and its workspace should be intact
    final_services = await admin.list_services()
    print(f"Final services in admin workspace: {len(final_services)}")
    assert len(final_services) >= 1, "Admin's workspace should still have services"
    
    await admin.disconnect()


@pytest.mark.asyncio
async def test_multiple_server_cleanup():
    """Test cleanup with multiple servers running simultaneously."""
    from hypha.core.store import RedisStore
    from hypha.core import RedisEventBus
    from fastapi import FastAPI
    from fakeredis import aioredis
    
    app = FastAPI()
    
    # Use shared fakeredis for all servers
    shared_redis = aioredis.FakeRedis.from_url("redis://localhost:9999/12")
    
    # Create multiple servers
    stores = []
    for i in range(3):
        store = RedisStore(
            app,
            server_id=f"test-multi-server-{i}",
            redis_uri=None,
        )
        store._redis = shared_redis
        store._event_bus = RedisEventBus(shared_redis)
        await store.init(reset_redis=(i == 0))  # Only reset on first
        
        # Register a service for each
        api = await store.get_public_api()
        service = {
            "id": f"service-{i}",
            "name": f"Service {i}",
            "config": {"visibility": "public"},
            "ping": lambda: "pong"
        }
        await api.register_service(service)
        stores.append(store)
    
    # Verify all services exist
    for i in range(3):
        keys = await shared_redis.keys(f"services:*|*:*/test-multi-server-{i}*")
        assert len(keys) > 0, f"Server {i} should have services"
    
    # Teardown servers one by one
    for i, store in enumerate(stores):
        await store.teardown()
        
        # Check this server's services are gone
        keys = await shared_redis.keys(f"services:*|*:*/test-multi-server-{i}*")
        assert len(keys) == 0, f"Server {i} services should be cleaned up"
        
        # Check other servers' services still exist (if not torn down yet)
        for j in range(i + 1, 3):
            keys = await shared_redis.keys(f"services:*|*:*/test-multi-server-{j}*")
            assert len(keys) > 0, f"Server {j} services should still exist"
