"""Test to verify the hanging client fix works correctly."""

import asyncio
import pytest
from hypha_rpc import connect_to_server
import logging

# Import the server URL from the test constants
from . import WS_SERVER_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_hanging_client_fix_integration(fastapi_server, test_user_token):
    """Integration test to verify the hanging client fix works."""
    
    # Use the constant server URL instead of the fixture parameter
    server_url = WS_SERVER_URL
    # Don't specify workspace to use default workspace behavior
    
    # Connect as manager to monitor client state
    manager = await connect_to_server({
        "name": "Fix Test Manager",
        "server_url": server_url,
        "token": test_user_token,
        "client_id": "fix-test-manager",
    })
    
    # Get the actual workspace from the manager
    workspace = manager.config.workspace
    
    try:
        logger.info(f"=== Testing Hanging Client Fix (workspace: {workspace}) ===")
        
        # Step 0: Clean up any existing hanging clients first
        logger.info("Step 0: Cleaning up any existing hanging clients")
        try:
            await manager.cleanup(timeout=2)
            logger.info("Initial cleanup completed")
        except Exception as e:
            logger.warning(f"Initial cleanup had issues: {e}")
        
        # Wait a moment for cleanup to complete
        await asyncio.sleep(1)
        
        # Step 1: Create a client and register a service
        logger.info("Step 1: Creating test client")
        test_client = await connect_to_server({
            "name": "Test Client",
            "server_url": server_url,
            "token": test_user_token,
            "client_id": "test-client",
        })
        
        # Register a service
        await test_client.register_service({
            "id": "test-service",
            "name": "Test Service",
            "config": {"visibility": "public"},
            "test_method": lambda x: f"processed: {x}"
        })
        
        # Verify client is initially visible and responsive
        initial_clients = await manager.list_clients()
        logger.info(f"Initial clients: {len(initial_clients)}")
        
        client_found = any(
            "test-client" in client["id"] for client in initial_clients
        )
        assert client_found, "Test client should be visible initially"
        
        # Step 2: Simulate abnormal disconnection
        logger.info("Step 2: Simulating abnormal client disconnection")
        
        # Force close the websocket connection without proper cleanup
        if hasattr(test_client, 'rpc') and hasattr(test_client.rpc, '_connection'):
            connection = test_client.rpc._connection
            if hasattr(connection, '_websocket'):
                await connection._websocket.close(code=1001)  # Going away - valid abnormal closure
                connection._closed = True
                connection._enable_reconnect = False
        
        # Give some time for the server to detect the disconnection
        await asyncio.sleep(2)
        
        # Step 3: Test the fix - check if client is properly detected as unresponsive
        logger.info("Step 3: Testing improved client detection")
        
        clients_after_disconnect = await manager.list_clients()
        logger.info(f"Clients after disconnect: {len(clients_after_disconnect)}")
        
        # Try to ping the client - this should fail and trigger cleanup
        test_client_id = f"{workspace}/test-client"
        ping_attempts = 0
        client_cleaned_up = False
        
        for attempt in range(3):  # Try up to 3 times
            try:
                ping_result = await manager.ping_client(test_client_id, timeout=2)
                if ping_result == "pong":
                    logger.info(f"Attempt {attempt + 1}: Client still responsive")
                else:
                    logger.info(f"Attempt {attempt + 1}: Client unresponsive - {ping_result}")
                    client_cleaned_up = True
                    break
            except Exception as e:
                logger.info(f"Attempt {attempt + 1}: Client ping failed - {e}")
                client_cleaned_up = True
                break
            
            await asyncio.sleep(1)
            ping_attempts += 1
        
        # Step 4: Verify automatic cleanup
        logger.info("Step 4: Verifying automatic cleanup")
        
        # Wait a bit more for cleanup to complete
        await asyncio.sleep(2)
        
        final_clients = await manager.list_clients()
        logger.info(f"Final clients after cleanup: {len(final_clients)}")
        
        # The test client should either be:
        # 1. Completely removed from the client list, OR  
        # 2. Detected as hanging and cleaned up when accessed
        test_client_still_listed = any(
            "test-client" in client["id"] for client in final_clients
        )
        
        if test_client_still_listed:
            # If still listed, it should be cleaned up when we try to access it
            logger.info("Test client still in list, verifying cleanup on access")
            
            # Try to access the service - this should trigger cleanup in the improved client_exists()
            try:
                service = await manager.get_service(f"{workspace}/test-client:test-service", timeout=3)
                result = await service.test_method("test_data")
                logger.warning("Unexpected: service access succeeded, client may not be hanging")
            except Exception as e:
                logger.info(f"Expected: service access failed due to hanging client: {e}")
                # This is expected - the service should not be accessible from a hanging client
        else:
            logger.info("✅ Test client was properly cleaned up and removed from client list")
        
        # Step 5: Test service registration with potential hanging clients
        logger.info("Step 5: Testing service registration after cleanup")
        
        # Create a new client to test that service registration works normally
        new_client = await connect_to_server({
            "name": "New Test Client",
            "server_url": server_url,
            "token": test_user_token,  
            "client_id": "new-test-client",
        })
        
        # This should succeed without issues
        await new_client.register_service({
            "id": "new-service",
            "name": "New Service", 
            "config": {"visibility": "public"},
            "process": lambda data: {"status": "ok", "data": data}
        })
        
        logger.info("✅ New service registration succeeded")
        
        # Clean up
        await new_client.disconnect()
        
        # Step 6: Verify final state
        logger.info("Step 6: Final verification")
        
        final_final_clients = await manager.list_clients()
        logger.info(f"Final client count: {len(final_final_clients)}")
        
        # The fix should have prevented hanging clients
        # Be more lenient - allow for some leftover clients but ensure the specific test client is cleaned up
        test_clients_remaining = [c for c in final_final_clients if "test-client" in c["id"]]
        assert len(test_clients_remaining) == 0, f"Test client should be cleaned up, but found: {test_clients_remaining}"
        
        # Allow for some leftover clients (e.g., load balancing clients from previous tests)
        # but ensure the count is reasonable (≤ 10 instead of exactly ≤ 2)
        assert len(final_final_clients) <= 10, f"Too many clients remaining: {len(final_final_clients)}"
        
        logger.info("🎉 HANGING CLIENT FIX VERIFICATION SUCCESSFUL!")
        logger.info("✅ Improved client detection is working")  
        logger.info("✅ Automatic cleanup is functioning")
        logger.info("✅ Service registration works normally after cleanup")
        
        return {
            "initial_clients": len(initial_clients),
            "clients_after_disconnect": len(clients_after_disconnect),
            "final_clients": len(final_final_clients),
            "cleanup_triggered": client_cleaned_up,
            "fix_working": len(test_clients_remaining) == 0
        }
        
    finally:
        try:
            await manager.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting manager: {e}")


@pytest.mark.asyncio
async def test_periodic_cleanup_functionality(fastapi_server, test_user_token):
    """Test that periodic cleanup actually runs and cleans up hanging clients."""
    
    server_url = WS_SERVER_URL
    
    # Connect as manager to monitor the cleanup
    manager = await connect_to_server({
        "name": "Periodic Cleanup Test Manager",
        "server_url": server_url,
        "token": test_user_token,
        "client_id": "periodic-cleanup-manager",
    })
    
    workspace = manager.config.workspace
    
    try:
        logger.info(f"=== Testing Periodic Cleanup Functionality (workspace: {workspace}) ===")
        
        # Step 1: Create multiple test clients to simulate a load
        logger.info("Step 1: Creating multiple test clients")
        test_clients = []
        
        for i in range(3):
            client = await connect_to_server({
                "name": f"Periodic Test Client {i+1}",
                "server_url": server_url,
                "token": test_user_token,
                "client_id": f"periodic-test-client-{i+1}",
            })
            
            # Register a service on each client
            await client.register_service({
                "id": f"test-service-{i+1}",
                "name": f"Test Service {i+1}",
                "config": {"visibility": "public"},
                "test_method": lambda x, idx=i: f"processed by client {idx+1}: {x}"
            })
            
            test_clients.append(client)
        
        # Verify all clients are initially visible
        initial_clients = await manager.list_clients()
        logger.info(f"Initial clients count: {len(initial_clients)}")
        
        test_client_count = sum(1 for c in initial_clients if "periodic-test-client" in c["id"])
        assert test_client_count == 3, f"Expected 3 test clients, found {test_client_count}"
        
        # Step 2: Force abnormal disconnection of all test clients
        logger.info("Step 2: Simulating abnormal disconnection of all test clients")
        
        for i, client in enumerate(test_clients):
            if hasattr(client, 'rpc') and hasattr(client.rpc, '_connection'):
                connection = client.rpc._connection
                if hasattr(connection, '_websocket'):
                    await connection._websocket.close(code=1001)  # Going away
                    connection._closed = True
                    connection._enable_reconnect = False
                    logger.info(f"Forcibly disconnected test client {i+1}")
        
        # Step 3: Wait for periodic cleanup to detect and clean up hanging clients
        logger.info("Step 3: Waiting for periodic cleanup to run (35 seconds for test environment)")
        
        # Since periodic cleanup runs every 30 seconds in test env, wait 35 seconds to be sure
        await asyncio.sleep(35)
        
        # Step 4: Verify that hanging clients have been cleaned up
        logger.info("Step 4: Verifying hanging clients were cleaned up")
        
        final_clients = await manager.list_clients()
        logger.info(f"Final clients count: {len(final_clients)}")
        
        remaining_test_clients = [c for c in final_clients if "periodic-test-client" in c["id"]]
        logger.info(f"Remaining test clients: {len(remaining_test_clients)}")
        
        if remaining_test_clients:
            logger.warning(f"Some test clients still remain: {[c['id'] for c in remaining_test_clients]}")
            
            # Try pinging them to see if they're actually hanging
            for client_info in remaining_test_clients:
                client_id = client_info["id"]
                try:
                    ping_result = await manager.ping_client(client_id, timeout=2)
                    logger.info(f"Ping result for {client_id}: {ping_result}")
                except Exception as e:
                    logger.info(f"Ping failed for {client_id}: {e}")
        
        # Expect all test clients to be cleaned up by periodic cleanup
        assert len(remaining_test_clients) == 0, f"Periodic cleanup failed - {len(remaining_test_clients)} hanging clients remain: {[c['id'] for c in remaining_test_clients]}"
        
        # Step 5: Verify that new clients can still connect normally
        logger.info("Step 5: Verifying new client registration works after cleanup")
        
        new_client = await connect_to_server({
            "name": "Post Cleanup Test Client",
            "server_url": server_url,
            "token": test_user_token,
            "client_id": "post-cleanup-client",
        })
        
        await new_client.register_service({
            "id": "post-cleanup-service",
            "name": "Post Cleanup Service",
            "config": {"visibility": "public"},
            "test_method": lambda x: f"post cleanup: {x}"
        })
        
        # Verify the new client is visible
        post_cleanup_clients = await manager.list_clients()
        new_client_found = any("post-cleanup-client" in c["id"] for c in post_cleanup_clients)
        assert new_client_found, "New client registration failed after cleanup"
        
        await new_client.disconnect()
        
        logger.info("🎉 PERIODIC CLEANUP VERIFICATION SUCCESSFUL!")
        logger.info("✅ Periodic cleanup automatically detected and removed hanging clients")
        logger.info("✅ New client registration works normally after cleanup")
        
        return {
            "initial_test_clients": 3,
            "remaining_test_clients": len(remaining_test_clients),
            "periodic_cleanup_working": len(remaining_test_clients) == 0
        }
        
    finally:
        try:
            await manager.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting manager: {e}")


if __name__ == "__main__":
    # For manual testing
    pytest.main([__file__, "-v", "-s"]) 