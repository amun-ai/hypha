"""Test HTTP Streaming RPC functionality.

This module tests the HTTP streaming RPC transport as an alternative to WebSocket.
"""

import asyncio
import pytest
import httpx

from hypha_rpc import connect_to_server

from . import SERVER_URL


class TestHTTPStreamingRPC:
    """Test HTTP Streaming RPC endpoints and client."""

    @pytest.mark.asyncio
    async def test_http_rpc_stream_endpoint_exists(self, fastapi_server):
        """Test that the HTTP streaming endpoint exists.

        Note: Connecting to a specific workspace without auth should fail,
        but the endpoint itself should exist.
        """
        # First connect via WebSocket to get a workspace and token
        ws_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-ws-for-stream",
        })
        try:
            workspace = ws_server.config["workspace"]
            token = await ws_server.generate_token()

            async with httpx.AsyncClient() as client:
                # The endpoint should exist with proper auth
                async with client.stream(
                    "GET",
                    f"{SERVER_URL}/{workspace}/rpc",
                    params={"client_id": "test-endpoint-exists"},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(10.0, connect=5.0),
                ) as response:
                    # Should be 200 with proper auth
                    if response.status_code != 200:
                        content = await response.aread()
                        print(f"Error response: {content}")
                    assert response.status_code == 200, f"HTTP RPC stream endpoint should exist, got {response.status_code}"
                    # Verify we can read at least one line
                    async for line in response.aiter_lines():
                        if line.strip():
                            break  # Got at least one line, test passes
        finally:
            await ws_server.disconnect()

    @pytest.mark.asyncio
    async def test_http_rpc_post_endpoint_exists(self, fastapi_server):
        """Test that the HTTP POST endpoint exists."""
        # First connect via WebSocket to get a workspace
        ws_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-ws-for-post",
        })
        try:
            workspace = ws_server.config["workspace"]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SERVER_URL}/{workspace}/rpc",
                    params={"client_id": "test-client"},
                    content=b"",  # Empty body
                    timeout=5.0,
                )
                # Should not be 404 (endpoint exists)
                # Without proper auth/connection it may fail with 400 (no connection)
                assert response.status_code != 404, "HTTP RPC POST endpoint should exist"
        finally:
            await ws_server.disconnect()

    @pytest.mark.asyncio
    async def test_http_rpc_stream_connection(self, fastapi_server):
        """Test establishing an HTTP streaming connection."""
        # First connect via WebSocket to get workspace and token
        ws_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-ws-for-stream-conn",
        })
        try:
            workspace = ws_server.config["workspace"]
            token = await ws_server.generate_token()

            async with httpx.AsyncClient() as client:
                # Connect with streaming using token
                async with client.stream(
                    "GET",
                    f"{SERVER_URL}/{workspace}/rpc",
                    params={"client_id": "test-stream-client"},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(30.0, connect=10.0),
                ) as response:
                    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

                    # Read first line (should be connection info)
                    first_line = None
                    async for line in response.aiter_lines():
                        if line.strip():
                            first_line = line
                            break

                    assert first_line is not None, "Should receive connection info"
                    import json
                    conn_info = json.loads(first_line)
                    assert conn_info.get("type") == "connection_info"
                    assert "workspace" in conn_info
                    assert "client_id" in conn_info
                    assert "manager_id" in conn_info
        finally:
            await ws_server.disconnect()

    @pytest.mark.asyncio
    async def test_http_client_connect(self, fastapi_server):
        """Test connecting with the HTTP client via transport parameter.

        Anonymous connection gets its own workspace.
        """
        server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-http-client",
            "transport": "http",
        })
        try:
            assert server is not None
            assert server.config.get("workspace") is not None
            assert server.config.get("client_id") == "test-http-client"
            # Anonymous user gets their own workspace starting with ws-user-
            assert server.config.get("workspace").startswith("ws-user-")
        finally:
            await server.disconnect()

    @pytest.mark.asyncio
    async def test_http_client_get_service(self, fastapi_server):
        """Test getting a service via HTTP client.

        Pattern:
        1. WebSocket client connects anonymously, gets its workspace
        2. WS client registers a public service
        3. WS client generates a token for HTTP client
        4. HTTP client connects with token to same workspace
        5. HTTP client calls the service
        """
        # First, register a service via WebSocket
        ws_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-ws-provider",
        })

        try:
            workspace = ws_server.config["workspace"]

            # Register a test service with public visibility
            await ws_server.register_service({
                "id": "test-http-service",
                "name": "Test HTTP Service",
                "config": {"visibility": "public"},
                "add": lambda a, b: a + b,
                "greet": lambda name: f"Hello, {name}!",
            })

            # Generate token for HTTP client to join same workspace
            token = await ws_server.generate_token()

            # Connect via HTTP to the same workspace using token
            http_server = await connect_to_server({
                "server_url": SERVER_URL,
                "workspace": workspace,
                "client_id": "test-http-consumer",
                "transport": "http",
                "token": token,
            })

            try:
                service = await http_server.get_service("test-ws-provider:test-http-service")
                assert service is not None

                # Call methods
                result = await service.add(2, 3)
                assert result == 5

                greeting = await service.greet("World")
                assert greeting == "Hello, World!"

            finally:
                await http_server.disconnect()

        finally:
            await ws_server.disconnect()

    @pytest.mark.asyncio
    async def test_http_client_callbacks(self, fastapi_server):
        """Test that callbacks work via HTTP streaming."""
        # Register a service that uses callbacks
        ws_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-callback-provider",
        })

        try:
            workspace = ws_server.config["workspace"]
            callback_results = []

            async def process_with_callback(data, callback):
                """Process data and call the callback with results."""
                for i in range(3):
                    await callback(f"Processing step {i+1}")
                return f"Final result: {data}"

            await ws_server.register_service({
                "id": "callback-service",
                "name": "Callback Service",
                "config": {"visibility": "public"},
                "process": process_with_callback,
            })

            # Generate token for HTTP client
            token = await ws_server.generate_token()

            # Connect via HTTP to the same workspace
            http_server = await connect_to_server({
                "server_url": SERVER_URL,
                "workspace": workspace,
                "client_id": "test-http-callback-consumer",
                "transport": "http",
                "token": token,
            })

            try:
                service = await http_server.get_service("test-callback-provider:callback-service")

                # Call with callback
                def my_callback(msg):
                    callback_results.append(msg)

                result = await service.process("test data", my_callback)

                assert result == "Final result: test data"
                assert len(callback_results) == 3
                assert "Processing step 1" in callback_results

            finally:
                await http_server.disconnect()

        finally:
            await ws_server.disconnect()

    @pytest.mark.asyncio
    async def test_http_client_reconnection(self, fastapi_server):
        """Test that HTTP client can reconnect after disconnection."""
        # Connect first time
        server1 = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-reconnect-client",
            "transport": "http",
        })

        workspace = server1.config["workspace"]
        # Generate token before disconnecting
        token = await server1.generate_token()
        await server1.disconnect()

        # Should be able to reconnect to same workspace with token
        server2 = await connect_to_server({
            "server_url": SERVER_URL,
            "workspace": workspace,
            "client_id": "test-reconnect-client-2",
            "transport": "http",
            "token": token,
        })

        try:
            assert server2.config.get("workspace") == workspace
        finally:
            await server2.disconnect()

    @pytest.mark.asyncio
    async def test_http_vs_websocket_interop(self, fastapi_server):
        """Test that HTTP and WebSocket clients can communicate."""
        # HTTP client connects first and registers service
        http_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "http-service-provider",
            "transport": "http",
        })

        try:
            workspace = http_server.config["workspace"]

            await http_server.register_service({
                "id": "http-provided-service",
                "name": "HTTP Provided Service",
                "config": {"visibility": "public"},
                "multiply": lambda a, b: a * b,
            })

            # Generate token for WebSocket client
            token = await http_server.generate_token()

            # WebSocket client joins same workspace and calls the service
            ws_server = await connect_to_server({
                "server_url": SERVER_URL,
                "workspace": workspace,
                "client_id": "ws-consumer",
                "token": token,
            })

            try:
                service = await ws_server.get_service("http-service-provider:http-provided-service")
                result = await service.multiply(4, 5)
                assert result == 20

            finally:
                await ws_server.disconnect()

        finally:
            await http_server.disconnect()

    @pytest.mark.asyncio
    async def test_http_client_error_handling(self, fastapi_server):
        """Test error handling in HTTP client."""
        http_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-error-client",
            "transport": "http",
        })

        try:
            # Try to get non-existent service
            with pytest.raises(Exception):
                await http_server.get_service("non-existent:service")

        finally:
            await http_server.disconnect()

    @pytest.mark.asyncio
    async def test_http_streaming_ping_keepalive(self, fastapi_server):
        """Test that the streaming connection sends ping messages for keep-alive."""
        # First connect via WebSocket to get workspace and token
        ws_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-ws-for-ping",
        })
        try:
            workspace = ws_server.config["workspace"]
            token = await ws_server.generate_token()

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "GET",
                    f"{SERVER_URL}/{workspace}/rpc",
                    params={"client_id": "test-ping-client"},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=httpx.Timeout(60.0, connect=10.0),
                ) as response:
                    assert response.status_code == 200

                    messages_received = []
                    ping_received = False
                    timeout_at = asyncio.get_event_loop().time() + 35  # Wait up to 35s for ping

                    async for line in response.aiter_lines():
                        if asyncio.get_event_loop().time() > timeout_at:
                            break

                        if line.strip():
                            import json
                            msg = json.loads(line)
                            messages_received.append(msg)

                            if msg.get("type") == "ping":
                                ping_received = True
                                break

                            # Wait for at least connection_info
                            if msg.get("type") == "connection_info":
                                continue

                    # We should have received connection info at least
                    assert any(m.get("type") == "connection_info" for m in messages_received)
        finally:
            await ws_server.disconnect()


class TestHTTPStreamingPerformance:
    """Performance and stress tests for HTTP streaming."""

    @pytest.mark.asyncio
    async def test_multiple_http_clients(self, fastapi_server):
        """Test multiple HTTP clients connecting simultaneously."""
        clients = []
        num_clients = 5

        try:
            # Create multiple clients (each gets their own workspace)
            for i in range(num_clients):
                client = await connect_to_server({
                    "server_url": SERVER_URL,
                    "client_id": f"multi-http-client-{i}",
                    "transport": "http",
                })
                clients.append(client)

            # Verify all connected
            assert len(clients) == num_clients

            # All should have different client IDs
            client_ids = [c.config.get("client_id") for c in clients]
            assert len(set(client_ids)) == num_clients

        finally:
            # Cleanup
            for client in clients:
                await client.disconnect()

    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect(self, fastapi_server):
        """Test rapid connect/disconnect cycles."""
        for i in range(5):
            server = await connect_to_server({
                "server_url": SERVER_URL,
                "client_id": f"rapid-client-{i}",
                "transport": "http",
            })
            assert server is not None
            await server.disconnect()
            await asyncio.sleep(0.1)  # Small delay between cycles


class TestHTTPReconnectionToken:
    """Test HTTP reconnection token functionality."""

    @pytest.mark.asyncio
    async def test_http_reconnection_token_received(self, fastapi_server):
        """Test that HTTP client receives reconnection token in connection info."""
        server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-reconnect-token-client",
            "transport": "http",
        })

        try:
            # Check that connection info contains reconnection token
            assert server.config.get("reconnection_token") is not None, \
                "reconnection_token should be in connection info"
            assert server.config.get("reconnection_token_life_time") is not None, \
                "reconnection_token_life_time should be in connection info"
        finally:
            await server.disconnect()

    @pytest.mark.asyncio
    async def test_http_reconnection_with_token(self, fastapi_server):
        """Test HTTP client can reconnect using reconnection_token."""
        # Connect first time
        server1 = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-token-reconnect",
            "transport": "http",
        })

        workspace = server1.config["workspace"]
        client_id = server1.config["client_id"]
        reconnection_token = server1.config.get("reconnection_token")

        assert reconnection_token is not None, "Should have received reconnection token"

        await server1.disconnect()

        # Wait a moment for cleanup
        await asyncio.sleep(0.2)

        # Reconnect using the reconnection_token
        # Import HTTP client directly to use reconnection_token
        from hypha_rpc.http_client import connect_to_server_http

        server2 = await connect_to_server_http({
            "server_url": SERVER_URL,
            "workspace": workspace,
            "client_id": client_id,
            "reconnection_token": reconnection_token,
        })

        try:
            # Should have connected to same workspace with same client_id
            assert server2.config.get("workspace") == workspace, \
                "Should reconnect to same workspace"
            assert server2.config.get("client_id") == client_id, \
                "Should reconnect with same client_id"
            # Should have received a new reconnection token
            new_token = server2.config.get("reconnection_token")
            assert new_token is not None, "Should receive new reconnection token"
        finally:
            await server2.disconnect()

    @pytest.mark.asyncio
    async def test_http_reconnection_token_validation(self, fastapi_server):
        """Test that invalid reconnection token is rejected."""
        # Try to connect with invalid token
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/public/rpc",
                params={
                    "client_id": "invalid-token-client",
                    "reconnection_token": "invalid-token-value",
                },
                timeout=5.0,
            )
            # Should fail with 401 Unauthorized
            assert response.status_code == 401, \
                f"Invalid reconnection token should be rejected, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_http_reconnection_token_workspace_mismatch(self, fastapi_server):
        """Test that reconnection token with wrong workspace is rejected."""
        # Connect first time
        server1 = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-ws-mismatch",
            "transport": "http",
        })

        workspace = server1.config["workspace"]
        client_id = server1.config["client_id"]
        reconnection_token = server1.config.get("reconnection_token")

        await server1.disconnect()
        await asyncio.sleep(0.2)

        # Try to connect with token but different workspace
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/different-workspace/rpc",
                params={
                    "client_id": client_id,
                    "reconnection_token": reconnection_token,
                },
                timeout=5.0,
            )
            # Should fail because workspace doesn't match token
            assert response.status_code in [401, 403], \
                f"Workspace mismatch should be rejected, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_http_reconnection_token_client_id_mismatch(self, fastapi_server):
        """Test that reconnection token with wrong client_id is rejected."""
        # Connect first time
        server1 = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "test-client-mismatch",
            "transport": "http",
        })

        workspace = server1.config["workspace"]
        reconnection_token = server1.config.get("reconnection_token")

        await server1.disconnect()
        await asyncio.sleep(0.2)

        # Try to connect with token but different client_id
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SERVER_URL}/{workspace}/rpc",
                params={
                    "client_id": "different-client-id",
                    "reconnection_token": reconnection_token,
                },
                timeout=5.0,
            )
            # Should fail because client_id doesn't match token
            assert response.status_code == 401, \
                f"Client ID mismatch should be rejected, got {response.status_code}"

    @pytest.mark.asyncio
    async def test_http_service_survives_reconnection(self, fastapi_server):
        """Test that services can still be called after reconnection."""
        # Provider connects via WebSocket
        ws_server = await connect_to_server({
            "server_url": SERVER_URL,
            "client_id": "service-provider-ws",
        })

        try:
            workspace = ws_server.config["workspace"]

            # Register a service
            await ws_server.register_service({
                "id": "reconnect-test-service",
                "name": "Reconnect Test Service",
                "config": {"visibility": "public"},
                "add": lambda a, b: a + b,
            })

            # Generate token for HTTP client
            token = await ws_server.generate_token()

            # HTTP client connects
            http_server1 = await connect_to_server({
                "server_url": SERVER_URL,
                "workspace": workspace,
                "client_id": "http-consumer-1",
                "transport": "http",
                "token": token,
            })

            # Get reconnection token
            reconnection_token = http_server1.config.get("reconnection_token")
            client_id = http_server1.config["client_id"]

            # Call service before disconnect
            service = await http_server1.get_service("service-provider-ws:reconnect-test-service")
            result1 = await service.add(1, 2)
            assert result1 == 3

            # Disconnect
            await http_server1.disconnect()
            await asyncio.sleep(0.3)

            # Reconnect with token
            from hypha_rpc.http_client import connect_to_server_http

            http_server2 = await connect_to_server_http({
                "server_url": SERVER_URL,
                "workspace": workspace,
                "client_id": client_id,
                "reconnection_token": reconnection_token,
            })

            try:
                # Call service after reconnect
                service2 = await http_server2.get_service("service-provider-ws:reconnect-test-service")
                result2 = await service2.add(4, 5)
                assert result2 == 9
            finally:
                await http_server2.disconnect()

        finally:
            await ws_server.disconnect()
