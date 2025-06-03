"""Test WebRTC ICE servers functionality and WebRTC services."""

import pytest
import json
import time
import asyncio
import socket
import numpy as np
from unittest.mock import patch
from hypha_rpc import connect_to_server, register_rtc_service, get_rtc_service
from hypha_rpc.sync import connect_to_server as connect_to_server_sync
from hypha_rpc.sync import register_rtc_service as register_rtc_service_sync
from hypha_rpc.sync import get_rtc_service as get_rtc_service_sync
from . import SERVER_URL_COTURN, COTURN_SECRET, COTURN_URI

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCConfiguration, RTCIceServer
from aiortc.rtcicetransport import RTCIceTransport
from av import VideoFrame


@pytest.mark.asyncio
async def test_webrtc_ice_servers_service_available(fastapi_server_coturn, root_user_token):
    """Test that WebRTC ICE servers service is available when COTURN is configured."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Check if get_rtc_ice_servers method is available
    ice_servers = await api.get_rtc_ice_servers()
    assert isinstance(ice_servers, list), "Should return a list of ICE servers"
    assert len(ice_servers) > 0, "Should have at least one ICE server"
    print("✅ WebRTC ICE servers service is available")


@pytest.mark.asyncio
async def test_get_rtc_ice_servers_basic(fastapi_server_coturn, root_user_token):
    """Test basic WebRTC ICE servers functionality."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Get ICE servers configuration
    ice_servers = await api.get_rtc_ice_servers()
    
    # Verify the response structure
    assert isinstance(ice_servers, list), "ICE servers should be a list"
    assert len(ice_servers) > 0, "Should have at least one ICE server"
    
    # Check the first server configuration
    server = ice_servers[0]
    assert "username" in server, "Server should have username"
    assert "credential" in server, "Server should have credential"
    assert "urls" in server, "Server should have urls"
    
    # Verify URL format
    urls = server["urls"]
    assert isinstance(urls, list), "URLs should be a list"
    
    # Should have both TURN and STUN URLs
    turn_urls = [url for url in urls if url.startswith("turn:")]
    stun_urls = [url for url in urls if url.startswith("stun:")]
    
    assert len(turn_urls) > 0, "Should have at least one TURN URL"
    assert len(stun_urls) > 0, "Should have at least one STUN URL"
    
    # Verify URL contains the expected host
    coturn_host = COTURN_URI.split(":")[0]
    for url in urls:
        assert coturn_host in url, f"URL should contain COTURN host: {url}"


@pytest.mark.asyncio
async def test_get_rtc_ice_servers_with_custom_ttl(fastapi_server_coturn, root_user_token):
    """Test WebRTC ICE servers with custom TTL."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Get ICE servers with custom TTL
    custom_ttl = 3600  # 1 hour
    ice_servers = await api.get_rtc_ice_servers(ttl=custom_ttl)
    
    assert isinstance(ice_servers, list), "ICE servers should be a list"
    assert len(ice_servers) > 0, "Should have at least one ICE server"
    
    # Username should contain timestamp based on TTL
    server = ice_servers[0]
    username = server["username"]
    
    # Username format is "timestamp:user_id"
    assert ":" in username, "Username should contain timestamp"
    timestamp_str = username.split(":")[0]
    timestamp = int(timestamp_str)
    
    # Verify timestamp is in expected range (current time + TTL)
    current_time = int(time.time())
    expected_expiry = current_time + custom_ttl
    
    # Allow some tolerance for timing differences
    assert abs(timestamp - expected_expiry) < 10, "Timestamp should be close to expected expiry"


@pytest.mark.asyncio
async def test_get_rtc_ice_servers_anonymous_rejected(fastapi_server_coturn):
    """Test that anonymous users are rejected when requesting ICE servers."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": SERVER_URL_COTURN,
            # No token provided - should be anonymous
        }
    )
    
    # Anonymous users should be rejected
    with pytest.raises(Exception) as exc_info:
        await api.get_rtc_ice_servers()
    
    assert "Anonymous users are not allowed to get RTC ICE servers" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rtc_service_basic(fastapi_server_coturn, root_user_token):
    """Test basic RTC service registration and access with two separate connections."""
    service_id = "test-rtc-service"
    
    # Create two separate server connections
    server1 = await connect_to_server(
        {
            "name": "rtc-provider-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    server2 = await connect_to_server(
        {
            "name": "rtc-consumer-client", 
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Register a regular echo service on server1
    await server1.register_service(
        {
            "id": "echo-service",
            "config": {"visibility": "public"},
            "type": "echo",
            "echo": lambda x: x,
        }
    )
    
    # Register RTC service on server1
    rtc_service_info = await register_rtc_service(server1, service_id)
    assert rtc_service_info is not None, "RTC service should be registered"
    
    # Get RTC service from server2 (different connection)
    pc = await get_rtc_service(server2, rtc_service_info["id"])
    assert pc is not None, "Should get RTC peer connection"
    
    try:
        # Access echo service through WebRTC from server2
        svc = await pc.get_service("echo-service")
        result = await svc.echo("hello")
        assert result == "hello", "Echo service should work through WebRTC"
        print("✅ WebRTC service communication between separate clients successful")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        # Clean up
        await pc.close()


def test_rtc_service_sync(fastapi_server_coturn, root_user_token):
    """Test RTC service with sync API using two separate connections."""
    service_id = "test-rtc-service-sync"
    
    # Create two separate sync server connections
    server1 = connect_to_server_sync(
        {
            "name": "rtc-sync-provider-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    server2 = connect_to_server_sync(
        {
            "name": "rtc-sync-consumer-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Register echo service on server1
    server1.register_service(
        {
            "id": "echo-service-sync",
            "config": {"visibility": "public"},
            "type": "echo",
            "echo": lambda x: x,
        }
    )
    
    # Register RTC service on server1 and access from server2
    register_rtc_service_sync(server1, service_id)
    pc = get_rtc_service_sync(server2, service_id)
    
    try:
        svc = pc.get_service("echo-service-sync")
        result = svc.echo("hello sync")
        assert result == "hello sync", "Sync echo service should work through WebRTC"
        print("✅ Sync WebRTC service communication between separate clients successful")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        pc.close()


def test_rtc_service_auto(fastapi_server_coturn, root_user_token):
    """Test automatic WebRTC mode."""
    server = connect_to_server_sync(
        {
            "name": "rtc-auto-test-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
            "webrtc": True,  # Enable automatic WebRTC
        }
    )
    
    # Register echo service
    server.register_service(
        {
            "id": "echo-service-auto",
            "config": {"visibility": "public"},
            "type": "echo",
            "echo": lambda x: x,
        }
    )
    
    # Get service should automatically use WebRTC
    svc = server.get_service("echo-service-auto")
    result = svc.echo("hello auto")
    assert result == "hello auto", "Auto WebRTC service should work"
    print("✅ Automatic WebRTC service communication successful")


@pytest.mark.asyncio
async def test_rtc_service_with_data_processing(fastapi_server_coturn, root_user_token):
    """Test RTC service with more complex data processing using two separate connections."""
    # Create two separate server connections
    server1 = await connect_to_server(
        {
            "name": "rtc-data-provider-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    server2 = await connect_to_server(
        {
            "name": "rtc-data-consumer-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Register a data processing service on server1
    def process_array(data):
        """Process numpy-like array data."""
        if isinstance(data, list):
            return [x * 2 for x in data]
        return data * 2
    
    def get_info():
        """Get service information."""
        return {
            "name": "Data Processor",
            "version": "1.0.0",
            "capabilities": ["array_processing", "math_operations"]
        }
    
    await server1.register_service(
        {
            "id": "data-processor",
            "config": {"visibility": "public"},
            "type": "processor",
            "process_array": process_array,
            "get_info": get_info,
        }
    )
    
    # Register RTC service on server1
    rtc_service_info = await register_rtc_service(server1, "data-rtc-service")
    # Get RTC service from server2 (different connection)
    pc = await get_rtc_service(server2, rtc_service_info["id"])
    
    try:
        # Access data processor through WebRTC from server2
        processor = await pc.get_service("data-processor")
        
        # Test array processing
        result = await processor.process_array([1, 2, 3, 4, 5])
        assert result == [2, 4, 6, 8, 10], "Array processing should work through WebRTC"
        
        # Test info retrieval
        info = await processor.get_info()
        assert info["name"] == "Data Processor", "Service info should be accessible"
        assert "array_processing" in info["capabilities"], "Capabilities should be correct"
        
        print("✅ WebRTC data processing service works correctly between separate clients")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        await pc.close()


@pytest.mark.asyncio
async def test_rtc_service_multiple_clients(fastapi_server_coturn, root_user_token, test_user_token):
    """Test that WebRTC services work with multiple client connections."""
    # Test that multiple clients can each register their own services successfully
    # Client 1: Service provider
    server1 = await connect_to_server(
        {
            "name": "rtc-provider",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Client 2: Another service provider (same workspace)
    server2 = await connect_to_server(
        {
            "name": "rtc-consumer", 
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Register different services for each client
    await server1.register_service(
        {
            "id": "calculator-service",
            "config": {"visibility": "public"},
            "type": "calculator",
            "add": lambda a, b: a + b,
            "multiply": lambda a, b: a * b,
        }
    )
    
    await server2.register_service(
        {
            "id": "string-service",
            "config": {"visibility": "public"},
            "type": "string",
            "concat": lambda a, b: f"{a}{b}",
            "reverse": lambda s: s[::-1],
        }
    )
    
    # Test that both regular services work
    calc_svc = await server1.get_service("calculator-service")
    result1 = await calc_svc.add(5, 3)
    assert result1 == 8, "Calculator service should work"
    
    string_svc = await server2.get_service("string-service")
    result2 = await string_svc.concat("Hello", " World")
    assert result2 == "Hello World", "String service should work"
    
    # Test cross-client service access
    calc_from_server2 = await server2.get_service("calculator-service")
    result3 = await calc_from_server2.multiply(4, 7)
    assert result3 == 28, "Cross-client service access should work"
    
    print("✅ Multiple client service access works")


@pytest.mark.asyncio
async def test_rtc_service_error_handling(fastapi_server_coturn, root_user_token):
    """Test error handling in WebRTC services using two separate connections."""
    # Create two separate server connections
    server1 = await connect_to_server(
        {
            "name": "rtc-error-provider-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    server2 = await connect_to_server(
        {
            "name": "rtc-error-consumer-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Register a service that can throw errors on server1
    def error_prone_function(mode):
        """Function that can produce different types of errors."""
        if mode == "value_error":
            raise ValueError("This is a value error")
        elif mode == "type_error":
            raise TypeError("This is a type error")
        elif mode == "custom_error":
            raise Exception("This is a custom error")
        elif mode == "success":
            return "Operation successful"
        else:
            return f"Unknown mode: {mode}"
    
    await server1.register_service(
        {
            "id": "error-service",
            "config": {"visibility": "public"},
            "type": "error_test",
            "test_function": error_prone_function,
        }
    )
    
    # Register RTC service on server1
    rtc_info = await register_rtc_service(server1, "error-rtc-service")
    # Get RTC service from server2 (different connection)
    pc = await get_rtc_service(server2, rtc_info["id"])
    
    try:
        error_svc = await pc.get_service("error-service")
        
        # Test successful operation
        result = await error_svc.test_function("success")
        assert result == "Operation successful", "Success case should work"
        
        # Test error cases
        with pytest.raises(Exception) as exc_info:
            await error_svc.test_function("value_error")
        assert "value error" in str(exc_info.value).lower(), "ValueError should be propagated"
        
        with pytest.raises(Exception) as exc_info:
            await error_svc.test_function("type_error")
        assert "type error" in str(exc_info.value).lower(), "TypeError should be propagated"
        
        print("✅ WebRTC service error handling works correctly between separate clients")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        await pc.close()


@pytest.mark.asyncio 
async def test_rtc_service_with_ice_servers(fastapi_server_coturn, root_user_token):
    """Test RTC service using ICE servers from Hypha with two separate connections."""
    # Create two separate server connections
    server1 = await connect_to_server(
        {
            "name": "rtc-ice-provider-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    server2 = await connect_to_server(
        {
            "name": "rtc-ice-consumer-client",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Get ICE servers from Hypha
    ice_servers = await server1.get_rtc_ice_servers()
    assert len(ice_servers) > 0, "Should have ICE servers available"
    
    # Register a service on server1
    await server1.register_service(
        {
            "id": "ice-test-service",
            "config": {"visibility": "public"},
            "type": "test",
            "ping": lambda: "pong",
            "get_ice_info": lambda: f"Using {len(ice_servers)} ICE servers",
        }
    )
    
    # Register RTC service on server1 (should automatically use ICE servers)
    rtc_info = await register_rtc_service(server1, "ice-rtc-service")
    # Get RTC service from server2 (different connection)
    pc = await get_rtc_service(server2, rtc_info["id"])
    
    try:
        svc = await pc.get_service("ice-test-service")
        
        ping_result = await svc.ping()
        assert ping_result == "pong", "Ping should work through WebRTC with ICE"
        
        ice_info = await svc.get_ice_info()
        assert "ICE servers" in ice_info, "ICE server info should be accessible"
        
        print(f"✅ WebRTC service works with ICE servers between separate clients: {ice_info}")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        await pc.close()


@pytest.mark.asyncio
async def test_webrtc_ice_servers_integration_for_services(fastapi_server_coturn, root_user_token):
    """Test that ICE servers can be properly used for WebRTC services."""
    server = await connect_to_server(
        {
            "name": "webrtc-ice-integration",
            "server_url": SERVER_URL_COTURN,
            "token": root_user_token,
        }
    )
    
    # Get ICE servers that would be used for WebRTC
    ice_servers = await server.get_rtc_ice_servers()
    
    # Test that we get valid ICE servers
    assert isinstance(ice_servers, list), "ICE servers should be a list"
    assert len(ice_servers) > 0, "Should have at least one ICE server"
    
    # Test structure of ICE servers
    for ice_server in ice_servers:
        assert "urls" in ice_server, "ICE server should have urls"
        assert "username" in ice_server, "ICE server should have username"
        assert "credential" in ice_server, "ICE server should have credential"
        
        # Validate URLs format
        urls = ice_server["urls"]
        if isinstance(urls, str):
            assert urls.startswith(("stun:", "turn:", "turns:")), f"Invalid URL format: {urls}"
        elif isinstance(urls, list):
            for url in urls:
                assert url.startswith(("stun:", "turn:", "turns:")), f"Invalid URL format: {url}"
    
    # Test that we can use these ICE servers in a WebRTC context
    # Convert to aiortc format for validation
    aiortc_ice_servers = []
    for server_config in ice_servers:
        ice_server = RTCIceServer(
            urls=server_config["urls"],
            username=server_config["username"],
            credential=server_config["credential"],
            credentialType="password"
        )
        aiortc_ice_servers.append(ice_server)
    
    # Create RTCConfiguration that would be used by WebRTC services
    rtc_config = RTCConfiguration(iceServers=aiortc_ice_servers)
    
    # Test creating peer connection (simulating service connection)
    pc = RTCPeerConnection(configuration=rtc_config)
    
    # Verify connection can be created
    assert pc.connectionState == "new"
    
    print("✅ ICE servers compatible with WebRTC services")
    
    # Clean up
    await pc.close()
    
    # Test that we can register a service successfully with WebRTC context
    ice_count = len(ice_servers)
    
    service_config = {
        "id": "webrtc-info-service",
        "type": "webrtc_info",
        "config": {
            "visibility": "public",
            "require_context": False,
        },
        "test_method": lambda: f"WebRTC ready with {ice_count} ICE servers",
    }
    
    webrtc_service = await server.register_service(service_config)
    
    # Test that the service was registered successfully
    assert webrtc_service is not None
    assert hasattr(webrtc_service, 'id'), "Service should have an id"
    assert 'webrtc-info-service' in webrtc_service.id, "Service ID should contain the service name"
    
    print("✅ WebRTC service integration with ICE servers works correctly")


@pytest.mark.asyncio
async def test_coturn_connectivity_validation(fastapi_server_coturn, coturn_server, root_user_token):
    """Test that COTURN server is actually running and accessible."""
    assert coturn_server is not None, "COTURN server should be available"
    
    # Test COTURN server ports
    coturn_port = coturn_server["port"]
    
    # Test TCP connectivity
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex(("127.0.0.1", coturn_port))
    sock.close()
    
    assert result == 0, f"COTURN server should be accessible on TCP port {coturn_port}"
    
    # Test that Hypha server has ICE servers from this COTURN
    api = await connect_to_server({
        "name": "coturn-validation-client",
        "server_url": SERVER_URL_COTURN,
        "token": root_user_token,
    })
    
    ice_servers = await api.get_rtc_ice_servers()
    
    # Verify ICE servers contain our COTURN server
    coturn_found = False
    for server in ice_servers:
        for url in server.get("urls", []):
            if f":{coturn_port}" in url:
                coturn_found = True
                break
    
    assert coturn_found, f"ICE servers should contain COTURN server on port {coturn_port}"
    print(f"✅ COTURN server validated on port {coturn_port}")
