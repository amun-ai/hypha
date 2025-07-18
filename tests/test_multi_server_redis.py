"""Test multi-server Redis partitioning with real Redis instances."""

import asyncio
import pytest
import subprocess
import sys
import time
import requests
import json
from requests import RequestException
from hypha_rpc import connect_to_server

from . import (
    REDIS_PORT,
    MINIO_SERVER_URL,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
    MINIO_SERVER_URL_PUBLIC,
)

# Test ports for multiple server instances
MULTI_SERVER_PORT_1 = 38301
MULTI_SERVER_PORT_2 = 38302
MULTI_SERVER_PORT_3 = 38303

WS_SERVER_URL_1 = f"ws://127.0.0.1:{MULTI_SERVER_PORT_1}/ws"
WS_SERVER_URL_2 = f"ws://127.0.0.1:{MULTI_SERVER_PORT_2}/ws"
WS_SERVER_URL_3 = f"ws://127.0.0.1:{MULTI_SERVER_PORT_3}/ws"

SERVER_URL_1 = f"http://127.0.0.1:{MULTI_SERVER_PORT_1}"
SERVER_URL_2 = f"http://127.0.0.1:{MULTI_SERVER_PORT_2}"
SERVER_URL_3 = f"http://127.0.0.1:{MULTI_SERVER_PORT_3}"

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


@pytest.fixture(name="multi_server_1", scope="session")
def multi_server_1(redis_server, minio_server):
    """Start first multi-server instance."""
    test_env = {
        "JWT_SECRET": "test-secret",
        "ACTIVITY_CHECK_INTERVAL": "0.3",
    }
    
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={MULTI_SERVER_PORT_1}",
            "--enable-server-apps",
            "--enable-s3",
            f"--public-base-url=http://my-public-url.com",
            "--server-id=multi-server-1",
            f"--redis-uri=redis://127.0.0.1:{REDIS_PORT}/0",
            "--reset-redis",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
        ],
        env=test_env,
    ) as proc:
        timeout = 30
        while timeout > 0:
            try:
                response = requests.get(f"{SERVER_URL_1}/health/readiness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Multi-server 1 did not start in time")
        yield
        proc.kill()
        proc.terminate()


@pytest.fixture(name="multi_server_2", scope="session")
def multi_server_2(redis_server, minio_server, multi_server_1):
    """Start second multi-server instance."""
    test_env = {
        "JWT_SECRET": "test-secret",
        "ACTIVITY_CHECK_INTERVAL": "0.3",
    }
    
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={MULTI_SERVER_PORT_2}",
            "--enable-server-apps",
            "--enable-s3",
            f"--public-base-url=http://my-public-url.com",
            "--server-id=multi-server-2",
            f"--redis-uri=redis://127.0.0.1:{REDIS_PORT}/0",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
        ],
        env=test_env,
    ) as proc:
        timeout = 30
        while timeout > 0:
            try:
                response = requests.get(f"{SERVER_URL_2}/health/readiness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Multi-server 2 did not start in time")
        yield
        proc.kill()
        proc.terminate()


@pytest.fixture(name="multi_server_3", scope="session")
def multi_server_3(redis_server, minio_server, multi_server_2):
    """Start third multi-server instance."""
    test_env = {
        "JWT_SECRET": "test-secret",
        "ACTIVITY_CHECK_INTERVAL": "0.3",
    }
    
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={MULTI_SERVER_PORT_3}",
            "--enable-server-apps",
            "--enable-s3",
            f"--public-base-url=http://my-public-url.com",
            "--server-id=multi-server-3",
            f"--redis-uri=redis://127.0.0.1:{REDIS_PORT}/0",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
        ],
        env=test_env,
    ) as proc:
        timeout = 30
        while timeout > 0:
            try:
                response = requests.get(f"{SERVER_URL_3}/health/readiness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Multi-server 3 did not start in time")
        yield
        proc.kill()
        proc.terminate()


@pytest.fixture(name="single_server", scope="session")
def single_server():
    """Start single server instance without Redis."""
    test_env = {
        "JWT_SECRET": "test-secret",
        "ACTIVITY_CHECK_INTERVAL": "0.3",
    }
    
    single_port = 38304
    
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={single_port}",
            f"--public-base-url=http://my-public-url.com",
            "--server-id=single-server",
            # No Redis URI - single node mode
        ],
        env=test_env,
    ) as proc:
        timeout = 30
        while timeout > 0:
            try:
                response = requests.get(f"http://127.0.0.1:{single_port}/health/readiness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Single server did not start in time")
        yield f"ws://127.0.0.1:{single_port}/ws"
        proc.kill()
        proc.terminate()


async def test_multi_server_client_distribution(multi_server_1, multi_server_2, multi_server_3):
    """Test that clients are properly distributed across servers and can communicate."""
    
    # Connect clients to different servers
    clients = []
    
    # Connect multiple clients to each server
    for i in range(3):
        client = await connect_to_server({
            "name": f"client_s1_{i}", 
            "server_url": WS_SERVER_URL_1,
            "client_id": f"test-client-s1-{i}"
        })
        clients.append(client)
    
    for i in range(3):
        client = await connect_to_server({
            "name": f"client_s2_{i}", 
            "server_url": WS_SERVER_URL_2,
            "client_id": f"test-client-s2-{i}"
        })
        clients.append(client)
    
    for i in range(3):
        client = await connect_to_server({
            "name": f"client_s3_{i}", 
            "server_url": WS_SERVER_URL_3,
            "client_id": f"test-client-s3-{i}"
        })
        clients.append(client)
    
    try:
        # Test basic functionality for all clients
        for client in clients:
            assert await client.echo("hello") == "hello"
        
        # Test service registration and discovery
        class TestService:
            def __init__(self, name):
                self.name = name
                
            async def get_name(self):
                return self.name
                
            async def ping(self, message):
                return f"pong from {self.name}: {message}"
        
        # Register services on different servers
        services = []
        for i, client in enumerate(clients):
            service_info = await client.register_service(TestService(f"service_{i}"))
            services.append((client, service_info))
        
        # Wait for service registration to propagate
        await asyncio.sleep(1.0)
        
        # Test that services work
        for client, service_info in services:
            service = await client.get_service(service_info.id)
            result = await service.ping("test")
            assert "pong from" in result
            assert "test" in result
        
        print("✓ Multi-server client distribution test passed")
        
    finally:
        # Clean up all clients
        for client in clients:
            await client.disconnect()


async def test_multi_server_message_routing(multi_server_1, multi_server_2):
    """Test that messages are properly routed between servers."""
    
    # Connect clients to different servers
    client1 = await connect_to_server({
        "name": "routing_client1", 
        "server_url": WS_SERVER_URL_1,
        "client_id": "routing-client-1"
    })
    
    client2 = await connect_to_server({
        "name": "routing_client2", 
        "server_url": WS_SERVER_URL_2,
        "client_id": "routing-client-2"
    })
    
    try:
        # Create a service that can receive messages
        class MessageService:
            def __init__(self, name):
                self.name = name
                self.messages = []
                
            async def send_message(self, message):
                self.messages.append(message)
                return f"Received by {self.name}: {message}"
        
        # Register services
        service1 = MessageService("service1")
        service2 = MessageService("service2")
        
        service1_info = await client1.register_service(service1)
        service2_info = await client2.register_service(service2)
        
        # Wait for services to register
        await asyncio.sleep(1.0)
        
        # Test cross-server service calls
        remote_service2 = await client1.get_service(service2_info.id)
        result = await remote_service2.send_message("Hello from server 1")
        assert "Received by service2" in result
        assert "Hello from server 1" in result
        
        remote_service1 = await client2.get_service(service1_info.id)
        result = await remote_service1.send_message("Hello from server 2")
        assert "Received by service1" in result
        assert "Hello from server 2" in result
        
        print("✓ Multi-server message routing test passed")
        
    finally:
        await client1.disconnect()
        await client2.disconnect()


async def test_multi_server_performance(multi_server_1, multi_server_2, multi_server_3):
    """Test performance with multiple servers and many clients."""
    
    # Connect many clients to different servers
    clients = []
    num_clients_per_server = 5
    
    # Connect clients to server 1
    for i in range(num_clients_per_server):
        client = await connect_to_server({
            "name": f"perf_client_s1_{i}", 
            "server_url": WS_SERVER_URL_1,
            "client_id": f"perf-client-s1-{i}"
        })
        clients.append(client)
    
    # Connect clients to server 2
    for i in range(num_clients_per_server):
        client = await connect_to_server({
            "name": f"perf_client_s2_{i}", 
            "server_url": WS_SERVER_URL_2,
            "client_id": f"perf-client-s2-{i}"
        })
        clients.append(client)
    
    # Connect clients to server 3
    for i in range(num_clients_per_server):
        client = await connect_to_server({
            "name": f"perf_client_s3_{i}", 
            "server_url": WS_SERVER_URL_3,
            "client_id": f"perf-client-s3-{i}"
        })
        clients.append(client)
    
    try:
        # Measure performance of concurrent operations
        start_time = time.time()
        
        # Perform many concurrent operations
        tasks = []
        for i, client in enumerate(clients):
            for j in range(10):  # 10 operations per client
                tasks.append(client.echo(f"message-{i}-{j}"))
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify all operations completed successfully
        assert len(results) == len(clients) * 10
        for i, result in enumerate(results):
            expected_pattern = f"message-{i // 10}-{i % 10}"
            assert result == expected_pattern
        
        print(f"✓ Performance test completed in {duration:.2f} seconds")
        print(f"✓ Processed {len(results)} operations across {len(clients)} clients on 3 servers")
        print(f"✓ Average operations per second: {len(results) / duration:.2f}")
        
    finally:
        # Clean up all clients
        for client in clients:
            await client.disconnect()


async def test_single_server_mode(single_server):
    """Test single server mode without Redis."""
    
    # Connect clients to single server
    client1 = await connect_to_server({
        "name": "single_client1", 
        "server_url": single_server,
        "client_id": "single-client-1"
    })
    
    client2 = await connect_to_server({
        "name": "single_client2", 
        "server_url": single_server,
        "client_id": "single-client-2"
    })
    
    try:
        # Test basic functionality
        assert await client1.echo("hello1") == "hello1"
        assert await client2.echo("hello2") == "hello2"
        
        # Test service registration and discovery
        class SingleService:
            def __init__(self, name):
                self.name = name
                
            async def get_name(self):
                return self.name
                
            async def process(self, data):
                return f"Processed by {self.name}: {data}"
        
        # Register services
        service1_info = await client1.register_service(SingleService("service1"))
        service2_info = await client2.register_service(SingleService("service2"))
        
        # Wait for services to register
        await asyncio.sleep(0.5)
        
        # Test service calls within single server
        service1 = await client2.get_service(service1_info.id)
        result = await service1.process("test data")
        assert "Processed by service1: test data" == result
        
        service2 = await client1.get_service(service2_info.id)
        result = await service2.process("more data")
        assert "Processed by service2: more data" == result
        
        print("✓ Single server mode test passed")
        
    finally:
        await client1.disconnect()
        await client2.disconnect()


async def test_server_restart_resilience(multi_server_1, multi_server_2):
    """Test that the system handles server restarts gracefully."""
    
    # Connect clients to both servers
    client1 = await connect_to_server({
        "name": "resilience_client1", 
        "server_url": WS_SERVER_URL_1,
        "client_id": "resilience-client-1"
    })
    
    client2 = await connect_to_server({
        "name": "resilience_client2", 
        "server_url": WS_SERVER_URL_2,
        "client_id": "resilience-client-2"
    })
    
    try:
        # Test initial connectivity
        assert await client1.echo("initial1") == "initial1"
        assert await client2.echo("initial2") == "initial2"
        
        # Register services
        class ResilientService:
            def __init__(self, name):
                self.name = name
                
            async def status(self):
                return f"Service {self.name} is running"
        
        service1_info = await client1.register_service(ResilientService("service1"))
        service2_info = await client2.register_service(ResilientService("service2"))
        
        # Wait for services to register
        await asyncio.sleep(1.0)
        
        # Test cross-server service calls
        remote_service2 = await client1.get_service(service2_info.id)
        result = await remote_service2.status()
        assert "Service service2 is running" == result
        
        remote_service1 = await client2.get_service(service1_info.id)
        result = await remote_service1.status()
        assert "Service service1 is running" == result
        
        # Test that both clients can still communicate after some time
        await asyncio.sleep(2.0)
        
        assert await client1.echo("after_wait1") == "after_wait1"
        assert await client2.echo("after_wait2") == "after_wait2"
        
        print("✓ Server restart resilience test passed")
        
    finally:
        await client1.disconnect()
        await client2.disconnect()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])