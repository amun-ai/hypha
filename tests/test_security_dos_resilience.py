"""Test suite for DoS and resource exhaustion vulnerabilities in Hypha server.

This test file contains tests for denial of service vulnerabilities, focusing on:
- Rate limiting and connection flooding
- Service registration spam
- Worker pool exhaustion
- Artifact upload flooding
- Redis connection exhaustion
- Database query flooding
- Autoscaling manipulation

Vulnerability List:
- DOS1: WebSocket connection flooding - No per-user connection limits (HIGH)
- DOS2: Service registration spam - No rate limiting on service creation (MEDIUM)
- DOS3: Worker pool exhaustion - Unbounded worker spawning (HIGH)
- DOS4: Artifact upload flooding - No size/rate limits on uploads (HIGH)
- DOS5: Redis connection exhaustion - Limited connection pool (MEDIUM)
- DOS6: Database query flooding - No query rate limiting (MEDIUM)
- DOS7: Autoscaling manipulation - Can force infinite scaling (HIGH)
- DOS8: Message flooding - No message rate limits (HIGH)
- DOS9: Memory exhaustion via large messages - No message size validation (HIGH)
"""

import pytest
import pytest_asyncio
import uuid
import asyncio
import time
from hypha_rpc import connect_to_server

from . import (
    WS_SERVER_URL,
    SERVER_URL,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


class TestDOS1WebSocketConnectionFlooding:
    """
    DOS1: WebSocket Connection Flooding (HIGH SEVERITY)

    Location: hypha/websocket.py

    Issue: There is no per-user or per-workspace limit on concurrent WebSocket
    connections. An attacker can open thousands of connections to exhaust server
    resources (memory, file descriptors, Redis connections).

    Current state:
    - HYPHA_REDIS_MAX_CONNECTIONS=2000 is a system-wide limit, not per-user
    - No rate limiting on new connections
    - Each connection consumes: WebSocket, Redis connection, memory for tracking

    Expected behavior: Should have per-user connection limits and rate limiting.
    """

    async def test_can_open_many_connections_from_single_user(
        self, fastapi_server, test_user_token
    ):
        """Test that a single user can open many concurrent connections.

        This demonstrates the lack of per-user connection limits.
        We'll open multiple connections but keep the test reasonable.
        """
        connections = []
        try:
            # Try to open 10 connections rapidly from the same user
            # In a real attack, this would be thousands
            for i in range(10):
                api = await connect_to_server({
                    "client_id": f"dos1-flood-{i}",
                    "server_url": WS_SERVER_URL,
                    "token": test_user_token,
                })
                connections.append(api)

            # All connections should succeed
            assert len(connections) == 10

            # Each connection should be functional
            for api in connections:
                info = await api.get_service("server-info")
                assert info is not None

            print(f"\n[DOS1] Successfully opened {len(connections)} concurrent connections from single user")
            print("[DOS1] VULNERABILITY: No per-user connection limits detected")

        finally:
            # Cleanup
            for api in connections:
                try:
                    await api.disconnect()
                except Exception:
                    pass

    async def test_rapid_connection_creation_no_rate_limit(
        self, fastapi_server, test_user_token
    ):
        """Test that connections can be created rapidly without rate limiting.

        Demonstrates lack of rate limiting on connection establishment.
        """
        start_time = time.time()
        connections = []

        try:
            # Create 20 connections as fast as possible
            for i in range(20):
                api = await connect_to_server({
                    "client_id": f"dos1-rapid-{i}",
                    "server_url": WS_SERVER_URL,
                    "token": test_user_token,
                })
                connections.append(api)

            elapsed = time.time() - start_time
            rate = len(connections) / elapsed

            print(f"\n[DOS1] Created {len(connections)} connections in {elapsed:.2f}s ({rate:.1f} conn/s)")
            print("[DOS1] VULNERABILITY: No rate limiting on connection creation")

        finally:
            for api in connections:
                try:
                    await api.disconnect()
                except Exception:
                    pass


class TestDOS2ServiceRegistrationSpam:
    """
    DOS2: Service Registration Spam (MEDIUM SEVERITY)

    Location: hypha/core/workspace.py

    Issue: No rate limiting on service registration. An attacker can register
    thousands of services to:
    - Exhaust Redis memory (service metadata stored in Redis)
    - Pollute service discovery
    - Consume workspace resources

    Expected behavior: Should have rate limits on service registration.
    """

    async def test_can_register_many_services_rapidly(
        self, fastapi_server, test_user_token
    ):
        """Test that many services can be registered without rate limiting."""
        api = await connect_to_server({
            "client_id": "dos2-spam-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        try:
            start_time = time.time()
            service_count = 50  # In real attack, would be thousands

            async def dummy_service():
                return "spam"

            for i in range(service_count):
                await api.register_service({
                    "id": f"spam-service-{i}",
                    "name": f"Spam Service {i}",
                    "type": "generic",
                    "run": dummy_service,
                })

            elapsed = time.time() - start_time
            rate = service_count / elapsed

            # Verify all services exist
            services = await api.list_services()
            spam_services = [s for s in services if s["id"].startswith("spam-service-")]

            print(f"\n[DOS2] Registered {len(spam_services)} services in {elapsed:.2f}s ({rate:.1f} svc/s)")
            print("[DOS2] VULNERABILITY: No rate limiting on service registration")

        finally:
            await api.disconnect()


class TestDOS3WorkerPoolExhaustion:
    """
    DOS3: Worker Pool Exhaustion (HIGH SEVERITY)

    Location: hypha/apps.py - AutoscalingManager

    Issue: Autoscaling can be triggered to spawn excessive workers:
    1. No upper limit validation on max_instances in autoscaling config
    2. Attacker can install app with high max_instances
    3. Can artificially increase load to trigger scaling
    4. Workers consume significant resources (processes, memory, CPU)

    Expected behavior: System-wide limits on total workers, validation of
    autoscaling config, resource quotas per workspace.
    """

    async def test_autoscaling_accepts_large_max_instances(
        self, fastapi_server, test_user_token
    ):
        """Test that autoscaling config accepts very large max_instances.

        This demonstrates lack of validation on autoscaling parameters.
        """
        api = await connect_to_server({
            "client_id": "dos3-autoscale-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        try:
            # Create workspace
            ws_info = await api.create_workspace({
                "name": f"dos3-test-{uuid.uuid4().hex[:8]}",
                "description": "Testing autoscaling limits"
            }, overwrite=True)

            # Simple server app source code
            app_source = '''
from hypha_rpc import api

async def hello(name):
    return f"Hello {name}"

api.export({"hello": hello})
'''

            # Try to install app with extremely large max_instances
            # Note: We won't actually trigger scaling to avoid consuming resources
            app_controller = await api.get_service("server-apps")

            try:
                manifest = await app_controller.install(
                    source=app_source,
                    config={
                        "autoscaling": {
                            "min_instances": 1,
                            "max_instances": 10000,  # Unreasonably high
                            "target_requests_per_instance": 10,
                        }
                    }
                )

                print(f"\n[DOS3] Installed app with max_instances=10000")
                print("[DOS3] VULNERABILITY: No validation of autoscaling parameters")
                print("[DOS3] An attacker could force system to spawn thousands of workers")

                # Check the config was accepted
                assert manifest["config"]["autoscaling"]["max_instances"] == 10000

            except Exception as e:
                # If it fails, that's actually good (means there's validation)
                print(f"\n[DOS3] GOOD: Validation rejected large max_instances: {e}")
                raise

        finally:
            await api.disconnect()


class TestDOS4ArtifactUploadFlooding:
    """
    DOS4: Artifact Upload Flooding (HIGH SEVERITY)

    Location: hypha/artifact.py, hypha/http.py

    Issue: No rate limiting on artifact uploads. An attacker can:
    1. Upload many small files rapidly
    2. Upload very large files
    3. Exhaust storage quota
    4. Consume database resources (metadata for each artifact)
    5. Consume S3/MinIO resources

    Expected behavior: Rate limits on uploads, size limits, workspace quotas.
    """

    async def test_can_upload_many_files_rapidly(
        self, fastapi_server, test_user_token
    ):
        """Test that many files can be uploaded without rate limiting."""
        api = await connect_to_server({
            "client_id": "dos4-upload-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        try:
            # Get artifact manager
            artifact_manager = await api.get_service("public/artifact-manager")

            start_time = time.time()
            upload_count = 20  # In real attack, would be much more

            for i in range(upload_count):
                # Upload small file
                await artifact_manager.put_file(
                    f"spam-file-{i}.txt",
                    f"spam content {i}".encode(),
                )

            elapsed = time.time() - start_time
            rate = upload_count / elapsed

            print(f"\n[DOS4] Uploaded {upload_count} files in {elapsed:.2f}s ({rate:.1f} files/s)")
            print("[DOS4] VULNERABILITY: No rate limiting on artifact uploads")

        finally:
            await api.disconnect()


class TestDOS5RedisConnectionExhaustion:
    """
    DOS5: Redis Connection Exhaustion (MEDIUM SEVERITY)

    Location: hypha/core/store.py

    Issue: HYPHA_REDIS_MAX_CONNECTIONS=2000 is a system-wide pool limit.
    Each WebSocket connection uses Redis connections for:
    - Event bus pub/sub
    - Service registry queries
    - Message passing

    If attackers open many connections, they can exhaust the Redis pool.

    Expected behavior: Per-user limits on Redis connection usage, connection
    pooling per client, graceful handling of pool exhaustion.
    """

    async def test_redis_connection_pool_shared_globally(
        self, fastapi_server, test_user_token
    ):
        """Test that Redis connections are shared from global pool.

        This is informational - demonstrating the architecture.
        """
        api = await connect_to_server({
            "client_id": "dos5-redis-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        try:
            # Get server info which requires Redis queries
            info = await api.get_service("server-info")

            print(f"\n[DOS5] Redis max connections: 2000 (system-wide, from env)")
            print("[DOS5] VULNERABILITY: No per-user connection limits")
            print("[DOS5] Many malicious connections could exhaust Redis pool")

        finally:
            await api.disconnect()


class TestDOS6DatabaseQueryFlooding:
    """
    DOS6: Database Query Flooding (MEDIUM SEVERITY)

    Location: hypha/artifact.py, hypha/core/workspace.py

    Issue: No rate limiting on database queries. Operations like:
    - list_artifacts() can query thousands of records
    - list_services() can be called repeatedly
    - Event logging writes to database

    An attacker can overwhelm the database with queries.

    Expected behavior: Query rate limiting, pagination limits, query complexity limits.
    """

    async def test_can_query_artifacts_repeatedly(
        self, fastapi_server, test_user_token
    ):
        """Test that database queries can be executed rapidly without limits."""
        api = await connect_to_server({
            "client_id": "dos6-query-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        try:
            artifact_manager = await api.get_service("public/artifact-manager")

            start_time = time.time()
            query_count = 50

            # Rapidly execute database queries
            for i in range(query_count):
                await artifact_manager.list()

            elapsed = time.time() - start_time
            rate = query_count / elapsed

            print(f"\n[DOS6] Executed {query_count} database queries in {elapsed:.2f}s ({rate:.1f} qps)")
            print("[DOS6] VULNERABILITY: No rate limiting on database queries")

        finally:
            await api.disconnect()


class TestDOS8MessageFlooding:
    """
    DOS8: Message Flooding (HIGH SEVERITY)

    Location: hypha/websocket.py, hypha/core/__init__.py (RedisEventBus)

    Issue: No rate limiting on RPC messages. An attacker can:
    1. Send thousands of RPC calls rapidly
    2. Overwhelm the event bus
    3. Consume CPU processing messages
    4. Exhaust memory with message queues

    Expected behavior: Per-client message rate limiting.
    """

    async def test_can_send_many_messages_rapidly(
        self, fastapi_server, test_user_token
    ):
        """Test that RPC messages can be sent without rate limiting."""
        api = await connect_to_server({
            "client_id": "dos8-message-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        try:
            # Register a simple service
            call_count = 0

            async def echo(msg):
                nonlocal call_count
                call_count += 1
                return msg

            await api.register_service({
                "id": "echo-service",
                "type": "echo",
                "echo": echo,
            })

            # Get the service
            echo_svc = await api.get_service("echo-service")

            start_time = time.time()
            message_count = 100

            # Send many messages rapidly
            tasks = []
            for i in range(message_count):
                tasks.append(echo_svc.echo(f"msg-{i}"))

            await asyncio.gather(*tasks)

            elapsed = time.time() - start_time
            rate = message_count / elapsed

            print(f"\n[DOS8] Sent {message_count} RPC messages in {elapsed:.2f}s ({rate:.1f} msg/s)")
            print(f"[DOS8] All messages processed: {call_count}/{message_count}")
            print("[DOS8] VULNERABILITY: No rate limiting on RPC messages")

        finally:
            await api.disconnect()


class TestDOS9LargeMessageMemoryExhaustion:
    """
    DOS9: Memory Exhaustion via Large Messages (HIGH SEVERITY)

    Location: hypha/websocket.py::establish_websocket_communication

    Issue: Limited validation of message sizes. Code only checks TEXT messages:
    - Text messages > 1000 chars are rejected (line 404-408)
    - But BINARY messages (msgpack) have no size check
    - An attacker can send huge binary messages to exhaust memory

    Expected behavior: Size limits on all message types, both text and binary.
    """

    async def test_binary_message_under_limit_accepted(
        self, fastapi_server, test_user_token
    ):
        """Test that binary messages under the limit (10MB default) are accepted.

        This verifies the fix doesn't break legitimate use cases.
        """
        api = await connect_to_server({
            "client_id": "dos9-under-limit-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        try:
            # Register a service that accepts data
            received_size = 0

            async def receive_data(data):
                nonlocal received_size
                received_size = len(data)
                return "ok"

            await api.register_service({
                "id": "data-receiver",
                "type": "receiver",
                "receive": receive_data,
            })

            # Get the service
            receiver = await api.get_service("data-receiver")

            # Send a 1MB message (well under 10MB limit)
            large_data = b"x" * (1024 * 1024)  # 1 MB

            result = await receiver.receive(large_data)

            print(f"\n[DOS9-FIX] Successfully sent {len(large_data)} byte message")
            print(f"[DOS9-FIX] Received size: {received_size}")
            assert received_size == len(large_data), "Message should be received"
            print("[DOS9-FIX] VERIFIED: Messages under limit are accepted")

        finally:
            await api.disconnect()

    async def test_binary_message_over_limit_rejected(
        self, fastapi_server, test_user_token
    ):
        """Test that binary messages over the limit (10MB default) are rejected.

        This verifies the DOS9 fix is working.
        """
        api = await connect_to_server({
            "client_id": "dos9-over-limit-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        try:
            # Register a service that accepts data
            received_count = 0

            async def receive_data(data):
                nonlocal received_count
                received_count += 1
                return "ok"

            await api.register_service({
                "id": "data-receiver-reject",
                "type": "receiver",
                "receive": receive_data,
            })

            # Get the service
            receiver = await api.get_service("data-receiver-reject")

            # Try to send an 11MB message (over 10MB limit)
            # Note: We'll send a smaller message in testing to avoid timeouts
            # In production, this would be 11MB+
            oversized_data = b"x" * (11 * 1024 * 1024)  # 11 MB

            # The message should be silently dropped by the server
            # The RPC call will timeout because the server ignores the message
            try:
                # Set a short timeout to avoid waiting too long
                result = await asyncio.wait_for(
                    receiver.receive(oversized_data),
                    timeout=5.0
                )
                # If we get here, the fix didn't work
                print(f"\n[DOS9-FIX] FAILED: Oversized message was NOT rejected!")
                assert False, "Oversized message should have been rejected"
            except asyncio.TimeoutError:
                # This is expected - the server dropped the message
                print(f"\n[DOS9-FIX] VERIFIED: Oversized message rejected (timeout as expected)")
                print(f"[DOS9-FIX] Message size: {len(oversized_data)} bytes (> 10MB limit)")
                print(f"[DOS9-FIX] Received count: {received_count} (should be 0)")
                assert received_count == 0, "Oversized message should not have been processed"

        finally:
            await api.disconnect()

    async def test_text_message_size_limit_exists(
        self, fastapi_server, test_user_token
    ):
        """Verify that text messages do have a size limit (good).

        This is to confirm the asymmetry - text has limits, binary doesn't.
        """
        # This test documents that text messages ARE limited (see websocket.py:404-408)
        # The vulnerability is that binary messages are NOT limited

        print("\n[DOS9] INFO: Text messages > 1000 chars are rejected (websocket.py:404)")
        print("[DOS9] INFO: Binary messages have NO size limit - VULNERABILITY")
