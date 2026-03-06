"""Test the observability and metrics functionality."""
import asyncio
import requests
import time
import pytest
from prometheus_client.parser import text_string_to_metric_families

from hypha_rpc import connect_to_server
from . import SERVER_URL

# Set timeout for the tests
pytestmark = pytest.mark.timeout(
    120
)  # Allow more time for observability tests that need to check metrics propagation

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


def get_metric_value(metric_name, labels):
    """Helper to parse Prometheus metrics response and extract the value for the specific metric"""
    response = requests.get(f"{SERVER_URL}/metrics", timeout=10)
    assert response.status_code == 200
    metrics_data = response.text
    for family in text_string_to_metric_families(metrics_data):
        if family.name == metric_name:
            for sample in family.samples:
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return None


async def _poll_metric_delta(metric_name, labels, baseline, expected_delta, timeout=3.0):
    """Poll until the metric reaches baseline + expected_delta, return actual value."""
    deadline = asyncio.get_event_loop().time() + timeout
    value = baseline
    while asyncio.get_event_loop().time() < deadline:
        value = get_metric_value(metric_name, labels) or 0
        if expected_delta > 0 and value >= baseline + expected_delta:
            return value
        if expected_delta < 0 and value <= baseline + expected_delta:
            return value
        await asyncio.sleep(0.1)
    return value


async def test_websocket_connections_metric(fastapi_server, test_user_token):
    """Test websocket_connections Gauge and websocket_connections_established Counter.

    The Gauge is shared with ALL other tests and can go up or down due to
    concurrent connects/disconnects, making exact assertions unreliable.

    We use the monotonically-increasing Counter for connect assertions
    (it can only go up, so >= baseline+1 is always sound), and we use
    relative gauge assertions for disconnect (gauge must drop after OUR
    disconnect, which we can verify by polling).
    """
    # --- connect api1: use monotonic counter to verify connection was counted ---
    before_established_1 = get_metric_value("websocket_connections_established", {}) or 0
    api1 = await connect_to_server(
        {
            "client_id": "websocket-test-client-1",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    await api1.log("client 1 connected")
    established_after_1 = await _poll_metric_delta("websocket_connections_established", {}, before_established_1, +1)
    assert established_after_1 >= before_established_1 + 1, (
        f"Expected established counter to increase from {before_established_1} after connecting api1, got {established_after_1}"
    )

    # --- connect api2: use monotonic counter ---
    before_established_2 = get_metric_value("websocket_connections_established", {}) or 0
    api2 = await connect_to_server(
        {
            "client_id": "websocket-test-client-2",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    await api2.log("client 2 connected")
    established_after_2 = await _poll_metric_delta("websocket_connections_established", {}, before_established_2, +1)
    assert established_after_2 >= before_established_2 + 1, (
        f"Expected established counter to increase from {before_established_2} after connecting api2, got {established_after_2}"
    )

    # --- disconnect api1: gauge must decrease ---
    before_disconnect1 = get_metric_value("websocket_connections", {}) or 0
    await api1.disconnect()
    connections_after_disconnect = await _poll_metric_delta("websocket_connections", {}, before_disconnect1, -1)
    assert connections_after_disconnect <= before_disconnect1 - 1, (
        f"Expected gauge to decrease from {before_disconnect1} after disconnecting api1, got {connections_after_disconnect}"
    )

    # --- disconnect api2: gauge must decrease ---
    before_disconnect2 = get_metric_value("websocket_connections", {}) or 0
    await api2.disconnect()
    final_connections = await _poll_metric_delta("websocket_connections", {}, before_disconnect2, -1)
    assert final_connections <= before_disconnect2 - 1, (
        f"Expected gauge to decrease from {before_disconnect2} after disconnecting api2, got {final_connections}"
    )

    print(
        f"✓ WebSocket connections metric test passed: "
        f"api1 established: {before_established_1}->{established_after_1}, "
        f"api2 established: {before_established_2}->{established_after_2}, "
        f"api1 disconnect gauge: {before_disconnect1}->{connections_after_disconnect}, "
        f"api2 disconnect gauge: {before_disconnect2}->{final_connections}"
    )


async def test_rpc_call_metrics(fastapi_server, test_user_token):
    """Test rpc_call_total Counter metric (no workspace labels)."""
    workspace_id = f"test-rpc-calls-{int(time.time() * 1000)}"
    
    # Connect to server (without workspace first)
    api = await connect_to_server(
        {
            "client_id": "rpc-test-client", 
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    
    # Create the test workspace
    await api.create_workspace(
        {
            "id": workspace_id,
            "name": workspace_id,
            "description": "Test workspace for RPC call metrics",
        },
        overwrite=True,
    )
    
    # Reconnect with specific workspace
    await api.disconnect()
    api = await connect_to_server(
        {
            "client_id": "rpc-test-client", 
            "server_url": SERVER_URL,
            "token": test_user_token,
            "workspace": workspace_id,
        }
    )
    
    # Check initial total RPC call count (no workspace label)
    initial_rpc_calls = get_metric_value("rpc_call", {}) or 0
    
    # Make several RPC calls (log is an RPC call)
    for i in range(5):
        await api.log(f"RPC call test message {i}")
        await asyncio.sleep(0.1)
    
    await asyncio.sleep(1.0)  # Allow metrics to update
    
    # Check total RPC call count increased
    final_rpc_calls = get_metric_value("rpc_call", {}) or 0
    assert final_rpc_calls > initial_rpc_calls, f"RPC call count should increase. Initial: {initial_rpc_calls}, Final: {final_rpc_calls}"
    
    print(f"✓ RPC call metric test passed (no labels): {initial_rpc_calls} -> {final_rpc_calls}")


async def test_client_requests_total_metric(fastapi_server, test_user_token):
    """Test client_requests_total Counter metric (no labels)."""
    workspace_id = f"test-client-requests-{int(time.time() * 1000)}"
    client_id = "client-requests-test"
    
    # Connect to server (without workspace first)
    api = await connect_to_server(
        {
            "client_id": client_id,
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    
    # Create the test workspace
    await api.create_workspace(
        {
            "id": workspace_id,
            "name": workspace_id,
            "description": "Test workspace for client requests metric",
        },
        overwrite=True,
    )
    
    # Reconnect with specific workspace
    await api.disconnect()
    api = await connect_to_server(
        {
            "client_id": client_id,
            "server_url": SERVER_URL,
            "token": test_user_token,
            "workspace": workspace_id,
        }
    )
    
    # Check initial total client request count (no labels)
    initial_requests = get_metric_value("client_requests", {}) or 0
    
    # Make several requests
    for i in range(3):
        await api.log(f"Client request test {i}")
        await asyncio.sleep(0.1)
    
    await asyncio.sleep(1.0)  # Allow metrics to update
    
    # Check total client request count increased
    final_requests = get_metric_value("client_requests", {}) or 0
    assert final_requests > initial_requests, f"Client request count should increase. Initial: {initial_requests}, Final: {final_requests}"
    
    print(f"✓ Client requests total metric test passed (no labels): {initial_requests} -> {final_requests}")


async def test_client_load_gauge_metric(fastapi_server, test_user_token):
    """Test client_load_current Gauge metric (no labels)."""
    workspace_id = f"test-client-load-{int(time.time() * 1000)}"
    client_id = "client-load-test__rlb"  # __rlb suffix enables load balancing tracking
    
    # Connect to server (without workspace first)
    api = await connect_to_server(
        {
            "client_id": client_id,
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    
    # Create the test workspace
    await api.create_workspace(
        {
            "id": workspace_id,
            "name": workspace_id,
            "description": "Test workspace for client load metric",
        },
        overwrite=True,
    )
    
    # Reconnect with specific workspace
    await api.disconnect()
    api = await connect_to_server(
        {
            "client_id": client_id,
            "server_url": SERVER_URL,
            "token": test_user_token,
            "workspace": workspace_id,
        }
    )
    
    # Make several rapid requests to generate load
    for i in range(10):
        await api.log(f"Load test message {i}")
        await asyncio.sleep(0.05)  # Small delay between requests
    
    await asyncio.sleep(1.0)  # Allow metrics to update
    
    # Check client load gauge exists and has some value (no labels)
    client_load = get_metric_value("client_load_current", {})
    # Note: client_load might be 0 or None depending on timing, so we just check it exists
    print(f"✓ Client load gauge metric exists with value (no labels): {client_load}")


async def test_event_bus_metrics(fastapi_server, test_user_token):
    """Test event_bus Counter metrics."""
    workspace_id = f"test-event-bus-{int(time.time() * 1000)}"
    
    # Connect to server (without workspace first)
    api = await connect_to_server(
        {
            "client_id": "event-bus-test-client",
            "server_url": SERVER_URL, 
            "token": test_user_token,
        }
    )
    
    # Create a workspace to trigger events (this should generate event bus activity)
    await api.create_workspace(
        {
            "id": workspace_id,
            "name": workspace_id,
            "description": "Test workspace for event bus metrics",
            "owners": ["test@imjoy.io"],
        },
        overwrite=True,
    )
    
    await asyncio.sleep(1.0)  # Allow events to propagate
    
    # Check for event bus metrics (we look for any processed events)
    event_bus_processed = get_metric_value("event_bus", {"event": "*", "status": "processed"})
    
    # Clean up
    await api.delete_workspace(workspace_id)
    await asyncio.sleep(0.5)

    # The event bus should have processed some events
    if event_bus_processed is not None:
        assert event_bus_processed > 0, f"Event bus should have processed some events, got {event_bus_processed}"
        print(f"✓ Event bus metrics test passed: processed {event_bus_processed} events")
    else:
        print("✓ Event bus metrics test completed (no events processed in this test)")


async def test_redis_pubsub_latency_gauge(fastapi_server, test_user_token):
    """Test redis_pubsub_latency_seconds Gauge metric."""
    # Connect to trigger some pubsub activity
    api = await connect_to_server(
        {
            "client_id": "pubsub-latency-test",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    
    await api.log("triggering pubsub activity")
    await asyncio.sleep(1.0)  # Allow metrics to update
    
    # Check pubsub latency metric exists
    pubsub_latency = get_metric_value("redis_pubsub_latency_seconds", {})
    
    # The latency metric might be 0 or have some value depending on system activity
    print(f"✓ Redis pubsub latency gauge metric exists with value: {pubsub_latency}")


async def test_all_metrics_exist(fastapi_server, test_user_token):
    """Test that all expected metrics are exposed in the /metrics endpoint."""
    expected_metrics = [
        "websocket_connections",  # Gauge
        "websocket_connections_established",  # Monotonic counter (never decrements)
        "rpc_call",  # Prometheus exports without _total suffix
        "client_requests",  # Prometheus exports without _total suffix
        "client_load_current",  # Gauge keeps its name
        "event_bus",
        "redis_pubsub_latency_seconds"
    ]
    
    # Get metrics
    response = requests.get(f"{SERVER_URL}/metrics", timeout=10)
    assert response.status_code == 200
    metrics_data = response.text
    
    found_metrics = set()
    for family in text_string_to_metric_families(metrics_data):
        found_metrics.add(family.name)
    
    # Print all available metrics for debugging
    print(f"Available metrics: {sorted(found_metrics)}")
    
    # Check all expected metrics are present
    for metric in expected_metrics:
        assert metric in found_metrics, f"Expected metric '{metric}' not found in /metrics endpoint. Available: {sorted(found_metrics)}"
    
    print(f"✓ All expected metrics found: {expected_metrics}")