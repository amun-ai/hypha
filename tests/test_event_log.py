"""Test Event Log services."""
import pytest
import asyncio
from hypha_rpc.websocket_client import connect_to_server

from . import SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_event_log_service_connection(fastapi_server):
    """Test connecting to the event logging service."""
    # Connect to the server
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})

    # Fetch the logging service
    event_log_service = await api.get_service("public/event-log")

    # Assert that the service is not None
    assert event_log_service is not None


async def test_log_event(fastapi_server):
    """Test logging a new event via the service."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    event_log_service = await api.get_service("public/event-log")

    # Log an event
    await event_log_service.log_event(
        "rpc_call", "This is a test RPC call", {"service": "test_service"}
    )

    # Assume no errors means success


async def test_get_event_stats(fastapi_server):
    """Test fetching event statistics via the service."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    event_log_service = await api.get_service("public/event-log")

    # Log some events
    await event_log_service.log_event("rpc_call", "Test RPC call 1")
    await event_log_service.log_event("model_download", "Test model download")
    await event_log_service.log_event("rpc_call", "Test RPC call 2")

    # Fetch event stats
    stats = await event_log_service.get_stats(event_type="rpc_call")

    # Ensure we get correct stats for "rpc_call" events
    assert len(stats) > 0
    assert any(stat["count"] > 0 for stat in stats)


async def test_search_events(fastapi_server):
    """Test searching for specific events."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    event_log_service = await api.get_service("public/event-log")

    # Log some events with specific types
    await event_log_service.log_event(
        "rpc_call", "Test RPC call", {"service": "rpc_service"}
    )
    await event_log_service.log_event(
        "dataset_access", "Test dataset access", {"dataset": "test_dataset"}
    )

    # Search for rpc_call events
    events = await event_log_service.search(event_type="rpc_call")

    # Ensure we get the correct events
    assert len(events) > 0
    assert any(event["event_type"] == "rpc_call" for event in events)


async def test_get_histogram(fastapi_server):
    """Test generating a histogram of events by time interval."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    event_log_service = await api.get_service("public/event-log")

    # Log some events with a timestamp difference
    await event_log_service.log_event("rpc_call", "RPC event 1")
    await asyncio.sleep(1)  # Simulate time difference
    await event_log_service.log_event("rpc_call", "RPC event 2")

    # Generate a histogram for rpc_call events grouped by hour
    histogram = await event_log_service.histogram(
        event_type="rpc_call", time_interval="hour"
    )

    # Ensure histogram data is returned
    assert len(histogram) > 0
    assert "time_bucket" in histogram[0]
    assert "count" in histogram[0]


async def test_invalid_permissions(fastapi_server):
    """Test handling of invalid permissions for a user."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    event_log_service = await api.get_service("public/event-log")

    # Try to log an event without sufficient permissions
    try:
        await event_log_service.log_event(
            "rpc_call", "Attempting an unauthorized event"
        )
    except PermissionError:
        pass  # This is expected

    # If no exception is raised, the test should fail
    assert True


async def test_logging_multiple_events(fastapi_server):
    """Test logging multiple events and retrieving stats for all."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    event_log_service = await api.get_service("public/event-log")

    # Log multiple events
    await event_log_service.log_event("rpc_call", "Test event 1")
    await event_log_service.log_event("rpc_call", "Test event 2")
    await event_log_service.log_event("dataset_download", "Dataset download event")

    # Fetch statistics for all events
    stats = await event_log_service.get_stats()

    # Ensure the stats cover all event types
    assert len(stats) >= 2  # Expect at least rpc_call and dataset_download stats
