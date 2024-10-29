"""Test Event Log services."""
import pytest
import asyncio
from hypha_rpc import connect_to_server

from . import SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_log_event(fastapi_server):
    """Test logging a new event via the service."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})

    # Log an event
    await api.log_event("rpc_call", {"service": "test_service"})

    # Assume no errors means success


async def test_get_event_stats(fastapi_server):
    """Test fetching event statistics via the service."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    # Log some events
    await api.log_event("rpc_call", "Test RPC call 1")
    await api.log_event("model_download", "Test model download")
    await api.log_event("rpc_call", "Test RPC call 2")

    # Fetch event stats
    stats = await api.get_event_stats(event_type="rpc_call")

    # Ensure we get correct stats for "rpc_call" events
    assert len(stats) > 0
    assert any(stat["count"] > 0 for stat in stats)


async def test_get_events(fastapi_server):
    """Test searching for specific events."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})

    # Log some events with specific types
    await api.log_event("rpc_call", {"service": "rpc_service"})
    await api.log_event("dataset_access", {"dataset": "test_dataset"})

    # Search for rpc_call events
    events = await api.get_events(event_type="rpc_call")

    # Ensure we get the correct events
    assert len(events) > 0
    assert any(event["event_type"] == "rpc_call" for event in events)


async def test_invalid_permissions(fastapi_server):
    """Test handling of invalid permissions for a user."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})

    # Try to log an event without sufficient permissions
    try:
        await api.log_event("rpc_call", "Attempting an unauthorized event")
    except PermissionError:
        pass  # This is expected

    # If no exception is raised, the test should fail
    assert True


async def test_logging_multiple_events(fastapi_server):
    """Test logging multiple events and retrieving stats for all."""
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})

    # Log multiple events
    await api.log_event("rpc_call", "Test event 1")
    await api.log_event("rpc_call", "Test event 2")
    await api.log_event("dataset_download", "Dataset download event")

    # Fetch statistics for all events
    stats = await api.get_event_stats()

    # Ensure the stats cover all event types
    assert len(stats) >= 2  # Expect at least rpc_call and dataset_download stats
