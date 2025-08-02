"""Test the hypha server."""

import os
import subprocess
import sys
import asyncio
import time

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families
from hypha_rpc import connect_to_server

from . import (
    SERVER_URL,
    SERVER_URL_REDIS_1,
    SERVER_URL_REDIS_2,
    SIO_PORT2,
    WS_SERVER_URL,
    find_item,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


def get_metric_value(metric_name, labels):
    """Helper to parse Prometheus metrics response and extract the value for the specific metric"""
    response = requests.get(f"{SERVER_URL}/metrics")
    assert response.status_code == 200
    metrics_data = response.text
    for family in text_string_to_metric_families(metrics_data):
        if family.name == metric_name:
            for sample in family.samples:
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return None


async def test_metrics(fastapi_server, test_user_token):
    """Test Prometheus metrics for workspace management."""
    # Create a unique workspace ID using timestamp to ensure it's always new
    workspace_id = f"test-metrics-ws-{int(time.time() * 1000)}"
    
    # Connect to the server
    api = await connect_to_server(
        {
            "client_id": "metrics-test-client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    await api.log("hello")

    # Check the initial number of active workspaces
    initial_active_workspaces = get_metric_value("active_workspaces", {})
    assert initial_active_workspaces is not None, "active_workspaces metric should exist"

    # Create a new workspace
    workspace_info = await api.create_workspace(
        {
            "id": workspace_id,
            "name": workspace_id,
            "description": "Test workspace for metrics",
            "owners": ["test@imjoy.io"],
        },
        overwrite=False,
    )

    # Wait for metrics to update
    await asyncio.sleep(1.0)

    # Check if the number of active workspaces has increased
    active_workspaces = get_metric_value("active_workspaces", {})
    assert active_workspaces is not None, "active_workspaces metric should exist after creation"
    assert (
        active_workspaces == initial_active_workspaces + 1
    ), f"Active workspace count should increase by 1. Expected: {initial_active_workspaces + 1}, Got: {active_workspaces}"

    # Clean up: delete the workspace
    await api.delete_workspace(workspace_id)
    await asyncio.sleep(1.0)
    
    # Check if workspace count decreased
    active_workspaces_after = get_metric_value("active_workspaces", {})
    assert active_workspaces_after is not None, "active_workspaces metric should still exist"
    assert (
        active_workspaces_after == initial_active_workspaces
    ), f"Active workspace count should return to initial value. Expected: {initial_active_workspaces}, Got: {active_workspaces_after}"
    
    print(f"✓ Metrics test passed: workspace count {initial_active_workspaces} -> {active_workspaces} -> {active_workspaces_after}")
