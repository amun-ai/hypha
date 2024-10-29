"""Test the hypha server."""
import os
import subprocess
import sys
import asyncio

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
    """Test Prometheus metrics for workspace and service creation."""
    # Connect to the server and create a new workspace
    api = await connect_to_server(
        {
            "client_id": "my-app-99",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    await api.log("hello")

    # Check the initial number of active workspaces
    initial_active_workspaces = get_metric_value("active_workspaces", {})
    assert initial_active_workspaces is not None

    # Create a new workspace
    await api.create_workspace(
        {
            "name": "my-test-workspace-metrics",
            "description": "This is a test workspace",
            "owners": ["user1@imjoy.io", "user2@imjoy.io"],
        },
        overwrite=True,
    )

    # Check if the number of active workspaces has increased
    active_workspaces = get_metric_value("active_workspaces", {})
    assert (
        active_workspaces == initial_active_workspaces + 1
    ), "Active workspace count did not increase."

    # Check if a service was added to the workspace
    active_services = get_metric_value(
        "active_services", {"workspace": "my-test-workspace-metrics"}
    )
    assert active_services is None, "Expected no active services in the new workspace."

    # Verify that other services and workspaces haven't changed unexpectedly
    public_active_services = get_metric_value(
        "active_services", {"workspace": "public"}
    )
    assert (
        public_active_services is not None
    ), "Public active services metric is missing."

    # Optionally check that the RPC call metric is functioning
    rpc_call_count = get_metric_value("rpc_call", {"workspace": api.config.workspace})
    assert (
        rpc_call_count is not None
    ), "Expected an RPC call metric for the new workspace."
