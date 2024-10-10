"""Test the hypha server."""
import os
import subprocess
import sys
import asyncio

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families
from hypha_rpc.websocket_client import connect_to_server

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


async def test_metrics(fastapi_server):
    """Test metrics."""
    assert get_metric_value("active_services", {"workspace": "ws-user-root"}) >= 1
