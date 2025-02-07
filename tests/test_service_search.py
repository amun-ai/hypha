"""Test the hypha server."""

import numpy as np

import pytest
from hypha_rpc import connect_to_server

from . import (
    SERVER_URL_REDIS_1,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_service_search(fastapi_server_redis_1, test_user_token):
    """Test service search with registered services."""
    api = await connect_to_server(
        {
            "client_id": "my-app-99",
            "server_url": SERVER_URL_REDIS_1,
            "token": test_user_token,
        }
    )

    # Register sample services with unique `docs` field
    await api.register_service(
        {
            "id": "service-1",
            "name": "example service one",
            "type": "my-type",
            "description": "This is the first test service.",
            "app_id": "my-app",
            "service_schema": {"example_key": "example_value"},
            "config": {"setting": "value1"},
            "docs": "This service handles data analysis workflows for genomics.",
        }
    )
    await api.register_service(
        {
            "id": "service-2",
            "name": "example service two",
            "type": "another-type",
            "description": "This is the second test service.",
            "app_id": "another-app",
            "service_schema": {"example_key": "another_value"},
            "config": {"setting": "value2"},
            "docs": "This service focuses on image processing for medical imaging.",
        }
    )

    await api.register_service(
        {
            "id": "service-3",
            "name": "example service three",
            "type": "my-type",
            "description": "This is the third test service.",
            "app_id": "my-app",
            "service_schema": {"example_key": "yet_another_value"},
            "config": {"setting": "value3"},
            "docs": "This service specializes in natural language processing and AI chatbots.",
        }
    )

    # Test semantic search using `text_query`
    text_query = "NLP"
    services = await api.search_services(query=text_query, limit=3)
    assert isinstance(services, list)
    assert len(services) <= 3
    assert services[0]["id"].count(":") == 1 and services[0]["id"].count("/") == 1
    # The top hit should be the service with "natural language processing" in the `docs` field
    assert "natural language processing" in services[0]["docs"]
    assert services[0]["score"] < services[1]["score"]

    results = await api.search_services(query=text_query, limit=3, pagination=True)
    assert results["total"] >= 1

    embedding = np.ones(384).astype(np.float32)
    await api.register_service(
        {
            "id": "service-88",
            "name": "example service 88",
            "type": "another-type",
            "description": "This is the 88-th test service.",
            "app_id": "another-app",
            "service_schema": {"example_key": "another_value"},
            "config": {"setting": "value2", "service_embedding": embedding},
            "docs": "This service is used for performing alphafold calculations.",
        }
    )

    # Test vector query with the exact embedding
    services = await api.search_services(query=embedding, limit=3)
    assert isinstance(services, list)
    assert len(services) <= 3
    assert "service-88" in services[0]["id"]

    # Test filter-based search with fuzzy matching on the `docs` field
    filters = {"docs": "calculations*"}
    services = await api.search_services(filters=filters, limit=3)
    assert isinstance(services, list)
    assert len(services) <= 3
    assert "calculations" in services[0]["docs"]

    # Test hybrid search (text query + filters)
    filters = {"type": "my-type"}
    text_query = "genomics workflows"
    services = await api.search_services(query=text_query, filters=filters, limit=3)
    assert isinstance(services, list)
    assert all(service["type"] == "my-type" for service in services)
    # The top hit should be the service with "genomics" in the `docs` field
    assert "genomics" in services[0]["docs"].lower()

    # Test hybrid search (embedding + filters)
    filters = {"type": "my-type"}
    services = await api.search_services(
        query=np.random.rand(384), filters=filters, limit=3
    )
    assert isinstance(services, list)
    assert all(service["type"] == "my-type" for service in services)
