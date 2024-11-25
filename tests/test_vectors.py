"""Test Vector Search."""
import pytest
import numpy as np
import random
from hypha_rpc import connect_to_server


from . import SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_artifact_vector_collection(
    minio_server, fastapi_server, test_user_token
):
    """Test vector-related functions within a vector-collection artifact."""

    # Connect to the server and set up the artifact manager
    api = await connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a vector-collection artifact
    vector_collection_manifest = {
        "name": "vector-collection",
        "description": "A test vector collection",
    }
    vector_collection_config = {
        "vectors_config": {
            "size": 384,
            "distance": "Cosine",
        },
        "embedding_model": "fastembed:BAAI/bge-small-en-v1.5",
    }
    vector_collection = await artifact_manager.create(
        type="vector-collection",
        manifest=vector_collection_manifest,
        config=vector_collection_config,
    )
    # Add vectors to the collection
    vectors = [
        {
            "vector": [random.random() for _ in range(384)],
            "payload": {
                "text": "This is a test document.",
                "label": "doc1",
                "rand_number": random.randint(0, 10),
            },
        },
        {
            "vector": np.random.rand(384),
            "payload": {
                "text": "Another document.",
                "label": "doc2",
                "rand_number": random.randint(0, 10),
            },
        },
        {
            "vector": np.random.rand(384),
            "payload": {
                "text": "Yet another document.",
                "label": "doc3",
                "rand_number": random.randint(0, 10),
            },
        },
    ]
    await artifact_manager.add_vectors(
        artifact_id=vector_collection.id,
        vectors=vectors,
    )

    vc = await artifact_manager.read(artifact_id=vector_collection.id)
    assert vc["config"]["vector_count"] == 3

    # Search for vectors by query vector
    query_vector = [random.random() for _ in range(384)]
    search_results = await artifact_manager.search_vectors(
        artifact_id=vector_collection.id,
        query_vector=query_vector,
        limit=2,
    )
    assert len(search_results) <= 2

    results = await artifact_manager.search_vectors(
        artifact_id=vector_collection.id,
        query_vector=query_vector,
        limit=2,
        pagination=True,
    )
    assert results["total"] == 3

    query_filter = {
        "should": None,
        "min_should": None,
        "must": [
            {
                "key": "rand_number",
                "match": None,
                "range": {"lt": None, "gt": None, "gte": 3.0, "lte": None},
                "geo_bounding_box": None,
                "geo_radius": None,
                "geo_polygon": None,
                "values_count": None,
            }
        ],
        "must_not": None,
    }

    search_results = await artifact_manager.search_vectors(
        artifact_id=vector_collection.id,
        query_filter=query_filter,
        query_vector=np.random.rand(384),
        limit=2,
    )
    assert len(search_results) <= 2

    # Search for vectors by text
    documents = [
        {"text": "This is a test document.", "label": "doc1"},
        {"text": "Another test document.", "label": "doc2"},
    ]
    await artifact_manager.add_documents(
        artifact_id=vector_collection.id,
        documents=documents,
    )
    text_query = "test document"
    text_search_results = await artifact_manager.search_vectors(
        artifact_id=vector_collection.id,
        query_text=text_query,
        limit=2,
    )
    assert len(text_search_results) <= 2

    # Retrieve a specific vector
    retrieved_vector = await artifact_manager.get_vector(
        artifact_id=vector_collection.id,
        id=text_search_results[0]["id"],
    )
    assert retrieved_vector.id == text_search_results[0]["id"]

    # List vectors in the collection
    vector_list = await artifact_manager.list_vectors(
        artifact_id=vector_collection.id,
        offset=0,
        limit=10,
    )
    assert len(vector_list) > 0

    # Remove a vector from the collection
    await artifact_manager.remove_vectors(
        artifact_id=vector_collection.id,
        ids=[vector_list[0]["id"]],
    )
    remaining_vectors = await artifact_manager.list_vectors(
        artifact_id=vector_collection.id,
        offset=0,
        limit=10,
    )
    assert all(v["id"] != vector_list[0]["id"] for v in remaining_vectors)

    # Clean up by deleting the vector collection
    await artifact_manager.delete(artifact_id=vector_collection.id)
