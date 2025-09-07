"""Test Vector Search."""

import pytest
import numpy as np
from redis import asyncio as aioredis
import pytest_asyncio
from aiobotocore.session import get_session
import time

from hypha.vectors import VectorSearchEngine
from . import (
    find_item,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(name="vector_search_engine")
async def vector_search_engine(redis_server):
    """Fixture to set up VectorSearchEngine instance."""
    redis = aioredis.from_url(redis_server)
    yield VectorSearchEngine(redis=redis)


async def test_create_and_list_collection(vector_search_engine):
    """Test creating a collection and listing collections."""
    # Create a collection
    collection_name = "test_collection"
    vector_fields = [
        {"type": "TAG", "name": "id"},
        {
            "type": "VECTOR",
            "name": "vector",
            "algorithm": "FLAT",
            "attributes": {"TYPE": "FLOAT32", "DIM": 128, "DISTANCE_METRIC": "COSINE"},
        },
    ]

    await vector_search_engine.create_collection(
        collection_name, vector_fields, overwrite=True
    )

    # Verify the collection exists
    collections = await vector_search_engine.list_collections()
    assert find_item(collections, "name", collection_name)


@pytest.mark.asyncio
async def test_add_and_search_vectors(vector_search_engine):
    """Test adding and searching vectors in a collection."""
    # Setup
    collection_name = "vector_test"
    vector_fields = [
        {"type": "TAG", "name": "id"},
        {
            "type": "VECTOR",
            "name": "vector",
            "algorithm": "FLAT",
            "attributes": {"TYPE": "FLOAT32", "DIM": 128, "DISTANCE_METRIC": "COSINE"},
        },
    ]
    await vector_search_engine.create_collection(
        collection_name, vector_fields, overwrite=True
    )

    # Add vectors
    vectors = [
        {"id": "vec1", "vector": np.random.rand(128).tolist()},
        {"id": "vec2", "vector": np.random.rand(128).tolist()},
    ]
    await vector_search_engine.add_vectors(collection_name, vectors)

    # Search for a vector
    query_vector = np.random.rand(128).tolist()
    results = await vector_search_engine.search_vectors(
        collection_name,
        query={"vector": query_vector},
        limit=1,
    )

    assert len(results) == 1
    assert "id" in results[0]


@pytest.mark.asyncio
async def test_dump_and_load_collection(vector_search_engine, minio_server):
    """Test dumping and loading a collection to/from S3."""
    # Setup
    async with get_session().create_client(
        "s3",
        endpoint_url=minio_server,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
        region_name="us-east-1",
    ) as s3_client:
        bucket = "test-bucket"
        prefix = "test-prefix"

        # Create a bucket if it doesn't exist
        try:
            await s3_client.create_bucket(Bucket=bucket)
        except s3_client.meta.client.exceptions.BucketAlreadyOwnedByYou:
            pass

        collection_name = "s3_test_collection"
        vector_fields = [
            {"type": "TAG", "name": "id"},
            {
                "type": "VECTOR",
                "name": "vector",
                "algorithm": "FLAT",
                "attributes": {
                    "TYPE": "FLOAT32",
                    "DIM": 128,
                    "DISTANCE_METRIC": "COSINE",
                },
            },
        ]
        await vector_search_engine.create_collection(
            collection_name, vector_fields, overwrite=True
        )

        # Add vectors
        vectors = [
            {"id": "vec1", "vector": np.random.rand(128).tolist()},
            {"id": "vec2", "vector": np.random.rand(128).tolist()},
        ]
        await vector_search_engine.add_vectors(collection_name, vectors)

        # Dump collection
        await vector_search_engine.dump_collection(
            collection_name, s3_client, bucket, prefix
        )

        # Verify collection was dumped
        objects = await s3_client.list_objects(Bucket=bucket, Prefix=prefix)
        assert find_item(objects["Contents"], "Key", f"{prefix}/vec1.json")
        assert find_item(objects["Contents"], "Key", f"{prefix}/vec2.json")

        # Load collection
        await vector_search_engine.load_collection(
            collection_name, s3_client, bucket, prefix, overwrite=True
        )

        # Verify vectors and index were loaded correctly
        collections = await vector_search_engine.list_collections()
        assert find_item(collections, "name", collection_name)


@pytest.mark.asyncio
async def test_delete_collection(vector_search_engine):
    """Test deleting a collection."""
    collection_name = "delete_test"
    vector_fields = [
        {"type": "TAG", "name": "id"},
        {
            "type": "VECTOR",
            "name": "vector",
            "algorithm": "FLAT",
            "attributes": {"TYPE": "FLOAT32", "DIM": 128, "DISTANCE_METRIC": "COSINE"},
        },
    ]
    await vector_search_engine.create_collection(
        collection_name, vector_fields, overwrite=True
    )

    # Add vectors
    vectors = [
        {"id": "vec1", "vector": np.random.rand(128).tolist()},
        {"id": "vec2", "vector": np.random.rand(128).tolist()},
    ]
    await vector_search_engine.add_vectors(collection_name, vectors)

    # Delete collection
    await vector_search_engine.delete_collection(collection_name)

    # Verify deletion
    collections = await vector_search_engine.list_collections()
    assert not find_item(collections, "name", collection_name)
