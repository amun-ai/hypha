"""Test Vector Search."""

import asyncio
import pytest
import numpy as np
import random
from hypha_rpc import connect_to_server
from redis import asyncio as aioredis
import pytest_asyncio
from aiobotocore.session import get_session
import time

from hypha.vectors import VectorSearchEngine
from . import (
    SERVER_URL_REDIS_1,
    find_item,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    wait_for_workspace_ready,
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


async def test_artifact_vector_collection(
    minio_server, fastapi_server_redis_1, test_user_token_9
):
    """Test vector-related functions within a vector-collection artifact."""

    # Connect to the server and set up the artifact manager
    async with connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL_REDIS_1,
            "token": test_user_token_9,
        }
    ) as api:
        await api.create_workspace(
            {
                "name": "my-vector-test-workspace",
                "description": "This is a test workspace",
                "persistent": True,
            },
            overwrite=True,
        )

    async with connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL_REDIS_1,
            "token": test_user_token_9,
            "workspace": "my-vector-test-workspace",
        }
    ) as api:
        artifact_manager = await api.get_service("public/artifact-manager")

        # Create a vector-collection artifact
        vector_collection_manifest = {
            "name": "vector-collection",
            "description": "A test vector collection",
        }
        vector_collection_config = {
            "vector_fields": [
                {
                    "type": "VECTOR",
                    "name": "vector",
                    "algorithm": "FLAT",
                    "attributes": {
                        "TYPE": "FLOAT32",
                        "DIM": 384,
                        "DISTANCE_METRIC": "COSINE",
                    },
                },
                {"type": "TEXT", "name": "text"},
                {"type": "TAG", "name": "label"},
                {"type": "NUMERIC", "name": "rand_number"},
            ],
            "embedding_models": {
                "vector": "fastembed:BAAI/bge-small-en-v1.5",
            },
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
                "text": "This is a test document.",
                "label": "doc1",
                "rand_number": random.randint(0, 10),
            },
            {
                "vector": np.random.rand(384),
                "text": "Another document.",
                "label": "doc2",
                "rand_number": random.randint(0, 10),
            },
            {
                "vector": np.random.rand(384),
                "text": "Yet another document.",
                "label": "doc3",
                "rand_number": random.randint(0, 10),
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
            query={"vector": query_vector},
            limit=2,
        )
        assert len(search_results) <= 2

        results = await artifact_manager.search_vectors(
            artifact_id=vector_collection.id,
            query={"vector": query_vector},
            limit=3,
            pagination=True,
        )
        assert results["total"] == 3

        search_results = await artifact_manager.search_vectors(
            artifact_id=vector_collection.id,
            filters={"rand_number": [-2, -1]},
            query={"vector": np.random.rand(384)},
            limit=2,
        )
        assert len(search_results) == 0

        search_results = await artifact_manager.search_vectors(
            artifact_id=vector_collection.id,
            filters={"rand_number": [0, 10]},
            query={"vector": np.random.rand(384)},
            limit=2,
        )
        assert len(search_results) > 0

        # Search for vectors by text
        vectors = [
            {"vector": "This is a test document.", "label": "doc1"},
            {"vector": "Another test document.", "label": "doc2"},
        ]
        await artifact_manager.add_vectors(
            artifact_id=vector_collection.id,
            vectors=vectors,
        )

        text_search_results = await artifact_manager.search_vectors(
            artifact_id=vector_collection.id,
            query={"vector": "test document"},
            limit=2,
        )
        assert len(text_search_results) <= 2

        # Retrieve a specific vector
        retrieved_vector = await artifact_manager.get_vector(
            artifact_id=vector_collection.id,
            id=text_search_results[0]["id"],
        )
        assert "label" in retrieved_vector

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


async def test_load_dump_vector_collections(
    minio_server, fastapi_server_redis_1, test_user_token, root_user_token
):
    """Test loading and dumping vector collections."""
    # Connect to the server and set up the artifact manager
    async with connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL_REDIS_1,
            "token": test_user_token,
        }
    ) as api:
        await api.create_workspace(
            {
                "name": "my-vector-dump-workspace",
                "description": "This is a test workspace for dumping vector collections",
                "persistent": True,
            },
            overwrite=True,
        )
    async with connect_to_server(
        {
            "name": "test deploy client",
            "server_url": SERVER_URL_REDIS_1,
            "token": test_user_token,
            "workspace": "my-vector-dump-workspace",
        }
    ) as api:
        await wait_for_workspace_ready(api, timeout=60)
        artifact_manager = await api.get_service("public/artifact-manager")

        # Create a vector-collection artifact
        vector_collection_manifest = {
            "name": "vector-collection",
            "description": "A test vector collection",
        }
        vector_collection_config = {
            "vector_fields": [
                {
                    "type": "VECTOR",
                    "name": "vector",
                    "algorithm": "FLAT",
                    "attributes": {
                        "TYPE": "FLOAT32",
                        "DIM": 384,
                        "DISTANCE_METRIC": "COSINE",
                    },
                },
                {"type": "TEXT", "name": "text"},
                {"type": "TAG", "name": "label"},
                {"type": "NUMERIC", "name": "rand_number"},
            ],
            "embedding_models": {
                "vector": "fastembed:BAAI/bge-small-en-v1.5",
            },
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
                "text": "This is a test document.",
                "label": "doc1",
                "rand_number": random.randint(0, 10),
            },
            {
                "vector": np.random.rand(384),
                "text": "Another document.",
                "label": "doc2",
                "rand_number": random.randint(0, 10),
            },
            {
                "vector": np.random.rand(384),
                "text": "Yet another document.",
                "label": "doc3",
                "rand_number": random.randint(0, 10),
            },
        ]
        await artifact_manager.add_vectors(
            artifact_id=vector_collection.id,
            vectors=vectors,
        )

        vc = await artifact_manager.read(artifact_id=vector_collection.id)
        assert vc["config"]["vector_count"] == 3
        assert (
            vc["config"]["embedding_models"]["vector"]
            == "fastembed:BAAI/bge-small-en-v1.5"
        )


async def test_vector_operation_retries(vector_search_engine):
    """Test retry mechanism for vector operations."""
    # Create a vector collection with required vector fields
    collection_name = "retry_test_collection"
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

    # Add vectors to test retry mechanism
    vectors = [
        {"id": "vec1", "vector": np.random.rand(128).tolist()},
        {"id": "vec2", "vector": np.random.rand(128).tolist()},
    ]

    # The operation should succeed with retries
    await vector_search_engine.add_vectors(collection_name, vectors)

    # Verify vectors were added successfully
    results = await vector_search_engine.search_vectors(
        collection_name, query={"vector": np.random.rand(128).tolist()}, limit=2
    )
    assert len(results) == 2

    # Clean up
    await vector_search_engine.delete_collection(collection_name)
