"""Test dynamic vector engine configuration in artifacts."""

import pytest
import os
import random
import numpy as np
from hypha_rpc import connect_to_server
from . import SERVER_URL, SERVER_URL_SQLITE, wait_for_workspace_ready

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_dynamic_vector_engine_pgvector(
    minio_server, fastapi_server, test_user_token
):
    """Test creating vector collections with pgvector engine using PostgreSQL."""
    
    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "dynamic vector test client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    
    # Create a collection with pgvector engine - should work with PostgreSQL
    collection = await artifact_manager.create(
        type="vector-collection",
        alias="pgvector-embeddings",
        manifest={
            "name": "PgVector Embeddings",
            "description": "Vector collection using pgvector engine",
        },
        config={
            "vector_engine": "pgvector",
            "dimension": 768,
            "distance_metric": "cosine",
            "permissions": {"*": "r", "@": "rw+"},
        },
    )
    
    # Verify the collection was created
    assert collection["alias"] == "pgvector-embeddings"
    assert collection["type"] == "vector-collection"
    assert collection["config"]["vector_engine"] == "pgvector"
    assert collection["config"]["dimension"] == 768
    
    # Test adding vectors
    vectors = []
    for i in range(5):
        vector = {
            "id": f"vec_{i}",
            "vector": np.random.rand(768).tolist(),
            "metadata": {"text": f"Sample text {i}", "index": i},
        }
        vectors.append(vector)
    
    added_ids = await artifact_manager.add_vectors(
        artifact_id=collection["id"],
        vectors=vectors,
    )
    assert len(added_ids) == 5


async def test_dynamic_vector_engine_s3vector(
    minio_server, fastapi_server, test_user_token
):
    """Test creating vector collections with s3vector engine specified in config."""
    
    # Check if S3Vector is available (check zarr availability)
    try:
        import zarr
        if not hasattr(zarr, '__version__') or zarr.__version__ < '3.0':
            pytest.skip("S3Vector requires zarr>=3.1.0")
        from hypha.s3vector import S3VectorSearchEngine
    except ImportError:
        pytest.skip("S3Vector dependencies not available (requires Python>=3.11 with zarr>=3.1.0)")
    
    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "s3vector test client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    
    # Create a parent collection to test S3 config inheritance
    parent_collection = await artifact_manager.create(
        type="collection",
        alias="parent-collection",
        manifest={
            "name": "Parent Collection",
            "description": "Parent collection with S3 config",
        },
        config={
            "s3_config": {
                "endpoint_url": minio_server,
                "access_key_id": "minio",
                "secret_access_key": "test-minio-password-123",
                "region_name": "us-east-1",
                "bucket": "test-vectors",
                "prefix": "parent-vectors/",
            },
            "permissions": {"*": "r", "@": "rw+"},
        },
    )
    
    # Create a vector collection with s3vector engine under the parent
    vector_collection = await artifact_manager.create(
        type="vector-collection",
        alias="s3vector-embeddings",
        parent_id=parent_collection["id"],
        manifest={
            "name": "S3Vector Embeddings",
            "description": "Vector collection using s3vector engine with inherited S3 config",
        },
        config={
            "vector_engine": "s3vector",
            "dimension": 384,
            "distance_metric": "l2",
            # S3 config should be inherited from parent
        },
    )
    
    # Verify the collection was created with s3vector engine
    assert vector_collection["alias"] == "s3vector-embeddings"
    assert vector_collection["type"] == "vector-collection"
    assert vector_collection["config"]["vector_engine"] == "s3vector"
    assert vector_collection["config"]["dimension"] == 384
    assert vector_collection["config"]["distance_metric"] == "l2"
    
    # Test adding vectors
    vectors = []
    for i in range(3):
        vector = {
            "id": f"s3_vec_{i}",
            "vector": np.random.rand(384).tolist(),
            "metadata": {"text": f"S3 sample text {i}", "index": i},
        }
        vectors.append(vector)
    
    added_ids = await artifact_manager.add_vectors(
        artifact_id=vector_collection["id"],
        vectors=vectors,
    )
    assert len(added_ids) == 3
    
    # S3vector search should work with the search_vectors method
    # Skipping search test for now to focus on configuration


async def test_vector_engine_with_custom_s3_config(
    minio_server, fastapi_server, test_user_token
):
    """Test creating vector collection with s3vector engine and custom S3 config."""
    
    # Check if S3Vector is available (check zarr availability)
    try:
        import zarr
        if not hasattr(zarr, '__version__') or zarr.__version__ < '3.0':
            pytest.skip("S3Vector requires zarr>=3.1.0")
        from hypha.s3vector import S3VectorSearchEngine
    except ImportError:
        pytest.skip("S3Vector dependencies not available (requires Python>=3.11 with zarr>=3.1.0)")
    
    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "custom s3 test client",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    
    # Create a vector collection with s3vector engine and custom S3 config
    vector_collection = await artifact_manager.create(
        type="vector-collection",
        alias="custom-s3-vectors",
        manifest={
            "name": "Custom S3 Vectors",
            "description": "Vector collection with custom S3 configuration",
        },
        config={
            "vector_engine": "s3vector",
            "dimension": 512,
            "distance_metric": "cosine",
            "s3_config": {
                "endpoint_url": minio_server,
                "access_key_id": "minio",
                "secret_access_key": "test-minio-password-123",
                "region_name": "us-west-2",
                "bucket": "my-workspaces",
                "prefix": "embeddings/",
            },
            "permissions": {"*": "r", "@": "rw+"},
        },
    )
    
    # Verify the collection was created with custom config
    assert vector_collection["alias"] == "custom-s3-vectors"
    assert vector_collection["config"]["vector_engine"] == "s3vector"
    assert vector_collection["config"]["dimension"] == 512
    assert vector_collection["config"]["s3_config"]["bucket"] == "my-workspaces"
    assert vector_collection["config"]["s3_config"]["prefix"] == "embeddings/"
    
    # Test adding a vector
    vector = {
        "id": "custom_vec_1",
        "vector": np.random.rand(512).tolist(),
        "metadata": {"text": "Custom S3 vector", "source": "test"},
    }
    
    added_ids = await artifact_manager.add_vectors(
        artifact_id=vector_collection["id"],
        vectors=[vector],
    )
    assert len(added_ids) == 1
    assert added_ids[0] == "custom_vec_1"


async def test_default_vector_engine_fallback(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """Test that SQLite requires explicit engine specification when no engine is specified."""
    
    # Set up connection and artifact manager
    api = await connect_to_server(
        {
            "name": "default engine test client",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    
    # Create a vector collection without specifying engine - should fail with SQLite
    with pytest.raises(Exception) as exc_info:
        await artifact_manager.create(
            type="vector-collection",
            alias="default-engine-vectors",
            manifest={
                "name": "Default Engine Vectors",
                "description": "Vector collection using default engine",
            },
            config={
                # No vector_engine specified, should fail with SQLite
                "dimension": 256,
                "distance_metric": "cosine",
                "permissions": {"*": "r", "@": "rw+"},
            },
        )
    
    # Verify we get the expected error about needing to specify engine for SQLite
    assert "SQLite" in str(exc_info.value)
    assert "vector_engine" in str(exc_info.value)