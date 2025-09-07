"""Test PgVector Search Engine."""

import asyncio
import pytest
import numpy as np
import random
import os
import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from aiobotocore.session import get_session
import pytest_asyncio

from hypha.pgvector import PgVectorSearchEngine
from . import (
    find_item,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(name="pg_engine")
async def pg_engine(postgres_server):
    """Fixture to set up PostgreSQL async engine."""
    # Use the test database URL from the postgres_server fixture
    from . import POSTGRES_URI
    import asyncpg
    
    # Wait for PostgreSQL to be ready
    max_retries = 20
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # Connect directly with asyncpg to check if ready
            conn = await asyncpg.connect(
                host="127.0.0.1",
                port=15432,
                user="postgres",
                password="mysecretpassword",
                database="postgres"
            )
            try:
                # Create pgvector extension if not exists
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                print("pgvector extension ready")
            except asyncpg.exceptions.DuplicateObjectError:
                print("pgvector extension already exists")
            except Exception as e:
                print(f"Warning: Could not create pgvector extension: {e}")
            finally:
                await conn.close()
            break
        except (asyncpg.exceptions.InvalidPasswordError, ConnectionRefusedError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"Waiting for PostgreSQL to be ready (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(retry_delay)
            else:
                raise Exception(f"Failed to connect to PostgreSQL after {max_retries} attempts: {e}")
    
    # Now create the SQLAlchemy engine
    # Use the imported POSTGRES_URI which has the correct port
    engine = create_async_engine(
        POSTGRES_URI,
        echo=False,
        pool_pre_ping=True,
    )
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture(name="pgvector_search_engine")
async def pgvector_search_engine(pg_engine):
    """Fixture to set up PgVectorSearchEngine instance."""
    # Create a fresh instance for each test
    search_engine = PgVectorSearchEngine(engine=pg_engine, prefix="test")
    yield search_engine
    
    # Cleanup: Drop all collections created in test
    try:
        collections = await search_engine.list_collections()
        for collection in collections:
            try:
                await search_engine.delete_collection(collection["name"])
            except Exception as e:
                print(f"Warning: Could not delete collection {collection['name']}: {e}")
    except Exception as e:
        print(f"Warning during cleanup: {e}")


async def test_create_and_list_collection(pgvector_search_engine):
    """Test creating a collection and listing collections."""
    # Create a collection
    collection_name = "test_collection_pg"
    
    await pgvector_search_engine.create_collection(
        collection_name, dimension=128, distance_metric="cosine", overwrite=True
    )

    # Verify the collection exists
    collections = await pgvector_search_engine.list_collections()
    assert find_item(collections, "name", collection_name)
    
    # Test count on empty collection
    count = await pgvector_search_engine.count(collection_name)
    assert count == 0


async def test_add_and_search_vectors(pgvector_search_engine):
    """Test adding and searching vectors in a collection."""
    # Setup
    collection_name = "vector_test_pg"
    await pgvector_search_engine.create_collection(
        collection_name, dimension=128, distance_metric="cosine", overwrite=True
    )

    # Add vectors
    vectors = [
        {
            "id": "vec1",
            "vector": np.random.rand(128).tolist(),
            "manifest": {
                "description": "First vector",
                "score": 0.8,
            }
        },
        {
            "id": "vec2",
            "vector": np.random.rand(128).tolist(),
            "manifest": {
                "description": "Second vector",
                "score": 0.6,
            }
        },
        {
            "id": "vec3",
            "vector": np.random.rand(128).tolist(),
            "manifest": {
                "description": "Third vector",
                "score": 0.9,
            }
        },
    ]
    ids = await pgvector_search_engine.add_vectors(collection_name, vectors)
    assert len(ids) == 3

    # Count vectors
    count = await pgvector_search_engine.count(collection_name)
    assert count == 3

    # Search for a vector by similarity
    query_vector = np.random.rand(128).tolist()
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        query_vector=query_vector,
        limit=2,
    )
    assert len(results) == 2
    assert "id" in results[0]
    assert "_score" in results[0]  # Similarity score

    # Search with filters
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        query_vector=query_vector,
        filters={"score": [0.7, 1.0]},  # Numeric range filter
        limit=10,
    )
    # Should return vectors with score >= 0.7
    assert all(r["score"] >= 0.7 for r in results if "score" in r)

    # Search with text filter
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"description": "First vector"},  # Exact match
        limit=10,
    )
    assert len(results) >= 1
    
    # Get specific vector
    vector = await pgvector_search_engine.get_vector(collection_name, "vec1")
    assert vector["id"] == "vec1"
    assert "description" in vector


async def test_update_vectors(pgvector_search_engine):
    """Test updating existing vectors."""
    collection_name = "update_test_pg"
    
    await pgvector_search_engine.create_collection(
        collection_name, dimension=64, distance_metric="l2", overwrite=True
    )
    
    # Add initial vectors
    vectors = [
        {"id": "update1", "vector": np.ones(64).tolist(), "manifest": {"label": "original"}},
    ]
    await pgvector_search_engine.add_vectors(collection_name, vectors)
    
    # Update the vector
    updated_vectors = [
        {"id": "update1", "vector": np.zeros(64).tolist(), "manifest": {"label": "updated"}},
    ]
    await pgvector_search_engine.add_vectors(
        collection_name, updated_vectors, update=True
    )
    
    # Verify update
    vector = await pgvector_search_engine.get_vector(collection_name, "update1")
    assert vector["label"] == "updated"
    assert np.allclose(vector["vector"], np.zeros(64))


async def test_list_vectors_pagination(pgvector_search_engine):
    """Test listing vectors with pagination."""
    collection_name = "pagination_test_pg"
    
    await pgvector_search_engine.create_collection(
        collection_name, dimension=128, distance_metric="cosine", overwrite=True
    )
    
    # Add multiple vectors
    vectors = [
        {"id": f"vec{i}", "vector": np.random.rand(128).tolist(), "manifest": {"index": i}}
        for i in range(25)
    ]
    await pgvector_search_engine.add_vectors(collection_name, vectors)
    
    # Test pagination
    page1 = await pgvector_search_engine.list_vectors(
        collection_name,
        offset=0,
        limit=10,
        pagination=True,
    )
    assert page1["total"] == 25
    assert len(page1["items"]) == 10
    assert page1["offset"] == 0
    assert page1["limit"] == 10
    
    page2 = await pgvector_search_engine.list_vectors(
        collection_name,
        offset=10,
        limit=10,
        pagination=True,
    )
    assert len(page2["items"]) == 10


async def test_remove_vectors(pgvector_search_engine):
    """Test removing vectors from a collection."""
    collection_name = "remove_test_pg"
    
    await pgvector_search_engine.create_collection(
        collection_name, dimension=128, distance_metric="cosine", overwrite=True
    )
    
    # Add vectors
    vectors = [
        {"id": f"rm{i}", "vector": np.random.rand(128).tolist(), "manifest": {"content": f"Content {i}"}}
        for i in range(5)
    ]
    await pgvector_search_engine.add_vectors(collection_name, vectors)
    
    # Remove some vectors
    await pgvector_search_engine.remove_vectors(
        collection_name, ["rm1", "rm3"]
    )
    
    # Verify removal
    count = await pgvector_search_engine.count(collection_name)
    assert count == 3
    
    remaining = await pgvector_search_engine.list_vectors(collection_name)
    remaining_ids = [v["id"] for v in remaining]
    assert "rm1" not in remaining_ids
    assert "rm3" not in remaining_ids
    assert "rm0" in remaining_ids
    assert "rm2" in remaining_ids
    assert "rm4" in remaining_ids


async def test_embedding_models(pgvector_search_engine):
    """Test text embedding generation for vector search."""
    collection_name = "embedding_test_pg"
    
    await pgvector_search_engine.create_collection(
        collection_name, dimension=384, distance_metric="cosine", overwrite=True
    )
    
    # Add vectors with text that will be embedded
    embedding_model = "fastembed:BAAI/bge-small-en-v1.5"
    
    vectors = [
        {"id": "doc1", "vector": "This is a test document", "manifest": {"text": "Test document"}},
        {"id": "doc2", "vector": "Another sample text", "manifest": {"text": "Sample text"}},
        {"id": "doc3", "vector": "Machine learning example", "manifest": {"text": "ML example"}},
    ]
    
    await pgvector_search_engine.add_vectors(
        collection_name, vectors, embedding_model=embedding_model
    )
    
    # Search with text query
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        query_vector="test document search",
        embedding_model=embedding_model,
        limit=2,
    )
    
    assert len(results) <= 2
    assert all("id" in r for r in results)


async def test_complex_filters(pgvector_search_engine):
    """Test complex filtering scenarios."""
    collection_name = "complex_filter_pg"
    
    await pgvector_search_engine.create_collection(
        collection_name, dimension=64, distance_metric="l2", overwrite=True
    )
    
    # Add diverse data
    vectors = [
        {
            "id": "prod1",
            "vector": np.random.rand(64).tolist(),
            "manifest": {
                "category": "electronics",
                "title": "Smartphone Pro",
                "price": 999.99,
            }
        },
        {
            "id": "prod2",
            "vector": np.random.rand(64).tolist(),
            "manifest": {
                "category": "electronics",
                "title": "Laptop Ultra",
                "price": 1499.99,
            }
        },
        {
            "id": "prod3",
            "vector": np.random.rand(64).tolist(),
            "manifest": {
                "category": "clothing",
                "title": "Smart Watch",
                "price": 299.99,
            }
        },
        {
            "id": "prod4",
            "vector": np.random.rand(64).tolist(),
            "manifest": {
                "category": "electronics",
                "title": "Tablet Pro",
                "price": 599.99,
            }
        },
    ]
    
    await pgvector_search_engine.add_vectors(collection_name, vectors)
    
    # Test category filter
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"category": "electronics"},
        limit=10,
    )
    assert len(results) == 3
    assert all(r["category"] == "electronics" for r in results)
    
    # Test price range filter
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"price": [500, 1000]},
        limit=10,
    )
    assert all(500 <= r["price"] <= 1000 for r in results)
    
    # Test combined filters with vector search
    query_vector = np.random.rand(64).tolist()
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        query_vector=query_vector,
        filters={
            "category": "electronics",
            "price": [0, 1000],
        },
        limit=10,
    )
    assert all(r["category"] == "electronics" for r in results)
    assert all(r["price"] <= 1000 for r in results)


async def test_delete_collection(pgvector_search_engine):
    """Test deleting a collection."""
    collection_name = "delete_test_pg"
    
    await pgvector_search_engine.create_collection(
        collection_name, dimension=128, distance_metric="cosine", overwrite=True
    )
    
    # Add vectors
    vectors = [
        {"id": "vec1", "vector": np.random.rand(128).tolist()},
        {"id": "vec2", "vector": np.random.rand(128).tolist()},
    ]
    await pgvector_search_engine.add_vectors(collection_name, vectors)
    
    # Delete collection
    await pgvector_search_engine.delete_collection(collection_name)
    
    # Verify deletion
    collections = await pgvector_search_engine.list_collections()
    assert not find_item(collections, "name", collection_name)
    
    # Try to access deleted collection should raise error
    with pytest.raises(Exception):
        await pgvector_search_engine.get_vector(collection_name, "vec1")


async def test_inner_product_distance(pgvector_search_engine):
    """Test inner product distance metric."""
    collection_name = "ip_test_pg"
    
    await pgvector_search_engine.create_collection(
        collection_name, dimension=32, distance_metric="ip", overwrite=True
    )
    
    # Add normalized vectors for inner product
    vec1 = np.random.rand(32)
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = np.random.rand(32)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    vectors = [
        {"id": "ip1", "vector": vec1.tolist()},
        {"id": "ip2", "vector": vec2.tolist()},
    ]
    await pgvector_search_engine.add_vectors(collection_name, vectors)
    
    # Search with inner product
    query = np.random.rand(32)
    query = query / np.linalg.norm(query)
    
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        query_vector=query.tolist(),
        limit=2,
    )
    
    assert len(results) == 2
    assert "_score" in results[0]


async def test_multiple_dimensions(pgvector_search_engine):
    """Test that different collections can use different dimensions."""
    # Create collection with 128 dimensions
    collection_128 = "test_128_dim"
    
    await pgvector_search_engine.create_collection(
        collection_128, dimension=128, distance_metric="cosine", overwrite=True
    )
    
    # Create collection with 384 dimensions  
    collection_384 = "test_384_dim"
    
    await pgvector_search_engine.create_collection(
        collection_384, dimension=384, distance_metric="cosine", overwrite=True
    )
    
    # Add vectors to both collections
    vectors_128 = [
        {"id": "v128_1", "vector": np.random.rand(128).tolist()},
        {"id": "v128_2", "vector": np.random.rand(128).tolist()},
    ]
    await pgvector_search_engine.add_vectors(collection_128, vectors_128)
    
    vectors_384 = [
        {"id": "v384_1", "vector": np.random.rand(384).tolist()},
        {"id": "v384_2", "vector": np.random.rand(384).tolist()},
    ]
    await pgvector_search_engine.add_vectors(collection_384, vectors_384)
    
    # Verify both collections work independently
    count_128 = await pgvector_search_engine.count(collection_128)
    count_384 = await pgvector_search_engine.count(collection_384)
    
    assert count_128 == 2
    assert count_384 == 2
    
    # Search in both collections
    results_128 = await pgvector_search_engine.search_vectors(
        collection_128,
        query_vector=np.random.rand(128).tolist(),
        limit=2,
    )
    
    results_384 = await pgvector_search_engine.search_vectors(
        collection_384,
        query_vector=np.random.rand(384).tolist(),
        limit=2,
    )
    
    assert len(results_128) == 2
    assert len(results_384) == 2
    
    # Verify metadata info
    metadata_128 = await pgvector_search_engine._get_metadata(collection_128)
    metadata_384 = await pgvector_search_engine._get_metadata(collection_384)
    
    assert metadata_128["dimension"] == 128
    assert metadata_384["dimension"] == 384


async def test_invalid_dimension(pgvector_search_engine):
    """Test that invalid dimensions are rejected."""
    collection_name = "invalid_dim"
    
    # Test dimension too large
    with pytest.raises(ValueError, match="Dimension 3000 must be between 1 and 2048"):
        await pgvector_search_engine.create_collection(
            collection_name, dimension=3000, overwrite=True
        )
    
    # Test dimension too small
    with pytest.raises(ValueError, match="Dimension 0 must be between 1 and 2048"):
        await pgvector_search_engine.create_collection(
            collection_name, dimension=0, overwrite=True
        )


async def test_dynamic_dimension_support(pgvector_search_engine):
    """Test that any reasonable dimension is supported dynamically."""
    # Test various unusual but valid dimensions
    test_dimensions = [7, 100, 512, 999, 1536]
    
    for dim in test_dimensions:
        collection_name = f"test_dim_{dim}"
        
        # Should create successfully
        await pgvector_search_engine.create_collection(
            collection_name, dimension=dim, distance_metric="cosine", overwrite=True
        )
        
        # Add a vector with the correct dimension
        vectors = [
            {"id": f"vec_{dim}", "vector": np.random.rand(dim).tolist()},
        ]
        await pgvector_search_engine.add_vectors(collection_name, vectors)
        
        # Verify it works
        count = await pgvector_search_engine.count(collection_name)
        assert count == 1
        
        # Search works
        results = await pgvector_search_engine.search_vectors(
            collection_name,
            query_vector=np.random.rand(dim).tolist(),
            limit=1,
        )
        assert len(results) == 1
        assert results[0]["id"] == f"vec_{dim}"
    
    # Verify all collections exist
    collections = await pgvector_search_engine.list_collections()
    collection_names = [c["name"] for c in collections]
    for dim in test_dimensions:
        assert f"test_dim_{dim}" in collection_names


@pytest.mark.asyncio
async def test_advanced_filter_operators(pgvector_search_engine):
    """Test advanced filter operators like $in, $like, $gt, $gte, $lt, $lte, $ne, $regex, $all."""
    collection_name = "test_advanced_filters"
    dimension = 64
    
    # Create collection
    await pgvector_search_engine.create_collection(
        collection_name, dimension=dimension, distance_metric="cosine", overwrite=True
    )
    
    # Add test vectors with various payload fields
    vectors = [
        {
            "id": "vec1",
            "vector": np.random.rand(dimension).tolist(),
            "manifest": {
                "category": "electronics",
                "price": 100.0,
                "name": "Laptop Computer",
                "tags": ["computer", "electronics", "portable"],
                "rating": 4.5,
            }
        },
        {
            "id": "vec2", 
            "vector": np.random.rand(dimension).tolist(),
            "manifest": {
                "category": "electronics",
                "price": 50.0,
                "name": "Wireless Mouse",
                "tags": ["mouse", "electronics", "wireless"],
                "rating": 4.0,
            }
        },
        {
            "id": "vec3",
            "vector": np.random.rand(dimension).tolist(),
            "manifest": {
                "category": "furniture",
                "price": 200.0,
                "name": "Office Chair",
                "tags": ["chair", "furniture", "office"],
                "rating": 4.8,
            }
        },
        {
            "id": "vec4",
            "vector": np.random.rand(dimension).tolist(),
            "manifest": {
                "category": "books",
                "price": 25.0,
                "name": "Python Programming",
                "tags": ["book", "programming", "python"],
                "rating": 4.2,
            }
        },
        {
            "id": "vec5",
            "vector": np.random.rand(dimension).tolist(),
            "manifest": {
                "category": "electronics",
                "price": 150.0,
                "name": "Tablet Device",
                "tags": ["tablet", "electronics", "touchscreen"],
                "rating": 3.9,
            }
        },
    ]
    
    await pgvector_search_engine.add_vectors(collection_name, vectors)
    
    # Test $in operator
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"category": {"$in": ["electronics", "books"]}},
        limit=10
    )
    assert len(results) == 4  # vec1, vec2, vec4, vec5
    categories = {r["category"] for r in results}
    assert categories == {"electronics", "books"}
    
    # Test $like operator
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"name": {"$like": "%Computer%"}},
        limit=10
    )
    assert len(results) == 1
    assert results[0]["id"] == "vec1"
    
    # Test pattern with wildcard at beginning
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"name": {"$like": "%Programming"}},
        limit=10
    )
    assert len(results) == 1
    assert results[0]["id"] == "vec4"
    
    # Test $gt and $lt operators
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"price": {"$gt": 50, "$lt": 150}},
        limit=10
    )
    assert len(results) == 1
    assert results[0]["id"] == "vec1"
    assert results[0]["price"] == 100.0
    
    # Test $gte and $lte operators
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"rating": {"$gte": 4.0, "$lte": 4.5}},
        limit=10
    )
    assert len(results) == 3  # vec1 (4.5), vec2 (4.0), vec4 (4.2)
    ratings = {r["rating"] for r in results}
    assert all(4.0 <= r <= 4.5 for r in ratings)
    
    # Test $ne operator
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"category": {"$ne": "electronics"}},
        limit=10
    )
    assert len(results) == 2  # vec3 (furniture), vec4 (books)
    categories = {r["category"] for r in results}
    assert "electronics" not in categories
    
    # Test $regex operator
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"name": {"$regex": "^(Wireless|Office)"}},
        limit=10
    )
    assert len(results) == 2  # vec2 (Wireless Mouse), vec3 (Office Chair)
    names = {r["name"] for r in results}
    assert all(n.startswith("Wireless") or n.startswith("Office") for n in names)
    
    # Test $all operator for array containment
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={"tags": {"$all": ["electronics", "portable"]}},
        limit=10
    )
    assert len(results) == 1
    assert results[0]["id"] == "vec1"
    
    # Test combination of operators
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        filters={
            "category": {"$in": ["electronics", "furniture"]},
            "price": {"$gte": 100},
            "rating": {"$gt": 4.0}
        },
        limit=10
    )
    assert len(results) == 2  # vec1 (electronics, 100, 4.5), vec3 (furniture, 200, 4.8)
    
    # Test with vector search and filters
    query_vector = np.random.rand(dimension).tolist()
    results = await pgvector_search_engine.search_vectors(
        collection_name,
        query_vector=query_vector,
        filters={"category": "electronics", "price": {"$lt": 100}},
        limit=10
    )
    assert len(results) == 1
    assert results[0]["id"] == "vec2"
    assert results[0]["category"] == "electronics"
    assert results[0]["price"] == 50.0
    
    # Clean up
    await pgvector_search_engine.delete_collection(collection_name)
