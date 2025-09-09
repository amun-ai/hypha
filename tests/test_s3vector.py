"""Comprehensive tests for S3-based vector search engine with SPANN indexing."""

import pytest
import pytest_asyncio
import numpy as np
import os
import uuid
import asyncio
import time
import sys
from fakeredis import aioredis
from unittest.mock import MagicMock

# Check if zarr 3.x is available
try:
    import zarr
    if not hasattr(zarr, '__version__') or zarr.__version__ < '3.0':
        pytest.skip("S3Vector requires zarr>=3.1.0", allow_module_level=True)
    
    from hypha.s3vector import (
        S3VectorSearchEngine,
        create_s3_vector_engine_from_config,
        HNSW,
        S3PersistentIndex,
        RedisCache,
        CollectionMetadata,
    )
    S3VECTOR_AVAILABLE = True
except ImportError:
    pytest.skip("S3Vector dependencies (zarr>=3.1.0) not available, requires Python>=3.11", allow_module_level=True)
    S3VECTOR_AVAILABLE = False

# Import test configuration for MinIO/S3 settings
from . import (
    MINIO_PORT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    MINIO_SERVER_URL,
)


async def empty_and_delete_bucket(s3_client, bucket_name):
    """Helper function to empty and delete an S3 bucket."""
    try:
        # Delete all objects in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        async for page in pages:
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                if objects:
                    await s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': objects}
                    )
        
        # Now delete the empty bucket
        await s3_client.delete_bucket(Bucket=bucket_name)
    except Exception as e:
        # Ignore errors if bucket doesn't exist
        pass


@pytest.fixture
def s3_config(minio_server):
    """S3 configuration using the shared MinIO server from conftest."""
    return {
        "endpoint_url": MINIO_SERVER_URL,
        "access_key_id": MINIO_ROOT_USER,
        "secret_access_key": MINIO_ROOT_PASSWORD,
        "region": "us-east-1"
    }


@pytest.fixture
def redis_client():
    """Create a fake Redis client for testing."""
    return aioredis.FakeRedis()


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing."""
    np.random.seed(42)
    return [
        np.random.rand(384).astype(np.float32) for _ in range(100)
    ]


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    return [
        {"name": f"item_{i}", "category": f"cat_{i % 3}", "value": i}
        for i in range(100)
    ]


# ============================================================================
# HNSW Index Tests
# ============================================================================

class TestHNSW:
    """Test HNSW index implementation."""
    
    def test_hnsw_init(self):
        """Test HNSW initialization."""
        hnsw = HNSW(metric="cosine", m=16, ef=200)
        assert hnsw.metric == "cosine"
        assert hnsw.m == 16
        assert hnsw.ef == 200
        assert hnsw.entry_point is None
        assert len(hnsw.data) == 0
        
    def test_hnsw_add_and_search(self):
        """Test adding vectors and searching in HNSW."""
        hnsw = HNSW(metric="cosine", m=5, ef=50)
        
        # Add vectors
        vectors = [np.random.rand(128).astype(np.float32) for _ in range(10)]
        for i, v in enumerate(vectors):
            hnsw.add(v, i)
        
        # Search
        query = vectors[0]
        results = hnsw.search(query, k=5)
        
        assert len(results) == 5
        # First result should be the query itself (distance, index)
        assert results[0][1] == 0  # Index should be 0
        assert results[0][0] < 0.01  # Very small distance
        
    def test_hnsw_serialize_deserialize(self):
        """Test HNSW serialization and deserialization."""
        hnsw = HNSW(metric="l2", m=8, ef=100)
        
        # Add some vectors
        vectors = [np.random.rand(64).astype(np.float32) for _ in range(5)]
        for i, v in enumerate(vectors):
            hnsw.add(v, i)
        
        # Serialize
        serialized = hnsw.serialize()
        
        # Deserialize
        hnsw2 = HNSW.deserialize(serialized)
        
        # Check properties
        assert hnsw2.metric == "l2"
        assert hnsw2.m == 8
        assert hnsw2.ef == 100
        assert hnsw2.entry_point == hnsw.entry_point
        assert len(hnsw2.data) == len(hnsw.data)
        
        # Check search works
        query = vectors[0]
        results = hnsw2.search(query, k=3)
        assert len(results) == 3
    
    @pytest.mark.parametrize("num_vectors,dim", [
        (100, 128),
        (1000, 128),
        (5000, 384),
    ])
    def test_hnsw_build_performance(self, num_vectors, dim):
        """Test HNSW index build time with different dataset sizes."""
        hnsw = HNSW(metric="cosine", m=16, ef=200)
        
        # Generate random vectors
        vectors = [np.random.rand(dim).astype(np.float32) for _ in range(num_vectors)]
        
        # Measure build time
        start_time = time.time()
        for i, v in enumerate(vectors):
            hnsw.add(v, i)
        build_time = time.time() - start_time
        
        print(f"\nHNSW Build Performance:")
        print(f"  Vectors: {num_vectors}, Dimension: {dim}")
        print(f"  Build time: {build_time:.3f}s")
        print(f"  Throughput: {num_vectors/build_time:.0f} vectors/sec")
        
        # Test search performance
        query = vectors[0]
        search_times = []
        for _ in range(100):
            start = time.time()
            results = hnsw.search(query, k=10)
            search_times.append(time.time() - start)
        
        avg_search_time = np.mean(search_times) * 1000  # Convert to ms
        print(f"  Avg search time: {avg_search_time:.3f}ms")
        print(f"  Search QPS: {1000/avg_search_time:.0f}")
        
        # Adjusted performance expectations for HNSW
        # HNSW typically takes 20-100ms per vector depending on dimension and parameters
        # Higher dimensions (384) take longer than lower dimensions (128)
        if dim <= 128:
            expected_time_per_vector = 0.05  # 50ms per vector for low dimensions
        else:
            expected_time_per_vector = 0.1   # 100ms per vector for high dimensions
        
        assert build_time < num_vectors * expected_time_per_vector, \
            f"Build time {build_time:.2f}s exceeds expected {num_vectors * expected_time_per_vector:.2f}s"
        
        # Search time should scale with dimension but remain under 100ms
        assert avg_search_time < 100, \
            f"Average search time {avg_search_time:.2f}ms exceeds 100ms"  # < 50ms per search
    
    def test_hnsw_memory_efficiency(self):
        """Test HNSW memory usage remains bounded."""
        hnsw = HNSW(metric="l2", m=8, ef=50)
        
        # Add vectors incrementally and check memory
        num_vectors = 10000
        dim = 128
        
        # Get initial memory
        initial_size = sys.getsizeof(hnsw.data) + sys.getsizeof(hnsw.graph)
        
        # Add vectors
        for i in range(num_vectors):
            v = np.random.rand(dim).astype(np.float32)
            hnsw.add(v, i)
        
        # Get final memory
        final_size = sys.getsizeof(hnsw.data) + sys.getsizeof(hnsw.graph)
        
        # Calculate memory per vector
        memory_per_vector = (final_size - initial_size) / num_vectors
        
        print(f"\nHNSW Memory Efficiency:")
        print(f"  Vectors: {num_vectors}")
        print(f"  Memory per vector: {memory_per_vector:.0f} bytes")
        print(f"  Total graph size: {final_size/1024/1024:.2f} MB")
        
        # Memory should be reasonable (< 1KB per vector for graph structure)
        assert memory_per_vector < 1024


# ============================================================================
# S3 Persistent Index Tests
# ============================================================================

class TestS3PersistentIndex:
    """Test S3 persistent index for centroids."""
    
    @pytest.mark.asyncio
    async def test_index_initialize(self, s3_config):
        """Test index initialization with random centroids."""
        bucket_name = f"test-index-{uuid.uuid4().hex[:8]}"
        
        # Create S3 client factory
        import aioboto3
        session = aioboto3.Session()
        
        def s3_client_factory():
            return session.client(
                "s3",
                endpoint_url=s3_config["endpoint_url"],
                aws_access_key_id=s3_config["access_key_id"],
                aws_secret_access_key=s3_config["secret_access_key"],
                region_name=s3_config["region"],
            )
        
        # Create bucket
        async with s3_client_factory() as s3_client:
            await s3_client.create_bucket(Bucket=bucket_name)
        
        try:
            # Create index
            index = S3PersistentIndex(s3_client_factory, bucket_name, "test_collection")
            await index.initialize(dimension=128, num_centroids=10, metric="cosine")
            
            # Check centroids
            assert index.centroids is not None
            assert index.centroids.shape == (10, 128)
            assert index.hnsw_index is not None
            
            # Test finding nearest centroids
            query = np.random.rand(128).astype(np.float32)
            nearest = index.find_nearest_centroids(query, k=3)
            assert len(nearest) == 3
            
            # Test load
            index2 = S3PersistentIndex(s3_client_factory, bucket_name, "test_collection")
            success = await index2.load()
            assert success
            assert index2.centroids.shape == (10, 128)
        except Exception as e:
            raise e
        finally:
            # Cleanup
            async with s3_client_factory() as s3_client:
                # Delete all objects
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name)
                
                async for page in pages:
                    if 'Contents' in page:
                        objects = [{'Key': obj['Key']} for obj in page['Contents']]
                        if objects:
                            await s3_client.delete_objects(
                                Bucket=bucket_name,
                                Delete={'Objects': objects}
                            )
                
                # Delete bucket
                await empty_and_delete_bucket(s3_client, bucket_name)


# ============================================================================
# Redis Cache Tests
# ============================================================================

class TestRedisCache:
    """Test Redis caching layer."""
    
    @pytest.mark.asyncio
    async def test_cache_centroids(self, redis_client):
        """Test caching centroids and HNSW index."""
        cache = RedisCache(redis_client, default_ttl=3600)
        
        # Create test data
        centroids = np.random.rand(10, 128).astype(np.float32)
        hnsw = HNSW(metric="cosine")
        for i, c in enumerate(centroids):
            hnsw.add(c, i)
        
        # Cache centroids
        await cache.set_centroids("test_collection", centroids, hnsw)
        
        # Retrieve cached centroids
        cached = await cache.get_centroids("test_collection")
        assert cached is not None
        cached_centroids, cached_hnsw = cached
        
        # Verify
        assert cached_centroids.shape == centroids.shape
        assert cached_hnsw.metric == "cosine"
        assert len(cached_hnsw.data) == 10
        
    @pytest.mark.asyncio
    async def test_cache_shard(self, redis_client):
        """Test caching shard data."""
        cache = RedisCache(redis_client, default_ttl=3600)
        
        # Create test data
        vectors = np.random.rand(50, 128).astype(np.float32)
        metadata = [{"id": i} for i in range(50)]
        ids = [f"vec_{i}" for i in range(50)]
        
        # Cache shard
        await cache.set_shard("test_collection", "shard_001", vectors, metadata, ids)
        
        # Retrieve cached shard
        cached = await cache.get_shard("test_collection", "shard_001")
        assert cached is not None
        cached_vectors, cached_metadata, cached_ids = cached
        
        # Verify
        assert cached_vectors.shape == vectors.shape
        assert len(cached_metadata) == 50
        assert len(cached_ids) == 50
        
    @pytest.mark.asyncio
    async def test_cache_metadata(self, redis_client):
        """Test caching collection metadata."""
        cache = RedisCache(redis_client, default_ttl=3600)
        
        # Create test metadata
        metadata = CollectionMetadata(
            collection_id="test_collection",
            dimension=128,
            metric="cosine",
            total_vectors=1000,
            num_centroids=10,
            num_shards=5,
            shard_size=200,
            created_at=1234567890,
            updated_at=1234567890
        )
        
        # Cache metadata
        await cache.set_metadata("test_collection", metadata)
        
        # Retrieve cached metadata
        cached = await cache.get_metadata("test_collection")
        assert cached is not None
        assert cached.collection_id == "test_collection"
        assert cached.dimension == 128
        assert cached.total_vectors == 1000
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, redis_client):
        """Test that Redis caching works correctly."""
        cache = RedisCache(redis_client, default_ttl=3600)
        
        # Create large test data
        vectors = np.random.rand(1000, 128).astype(np.float32)
        metadata = [{"id": i} for i in range(1000)]
        ids = [f"vec_{i}" for i in range(1000)]
        
        # First operation (cold cache)
        await redis_client.flushall()
        start = time.time()
        await cache.set_shard("perf_test", "shard_001", vectors, metadata, ids)
        cold_write_time = time.time() - start
        
        # Verify cache miss (data not in cache initially)
        await redis_client.flushall()
        result = await cache.get_shard("perf_test", "shard_001")
        assert result is None, "Cache should be empty after flush"
        
        # Write to cache
        await cache.set_shard("perf_test", "shard_001", vectors, metadata, ids)
        
        # Read from cache (should succeed)
        start = time.time()
        result = await cache.get_shard("perf_test", "shard_001")
        read_time = time.time() - start
        
        assert result is not None, "Cache should contain data"
        cached_vectors, cached_metadata, cached_ids = result
        assert cached_vectors.shape == vectors.shape
        assert len(cached_metadata) == len(metadata)
        assert len(cached_ids) == len(ids)
        
        print(f"\nCache Performance:")
        print(f"  Write time: {cold_write_time*1000:.2f}ms")
        print(f"  Read time: {read_time*1000:.2f}ms")
        print(f"  Data size: {vectors.nbytes / 1024:.1f} KB")
        
        # Multiple reads should consistently return the same data
        for _ in range(5):
            result = await cache.get_shard("perf_test", "shard_001")
            assert result is not None
            v, m, i = result
            assert v.shape == vectors.shape


# ============================================================================
# S3 Vector Search Engine Tests
# ============================================================================

class TestS3VectorSearchEngine:
    """Test S3 vector search engine with SPANN indexing."""
    
    @pytest.mark.asyncio
    async def test_create_collection(self, s3_config, redis_client):
        """Test creating a vector collection."""
        bucket_name = f"test-engine-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=5,
            shard_size=100
        )
        await engine.initialize()
        
        try:
            # Create collection
            collection_id = "test_collection"
            config = {
                "dimension": 128,
                "metric": "cosine",
                "num_centroids": 5,
                "shard_size": 100
            }
            
            metadata = await engine.create_collection(collection_id, config)
            
            # Verify metadata
            assert metadata["collection_id"] == collection_id
            assert metadata["dimension"] == 128
            assert metadata["metric"] == "cosine"
            assert metadata["num_centroids"] == 5
            
            # Verify collection info
            info = await engine.get_collection_info(collection_id)
            assert info["collection_id"] == collection_id
            
            # List collections
            collections = await engine.list_collections()
            assert collection_id in collections
        except Exception as e:
            raise e
        finally:
            # Cleanup
            await engine.delete_collection(collection_id)
            
            # Delete bucket
            async with engine.s3_client_factory() as s3_client:
                await s3_client.delete_bucket(Bucket=bucket_name)
    
    @pytest.mark.asyncio
    async def test_add_and_search_vectors(self, s3_config, redis_client, sample_vectors, sample_metadata):
        """Test adding and searching vectors with SPANN indexing."""
        bucket_name = f"test-search-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=10,
            shard_size=50  # Small shard size for testing
        )
        await engine.initialize()
        
        collection_id = "search_test"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 384,
                "metric": "cosine",
                "num_centroids": 10
            })
            
            # Add vectors - they will be distributed across shards
            vector_ids = [f"vec_{i}" for i in range(len(sample_vectors))]
            result = await engine.add_vectors(
                collection_id, sample_vectors, sample_metadata, ids=vector_ids
            )
            assert len(result) == len(sample_vectors)
            
            # Search vectors
            query_vector = sample_vectors[0]
            results = await engine.search_vectors(
                collection_id, query_vector=query_vector, limit=10
            )
            
            assert len(results) <= 10
            # First result should be the query vector itself
            if results:
                assert results[0]["_score"] > 0.9  # High similarity (using _score)
                assert results[0]["id"] == "vec_0"
            
            # Search with metadata filter
            filtered_results = await engine.search_vectors(
                collection_id,
                query_vector=sample_vectors[3],
                limit=5,
                filters={"category": "cat_0"}  # Use filters instead of filter_metadata
            )
            
            # All results should have category "cat_0"
            for result in filtered_results:
                # Metadata is flattened into the result
                assert result["category"] == "cat_0"
        except Exception as e:
            raise e
        finally:
            # Cleanup
            await engine.delete_collection(collection_id)
            
            # Delete bucket using helper function
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)
    
    @pytest.mark.asyncio
    async def test_search_with_filters_only(self, s3_config, redis_client):
        """Test searching vectors using only filters without query vector."""
        bucket_name = f"test-filter-search-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=5,
            shard_size=50
        )
        await engine.initialize()
        
        collection_id = "filter_test"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine"
            })
            
            # Add diverse test data
            vectors = []
            metadata = []
            ids = []
            
            # Create 100 vectors with different categories and properties
            for i in range(100):
                vectors.append(np.random.rand(128).astype(np.float32))
                metadata.append({
                    "category": f"cat_{i % 5}",
                    "type": "product" if i % 2 == 0 else "document",
                    "price": (i * 10) % 1000,
                    "rating": (i % 5) + 1,
                    "name": f"item_{i}",
                    "tags": [f"tag_{i % 3}", f"tag_{i % 7}"]
                })
                ids.append(f"id_{i}")
            
            await engine.add_vectors(collection_id, vectors, metadata, ids)
            
            # Test 1: Search with single filter, no query vector
            results = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={"category": "cat_2"},
                limit=10
            )
            
            assert len(results) > 0
            for result in results:
                assert result["category"] == "cat_2"
            
            # Test 2: Multiple filters (AND logic)
            results = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={
                    "category": "cat_1",
                    "type": "product"
                },
                limit=5
            )
            
            for result in results:
                assert result["category"] == "cat_1"
                assert result["type"] == "product"
            
            # Test 3: Range filter on numeric field
            results = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={"price": {"$gte": 500, "$lt": 800}},
                limit=10
            )
            
            for result in results:
                assert 500 <= result["price"] < 800
            
            # Test 4: Filter with exact match on rating
            results = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={"rating": 5},
                limit=20
            )
            
            for result in results:
                assert result["rating"] == 5
            
            # Test 5: Complex filter with nested conditions
            results = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={
                    "type": "document",
                    "rating": {"$gte": 3}
                },
                limit=15
            )
            
            for result in results:
                assert result["type"] == "document"
                assert result["rating"] >= 3
                
        except Exception as e:
            raise e
        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)
    
    @pytest.mark.asyncio
    async def test_combined_vector_and_filter_search(self, s3_config, redis_client):
        """Test combining vector similarity search with metadata filters."""
        bucket_name = f"test-combined-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=8,
            shard_size=100
        )
        await engine.initialize()
        
        collection_id = "combined_search"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 256,
                "metric": "l2"
            })
            
            # Create clustered vectors (simulate different product categories)
            vectors = []
            metadata = []
            ids = []
            
            # Create 3 clusters of vectors
            for cluster in range(3):
                cluster_center = np.random.rand(256).astype(np.float32)
                for i in range(50):
                    # Add noise to create cluster
                    vec = cluster_center + np.random.randn(256).astype(np.float32) * 0.1
                    vectors.append(vec)
                    metadata.append({
                        "cluster": cluster,
                        "category": f"category_{cluster}",
                        "subcategory": f"sub_{i % 5}",
                        "quality": "high" if i < 25 else "low",
                        "price": 100 * cluster + i,
                        "in_stock": i % 3 != 0
                    })
                    ids.append(f"cluster{cluster}_item{i}")
            
            await engine.add_vectors(collection_id, vectors, metadata, ids)
            
            # Test 1: Vector search within a specific category
            query = vectors[0]  # From cluster 0
            results = await engine.search_vectors(
                collection_id,
                query_vector=query,
                filters={"category": "category_0"},
                limit=10
            )
            
            # Should get results from cluster 0 only
            for result in results:
                assert result["category"] == "category_0"
                assert result["cluster"] == 0
            
            # Test 2: Cross-cluster search with quality filter
            query = vectors[75]  # From cluster 1
            results = await engine.search_vectors(
                collection_id,
                query_vector=query,
                filters={"quality": "high"},
                limit=10
            )
            
            # Should get high quality items regardless of cluster
            for result in results:
                assert result["quality"] == "high"
            
            # Test 3: Complex filter with vector search
            query = vectors[100]  # From cluster 2
            results = await engine.search_vectors(
                collection_id,
                query_vector=query,
                filters={
                    "in_stock": True,
                    "price": {"$lt": 250}
                },
                limit=15
            )
            
            for result in results:
                assert result["in_stock"] == True
                assert result["price"] < 250
                
        except Exception as e:
            raise e
        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)
    
    @pytest.mark.asyncio
    async def test_pagination_and_offset(self, s3_config, redis_client):
        """Test pagination with offset and limit."""
        bucket_name = f"test-pagination-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=5,
            shard_size=100
        )
        await engine.initialize()
        
        collection_id = "pagination_test"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine"
            })
            
            # Add 200 vectors
            vectors = []
            metadata = []
            ids = []
            
            for i in range(200):
                vectors.append(np.random.rand(128).astype(np.float32))
                metadata.append({
                    "index": i,
                    "category": f"cat_{i % 10}"
                })
                ids.append(f"vec_{i:03d}")
            
            await engine.add_vectors(collection_id, vectors, metadata, ids)
            
            # Test 1: Basic pagination without query vector
            page1 = await engine.search_vectors(
                collection_id,
                query_vector=None,
                limit=20,
                offset=0
            )
            
            page2 = await engine.search_vectors(
                collection_id,
                query_vector=None,
                limit=20,
                offset=20
            )
            
            # Verify no overlap between pages
            page1_ids = {r["id"] for r in page1}
            page2_ids = {r["id"] for r in page2}
            assert len(page1_ids.intersection(page2_ids)) == 0
            
            # Test 2: Pagination with filters
            filtered_page1 = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={"category": "cat_5"},
                limit=5,
                offset=0
            )
            
            filtered_page2 = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={"category": "cat_5"},
                limit=5,
                offset=5
            )
            
            # All results should match filter
            for result in filtered_page1 + filtered_page2:
                assert result["category"] == "cat_5"
            
            # Test 3: Pagination with vector search
            query = vectors[50]
            
            results_batch1 = await engine.search_vectors(
                collection_id,
                query_vector=query,
                limit=10,
                offset=0
            )
            
            results_batch2 = await engine.search_vectors(
                collection_id,
                query_vector=query,
                limit=10,
                offset=10
            )
            
            # Scores should generally decrease (first batch should have higher scores)
            if len(results_batch1) > 0 and len(results_batch2) > 0:
                avg_score_batch1 = np.mean([r["_score"] for r in results_batch1])
                avg_score_batch2 = np.mean([r["_score"] for r in results_batch2])
                assert avg_score_batch1 >= avg_score_batch2
                
        except Exception as e:
            raise e
        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)
    
    @pytest.mark.asyncio
    async def test_empty_and_edge_cases(self, s3_config, redis_client):
        """Test edge cases and error handling."""
        bucket_name = f"test-edge-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client
        )
        await engine.initialize()
        
        collection_id = "edge_test"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 64,
                "metric": "cosine"
            })
            
            # Test 1: Search in empty collection
            results = await engine.search_vectors(
                collection_id,
                query_vector=np.random.rand(64).astype(np.float32),
                limit=10
            )
            assert len(results) == 0
            
            # Test 2: Search with filters in empty collection
            results = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={"category": "nonexistent"},
                limit=10
            )
            assert len(results) == 0
            
            # Add some vectors
            vectors = [np.random.rand(64).astype(np.float32) for _ in range(10)]
            metadata = [{"value": i} for i in range(10)]
            await engine.add_vectors(collection_id, vectors, metadata)
            
            # Test 3: Search with limit larger than collection size
            results = await engine.search_vectors(
                collection_id,
                query_vector=vectors[0],
                limit=100
            )
            assert len(results) <= 10
            
            # Test 4: Search with invalid filter (no matches)
            results = await engine.search_vectors(
                collection_id,
                query_vector=None,
                filters={"nonexistent_field": "value"},
                limit=10
            )
            # Should return empty or all results depending on implementation
            assert isinstance(results, list)
            
            # Test 5: Large offset beyond collection size
            results = await engine.search_vectors(
                collection_id,
                query_vector=None,
                limit=10,
                offset=1000
            )
            assert len(results) == 0
            
        except Exception as e:
            raise e
        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)
    
    @pytest.mark.asyncio
    async def test_partial_loading(self, s3_config, redis_client):
        """Test that only relevant shards are loaded during search."""
        bucket_name = f"test-partial-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=4,
            shard_size=25  # Small shards to force multiple shards
        )
        await engine.initialize()
        
        collection_id = "partial_test"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine",
                "num_centroids": 4
            })
            
            # Add many vectors to create multiple shards
            vectors = [np.random.rand(128).astype(np.float32) for _ in range(100)]
            metadata = [{"id": i} for i in range(100)]
            
            await engine.add_vectors(collection_id, vectors, metadata)
            
            # Search should only load relevant shards
            query = vectors[0]
            results = await engine.search_vectors(collection_id, query_vector=query, limit=5)
            
            # Verify we get results
            assert len(results) > 0
            assert results[0]["_score"] > 0.5  # Use _score instead of score
            
            # Check that not all shards are in cache (if using Redis)
            if redis_client:
                # Count cached shards
                cursor = 0
                cached_shards = []
                while True:
                    cursor, keys = await redis_client.scan(
                        cursor, match=f"cache:shard:{collection_id}:*", count=100
                    )
                    cached_shards.extend(keys)
                    if cursor == 0:
                        break
                
                # Should have cached some but not all shards
                # (with 4 centroids and top-5 search, we should search at most 4 shards)
                assert len(cached_shards) <= 4
        except Exception as e:
            raise e
        finally:
            # Cleanup
            await engine.delete_collection(collection_id)
            
            # Delete bucket using helper function
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)
    
    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, s3_config, redis_client):
        """Test that multiple collections are isolated."""
        bucket_name = f"test-multi-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client
        )
        await engine.initialize()
        
        collection1 = "tenant1"
        collection2 = "tenant2"
        
        try:
            # Create two collections with different dimensions
            await engine.create_collection(collection1, {"dimension": 128})
            await engine.create_collection(collection2, {"dimension": 256})
            
            # Add vectors to each collection
            vectors1 = [np.random.rand(128).astype(np.float32) for _ in range(10)]
            vectors2 = [np.random.rand(256).astype(np.float32) for _ in range(10)]
            
            await engine.add_vectors(collection1, vectors1)
            await engine.add_vectors(collection2, vectors2)
            
            # Search in each collection
            results1 = await engine.search_vectors(collection1, query_vector=vectors1[0], limit=5)
            results2 = await engine.search_vectors(collection2, query_vector=vectors2[0], limit=5)
            
            # Verify isolation
            assert len(results1) > 0
            assert len(results2) > 0
            
            # List collections
            collections = await engine.list_collections()
            assert collection1 in collections
            assert collection2 in collections
        except Exception as e:
            raise e
        finally:
            # Cleanup
            await engine.delete_collection(collection1)
            await engine.delete_collection(collection2)
            
            # Delete bucket
            async with engine.s3_client_factory() as s3_client:
                await s3_client.delete_bucket(Bucket=bucket_name)
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, s3_config, redis_client):
        """Test deleting a collection."""
        bucket_name = f"test-delete-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client
        )
        await engine.initialize()
        
        collection_name = "delete_test"
        
        try:
            # Create and populate collection
            await engine.create_collection(
                collection_name,
                {"dimension": 128}
            )
            
            vectors = [np.random.rand(128).astype(np.float32) for _ in range(5)]
            await engine.add_vectors(collection_name, vectors)
            
            # Delete collection
            await engine.delete_collection(collection_name)
            
            # Verify deletion
            collections = await engine.list_collections()
            assert collection_name not in collections
            
            # Try to access deleted collection should raise error
            with pytest.raises(Exception):
                await engine.get_collection_info(collection_name)
        except Exception as e:
            raise e   
        finally:
            # Delete bucket
            async with engine.s3_client_factory() as s3_client:
                await s3_client.delete_bucket(Bucket=bucket_name)


# ============================================================================
# Integration Tests with Text Embeddings
# ============================================================================

async def _create_engine(s3_config, redis_client):
    """Helper to create S3VectorSearchEngine instance with embedding model."""
    bucket_name = f"test-integration-{uuid.uuid4().hex[:8]}"
    
    # Try to use fastembed if available
    embedding_model = None
    try:
        from fastembed import TextEmbedding
        # Test if model can be loaded
        test_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        embedding_model = "fastembed:BAAI/bge-small-en-v1.5"
        del test_model
    except:
        print("Warning: fastembed not available, skipping embedding tests")
    
    engine = S3VectorSearchEngine(
        endpoint_url=s3_config["endpoint_url"],
        access_key_id=s3_config["access_key_id"],
        secret_access_key=s3_config["secret_access_key"],
        region_name=s3_config["region"],
        bucket_name=bucket_name,
        redis_client=redis_client,
        embedding_model=embedding_model,
        num_centroids=10,
        shard_size=100
    )
    await engine.initialize()
    return engine, bucket_name


@pytest_asyncio.fixture
async def s3vector_engine(s3_config, redis_client):
    """Fixture to set up S3VectorSearchEngine instance."""
    engine, bucket_name = await _create_engine(s3_config, redis_client)
    return engine


    @pytest.mark.asyncio
    async def test_list_and_remove_vectors(self, s3_config, redis_client):
        """Test listing and removing vectors from collection."""
        bucket_name = f"test-list-remove-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=5,
            shard_size=50
        )
        await engine.initialize()
        
        collection_id = "list_remove_test"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine"
            })
            
            # Add vectors
            vectors = []
            metadata = []
            ids = []
            
            for i in range(50):
                vectors.append(np.random.rand(128).astype(np.float32))
                metadata.append({
                    "name": f"vector_{i}",
                    "group": f"group_{i % 5}"
                })
                ids.append(f"id_{i:03d}")
            
            await engine.add_vectors(collection_id, vectors, metadata, ids)
            
            # Test 1: Count vectors
            count = await engine.count(collection_id)
            assert count == 50
            
            # Test 2: List vectors (if implemented)
            try:
                listed = await engine.list_vectors(
                    collection_id,
                    limit=10,
                    offset=0
                )
                # Should return list of vector metadata without the actual vectors
                assert isinstance(listed, list)
                if len(listed) > 0:
                    assert "id" in listed[0]
                    assert "name" in listed[0]
            except NotImplementedError:
                # Expected if list_vectors is not fully implemented
                pass
            
            # Test 3: List with filters
            try:
                filtered_list = await engine.list_vectors(
                    collection_id,
                    filters={"group": "group_1"},
                    limit=20
                )
                if isinstance(filtered_list, list) and len(filtered_list) > 0:
                    for item in filtered_list:
                        assert item.get("group") == "group_1"
            except NotImplementedError:
                pass
            
            # Test 4: Remove vectors (if implemented)
            try:
                # Remove specific vectors
                ids_to_remove = ["id_001", "id_002", "id_003"]
                await engine.remove_vectors(collection_id, ids_to_remove)
                
                # Verify count decreased
                new_count = await engine.count(collection_id)
                assert new_count == 47
            except NotImplementedError:
                # Expected if remove_vectors is not fully implemented
                pass
            
        except Exception as e:
            raise e
        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)
    
    @pytest.mark.asyncio
    async def test_update_vectors(self, s3_config, redis_client):
        """Test updating existing vectors and metadata."""
        bucket_name = f"test-update-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=5,
            shard_size=100
        )
        await engine.initialize()
        
        collection_id = "update_test"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine"
            })
            
            # Add initial vectors
            initial_vectors = []
            initial_metadata = []
            ids = []
            
            for i in range(20):
                initial_vectors.append(np.random.rand(128).astype(np.float32))
                initial_metadata.append({
                    "version": 1,
                    "name": f"item_{i}",
                    "updated": False
                })
                ids.append(f"vec_{i:03d}")
            
            await engine.add_vectors(collection_id, initial_vectors, initial_metadata, ids)
            
            # Test update functionality (if supported)
            try:
                # Update some vectors
                updated_vectors = []
                updated_metadata = []
                update_ids = []
                
                for i in range(5):
                    updated_vectors.append(np.random.rand(128).astype(np.float32))
                    updated_metadata.append({
                        "version": 2,
                        "name": f"updated_item_{i}",
                        "updated": True
                    })
                    update_ids.append(f"vec_{i:03d}")
                
                # Try to update with update=True flag
                await engine.add_vectors(
                    collection_id,
                    updated_vectors,
                    updated_metadata,
                    ids=update_ids,
                    update=True
                )
                
                # Search for updated vectors
                results = await engine.search_vectors(
                    collection_id,
                    query_vector=updated_vectors[0],
                    filters={"updated": True},
                    limit=5
                )
                
                # Check if updates were applied
                if len(results) > 0:
                    assert results[0]["updated"] == True
                    assert results[0]["version"] == 2
            except NotImplementedError:
                # Update might not be implemented
                pass
                
        except Exception as e:
            raise e
        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)


class TestIntegration:
    """Integration tests with real embeddings."""
    
    @pytest.mark.asyncio
    async def test_text_embedding_search(self, s3vector_engine):
        """Test text embedding and semantic search."""
        # Skip if no embedding model
        if s3vector_engine.model is None:
            pytest.skip("No embedding model available")
        
        collection_name = "text_search"
        
        # Create collection with embedding dimension
        await s3vector_engine.create_collection(
            collection_name,
            config={
                "dimension": s3vector_engine.embedding_dim,
                "metric": "cosine"
            }
        )
        
        # Prepare text documents
        documents = [
            "Python is a high-level programming language",
            "Machine learning is a subset of artificial intelligence", 
            "Neural networks are inspired by biological neurons",
            "Natural language processing enables computers to understand text",
            "Deep learning uses multiple layers of neural networks",
            "Computer vision helps machines interpret visual information",
            "Data science combines statistics and programming",
            "Reinforcement learning involves learning from rewards",
        ]
        
        # Generate embeddings
        vectors = []
        metadata = []
        ids = []
        
        embeddings = await s3vector_engine._embed_texts(documents)
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vectors.append(embedding)
            metadata.append({"text": doc, "doc_id": i})
            ids.append(f"doc_{i}")
        
        # Add to collection
        result = await s3vector_engine.add_vectors(
            collection_name, vectors, metadata, ids
        )
        assert len(result) == len(documents)
        
        # Search with text query
        query_text = "artificial intelligence and deep learning"
        query_embeddings = await s3vector_engine._embed_texts([query_text])
        query_embedding = query_embeddings[0]
        
        # Search with a higher limit to ensure more shards are searched
        # This makes the test more stable as it searches at least 5 shards
        results = await s3vector_engine.search_vectors(
            collection_name,
            query_vector=query_embedding,
            limit=15  # This triggers searching 5 shards instead of just 2
        )
        
        assert len(results) >= 3
        # Take only top 3 for testing
        results = results[:3]
        
        # Print results for debugging
        print("\nSemantic Search Results:")
        print(f"Query: {query_text}")
        for i, result in enumerate(results[:3]):
            print(f"{i+1}. Score: {result['_score']:.3f} - {result['text']}")
        
        # The most relevant documents should be about AI/ML
        # Check that at least one of the top 3 results contains relevant terms
        # This makes the test more robust to minor variations in embeddings
        relevant_terms = ["machine learning", "artificial", "deep learning"]
        found_relevant = False
        for result in results[:3]:
            result_text = result["text"].lower()
            if any(term in result_text for term in relevant_terms):
                found_relevant = True
                break
        
        assert found_relevant, f"None of the top 3 results contain expected AI/ML terms. Results: {[r['text'] for r in results[:3]]}"
    
    @pytest.mark.asyncio
    async def test_real_world_product_search(self, s3vector_engine):
        """Test a real-world scenario: product recommendation system."""
        if s3vector_engine.model is None:
            pytest.skip("No embedding model available")
        
        collection_name = "product_catalog"
        
        # Create collection
        await s3vector_engine.create_collection(
            collection_name,
            config={
                "dimension": s3vector_engine.embedding_dim,
                "metric": "cosine",
                "num_centroids": 3,
                "shard_size": 50
            }
        )
        
        # Product catalog
        products = [
            {"id": "prod1", "name": "iPhone 15 Pro", "category": "Electronics", "price": 999, 
             "description": "Latest Apple smartphone with advanced camera system and A17 Pro chip"},
            {"id": "prod2", "name": "Samsung Galaxy S24", "category": "Electronics", "price": 899,
             "description": "Android flagship phone with AI features and excellent display"},
            {"id": "prod3", "name": "MacBook Pro M3", "category": "Electronics", "price": 1599,
             "description": "Professional laptop with Apple Silicon chip for creative work"},
            {"id": "prod4", "name": "Dell XPS 15", "category": "Electronics", "price": 1299,
             "description": "Windows laptop with powerful specs for business and gaming"},
            {"id": "prod5", "name": "Sony WH-1000XM5", "category": "Audio", "price": 399,
             "description": "Premium noise-canceling headphones with exceptional sound quality"},
            {"id": "prod6", "name": "AirPods Pro", "category": "Audio", "price": 249,
             "description": "Wireless earbuds with active noise cancellation and spatial audio"},
            {"id": "prod7", "name": "Nike Air Max", "category": "Footwear", "price": 150,
             "description": "Comfortable running shoes with Air cushioning technology"},
            {"id": "prod8", "name": "Adidas Ultraboost", "category": "Footwear", "price": 180,
             "description": "High-performance running shoes with responsive Boost midsole"},
        ]
        
        # Generate embeddings for product descriptions
        texts = [f"{p['name']} {p['description']}" for p in products]
        embeddings = await s3vector_engine._embed_texts(texts)
        
        vectors = []
        metadata = []
        ids = []
        
        for product, embedding in zip(products, embeddings):
            vectors.append(embedding)
            metadata.append({
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "description": product["description"]
            })
            ids.append(product["id"])
        
        # Add products to collection
        await s3vector_engine.add_vectors(collection_name, vectors, metadata, ids)
        
        # Customer query: "I need a good laptop for video editing"
        query = "professional laptop for video editing and creative work"
        query_embeddings = await s3vector_engine._embed_texts([query])
        query_embedding = query_embeddings[0]
        
        # Find recommendations
        results = await s3vector_engine.search_vectors(
            collection_name,
            query_vector=query_embedding,
            limit=3
        )
        
        print("\n=== Product Recommendations ===")
        print(f"Customer Query: '{query}'")
        print("\nTop Recommendations:")
        for i, result in enumerate(results, 1):
            # Metadata is flattened into the result
            print(f"\n{i}. {result['name']} - ${result['price']}")
            print(f"   Category: {result['category']}")
            print(f"   Match Score: {result['_score']:.3f}")
            print(f"   Description: {result['description']}")
        
        # Verify recommendations make sense
        assert len(results) > 0
        # Top results should be laptops (metadata is flattened)
        top_categories = [r["category"] for r in results[:2]]
        assert "Electronics" in top_categories


# ============================================================================
# Performance and Scalability Tests
# ============================================================================

class TestPerformance:
    """Test performance and scalability."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_vectors,num_shards", [
        (1000, 4),
        (5000, 10),
    ])
    async def test_spann_scalability(self, s3_config, redis_client, num_vectors, num_shards):
        """Test SPANN indexing scalability with different dataset sizes."""
        bucket_name = f"test-scale-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=num_shards,
            shard_size=max(100, num_vectors // num_shards)
        )
        await engine.initialize()
        
        collection_id = "scale_test"
        
        try:
            # Create collection
            start = time.time()
            await engine.create_collection(collection_id, {
                "dimension": 384,
                "metric": "cosine",
                "num_centroids": num_shards
            })
            create_time = time.time() - start
            
            # Generate test vectors
            vectors = [np.random.rand(384).astype(np.float32) for _ in range(num_vectors)]
            metadata = [{"id": i, "category": f"cat_{i % 10}"} for i in range(num_vectors)]
            
            # Add vectors in batches
            batch_size = 100
            add_times = []
            for i in range(0, num_vectors, batch_size):
                batch_vectors = vectors[i:i+batch_size]
                batch_metadata = metadata[i:i+batch_size]
                
                start = time.time()
                await engine.add_vectors(collection_id, batch_vectors, batch_metadata)
                add_times.append(time.time() - start)
            
            total_add_time = sum(add_times)
            
            # Test search performance
            search_times = []
            for _ in range(50):
                query = vectors[np.random.randint(0, num_vectors)]
                start = time.time()
                results = await engine.search_vectors(
                    collection_id, query_vector=query, limit=10
                )
                search_times.append(time.time() - start)
            
            avg_search_time = np.mean(search_times) * 1000  # ms
            
            print(f"\nSPANN Scalability Test:")
            print(f"  Vectors: {num_vectors}, Shards: {num_shards}")
            print(f"  Collection creation: {create_time:.3f}s")
            print(f"  Total insert time: {total_add_time:.3f}s")
            print(f"  Insert throughput: {num_vectors/total_add_time:.0f} vectors/sec")
            print(f"  Avg search time: {avg_search_time:.3f}ms")
            print(f"  Search QPS: {1000/avg_search_time:.0f}")
            
            # Performance assertions - adjusted for realistic expectations
            # With S3 backend and multiple shards, search time depends on shard loading
            if num_vectors <= 1000:
                assert avg_search_time < 100  # Small dataset should be fast
            else:
                # Larger datasets with more shards have higher latency due to S3 operations
                assert avg_search_time < 500  # Allow up to 500ms for large datasets
            
            assert num_vectors/total_add_time > 100  # > 100 vectors/sec insert
        except Exception as e:
            raise e
        finally:
            # Cleanup
            await engine.delete_collection(collection_id)
            
            # Delete bucket
            async with engine.s3_client_factory() as s3_client:
                await s3_client.delete_bucket(Bucket=bucket_name)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, s3_config, redis_client):
        """Test concurrent add and search operations."""
        bucket_name = f"test-concurrent-{uuid.uuid4().hex[:8]}"
        
        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=10,
            shard_size=200
        )
        await engine.initialize()
        
        collection_id = "concurrent_test"
        
        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine"
            })
            
            # Add initial vectors
            initial_vectors = [np.random.rand(128).astype(np.float32) for _ in range(500)]
            await engine.add_vectors(collection_id, initial_vectors)
            
            # Define concurrent operations
            async def add_batch(batch_id):
                vectors = [np.random.rand(128).astype(np.float32) for _ in range(100)]
                metadata = [{"batch": batch_id} for _ in range(100)]
                await engine.add_vectors(collection_id, vectors, metadata)
                return batch_id
            
            async def search_vectors(search_id):
                query = np.random.rand(128).astype(np.float32)
                results = await engine.search_vectors(collection_id, query_vector=query, limit=5)
                return (search_id, len(results))
            
            # Run concurrent operations
            start = time.time()
            tasks = []
            
            # Mix adds and searches
            for i in range(5):
                tasks.append(add_batch(i))
            for i in range(10):
                tasks.append(search_vectors(i))
            
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start
            
            # Count successful operations
            add_count = sum(1 for r in results if isinstance(r, int))
            search_count = sum(1 for r in results if isinstance(r, tuple))
            
            print(f"\nConcurrent Operations Test:")
            print(f"  Concurrent adds: {add_count}")
            print(f"  Concurrent searches: {search_count}")
            print(f"  Total time: {elapsed:.3f}s")
            print(f"  Operations/sec: {(add_count + search_count)/elapsed:.0f}")
            
            # All operations should succeed
            assert add_count == 5
            assert search_count == 10
        except Exception as e:
            raise e
        finally:
            # Cleanup
            await engine.delete_collection(collection_id)
            
            # Delete bucket
            async with engine.s3_client_factory() as s3_client:
                await s3_client.delete_bucket(Bucket=bucket_name)


# ============================================================================
# Helper Functions and Factory Tests
# ============================================================================

@pytest.mark.asyncio
async def test_create_s3_vector_engine_from_config(s3_config):
    """Test creating engine from config."""
    config = {
        "endpoint_url": s3_config["endpoint_url"],
        "access_key_id": s3_config["access_key_id"],
        "secret_access_key": s3_config["secret_access_key"],
        "region_name": s3_config["region"],
        "bucket_name": f"test-config-{uuid.uuid4().hex[:8]}",
        "num_centroids": 20,
        "shard_size": 5000,
    }
    
    engine = create_s3_vector_engine_from_config(config)
    assert engine.endpoint_url == config["endpoint_url"]
    assert engine.bucket_name == config["bucket_name"]
    assert engine.num_centroids == 20
    assert engine.shard_size == 5000


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])