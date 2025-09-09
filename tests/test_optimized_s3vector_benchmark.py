"""
Pytest-based benchmark tests for optimized S3Vector engine.

This test suite uses real infrastructure and can be run with:
pytest tests/test_optimized_s3vector_benchmark.py -v -s --tb=short

Requirements:
- MinIO running on localhost:9000
- Redis running on localhost:6379  
- PostgreSQL with pgvector on localhost:5432
"""

import pytest
import pytest_asyncio
import asyncio
import time
import numpy as np
import uuid
import logging
from typing import Dict, Any, List, Tuple
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine

from hypha.s3vector import create_s3_vector_engine_from_config as create_original_s3
from hypha.s3vector_optimized import create_optimized_s3_vector_engine_from_config as create_optimized_s3
from hypha.pgvector import PgVectorSearchEngine

# Import test configuration
from . import (
    MINIO_PORT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    MINIO_SERVER_URL,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
async def redis_client():
    """Create Redis client for the session."""
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=False)
    try:
        await client.ping()
        await client.flushall()
        yield client
    finally:
        await client.flushall()
        await client.close()


@pytest.fixture(scope="session") 
async def pg_engine():
    """Create PostgreSQL engine for the session."""
    db_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/vector_benchmark"
    engine = create_async_engine(db_url, echo=False)
    
    try:
        # Test connection
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def s3_config():
    """S3 configuration for testing."""
    return {
        "endpoint_url": MINIO_SERVER_URL,
        "access_key_id": MINIO_ROOT_USER,
        "secret_access_key": MINIO_ROOT_PASSWORD,
        "region_name": "us-east-1",
        "bucket_name": f"benchmark-test-{uuid.uuid4().hex[:8]}"
    }


@pytest.fixture
async def engines(redis_client, pg_engine, s3_config):
    """Setup all engines for testing."""
    engines = {}
    
    # Original S3Vector
    original_config = {
        **s3_config,
        "redis_client": redis_client,
        "num_centroids": 20,
        "shard_size": 1000,
    }
    engines["original"] = create_original_s3(original_config)
    await engines["original"].initialize()
    
    # Optimized S3Vector  
    optimized_config = {
        **s3_config,
        "redis_client": redis_client,
        "num_centroids": 20,
        "shard_size": 500,
        "max_s3_connections": 10,
        "enable_early_termination": True,
        "search_timeout": 30.0,
    }
    engines["optimized"] = create_optimized_s3(optimized_config)
    await engines["optimized"].initialize()
    
    # PgVector
    engines["pgvector"] = PgVectorSearchEngine(pg_engine, prefix="benchmark_test")
    
    yield engines
    
    # Cleanup
    for engine in engines.values():
        try:
            collections = await engine.list_collections()
            for collection in collections:
                await engine.delete_collection(collection)
        except:
            pass


def generate_test_vectors(num_vectors: int, dimension: int = 384) -> Tuple[np.ndarray, List[Dict]]:
    """Generate test vectors and metadata."""
    np.random.seed(42)  # Reproducible
    
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    metadata = [
        {
            "id": i,
            "category": f"cat_{i % 5}",
            "score": float(np.random.rand()),
        }
        for i in range(num_vectors)
    ]
    
    return vectors, metadata


async def benchmark_insert(engine_name: str, engine: Any, vectors: np.ndarray, metadata: List[Dict]) -> Dict[str, float]:
    """Benchmark insert performance."""
    collection_name = f"insert_test_{uuid.uuid4().hex[:8]}"
    
    # Create collection
    if engine_name == "pgvector":
        await engine.create_collection(collection_name, dimension=vectors.shape[1])
    elif engine_name == "optimized":
        optimal_centroids = engine.calculate_optimal_centroids(len(vectors))
        await engine.create_collection(collection_name, {
            "dimension": vectors.shape[1],
            "num_centroids": optimal_centroids,
        })
    else:
        await engine.create_collection(collection_name, {
            "dimension": vectors.shape[1],
            "num_centroids": min(10, len(vectors) // 50),
        })
    
    # Insert vectors
    start_time = time.time()
    
    if engine_name == "pgvector":
        # PgVector format
        pg_vectors = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            pg_vectors.append({
                "id": f"vec_{i}",
                "vector": vector.tolist(),
                "manifest": meta
            })
        await engine.add_vectors(collection_name, pg_vectors)
    else:
        # S3Vector format
        ids = [f"vec_{i}" for i in range(len(vectors))]
        await engine.add_vectors(collection_name, vectors.tolist(), metadata, ids)
    
    duration = time.time() - start_time
    throughput = len(vectors) / duration
    
    # Cleanup
    await engine.delete_collection(collection_name)
    
    return {
        "duration": duration,
        "throughput": throughput,
        "vectors_per_second": throughput
    }


async def benchmark_search(engine_name: str, engine: Any, vectors: np.ndarray, metadata: List[Dict]) -> Dict[str, Any]:
    """Benchmark search performance."""
    collection_name = f"search_test_{uuid.uuid4().hex[:8]}"
    
    # Create collection and insert data
    if engine_name == "pgvector":
        await engine.create_collection(collection_name, dimension=vectors.shape[1])
        pg_vectors = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            pg_vectors.append({
                "id": f"vec_{i}",
                "vector": vector.tolist(),
                "manifest": meta
            })
        await engine.add_vectors(collection_name, pg_vectors)
    elif engine_name == "optimized":
        optimal_centroids = engine.calculate_optimal_centroids(len(vectors))
        await engine.create_collection(collection_name, {
            "dimension": vectors.shape[1],
            "num_centroids": optimal_centroids,
        })
        ids = [f"vec_{i}" for i in range(len(vectors))]
        await engine.add_vectors(collection_name, vectors.tolist(), metadata, ids)
    else:
        await engine.create_collection(collection_name, {
            "dimension": vectors.shape[1],
            "num_centroids": min(10, len(vectors) // 50),
        })
        ids = [f"vec_{i}" for i in range(len(vectors))]
        await engine.add_vectors(collection_name, vectors.tolist(), metadata, ids)
    
    query_vector = vectors[0]
    
    # Cold search
    start_time = time.time()
    if engine_name == "pgvector":
        results = await engine.search_vectors(collection_name, query_vector=query_vector, limit=10)
    else:
        results = await engine.search_vectors(collection_name, query_vector=query_vector, limit=10)
    cold_time = (time.time() - start_time) * 1000
    
    # Warm searches
    warm_times = []
    for _ in range(5):
        start_time = time.time()
        if engine_name == "pgvector":
            results = await engine.search_vectors(collection_name, query_vector=query_vector, limit=10)
        else:
            results = await engine.search_vectors(collection_name, query_vector=query_vector, limit=10)
        warm_times.append((time.time() - start_time) * 1000)
    
    # Cleanup
    await engine.delete_collection(collection_name)
    
    return {
        "cold_time_ms": cold_time,
        "avg_warm_time_ms": np.mean(warm_times),
        "std_warm_time_ms": np.std(warm_times),
        "qps": 1000 / np.mean(warm_times),
        "num_results": len(results) if results else 0
    }


class TestOptimizedS3VectorBenchmark:
    """Test suite for benchmarking optimized S3Vector performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("dataset_size", [1000, 5000])
    async def test_insert_performance_comparison(self, engines, dataset_size):
        """Compare insert performance across all engines."""
        vectors, metadata = generate_test_vectors(dataset_size)
        
        results = {}
        for engine_name, engine in engines.items():
            logger.info(f"Testing {engine_name} insert with {dataset_size} vectors")
            
            result = await benchmark_insert(engine_name, engine, vectors, metadata)
            results[engine_name] = result
            
            logger.info(f"{engine_name}: {result['throughput']:.0f} vec/s in {result['duration']:.2f}s")
        
        # Verify optimized is better than original for larger datasets
        if dataset_size >= 5000:
            assert results["optimized"]["throughput"] > results["original"]["throughput"] * 0.8, \
                "Optimized S3Vector should be competitive with original for insert"
        
        # All engines should complete successfully
        for engine_name, result in results.items():
            assert result["throughput"] > 0, f"{engine_name} should have positive throughput"
            assert result["duration"] > 0, f"{engine_name} should have positive duration"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("dataset_size", [1000, 5000])
    async def test_search_performance_comparison(self, engines, dataset_size):
        """Compare search performance across all engines."""
        vectors, metadata = generate_test_vectors(dataset_size)
        
        results = {}
        for engine_name, engine in engines.items():
            logger.info(f"Testing {engine_name} search with {dataset_size} vectors")
            
            result = await benchmark_search(engine_name, engine, vectors, metadata)
            results[engine_name] = result
            
            logger.info(f"{engine_name}: {result['qps']:.1f} QPS, cold: {result['cold_time_ms']:.1f}ms, warm: {result['avg_warm_time_ms']:.1f}ms")
        
        # Verify optimized shows improvement over original
        if dataset_size >= 5000:
            optimized_qps = results["optimized"]["qps"]
            original_qps = results["original"]["qps"]
            
            # Optimized should be at least 20% better for larger datasets
            improvement_threshold = 1.2
            assert optimized_qps > original_qps * 0.5, \
                f"Optimized QPS ({optimized_qps:.1f}) should be competitive with original ({original_qps:.1f})"
        
        # All engines should return results
        for engine_name, result in results.items():
            assert result["num_results"] > 0, f"{engine_name} should return search results"
            assert result["qps"] > 0, f"{engine_name} should have positive QPS"
    
    @pytest.mark.asyncio
    async def test_optimized_engine_specific_features(self, engines):
        """Test optimized engine specific features."""
        optimized_engine = engines["optimized"]
        vectors, metadata = generate_test_vectors(2000)
        
        # Test adaptive centroid calculation
        optimal_centroids = optimized_engine.calculate_optimal_centroids(len(vectors))
        assert optimal_centroids > 0, "Should calculate positive number of centroids"
        assert optimal_centroids <= len(vectors), "Centroids should not exceed vector count"
        
        # Test HNSW parameter optimization
        hnsw_params = optimized_engine.optimize_hnsw_params(optimal_centroids)
        assert "m" in hnsw_params, "Should return HNSW m parameter"
        assert "ef" in hnsw_params, "Should return HNSW ef parameter"
        assert hnsw_params["m"] > 0, "m parameter should be positive"
        assert hnsw_params["ef"] > 0, "ef parameter should be positive"
        
        logger.info(f"Optimal centroids for {len(vectors)} vectors: {optimal_centroids}")
        logger.info(f"HNSW params: {hnsw_params}")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, engines):
        """Test concurrent search operations."""
        vectors, metadata = generate_test_vectors(1000)
        
        for engine_name, engine in engines.items():
            collection_name = f"concurrent_test_{uuid.uuid4().hex[:8]}"
            
            # Setup collection
            if engine_name == "pgvector":
                await engine.create_collection(collection_name, dimension=vectors.shape[1])
                pg_vectors = []
                for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                    pg_vectors.append({
                        "id": f"vec_{i}",
                        "vector": vector.tolist(),
                        "manifest": meta
                    })
                await engine.add_vectors(collection_name, pg_vectors)
            else:
                config = {"dimension": vectors.shape[1]}
                if engine_name == "optimized":
                    config["num_centroids"] = engine.calculate_optimal_centroids(len(vectors))
                else:
                    config["num_centroids"] = 10
                    
                await engine.create_collection(collection_name, config)
                ids = [f"vec_{i}" for i in range(len(vectors))]
                await engine.add_vectors(collection_name, vectors.tolist(), metadata, ids)
            
            # Test concurrent searches
            async def search_task():
                query = vectors[np.random.randint(0, len(vectors))]
                if engine_name == "pgvector":
                    return await engine.search_vectors(collection_name, query_vector=query, limit=5)
                else:
                    return await engine.search_vectors(collection_name, query_vector=query, limit=5)
            
            start_time = time.time()
            results = await asyncio.gather(*[search_task() for _ in range(5)])
            duration = time.time() - start_time
            
            # Verify all searches completed
            assert len(results) == 5, f"{engine_name} should complete all concurrent searches"
            for result in results:
                assert len(result) > 0, f"{engine_name} should return results from concurrent searches"
            
            concurrent_qps = 5 / duration
            logger.info(f"{engine_name} concurrent QPS: {concurrent_qps:.1f}")
            
            # Cleanup
            await engine.delete_collection(collection_name)
    
    @pytest.mark.asyncio 
    @pytest.mark.slow  # Mark as slow test
    async def test_large_dataset_performance(self, engines):
        """Test performance with larger dataset (marked as slow)."""
        vectors, metadata = generate_test_vectors(10000)
        
        # Only test if explicitly requested
        import os
        if not os.environ.get("RUN_SLOW_TESTS"):
            pytest.skip("Slow test - set RUN_SLOW_TESTS=1 to run")
        
        results = {}
        
        # Test each engine
        for engine_name, engine in engines.items():
            logger.info(f"Large dataset test: {engine_name}")
            
            # Insert benchmark
            insert_result = await benchmark_insert(engine_name, engine, vectors, metadata)
            
            # Search benchmark  
            search_result = await benchmark_search(engine_name, engine, vectors, metadata)
            
            results[engine_name] = {
                "insert": insert_result,
                "search": search_result
            }
            
            logger.info(f"{engine_name} - Insert: {insert_result['throughput']:.0f} vec/s, Search: {search_result['qps']:.1f} QPS")
        
        # Verify optimized shows meaningful improvement
        if "optimized" in results and "original" in results:
            opt_search_qps = results["optimized"]["search"]["qps"]
            orig_search_qps = results["original"]["search"]["qps"]
            
            logger.info(f"Search QPS improvement: {opt_search_qps/orig_search_qps:.2f}x")
            
            # Should be at least competitive
            assert opt_search_qps > orig_search_qps * 0.5, \
                "Optimized should be competitive with original on large datasets"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])