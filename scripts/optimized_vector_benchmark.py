#!/usr/bin/env python3
"""
Comprehensive benchmark comparing Optimized S3Vector vs Original S3Vector vs PgVector.

This benchmark uses real infrastructure:
- MinIO for S3 storage
- Redis for caching
- PostgreSQL with pgvector extension
"""

import asyncio
import time
import numpy as np
import os
import sys
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from tabulate import tabulate
import redis.asyncio as redis
import psutil
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypha.s3vector import create_s3_vector_engine_from_config as create_original_s3
from hypha.s3vector_optimized import create_optimized_s3_vector_engine_from_config as create_optimized_s3
from hypha.pgvector import PgVectorSearchEngine
from sqlalchemy.ext.asyncio import create_async_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    engine_name: str
    operation: str
    dataset_size: int
    duration: float
    throughput: float
    memory_usage_mb: float
    additional_metrics: Dict[str, Any]


class InfrastructureManager:
    """Manages test infrastructure setup and teardown."""
    
    def __init__(self):
        self.redis_client = None
        self.pg_engine = None
        
    async def setup_infrastructure(self):
        """Setup test infrastructure."""
        logger.info("Setting up test infrastructure...")
        
        # Setup Redis connection
        try:
            self.redis_client = redis.Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=False)
            await self.redis_client.ping()
            await self.redis_client.flushall()  # Clean slate
            logger.info("✓ Redis connected and cleared")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.info("Please ensure Redis is running on localhost:6379")
            raise
        
        # Setup PostgreSQL connection
        try:
            db_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/vector_benchmark"
            self.pg_engine = create_async_engine(db_url, echo=False)
            
            # Test connection
            async with self.pg_engine.begin() as conn:
                await conn.execute("SELECT 1")
            logger.info("✓ PostgreSQL connected")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            logger.info("Please ensure PostgreSQL with pgvector is running on localhost:5432")
            logger.info("Database: vector_benchmark, User: postgres, Password: postgres")
            raise
    
    async def cleanup_infrastructure(self):
        """Cleanup test infrastructure."""
        if self.redis_client:
            await self.redis_client.flushall()
            await self.redis_client.close()
        
        if self.pg_engine:
            await self.pg_engine.dispose()


class OptimizedVectorBenchmark:
    """Comprehensive benchmark with optimized S3Vector engine."""
    
    def __init__(self):
        self.infra = InfrastructureManager()
        self.results: List[BenchmarkResult] = []
        self.dimension = 384
        
        # Test configuration
        self.s3_config = {
            "endpoint_url": "http://localhost:9000",  # MinIO
            "access_key_id": "minioadmin",
            "secret_access_key": "minioadmin",
            "region_name": "us-east-1",
            "bucket_name": f"benchmark-{uuid.uuid4().hex[:8]}"
        }
    
    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def generate_test_data(self, num_vectors: int, seed: int = 42) -> Tuple[np.ndarray, List[Dict]]:
        """Generate reproducible test data."""
        np.random.seed(seed)
        
        # Generate normalized vectors for consistent similarity scores
        vectors = np.random.randn(num_vectors, self.dimension).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        metadata = [
            {
                "id": i,
                "category": f"cat_{i % 10}",
                "score": float(np.random.rand()),
                "text": f"Document {i} with some test content"
            }
            for i in range(num_vectors)
        ]
        
        return vectors, metadata
    
    async def setup_engines(self) -> Dict[str, Any]:
        """Setup all engines for testing."""
        await self.infra.setup_infrastructure()
        
        engines = {}
        
        # Original S3Vector Engine
        original_config = {
            **self.s3_config,
            "redis_client": self.infra.redis_client,
            "num_centroids": 50,  # Reasonable default
            "shard_size": 10000,
        }
        engines["S3Vector_Original"] = create_original_s3(original_config)
        await engines["S3Vector_Original"].initialize()
        
        # Optimized S3Vector Engine
        optimized_config = {
            **self.s3_config,
            "redis_client": self.infra.redis_client,
            "num_centroids": 50,  # Will be auto-optimized
            "shard_size": 5000,   # Smaller shards for better parallelism
            "max_s3_connections": 20,
            "enable_early_termination": True,
            "search_timeout": 60.0,
        }
        engines["S3Vector_Optimized"] = create_optimized_s3(optimized_config)
        await engines["S3Vector_Optimized"].initialize()
        
        # PgVector Engine
        engines["PgVector"] = PgVectorSearchEngine(self.infra.pg_engine, prefix="benchmark")
        
        logger.info("✓ All engines initialized")
        return engines
    
    async def benchmark_insert_performance(
        self, 
        engine_name: str, 
        engine: Any, 
        vectors: np.ndarray, 
        metadata: List[Dict],
        collection_name: str
    ) -> BenchmarkResult:
        """Benchmark insert performance for an engine."""
        logger.info(f"Benchmarking {engine_name} insert performance...")
        
        start_memory = self.get_memory_usage()
        
        # Create collection with optimal settings
        if "S3Vector_Optimized" in engine_name:
            optimal_centroids = engine.calculate_optimal_centroids(len(vectors))
            await engine.create_collection(collection_name, {
                "dimension": self.dimension,
                "num_centroids": optimal_centroids,
                "shard_size": engine.shard_size
            })
            logger.info(f"  Using {optimal_centroids} centroids for optimized engine")
        elif "S3Vector" in engine_name:
            await engine.create_collection(collection_name, {
                "dimension": self.dimension,
                "num_centroids": min(50, len(vectors) // 100),
            })
        else:  # PgVector
            await engine.create_collection(collection_name, dimension=self.dimension)
        
        # Insert vectors in batches
        batch_size = min(1000, len(vectors) // 10) if len(vectors) > 10000 else 500
        batch_times = []
        
        start_time = time.time()
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            batch_start = time.time()
            
            if "PgVector" in engine_name:
                # PgVector format
                pg_vectors = []
                for j, (vector, meta) in enumerate(zip(batch_vectors, batch_metadata)):
                    pg_vectors.append({
                        "id": f"vec_{i+j}",
                        "vector": vector.tolist(),
                        "manifest": meta
                    })
                await engine.add_vectors(collection_name, pg_vectors)
            else:
                # S3Vector format
                batch_ids = [f"vec_{i+j}" for j in range(len(batch_vectors))]
                await engine.add_vectors(collection_name, batch_vectors.tolist(), batch_metadata, batch_ids)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Progress logging for large datasets
            if len(vectors) > 10000 and (i // batch_size) % 10 == 0:
                progress = (i + batch_size) / len(vectors) * 100
                logger.info(f"  Insert progress: {progress:.1f}%")
        
        total_time = time.time() - start_time
        end_memory = self.get_memory_usage()
        
        return BenchmarkResult(
            engine_name=engine_name,
            operation="insert",
            dataset_size=len(vectors),
            duration=total_time,
            throughput=len(vectors) / total_time,
            memory_usage_mb=end_memory - start_memory,
            additional_metrics={
                "avg_batch_time": np.mean(batch_times),
                "batch_size": batch_size,
                "num_batches": len(batch_times)
            }
        )
    
    async def benchmark_search_performance(
        self,
        engine_name: str,
        engine: Any,
        vectors: np.ndarray,
        collection_name: str
    ) -> BenchmarkResult:
        """Benchmark search performance for an engine."""
        logger.info(f"Benchmarking {engine_name} search performance...")
        
        query_vector = vectors[0]  # Use first vector as query
        k_values = [10, 50] if len(vectors) > 50000 else [10, 50, 100]
        
        search_metrics = {}
        all_search_times = []
        
        for k in k_values:
            # Cold search
            start_time = time.time()
            if "PgVector" in engine_name:
                results = await engine.search_vectors(collection_name, query_vector=query_vector, limit=k)
            else:
                results = await engine.search_vectors(collection_name, query_vector=query_vector, limit=k)
            cold_time = (time.time() - start_time) * 1000
            
            # Warm searches
            warm_times = []
            num_warm_searches = 5 if len(vectors) > 50000 else 10
            
            for _ in range(num_warm_searches):
                start_time = time.time()
                if "PgVector" in engine_name:
                    results = await engine.search_vectors(collection_name, query_vector=query_vector, limit=k)
                else:
                    results = await engine.search_vectors(collection_name, query_vector=query_vector, limit=k)
                warm_times.append((time.time() - start_time) * 1000)
            
            all_search_times.extend(warm_times)
            
            search_metrics[f"k_{k}"] = {
                "cold_time_ms": cold_time,
                "avg_warm_time_ms": np.mean(warm_times),
                "std_warm_time_ms": np.std(warm_times),
                "num_results": len(results) if results else 0
            }
        
        # Concurrent search test
        concurrent_times = []
        num_concurrent = 5 if len(vectors) > 50000 else 10
        
        async def concurrent_search():
            query = vectors[np.random.randint(0, min(1000, len(vectors)))]
            start = time.time()
            if "PgVector" in engine_name:
                await engine.search_vectors(collection_name, query_vector=query, limit=10)
            else:
                await engine.search_vectors(collection_name, query_vector=query, limit=10)
            return (time.time() - start) * 1000
        
        concurrent_start = time.time()
        concurrent_results = await asyncio.gather(*[concurrent_search() for _ in range(num_concurrent)])
        concurrent_total = time.time() - concurrent_start
        
        search_metrics["concurrent"] = {
            "avg_time_ms": np.mean(concurrent_results),
            "qps": num_concurrent / concurrent_total,
            "num_queries": num_concurrent
        }
        
        return BenchmarkResult(
            engine_name=engine_name,
            operation="search",
            dataset_size=len(vectors),
            duration=np.mean(all_search_times) / 1000,  # Average search time in seconds
            throughput=1000 / np.mean(all_search_times),  # QPS
            memory_usage_mb=0,  # Not measured for search
            additional_metrics=search_metrics
        )
    
    async def run_dataset_benchmark(self, dataset_size: int):
        """Run benchmark for a specific dataset size."""
        logger.info(f"\n{'='*60}")
        logger.info(f"BENCHMARKING {dataset_size:,} VECTORS")
        logger.info(f"{'='*60}")
        
        # Generate test data
        vectors, metadata = self.generate_test_data(dataset_size)
        
        # Setup engines
        engines = await self.setup_engines()
        
        dataset_results = []
        
        for engine_name, engine in engines.items():
            collection_name = f"{engine_name.lower()}_{dataset_size}_{uuid.uuid4().hex[:8]}"
            
            try:
                logger.info(f"\n--- Testing {engine_name} ---")
                
                # Insert benchmark
                insert_result = await self.benchmark_insert_performance(
                    engine_name, engine, vectors, metadata, collection_name
                )
                dataset_results.append(insert_result)
                self.results.append(insert_result)
                
                logger.info(f"Insert: {insert_result.throughput:.0f} vec/s, {insert_result.duration:.2f}s")
                
                # Search benchmark
                search_result = await self.benchmark_search_performance(
                    engine_name, engine, vectors, collection_name
                )
                dataset_results.append(search_result)
                self.results.append(search_result)
                
                logger.info(f"Search: {search_result.throughput:.1f} QPS, {search_result.duration*1000:.1f}ms avg")
                
                # Cleanup
                try:
                    await engine.delete_collection(collection_name)
                except:
                    pass  # Best effort cleanup
                    
            except Exception as e:
                logger.error(f"Failed to benchmark {engine_name}: {e}")
                continue
        
        # Cleanup engines
        for engine in engines.values():
            if hasattr(engine, 's3_pool'):
                # Cleanup S3 connections for optimized engine
                pass
        
        return dataset_results
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report = "# Optimized S3Vector vs Original vs PgVector Benchmark Report\n\n"
        report += f"**Test Configuration:**\n"
        report += f"- Vector Dimension: {self.dimension}\n"
        report += f"- Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"- Infrastructure: MinIO (S3), Redis, PostgreSQL with pgvector\n\n"
        
        # Group results by dataset size and operation
        by_dataset = {}
        for result in self.results:
            key = (result.dataset_size, result.operation)
            if key not in by_dataset:
                by_dataset[key] = {}
            by_dataset[key][result.engine_name] = result
        
        # Insert Performance Table
        report += "## Insert Performance Comparison\n\n"
        insert_headers = ["Dataset Size", "S3Vector Original (vec/s)", "S3Vector Optimized (vec/s)", "PgVector (vec/s)", "Best"]
        insert_rows = []
        
        dataset_sizes = sorted(set(r.dataset_size for r in self.results))
        
        for size in dataset_sizes:
            key = (size, "insert")
            if key in by_dataset:
                results = by_dataset[key]
                
                original_throughput = results.get("S3Vector_Original", {}).throughput or 0
                optimized_throughput = results.get("S3Vector_Optimized", {}).throughput or 0  
                pg_throughput = results.get("PgVector", {}).throughput or 0
                
                best = max([
                    ("S3Vector Original", original_throughput),
                    ("S3Vector Optimized", optimized_throughput),
                    ("PgVector", pg_throughput)
                ], key=lambda x: x[1])[0]
                
                insert_rows.append([
                    f"{size:,}",
                    f"{original_throughput:.0f}",
                    f"{optimized_throughput:.0f}",
                    f"{pg_throughput:.0f}",
                    f"**{best}**"
                ])
        
        report += tabulate(insert_rows, headers=insert_headers, tablefmt="pipe") + "\n\n"
        
        # Search Performance Table
        report += "## Search Performance Comparison\n\n"
        search_headers = ["Dataset Size", "S3Vector Original (QPS)", "S3Vector Optimized (QPS)", "PgVector (QPS)", "Best"]
        search_rows = []
        
        for size in dataset_sizes:
            key = (size, "search")
            if key in by_dataset:
                results = by_dataset[key]
                
                original_qps = results.get("S3Vector_Original", {}).throughput or 0
                optimized_qps = results.get("S3Vector_Optimized", {}).throughput or 0
                pg_qps = results.get("PgVector", {}).throughput or 0
                
                best = max([
                    ("S3Vector Original", original_qps),
                    ("S3Vector Optimized", optimized_qps),
                    ("PgVector", pg_qps)
                ], key=lambda x: x[1])[0]
                
                search_rows.append([
                    f"{size:,}",
                    f"{original_qps:.1f}",
                    f"{optimized_qps:.1f}",
                    f"{pg_qps:.1f}",
                    f"**{best}**"
                ])
        
        report += tabulate(search_rows, headers=search_headers, tablefmt="pipe") + "\n\n"
        
        # Detailed Performance Metrics
        report += "## Detailed Search Metrics\n\n"
        
        for size in dataset_sizes:
            report += f"### Dataset Size: {size:,} vectors\n\n"
            
            key = (size, "search")
            if key in by_dataset:
                results = by_dataset[key]
                
                detail_headers = ["Engine", "Cold Search (ms)", "Warm Search (ms)", "Concurrent QPS"]
                detail_rows = []
                
                for engine_name, result in results.items():
                    metrics = result.additional_metrics
                    k10_metrics = metrics.get("k_10", {})
                    concurrent_metrics = metrics.get("concurrent", {})
                    
                    detail_rows.append([
                        engine_name.replace("_", " "),
                        f"{k10_metrics.get('cold_time_ms', 0):.1f}",
                        f"{k10_metrics.get('avg_warm_time_ms', 0):.1f}",
                        f"{concurrent_metrics.get('qps', 0):.1f}"
                    ])
                
                report += tabulate(detail_rows, headers=detail_headers, tablefmt="pipe") + "\n\n"
        
        # Performance Analysis
        report += "## Performance Analysis\n\n"
        
        # Calculate improvements
        improvements = {}
        for size in dataset_sizes:
            insert_key = (size, "insert")
            search_key = (size, "search")
            
            if insert_key in by_dataset and search_key in by_dataset:
                insert_results = by_dataset[insert_key]
                search_results = by_dataset[search_key]
                
                # Compare optimized vs original
                if "S3Vector_Original" in insert_results and "S3Vector_Optimized" in insert_results:
                    orig_insert = insert_results["S3Vector_Original"].throughput
                    opt_insert = insert_results["S3Vector_Optimized"].throughput
                    insert_improvement = (opt_insert / orig_insert - 1) * 100 if orig_insert > 0 else 0
                    
                    orig_search = search_results["S3Vector_Original"].throughput
                    opt_search = search_results["S3Vector_Optimized"].throughput  
                    search_improvement = (opt_search / orig_search - 1) * 100 if orig_search > 0 else 0
                    
                    improvements[size] = {
                        "insert": insert_improvement,
                        "search": search_improvement
                    }
        
        report += "### S3Vector Optimization Improvements:\n\n"
        for size, impr in improvements.items():
            report += f"**{size:,} vectors:**\n"
            report += f"- Insert throughput: {impr['insert']:+.1f}%\n"
            report += f"- Search throughput: {impr['search']:+.1f}%\n\n"
        
        report += "### Key Optimizations Implemented:\n"
        report += "1. **Enhanced HNSW Index**: Multi-level graph with better connectivity\n"
        report += "2. **Parallel Shard Loading**: Concurrent S3 operations with connection pooling\n"
        report += "3. **Smart Caching**: LRU cache with compression and batch operations\n"
        report += "4. **Early Termination**: Stop search when sufficient high-quality results found\n"
        report += "5. **Vectorized Operations**: NumPy optimizations for similarity calculations\n"
        report += "6. **Adaptive Query Planning**: Dynamic shard selection based on dataset size\n\n"
        
        report += "### Recommendations:\n\n"
        report += "**Use Optimized S3Vector when:**\n"
        report += "- Dataset size > 100K vectors\n"
        report += "- Cost optimization is critical\n" 
        report += "- Horizontal scaling needed\n"
        report += "- Can tolerate 100-500ms search latency\n\n"
        
        report += "**Use PgVector when:**\n"
        report += "- Dataset size < 100K vectors\n"
        report += "- Sub-50ms latency required\n"
        report += "- Complex SQL queries needed\n"
        report += "- ACID compliance required\n"
        
        return report
    
    async def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        logger.info("Starting Optimized S3Vector Benchmark Suite")
        
        try:
            # Test different dataset sizes - progressive scaling
            test_sizes = [1000, 10000, 100000]  # Scale up to 100K for thorough testing
            
            for size in test_sizes:
                await self.run_dataset_benchmark(size)
                
                # Clear Redis cache between tests
                if self.infra.redis_client:
                    await self.infra.redis_client.flushall()
            
            # Generate and save report
            report = self.generate_report()
            
            report_filename = f"optimized_s3vector_benchmark_report_{int(time.time())}.md"
            with open(report_filename, "w") as f:
                f.write(report)
            
            logger.info(f"\n{'='*60}")
            logger.info("BENCHMARK COMPLETE!")
            logger.info(f"Report saved to: {report_filename}")
            logger.info(f"{'='*60}")
            
            return report
            
        finally:
            await self.infra.cleanup_infrastructure()


async def main():
    """Main function to run the optimized benchmark."""
    benchmark = OptimizedVectorBenchmark()
    
    try:
        report = await benchmark.run_full_benchmark()
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(report)
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())