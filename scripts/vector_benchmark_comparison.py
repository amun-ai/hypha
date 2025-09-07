#!/usr/bin/env python3
"""Comprehensive benchmark comparing S3Vector vs PgVector performance."""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os
from tabulate import tabulate
import redis.asyncio as redis

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypha.s3vector import create_s3_vector_engine_from_config
from hypha.pgvector import PgVectorSearchEngine
from sqlalchemy.ext.asyncio import create_async_engine


class VectorBenchmarkComparison:
    """Comprehensive benchmark comparing S3Vector vs PgVector."""
    
    def __init__(self):
        self.results = {}
        self.dimension = 384
        self.s3_engine = None
        self.pg_engine = None
    
    async def setup_engines(self):
        """Setup both S3Vector and PgVector engines."""
        print("Setting up engines...")
        
        # Create Redis client
        redis_client = redis.Redis.from_url("redis://127.0.0.1:6338/0")
        
        # S3Vector engine with performance improvements
        s3_config = {
            "bucket_name": "test-bucket",
            "redis_client": redis_client,  # Use real Redis client
            "endpoint_url": "http://localhost:9002",  # Local MinIO
            "access_key_id": "minioadmin",
            "secret_access_key": "minioadmin",
        }
        
        self.s3_engine = create_s3_vector_engine_from_config(s3_config)
        await self.s3_engine.initialize()
        
        # PgVector engine
        db_url = "postgresql+asyncpg://postgres:mysecretpassword@localhost:15433/benchmark"
        pg_sqlalchemy_engine = create_async_engine(db_url, echo=False)
        self.pg_engine = PgVectorSearchEngine(pg_sqlalchemy_engine, prefix="benchmark")
        
        print("✓ Engines setup complete\n")
    
    def generate_test_data(self, num_vectors: int) -> tuple[np.ndarray, List[Dict]]:
        """Generate test vectors and metadata."""
        np.random.seed(42)  # Reproducible results
        vectors = np.random.randn(num_vectors, self.dimension).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        metadata = [
            {
                "id": i,
                "category": f"cat_{i % 10}",
                "score": np.random.rand(),
                "text": f"Document {i} content"
            }
            for i in range(num_vectors)
        ]
        
        return vectors, metadata
    
    async def benchmark_insert_performance(self, vectors: np.ndarray, metadata: List[Dict], engine_name: str, engine, collection_name: str) -> Dict[str, float]:
        """Benchmark insert performance for an engine."""
        print(f"  Benchmarking {engine_name} insert performance...")
        
        # Create collection with optimal settings
        if engine_name == "S3Vector":
            optimal_centroids = self.s3_engine.calculate_optimal_centroids(len(vectors))
            hnsw_params = self.s3_engine.optimize_hnsw_params(optimal_centroids)
            await engine.create_collection(collection_name, {
                "dimension": self.dimension,
                "num_centroids": optimal_centroids,
            })
            print(f"    S3Vector using {optimal_centroids} centroids, HNSW m={hnsw_params['m']}, ef={hnsw_params['ef']}")
        else:
            await engine.create_collection(collection_name, dimension=self.dimension)
        
        # Insert in batches - much larger batches for 1M+ datasets
        if len(vectors) > 500000:
            batch_size = 20000  # 20K batch size for 1M vectors
        elif len(vectors) > 100000:
            batch_size = 5000
        else:
            batch_size = 500
        batch_times = []
        total_start = time.time()
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            start = time.time()
            if engine_name == "S3Vector":
                await engine.add_vectors(collection_name, batch_vectors.tolist(), batch_metadata)
            else:
                # PgVector expects list of dict with vector and metadata combined
                pg_vectors = []
                for j, (vector, meta) in enumerate(zip(batch_vectors, batch_metadata)):
                    pg_vectors.append({
                        "id": f"vec_{i+j}",
                        "vector": vector.tolist(),
                        **meta  # Include metadata fields
                    })
                await engine.add_vectors(collection_name, pg_vectors)
            batch_times.append(time.time() - start)
        
        total_time = time.time() - total_start
        throughput = len(vectors) / total_time
        
        return {
            "total_time": total_time,
            "throughput": throughput,
            "avg_batch_time": np.mean(batch_times),
            "batch_times": batch_times
        }
    
    async def benchmark_search_performance(self, vectors: np.ndarray, engine_name: str, engine, collection_name: str) -> Dict[str, Any]:
        """Benchmark search performance for an engine."""
        print(f"  Benchmarking {engine_name} search performance...")
        
        query_vector = vectors[0]
        # Reduce k_values for large datasets to prevent timeout
        if len(vectors) > 100000:
            k_values = [10]  # Only test k=10 for 1M+ vectors
        else:
            k_values = [10, 50, 100]
        search_results = {}
        
        for k in k_values:
            # Cold search (first query)
            start = time.time()
            if engine_name == "S3Vector":
                results = await engine.search_vectors(collection_name, query_vector, limit=k)
            else:
                results = await engine.search_vectors(collection_name, query_vector, limit=k)
            cold_time = (time.time() - start) * 1000
            
            # Warm searches (cache warmed up) - reduce iterations for large datasets
            num_warm_searches = 5 if len(vectors) > 100000 else 10
            warm_times = []
            for _ in range(num_warm_searches):
                start = time.time()
                if engine_name == "S3Vector":
                    results = await engine.search_vectors(collection_name, query_vector, limit=k)
                else:
                    results = await engine.search_vectors(collection_name, query_vector, limit=k)
                warm_times.append((time.time() - start) * 1000)
            
            # Concurrent searches (test parallel loading for S3Vector) - heavily reduce for large datasets
            if len(vectors) > 500000:
                num_concurrent = 5  # Only 5 concurrent for 1M vectors
            elif len(vectors) > 100000:
                num_concurrent = 10
            else:
                num_concurrent = 20
            async def search_task():
                query = vectors[np.random.randint(0, min(1000, len(vectors)))]
                start = time.time()
                if engine_name == "S3Vector":
                    await engine.search_vectors(collection_name, query, limit=k)
                else:
                    await engine.search_vectors(collection_name, query, limit=k)
                return (time.time() - start) * 1000
            
            concurrent_start = time.time()
            concurrent_results = await asyncio.gather(*[search_task() for _ in range(num_concurrent)])
            concurrent_total = time.time() - concurrent_start
            
            search_results[k] = {
                "cold_time": cold_time,
                "avg_warm_time": np.mean(warm_times),
                "std_warm_time": np.std(warm_times),
                "avg_concurrent_time": np.mean(concurrent_results),
                "concurrent_qps": num_concurrent / concurrent_total,
                "num_results": len(results) if results else 0
            }
        
        return search_results
    
    async def benchmark_dataset_size(self, num_vectors: int):
        """Benchmark both engines with a specific dataset size."""
        print(f"\n=== Benchmarking {num_vectors:,} vectors ===")
        
        vectors, metadata = self.generate_test_data(num_vectors)
        
        s3_collection = f"s3_test_{num_vectors}"
        pg_collection = f"pg_test_{num_vectors}"
        
        # Benchmark both engines
        engines = [
            ("S3Vector", self.s3_engine, s3_collection),
            ("PgVector", self.pg_engine, pg_collection)
        ]
        
        dataset_results = {"num_vectors": num_vectors}
        
        for engine_name, engine, collection_name in engines:
            try:
                # Clean up any existing collection first
                try:
                    await engine.delete_collection(collection_name)
                except:
                    pass  # Collection might not exist
                
                # Insert benchmark
                insert_results = await self.benchmark_insert_performance(
                    vectors, metadata, engine_name, engine, collection_name
                )
                
                # Search benchmark
                search_results = await self.benchmark_search_performance(
                    vectors, engine_name, engine, collection_name
                )
                
                dataset_results[engine_name] = {
                    "insert": insert_results,
                    "search": search_results
                }
                
                # Cleanup
                await engine.delete_collection(collection_name)
                
            except Exception as e:
                print(f"  ⚠️ {engine_name} failed: {e}")
                dataset_results[engine_name] = {"error": str(e)}
                # Try to cleanup on error
                try:
                    await engine.delete_collection(collection_name)
                except:
                    pass
        
        self.results[num_vectors] = dataset_results
    
    def format_results_table(self) -> str:
        """Format results into a comprehensive report."""
        report = "# Vector Search Engine Performance Comparison\n\n"
        report += f"**Test Configuration:**\n"
        report += f"- Vector Dimension: {self.dimension}\n"
        report += f"- Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"- S3Vector Improvements: Adaptive centroids, Parallel shard loading, HNSW tuning\n\n"
        
        # Insert Performance Summary
        report += "## Insert Performance Comparison\n\n"
        insert_headers = ["Dataset Size", "S3Vector (vec/s)", "PgVector (vec/s)", "S3Vector Time (s)", "PgVector Time (s)", "Winner"]
        insert_rows = []
        
        for num_vectors in sorted(self.results.keys()):
            data = self.results[num_vectors]
            s3_insert = data.get("S3Vector", {}).get("insert", {})
            pg_insert = data.get("PgVector", {}).get("insert", {})
            
            if s3_insert and pg_insert:
                s3_throughput = s3_insert.get("throughput", 0)
                pg_throughput = pg_insert.get("throughput", 0)
                s3_time = s3_insert.get("total_time", 0)
                pg_time = pg_insert.get("total_time", 0)
                winner = "S3Vector" if s3_throughput > pg_throughput else "PgVector"
                
                insert_rows.append([
                    f"{num_vectors:,}",
                    f"{s3_throughput:.0f}",
                    f"{pg_throughput:.0f}",
                    f"{s3_time:.2f}",
                    f"{pg_time:.2f}",
                    f"**{winner}**"
                ])
        
        report += tabulate(insert_rows, headers=insert_headers, tablefmt="pipe") + "\n\n"
        
        # Search Performance Summary
        for k in [10, 50, 100]:
            report += f"## Search Performance (k={k})\n\n"
            search_headers = ["Dataset Size", "S3Vector Cold (ms)", "PgVector Cold (ms)", "S3Vector Warm (ms)", "PgVector Warm (ms)", "S3Vector QPS", "PgVector QPS"]
            search_rows = []
            
            for num_vectors in sorted(self.results.keys()):
                data = self.results[num_vectors]
                s3_search = data.get("S3Vector", {}).get("search", {}).get(k, {})
                pg_search = data.get("PgVector", {}).get("search", {}).get(k, {})
                
                if s3_search and pg_search:
                    search_rows.append([
                        f"{num_vectors:,}",
                        f"{s3_search.get('cold_time', 0):.1f}",
                        f"{pg_search.get('cold_time', 0):.1f}",
                        f"{s3_search.get('avg_warm_time', 0):.1f}",
                        f"{pg_search.get('avg_warm_time', 0):.1f}",
                        f"{s3_search.get('concurrent_qps', 0):.1f}",
                        f"{pg_search.get('concurrent_qps', 0):.1f}"
                    ])
            
            report += tabulate(search_rows, headers=search_headers, tablefmt="pipe") + "\n\n"
        
        # Analysis and Conclusions
        report += "## Performance Analysis\n\n"
        report += "### S3Vector Advantages:\n"
        report += "- **Cost Efficiency**: S3 storage costs ~10x less than PostgreSQL\n"
        report += "- **Scalability**: Can handle billions of vectors\n"
        report += "- **Memory Efficiency**: Constant memory usage regardless of dataset size\n"
        report += "- **Elastic Scaling**: Scales to zero when idle\n\n"
        
        report += "### PgVector Advantages:\n"
        report += "- **Low Latency**: Better performance for small datasets and real-time queries\n"
        report += "- **ACID Compliance**: Full transactional consistency\n"
        report += "- **SQL Integration**: Native PostgreSQL queries and joins\n"
        report += "- **Mature Ecosystem**: Well-established tooling and monitoring\n\n"
        
        report += "### Recommendations:\n\n"
        report += "**Use S3Vector when:**\n"
        report += "- Dataset size > 1M vectors\n"
        report += "- Cost optimization is critical\n"
        report += "- Serverless/elastic scaling needed\n"
        report += "- Write-once, read-many patterns\n\n"
        
        report += "**Use PgVector when:**\n"
        report += "- Dataset size < 1M vectors\n"
        report += "- Sub-100ms latency required\n"
        report += "- Complex SQL queries with vectors\n"
        report += "- Strong consistency requirements\n"
        
        return report
    
    async def run_benchmark(self):
        """Run the complete benchmark suite."""
        await self.setup_engines()
        
        # Test different dataset sizes - optimized for large scale
        test_sizes = [1000, 10000, 1000000]  # Progressive scaling including 1M
        
        for size in test_sizes:
            await self.benchmark_dataset_size(size)
        
        # Generate and save report
        report = self.format_results_table()
        
        with open("vector_search_benchmark_report.md", "w") as f:
            f.write(report)
        
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE")
        print("Report saved to: vector_search_benchmark_report.md")
        print(f"{'='*60}")
        
        return report


async def main():
    """Main function to run the benchmark."""
    benchmark = VectorBenchmarkComparison()
    report = await benchmark.run_benchmark()
    print(report)


if __name__ == "__main__":
    asyncio.run(main())