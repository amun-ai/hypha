#!/usr/bin/env python3
"""Comprehensive benchmark comparing S3Vector (Zarr & Parquet) vs PgVector performance."""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os
from tabulate import tabulate
import redis.asyncio as redis
import shutil
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypha.s3vector import create_s3_vector_engine_from_config
from hypha.pgvector import PgVectorSearchEngine
from sqlalchemy.ext.asyncio import create_async_engine


class VectorBenchmarkComparison:
    """Comprehensive benchmark comparing S3Vector (Zarr & Parquet) vs PgVector."""
    
    def __init__(self):
        self.results = {}
        self.dimension = 384
        self.s3_zarr_engine = None
        self.s3_parquet_engine = None
        self.pg_engine = None
    
    async def setup_engines(self):
        """Setup S3Vector (both Zarr and Parquet) and PgVector engines."""
        print("Setting up engines...")
        
        # Create Redis client
        redis_client = redis.Redis.from_url("redis://127.0.0.1:6338/0")
        
        # S3Vector engine with Zarr (current implementation)
        s3_zarr_config = {
            "bucket_name": "test-bucket-zarr",
            "redis_client": redis_client,  # Use real Redis client
            "endpoint_url": "http://localhost:9002",  # Local MinIO
            "access_key_id": "minioadmin",
            "secret_access_key": "minioadmin",
        }
        
        self.s3_zarr_engine = create_s3_vector_engine_from_config(s3_zarr_config)
        await self.s3_zarr_engine.initialize()
        
        # For S3-Parquet, we'll need to temporarily switch implementations
        # Save current s3vector.py and restore parquet version for comparison
        print("Note: S3-Parquet comparison requires switching implementation")
        self.s3_parquet_engine = None  # Will be set if Parquet version is available
        
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
        if engine_name.startswith("S3"):
            optimal_centroids = engine.calculate_optimal_centroids(len(vectors))
            hnsw_params = engine.optimize_hnsw_params(optimal_centroids)
            await engine.create_collection(collection_name, {
                "dimension": self.dimension,
                "num_centroids": optimal_centroids,
            })
            print(f"    {engine_name} using {optimal_centroids} centroids, HNSW m={hnsw_params['m']}, ef={hnsw_params['ef']}")
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
            if engine_name.startswith("S3"):
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
            results = await engine.search_vectors(collection_name, query_vector, limit=k)
            cold_time = (time.time() - start) * 1000
            
            # Warm searches (cache warmed up) - reduce iterations for large datasets
            num_warm_searches = 5 if len(vectors) > 100000 else 10
            warm_times = []
            for _ in range(num_warm_searches):
                start = time.time()
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
        """Benchmark all engines with a specific dataset size."""
        print(f"\n=== Benchmarking {num_vectors:,} vectors ===")
        
        vectors, metadata = self.generate_test_data(num_vectors)
        
        s3_zarr_collection = f"s3_zarr_test_{num_vectors}"
        s3_parquet_collection = f"s3_parquet_test_{num_vectors}"
        pg_collection = f"pg_test_{num_vectors}"
        
        # Benchmark all engines
        engines = [
            ("S3-Zarr", self.s3_zarr_engine, s3_zarr_collection),
            ("PgVector", self.pg_engine, pg_collection)
        ]
        
        # Add S3-Parquet if available
        if self.s3_parquet_engine:
            engines.append(("S3-Parquet", self.s3_parquet_engine, s3_parquet_collection))
        
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
        report += f"- S3Vector Zarr: Using Zarr format with Blosc LZ4 compression\n"
        report += f"- S3Vector Parquet: Using Parquet format (if available)\n"
        report += f"- Optimizations: Adaptive centroids, Parallel shard loading, HNSW tuning\n\n"
        
        # Insert Performance Summary
        report += "## Insert Performance Comparison\n\n"
        insert_headers = ["Dataset Size", "S3-Zarr (vec/s)", "S3-Parquet (vec/s)", "PgVector (vec/s)", "S3-Zarr Time (s)", "S3-Parquet Time (s)", "PgVector Time (s)"]
        insert_rows = []
        
        for num_vectors in sorted(self.results.keys()):
            data = self.results[num_vectors]
            s3_zarr_insert = data.get("S3-Zarr", {}).get("insert", {})
            s3_parquet_insert = data.get("S3-Parquet", {}).get("insert", {})
            pg_insert = data.get("PgVector", {}).get("insert", {})
            
            row = [f"{num_vectors:,}"]
            
            # Add throughput data
            row.append(f"{s3_zarr_insert.get('throughput', 0):.0f}" if s3_zarr_insert else "N/A")
            row.append(f"{s3_parquet_insert.get('throughput', 0):.0f}" if s3_parquet_insert else "N/A")
            row.append(f"{pg_insert.get('throughput', 0):.0f}" if pg_insert else "N/A")
            
            # Add time data
            row.append(f"{s3_zarr_insert.get('total_time', 0):.2f}" if s3_zarr_insert else "N/A")
            row.append(f"{s3_parquet_insert.get('total_time', 0):.2f}" if s3_parquet_insert else "N/A")
            row.append(f"{pg_insert.get('total_time', 0):.2f}" if pg_insert else "N/A")
            
            insert_rows.append(row)
        
        report += tabulate(insert_rows, headers=insert_headers, tablefmt="pipe") + "\n\n"
        
        # Search Performance Summary
        for k in [10, 50, 100]:
            report += f"## Search Performance (k={k})\n\n"
            search_headers = ["Dataset Size", "S3-Zarr Cold (ms)", "S3-Parquet Cold (ms)", "PgVector Cold (ms)", "S3-Zarr Warm (ms)", "S3-Parquet Warm (ms)", "PgVector Warm (ms)", "S3-Zarr QPS", "S3-Parquet QPS", "PgVector QPS"]
            search_rows = []
            
            for num_vectors in sorted(self.results.keys()):
                data = self.results[num_vectors]
                s3_zarr_search = data.get("S3-Zarr", {}).get("search", {}).get(k, {})
                s3_parquet_search = data.get("S3-Parquet", {}).get("search", {}).get(k, {})
                pg_search = data.get("PgVector", {}).get("search", {}).get(k, {})
                
                if s3_zarr_search or s3_parquet_search or pg_search:
                    row = [f"{num_vectors:,}"]
                    
                    # Cold times
                    row.append(f"{s3_zarr_search.get('cold_time', 0):.1f}" if s3_zarr_search else "N/A")
                    row.append(f"{s3_parquet_search.get('cold_time', 0):.1f}" if s3_parquet_search else "N/A")
                    row.append(f"{pg_search.get('cold_time', 0):.1f}" if pg_search else "N/A")
                    
                    # Warm times
                    row.append(f"{s3_zarr_search.get('avg_warm_time', 0):.1f}" if s3_zarr_search else "N/A")
                    row.append(f"{s3_parquet_search.get('avg_warm_time', 0):.1f}" if s3_parquet_search else "N/A")
                    row.append(f"{pg_search.get('avg_warm_time', 0):.1f}" if pg_search else "N/A")
                    
                    # QPS
                    row.append(f"{s3_zarr_search.get('concurrent_qps', 0):.1f}" if s3_zarr_search else "N/A")
                    row.append(f"{s3_parquet_search.get('concurrent_qps', 0):.1f}" if s3_parquet_search else "N/A")
                    row.append(f"{pg_search.get('concurrent_qps', 0):.1f}" if pg_search else "N/A")
                    
                    search_rows.append(row)
            
            if search_rows:  # Only add table if there's data
                report += tabulate(search_rows, headers=search_headers, tablefmt="pipe") + "\n\n"
        
        # Analysis and Conclusions
        report += "## Performance Analysis\n\n"
        
        report += "### S3-Zarr Advantages:\n"
        report += "- **Smaller Chunks**: Zarr uses configurable chunk sizes (1MB target) for faster partial reads\n"
        report += "- **Better Compression**: Blosc LZ4 provides fast compression/decompression\n"
        report += "- **Parallel Access**: Zarr format designed for concurrent chunk access\n"
        report += "- **Flexible Schema**: Easy to add metadata and extend arrays\n\n"
        
        report += "### S3-Parquet Advantages:\n"
        report += "- **Columnar Format**: Efficient for batch operations\n"
        report += "- **Wide Ecosystem**: Supported by many data processing tools\n"
        report += "- **Predicate Pushdown**: Can filter data at storage level\n\n"
        
        report += "### S3Vector (Both Formats) Advantages:\n"
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
        report += "**Use S3-Zarr when:**\n"
        report += "- Need fastest cold start times\n"
        report += "- Frequent partial reads of vectors\n"
        report += "- Want better compression ratios\n"
        report += "- Building streaming/online systems\n\n"
        
        report += "**Use S3-Parquet when:**\n"
        report += "- Integrating with data lake tools\n"
        report += "- Batch processing workflows\n"
        report += "- Need wide tool compatibility\n\n"
        
        report += "**Use PgVector when:**\n"
        report += "- Dataset size < 1M vectors\n"
        report += "- Sub-100ms latency required\n"
        report += "- Complex SQL queries with vectors\n"
        report += "- Strong consistency requirements\n"
        
        return report
    
    async def run_benchmark(self):
        """Run the complete benchmark suite."""
        await self.setup_engines()
        
        # Test different dataset sizes - start with smaller sizes for quick testing
        test_sizes = [1000, 10000, 50000]  # Progressive scaling
        
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