#!/usr/bin/env python3
"""
Simple script to run the optimized S3Vector benchmark.

This script will:
1. Check if required services are running
2. Run a quick benchmark comparing engines
3. Generate a performance report

Requirements:
- MinIO running on localhost:9000 (or adjust MINIO_SERVER_URL)
- Redis running on localhost:6379
- PostgreSQL with pgvector on localhost:5432

Run with: python run_benchmark.py
"""

import asyncio
import sys
import os
import logging
import time
from pathlib import Path

# Add the workspace to Python path
workspace_path = Path(__file__).parent
sys.path.insert(0, str(workspace_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_services():
    """Check if required services are available."""
    logger.info("Checking required services...")
    
    # Check Redis
    try:
        import redis.asyncio as redis
        redis_client = redis.Redis.from_url("redis://127.0.0.1:6379/0")
        await redis_client.ping()
        await redis_client.close()
        logger.info("✓ Redis is available")
    except Exception as e:
        logger.error(f"✗ Redis not available: {e}")
        logger.error("Please start Redis: docker run -d -p 6379:6379 redis:alpine")
        return False
    
    # Check MinIO (S3)
    try:
        import aioboto3
        session = aioboto3.Session()
        async with session.client(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            region_name="us-east-1"
        ) as s3_client:
            await s3_client.list_buckets()
        logger.info("✓ MinIO (S3) is available")
    except Exception as e:
        logger.error(f"✗ MinIO not available: {e}")
        logger.error("Please start MinIO: docker run -d -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ':9001'")
        return False
    
    # Check PostgreSQL
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        db_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        await engine.dispose()
        logger.info("✓ PostgreSQL is available")
    except Exception as e:
        logger.error(f"✗ PostgreSQL not available: {e}")
        logger.error("Please start PostgreSQL with pgvector:")
        logger.error("docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16")
        return False
    
    return True


async def run_quick_benchmark():
    """Run a quick benchmark to compare engines."""
    logger.info("Starting quick performance benchmark...")
    
    try:
        # Import benchmark
        from scripts.optimized_vector_benchmark import OptimizedVectorBenchmark
        
        # Create benchmark instance
        benchmark = OptimizedVectorBenchmark()
        
        # Override test sizes for quick run
        original_run = benchmark.run_full_benchmark
        
        async def quick_run():
            logger.info("Running quick benchmark with smaller datasets...")
            
            try:
                # Test with smaller datasets for quick feedback
                test_sizes = [1000, 5000]  # Smaller datasets for quick test
                
                for size in test_sizes:
                    await benchmark.run_dataset_benchmark(size)
                    
                    # Clear Redis between tests
                    if benchmark.infra.redis_client:
                        await benchmark.infra.redis_client.flushall()
                
                # Generate report
                report = benchmark.generate_report()
                
                report_filename = f"quick_benchmark_report_{int(time.time())}.md"
                with open(report_filename, "w") as f:
                    f.write(report)
                
                logger.info(f"Quick benchmark complete! Report: {report_filename}")
                return report
                
            finally:
                await benchmark.infra.cleanup_infrastructure()
        
        # Run the quick benchmark
        return await quick_run()
        
    except ImportError as e:
        logger.error(f"Failed to import benchmark: {e}")
        logger.error("Make sure all dependencies are installed:")
        logger.error("pip install numpy redis asyncio-redis aioboto3 sqlalchemy asyncpg psutil tabulate lz4")
        return None
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return None


async def run_pytest_tests():
    """Run the pytest-based benchmark tests."""
    logger.info("Running pytest benchmark tests...")
    
    try:
        import subprocess
        
        # Run the benchmark tests
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_optimized_s3vector_benchmark.py",
            "-v", "-s", "--tb=short",
            "-k", "not test_large_dataset_performance"  # Skip slow tests by default
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=workspace_path)
        
        if result.returncode == 0:
            logger.info("✓ Pytest benchmark tests passed!")
            print("\nTest Output:")
            print(result.stdout)
        else:
            logger.error("✗ Pytest benchmark tests failed!")
            print("\nError Output:")
            print(result.stderr)
            print("\nStdout:")
            print(result.stdout)
            
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Failed to run pytest tests: {e}")
        return False


async def run_zarr_benchmark():
    """Run the Zarr comparison benchmark."""
    logger.info("Running Zarr benchmark comparison...")
    
    try:
        from scripts.zarr_benchmark_comparison import ZarrVectorBenchmark
        
        benchmark = ZarrVectorBenchmark()
        report = await benchmark.run_zarr_benchmark()
        
        if report:
            logger.info("✓ Zarr benchmark completed successfully!")
            print("\nZarr Benchmark Results:")
            # Print key insights about Zarr performance
            lines = report.split('\n')
            for line in lines:
                if 'Zarr' in line and ('faster' in line.lower() or 'compression' in line.lower()):
                    print(f"  • {line}")
            return True
        else:
            logger.error("✗ Zarr benchmark failed!")
            return False
            
    except ImportError as e:
        logger.error(f"Failed to import Zarr benchmark: {e}")
        logger.error("Make sure zarr and numcodecs are installed:")
        logger.error("pip install zarr>=2.16.0 numcodecs>=0.11.0")
        return False
    except Exception as e:
        logger.error(f"Zarr benchmark failed: {e}")
        return False


async def main():
    """Main function to run the benchmark."""
    print("="*60)
    print("OPTIMIZED S3VECTOR BENCHMARK RUNNER")
    print("="*60)
    
    # Check services
    if not await check_services():
        print("\n❌ Required services are not available. Please start them and try again.")
        return 1
    
    print("\n✅ All required services are available!")
    
    # Ask user what to run
    print("\nChoose benchmark type:")
    print("1. Quick benchmark (recommended for first run)")
    print("2. Pytest-based tests")
    print("3. Zarr benchmark (compare Zarr storage)")
    print("4. All benchmarks")
    
    try:
        choice = input("\nEnter choice (1-4, default=1): ").strip() or "1"
    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user.")
        return 0
    
    success = True
    
    if choice in ["1", "4"]:
        print("\n" + "="*50)
        print("RUNNING QUICK BENCHMARK")
        print("="*50)
        
        report = await run_quick_benchmark()
        if report:
            print("\n📊 BENCHMARK RESULTS SUMMARY:")
            print("-" * 40)
            # Print key results from report
            lines = report.split('\n')
            in_table = False
            for line in lines:
                if '|' in line and ('vec/s' in line or 'QPS' in line or 'Dataset Size' in line):
                    print(line)
                    in_table = True
                elif in_table and line.strip() == '':
                    break
        else:
            success = False
    
    if choice in ["2", "4"]:
        print("\n" + "="*50)
        print("RUNNING PYTEST TESTS")
        print("="*50)
        
        if not await run_pytest_tests():
            success = False
    
    if choice in ["3", "4"]:
        print("\n" + "="*50)
        print("RUNNING ZARR BENCHMARK")
        print("="*50)
        
        if not await run_zarr_benchmark():
            success = False
    
    if success:
        print("\n✅ Benchmark completed successfully!")
        print("\nKey Findings:")
        print("- Check the generated report files for detailed results")
        print("- Optimized S3Vector should show improved performance")
        print("- PgVector will be faster for small datasets")
        print("- S3Vector scales better for large datasets")
    else:
        print("\n❌ Benchmark completed with errors. Check logs above.")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)