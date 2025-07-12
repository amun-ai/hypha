#!/usr/bin/env python3
"""
Comprehensive Hypha Server Stress Test Suite
Consolidates all stress testing functionality into one comprehensive test file.

Usage:
    # Run all tests
    pytest tests/hypha_stress_test_suite.py -v -s

    # Run specific test
    pytest tests/hypha_stress_test_suite.py::test_concurrent_clients_basic -v -s

    # Run standalone extreme load test
    python tests/hypha_stress_test_suite.py --extreme-load

    # Run performance benchmark
    python tests/hypha_stress_test_suite.py --benchmark
"""

import asyncio
import argparse
import gc
import json
import logging
import numpy as np
import os
import psutil
import pytest
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import threading
import websockets
from datetime import datetime

from hypha_rpc import connect_to_server
from hypha.core import UserInfo
from hypha.core.store import RedisStore
from hypha.utils import random_id

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test configuration with environment variable support
STRESS_TEST_CONFIG = {
    "max_concurrent_clients": int(
        os.environ.get("STRESS_MAX_CONCURRENT_CLIENTS", "50")
    ),
    "large_array_sizes": [
        int(x)
        for x in os.environ.get("STRESS_LARGE_ARRAY_SIZES", "1024,10240,102400").split(
            ","
        )
    ],
    "test_duration": int(os.environ.get("STRESS_TEST_DURATION", "30")),
    "memory_check_interval": int(os.environ.get("STRESS_MEMORY_CHECK_INTERVAL", "1")),
    "websocket_timeout": int(os.environ.get("STRESS_WEBSOCKET_TIMEOUT", "10")),
    "extreme_load_max_clients": int(
        os.environ.get("STRESS_EXTREME_MAX_CLIENTS", "10000")
    ),
    "extreme_load_batch_size": int(os.environ.get("STRESS_EXTREME_BATCH_SIZE", "100")),
}

# Default server URL
SERVER_URL = "ws://127.0.0.1:38283/ws"


class SystemMonitor:
    """Monitor system resources during stress testing."""

    def __init__(self, interval=1):
        self.interval = interval
        self.process = psutil.Process()
        self.measurements = []
        self.monitoring = False
        self._monitor_task = None
        self.start_memory = 0
        self.peak_memory = 0

    async def start(self):
        """Start continuous monitoring."""
        self.monitoring = True
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.measurements = []
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Monitor system resources in a loop."""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()

                # Get system-wide info
                system_memory = psutil.virtual_memory()

                measurement = {
                    "timestamp": time.time(),
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "memory_percent": memory_info.rss / system_memory.total * 100,
                    "cpu_percent": cpu_percent,
                    "system_memory_available_gb": system_memory.available
                    / 1024
                    / 1024
                    / 1024,
                    "open_files": len(self.process.open_files()),
                    "num_threads": self.process.num_threads(),
                }

                self.measurements.append(measurement)
                self.peak_memory = max(self.peak_memory, measurement["memory_mb"])
                await asyncio.sleep(self.interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")

    def get_current_stats(self):
        """Get current resource usage."""
        if self.measurements:
            return self.measurements[-1]
        return {}

    def get_peak_stats(self):
        """Get peak resource usage."""
        if not self.measurements:
            return {}

        return {
            "start_memory_mb": self.start_memory,
            "peak_memory_mb": self.peak_memory,
            "final_memory_mb": (
                self.measurements[-1]["memory_mb"] if self.measurements else 0
            ),
            "memory_increase_mb": self.peak_memory - self.start_memory,
            "avg_memory_mb": sum(m["memory_mb"] for m in self.measurements)
            / len(self.measurements),
            "peak_cpu_percent": max(m["cpu_percent"] for m in self.measurements),
            "peak_open_files": max(m["open_files"] for m in self.measurements),
            "peak_threads": max(m["num_threads"] for m in self.measurements),
            "sample_count": len(self.measurements),
        }


class StressTestClient:
    """A comprehensive client for stress testing."""

    def __init__(
        self, client_id: str, server_url: str = SERVER_URL, token: Optional[str] = None
    ):
        self.client_id = client_id
        self.server_url = server_url
        self.token = token
        self.api = None
        self.connected = False
        self.operations_count = 0
        self.errors = []
        self.stats = {
            "connection_time": 0,
            "operation_times": [],
            "data_transfer_speeds": [],
        }

    async def connect(self):
        """Connect to the server."""
        start_time = time.time()
        try:
            self.api = await connect_to_server(
                {
                    "client_id": f"stress-test-{self.client_id}",
                    "name": f"stress-test-{self.client_id}",
                    "server_url": self.server_url,
                    "token": self.token,
                }
            )
            self.connected = True
            self.stats["connection_time"] = time.time() - start_time
            logger.debug(
                f"Client {self.client_id} connected in {self.stats['connection_time']:.3f}s"
            )
        except Exception as e:
            self.errors.append(f"Connection error: {e}")
            logger.error(f"Client {self.client_id} failed to connect: {e}")

    async def disconnect(self):
        """Disconnect from the server."""
        if self.api:
            try:
                await self.api.disconnect()
                self.connected = False
                logger.debug(f"Client {self.client_id} disconnected")
            except Exception as e:
                self.errors.append(f"Disconnection error: {e}")
                logger.error(f"Client {self.client_id} failed to disconnect: {e}")

    async def ping_test(self, count: int = 10):
        """Perform ping test."""
        if not self.connected:
            return

        for i in range(count):
            try:
                start_time = time.time()
                result = await self.api.echo(f"ping-{i}")
                operation_time = time.time() - start_time

                if result != f"ping-{i}":
                    self.errors.append(
                        f"Echo mismatch: expected 'ping-{i}', got '{result}'"
                    )

                self.stats["operation_times"].append(operation_time)
                self.operations_count += 1

            except Exception as e:
                self.errors.append(f"Ping error: {e}")

    async def large_data_test(self, array_size: int):
        """Test with large numpy arrays."""
        if not self.connected:
            return

        try:
            # Create large numpy array
            test_array = np.random.rand(array_size).astype(np.float32)

            # Send via echo
            start_time = time.time()
            result = await self.api.echo(test_array)
            transfer_time = time.time() - start_time

            # Verify result
            if not np.array_equal(test_array, result):
                self.errors.append(f"Array mismatch for size {array_size}")
            else:
                data_size_mb = test_array.nbytes / 1024 / 1024
                throughput_mbps = data_size_mb / transfer_time
                self.stats["data_transfer_speeds"].append(throughput_mbps)
                self.stats["operation_times"].append(transfer_time)
                logger.info(
                    f"Client {self.client_id}: {data_size_mb:.2f}MB in {transfer_time:.3f}s ({throughput_mbps:.2f}MB/s)"
                )

            self.operations_count += 1

        except Exception as e:
            self.errors.append(f"Large data test error: {e}")

    async def service_registration_test(self, count: int = 5):
        """Test service registration/unregistration."""
        if not self.connected:
            return

        services = []
        try:
            # Register services
            for i in range(count):
                service_id = f"test-service-{self.client_id}-{i}"
                start_time = time.time()
                service_info = await self.api.register_service(
                    {
                        "id": service_id,
                        "name": f"Test Service {i}",
                        "echo": lambda x: x,
                    }
                )
                self.stats["operation_times"].append(time.time() - start_time)
                services.append(service_info)
                self.operations_count += 1

            # Unregister services
            for service in services:
                start_time = time.time()
                await self.api.unregister_service(service["id"])
                self.stats["operation_times"].append(time.time() - start_time)
                self.operations_count += 1

        except Exception as e:
            self.errors.append(f"Service registration error: {e}")

    def get_stats(self):
        """Get performance statistics."""
        stats = self.stats.copy()
        if self.stats["operation_times"]:
            stats["avg_operation_time"] = statistics.mean(self.stats["operation_times"])
            stats["min_operation_time"] = min(self.stats["operation_times"])
            stats["max_operation_time"] = max(self.stats["operation_times"])
        if self.stats["data_transfer_speeds"]:
            stats["avg_transfer_speed"] = statistics.mean(
                self.stats["data_transfer_speeds"]
            )
            stats["max_transfer_speed"] = max(self.stats["data_transfer_speeds"])
        return stats


class ExtremeLoadTester:
    """Extreme load tester for finding breaking points."""

    def __init__(self, server_url: str = SERVER_URL, token: Optional[str] = None):
        self.server_url = server_url
        self.token = token
        self.monitor = SystemMonitor()
        self.all_clients = []
        self.breaking_point = None
        self.results = []

    async def create_client_batch(
        self, batch_id: int, batch_size: int, max_concurrent: int = 50
    ):
        """Create a batch of clients with controlled concurrency."""
        logger.info(f"Creating batch {batch_id} with {batch_size} clients...")

        clients = []
        errors = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def create_single_client(client_id):
            async with semaphore:
                try:
                    client = StressTestClient(
                        f"extreme-{batch_id}-{client_id}", self.server_url, self.token
                    )
                    await client.connect()
                    return client
                except Exception as e:
                    return Exception(f"Client {batch_id}-{client_id}: {str(e)}")

        # Create all clients concurrently but with limited concurrency
        tasks = [create_single_client(i) for i in range(batch_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful clients from errors
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            else:
                clients.append(result)

        # Test operations with connected clients
        operation_success = 0
        if clients:
            test_clients = clients[: min(5, len(clients))]
            try:
                tasks = [client.ping_test(1) for client in test_clients]
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=30
                )
                operation_success = len(test_clients)
            except Exception as e:
                errors.append(f"Operations failed: {str(e)}")

        logger.info(
            f"Batch {batch_id}: {len(clients)}/{batch_size} clients created, {operation_success} operations successful"
        )
        return clients, errors

    async def run_extreme_test(
        self, start_clients: int = 1000, max_clients: int = 10000, step_size: int = 1000
    ):
        """Run extreme load test to find breaking point."""
        print(f"🚀 EXTREME LOAD TEST: {start_clients} to {max_clients} clients")

        # Start monitoring
        monitor_task = asyncio.create_task(self.monitor.start())

        try:
            current_clients = 0

            for target in range(start_clients, max_clients + 1, step_size):
                print(f"\n🎯 TARGET: {target} concurrent clients")

                # Calculate how many new clients we need
                new_clients_needed = target - current_clients
                if new_clients_needed <= 0:
                    continue

                start_time = time.time()

                # Create new clients in batches
                batch_size = min(
                    STRESS_TEST_CONFIG["extreme_load_batch_size"], new_clients_needed
                )
                num_batches = (new_clients_needed + batch_size - 1) // batch_size

                all_new_clients = []
                total_errors = []

                for batch_id in range(num_batches):
                    actual_batch_size = min(
                        batch_size, new_clients_needed - batch_id * batch_size
                    )
                    if actual_batch_size <= 0:
                        break

                    batch_clients, batch_errors = await self.create_client_batch(
                        batch_id, actual_batch_size, max_concurrent=100
                    )
                    all_new_clients.extend(batch_clients)
                    total_errors.extend(batch_errors)

                    await asyncio.sleep(0.1)  # Brief pause between batches

                self.all_clients.extend(all_new_clients)
                current_clients = len(self.all_clients)
                connection_time = time.time() - start_time

                # Get current system stats
                current_stats = self.monitor.get_current_stats()
                success_rate = current_clients / target if target > 0 else 0

                result = {
                    "target": target,
                    "connected": current_clients,
                    "success_rate": success_rate,
                    "connection_time": connection_time,
                    "total_errors": len(total_errors),
                    "memory_mb": current_stats.get("memory_mb", 0),
                    "cpu_percent": current_stats.get("cpu_percent", 0),
                    "open_files": current_stats.get("open_files", 0),
                    "timestamp": time.time(),
                }

                self.results.append(result)

                print(f"📊 RESULTS:")
                print(
                    f"   ✅ Connected: {current_clients}/{target} ({success_rate:.1%})"
                )
                print(f"   ⏱️  Time: {connection_time:.1f}s")
                print(f"   ❌ Errors: {len(total_errors)}")
                print(f"   💾 Memory: {current_stats.get('memory_mb', 0):.1f}MB")
                print(f"   🔧 CPU: {current_stats.get('cpu_percent', 0):.1f}%")

                # Check for breaking conditions
                if success_rate < 0.3:
                    self.breaking_point = (
                        f"Low success rate ({success_rate:.1%}) at {target} clients"
                    )
                    print(f"❌ BREAKING POINT: {self.breaking_point}")
                    break

                if current_stats.get("memory_mb", 0) > 12000:  # > 12GB
                    self.breaking_point = f"Memory limit ({current_stats.get('memory_mb', 0):.1f}MB) at {target} clients"
                    print(f"❌ MEMORY LIMIT: {self.breaking_point}")
                    break

                # Brief pause and garbage collection
                await asyncio.sleep(2)
                gc.collect()

        except Exception as e:
            print(f"❌ CRASH: {str(e)}")
            self.breaking_point = (
                f"Server crash with {current_clients} clients: {str(e)}"
            )

        finally:
            # Stop monitoring
            self.monitor.stop()
            monitor_task.cancel()

            # Clean up clients
            await self.cleanup_clients()

        return self.results, self.breaking_point

    async def cleanup_clients(self):
        """Clean up all clients."""
        if not self.all_clients:
            return

        print(f"🧹 Cleaning up {len(self.all_clients)} clients...")

        batch_size = 100
        for i in range(0, len(self.all_clients), batch_size):
            batch = self.all_clients[i : i + batch_size]

            cleanup_tasks = []
            for client in batch:
                try:
                    cleanup_tasks.append(client.disconnect())
                except Exception:
                    pass

            if cleanup_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*cleanup_tasks, return_exceptions=True),
                        timeout=30,
                    )
                except Exception:
                    pass

            await asyncio.sleep(0.1)


class PerformanceBenchmarker:
    """Performance benchmarking suite."""

    def __init__(self, server_url: str = SERVER_URL, token: Optional[str] = None):
        self.server_url = server_url
        self.token = token
        self.results = {}

    async def run_benchmark(self):
        """Run comprehensive performance benchmark."""
        print("🏆 PERFORMANCE BENCHMARK SUITE")

        # Test 1: Concurrent clients
        print("\n📊 Testing concurrent client performance...")
        self.results["concurrent_clients"] = await self.test_concurrent_clients()

        # Test 2: Throughput and latency
        print("\n🚀 Testing throughput and latency...")
        self.results["throughput_latency"] = await self.test_throughput_latency()

        # Test 3: Redis scaling simulation
        print("\n🔗 Testing Redis scaling simulation...")
        self.results["redis_scaling"] = await self.test_redis_scaling()

        return self.results

    async def test_concurrent_clients(self, max_clients: int = 100, step: int = 10):
        """Test concurrent client performance."""
        results = []

        for num_clients in range(step, max_clients + 1, step):
            logger.info(f"Testing {num_clients} concurrent clients...")

            clients = []
            connected_count = 0
            start_time = time.time()

            # Create clients in batches
            batch_size = 5
            for i in range(0, num_clients, batch_size):
                batch_clients = []
                for j in range(min(batch_size, num_clients - i)):
                    client = StressTestClient(
                        f"bench-{i+j}", self.server_url, self.token
                    )
                    await client.connect()
                    if client.connected:
                        batch_clients.append(client)
                        connected_count += 1

                clients.extend(batch_clients)
                await asyncio.sleep(0.1)

            connection_time = time.time() - start_time
            success_rate = connected_count / num_clients

            # Test operations
            operation_success = 0
            if connected_count > 0:
                test_clients = clients[: min(5, len(clients))]
                try:
                    tasks = [client.ping_test(1) for client in test_clients]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    operation_success = len(test_clients)
                except Exception as e:
                    logger.error(f"Operation test failed: {e}")

            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

            result = {
                "num_clients": num_clients,
                "connected": connected_count,
                "success_rate": success_rate,
                "connection_time": connection_time,
                "operations_successful": operation_success,
                "memory_mb": memory_mb,
            }

            results.append(result)
            logger.info(
                f"📊 {num_clients} clients: {connected_count} connected ({success_rate:.1%}), "
                f"time: {connection_time:.2f}s, memory: {memory_mb:.1f}MB"
            )

            # Cleanup
            for client in clients:
                await client.disconnect()

            await asyncio.sleep(1)

            # Stop if success rate drops below 80%
            if success_rate < 0.8:
                logger.info(f"Success rate dropped to {success_rate:.1%}, stopping")
                break

        return results

    async def test_throughput_latency(
        self, data_sizes: List[int] = None, clients_count: int = 10
    ):
        """Test throughput and latency with different data sizes."""
        if data_sizes is None:
            data_sizes = STRESS_TEST_CONFIG["large_array_sizes"]

        results = []

        for data_size in data_sizes:
            logger.info(f"Testing with {data_size} element arrays...")

            # Create test data
            test_data = np.random.rand(data_size).astype(np.float32)
            data_size_mb = test_data.nbytes / 1024 / 1024

            # Connect clients
            clients = []
            for i in range(clients_count):
                client = StressTestClient(
                    f"throughput-{i}", self.server_url, self.token
                )
                await client.connect()
                if client.connected:
                    clients.append(client)

            if not clients:
                logger.error("No clients connected for throughput test")
                continue

            # Test latency (sequential)
            latencies = []
            for _ in range(5):
                start = time.time()
                try:
                    await clients[0].api.echo(test_data)
                    latency = time.time() - start
                    latencies.append(latency)
                except Exception as e:
                    logger.error(f"Latency test failed: {e}")

            avg_latency = statistics.mean(latencies) if latencies else 0

            # Test throughput (concurrent)
            start_time = time.time()
            tasks = [client.large_data_test(data_size) for client in clients]
            await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            successful_ops = sum(1 for client in clients if client.operations_count > 0)
            total_data_mb = data_size_mb * successful_ops
            throughput_mbps = total_data_mb / total_time if total_time > 0 else 0
            ops_per_second = successful_ops / total_time if total_time > 0 else 0

            result = {
                "data_size_elements": data_size,
                "data_size_mb": data_size_mb,
                "clients": len(clients),
                "avg_latency_ms": avg_latency * 1000,
                "throughput_mbps": throughput_mbps,
                "ops_per_second": ops_per_second,
                "successful_ops": successful_ops,
                "total_time": total_time,
            }

            results.append(result)
            logger.info(
                f"📈 {data_size} elements: {avg_latency*1000:.1f}ms latency, "
                f"{throughput_mbps:.1f}MB/s throughput, {ops_per_second:.1f} ops/s"
            )

            # Cleanup
            for client in clients:
                await client.disconnect()

            await asyncio.sleep(1)

        return results

    async def test_redis_scaling(
        self, max_servers: int = 5, clients_per_server: int = 20
    ):
        """Test Redis scaling with simulated multiple servers."""
        results = []

        for server_count in range(1, max_servers + 1):
            logger.info(
                f"Simulating {server_count} servers with {clients_per_server} clients each..."
            )

            all_clients = []
            start_time = time.time()

            # Create clients simulating multiple servers
            for server_id in range(server_count):
                for client_id in range(clients_per_server):
                    client = StressTestClient(
                        f"server-{server_id}-client-{client_id}",
                        self.server_url,
                        self.token,
                    )
                    await client.connect()
                    if client.connected:
                        all_clients.append(client)

                await asyncio.sleep(0.1)

            connection_time = time.time() - start_time
            total_connected = len(all_clients)
            expected_clients = server_count * clients_per_server
            success_rate = total_connected / expected_clients

            # Test cross-server operations
            operation_start = time.time()
            if len(all_clients) >= 2:
                tasks = [
                    client.ping_test(1)
                    for client in all_clients[: min(10, len(all_clients))]
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            operation_time = time.time() - operation_start
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

            result = {
                "simulated_servers": server_count,
                "clients_per_server": clients_per_server,
                "total_expected": expected_clients,
                "total_connected": total_connected,
                "connection_success_rate": success_rate,
                "connection_time": connection_time,
                "operation_time": operation_time,
                "memory_mb": memory_mb,
            }

            results.append(result)
            logger.info(
                f"🖥️  {server_count} servers: {total_connected}/{expected_clients} clients "
                f"({success_rate:.1%}), {connection_time:.2f}s, {memory_mb:.1f}MB"
            )

            # Cleanup
            for client in all_clients:
                await client.disconnect()

            await asyncio.sleep(1)

            # Stop if success rate drops significantly
            if success_rate < 0.7:
                logger.info(
                    f"Success rate dropped to {success_rate:.1%}, stopping scaling test"
                )
                break

        return results

    def print_summary(self):
        """Print benchmark summary."""
        print("\n🏆 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 50)

        if "concurrent_clients" in self.results:
            print("\n📊 CONCURRENT CLIENTS:")
            for result in self.results["concurrent_clients"]:
                print(
                    f"  {result['num_clients']} clients: {result['connected']} connected "
                    f"({result['success_rate']:.1%}), {result['connection_time']:.2f}s, "
                    f"{result['memory_mb']:.1f}MB"
                )

        if "throughput_latency" in self.results:
            print("\n🚀 THROUGHPUT & LATENCY:")
            for result in self.results["throughput_latency"]:
                print(
                    f"  {result['data_size_elements']} elements: "
                    f"{result['avg_latency_ms']:.1f}ms latency, "
                    f"{result['throughput_mbps']:.1f}MB/s throughput"
                )

        if "redis_scaling" in self.results:
            print("\n🔗 REDIS SCALING:")
            for result in self.results["redis_scaling"]:
                print(
                    f"  {result['simulated_servers']} servers: "
                    f"{result['total_connected']}/{result['total_expected']} clients "
                    f"({result['connection_success_rate']:.1%})"
                )

        print("\n✅ Benchmark completed")

    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            filename = (
                f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"📄 Results saved to: {filename}")
        return filename


# ==================== PYTEST TEST FUNCTIONS ====================


@pytest.mark.asyncio
async def test_concurrent_clients_basic(fastapi_server, test_user_token):
    """Test basic operations with concurrent clients."""
    from tests import SERVER_URL

    monitor = SystemMonitor()
    await monitor.start()

    try:
        num_clients = min(STRESS_TEST_CONFIG["max_concurrent_clients"], 20)
        clients = []

        # Create clients
        logger.info(f"Creating {num_clients} concurrent clients...")
        for i in range(num_clients):
            client = StressTestClient(f"client-{i}", SERVER_URL, test_user_token)
            clients.append(client)

        # Connect clients in batches
        batch_size = 5
        for i in range(0, len(clients), batch_size):
            batch = clients[i : i + batch_size]
            await asyncio.gather(*[client.connect() for client in batch])
            await asyncio.sleep(0.1)

        # Count successful connections
        connected_clients = [c for c in clients if c.connected]
        logger.info(
            f"Successfully connected {len(connected_clients)}/{num_clients} clients"
        )

        # Perform ping tests
        logger.info("Starting ping tests...")
        await asyncio.gather(*[client.ping_test(5) for client in connected_clients])

        # Collect results
        total_operations = sum(c.operations_count for c in clients)
        total_errors = sum(len(c.errors) for c in clients)

        logger.info(
            f"Completed {total_operations} operations with {total_errors} errors"
        )

        # Verify results
        assert (
            len(connected_clients) >= num_clients * 0.8
        ), "Too many connection failures"
        assert total_errors < total_operations * 0.1, "Too many operation errors"

    finally:
        # Clean up
        await monitor.stop()
        for client in clients:
            await client.disconnect()


@pytest.mark.asyncio
async def test_large_data_transmission(fastapi_server, test_user_token):
    """Test large data transmission performance."""
    from tests import SERVER_URL

    monitor = SystemMonitor()
    await monitor.start()

    try:
        array_sizes = STRESS_TEST_CONFIG["large_array_sizes"]
        num_clients = 3

        for array_size in array_sizes:
            logger.info(
                f"Testing {array_size} element arrays with {num_clients} clients..."
            )

            clients = []
            for i in range(num_clients):
                client = StressTestClient(
                    f"large-data-{i}", SERVER_URL, test_user_token
                )
                await client.connect()
                if client.connected:
                    clients.append(client)

            if not clients:
                logger.error(f"No clients connected for array size {array_size}")
                continue

            # Test concurrent large data transmission
            await asyncio.gather(
                *[client.large_data_test(array_size) for client in clients]
            )

            # Verify results
            successful_transfers = sum(
                1 for client in clients if client.operations_count > 0
            )
            total_errors = sum(len(client.errors) for client in clients)

            logger.info(
                f"Array size {array_size}: {successful_transfers}/{len(clients)} successful transfers, "
                f"{total_errors} errors"
            )

            assert (
                successful_transfers >= len(clients) * 0.8
            ), f"Too many failures for array size {array_size}"

            # Clean up
            for client in clients:
                await client.disconnect()

            await asyncio.sleep(1)

    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_service_registration_stress(fastapi_server, test_user_token):
    """Test service registration under stress."""
    from tests import SERVER_URL

    monitor = SystemMonitor()
    await monitor.start()

    try:
        num_clients = 10
        services_per_client = 5

        clients = []
        for i in range(num_clients):
            client = StressTestClient(f"service-test-{i}", SERVER_URL, test_user_token)
            await client.connect()
            if client.connected:
                clients.append(client)

        logger.info(
            f"Testing service registration with {len(clients)} clients, "
            f"{services_per_client} services each..."
        )

        # Perform service registration tests
        await asyncio.gather(
            *[
                client.service_registration_test(services_per_client)
                for client in clients
            ]
        )

        # Verify results
        total_operations = sum(c.operations_count for c in clients)
        total_errors = sum(len(c.errors) for c in clients)
        expected_operations = (
            len(clients) * services_per_client * 2
        )  # register + unregister

        logger.info(
            f"Service registration: {total_operations}/{expected_operations} operations, "
            f"{total_errors} errors"
        )

        assert (
            total_operations >= expected_operations * 0.8
        ), "Too many service operation failures"
        assert (
            total_errors < total_operations * 0.1
        ), "Too many service operation errors"

    finally:
        await monitor.stop()
        for client in clients:
            await client.disconnect()


@pytest.mark.asyncio
async def test_memory_leak_detection(fastapi_server, test_user_token):
    """Test for memory leaks during repeated operations."""
    from tests import SERVER_URL

    monitor = SystemMonitor()
    await monitor.start()

    try:
        iterations = 5
        clients_per_iteration = 10

        logger.info(
            f"Testing memory leaks over {iterations} iterations with "
            f"{clients_per_iteration} clients each..."
        )

        for iteration in range(iterations):
            logger.info(f"Iteration {iteration + 1}/{iterations}")

            clients = []
            for i in range(clients_per_iteration):
                client = StressTestClient(
                    f"leak-test-{iteration}-{i}", SERVER_URL, test_user_token
                )
                await client.connect()
                if client.connected:
                    clients.append(client)

            # Perform operations
            await asyncio.gather(*[client.ping_test(10) for client in clients])

            # Clean up
            for client in clients:
                await client.disconnect()

            # Force garbage collection
            gc.collect()

            # Log memory usage
            current_stats = monitor.get_current_stats()
            logger.info(
                f"Iteration {iteration + 1} memory: {current_stats.get('memory_mb', 0):.1f}MB"
            )

            await asyncio.sleep(1)

        # Check for memory leaks
        peak_stats = monitor.get_peak_stats()
        memory_increase = peak_stats.get("memory_increase_mb", 0)

        logger.info(f"Memory increase: {memory_increase:.1f}MB")

        # Memory increase should be reasonable (< 100MB for this test)
        assert (
            memory_increase < 100
        ), f"Potential memory leak detected: {memory_increase:.1f}MB increase"

    finally:
        await monitor.stop()


@pytest.mark.asyncio
async def test_connection_pool_stress(fastapi_server, test_user_token):
    """Test Redis connection pool under stress."""
    from tests import SERVER_URL

    monitor = SystemMonitor()
    await monitor.start()

    try:
        # Test with more clients than default Redis pool size
        num_clients = 40

        logger.info(f"Testing connection pool with {num_clients} concurrent clients...")

        clients = []
        for i in range(num_clients):
            client = StressTestClient(f"pool-test-{i}", SERVER_URL, test_user_token)
            clients.append(client)

        # Connect all clients simultaneously
        await asyncio.gather(*[client.connect() for client in clients])

        connected_clients = [c for c in clients if c.connected]
        success_rate = len(connected_clients) / num_clients

        logger.info(
            f"Connection pool test: {len(connected_clients)}/{num_clients} clients "
            f"connected ({success_rate:.1%})"
        )

        # Perform concurrent operations
        await asyncio.gather(*[client.ping_test(5) for client in connected_clients])

        # Verify results
        total_operations = sum(c.operations_count for c in connected_clients)
        total_errors = sum(len(c.errors) for c in connected_clients)

        logger.info(
            f"Pool stress test: {total_operations} operations, {total_errors} errors"
        )

        assert success_rate >= 0.7, f"Too many connection failures: {success_rate:.1%}"
        assert (
            total_errors < total_operations * 0.15
        ), "Too many operation errors under pool stress"

    finally:
        await monitor.stop()
        for client in clients:
            await client.disconnect()


@pytest.mark.asyncio
async def test_extreme_load_pytest(fastapi_server, test_user_token):
    """Extreme load test using pytest framework."""
    from tests import SERVER_URL

    extreme_tester = ExtremeLoadTester(SERVER_URL, test_user_token)

    # Run extreme test with moderate limits for CI
    results, breaking_point = await extreme_tester.run_extreme_test(
        start_clients=100, max_clients=1000, step_size=200
    )

    logger.info(f"Extreme load test completed. Breaking point: {breaking_point}")

    # Verify we handled at least some load
    max_clients = max(result["connected"] for result in results) if results else 0
    assert max_clients >= 50, f"Could not handle minimum load: {max_clients} clients"


# ==================== STANDALONE EXECUTION ====================


async def run_extreme_load_test():
    """Run standalone extreme load test."""
    print("🚀 Running Extreme Load Test...")

    tester = ExtremeLoadTester()
    results, breaking_point = await tester.run_extreme_load_test()

    print(f"\n🎯 EXTREME LOAD TEST COMPLETED")
    print(f"Breaking point: {breaking_point}")
    print(f"Results: {len(results)} test scenarios")

    # Save results
    filename = f"extreme_load_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(
            {
                "results": results,
                "breaking_point": breaking_point,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"📄 Results saved to: {filename}")


async def run_performance_benchmark():
    """Run standalone performance benchmark."""
    print("🏆 Running Performance Benchmark...")

    benchmarker = PerformanceBenchmarker()
    results = await benchmarker.run_benchmark()

    benchmarker.print_summary()
    benchmarker.save_results()


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Hypha Server Stress Test Suite")
    parser.add_argument(
        "--extreme-load", action="store_true", help="Run extreme load test"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument(
        "--max-clients",
        type=int,
        default=10000,
        help="Maximum clients for extreme test",
    )
    parser.add_argument(
        "--start-clients",
        type=int,
        default=1000,
        help="Starting clients for extreme test",
    )
    parser.add_argument(
        "--step-size", type=int, default=1000, help="Step size for extreme test"
    )

    args = parser.parse_args()

    if args.extreme_load:
        # Update configuration
        STRESS_TEST_CONFIG["extreme_load_max_clients"] = args.max_clients
        asyncio.run(run_extreme_load_test())
    elif args.benchmark:
        asyncio.run(run_performance_benchmark())
    else:
        print("Use --extreme-load or --benchmark to run standalone tests")
        print("Or use pytest to run the test suite:")
        print("  pytest tests/hypha_stress_test_suite.py -v -s")


if __name__ == "__main__":
    main()
