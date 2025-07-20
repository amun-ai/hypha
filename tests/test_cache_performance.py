"""Performance tests for browser worker caching system."""

import asyncio
import time
import pytest
import hypha_rpc
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SERVER_URL, find_item, wait_for_workspace_ready

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_cache_performance_web_python(fastapi_server, test_user_token):
    """Test performance improvement with caching for web-python apps."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 120,  # Longer timeout for performance tests
            "token": test_user_token,
        }
    )

    controller = await api.get_service("public/server-apps")

    # Simple web-python app that would load Pyodide
    web_python_source = """
# Simple test app
from js import console
console.log("Web Python performance test app loaded")

api.export({
    "setup": lambda: "App setup complete",
    "test": lambda: "Performance test successful"
})
"""

    print("🧪 Testing Web-Python App Performance with Caching")
    print("=" * 60)

    # Test 1: App WITHOUT explicit cache routes (baseline)
    print("\n📊 Phase 1: Installing app WITHOUT cache routes...")
    
    app_no_cache = await controller.install(
        source=web_python_source,
        manifest={
            "name": "No Cache Performance Test",
            "type": "web-python",
            "version": "1.0.0",
            "cache_routes": []  # Explicitly disable caching
        },
        stage=False,  # Actually start the app
        overwrite=True,
    )

    print(f"✅ Installed no-cache app: {app_no_cache['id']}")

    # Measure first startup time (no cache)
    print("⏱️ Measuring first startup time (no cache)...")
    start_time = time.time()
    
    session_id_1 = await controller.start(app_no_cache["id"])
    
    # Wait for app to fully initialize
    await asyncio.sleep(5)
    
    # Try to connect to verify it's running
    services = await api.list_services({"workspace": "default"})
    app_services = [s for s in services if s["id"].startswith(app_no_cache["id"])]
    
    if app_services:
        app_service = await api.get_service(app_services[0]["id"])
        result = await app_service.test()
        assert result == "Performance test successful"
        print(f"✅ App service connected and working")
    
    no_cache_time = time.time() - start_time
    print(f"📈 No-cache startup time: {no_cache_time:.2f} seconds")

    # Stop the app
    await controller.stop(session_id_1)
    await asyncio.sleep(2)

    # Test 2: App WITH cache routes (should be similar first time)
    print("\n📊 Phase 2: Installing app WITH cache routes...")
    
    app_with_cache = await controller.install(
        source=web_python_source,
        manifest={
            "name": "Cached Performance Test",
            "type": "web-python",
            "version": "1.0.0",
            # Use default cache routes for web-python
        },
        stage=False,  # Actually start the app
        overwrite=True,
    )

    print(f"✅ Installed cached app: {app_with_cache['id']}")

    # Measure first startup time (with cache recording)
    print("⏱️ Measuring first startup time (cache recording)...")
    start_time = time.time()
    
    session_id_2 = await controller.start(app_with_cache["id"])
    
    # Wait for app to fully initialize and populate cache
    await asyncio.sleep(5)
    
    # Verify it's working
    services = await api.list_services({"workspace": "default"})
    app_services = [s for s in services if s["id"].startswith(app_with_cache["id"])]
    
    if app_services:
        app_service = await api.get_service(app_services[0]["id"])
        result = await app_service.test()
        assert result == "Performance test successful"
        print(f"✅ Cached app service connected and working")
    
    first_cached_time = time.time() - start_time
    print(f"📈 First cached startup time: {first_cached_time:.2f} seconds")

    # Check cache stats
    browser_worker = await api.get_service("public/browser-worker")
    stats = await browser_worker.get_app_cache_stats("default", app_with_cache["id"])
    print(f"📊 Cache populated: {stats['entry_count']} entries, {stats['total_size_bytes']} bytes")

    # Stop the app
    await controller.stop(session_id_2)
    await asyncio.sleep(2)

    # Test 3: Restart cached app (should be faster)
    print("\n📊 Phase 3: Restarting cached app...")
    print("⏱️ Measuring second startup time (using cache)...")
    
    start_time = time.time()
    
    session_id_3 = await controller.start(app_with_cache["id"])
    
    # Wait for app to initialize (should be faster)
    await asyncio.sleep(5)
    
    # Verify it's working
    services = await api.list_services({"workspace": "default"})
    app_services = [s for s in services if s["id"].startswith(app_with_cache["id"])]
    
    if app_services:
        app_service = await api.get_service(app_services[0]["id"])
        result = await app_service.test()
        assert result == "Performance test successful"
        print(f"✅ Second cached app service connected and working")
    
    second_cached_time = time.time() - start_time
    print(f"📈 Second cached startup time: {second_cached_time:.2f} seconds")

    # Stop the app
    await controller.stop(session_id_3)

    # Performance Analysis
    print("\n📈 PERFORMANCE ANALYSIS")
    print("=" * 40)
    print(f"No cache:           {no_cache_time:.2f}s")
    print(f"First cached:       {first_cached_time:.2f}s")
    print(f"Second cached:      {second_cached_time:.2f}s")
    
    improvement_vs_no_cache = ((no_cache_time - second_cached_time) / no_cache_time) * 100
    improvement_vs_first = ((first_cached_time - second_cached_time) / first_cached_time) * 100
    
    print(f"\n🎯 Performance Improvements:")
    print(f"   vs No Cache:     {improvement_vs_no_cache:+.1f}%")
    print(f"   vs First Load:   {improvement_vs_first:+.1f}%")
    
    # Cache effectiveness
    final_stats = await browser_worker.get_app_cache_stats("default", app_with_cache["id"])
    print(f"\n💾 Cache Effectiveness:")
    print(f"   Entries cached:  {final_stats['entry_count']}")
    print(f"   Total size:      {final_stats['total_size_bytes']} bytes")
    print(f"   Recording:       {final_stats['recording']}")
    
    # The second cached load should be same or better than first
    if second_cached_time <= first_cached_time:
        print("✅ Cache provided performance benefit or maintained speed")
    else:
        print("⚠️ Second load was slower (cache infrastructure overhead)")
    
    # Cache should have some entries if external resources were loaded
    if final_stats['entry_count'] > 0:
        print("✅ Cache successfully captured external resources")
    else:
        print("ℹ️ No external resources cached (may be expected in test environment)")

    # Clean up
    await controller.uninstall(app_no_cache["id"])
    await controller.uninstall(app_with_cache["id"])

    await api.disconnect()

    # Return performance metrics for analysis
    return {
        "no_cache_time": no_cache_time,
        "first_cached_time": first_cached_time, 
        "second_cached_time": second_cached_time,
        "improvement_vs_no_cache": improvement_vs_no_cache,
        "improvement_vs_first": improvement_vs_first,
        "cache_entries": final_stats['entry_count'],
        "cache_size": final_stats['total_size_bytes']
    }


async def test_web_app_performance(fastapi_server, test_user_token):
    """Test performance of web-app type with direct URL navigation."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 60,
            "token": test_user_token,
        }
    )

    controller = await api.get_service("public/server-apps")

    print("\n🧪 Testing Web-App Performance")
    print("=" * 40)

    # Test web-app with httpbin.org (simple, reliable test endpoint)
    app_info = await controller.install(
        manifest={
            "name": "HTTPBin Test App",
            "type": "web-app",
            "version": "1.0.0",
            "url": "https://httpbin.org/get",
            "cache_routes": [
                "https://httpbin.org/*"
            ]
        },
        files=[],
        stage=False,  # Actually start to test navigation
        overwrite=True,
    )

    print(f"✅ Installed web-app: {app_info['id']}")

    # Measure startup time
    print("⏱️ Measuring web-app startup time...")
    start_time = time.time()
    
    session_id = await controller.start(app_info["id"])
    
    # Wait for page to load
    await asyncio.sleep(5)
    
    startup_time = time.time() - start_time
    print(f"📈 Web-app startup time: {startup_time:.2f} seconds")

    # Check if any requests were cached
    browser_worker = await api.get_service("public/browser-worker")
    stats = await browser_worker.get_app_cache_stats("default", app_info["id"])
    print(f"📊 Web-app cache stats: {stats['entry_count']} entries, {stats['total_size_bytes']} bytes")

    # Stop and restart to test cache usage
    await controller.stop(session_id)
    await asyncio.sleep(2)

    print("⏱️ Measuring web-app restart time (with cache)...")
    start_time = time.time()
    
    session_id_2 = await controller.start(app_info["id"])
    
    # Wait for page to load
    await asyncio.sleep(5)
    
    restart_time = time.time() - start_time
    print(f"📈 Web-app restart time: {restart_time:.2f} seconds")

    # Performance comparison
    improvement = ((startup_time - restart_time) / startup_time) * 100
    print(f"🎯 Web-app restart improvement: {improvement:+.1f}%")

    if restart_time <= startup_time:
        print("✅ Web-app restart was same or faster")
    else:
        print("⚠️ Web-app restart was slower")

    # Stop and clean up
    await controller.stop(session_id_2)
    await controller.uninstall(app_info["id"])

    await api.disconnect()

    return {
        "startup_time": startup_time,
        "restart_time": restart_time,
        "improvement": improvement,
        "cache_entries": stats['entry_count']
    }


async def test_cache_route_patterns_performance(fastapi_server, test_user_token):
    """Test different cache route patterns and their effectiveness."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 60,
            "token": test_user_token,
        }
    )

    controller = await api.get_service("public/server-apps")
    browser_worker = await api.get_service("public/browser-worker")

    print("\n🧪 Testing Cache Route Pattern Effectiveness")
    print("=" * 50)

    # Test different cache patterns
    cache_patterns = [
        {
            "name": "No Cache",
            "routes": []
        },
        {
            "name": "Specific Domain",
            "routes": ["https://httpbin.org/*"]
        },
        {
            "name": "Multiple Domains",
            "routes": ["https://httpbin.org/*", "https://cdn.jsdelivr.net/*", "*.js"]
        },
        {
            "name": "File Extensions",
            "routes": ["*.js", "*.css", "*.json", "*.wasm"]
        }
    ]

    results = {}

    for i, pattern_config in enumerate(cache_patterns):
        print(f"\n📊 Testing Pattern {i+1}: {pattern_config['name']}")
        print(f"   Routes: {pattern_config['routes']}")

        # Install app with this cache pattern
        app_info = await controller.install(
            manifest={
                "name": f"Cache Pattern Test {i+1}",
                "type": "web-app",
                "version": "1.0.0",
                "url": "https://httpbin.org/json",
                "cache_routes": pattern_config['routes']
            },
            files=[],
            stage=False,
            overwrite=True,
        )

        # Measure startup time
        start_time = time.time()
        session_id = await controller.start(app_info["id"])
        await asyncio.sleep(3)
        startup_time = time.time() - start_time

        # Get cache stats
        stats = await browser_worker.get_app_cache_stats("default", app_info["id"])
        
        # Restart and measure
        await controller.stop(session_id)
        await asyncio.sleep(1)
        
        start_time = time.time()
        session_id_2 = await controller.start(app_info["id"])
        await asyncio.sleep(3)
        restart_time = time.time() - start_time
        
        await controller.stop(session_id_2)

        # Store results
        results[pattern_config['name']] = {
            "startup_time": startup_time,
            "restart_time": restart_time,
            "cache_entries": stats['entry_count'],
            "cache_size": stats['total_size_bytes'],
            "improvement": ((startup_time - restart_time) / startup_time) * 100 if startup_time > 0 else 0
        }

        print(f"   Startup:  {startup_time:.2f}s")
        print(f"   Restart:  {restart_time:.2f}s") 
        print(f"   Cached:   {stats['entry_count']} entries ({stats['total_size_bytes']} bytes)")
        print(f"   Improve:  {results[pattern_config['name']]['improvement']:+.1f}%")

        # Clean up
        await controller.uninstall(app_info["id"])

    # Summary
    print(f"\n📈 CACHE PATTERN EFFECTIVENESS SUMMARY")
    print("=" * 50)
    
    for name, result in results.items():
        print(f"{name:20} | {result['startup_time']:6.2f}s | {result['restart_time']:6.2f}s | {result['improvement']:+6.1f}% | {result['cache_entries']:3d} entries")

    await api.disconnect()

    return results