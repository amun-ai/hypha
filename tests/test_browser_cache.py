"""Test browser cache functionality."""

import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock
from fakeredis import aioredis

from hypha.workers.browser_cache import BrowserCache, CacheEntry


@pytest.fixture
async def fake_redis():
    """Create a fake Redis client for testing."""
    redis = aioredis.FakeRedis()
    await redis.flushall()
    return redis


@pytest.fixture
async def browser_cache(fake_redis):
    """Create a browser cache instance for testing."""
    return BrowserCache(fake_redis, cache_ttl=3600)


@pytest.mark.asyncio
async def test_cache_entry_serialization():
    """Test CacheEntry serialization and deserialization."""
    # Test data
    url = "https://example.com/test.js"
    status = 200
    headers = {"Content-Type": "application/javascript", "Cache-Control": "max-age=3600"}
    body = b"console.log('test');"
    timestamp = time.time()
    
    # Create entry
    entry = CacheEntry(url, status, headers, body, timestamp)
    
    # Test serialization
    data = entry.to_dict()
    assert data["url"] == url
    assert data["status"] == status
    assert data["headers"] == headers
    assert data["body"] == body.hex()
    assert data["timestamp"] == timestamp
    
    # Test deserialization
    restored_entry = CacheEntry.from_dict(data)
    assert restored_entry.url == url
    assert restored_entry.status == status
    assert restored_entry.headers == headers
    assert restored_entry.body == body
    assert restored_entry.timestamp == timestamp


@pytest.mark.asyncio
async def test_url_pattern_matching(browser_cache):
    """Test URL pattern matching functionality."""
    # Test exact match
    assert browser_cache._url_matches_pattern("https://example.com/test.js", "https://example.com/test.js")
    
    # Test wildcard patterns
    assert browser_cache._url_matches_pattern("https://example.com/test.js", "https://example.com/*")
    assert browser_cache._url_matches_pattern("https://example.com/path/test.js", "https://example.com/*")
    assert browser_cache._url_matches_pattern("https://cdn.jsdelivr.net/pyodide/pyodide.js", "https://cdn.jsdelivr.net/pyodide/*")
    
    # Test file extension patterns
    assert browser_cache._url_matches_pattern("https://example.com/package.whl", "*.whl")
    assert browser_cache._url_matches_pattern("https://files.pythonhosted.org/package.tar.gz", "*.tar.gz")
    
    # Test non-matching patterns
    assert not browser_cache._url_matches_pattern("https://example.com/test.js", "https://other.com/*")
    assert not browser_cache._url_matches_pattern("https://example.com/test.js", "*.css")
    
    # Test multiple patterns
    patterns = ["https://example.com/*", "*.whl", "https://pypi.org/*"]
    assert browser_cache._url_matches_patterns("https://example.com/test.js", patterns)
    assert browser_cache._url_matches_patterns("https://files.pythonhosted.org/package.whl", patterns)
    assert browser_cache._url_matches_patterns("https://pypi.org/simple/", patterns)
    assert not browser_cache._url_matches_patterns("https://other.com/test.js", patterns)


@pytest.mark.asyncio
async def test_cache_key_generation(browser_cache):
    """Test cache key generation."""
    workspace = "test-workspace"
    app_id = "test-app"
    url = "https://example.com/test.js"
    
    key = browser_cache._get_cache_key(workspace, app_id, url)
    assert key.startswith(f"acache:{workspace}/{app_id}:")
    assert len(key.split(":")) == 3  # acache:workspace/app_id:hash
    
    # Same URL should generate same key
    key2 = browser_cache._get_cache_key(workspace, app_id, url)
    assert key == key2
    
    # Different URL should generate different key
    key3 = browser_cache._get_cache_key(workspace, app_id, "https://example.com/other.js")
    assert key != key3


@pytest.mark.asyncio
async def test_recording_session_management(browser_cache):
    """Test recording session management."""
    workspace = "test-workspace"
    app_id = "test-app"
    
    # Initially not recording
    assert not browser_cache.recording_sessions.get(f"{workspace}/{app_id}", False)
    
    # Start recording
    await browser_cache.start_recording(workspace, app_id)
    assert browser_cache.recording_sessions[f"{workspace}/{app_id}"] is True
    
    # Stop recording
    await browser_cache.stop_recording(workspace, app_id)
    assert f"{workspace}/{app_id}" not in browser_cache.recording_sessions


@pytest.mark.asyncio
async def test_should_cache_url(browser_cache):
    """Test should_cache_url logic."""
    workspace = "test-workspace"
    app_id = "test-app"
    url = "https://cdn.jsdelivr.net/pyodide/pyodide.js"
    cache_routes = ["https://cdn.jsdelivr.net/pyodide/*"]
    
    # Not recording, should not cache
    should_cache = await browser_cache.should_cache_url(workspace, app_id, url, cache_routes)
    assert not should_cache
    
    # Start recording, should cache matching URLs
    await browser_cache.start_recording(workspace, app_id)
    should_cache = await browser_cache.should_cache_url(workspace, app_id, url, cache_routes)
    assert should_cache
    
    # Non-matching URL should not be cached
    non_matching_url = "https://example.com/test.js"
    should_cache = await browser_cache.should_cache_url(workspace, app_id, non_matching_url, cache_routes)
    assert not should_cache
    
    # No cache routes should not cache
    should_cache = await browser_cache.should_cache_url(workspace, app_id, url, [])
    assert not should_cache


@pytest.mark.asyncio
async def test_cache_response_and_retrieval(browser_cache):
    """Test caching and retrieving responses."""
    workspace = "test-workspace"
    app_id = "test-app"
    url = "https://example.com/test.js"
    status = 200
    headers = {"Content-Type": "application/javascript"}
    body = b"console.log('test');"
    
    # Cache the response
    await browser_cache.cache_response(workspace, app_id, url, status, headers, body)
    
    # Retrieve the cached response
    cached_entry = await browser_cache.get_cached_response(workspace, app_id, url)
    
    assert cached_entry is not None
    assert cached_entry.url == url
    assert cached_entry.status == status
    assert cached_entry.headers == headers
    assert cached_entry.body == body
    assert isinstance(cached_entry.timestamp, float)
    
    # Non-existent URL should return None
    non_cached_entry = await browser_cache.get_cached_response(workspace, app_id, "https://example.com/nonexistent.js")
    assert non_cached_entry is None


@pytest.mark.asyncio
async def test_clear_app_cache(browser_cache):
    """Test clearing app cache."""
    workspace = "test-workspace"
    app_id = "test-app"
    
    # Cache multiple responses
    urls = [
        "https://example.com/test1.js",
        "https://example.com/test2.js",
        "https://example.com/test3.js"
    ]
    
    for url in urls:
        await browser_cache.cache_response(workspace, app_id, url, 200, {}, b"test content")
    
    # Verify entries exist
    for url in urls:
        entry = await browser_cache.get_cached_response(workspace, app_id, url)
        assert entry is not None
    
    # Clear cache
    deleted_count = await browser_cache.clear_app_cache(workspace, app_id)
    assert deleted_count == 3
    
    # Verify entries are gone
    for url in urls:
        entry = await browser_cache.get_cached_response(workspace, app_id, url)
        assert entry is None
    
    # Clearing empty cache should return 0
    deleted_count = await browser_cache.clear_app_cache(workspace, app_id)
    assert deleted_count == 0


@pytest.mark.asyncio
async def test_cache_stats(browser_cache):
    """Test cache statistics."""
    workspace = "test-workspace"
    app_id = "test-app"
    
    # Initially empty stats
    stats = await browser_cache.get_cache_stats(workspace, app_id)
    assert stats["entry_count"] == 0
    assert stats["total_size_bytes"] == 0
    assert stats["oldest_entry"] is None
    assert stats["newest_entry"] is None
    assert stats["recording"] is False
    
    # Add some cached entries
    await browser_cache.cache_response(workspace, app_id, "https://example.com/test1.js", 200, {}, b"test1")
    await browser_cache.cache_response(workspace, app_id, "https://example.com/test2.js", 200, {}, b"test22")
    
    # Start recording
    await browser_cache.start_recording(workspace, app_id)
    
    # Check stats
    stats = await browser_cache.get_cache_stats(workspace, app_id)
    assert stats["entry_count"] == 2
    assert stats["total_size_bytes"] == 11  # len(b"test1") + len(b"test22")
    assert stats["oldest_entry"] is not None
    assert stats["newest_entry"] is not None
    assert stats["recording"] is True


@pytest.mark.asyncio
async def test_default_cache_routes():
    """Test default cache routes for different app types."""
    cache = BrowserCache(None)  # Redis client not needed for this test
    
    # web-python should have default routes
    web_python_routes = cache.get_default_cache_routes_for_type("web-python")
    assert len(web_python_routes) > 0
    assert any("pyodide" in route for route in web_python_routes)
    assert any("*.whl" in route for route in web_python_routes)
    
    # Other types should have no default routes
    other_routes = cache.get_default_cache_routes_for_type("window")
    assert len(other_routes) == 0


@pytest.mark.asyncio
async def test_cache_ttl_expiration(browser_cache):
    """Test cache TTL expiration (mock test since we can't wait for real expiration)."""
    # This test verifies that the TTL is set correctly when caching
    workspace = "test-workspace"
    app_id = "test-app"
    url = "https://example.com/test.js"
    
    # Mock the redis setex method to verify TTL is passed correctly
    original_setex = browser_cache.redis_client.setex
    setex_calls = []
    
    async def mock_setex(key, ttl, value):
        setex_calls.append((key, ttl, value))
        return await original_setex(key, ttl, value)
    
    browser_cache.redis_client.setex = mock_setex
    
    # Cache a response
    await browser_cache.cache_response(workspace, app_id, url, 200, {}, b"test")
    
    # Verify setex was called with correct TTL
    assert len(setex_calls) == 1
    key, ttl, value = setex_calls[0]
    assert ttl == browser_cache.cache_ttl
    assert key.startswith(f"acache:{workspace}/{app_id}:")


@pytest.mark.asyncio
async def test_cache_isolation_between_apps(browser_cache):
    """Test that cache entries are isolated between different apps."""
    workspace = "test-workspace"
    app_id_1 = "test-app-1"
    app_id_2 = "test-app-2"
    url = "https://example.com/test.js"
    
    # Cache same URL for different apps
    await browser_cache.cache_response(workspace, app_id_1, url, 200, {}, b"content1")
    await browser_cache.cache_response(workspace, app_id_2, url, 200, {}, b"content2")
    
    # Verify each app gets its own cached content
    entry1 = await browser_cache.get_cached_response(workspace, app_id_1, url)
    entry2 = await browser_cache.get_cached_response(workspace, app_id_2, url)
    
    assert entry1 is not None
    assert entry2 is not None
    assert entry1.body == b"content1"
    assert entry2.body == b"content2"
    
    # Clear cache for one app shouldn't affect the other
    deleted_count = await browser_cache.clear_app_cache(workspace, app_id_1)
    assert deleted_count == 1
    
    # App 1 cache should be cleared
    entry1 = await browser_cache.get_cached_response(workspace, app_id_1, url)
    assert entry1 is None
    
    # App 2 cache should still exist
    entry2 = await browser_cache.get_cached_response(workspace, app_id_2, url)
    assert entry2 is not None
    assert entry2.body == b"content2"