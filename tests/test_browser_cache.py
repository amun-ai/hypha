"""Unit tests for browser cache functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from hypha.workers.browser_cache import BrowserCache

# Mark all async functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.keys = AsyncMock(return_value=[])
    mock.exists = AsyncMock(return_value=0)
    return mock


@pytest.fixture
def cache_manager(mock_redis):
    """Create a BrowserCache instance with mocked Redis."""
    return BrowserCache(mock_redis)


async def test_cache_entry_serialization(cache_manager):
    """Test HTTP response serialization and deserialization."""
    from hypha.workers.browser_cache import CacheEntry

    # Create a cache entry
    url = "https://example.com/api"
    status = 200
    headers = {"content-type": "application/json", "cache-control": "max-age=3600"}
    body = b'{"test": "data"}'
    timestamp = 1234567890.0

    entry = CacheEntry(url, status, headers, body, timestamp)

    # Test serialization
    cache_dict = entry.to_dict()

    assert cache_dict["status"] == 200
    assert cache_dict["headers"]["content-type"] == "application/json"
    assert cache_dict["body"] == b'{"test": "data"}'.hex()
    assert cache_dict["url"] == "https://example.com/api"
    assert cache_dict["timestamp"] == timestamp

    # Test deserialization
    reconstructed = CacheEntry.from_dict(cache_dict)

    assert reconstructed.status == 200
    assert reconstructed.body == b'{"test": "data"}'
    assert reconstructed.url == url
    assert reconstructed.headers == headers
    assert reconstructed.headers["content-type"] == "application/json"


async def test_url_pattern_matching(cache_manager):
    """Test URL pattern matching with wildcards."""
    patterns = [
        "https://cdn.jsdelivr.net/*",
        "https://files.pythonhosted.org/*.whl",
        "https://pyodide.org/*/pyodide.js",
    ]

    # Enable recording so should_cache_url will check patterns
    await cache_manager.start_recording("test_workspace", "test_app")

    test_cases = [
        ("https://cdn.jsdelivr.net/pyodide/pyodide.js", True),
        ("https://cdn.jsdelivr.net/npm/react@18/index.js", True),
        ("https://files.pythonhosted.org/packages/numpy-1.24.0-py3-none-any.whl", True),
        ("https://pyodide.org/v0.24.1/pyodide.js", True),
        ("https://example.com/test.js", False),
        ("https://files.pythonhosted.org/packages/data.tar.gz", False),
    ]

    for url, should_match in test_cases:
        result = await cache_manager.should_cache_url(
            "test_workspace", "test_app", url, patterns
        )
        assert (
            result == should_match
        ), f"URL {url} should {'match' if should_match else 'not match'}"


async def test_cache_recording_session(cache_manager):
    """Test cache recording session management."""
    workspace = "test_workspace"
    app_id = "test_app"
    session_key = f"{workspace}/{app_id}"

    # Initially not recording
    assert not cache_manager.recording_sessions.get(session_key, False)

    # Start recording
    await cache_manager.start_recording(workspace, app_id)
    assert cache_manager.recording_sessions.get(session_key, False)

    # Stop recording
    await cache_manager.stop_recording(workspace, app_id)
    assert not cache_manager.recording_sessions.get(session_key, False)


async def test_cache_operations(cache_manager, mock_redis):
    """Test basic cache operations (get, set, clear)."""
    workspace = "test_workspace"
    app_id = "test_app"
    url = "https://example.com/test.js"
    status = 200
    headers = {"content-type": "application/javascript"}
    body = b"console.log('test');"

    # Test cache miss
    mock_redis.get.return_value = None
    cached = await cache_manager.get_cached_response(workspace, app_id, url)
    assert cached is None

    # Test cache set
    await cache_manager.cache_response(workspace, app_id, url, status, headers, body)
    mock_redis.setex.assert_called_once()

    # Test cache hit
    from hypha.workers.browser_cache import CacheEntry
    import json
    import time

    cache_entry = CacheEntry(url, status, headers, body, time.time())
    mock_redis.get.return_value = json.dumps(cache_entry.to_dict())

    cached = await cache_manager.get_cached_response(workspace, app_id, url)
    assert cached is not None
    assert cached.status == 200
    assert cached.body == b"console.log('test');"
    assert cached.url == url


async def test_default_cache_routes(cache_manager):
    """Test default cache routes for different app types."""
    # Test web-python defaults
    web_python_routes = cache_manager.get_default_cache_routes_for_type("web-python")

    assert len(web_python_routes) > 0
    assert any("pyodide" in route.lower() for route in web_python_routes)
    assert any("*.whl" in route for route in web_python_routes)

    # Test unsupported type
    unknown_routes = cache_manager.get_default_cache_routes_for_type("unknown")
    assert unknown_routes == []


async def test_cache_stats(cache_manager, mock_redis):
    """Test cache statistics collection."""
    workspace = "test_workspace"
    app_id = "test_app"

    # Create a proper async iterator mock
    async def mock_scan_iter(match):
        keys = [
            f"acache:{workspace}/{app_id}:url1",
            f"acache:{workspace}/{app_id}:url2",
        ]
        for key in keys:
            yield key

    mock_redis.scan_iter = mock_scan_iter

    # Mock cache entry data
    from hypha.workers.browser_cache import CacheEntry
    import json
    import time

    entry = CacheEntry("http://example.com", 200, {}, b"test data", time.time())
    mock_redis.get.return_value = json.dumps(entry.to_dict())

    stats = await cache_manager.get_cache_stats(workspace, app_id)

    assert "entry_count" in stats
    assert "total_size_bytes" in stats
    assert "oldest_entry" in stats
    assert "newest_entry" in stats
    assert "recording" in stats
    assert stats["entry_count"] == 2  # Two keys in our mock


async def test_error_propagation(cache_manager, mock_redis):
    """Test that errors propagate without defensive programming."""
    workspace = "test_workspace"
    app_id = "test_app"
    url = "https://example.com/test"

    # Test Redis connection error propagates
    mock_redis.get.side_effect = ConnectionError("Redis connection failed")

    with pytest.raises(ConnectionError):
        await cache_manager.get_cached_response(workspace, app_id, url)

    # Test JSON parsing error propagates
    mock_redis.get.side_effect = None
    mock_redis.get.return_value = "invalid json"

    with pytest.raises(Exception):  # JSON decode error should propagate
        await cache_manager.get_cached_response(workspace, app_id, url)
