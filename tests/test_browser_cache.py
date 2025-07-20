"""Unit tests for browser cache functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from hypha.workers.browser_cache import BrowserCache


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


class TestBrowserCacheUnit:
    """Unit tests for BrowserCache core functionality."""

    async def test_cache_entry_serialization(self, cache_manager):
        """Test HTTP response serialization and deserialization."""
        # Mock HTTP response data
        response_data = {
            "status": 200,
            "headers": {"content-type": "application/json", "cache-control": "max-age=3600"},
            "body": b'{"test": "data"}',
            "url": "https://example.com/api"
        }
        
        # Test serialization
        cache_entry = cache_manager._create_cache_entry(response_data)
        
        assert cache_entry["status"] == 200
        assert cache_entry["headers"]["content-type"] == "application/json"
        assert cache_entry["body_hex"] == b'{"test": "data"}'.hex()
        assert cache_entry["url"] == "https://example.com/api"
        assert "timestamp" in cache_entry
        
        # Test deserialization
        reconstructed = cache_manager._parse_cache_entry(cache_entry)
        
        assert reconstructed["status"] == 200
        assert reconstructed["body"] == b'{"test": "data"}'
        assert reconstructed["headers"]["content-type"] == "application/json"

    async def test_url_pattern_matching(self, cache_manager):
        """Test URL pattern matching with wildcards."""
        patterns = [
            "https://cdn.jsdelivr.net/*",
            "https://files.pythonhosted.org/*.whl",
            "https://pyodide.org/*/pyodide.js"
        ]
        
        test_cases = [
            ("https://cdn.jsdelivr.net/pyodide/pyodide.js", True),
            ("https://cdn.jsdelivr.net/npm/react@18/index.js", True),
            ("https://files.pythonhosted.org/packages/numpy-1.24.0-py3-none-any.whl", True),
            ("https://pyodide.org/v0.24.1/pyodide.js", True),
            ("https://example.com/test.js", False),
            ("https://files.pythonhosted.org/packages/data.tar.gz", False)
        ]
        
        for url, should_match in test_cases:
            result = await cache_manager.should_cache_url("test_workspace", "test_app", url, patterns)
            assert result == should_match, f"URL {url} should {'match' if should_match else 'not match'}"

    async def test_cache_recording_session(self, cache_manager):
        """Test cache recording session management."""
        workspace = "test_workspace"
        app_id = "test_app"
        
        # Initially not recording
        assert not cache_manager.is_recording(workspace, app_id)
        
        # Start recording
        await cache_manager.start_recording(workspace, app_id)
        assert cache_manager.is_recording(workspace, app_id)
        
        # Stop recording
        await cache_manager.stop_recording(workspace, app_id)
        assert not cache_manager.is_recording(workspace, app_id)

    async def test_cache_operations(self, cache_manager, mock_redis):
        """Test basic cache operations (get, set, clear)."""
        workspace = "test_workspace"
        app_id = "test_app"
        url = "https://example.com/test.js"
        
        response_data = {
            "status": 200,
            "headers": {"content-type": "application/javascript"},
            "body": b"console.log('test');",
            "url": url
        }
        
        # Test cache miss
        mock_redis.get.return_value = None
        cached = await cache_manager.get_cached_response(workspace, app_id, url)
        assert cached is None
        
        # Test cache set
        await cache_manager.cache_response(workspace, app_id, url, response_data)
        mock_redis.set.assert_called_once()
        
        # Test cache hit
        cache_entry = cache_manager._create_cache_entry(response_data)
        import json
        mock_redis.get.return_value = json.dumps(cache_entry)
        
        cached = await cache_manager.get_cached_response(workspace, app_id, url)
        assert cached is not None
        assert cached["status"] == 200
        assert cached["body"] == b"console.log('test');"

    async def test_default_cache_routes(self, cache_manager):
        """Test default cache routes for different app types."""
        # Test web-python defaults
        web_python_routes = cache_manager.get_default_cache_routes_for_type("web-python")
        
        assert len(web_python_routes) > 0
        assert any("pyodide" in route.lower() for route in web_python_routes)
        assert any("*.whl" in route for route in web_python_routes)
        
        # Test unsupported type
        unknown_routes = cache_manager.get_default_cache_routes_for_type("unknown")
        assert unknown_routes == []

    async def test_cache_stats(self, cache_manager, mock_redis):
        """Test cache statistics collection."""
        workspace = "test_workspace"
        app_id = "test_app"
        
        # Mock Redis responses for stats
        mock_redis.keys.return_value = [
            f"acache:{workspace}/{app_id}:url1",
            f"acache:{workspace}/{app_id}:url2"
        ]
        mock_redis.get.return_value = '{"size": 1024}'
        
        stats = await cache_manager.get_cache_stats(workspace, app_id)
        
        assert "entries" in stats
        assert "total_size" in stats
        assert "hits" in stats
        assert "misses" in stats

    async def test_error_propagation(self, cache_manager, mock_redis):
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