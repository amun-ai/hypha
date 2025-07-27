"""Browser worker cache manager using Redis."""

import asyncio
import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Pattern
import re
from urllib.parse import urlparse
from aiocache.backends.redis import RedisCache
from aiocache.serializers import PickleSerializer

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a cached HTTP response."""

    def __init__(
        self,
        url: str,
        status: int,
        headers: Dict[str, str],
        body: bytes,
        timestamp: float,
    ):
        self.url = url
        self.status = status
        self.headers = headers
        self.body = body
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "status": self.status,
            "headers": self.headers,
            "body": self.body.hex() if isinstance(self.body, bytes) else self.body,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        body = (
            bytes.fromhex(data["body"])
            if isinstance(data["body"], str)
            else data["body"]
        )
        return cls(
            url=data["url"],
            status=data["status"],
            headers=data["headers"],
            body=body,
            timestamp=data["timestamp"],
        )


class BrowserCache:
    """Browser worker cache manager."""

    def __init__(self, redis_client, cache_ttl: int = 24 * 60 * 60):  # 24 hours default
        """Initialize the cache manager.

        Args:
            redis_client: Redis client instance
            cache_ttl: Cache time-to-live in seconds
        """
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.recording_sessions: Dict[str, bool] = {}

        # Default cache routes for web-python apps
        self.DEFAULT_WEB_PYTHON_ROUTES = [
            "https://cdn.jsdelivr.net/pyodide/*",
            "https://pypi.org/*",
            "https://files.pythonhosted.org/*",
            "https://pyodide.org/*",
            "*.whl",
            "*.tar.gz",
        ]

    def _get_cache_key(self, workspace: str, app_id: str, url: str) -> str:
        """Generate cache key for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"acache:{workspace}/{app_id}:{url_hash}"

    def _get_app_cache_pattern(self, workspace: str, app_id: str) -> str:
        """Get cache pattern for all URLs of an app."""
        return f"acache:{workspace}/{app_id}:*"

    def _url_matches_patterns(self, url: str, patterns: List[str]) -> bool:
        """Check if URL matches any of the given patterns."""
        for pattern in patterns:
            if self._url_matches_pattern(url, pattern):
                return True
        return False

    def _url_matches_pattern(self, url: str, pattern: str) -> bool:
        """Check if URL matches a pattern (supports wildcards)."""
        # Convert glob-style pattern to regex
        # Escape special regex characters except * and ?
        pattern_escaped = re.escape(pattern)
        # Replace escaped wildcards with regex equivalents
        pattern_regex = pattern_escaped.replace(r"\*", ".*").replace(r"\?", ".")
        # Anchor the pattern
        pattern_regex = f"^{pattern_regex}$"

        return bool(re.match(pattern_regex, url))

    async def should_cache_url(
        self,
        workspace: str,
        app_id: str,
        url: str,
        cache_routes: Optional[List[str]] = None,
    ) -> bool:
        """Determine if a URL should be cached."""
        # Check if we're in recording mode for this session
        session_key = f"{workspace}/{app_id}"
        if not self.recording_sessions.get(session_key, False):
            return False

        if not cache_routes:
            return False

        return self._url_matches_patterns(url, cache_routes)

    async def get_cached_response(
        self, workspace: str, app_id: str, url: str
    ) -> Optional[CacheEntry]:
        """Get cached response for a URL."""
        cache_key = self._get_cache_key(workspace, app_id, url)

        cached_data = await self.redis_client.get(cache_key)
        if cached_data:
            data = json.loads(cached_data)
            return CacheEntry.from_dict(data)

        return None

    async def cache_response(
        self,
        workspace: str,
        app_id: str,
        url: str,
        status: int,
        headers: Dict[str, str],
        body: bytes,
    ) -> None:
        """Cache an HTTP response."""
        cache_key = self._get_cache_key(workspace, app_id, url)

        entry = CacheEntry(url, status, headers, body, time.time())
        data = json.dumps(entry.to_dict())

        await self.redis_client.setex(cache_key, self.cache_ttl, data)
        logger.info(f"Cached response for {url} (size: {len(body)} bytes)")

    async def start_recording(self, workspace: str, app_id: str) -> None:
        """Start recording/caching for an app session."""
        session_key = f"{workspace}/{app_id}"
        self.recording_sessions[session_key] = True
        logger.info(f"Started cache recording for {session_key}")

    async def stop_recording(self, workspace: str, app_id: str) -> None:
        """Stop recording/caching for an app session."""
        session_key = f"{workspace}/{app_id}"
        self.recording_sessions.pop(session_key, None)
        logger.info(f"Stopped cache recording for {session_key}")

    async def clear_app_cache(self, workspace: str, app_id: str) -> int:
        """Clear all cached entries for an app."""
        pattern = self._get_app_cache_pattern(workspace, app_id)

        # Find all keys matching the pattern
        keys = []
        async for key in self.redis_client.scan_iter(match=pattern):
            keys.append(key)

        if keys:
            deleted = await self.redis_client.delete(*keys)
            logger.info(f"Cleared {deleted} cache entries for {workspace}/{app_id}")
            return deleted

        return 0

    async def get_cache_stats(self, workspace: str, app_id: str) -> Dict[str, Any]:
        """Get cache statistics for an app."""
        pattern = self._get_app_cache_pattern(workspace, app_id)

        entry_count = 0
        total_size = 0
        oldest_timestamp = None
        newest_timestamp = None

        async for key in self.redis_client.scan_iter(match=pattern):
            entry_count += 1
            cached_data = await self.redis_client.get(key)
            if cached_data:
                data = json.loads(cached_data)
                body_size = (
                    len(bytes.fromhex(data["body"]))
                    if isinstance(data["body"], str)
                    else len(data["body"])
                )
                total_size += body_size

                timestamp = data["timestamp"]
                if oldest_timestamp is None or timestamp < oldest_timestamp:
                    oldest_timestamp = timestamp
                if newest_timestamp is None or timestamp > newest_timestamp:
                    newest_timestamp = timestamp

        return {
            "entry_count": entry_count,
            "total_size_bytes": total_size,
            "oldest_entry": oldest_timestamp,
            "newest_entry": newest_timestamp,
            "recording": self.recording_sessions.get(f"{workspace}/{app_id}", False),
        }

    def get_default_cache_routes_for_type(self, app_type: str) -> List[str]:
        """Get default cache routes for an app type."""
        if app_type == "web-python":
            return self.DEFAULT_WEB_PYTHON_ROUTES.copy()
        return []
