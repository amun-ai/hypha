"""Rate limiting infrastructure for DoS prevention.

This module provides a Redis-backed distributed rate limiting system to prevent
resource exhaustion attacks across multiple DoS vectors (DOS1-DOS8).

Key Features:
- Distributed rate limiting using Redis (supports horizontal scaling)
- Per-user, per-workspace, and global rate limits
- Sliding window algorithm for accurate rate limiting
- Configurable limits via environment variables
- Graceful degradation when Redis unavailable

Architecture:
- Uses Redis sorted sets for sliding window implementation
- Automatic cleanup via TTL
- Support for multiple scopes (user, workspace, global, custom)

Usage:
    from hypha.core.rate_limiter import RateLimiter, rate_limit

    # As a decorator
    @rate_limit(max_requests=100, window_seconds=60, scope="user")
    async def my_function(context: dict):
        ...

    # Manual usage
    limiter = RateLimiter(redis_client)
    allowed = await limiter.check_rate_limit(
        key="user:123",
        max_requests=100,
        window_seconds=60
    )
"""

import time
import logging
import os
from typing import Optional, Literal, Callable, Any
from functools import wraps
import asyncio
from prometheus_client import Counter

logger = logging.getLogger(__name__)

# Prometheus metrics for monitoring
_rate_limit_exceeded_counter = Counter(
    "rate_limit_exceeded_total",
    "Total number of rate limit exceeded events",
    ["endpoint"],
)

_rate_limit_errors_counter = Counter(
    "rate_limit_errors_total",
    "Total number of rate limiter errors (Redis failures)",
    ["error_type"],
)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    """Redis-backed distributed rate limiter using sliding window algorithm.

    Uses Redis sorted sets to implement accurate sliding window rate limiting.
    Each request is recorded with a timestamp, and old requests are automatically
    cleaned up.

    Attributes:
        redis: Redis client (can be async or sync)
        prefix: Key prefix for Redis keys (default: "ratelimit:")
        enabled: Whether rate limiting is enabled (default: True)
    """

    def __init__(
        self,
        redis_client: Any,
        prefix: str = "ratelimit:",
        enabled: bool = None,
    ):
        """Initialize rate limiter.

        Args:
            redis_client: Redis client instance (asyncio or sync)
            prefix: Prefix for all Redis keys
            enabled: Override enable/disable (default: from env HYPHA_RATE_LIMITING_ENABLED)
        """
        self.redis = redis_client
        self.prefix = prefix

        # Check if rate limiting is enabled
        if enabled is None:
            enabled = os.environ.get("HYPHA_RATE_LIMITING_ENABLED", "true").lower() == "true"
        self.enabled = enabled

        if not self.enabled:
            logger.warning("Rate limiting is DISABLED. Set HYPHA_RATE_LIMITING_ENABLED=true to enable.")

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
        increment: bool = True,
    ) -> tuple[bool, int, float]:
        """Check if request is within rate limit using sliding window.

        Args:
            key: Unique identifier for rate limit scope (e.g., "user:123", "ws:myworkspace")
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            increment: Whether to increment counter (default: True)

        Returns:
            Tuple of (allowed, current_count, retry_after_seconds)
            - allowed: True if request is allowed
            - current_count: Current number of requests in window
            - retry_after_seconds: Seconds until rate limit resets (0 if allowed)

        Raises:
            Exception: If Redis operation fails (gracefully degrades by allowing request)
        """
        if not self.enabled:
            return True, 0, 0.0

        redis_key = f"{self.prefix}{key}"
        current_time = time.time()
        window_start = current_time - window_seconds

        try:
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()

            # Remove old entries outside the window
            pipe.zremrangebyscore(redis_key, 0, window_start)

            # Count current requests in window
            pipe.zcard(redis_key)

            # Add current request if incrementing
            if increment:
                # Use timestamp + small random to avoid collisions
                request_id = f"{current_time}:{os.urandom(4).hex()}"
                pipe.zadd(redis_key, {request_id: current_time})

            # Set expiry to window + buffer for cleanup
            pipe.expire(redis_key, window_seconds + 60)

            # Execute pipeline
            results = await pipe.execute()

            # Extract count (index 1 in results)
            current_count = results[1]

            # If we incremented, count includes the new request
            if increment:
                current_count += 1

            allowed = current_count <= max_requests

            if allowed:
                retry_after = 0.0
            else:
                # Get oldest request timestamp to calculate retry_after
                oldest_requests = await self.redis.zrange(redis_key, 0, 0, withscores=True)
                if oldest_requests:
                    oldest_timestamp = oldest_requests[0][1]
                    retry_after = max(0, (oldest_timestamp + window_seconds) - current_time)
                else:
                    retry_after = window_seconds

            return allowed, current_count, retry_after

        except Exception as e:
            # Graceful degradation: allow request if Redis fails
            logger.error(f"Rate limiter Redis error (allowing request): {e}")
            # Track Redis errors for monitoring
            _rate_limit_errors_counter.labels(error_type="redis_failure").inc()
            return True, 0, 0.0

    async def reset(self, key: str):
        """Reset rate limit counter for a key.

        Args:
            key: Key to reset
        """
        if not self.enabled:
            return

        redis_key = f"{self.prefix}{key}"
        try:
            await self.redis.delete(redis_key)
        except Exception as e:
            logger.error(f"Failed to reset rate limit for {key}: {e}")

    async def get_usage(self, key: str, window_seconds: int) -> int:
        """Get current usage count for a key.

        Args:
            key: Key to check
            window_seconds: Time window

        Returns:
            Current number of requests in window
        """
        if not self.enabled:
            return 0

        redis_key = f"{self.prefix}{key}"
        current_time = time.time()
        window_start = current_time - window_seconds

        try:
            # Count requests in current window
            count = await self.redis.zcount(redis_key, window_start, current_time)
            return count
        except Exception as e:
            logger.error(f"Failed to get usage for {key}: {e}")
            return 0


def build_rate_limit_key(
    scope: Literal["user", "workspace", "global", "custom"],
    context: dict,
    custom_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    """Build rate limit key based on scope.

    Args:
        scope: Scope of rate limiting (user, workspace, global, custom)
        context: Request context with user and workspace info
        custom_key: Custom key for "custom" scope
        endpoint: Optional endpoint name to include in key

    Returns:
        Redis key for rate limiting

    Raises:
        ValueError: If scope is invalid or required context missing
    """
    parts = []

    if scope == "user":
        user_info = context.get("user", {})
        user_id = user_info.get("id")
        if not user_id:
            raise ValueError("User ID not found in context for user-scoped rate limiting")
        parts.append(f"user:{user_id}")

    elif scope == "workspace":
        workspace = context.get("ws")
        if not workspace:
            raise ValueError("Workspace not found in context for workspace-scoped rate limiting")
        parts.append(f"ws:{workspace}")

    elif scope == "global":
        parts.append("global")

    elif scope == "custom":
        if not custom_key:
            raise ValueError("custom_key required for custom scope")
        parts.append(custom_key)

    else:
        raise ValueError(f"Invalid rate limit scope: {scope}")

    if endpoint:
        parts.append(endpoint)

    return ":".join(parts)


def rate_limit(
    max_requests: int = None,
    window_seconds: int = None,
    scope: Literal["user", "workspace", "global", "custom"] = "user",
    custom_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    error_message: Optional[str] = None,
):
    """Decorator for rate limiting functions.

    Applies rate limiting to async functions. The decorated function must accept
    a 'context' parameter containing user and workspace information.

    Args:
        max_requests: Maximum requests allowed (default: from env or 100)
        window_seconds: Time window in seconds (default: from env or 60)
        scope: Scope of rate limiting (user, workspace, global, custom)
        custom_key: Custom key for "custom" scope
        endpoint: Endpoint name for key (default: function name)
        error_message: Custom error message when rate limited

    Example:
        @rate_limit(max_requests=10, window_seconds=60, scope="user")
        async def upload_file(path: str, context: dict):
            ...

    Raises:
        RateLimitExceeded: When rate limit is exceeded
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context from kwargs
            context = kwargs.get("context")
            if not context:
                # Try to find context in args (common pattern)
                for arg in args:
                    if isinstance(arg, dict) and ("user" in arg or "ws" in arg):
                        context = arg
                        break

            if not context:
                logger.warning(f"Rate limit decorator on {func.__name__} but no context found - allowing request")
                return await func(*args, **kwargs)

            # Get rate limiter from store (stored in context during request setup)
            # If not available, allow request (graceful degradation)
            rate_limiter = context.get("_rate_limiter")
            if not rate_limiter:
                logger.debug(f"No rate limiter in context for {func.__name__} - allowing request")
                return await func(*args, **kwargs)

            # Determine limits (env vars as fallback)
            _max_requests = max_requests
            if _max_requests is None:
                env_key = f"HYPHA_RATE_LIMIT_{endpoint or func.__name__}_MAX".upper()
                _max_requests = int(os.environ.get(env_key, "100"))

            _window_seconds = window_seconds
            if _window_seconds is None:
                env_key = f"HYPHA_RATE_LIMIT_{endpoint or func.__name__}_WINDOW".upper()
                _window_seconds = int(os.environ.get(env_key, "60"))

            # Build rate limit key
            try:
                key = build_rate_limit_key(
                    scope=scope,
                    context=context,
                    custom_key=custom_key,
                    endpoint=endpoint or func.__name__,
                )
            except ValueError as e:
                logger.error(f"Failed to build rate limit key: {e}")
                return await func(*args, **kwargs)

            # Check rate limit
            allowed, count, retry_after = await rate_limiter.check_rate_limit(
                key=key,
                max_requests=_max_requests,
                window_seconds=_window_seconds,
                increment=True,
            )

            if not allowed:
                # Track rate limit exceeded for monitoring
                endpoint_name = endpoint or func.__name__
                _rate_limit_exceeded_counter.labels(endpoint=endpoint_name).inc()

                message = error_message or (
                    f"Rate limit exceeded: {count}/{_max_requests} requests in {_window_seconds}s. "
                    f"Retry after {retry_after:.1f} seconds."
                )
                logger.warning(f"Rate limit exceeded for {key}: {message}")
                raise RateLimitExceeded(message, retry_after=retry_after)

            # Execute function
            return await func(*args, **kwargs)

        return wrapper
    return decorator
