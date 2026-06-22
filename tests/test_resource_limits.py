"""Tests for resource limits and admin blocking."""

import pytest
import pytest_asyncio

from hypha.resource_limits import ResourceLimitsManager, TokenBucket


# ── TokenBucket unit tests ─────────────────────────────────────────────


class TestTokenBucket:
    """Tests for the TokenBucket rate limiter."""

    def test_allows_within_burst(self):
        bucket = TokenBucket(rate=10, burst=5)
        for _ in range(5):
            assert bucket.consume() is True

    def test_rejects_after_burst(self):
        bucket = TokenBucket(rate=10, burst=3)
        for _ in range(3):
            assert bucket.consume() is True
        assert bucket.consume() is False

    def test_refills_over_time(self):
        bucket = TokenBucket(rate=1000, burst=1)
        assert bucket.consume() is True
        assert bucket.consume() is False
        # Simulate time passing by manipulating _last
        bucket._last -= 0.01  # 10ms ago, at 1000/s rate = 10 tokens
        assert bucket.consume() is True


# ── ResourceLimitsManager unit tests ───────────────────────────────────


class TestConnectionLimits:
    """Tests for per-user and per-workspace connection limits."""

    @pytest.fixture
    def rl(self):
        manager = ResourceLimitsManager(redis=None)
        manager.max_connections_per_user = 3
        manager.max_clients_per_workspace = 5
        return manager

    @pytest.mark.asyncio
    async def test_allows_connections_within_limit(self, rl):
        result = await rl.check_connection_allowed("user1", "1.2.3.4", "ws1")
        assert result is None  # No rejection

    @pytest.mark.asyncio
    async def test_rejects_after_user_limit(self, rl):
        # Simulate 3 connections
        for i in range(3):
            rl.on_client_connected(f"ws1/client{i}", "user1", "ws1")

        result = await rl.check_connection_allowed("user1", "1.2.3.4", "ws1")
        assert result is not None
        assert "Connection limit exceeded" in result
        assert "user1" in result

    @pytest.mark.asyncio
    async def test_rejects_after_workspace_limit(self, rl):
        rl.max_connections_per_user = 100  # Don't trigger user limit
        for i in range(5):
            rl.on_client_connected(f"ws1/client{i}", f"user{i}", "ws1")

        result = await rl.check_connection_allowed("user99", "1.2.3.4", "ws1")
        assert result is not None
        assert "workspace ws1" in result

    @pytest.mark.asyncio
    async def test_different_users_have_separate_limits(self, rl):
        for i in range(3):
            rl.on_client_connected(f"ws1/client{i}", "user1", "ws1")

        # user1 is at limit
        assert await rl.check_connection_allowed("user1", "1.2.3.4", "ws1") is not None
        # user2 is not
        assert await rl.check_connection_allowed("user2", "1.2.3.4", "ws1") is None

    @pytest.mark.asyncio
    async def test_counter_decrements_on_disconnect(self, rl):
        rl.on_client_connected("ws1/client0", "user1", "ws1")
        rl.on_client_connected("ws1/client1", "user1", "ws1")
        rl.on_client_connected("ws1/client2", "user1", "ws1")

        # At limit
        assert await rl.check_connection_allowed("user1", "1.2.3.4", "ws1") is not None

        # Disconnect one
        rl.on_client_disconnected("ws1/client0")

        # Now allowed again
        assert await rl.check_connection_allowed("user1", "1.2.3.4", "ws1") is None

    @pytest.mark.asyncio
    async def test_double_disconnect_no_negative(self, rl):
        rl.on_client_connected("ws1/client0", "user1", "ws1")
        rl.on_client_disconnected("ws1/client0")
        rl.on_client_disconnected("ws1/client0")  # Should not go negative

        assert rl._connections_per_user.get("user1", 0) == 0
        assert rl._connections_per_workspace.get("ws1", 0) == 0


class TestServiceLimits:
    """Tests for per-client service registration limits."""

    @pytest.fixture
    def rl(self):
        manager = ResourceLimitsManager(redis=None)
        manager.max_services_per_client = 3
        return manager

    def test_allows_services_within_limit(self, rl):
        assert rl.check_service_registration_allowed("ws1/client1") is None

    def test_rejects_after_limit(self, rl):
        for _ in range(3):
            rl.on_service_registered("ws1/client1")

        result = rl.check_service_registration_allowed("ws1/client1")
        assert result is not None
        assert "Service limit exceeded" in result

    def test_different_clients_have_separate_limits(self, rl):
        for _ in range(3):
            rl.on_service_registered("ws1/client1")

        # client1 at limit, client2 not
        assert rl.check_service_registration_allowed("ws1/client1") is not None
        assert rl.check_service_registration_allowed("ws1/client2") is None

    def test_clear_resets_counter(self, rl):
        for _ in range(3):
            rl.on_service_registered("ws1/client1")

        rl.on_client_services_cleared("ws1/client1")
        assert rl.check_service_registration_allowed("ws1/client1") is None

    def test_unregister_decrements(self, rl):
        for _ in range(3):
            rl.on_service_registered("ws1/client1")

        rl.on_service_unregistered("ws1/client1")
        assert rl.check_service_registration_allowed("ws1/client1") is None


class TestMessageRateLimiting:
    """Tests for per-client WebSocket message rate limiting."""

    def test_allows_messages_within_burst(self):
        rl = ResourceLimitsManager(redis=None)
        rl.ws_msg_rate_limit = 10
        rl.ws_msg_burst_limit = 5

        for _ in range(5):
            assert rl.check_message_allowed("ws1/client1") is True

    def test_rejects_after_burst(self):
        rl = ResourceLimitsManager(redis=None)
        rl.ws_msg_rate_limit = 10
        rl.ws_msg_burst_limit = 3

        for _ in range(3):
            assert rl.check_message_allowed("ws1/client1") is True
        assert rl.check_message_allowed("ws1/client1") is False

    def test_different_clients_have_separate_buckets(self):
        rl = ResourceLimitsManager(redis=None)
        rl.ws_msg_rate_limit = 10
        rl.ws_msg_burst_limit = 1

        assert rl.check_message_allowed("ws1/client1") is True
        assert rl.check_message_allowed("ws1/client1") is False
        # Different client should have its own bucket
        assert rl.check_message_allowed("ws1/client2") is True

    def test_rate_limiter_cleaned_on_disconnect(self):
        rl = ResourceLimitsManager(redis=None)
        rl.on_client_connected("ws1/client1", "user1", "ws1")
        rl.check_message_allowed("ws1/client1")  # Creates a bucket
        assert "ws1/client1" in rl._ws_rate_limiters

        rl.on_client_disconnected("ws1/client1")
        assert "ws1/client1" not in rl._ws_rate_limiters


class TestBlocklist:
    """Tests for admin blocking with FakeRedis."""

    @pytest_asyncio.fixture
    async def rl_with_redis(self):
        from fakeredis import aioredis

        redis = aioredis.FakeRedis()
        manager = ResourceLimitsManager(redis=redis)
        yield manager
        await redis.flushall()
        await redis.aclose()

    @pytest.mark.asyncio
    async def test_block_and_check_user(self, rl_with_redis):
        rl = rl_with_redis
        await rl.block_user("bad_user", duration=60, reason="spam")

        result = await rl.check_connection_allowed("bad_user", "1.2.3.4", "ws1")
        assert result is not None
        assert "User is blocked" in result

    @pytest.mark.asyncio
    async def test_unblock_user(self, rl_with_redis):
        rl = rl_with_redis
        await rl.block_user("bad_user", duration=60, reason="spam")
        await rl.unblock_user("bad_user")

        result = await rl.check_connection_allowed("bad_user", "1.2.3.4", "ws1")
        assert result is None

    @pytest.mark.asyncio
    async def test_block_and_check_ip(self, rl_with_redis):
        rl = rl_with_redis
        await rl.block_ip("10.0.0.1", duration=60, reason="abuse")

        result = await rl.check_connection_allowed("good_user", "10.0.0.1", "ws1")
        assert result is not None
        assert "IP is blocked" in result

    @pytest.mark.asyncio
    async def test_unblock_ip(self, rl_with_redis):
        rl = rl_with_redis
        await rl.block_ip("10.0.0.1", duration=60, reason="abuse")
        await rl.unblock_ip("10.0.0.1")

        result = await rl.check_connection_allowed("good_user", "10.0.0.1", "ws1")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_blocked(self, rl_with_redis):
        rl = rl_with_redis
        await rl.block_user("user1", duration=60, reason="spam")
        await rl.block_ip("10.0.0.1", duration=120, reason="ddos")

        blocked = await rl.list_blocked()
        assert len(blocked["users"]) == 1
        assert blocked["users"][0]["id"] == "user1"
        assert blocked["users"][0]["reason"] == "spam"
        assert len(blocked["ips"]) == 1
        assert blocked["ips"][0]["id"] == "10.0.0.1"

    @pytest.mark.asyncio
    async def test_block_with_zero_duration_uses_30_days(self, rl_with_redis):
        rl = rl_with_redis
        await rl.block_user("perma_ban", duration=0, reason="permanent")

        # Should be blocked
        result = await rl.check_connection_allowed("perma_ban", "1.2.3.4", "ws1")
        assert result is not None

        # Check TTL is approximately 30 days
        ttl = await rl._redis.ttl("blocked:user:perma_ban")
        assert ttl > 29 * 86400  # At least 29 days


class TestResourceUsage:
    """Tests for resource usage reporting."""

    def test_get_resource_usage(self):
        rl = ResourceLimitsManager(redis=None)
        rl.on_client_connected("ws1/c1", "user1", "ws1")
        rl.on_client_connected("ws1/c2", "user1", "ws1")
        rl.on_client_connected("ws2/c3", "user2", "ws2")
        rl.on_service_registered("ws1/c1")
        rl.on_service_registered("ws1/c1")

        usage = rl.get_resource_usage()
        assert usage["connections_per_user"]["user1"] == 2
        assert usage["connections_per_user"]["user2"] == 1
        assert usage["connections_per_workspace"]["ws1"] == 2
        assert usage["connections_per_workspace"]["ws2"] == 1
        assert usage["services_per_client"]["ws1/c1"] == 2
        assert "max_connections_per_user" in usage["limits"]

    def test_no_store_blocklist_block_raises(self):
        # With neither a durable SQL store nor Redis, a BLOCK has nowhere to persist
        # and must raise (unblock, by contrast, is a graceful no-op).
        rl = ResourceLimitsManager(redis=None, sql_engine=None)
        with pytest.raises(RuntimeError, match="No durable store or Redis"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                rl.block_user("user1")
            )


# ── Blocklist enforcement on the request boundary ──────────────────────


class _FakeRedis:
    """Minimal async Redis stub for blocklist tests."""

    def __init__(self, blocked=None):
        self._b = blocked or {}

    async def exists(self, key):
        return 1 if key in self._b else 0

    async def get(self, key):
        return self._b.get(key)


class TestCheckBlocked:
    """check_blocked is enforced on EVERY request (WS connect AND every HTTP/MCP
    request via store.login_optional), so block_user/block_ip stop a blocked principal
    from hammering stateless HTTP /rpc + /mcp endpoints — not just WebSocket connect."""

    @pytest.mark.asyncio
    async def test_blocked_user_returns_reason(self):
        uid = "google-oauth2|117932459319487424837"
        rl = ResourceLimitsManager(redis=_FakeRedis({f"blocked:user:{uid}": b"abuse"}))
        reason = await rl.check_blocked(uid, "1.2.3.4")
        assert reason is not None and "blocked" in reason.lower()

    @pytest.mark.asyncio
    async def test_blocked_ip_returns_reason(self):
        rl = ResourceLimitsManager(redis=_FakeRedis({"blocked:ip:9.9.9.9": b"bad"}))
        reason = await rl.check_blocked("some-user", "9.9.9.9")
        assert reason is not None and "IP is blocked" in reason

    @pytest.mark.asyncio
    async def test_not_blocked_returns_none(self):
        rl = ResourceLimitsManager(redis=_FakeRedis({}))
        assert await rl.check_blocked("some-user", "1.2.3.4") is None

    @pytest.mark.asyncio
    async def test_fail_open_on_redis_error(self):
        class _Boom:
            async def exists(self, key):
                raise RuntimeError("redis down")

        rl = ResourceLimitsManager(redis=_Boom())
        # Must fail OPEN — a Redis blip must not lock everyone out.
        assert await rl.check_blocked("some-user", "1.2.3.4") is None

    @pytest.mark.asyncio
    async def test_no_redis_returns_none(self):
        rl = ResourceLimitsManager(redis=None)
        assert await rl.check_blocked("some-user", "1.2.3.4") is None

    @pytest.mark.asyncio
    async def test_check_connection_allowed_still_enforces_block(self):
        # Refactor guard: WebSocket connect path still rejects blocked users.
        uid = "google-oauth2|117932459319487424837"
        rl = ResourceLimitsManager(redis=_FakeRedis({f"blocked:user:{uid}": b"abuse"}))
        result = await rl.check_connection_allowed(uid, "1.2.3.4", "ws1")
        assert result is not None and "blocked" in result.lower()


class _RichFakeRedis:
    """Fake async Redis supporting the blocklist cache ops."""

    def __init__(self):
        self.store = {}

    async def set(self, k, v, ex=None):
        self.store[k] = v

    async def get(self, k):
        return self.store.get(k)

    async def delete(self, k):
        self.store.pop(k, None)

    async def exists(self, k):
        return 1 if k in self.store else 0

    async def ttl(self, k):
        return -1

    async def scan(self, cursor=0, match=None, count=100):
        import fnmatch
        keys = [k for k in self.store if match is None or fnmatch.fnmatch(k, match)]
        return 0, keys


class TestDurableBlocklist:
    """Postgres (SQLite here) is the durable source of truth; Redis is a cache
    repopulated from SQL on startup — so admin blocks survive a Redis reset/flush
    (HYPHA_RESET_REDIS=true) / data-loss, not just pod restarts. (0.21.99)"""

    @pytest_asyncio.fixture
    async def engine(self):
        import tempfile
        import os as _os
        from sqlalchemy.ext.asyncio import create_async_engine

        fd, path = tempfile.mkstemp(suffix=".db")
        _os.close(fd)
        eng = create_async_engine(f"sqlite+aiosqlite:///{path}")
        yield eng
        await eng.dispose()
        _os.unlink(path)

    @pytest.mark.asyncio
    async def test_block_survives_redis_flush(self, engine):
        redis = _RichFakeRedis()
        rl = ResourceLimitsManager(redis=redis, sql_engine=engine)
        await rl.init_blocklist_db()

        uid = "google-oauth2|117932459319487424837"
        await rl.block_user(uid, duration=3600, reason="abuse")
        # active via the durable list + the Redis cache
        bl = await rl.list_blocked()
        assert any(u["id"] == uid for u in bl["users"])
        assert await rl.check_blocked(uid) is not None

        # Simulate a Redis FLUSH (what HYPHA_RESET_REDIS=true does on restart).
        redis.store.clear()
        assert await rl.check_blocked(uid) is None  # cache gone…
        # …but the durable SQL record still has it, and boot repopulation restores it.
        bl = await rl.list_blocked()
        assert any(u["id"] == uid for u in bl["users"])
        await rl.sync_blocklist_to_redis()
        assert await rl.check_blocked(uid) is not None  # back in the cache

    @pytest.mark.asyncio
    async def test_unblock_clears_sql_and_redis(self, engine):
        redis = _RichFakeRedis()
        rl = ResourceLimitsManager(redis=redis, sql_engine=engine)
        await rl.init_blocklist_db()
        await rl.block_ip("9.9.9.9", reason="x")
        await rl.unblock_ip("9.9.9.9")
        assert (await rl.list_blocked())["ips"] == []
        assert await rl.check_blocked(None, "9.9.9.9") is None

    @pytest.mark.asyncio
    async def test_expired_block_inactive_and_pruned(self, engine):
        from datetime import datetime, timezone, timedelta
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlmodel import select
        from hypha.resource_limits import BlockEntry

        redis = _RichFakeRedis()
        rl = ResourceLimitsManager(redis=redis, sql_engine=engine)
        await rl.init_blocklist_db()
        async with AsyncSession(engine) as s:
            s.add(BlockEntry(kind="user", key="stale", reason="r",
                             expires_at=datetime.now(timezone.utc) - timedelta(hours=1)))
            await s.commit()
        # expired → not listed as active
        assert all(u["id"] != "stale" for u in (await rl.list_blocked())["users"])
        # sync prunes it from SQL
        await rl.sync_blocklist_to_redis()
        async with AsyncSession(engine) as s:
            rows = (await s.execute(select(BlockEntry))).scalars().all()
        assert all(r.key != "stale" for r in rows)
