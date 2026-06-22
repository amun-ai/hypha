"""Resource limits and blocking for Hypha server.

Provides per-client, per-user, and per-workspace resource consumption limits
plus admin-controlled blocklists to protect server stability.
"""

import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlmodel import SQLModel, Field, select
from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("resource-limits")


class BlockEntry(SQLModel, table=True):
    """Durable (Postgres) record of a blocked user or IP.

    This is the SOURCE OF TRUTH for the blocklist; Redis is only a fast cache that
    is repopulated from this table on startup (see
    ResourceLimitsManager.sync_blocklist_to_redis). That makes blocks survive a Redis
    reset/flush/data-loss (e.g. HYPHA_RESET_REDIS=true, or a Redis restart without
    persistence) — not just pod restarts.
    """

    __tablename__ = "blocklist"

    kind: str = Field(primary_key=True)  # "user" | "ip"
    key: str = Field(primary_key=True)  # the user_id or IP address
    reason: str = ""
    # None = permanent; otherwise the block is inactive once now() > expires_at.
    expires_at: Optional[datetime] = Field(default=None, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class TokenBucket:
    """Token-bucket rate limiter for per-client message throttling."""

    __slots__ = ("rate", "burst", "_tokens", "_last")

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


class ResourceLimitsManager:
    """Manages per-client resource limits and blocklists.

    In-memory counters (per-process, O(1)) for connection and service limits.
    Redis-backed blocklists (cross-instance) for admin blocking.
    """

    def __init__(self, redis=None, sql_engine=None):
        self._redis = redis
        # Durable blocklist store (Postgres). When present, it is the source of
        # truth; Redis is a cache. When absent (e.g. tests/dev with no DB), the
        # blocklist degrades to Redis-only (its prior behavior).
        self._sql_engine = sql_engine

        # Limits from environment (with sensible defaults)
        self.max_connections_per_user = int(
            os.environ.get("HYPHA_MAX_CONNECTIONS_PER_USER", "200")
        )
        self.max_clients_per_workspace = int(
            os.environ.get("HYPHA_MAX_CLIENTS_PER_WORKSPACE", "1000")
        )
        self.max_services_per_client = int(
            os.environ.get("HYPHA_MAX_SERVICES_PER_CLIENT", "50")
        )
        self.ws_msg_rate_limit = float(
            os.environ.get("HYPHA_WS_MSG_RATE_LIMIT", "200")
        )
        self.ws_msg_burst_limit = int(
            os.environ.get("HYPHA_WS_MSG_BURST_LIMIT", "500")
        )

        # In-memory counters (per-process — WS connections are sticky)
        self._connections_per_user: dict[str, int] = {}
        self._connections_per_workspace: dict[str, int] = {}
        self._services_per_client: dict[str, int] = {}

        # Per-client WS message rate limiters
        self._ws_rate_limiters: dict[str, TokenBucket] = {}

        # Track user_id and workspace per conn_key for cleanup
        self._conn_user: dict[str, str] = {}  # conn_key -> user_id
        self._conn_workspace: dict[str, str] = {}  # conn_key -> workspace

    # ── Connection limits ──────────────────────────────────────────────

    async def check_blocked(
        self, user_id: str = None, ip: str = None
    ) -> Optional[str]:
        """Return a reason string if the user or IP is blocklisted, else None.

        Enforced on EVERY request boundary — WebSocket connect AND every HTTP/MCP
        request (via store.login_optional) — so a blocked principal cannot keep
        hammering stateless HTTP `/rpc` + `/mcp` endpoints. (Previously the blocklist
        was only consulted in check_connection_allowed, i.e. WebSocket connect, so
        block_user/block_ip had no effect on the HTTP/MCP vector.) Fails OPEN if Redis
        is unavailable — never lock everyone out on a Redis blip.
        """
        if self._redis is None:
            return None
        try:
            if user_id and await self._redis.exists(f"blocked:user:{user_id}"):
                reason = await self._redis.get(f"blocked:user:{user_id}")
                if isinstance(reason, bytes):
                    reason = reason.decode("utf-8", errors="replace")
                return f"User is blocked: {reason or 'no reason given'}"
            if ip and await self._redis.exists(f"blocked:ip:{ip}"):
                reason = await self._redis.get(f"blocked:ip:{ip}")
                if isinstance(reason, bytes):
                    reason = reason.decode("utf-8", errors="replace")
                return f"IP is blocked: {reason or 'no reason given'}"
        except Exception as e:
            logger.warning("Failed to check blocklist: %s", e)
            # Fail open — don't block users if Redis is down
        return None

    async def check_connection_allowed(
        self, user_id: str, ip: str, workspace: str
    ) -> Optional[str]:
        """Check if a new connection is allowed.

        Returns None if allowed, or an error message string if rejected.
        """
        # Check blocklists (Redis, cross-instance)
        blocked = await self.check_blocked(user_id, ip)
        if blocked:
            return blocked

        # Check per-user connection limit
        user_count = self._connections_per_user.get(user_id, 0)
        if user_count >= self.max_connections_per_user:
            return (
                f"Connection limit exceeded: user {user_id} has "
                f"{user_count}/{self.max_connections_per_user} connections"
            )

        # Check per-workspace client limit
        ws_count = self._connections_per_workspace.get(workspace, 0)
        if ws_count >= self.max_clients_per_workspace:
            return (
                f"Connection limit exceeded: workspace {workspace} has "
                f"{ws_count}/{self.max_clients_per_workspace} clients"
            )

        return None

    def on_client_connected(self, conn_key: str, user_id: str, workspace: str):
        """Increment counters when a client connects.

        If conn_key already exists (reconnection with the same client_id),
        decrement the old counters first to avoid double-counting.
        """
        old_user = self._conn_user.get(conn_key)
        old_workspace = self._conn_workspace.get(conn_key)
        if old_user is not None:
            self._decrement_connection(old_user, old_workspace)

        self._connections_per_user[user_id] = (
            self._connections_per_user.get(user_id, 0) + 1
        )
        self._connections_per_workspace[workspace] = (
            self._connections_per_workspace.get(workspace, 0) + 1
        )
        self._conn_user[conn_key] = user_id
        self._conn_workspace[conn_key] = workspace

    def on_client_disconnected(self, conn_key: str):
        """Decrement counters when a client disconnects."""
        user_id = self._conn_user.pop(conn_key, None)
        workspace = self._conn_workspace.pop(conn_key, None)

        self._decrement_connection(user_id, workspace)

        # Clean up rate limiter
        self._ws_rate_limiters.pop(conn_key, None)

    def _decrement_connection(self, user_id: str, workspace: str):
        """Decrement user and workspace connection counters.

        This is used both by on_client_disconnected (normal path) and directly
        when a superseded connection needs to release its counter without
        touching the conn_key maps (which belong to the newer connection).
        """
        if user_id and user_id in self._connections_per_user:
            self._connections_per_user[user_id] = max(
                0, self._connections_per_user[user_id] - 1
            )
            if self._connections_per_user[user_id] == 0:
                del self._connections_per_user[user_id]

        if workspace and workspace in self._connections_per_workspace:
            self._connections_per_workspace[workspace] = max(
                0, self._connections_per_workspace[workspace] - 1
            )
            if self._connections_per_workspace[workspace] == 0:
                del self._connections_per_workspace[workspace]

    # ── Service limits ─────────────────────────────────────────────────

    def check_service_registration_allowed(self, client_id: str) -> Optional[str]:
        """Check if a client can register another service.

        Returns None if allowed, or an error message string if rejected.
        """
        count = self._services_per_client.get(client_id, 0)
        if count >= self.max_services_per_client:
            return (
                f"Service limit exceeded: client {client_id} has "
                f"{count}/{self.max_services_per_client} services"
            )
        return None

    def on_service_registered(self, client_id: str):
        """Increment service counter for a client."""
        self._services_per_client[client_id] = (
            self._services_per_client.get(client_id, 0) + 1
        )

    def on_service_unregistered(self, client_id: str):
        """Decrement service counter for a client."""
        if client_id in self._services_per_client:
            self._services_per_client[client_id] = max(
                0, self._services_per_client[client_id] - 1
            )
            if self._services_per_client[client_id] == 0:
                del self._services_per_client[client_id]

    def on_client_services_cleared(self, client_id: str):
        """Remove all service counts for a client (on delete_client)."""
        self._services_per_client.pop(client_id, None)

    # ── WebSocket message rate limiting ────────────────────────────────

    def check_message_allowed(self, conn_key: str) -> bool:
        """Check if a WebSocket message is allowed (token bucket).

        Returns True if allowed, False if rate limited.
        """
        bucket = self._ws_rate_limiters.get(conn_key)
        if bucket is None:
            bucket = TokenBucket(self.ws_msg_rate_limit, self.ws_msg_burst_limit)
            self._ws_rate_limiters[conn_key] = bucket
        return bucket.consume()

    # ── Admin blocking ─────────────────────────────────────────────────

    @staticmethod
    def _ttl_to_expiry(duration: int):
        """duration seconds → (ttl_seconds, expires_at). 0 → 30 days (legacy)."""
        ttl = duration if duration and duration > 0 else 30 * 86400
        return ttl, datetime.now(timezone.utc) + timedelta(seconds=ttl)

    async def init_blocklist_db(self):
        """Create the blocklist table if it does not exist (idempotent)."""
        if self._sql_engine is None:
            return
        async with self._sql_engine.begin() as conn:
            await conn.run_sync(BlockEntry.metadata.create_all)

    async def _sql_put_block(self, kind, key, reason, expires_at):
        async with AsyncSession(self._sql_engine) as session:
            await session.execute(
                sql_delete(BlockEntry).where(
                    BlockEntry.kind == kind, BlockEntry.key == key
                )
            )
            session.add(
                BlockEntry(
                    kind=kind, key=key, reason=reason or "blocked",
                    expires_at=expires_at,
                )
            )
            await session.commit()

    async def _sql_remove_block(self, kind, key):
        async with AsyncSession(self._sql_engine) as session:
            await session.execute(
                sql_delete(BlockEntry).where(
                    BlockEntry.kind == kind, BlockEntry.key == key
                )
            )
            await session.commit()

    async def _set_redis_block(self, kind, key, reason, ttl):
        if self._redis is not None:
            await self._redis.set(f"blocked:{kind}:{key}", reason or "blocked", ex=ttl)

    async def _del_redis_block(self, kind, key):
        if self._redis is not None:
            await self._redis.delete(f"blocked:{kind}:{key}")

    async def block_user(self, user_id: str, duration: int = 3600, reason: str = ""):
        """Block a user. Duration in seconds (0 = 30 days). Durable (Postgres) +
        Redis cache."""
        if self._sql_engine is None and self._redis is None:
            raise RuntimeError("No durable store or Redis available for blocklist")
        ttl, expires_at = self._ttl_to_expiry(duration)
        if self._sql_engine is not None:
            await self._sql_put_block("user", user_id, reason, expires_at)
        await self._set_redis_block("user", user_id, reason, ttl)
        logger.info("Blocked user %s for %ds: %s", user_id, ttl, reason)

    async def unblock_user(self, user_id: str):
        """Unblock a user (durable store + Redis cache)."""
        if self._sql_engine is not None:
            await self._sql_remove_block("user", user_id)
        await self._del_redis_block("user", user_id)
        logger.info("Unblocked user %s", user_id)

    async def block_ip(self, ip: str, duration: int = 3600, reason: str = ""):
        """Block an IP address. Duration in seconds (0 = 30 days). Durable + cache."""
        if self._sql_engine is None and self._redis is None:
            raise RuntimeError("No durable store or Redis available for blocklist")
        ttl, expires_at = self._ttl_to_expiry(duration)
        if self._sql_engine is not None:
            await self._sql_put_block("ip", ip, reason, expires_at)
        await self._set_redis_block("ip", ip, reason, ttl)
        logger.info("Blocked IP %s for %ds: %s", ip, ttl, reason)

    async def unblock_ip(self, ip: str):
        """Unblock an IP address (durable store + Redis cache)."""
        if self._sql_engine is not None:
            await self._sql_remove_block("ip", ip)
        await self._del_redis_block("ip", ip)
        logger.info("Unblocked IP %s", ip)

    async def list_blocked(self) -> dict:
        """List currently-blocked users and IPs (from the durable store if present,
        else Redis). ttl_seconds = -1 means a permanent block."""
        result = {"users": [], "ips": []}
        if self._sql_engine is not None:
            now = datetime.now(timezone.utc)
            async with AsyncSession(self._sql_engine) as session:
                rows = (await session.execute(select(BlockEntry))).scalars().all()
            for r in rows:
                exp = r.expires_at
                if exp is not None:
                    if exp.tzinfo is None:  # older naive rows → treat as UTC
                        exp = exp.replace(tzinfo=timezone.utc)
                    if exp <= now:
                        continue  # expired — not active
                    ttl = int((exp - now).total_seconds())
                else:
                    ttl = -1  # permanent
                bucket = "users" if r.kind == "user" else "ips"
                result[bucket].append(
                    {"id": r.key, "reason": r.reason, "ttl_seconds": ttl}
                )
            return result
        # No durable store — scan Redis (legacy behavior).
        if self._redis is None:
            return result
        for prefix, key in [("blocked:user:", "users"), ("blocked:ip:", "ips")]:
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=f"{prefix}*", count=100
                )
                for k in keys:
                    if isinstance(k, bytes):
                        k = k.decode("utf-8", errors="replace")
                    entity_id = k[len(prefix):]
                    reason = await self._redis.get(k)
                    if isinstance(reason, bytes):
                        reason = reason.decode("utf-8", errors="replace")
                    ttl = await self._redis.ttl(k)
                    result[key].append(
                        {"id": entity_id, "reason": reason, "ttl_seconds": ttl}
                    )
                if cursor == 0:
                    break
        return result

    async def sync_blocklist_to_redis(self):
        """Repopulate the Redis cache from the Postgres source of truth (call on
        startup). Makes blocks survive a Redis reset/flush/data-loss (e.g.
        HYPHA_RESET_REDIS=true) and prunes expired rows. No-op without both stores."""
        if self._sql_engine is None or self._redis is None:
            return
        now = datetime.now(timezone.utc)
        async with AsyncSession(self._sql_engine) as session:
            await session.execute(
                sql_delete(BlockEntry).where(
                    BlockEntry.expires_at.is_not(None),
                    BlockEntry.expires_at <= now,
                )
            )
            await session.commit()
            rows = (await session.execute(select(BlockEntry))).scalars().all()
        restored = 0
        for r in rows:
            exp = r.expires_at
            if exp is not None and exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            ttl = int((exp - now).total_seconds()) if exp is not None else 30 * 86400
            if ttl <= 0:
                continue
            await self._redis.set(
                f"blocked:{r.kind}:{r.key}", r.reason or "blocked", ex=ttl
            )
            restored += 1
        if restored:
            logger.info(
                "Repopulated %d blocklist entries from Postgres into Redis", restored
            )

    # ── Connection reconciliation ─────────────────────────────────────

    def reconcile_connections(self, active_conn_keys: set):
        """Reconcile connection counters against actual active connections.

        This is a self-healing mechanism that corrects counter drift caused
        by missed disconnection events (e.g., reconnection races, unclean
        shutdowns).  It should be called periodically (e.g., every few
        minutes) by the WebSocket server.

        Two-phase approach:
        1. Remove stale entries from _conn_user/_conn_workspace
        2. Rebuild per-user/per-workspace counters from the cleaned maps
        """
        from collections import Counter

        corrections = 0

        # Phase 1: Remove stale conn_keys no longer in active WebSockets
        stale_keys = set(self._conn_user.keys()) - active_conn_keys
        for conn_key in stale_keys:
            self._conn_user.pop(conn_key, None)
            self._conn_workspace.pop(conn_key, None)
            self._ws_rate_limiters.pop(conn_key, None)

        # Phase 2: Rebuild counters from the authoritative conn maps
        correct_user = Counter(self._conn_user.values())
        correct_workspace = Counter(self._conn_workspace.values())

        # Detect and fix user counter drift
        all_users = set(self._connections_per_user.keys()) | set(correct_user.keys())
        for user_id in all_users:
            tracked = self._connections_per_user.get(user_id, 0)
            actual = correct_user.get(user_id, 0)
            if tracked != actual:
                corrections += 1
                if actual == 0:
                    self._connections_per_user.pop(user_id, None)
                else:
                    self._connections_per_user[user_id] = actual

        # Detect and fix workspace counter drift
        all_ws = set(self._connections_per_workspace.keys()) | set(
            correct_workspace.keys()
        )
        for ws in all_ws:
            tracked = self._connections_per_workspace.get(ws, 0)
            actual = correct_workspace.get(ws, 0)
            if tracked != actual:
                corrections += 1
                if actual == 0:
                    self._connections_per_workspace.pop(ws, None)
                else:
                    self._connections_per_workspace[ws] = actual

        if stale_keys or corrections:
            logger.warning(
                "Reconciled connections: removed %d stale entries, "
                "fixed %d counter(s)",
                len(stale_keys),
                corrections,
            )
        return len(stale_keys) + corrections

    # ── Resource usage snapshot ────────────────────────────────────────

    def get_resource_usage(self) -> dict:
        """Get a snapshot of current resource usage counters."""
        return {
            "connections_per_user": dict(self._connections_per_user),
            "connections_per_workspace": dict(self._connections_per_workspace),
            "services_per_client": dict(self._services_per_client),
            "active_rate_limiters": len(self._ws_rate_limiters),
            "limits": {
                "max_connections_per_user": self.max_connections_per_user,
                "max_clients_per_workspace": self.max_clients_per_workspace,
                "max_services_per_client": self.max_services_per_client,
                "ws_msg_rate_limit": self.ws_msg_rate_limit,
                "ws_msg_burst_limit": self.ws_msg_burst_limit,
            },
        }
