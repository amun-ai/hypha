"""Redis-backed single-leader election (F6, Phase 0).

An "exactly-one-runner" primitive for the control-plane singletons that must NOT
run on every replica in a multi-replica deployment — the autoscaling monitor
(``apps.py`` ``AutoscalingManager``) and the workspace activity-cleanup loop
(``workspace.py`` ``WorkspaceActivityManager``). Each replica runs a
``LeaderLease``; only the current leader runs the gated work, and the rest serve
all client traffic normally.

Mechanism:
- **Acquire**: ``SET key identity NX PX ttl`` — atomic, only one replica wins.
- **Renew**: a ``WATCH``/``MULTI`` compare-and-extend (``PEXPIRE`` iff we still
  hold the key) — atomic without Lua, so it works on both real Redis and the
  in-process ``fakeredis`` used for single-server/dev mode.
- **Failover**: on leader crash the key expires after ``ttl_ms`` and another
  replica acquires it on its next renew tick.

This is a *best-effort* lease, not a strongly-consistent lock: during failover
there can be a brief window with no leader, and (very briefly, around a missed
renew) two instances could both believe they hold it. The gated work is
therefore required to be **idempotent / overlap-tolerant** — which the autoscaling
and cleanup loops are (Redis is the source of truth and the operations are
idempotent). See ``docs/multi-replica-design.md`` §3.1.
"""
import asyncio
import logging

logger = logging.getLogger("leader")


class LeaderLease:
    """Best-effort single-leader election over a shared Redis key."""

    def __init__(
        self,
        redis,
        identity,
        *,
        key: str = "hypha:leader",
        ttl_ms: int = 15000,
        renew_interval: float = 5.0,
    ):
        """Create a leader lease.

        Args:
            redis: An async Redis client (real ``redis.asyncio`` or ``fakeredis``).
            identity: Unique identifier for this replica (e.g. the store's
                ``server_id``). Used as the lock value so renew/release only
                affect a lease we actually hold.
            key: The shared Redis key all replicas contend on.
            ttl_ms: Lease lifetime in milliseconds. Must exceed two renew
                intervals so a single missed renew does not drop the lease.
            renew_interval: Seconds between renew/acquire attempts.
        """
        if ttl_ms <= renew_interval * 1000 * 2:
            raise ValueError(
                f"ttl_ms ({ttl_ms}) must exceed 2x renew_interval "
                f"({renew_interval * 1000 * 2} ms) so one missed renew does not "
                "drop the lease"
            )
        self._redis = redis
        self._identity = (
            identity if isinstance(identity, bytes) else str(identity).encode()
        )
        self._key = key
        self._ttl_ms = int(ttl_ms)
        self._renew_interval = renew_interval
        self._is_leader = False
        self._task = None
        self._stopped = False

    def is_leader(self) -> bool:
        """Whether this replica currently believes it holds the lease.

        Updated on each renew tick; may lag reality by up to ``renew_interval``.
        Gated work must tolerate brief overlap (see module docstring).
        """
        return self._is_leader

    async def _acquire(self) -> bool:
        """Atomically acquire the lease iff it is currently free."""
        ok = await self._redis.set(
            self._key, self._identity, nx=True, px=self._ttl_ms
        )
        return bool(ok)

    async def _contend(self) -> bool:
        """Ensure leadership in one atomic step: extend if we hold it, acquire if
        it is free, yield if another replica holds it (CAS via WATCH/MULTI).

        This single operation covers both the leader (extend) and follower
        (acquire-if-free) states, so the renew loop need not track which branch
        it is in — avoiding a class of bugs where a key we already own is not
        recognized by a plain ``SET NX`` acquire. Returns whether we hold it
        afterward.
        """
        async with self._redis.pipeline() as pipe:
            try:
                await pipe.watch(self._key)
                val = await pipe.get(self._key)
                if val == self._identity:
                    pipe.multi()
                    pipe.pexpire(self._key, self._ttl_ms)
                    await pipe.execute()
                    return True
                await pipe.unwatch()
                if val is None:
                    # Lease lapsed/free — try to take it.
                    return await self._acquire()
                return False
            except Exception as e:
                # WatchError (key changed under us) or a transient Redis error:
                # treat as "not held" and re-contend on the next tick.
                logger.debug("leader renew transaction failed: %s", e)
                return False

    async def _release(self) -> None:
        """Delete the key iff we hold it (CAS), so we don't drop a peer's lease."""
        async with self._redis.pipeline() as pipe:
            try:
                await pipe.watch(self._key)
                val = await pipe.get(self._key)
                if val == self._identity:
                    pipe.multi()
                    pipe.delete(self._key)
                    await pipe.execute()
                else:
                    await pipe.unwatch()
            except Exception as e:
                logger.debug("leader release transaction failed: %s", e)

    async def _loop(self) -> None:
        while not self._stopped:
            try:
                was_leader = self._is_leader
                held = await self._contend()
                if held and not was_leader:
                    logger.info("Acquired leader lease (%s)", self._identity.decode())
                elif was_leader and not held:
                    logger.info("Lost leader lease (%s)", self._identity.decode())
                self._is_leader = held
            except Exception as e:
                logger.warning("leader loop error: %s", e)
                self._is_leader = False
            await asyncio.sleep(self._renew_interval)

    async def start(self) -> None:
        """Begin contending for leadership and start the renew loop.

        Does one immediate acquire attempt so ``is_leader()`` is meaningful
        immediately after ``start()`` returns.
        """
        self._stopped = False
        try:
            self._is_leader = await self._contend()
            if self._is_leader:
                logger.info("Acquired leader lease (%s)", self._identity.decode())
        except Exception as e:
            logger.warning("initial leader acquire failed: %s", e)
            self._is_leader = False
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the renew loop and release the lease if we hold it."""
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._is_leader:
            try:
                await self._release()
            except Exception as e:
                logger.debug("leader release on stop failed: %s", e)
        self._is_leader = False
