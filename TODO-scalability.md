# Scalability Improvements TODO

This document tracks planned scalability improvements for Hypha's RPC message routing system.

## Completed ✅

### 1. True Local Shortcut in Event Bus
**Status:** Implemented
**Location:** `hypha/core/__init__.py` - `RedisEventBus.emit()`
**Impact:** High
**Effort:** Low

Modified `emit()` to skip Redis entirely for targeted messages to local clients. When a message is addressed to a client connected to the same Hypha server instance, it now bypasses Redis pub/sub and delivers directly via the internal `_redis_event_bus` where RPC message handlers are registered.

**Implementation notes:**

- Uses `is_local_client(workspace, client_id)` to check if target is local
- Emits to `_redis_event_bus` (not `_local_event_bus`) to ensure handlers receive messages
- Handlers are registered via `RedisEventBus.on()` which uses `_redis_event_bus`
- The `_local_event_bus` is separate and used only for `on_local()`/`emit_local()` methods

**Expected improvement:** 50-80% reduction in Redis traffic for co-located client communication.

### 2. HTTP Client Connection Pooling
**Status:** Implemented
**Location:** `hypha_rpc/http_client.py` - `HTTPStreamingRPCConnection._create_http_client()`
**Impact:** Medium
**Effort:** Low

Added `httpx.Limits` configuration for connection pooling:
- `max_connections=100` - Maximum total connections
- `max_keepalive_connections=20` - Connections kept alive for reuse
- `keepalive_expiry=30.0` - Keep-alive duration in seconds

**Expected improvement:** Reduced connection overhead for HTTP transport, especially under high request volume.

---

## Planned Improvements

### 3. Reduce Pattern Subscriptions (High Priority)
**Status:** Not Started
**Location:** `hypha/core/__init__.py` - Pattern subscription logic
**Impact:** High
**Effort:** Medium

**Problem:** Each client creates a dedicated pattern subscription (`targeted:{workspace}/{client_id}:*`), leading to O(n) patterns per server where n = number of clients. Redis PSUBSCRIBE has O(N) complexity for pattern matching.

**Solution:** Aggregate patterns to workspace-level:
```python
# Instead of: targeted:{workspace}/{client_id}:*
# Use: targeted:{workspace}/*
# Then filter in _process_message based on registered local clients
```

**Changes needed:**
1. Modify `subscribe_to_client_events()` to use workspace-level patterns
2. Update `_process_message()` to filter by registered local clients
3. Add workspace-level pattern tracking

**Expected improvement:** Pattern count reduced from O(clients) to O(workspaces).

### 4. Use Redis Streams Instead of Pub/Sub (Medium Priority)
**Status:** Not Started
**Location:** `hypha/core/__init__.py`
**Impact:** High
**Effort:** High

**Problem:** Redis Pub/Sub doesn't guarantee message delivery if a subscriber is temporarily disconnected. Pattern matching performance degrades with many patterns.

**Solution:** Replace pattern-based pub/sub with Redis Streams:
- Each client gets a dedicated stream: `stream:{workspace}/{client_id}`
- Use `XREAD` with consumer groups for reliable delivery
- Built-in message persistence and replay capability

**Trade-offs:**
- More complex implementation
- Higher memory usage (streams persist messages)
- Better reliability and message ordering guarantees

### 5. Batch Message Processing (Medium Priority)
**Status:** Not Started
**Location:** `hypha/core/__init__.py` - `_subscribe_redis()`
**Impact:** Medium
**Effort:** Medium

**Problem:** Messages are processed one-by-one, creating task overhead per message.

**Solution:** Batch multiple messages before processing:
```python
async def _process_batch(self, messages, semaphore):
    """Process multiple messages in a single async context."""
    async with semaphore:
        for msg in messages:
            await self._process_single(msg)
```

**Changes needed:**
1. Accumulate messages in a small window (e.g., 10ms or 100 messages)
2. Process batch together with single semaphore acquisition
3. Balance latency vs throughput

### 6. Shard Redis by Workspace (Low Priority)
**Status:** Not Started
**Location:** Architecture change
**Impact:** High
**Effort:** Very High

**Problem:** Single Redis instance becomes bottleneck at very large scale.

**Solution:** Use Redis Cluster with workspace-based key distribution:
- Hash workspace names to determine Redis shard
- Each Hypha replica subscribes only to relevant workspace shards
- Requires load balancer awareness for workspace affinity

**Considerations:**
- Only needed for very large deployments (10,000+ concurrent users)
- Increases operational complexity
- May require workspace migration tooling

### 7. WebSocket Multiplexing (Low Priority)
**Status:** Not Started
**Location:** Protocol layer
**Impact:** Medium
**Effort:** Medium

**Problem:** Each RPC call uses separate WebSocket frame, creating overhead for high-frequency small messages.

**Solution:** Multiplex multiple RPC messages into single WebSocket frames:
- Buffer small messages for a short window
- Combine into single frame with message boundaries
- Decompose on receiving end

**Trade-offs:**
- Adds small latency (buffering window)
- Reduces frame overhead for chatty clients
- Requires client library changes

---

## Performance Benchmarks Needed

Before implementing further improvements, establish baseline metrics:

1. **Message throughput:** Messages/second at various client counts
2. **Latency distribution:** P50, P95, P99 for RPC calls
3. **Redis operations:** Ops/second, bandwidth usage
4. **CPU utilization:** Per-server under load
5. **Memory usage:** Per-client overhead

### Suggested Test Scenarios

1. **Local routing:** Two clients on same server, measure latency
2. **Cross-server routing:** Two clients on different servers
3. **Fan-out:** One client broadcasting to N clients
4. **Concurrent load:** 100/500/1000 simultaneous RPC calls

---

## Configuration Options to Add

Consider adding environment variables for tuning:

```bash
# Event bus optimization
HYPHA_LOCAL_ROUTING_ENABLED=true  # Enable/disable local shortcut
HYPHA_PATTERN_AGGREGATION=workspace  # none|workspace for pattern strategy

# Connection pooling
HYPHA_HTTP_MAX_CONNECTIONS=100
HYPHA_HTTP_KEEPALIVE_CONNECTIONS=20
HYPHA_HTTP_KEEPALIVE_EXPIRY=30

# Batch processing
HYPHA_MESSAGE_BATCH_SIZE=100
HYPHA_MESSAGE_BATCH_TIMEOUT_MS=10
```

---

## Related Files

- `hypha/core/__init__.py` - RedisEventBus, RedisRPCConnection
- `hypha/websocket.py` - WebsocketServer
- `hypha/http_rpc.py` - HTTPStreamingRPCServer
- `hypha_rpc/http_client.py` - HTTP client connection
- `hypha_rpc/websocket_client.py` - WebSocket client connection

## References

- [Redis Pub/Sub Documentation](https://redis.io/docs/manual/pubsub/)
- [Redis Streams Documentation](https://redis.io/docs/data-types/streams/)
- [HTTPX Connection Pooling](https://www.python-httpx.org/advanced/#pool-limit-configuration)
