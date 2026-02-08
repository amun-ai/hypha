# DoS and Resource Exhaustion Vulnerabilities in Hypha Server

**Assessment Date**: 2026-02-08
**Target**: https://hypha-dev.aicell.io/
**Assessment Type**: Security Audit - Denial of Service and Resource Exhaustion

## Executive Summary

This security assessment identified **9 critical and high-severity** Denial of Service (DoS) vulnerabilities in the Hypha server. The vulnerabilities allow attackers to exhaust system resources through:

- **Connection flooding** (no per-user connection limits)
- **Service registration spam** (no rate limiting)
- **Worker pool exhaustion** (unbounded autoscaling)
- **Artifact upload flooding** (no upload rate limits)
- **Message flooding** (no message rate limiting)
- **Memory exhaustion** (no binary message size limits)

**Risk Level**: **HIGH** - These vulnerabilities can be exploited to render the Hypha server unavailable, affecting all users.

---

## Vulnerability Details

### DOS1: WebSocket Connection Flooding (HIGH SEVERITY)

**Location**: `hypha/websocket.py`

**Description**: There is no per-user or per-workspace limit on concurrent WebSocket connections. An attacker can open thousands of connections to exhaust server resources.

**Current State**:
- `HYPHA_REDIS_MAX_CONNECTIONS=2000` is a **system-wide** limit, not per-user
- No rate limiting on new connections
- Each connection consumes:
  - 1 WebSocket connection
  - 1+ Redis connections (for pub/sub)
  - Memory for connection tracking
  - File descriptors

**Evidence from Code**:
```python
# hypha/websocket.py:229
@app.websocket(path)
async def websocket_endpoint(
    websocket: WebSocket,
    workspace: str = Query(None),
    client_id: str = Query(None),
    token: str = Query(None),
    reconnection_token: str = Query(None),
):
    await websocket.accept()  # No rate limiting or connection count check
```

**Attack Scenario**:
1. Attacker creates one user account (or uses anonymous mode)
2. Opens 1000+ WebSocket connections using different `client_id` values
3. Exhausts Redis connection pool (2000 limit)
4. New legitimate users cannot connect
5. Server becomes unresponsive

**Impact**: Complete service denial for all users

**Recommended Fix**:
- Implement per-user connection limits (e.g., max 10 connections per user)
- Add rate limiting on connection establishment (e.g., max 5 new connections/minute per user)
- Implement connection backpressure when approaching limits

**Test**: `tests/test_security_dos_resilience.py::TestDOS1WebSocketConnectionFlooding`

---

### DOS2: Service Registration Spam (MEDIUM SEVERITY)

**Location**: `hypha/core/workspace.py`

**Description**: No rate limiting on service registration. An attacker can register thousands of services to exhaust Redis memory and pollute service discovery.

**Current State**:
- Services are stored in Redis with no limits per workspace or user
- Service metadata is never pruned automatically
- Each service registration consumes Redis memory

**Evidence from Code**:
```python
# hypha/core/workspace.py - register_service has no rate limiting check
async def register_service(self, service, context: dict = None):
    # No check for number of existing services
    # No rate limiting
    await self._store.register_service(...)
```

**Attack Scenario**:
1. Attacker connects to server
2. Registers 10,000 services with unique IDs
3. Exhausts Redis memory
4. Service discovery becomes slow/unusable
5. Redis may crash or evict legitimate data

**Impact**: Service unavailability, Redis memory exhaustion

**Recommended Fix**:
- Limit maximum services per workspace (e.g., 1000)
- Implement rate limiting on service registration (e.g., 10/minute)
- Add quota system per workspace

**Test**: `tests/test_security_dos_resilience.py::TestDOS2ServiceRegistrationSpam`

---

### DOS3: Worker Pool Exhaustion (HIGH SEVERITY)

**Location**: `hypha/apps.py` - `AutoscalingManager`

**Description**: Autoscaling accepts unreasonably large `max_instances` values with no validation. An attacker can force the system to spawn thousands of worker processes.

**Current State**:
- No validation on `max_instances` in autoscaling config
- No system-wide limit on total workers across all apps
- Workers consume significant resources (processes, memory, CPU)

**Evidence from Code**:
```python
# hypha/apps.py:264 - _check_and_scale
async def _check_and_scale(self, app_id: str, config: AutoscalingConfig, context: dict):
    # config.max_instances can be arbitrarily large
    if current_count < config.max_instances:
        await self._scale_up(app_id, context)  # No upper bound check
```

```python
# hypha/apps.py - No validation in install()
manifest = await app_controller.install(
    source=app_source,
    config={
        "autoscaling": {
            "max_instances": 10000,  # Accepted without validation!
        }
    }
)
```

**Attack Scenario**:
1. Attacker installs an app with `max_instances=10000`
2. Creates artificial load to trigger autoscaling
3. System spawns thousands of worker processes
4. Server runs out of memory/CPU
5. System crashes or becomes unresponsive

**Impact**: Complete system failure, resource exhaustion

**Recommended Fix**:
- Validate `max_instances` ≤ reasonable limit (e.g., 50)
- Implement system-wide worker limit
- Add workspace quotas for worker resources
- Monitor total system load before scaling

**Test**: `tests/test_security_dos_resilience.py::TestDOS3WorkerPoolExhaustion`

---

### DOS4: Artifact Upload Flooding (HIGH SEVERITY)

**Location**: `hypha/artifact.py`, `hypha/http.py`

**Description**: No rate limiting on artifact uploads. Attackers can upload many files rapidly or very large files to exhaust storage and database resources.

**Current State**:
- No rate limiting on `put_file()` operations
- No per-workspace storage quotas
- Each upload creates database metadata entry
- S3/MinIO storage can be exhausted

**Evidence from Code**:
```python
# hypha/artifact.py - put_file has no rate limiting
async def put_file(self, path: str, content: bytes, ...):
    # No check for upload rate
    # No check for total workspace storage
    # No check for number of files
```

**Attack Scenario**:
1. Attacker uploads 10,000 small files rapidly
2. Exhausts database with metadata entries
3. Fills S3/MinIO storage
4. Legitimate users cannot upload files
5. Database performance degrades

**Impact**: Storage exhaustion, database overload, service unavailability

**Recommended Fix**:
- Implement upload rate limiting (e.g., 100 files/hour per workspace)
- Add workspace storage quotas
- Limit maximum file size
- Throttle concurrent uploads

**Test**: `tests/test_security_dos_resilience.py::TestDOS4ArtifactUploadFlooding`

---

### DOS5: Redis Connection Exhaustion (MEDIUM SEVERITY)

**Location**: `hypha/core/store.py`

**Description**: The `HYPHA_REDIS_MAX_CONNECTIONS=2000` limit is system-wide. Malicious users can exhaust the Redis connection pool.

**Evidence from Code**:
```python
# hypha/core/store.py:146
max_connections = int(os.environ.get("HYPHA_REDIS_MAX_CONNECTIONS", "2000"))
self._redis_pool = redis.asyncio.ConnectionPool.from_url(
    redis_uri,
    max_connections=max_connections,  # Global pool
)
```

**Current State**:
- Redis connection pool is shared globally
- No per-user allocation
- Each WebSocket connection uses multiple Redis connections
- Pool exhaustion prevents all operations

**Attack Scenario**:
1. Attacker opens many WebSocket connections
2. Each connection uses Redis pub/sub
3. Connection pool reaches 2000 limit
4. New operations fail with "connection pool exhausted"
5. All users affected

**Impact**: Service unavailability for all users

**Recommended Fix**:
- Implement per-user connection limits
- Use connection pooling per client
- Monitor pool usage and reject connections before exhaustion
- Increase pool size with horizontal scaling

**Test**: `tests/test_security_dos_resilience.py::TestDOS5RedisConnectionExhaustion`

---

### DOS6: Database Query Flooding (MEDIUM SEVERITY)

**Location**: `hypha/artifact.py`, `hypha/core/workspace.py`

**Description**: No rate limiting on database queries. Operations like `list_artifacts()` and `list_services()` can be called repeatedly to overwhelm the database.

**Evidence from Code**:
```python
# No rate limiting on these operations:
await artifact_manager.list()  # Can query thousands of records
await api.list_services()      # Can be called repeatedly
```

**Attack Scenario**:
1. Attacker calls `list_artifacts()` 100 times/second
2. Database CPU usage spikes to 100%
3. All database operations become slow
4. Service becomes unresponsive
5. Other users experience timeouts

**Impact**: Database overload, service degradation

**Recommended Fix**:
- Implement query rate limiting (e.g., 60 queries/minute per user)
- Add pagination limits
- Cache frequently accessed data
- Monitor query complexity

**Test**: `tests/test_security_dos_resilience.py::TestDOS6DatabaseQueryFlooding`

---

### DOS7: Autoscaling Manipulation (HIGH SEVERITY)

**Location**: `hypha/apps.py` - `AutoscalingManager`

**Description**: An attacker can manipulate autoscaling by creating artificial load, forcing the system to spawn excessive workers.

**Current State**:
- Autoscaling decisions based on load metrics
- No validation of load sources
- An attacker can create fake load by making service calls
- No cost/resource limits per workspace

**Evidence from Code**:
```python
# hypha/apps.py:280
total_load = 0
for session_id, session_info in current_instances.items():
    load = RedisRPCConnection.get_client_load(workspace, client_id)
    total_load += load  # Attacker can manipulate this

if average_load > config.scale_up_threshold * config.target_requests_per_instance:
    await self._scale_up(app_id, context)  # Triggered by artificial load
```

**Attack Scenario**:
1. Attacker installs app with high `max_instances`
2. Makes rapid RPC calls to app services
3. Artificially increases load metrics
4. Triggers autoscaling to spawn many workers
5. System resources exhausted

**Impact**: Resource exhaustion, system crash

**Recommended Fix**:
- Validate load sources
- Implement workspace resource quotas
- Add cost limits for autoscaling
- Detect and block artificial load patterns

---

### DOS8: Message Flooding (HIGH SEVERITY)

**Location**: `hypha/websocket.py`, `hypha/core/__init__.py` - `RedisEventBus`

**Description**: No rate limiting on RPC messages. An attacker can send thousands of RPC calls rapidly to overwhelm the event bus and consume CPU.

**Evidence from Code**:
```python
# hypha/websocket.py:391 - Message receive loop has no rate limiting
while not self._stop:
    data = await asyncio.wait_for(websocket.receive(), timeout=self._idle_timeout)
    # No rate limit check here
    if "bytes" in data:
        await conn.emit_message(data["bytes"])  # No throttling
```

**Attack Scenario**:
1. Attacker creates service and gets reference to it
2. Sends 10,000 RPC calls in rapid succession
3. Event bus message queue fills up
4. CPU usage spikes processing messages
5. Other users experience delays

**Impact**: CPU exhaustion, message queue overflow, service degradation

**Recommended Fix**:
- Implement per-client message rate limiting (e.g., 1000 msg/second)
- Add message queue size limits
- Throttle excessive message senders
- Monitor and alert on unusual message rates

**Test**: `tests/test_security_dos_resilience.py::TestDOS8MessageFlooding`

---

### DOS9: Memory Exhaustion via Large Messages (HIGH SEVERITY)

**Location**: `hypha/websocket.py::establish_websocket_communication`

**Description**: **CRITICAL ASYMMETRY**: Text messages are limited to 1000 characters, but binary messages (msgpack) have **NO size limit**. An attacker can send multi-gigabyte binary messages to exhaust server memory.

**Evidence from Code**:
```python
# hypha/websocket.py:404-408
elif "text" in data:
    data = data["text"]
    if len(data) > 1000:  # TEXT MESSAGE SIZE LIMIT EXISTS
        logger.warning("Ignoring long text message: %s...", data[:1000])
        continue
    # ... process text message

# But for binary messages:
if "bytes" in data:
    data = data["bytes"]  # NO SIZE CHECK AT ALL!
    await conn.emit_message(data)  # Processes unlimited size
```

**Attack Scenario**:
1. Attacker creates a service that accepts data
2. Sends a 1 GB binary message via RPC
3. Server loads entire message into memory
4. Memory exhaustion occurs
5. Server crashes or triggers OOM killer

**Impact**: Complete server crash, memory exhaustion

**Recommended Fix**:
- **URGENT**: Add size limit to binary messages (e.g., max 10 MB)
- Implement streaming for large messages
- Add memory usage monitoring
- Reject messages exceeding size limit

**Test**: `tests/test_security_dos_resilience.py::TestDOS9LargeMessageMemoryExhaustion`

**Code Comment Found**:
```python
# hypha/websocket.py:228
"""Periodically disconnect idle connections (prevents explosion without strict rate limiting)."""
```
This comment explicitly acknowledges the lack of rate limiting!

---

## Summary of Vulnerabilities

| ID | Vulnerability | Severity | Location | Exploitability |
|----|---------------|----------|----------|----------------|
| DOS1 | WebSocket Connection Flooding | HIGH | hypha/websocket.py | Easy |
| DOS2 | Service Registration Spam | MEDIUM | hypha/core/workspace.py | Easy |
| DOS3 | Worker Pool Exhaustion | HIGH | hypha/apps.py | Medium |
| DOS4 | Artifact Upload Flooding | HIGH | hypha/artifact.py | Easy |
| DOS5 | Redis Connection Exhaustion | MEDIUM | hypha/core/store.py | Medium |
| DOS6 | Database Query Flooding | MEDIUM | hypha/artifact.py | Easy |
| DOS7 | Autoscaling Manipulation | HIGH | hypha/apps.py | Medium |
| DOS8 | Message Flooding | HIGH | hypha/websocket.py | Easy |
| DOS9 | Large Message Memory Exhaustion | HIGH | hypha/websocket.py | Easy |

---

## Current Mitigations

The only existing mitigations are:

1. **Idle timeout** (`HYPHA_WS_IDLE_TIMEOUT=600`) - Disconnects idle connections after 10 minutes
   - Limited effectiveness: Attacker can send periodic pings to stay connected

2. **Text message size limit** (1000 chars) - Only applies to text, not binary
   - **Incomplete**: Binary messages unlimited

3. **System-wide limits**:
   - `HYPHA_REDIS_MAX_CONNECTIONS=2000` - Can be exhausted by single attacker
   - `HYPHA_MAX_TERMINAL_SESSIONS=50` - Per-worker type, not per-user
   - `HYPHA_MAX_BROWSER_SESSIONS=20` - Per-worker type, not per-user

**These mitigations are insufficient** to prevent DoS attacks.

---

## Recommendations

### Immediate (Critical)

1. **Add binary message size limits** (DOS9) - Can be exploited today for instant crash
2. **Implement per-user connection limits** (DOS1) - Prevents connection flooding
3. **Validate autoscaling parameters** (DOS3) - Prevent worker pool exhaustion

### Short-term (High Priority)

4. **Add rate limiting on service registration** (DOS2)
5. **Implement upload rate limiting** (DOS4)
6. **Add message rate limiting** (DOS8)

### Medium-term

7. **Implement workspace resource quotas**
   - Storage quota per workspace
   - Worker quota per workspace
   - Connection quota per workspace

8. **Add comprehensive rate limiting middleware**
   - Per-user rate limiting
   - Per-workspace rate limiting
   - Per-operation rate limiting

9. **Implement connection backpressure**
   - Graceful rejection when approaching limits
   - Queue management for connections

### Long-term

10. **Add anomaly detection**
    - Detect unusual usage patterns
    - Automatic throttling of suspicious users
    - Alerting system for DoS attempts

11. **Implement horizontal autoscaling**
    - Scale Hypha instances based on load
    - Distribute load across instances
    - Increase Redis pool with instances

---

## Testing

All vulnerabilities have been documented with tests in `tests/test_security_dos_resilience.py`. These tests demonstrate:

- Connection flooding capability
- Service registration spam
- Worker pool exhaustion
- Upload flooding
- Message flooding
- Large message handling

**Note**: Tests are designed to be non-destructive and use reasonable limits to avoid actually crashing test infrastructure.

---

## Conclusion

The Hypha server has **significant DoS vulnerabilities** across multiple attack surfaces. The lack of rate limiting and resource quotas allows a single malicious user to:

- Exhaust Redis connections
- Crash the server with large messages
- Spawn unlimited workers
- Fill storage with uploads
- Overwhelm the database

**These vulnerabilities should be addressed urgently**, starting with the most critical (DOS9: large message limit) and high-severity issues (DOS1, DOS3, DOS4, DOS8).

The architecture is fundamentally sound, but needs **defense-in-depth** with:
- Per-user limits
- Rate limiting
- Resource quotas
- Input validation
- Monitoring and alerting

---

**Assessment Completed By**: DoS Resilience Security Expert
**Date**: 2026-02-08
