# hypha-admin

A concise CLI for monitoring and managing a running Hypha server.
Provides health checks, workspace/service inspection, zombie detection, pod management, and live patching via `admin-utils`.

## Quick Start

```bash
# Run a health check
python3 skills/hypha-admin/hypha-admin.py health

# Fast zombie check (~1s, no ping)
python3 skills/hypha-admin/hypha-admin.py quick-zombies

# Full zombie scan with ping (~2s per suspect)
python3 skills/hypha-admin/hypha-admin.py zombies

# Run orphan cleanup (before/after counts)
python3 skills/hypha-admin/hypha-admin.py cleanup

# List top workspaces by service count
python3 skills/hypha-admin/hypha-admin.py workspaces

# Get machine-readable JSON health report
python3 skills/hypha-admin/hypha-admin.py report

# Execute arbitrary Python on the server
python3 skills/hypha-admin/hypha-admin.py exec "print('hello')"

# Full status (health + workspaces + quick-zombies)
python3 skills/hypha-admin/hypha-admin.py status

# Help
python3 skills/hypha-admin/hypha-admin.py help
```

> Requires `hypha-rpc` and `kubectl` installed.

## Commands

| Command | Description | Typical Time |
|---------|-------------|-------------|
| `health` | Process stats, metrics, pods, OOM/high-restart alerts | ~5s |
| `metrics` | RPC and event bus metrics JSON | ~3s |
| `workspaces` | List workspaces sorted by service count | ~5s |
| `services [ws]` | List services (optionally filter by workspace) | ~5s |
| `quick-zombies` | WS-dict based zombie check, no ping needed | ~1s |
| `zombies` | Ping only suspect (non-WS) clients; skips alive WS clients | ~5-30s |
| `cleanup` | Targeted orphan cleanup: ping only non-WS suspects, delete confirmed zombies | ~10-30s |
| `pods` | Pod status + resource usage | ~3s |
| `hc-pods` | Compute (hc-*) pods: workspace, mem limit, status, OOM | ~3s |
| `logs [pod]` | Tail logs from hypha-server or specific pod | ~3s |
| `exec "<code>"` | Execute Python in admin-utils context | varies |
| `tasks` | List all asyncio task types with counts; shows stuck heartbeats | ~5s |
| `cleanup-tasks` | Cancel stuck heartbeat tasks (method_task already done) | ~5s |
| `clients` | List all connected clients with workspace and WS status | ~5s |
| `sessions` | List active app sessions in Redis with app_id, status, TTL | ~5s |
| `report` | Full JSON health report (for automation/monitoring) | ~10s |
| `status` | Full status: health + workspaces + quick-zombies | ~15s |
| `kickout <ws> <client>` | Kick a client from a workspace | ~3s |
| `scale <deploy> <n>` | Scale a Kubernetes deployment | ~3s |

## Environment Variables

Use `HYPHA_ADMIN_*` prefix to avoid conflicts with the user's `HYPHA_TOKEN`/`HYPHA_WORKSPACE`.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPHA_ADMIN_URL` | `https://hypha.aicell.io` | Hypha server URL |
| `HYPHA_ADMIN_TOKEN` | admin token | Root admin bearer token |
| `HYPHA_ADMIN_WORKSPACE` | `ws-user-root` | Admin workspace |
| `HYPHA_ADMIN_SERVICE` | `admin-utils` | Admin service name |
| `KUBECTL_NS` | `hypha` | Kubernetes namespace |

> **Note**: The shell often has `HYPHA_TOKEN`/`HYPHA_WORKSPACE` set for the user's own session. Always use `HYPHA_ADMIN_*` for admin operations.

## Admin-Utils Context Variables

When executing code via `exec`, these variables are available:

| Variable | Type | Description |
|----------|------|-------------|
| `store` | `RedisStore` | Main store (workspace mgr, redis, etc.) |
| `wm` | `WorkspaceManager` | Workspace manager (same as `store._workspace_manager`) |
| `redis` | `aioredis.Redis` | Direct Redis client (same as `store.get_redis()`) |
| `pool` | `aioredis.ConnectionPool` | Redis connection pool |
| `admin` | object | Admin service interface |
| `app` | FastAPI | The FastAPI app |
| `ws_server` | object | WebSocket server |
| `psutil` | module | Process utilities |
| `os`, `sys`, `subprocess` | modules | Standard modules |

### Key store methods
```python
store.get_redis()               # Redis client
store._workspace_manager        # WorkspaceManager (wm)
store.list_all_workspaces()     # All WorkspaceInfo objects
store.get_metrics()             # RPC/eventbus metrics dict
store.get_root_user()           # Root UserInfo
store._server_id                # e.g. "tabby-auroraceratops-90271949"
store._websocket_server._websockets  # dict: {ws/cid: ws_obj}
store._cleanup_orphaned_client_services()  # built-in orphan cleanup
```

## Redis Key Patterns

```
services:<visibility>|<type>:<workspace>/<client_id>:<service_id>@*
```

Examples:
- `services:public|built-in:public/abc123:built-in@*`
- `services:protected|generic:ws-user-root/xyz:admin-utils@*`

## Zombie Detection Strategy

**Two-tier approach:**
1. **Quick check** (`quick-zombies`): Compare Redis built-in service keys against `_websockets` dict. O(n_services), ~1s. Non-WS clients may be HTTP transport (not zombies). Check list is `suspect`, not confirmed-dead.
2. **Ping check** (`zombies`): Only ping clients NOT in `_websockets` (the suspects). ~2s per suspect. Confirms dead vs. alive-via-HTTP.

**Known HTTP transport clients (appear in suspect list but ARE alive):**
- `hypha-agents/hypha-compute-worker`
- Any active svamp HTTP sessions (e.g., `ws-user-github|478667/5wko...`)

## Patterns & Gotchas

- `async for` must be wrapped in `async def` when using `exec`
- WorkspaceInfo attributes: `.id`, `.name` (NOT subscriptable like dict)
- UserInfo requires `is_anonymous` field — use `store.get_root_user().model_dump()` for admin context
- `_cleanup_orphaned_client_services` is O(N×3s) where N = all active clients — at 2000+ clients this takes 100+ minutes and is NOT usable at scale. The admin `cleanup` command avoids it by only pinging non-WS suspects (typically <10).
- `_rapp_*__rlb` clients are app load-balancer instances (hypha-agents) — expected alive in WS dict
- Typical cleanup: 0-5 zombie services removed per run; suspects are usually 4-7 (mostly HTTP transport alive clients)
- **Cleanup "net change: +N"**: If services count increases after cleanup, new services came online during the cleanup window. This is NOT an error — cleanup ran but concurrent registrations arrived.
- `redis` is available directly as a global in exec context; no need to call `store.get_redis()`
- **hypha-agents high service count is NORMAL**: `hypha-compute-worker` (HTTP transport) registers one proxy service per deployed app + 5 worker slots + 1 built-in. 300-700+ services in hypha-agents = many apps deployed, not a leak. The `HIGH WS SERVICES` alert only fires for `hypha-agents` above **1000** services (not 100) to avoid false positives.
- **`report` includes `top_workspaces`**: The services section now includes top 8 workspaces by service count. `HIGH WS SERVICES` alert threshold is 100 for normal workspaces, **1000** for compute worker workspaces (hypha-agents).
- **`rpc_object_entries` in tasks**: The `report` now tracks total entries across all RPC `_object_store` dicts. Normal baseline: 50,000-80,000 entries (mostly service method registrations for hypha-compute-worker). Rapid growth beyond 200,000 would indicate a leak. Each service with N methods contributes ~N entries; entries are cleaned up when the owning client disconnects.
- **`pending_rpc_calls` in tasks**: Count of `Timer._job` asyncio tasks, one per active in-flight RPC call (each call creates a 30s timeout timer). Normal: 10-50. A burst to 200-500 is transient — it means many concurrent calls fired at once (e.g. hypha-agents starting many services). Alert fires at >200. Does NOT indicate a leak — these tasks complete once the call resolves or times out.
- **`_session_gc_loop` tasks in top_types**: One `RPC._session_gc_loop` task per active RPC instance on the server (e.g., 7-10 tasks is normal). Each sweeps `_object_store` sessions with no activity older than `_session_ttl`. Not a leak; completely normal background maintenance.
- **`top_types` in tasks**: Top 8 asyncio task types by count. Typical profile: 6 WS infrastructure types × ~N per connection, plus heartbeats and Timer._job. Use to detect new accumulating patterns. `WebSocketCommonProtocol.*` tasks are 1:1 with WS connections — if they exceed active_ws×6 significantly, connections may be lingering in teardown.
- **Memory grows with service count**: Each new service registration creates Python routing objects. ~2-3 MB per service. hypha-agents 162 services ≈ +200 MB over baseline. This is expected, not a leak.

## Health Baselines (March 2026, updated for peak-load scale)

Peak hours (daytime UTC+1 to UTC+8) see many concurrent user sessions. Normal range is much higher than off-peak. Thresholds are set to catch genuine anomalies, not normal daily peaks.

| Metric | Off-peak Range | Peak Range | Alert Threshold |
|--------|---------------|-----------|-----------------|
| Active RPC connections | 100-500 | 2000-3000 | >5000 |
| Active WS connections | 100-500 | 2000-3000 | >5000 |
| Total Redis services | 200-800 | 2000-3500 | >8000 |
| Distinct workspaces connected | 50-200 | 1000-2500 | — |
| Redis clients | 100-300 | 2000-3000 | — |
| hypha-server CPU | 100-700m | 700-1200m | >1500m |
| hypha-server Memory (RSS) | 400-600 MB | 1000-2000 MB | >3000 MB |
| Container Memory (cgroup) | 500-700 Mi | 1400-2000 Mi | >5000 Mi |
| Open file descriptors | 500-1000 | 2000-3000 | >6000 |
| Active event bus patterns | 100-500 | 2000-3000 | — |
| RPC object store entries | 10-30 (0.21.77+) | ~30-100 | >500 sustained |
| Pending RPC calls (Timer._job) | 10-50 (burst: up to 500) | same | >200 sustained |

## Known Issues (March 2026)

| Pod | Restarts | Root Cause | Status |
|-----|----------|------------|--------|
| hypha-server | 3 | OOM killed (exit 137), 8G limit | Stable — last OOM >3 days ago |
| hypha-weaviate | 1 | 1 restart during Mar-7 server upgrade (liveness probe kill, NOT OOM). New pod stable at ~117 Mi. | Healthy |
| bioimageio-colab | 35 | SIGTERM (exit 15) — liveness probe fails when Hypha connection slow | Benign/recurring |
| deno-app-engine | 21 | Clean exit (0) — intentional restart loop | Normal |
| hypha-biomni | 17 | exit 252 (app error), old hypha-rpc causes built-in service registration to fail | Needs image update |
| hc-hypha-agents-* | recurring | OOM killed (reason=OOMKilled): 2Gi limit bumped to 4Gi (early Mar 7 2026), then OOMed again at 94-97% → bumped to **6Gi** via kubectl patch (Mar 7 2026) | All new pods get 6Gi; running pods keep old limit until OOM cycle |

## Version Notes

| Version | Key Fix | Deployed |
|---------|---------|---------|
| 0.21.78 | perf(workspace): batch Redis pipeline for list_services/list_clients/delete_workspace/cleanup_client (PR #934); admin: fast cleanup cmd (O(non-WS) not O(all)), updated thresholds for 2000+ connection scale | **LIVE** (deployed 2026-03-07, ~13:14 UTC) |
| 0.21.77 | remove limit_max_requests from uvicorn (PR #932): fixes "connection refused" liveness probe failures | Superseded by 0.21.78 |
| 0.21.76 | worker_managed session persist fix (PR #930) | Superseded |
| 0.21.74 | OOM fix: use reason=OOMKilled instead of exitCode=137; worker_id propagation; admin improvements | Superseded |
| 0.21.73 | hypha-rpc 0.21.33: heartbeat task leak fix; ghost _last_seen cleanup; scan/GC fixes | Superseded |

> **Note**: Live server is at **0.21.78** (deployed Mar 7 2026, ~13:14 UTC). Key changes: batched Redis pipeline for workspace list operations (N sequential HGETALL → 1 round-trip); admin `cleanup` cmd now O(non-WS suspects) not O(all clients); alert thresholds updated for peak production scale (5000/8000/6000).

### OOM Detection Note
- `reason == "OOMKilled"` is the only reliable OOM signal (k8s sets this for memory kills)
- `exitCode == 137` alone is NOT sufficient — liveness probe kills also use exit 137 with `reason = "Error"`
- Admin tool (`report`, `health`) correctly uses reason check

### Stuck heartbeat tasks (pre-0.21.73)
- Cause: hypha-rpc <0.21.33 did not cancel heartbeat task on exception/CancelledError
- Symptom: `report` shows `tasks.heartbeat_stuck > 0`
- Permanent fix: upgrade server to 0.21.73 (already deployed Mar 7 2026)

## Hypha-Compute (hc-*) Pod Notes

- `hc-*` pods are Kubernetes **Jobs** spawned by hypha-compute on demand — `restartPolicy: Never`
- They do NOT auto-restart after OOM; hypha-compute spawns a new pod on next request
- Image: `oeway/agent-sandbox:0.3.2`, memory limit: **6Gi** (bumped 2Gi→4Gi→6Gi, all Mar 7 2026)
- hypha-compute installs itself from `amun-ai/hypha-compute` GitHub main at startup — no version pin needed
- Verify active config: `kubectl exec hypha-compute-* -n hypha -- python3 -c "from hypha_compute.worker import ComputeAppConfig; print(ComputeAppConfig().resources)"`
- Use `hc-pods` command to inspect all compute pods and catch recurring OOM cycles
