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
| `cleanup` | Run built-in orphan cleanup with before/after counts | ~120s |
| `pods` | Pod status + resource usage | ~3s |
| `hc-pods` | Compute (hc-*) pods: workspace, mem limit, status, OOM | ~3s |
| `logs [pod]` | Tail logs from hypha-server or specific pod | ~3s |
| `exec "<code>"` | Execute Python in admin-utils context | varies |
| `tasks` | List all asyncio task types with counts; shows stuck heartbeats | ~5s |
| `cleanup-tasks` | Cancel stuck heartbeat tasks (method_task already done) | ~5s |
| `clients` | List all connected clients with workspace and WS status | ~5s |
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
- `_cleanup_orphaned_client_services` uses `_scan_keys` (cursor-based scan, NOT `redis.keys()`)
- `_rapp_*__rlb` clients are app load-balancer instances (hypha-agents) — expected alive in WS dict
- Typical cleanup: ~30 zombie services removed per run; post-cleanup ~3 suspects remain (all HTTP transport)
- **Cleanup "net change: +N"**: If services count increases after cleanup, new services came online during the cleanup window. This is NOT an error — cleanup ran but concurrent registrations arrived.
- `redis` is available directly as a global in exec context; no need to call `store.get_redis()`
- **hypha-agents high service count is NORMAL**: `hypha-compute-worker` (HTTP transport) registers one proxy service per deployed app + 5 worker slots + 1 built-in. 120+ services in hypha-agents = many apps deployed, not a leak. The `HIGH WS SERVICES` alert only fires for `hypha-agents` above 200 services (not 100) to avoid false positives.
- **`report` includes `top_workspaces`**: The services section now includes top 8 workspaces by service count. `HIGH WS SERVICES` alert threshold is 100 for normal workspaces, 200 for compute worker workspaces (hypha-agents).
- **`rpc_object_entries` in tasks**: The `report` now tracks total entries across all RPC `_object_store` dicts. Normal baseline: 50,000-80,000 entries (mostly service method registrations for hypha-compute-worker). Rapid growth beyond 200,000 would indicate a leak. Each service with N methods contributes ~N entries; entries are cleaned up when the owning client disconnects.

## Health Baselines (March 2026)

| Metric | Normal Range | Alert Threshold |
|--------|-------------|-----------------|
| Active RPC connections | 80-130 | >300 |
| Active WS connections | 80-115 | >300 |
| Total Redis services | 250-330 | >1000 |
| Redis clients | 100-140 | >400 |
| hypha-server CPU | 100-700m | >1500m |
| hypha-server Memory (RSS) | 800-1200 MB | >3000 MB |
| Container Memory (kubectl top) | 1400-1800 Mi | >4 Gi |
| Open file descriptors | 1200-1400 | >3000 |
| Active event bus patterns | 80-130 | >300 |
| Active workspaces | 35-45 | >200 |
| RPC object store entries | 50,000-80,000 | >200,000 |

## Known Issues (March 2026)

| Pod | Restarts | Root Cause | Status |
|-----|----------|------------|--------|
| hypha-server | 3 | OOM killed (exit 137), 8G limit | Monitor memory growth |
| hypha-weaviate | 0 (new pod) | Previous pod had 85 OOM restarts; replaced Mar 6 2026. New pod stable at ~115 Mi. | Monitor |
| bioimageio-colab | 34 | SIGTERM (exit 15) — liveness probe fails when Hypha connection slow | Benign/recurring |
| deno-app-engine | 21 | Clean exit (0) — intentional restart loop | Normal |
| hypha-compute | 10 | Clean exit (0) | Normal |
| hc-hypha-agents-* | recurring | OOM killed (exit 137), 2Gi limit — ML workloads in agent-sandbox exceed limit after ~1-2h | Needs higher memory limit |

## Version Notes

| Version | Key Fix | Deployed |
|---------|---------|---------|
| 0.21.73 | hypha-rpc 0.21.33: heartbeat task leak fix (cancel on exception/CancelledError) | **Not yet** (live: 0.21.70) |
| 0.21.72 | hypha-rpc 0.21.32: scan-vs-keys migration, GC fixes | Not yet deployed |
| 0.21.71 | All redis.keys() → scan fixes, anonymous-workspace-unload TOCTOU fix | Not yet deployed |

> **Note**: Live server is at 0.21.70. Use `report` → `server.version` to confirm. Upgrade to 0.21.73 will eliminate stuck heartbeat tasks entirely.

### Stuck heartbeat tasks (pre-0.21.73)
- Cause: hypha-rpc <0.21.33 did not cancel heartbeat task on exception/CancelledError
- Symptom: `report` shows `tasks.heartbeat_stuck > 0`
- Workaround: Run `cleanup-tasks` to cancel stuck tasks manually
- Permanent fix: upgrade server to 0.21.73

## Hypha-Compute (hc-*) Pod Notes

- `hc-*` pods are Kubernetes **Jobs** spawned by hypha-compute on demand — `restartPolicy: Never`
- They do NOT auto-restart after OOM; hypha-compute spawns a new pod on next request
- Image: `oeway/agent-sandbox:0.3.2`, memory limit: 2Gi (as of March 2026)
- OOM pattern for hypha-agents: pods run ~1-2h then OOM — 2Gi limit too low for ML workloads
- Use `hc-pods` command to inspect all compute pods and catch recurring OOM cycles
