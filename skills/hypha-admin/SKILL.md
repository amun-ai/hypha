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
| `logs [pod]` | Tail logs from hypha-server or specific pod | ~3s |
| `exec "<code>"` | Execute Python in admin-utils context | varies |
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

## Health Baselines (March 2026)

| Metric | Normal Range | Alert Threshold |
|--------|-------------|-----------------|
| Active RPC connections | 80-130 | >300 |
| Active WS connections | 80-115 | >300 |
| Total Redis services | 250-330 | >1000 |
| Redis clients | 100-140 | >400 |
| hypha-server CPU | 100-700m | >1500m |
| hypha-server Memory (RSS) | 800-1000 MB | >3000 MB |
| Container Memory (kubectl top) | 1200-1600 Mi | >4 Gi |
| Open file descriptors | 1200-1400 | >3000 |
| Active event bus patterns | 80-130 | >300 |
| Active workspaces | 35-45 | >200 |

## Known Issues (March 2026)

| Pod | Restarts | Root Cause | Status |
|-----|----------|------------|--------|
| hypha-server | 3 | OOM killed (exit 137), 8G limit | Monitor memory growth |
| hypha-weaviate | 85 | OOM killed (exit 137), 6Gi limit — memory bursts under vector search load | Under investigation |
| bioimageio-colab | 34 | SIGTERM (exit 15) — periodic external restart | Benign |
| deno-app-engine | 20 | Clean exit (0) — intentional restart loop | Normal |
| hypha-compute | 10 | Clean exit (0) | Normal |
