#!/usr/bin/env python3
"""hypha-admin.py — Hypha server admin CLI using hypha-rpc.

Usage:
    python hypha-admin.py <command> [args...]

See SKILL.md for full documentation.
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
# Use HYPHA_ADMIN_* vars to avoid conflict with user's HYPHA_TOKEN/HYPHA_WORKSPACE
HYPHA_URL   = os.environ.get("HYPHA_ADMIN_URL",   os.environ.get("HYPHA_URL", "https://hypha.aicell.io"))
HYPHA_TOKEN = os.environ.get("HYPHA_ADMIN_TOKEN", "0u39rlsdkfow34o3ijo09wu4o23ijosijr9238y43oirjoweihrowi3h43r")
HYPHA_WS    = os.environ.get("HYPHA_ADMIN_WORKSPACE", "ws-user-root")
HYPHA_SVC   = os.environ.get("HYPHA_ADMIN_SERVICE", "admin-utils")
KUBECTL_NS  = os.environ.get("KUBECTL_NS", "hypha")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kubectl(*args):
    result = subprocess.run(["kubectl", "-n", KUBECTL_NS] + list(args),
                            capture_output=True, text=True)
    return result.stdout + result.stderr


def _kubectl_json(*args):
    result = subprocess.run(["kubectl", "-n", KUBECTL_NS, "-o", "json"] + list(args),
                            capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except Exception:
        return {}


async def get_admin(server):
    """Return the admin-utils service proxy."""
    return await server.get_service(HYPHA_SVC)


async def exec_py(admin, code: str, timeout: int = 20) -> str:
    """Execute Python code on the server via admin-utils."""
    result = await admin.execute_command(command=code, timeout=timeout)
    return result.get("output", "") if isinstance(result, dict) else str(result)


async def connect():
    from hypha_rpc import connect_to_server
    return await connect_to_server(
        server_url=HYPHA_URL,
        token=HYPHA_TOKEN,
        workspace=HYPHA_WS,
        method_timeout=120,
    )


def _fmt_age(start_time_str: str) -> str:
    """Return human-readable age from ISO8601 timestamp."""
    try:
        from datetime import timezone as tz
        start = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        delta = datetime.now(tz.utc) - start
        d = delta.days
        h = delta.seconds // 3600
        if d > 0:
            return f"{d}d{h}h"
        return f"{h}h{(delta.seconds % 3600)//60}m"
    except Exception:
        return "?"


# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_health(server):
    """Quick health check: process stats, metrics, pod status, OOM alerts."""
    print("=== Hypha Server Health Check ===")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    admin = await get_admin(server)

    # Process stats
    print("\n--- Server Process ---")
    out = await exec_py(admin,
        'import json, psutil; p=psutil.Process(); '
        'print(json.dumps({"cpu_pct": p.cpu_percent(interval=0.5), '
        '"mem_mb": round(p.memory_info().rss/1024/1024), '
        '"threads": p.num_threads(), "fds": p.num_fds()}))', 15)
    try:
        stats = json.loads(out.strip())
        mem_flag = " ⚠ HIGH" if stats['mem_mb'] > 3000 else ""
        fds_flag = " ⚠ HIGH" if stats['fds'] > 3000 else ""
        print(f"  CPU: {stats['cpu_pct']}% | Memory: {stats['mem_mb']} MB{mem_flag} | "
              f"Threads: {stats['threads']} | FDs: {stats['fds']}{fds_flag}")
    except Exception:
        print(out)

    # RPC metrics
    print("\n--- Metrics ---")
    out = await exec_py(admin,
        'import json; m=await store.get_metrics(); '
        'rpc=m["rpc"]["rpc_connections"]; eb=m["eventbus"]["eventbus"]; '
        'print(json.dumps({"active_rpc": rpc["active"], '
        '"active_ws": len(store._websocket_server._websockets), '
        '"active_patterns": eb["patterns"]["active"]}))', 15)
    try:
        m = json.loads(out.strip())
        rpc_flag = " ⚠" if m['active_rpc'] > 300 else ""
        print(f"  Active RPC: {m['active_rpc']}{rpc_flag} | Active WS: {m['active_ws']} | "
              f"Event patterns: {m['active_patterns']}")
    except Exception:
        print(out)

    # Pod status with OOM detection
    print("\n--- Pods (infra) ---")
    pods_data = _kubectl_json("get", "pods")
    oom_alerts = []
    for item in pods_data.get("items", []):
        name = item["metadata"]["name"]
        if not any(k in name for k in ["hypha-server", "redis-b8", "hypha-sql"]):
            continue
        phase = item.get("status", {}).get("phase", "?")
        containers = item.get("status", {}).get("containerStatuses", [{}])
        container = containers[0] if containers else {}
        restarts = container.get("restartCount", 0)
        last_terminated = container.get("lastState", {}).get("terminated", {})
        cur_terminated = container.get("state", {}).get("terminated", {})
        exit_code = last_terminated.get("exitCode")
        oom_reason = last_terminated.get("reason") or cur_terminated.get("reason")
        if exit_code is None:
            exit_code = cur_terminated.get("exitCode")
        start = item.get("status", {}).get("startTime", "")
        age = _fmt_age(start)
        oom_flag = ""
        if oom_reason == "OOMKilled":
            oom_flag = " ⚠ OOM-KILLED"
            oom_alerts.append(f"{name} (OOMKilled)")
        print(f"  {name:<50} {phase:<10} restarts={restarts:<4} age={age}{oom_flag}")

    if oom_alerts:
        print("\n  *** OOM KILL ALERTS ***")
        for a in oom_alerts:
            print(f"  !! {a}")

    # High-restart pods
    print("\n--- High-Restart Pods (restarts≥5) ---")
    found = False
    for item in pods_data.get("items", []):
        name = item["metadata"]["name"]
        containers = item.get("status", {}).get("containerStatuses", [{}])
        container = containers[0] if containers else {}
        restarts = container.get("restartCount", 0)
        if restarts >= 5:
            last_terminated = container.get("lastState", {}).get("terminated", {})
            cur_terminated = container.get("state", {}).get("terminated", {})
            exit_code = last_terminated.get("exitCode")
            if exit_code is None:
                exit_code = cur_terminated.get("exitCode")
            if exit_code is None:
                exit_code = "?"
            phase = item.get("status", {}).get("phase", "?")
            print(f"  {name:<55} restarts={restarts:<4} last_exit={exit_code} phase={phase}")
            found = True
    if not found:
        print("  None")

    # Top CPU pods
    print("\n--- Top CPU Pods ---")
    top = _kubectl("top", "pods", "--sort-by=cpu")
    lines = top.strip().splitlines()
    print(f"  {lines[0]}")
    for line in lines[1:6]:
        print(f"  {line}")


async def cmd_metrics(server):
    admin = await get_admin(server)
    out = await exec_py(admin,
        'import json; m=await store.get_metrics(); print(json.dumps(m, indent=2))', 15)
    print(out)


async def cmd_workspaces(server):
    print("=== Workspaces by Service Count ===")
    admin = await get_admin(server)
    out = await exec_py(admin, '''
import json

async def ws_service_counts():
    redis = store.get_redis()
    ws_counts = {}
    async for key in redis.scan_iter(match="services:*", count=500):
        k = key.decode() if isinstance(key, bytes) else key
        parts = k.split(":")
        if len(parts) >= 3:
            ws = parts[2].split("/")[0] if "/" in parts[2] else parts[2]
            ws_counts[ws] = ws_counts.get(ws, 0) + 1
    return ws_counts

ws_counts = await ws_service_counts()
sorted_ws = sorted(ws_counts.items(), key=lambda x: -x[1])
total_ws = len(await store.list_all_workspaces())
print(json.dumps({"total_workspaces": total_ws, "total_services": sum(ws_counts.values()), "by_workspace": sorted_ws}))
''', 30)
    try:
        data = json.loads(out.strip())
        print(f"Total workspaces: {data['total_workspaces']} | Total services: {data['total_services']}")
        print(f"{'Workspace':<50} {'Services':>8}")
        print("-" * 60)
        for ws, count in data['by_workspace']:
            flag = " ⚠" if count > 100 else ""
            print(f"  {ws:<48} {count:>8}{flag}")
    except Exception:
        print(out)


async def cmd_services(server, workspace: str = ""):
    admin = await get_admin(server)
    if workspace:
        print(f"=== Services in workspace: {workspace} ===")
        out = await exec_py(admin, f'''
import json

async def get_ws_services(ws_id):
    redis = store.get_redis()
    keys = []
    async for key in redis.scan_iter(match=f"services:*:{{ws_id}}/*", count=200):
        k = key.decode() if isinstance(key, bytes) else key
        keys.append(k)
    return keys

svcs = await get_ws_services("{workspace}")
parsed = []
for s in svcs:
    parts = s.split(":")
    vis_type = parts[1] if len(parts) > 1 else "?"
    svc_id = parts[3].split("@")[0] if len(parts) > 3 else s
    client = parts[2].split("/")[1] if "/" in parts[2] else parts[2]
    parsed.append({{"vis": vis_type, "client": client[:20], "service": svc_id}})
print(json.dumps({{"count": len(parsed), "services": sorted(parsed, key=lambda x: x["service"])}}))
''', 20)
        try:
            data = json.loads(out.strip())
            print(f"Count: {data['count']}")
            for s in data['services']:
                print(f"  [{s['vis']:<30}] {s['service']:<40} client={s['client']}")
        except Exception:
            print(out)
    else:
        print("=== All Services by Type ===")
        out = await exec_py(admin, '''
import json

async def count_all():
    redis = store.get_redis()
    by_type = {}
    total = 0
    async for key in redis.scan_iter(match="services:*", count=500):
        k = key.decode() if isinstance(key, bytes) else key
        parts = k.split(":")
        t = parts[1] if len(parts) > 1 else "?"
        by_type[t] = by_type.get(t, 0) + 1
        total += 1
    return total, by_type

total, by_type = await count_all()
print(json.dumps({"total": total, "by_type": sorted(by_type.items(), key=lambda x: -x[1])}))
''', 30)
        try:
            data = json.loads(out.strip())
            print(f"Total: {data['total']}")
            for t, count in data['by_type']:
                print(f"  {t:<40} {count:>6}")
        except Exception:
            print(out)


async def cmd_quick_zombies(server):
    """Fast zombie check via WS dict (no ping, no network). ~1s.
    Note: HTTP transport clients (hypha-compute-worker, active HTTP clients) appear as
    'not_in_ws' but ARE alive. Use 'zombies' command to ping-verify suspicious ones."""
    print("=== Quick Zombie Check (WS dict, no ping) ===")
    admin = await get_admin(server)
    out = await exec_py(admin, '''
import json

async def quick_check():
    redis = store.get_redis()
    ws_set = store._websocket_server._websockets
    server_id = store._server_id
    skip = {f"public/{server_id}", f"ws-user-root/{server_id}"}

    all_clients = {}
    async for key in redis.scan_iter(match="services:*|built-in:*:built-in@*", count=500):
        k = key.decode() if isinstance(key, bytes) else key
        parts = k.split(":")
        if len(parts) >= 3 and "/" in parts[2]:
            ckey = parts[2]
            if ckey in skip:
                continue
            all_clients[ckey] = ckey in ws_set

    in_ws = [c for c, v in all_clients.items() if v]
    not_ws = [c for c, v in all_clients.items() if not v]
    return {"total": len(all_clients), "in_ws": len(in_ws), "not_in_ws": len(not_ws), "suspect": sorted(not_ws)}

r = await quick_check()
print(json.dumps(r, indent=2))
''', 20)
    try:
        data = json.loads(out.strip())
        print(f"Total clients: {data['total']} | In WS: {data['in_ws']} | Not in WS: {data['not_in_ws']}")
        if data['suspect']:
            print("\nNot in WebSocket (may be HTTP transport or zombies):")
            for c in data['suspect']:
                print(f"  {c}")
    except Exception:
        print(out)


async def cmd_zombies(server):
    """Detect zombie services by pinging client built-in services (~2s per client)."""
    print("=== Zombie / Orphan Service Detection (ping-based) ===")
    admin = await get_admin(server)
    out = await exec_py(admin, '''
import json

async def find_zombies():
    redis = store.get_redis()
    ws_set = store._websocket_server._websockets
    server_id = store._server_id

    # Pre-filter: only ping clients NOT already in WebSocket dict
    # (WS clients are definitely alive; HTTP transport clients need ping)
    skip = {f"public/{server_id}", f"ws-user-root/{server_id}"}
    clients_to_check = set()
    async for key in redis.scan_iter(match="services:*|built-in:*:built-in@*", count=500):
        k = key.decode() if isinstance(key, bytes) else key
        parts = k.split(":")
        if len(parts) >= 3 and "/" in parts[2]:
            ckey = parts[2]
            if ckey in skip or ckey in ws_set:
                continue  # skip server-owned and confirmed WS clients
            clients_to_check.add(ckey)

    # Ping only the suspect clients
    rpc = store.create_rpc("root", store._root_user, client_id="zombie-checker", silent=True)
    zombies = []
    alive_http = []
    try:
        for ws_client in clients_to_check:
            workspace, client_id = ws_client.split("/", 1)
            if client_id.startswith("manager-") or client_id == "zombie-checker":
                continue
            try:
                svc = await rpc.get_remote_service(f"{workspace}/{client_id}:built-in", {"timeout": 2})
                await svc.ping("ping")
                alive_http.append(ws_client)
            except Exception:
                zombies.append(ws_client)
    finally:
        await rpc.disconnect()

    # Count services for each zombie
    zombie_details = []
    for zc in zombies:
        count = 0
        async for _ in redis.scan_iter(match=f"services:*:{zc}:*", count=200):
            count += 1
        zombie_details.append({"client": zc, "services": count})

    return {
        "ws_clients": len(ws_set),
        "suspect_checked": len(clients_to_check),
        "alive_http": len(alive_http),
        "zombie_clients": len(zombies),
        "zombie_services": sum(z["services"] for z in zombie_details),
        "alive_http_clients": sorted(alive_http),
        "zombies": sorted(zombie_details, key=lambda x: -x["services"])
    }

result = await find_zombies()
print(json.dumps(result, indent=2))
''', 90)
    try:
        data = json.loads(out.strip())
        print(f"WS clients: {data['ws_clients']} | Suspect checked: {data['suspect_checked']} | "
              f"Alive (HTTP): {data['alive_http']} | "
              f"Zombies: {data['zombie_clients']} clients / {data['zombie_services']} services")
        if data.get('alive_http_clients'):
            print(f"\nAlive via HTTP transport: {', '.join(data['alive_http_clients'])}")
        if data['zombies']:
            print("\nZombie clients:")
            for z in data['zombies']:
                print(f"  {z['client']:<60} {z['services']:>4} services")
        else:
            print("\nNo zombies found!")
    except Exception:
        print(out)


async def cmd_cleanup_zombies(server):
    """Delete services from zombie clients (ping-verified dead)."""
    print("=== Cleanup Zombie Services ===")
    admin = await get_admin(server)

    count_code = '''
import json
async def _count_svcs():
    redis = store.get_redis()
    n = 0
    async for _ in redis.scan_iter(match="services:*", count=500):
        n += 1
    return n
print(json.dumps({"count": await _count_svcs()}))
'''

    # Count before
    before_out = await exec_py(admin, count_code, 30)

    # Run built-in cleanup
    out = await exec_py(admin,
        'await store._cleanup_orphaned_client_services(); print("Cleanup complete")', 120)
    print(out)

    # Count after
    after_out = await exec_py(admin, count_code, 30)

    try:
        before = json.loads(before_out.strip()).get("count", "?")
        after = json.loads(after_out.strip()).get("count", "?")
        if isinstance(before, int) and isinstance(after, int):
            removed = before - after
            if removed >= 0:
                label = f"removed {removed}"
            else:
                label = f"net change: +{-removed} (new services joined during cleanup window)"
        else:
            label = "?"
        print(f"Services: {before} → {after} ({label})")
    except Exception:
        print(f"Before: {before_out.strip()} | After: {after_out.strip()}")


async def cmd_pods(server):
    print("=== Pod Status ===")
    print(_kubectl("get", "pods", "-o", "wide"))
    print("=== Pod Resources ===")
    print(_kubectl("top", "pods", "--sort-by=memory"))


async def cmd_hc_pods(server):
    """Show hypha-compute (hc-*) pod status: workspace, memory limit, age, OOM info."""
    print("=== Hypha-Compute (hc-*) Pods ===")
    pods_data = _kubectl_json("get", "pods")
    hc_pods = [p for p in pods_data.get("items", []) if p["metadata"]["name"].startswith("hc-")]
    if not hc_pods:
        print("  No hc-* pods found.")
        return

    # Try to get live memory from kubectl top
    top_mem = {}
    top_out = _kubectl("top", "pods", "--no-headers", "--sort-by=memory")
    for line in top_out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].startswith("hc-"):
            top_mem[parts[0]] = parts[2]

    def _parse_mem_mi(s: str) -> float:
        """Parse memory string (e.g. '2Gi', '598Mi') to MiB float."""
        s = s.strip()
        if s.endswith("Gi"):
            return float(s[:-2]) * 1024
        if s.endswith("Mi"):
            return float(s[:-2])
        if s.endswith("G"):
            return float(s[:-1]) * 1024
        if s.endswith("M"):
            return float(s[:-1])
        if s.endswith("Ki"):
            return float(s[:-2]) / 1024
        return 0.0

    print(f"{'Pod':<50} {'Workspace':<25} {'Status':<12} {'MemLimit':<10} {'MemUsed':<10} {'Mem%':<8} {'Age'}")
    print("-" * 130)
    oom_count = 0
    running_count = 0
    for p in sorted(hc_pods, key=lambda x: x["metadata"]["name"]):
        name = p["metadata"]["name"]
        labels = p["metadata"].get("labels", {})
        workspace = labels.get("sandbox-namespace", "?").replace("hc-", "")
        spec_c = p["spec"]["containers"][0] if p["spec"].get("containers") else {}
        mem_limit = spec_c.get("resources", {}).get("limits", {}).get("memory", "?")
        start = p.get("status", {}).get("startTime", "")
        age = _fmt_age(start) if start else "?"
        cs = p.get("status", {}).get("containerStatuses", [{}])
        c = cs[0] if cs else {}
        state = c.get("state", {})
        cur_term = state.get("terminated", {})
        if cur_term.get("reason") == "OOMKilled":
            status = "OOMKilled"
            oom_count += 1
        elif cur_term.get("exitCode") == 0:
            status = "Completed"
        elif cur_term:
            status = f"Exit:{cur_term.get('exitCode','?')}"
        elif state.get("running"):
            status = "Running"
            running_count += 1
        else:
            status = p.get("status", {}).get("phase", "?")
        mem_used = top_mem.get(name, "-")
        mem_pct_str = "-"
        if mem_used != "-" and mem_limit != "?":
            limit_mi = _parse_mem_mi(mem_limit)
            used_mi = _parse_mem_mi(mem_used)
            if limit_mi > 0:
                pct = used_mi / limit_mi * 100
                mem_pct_str = f"{pct:.0f}%"
                if pct >= 80:
                    mem_pct_str += " ⚠⚠"
                elif pct >= 60:
                    mem_pct_str += " ⚠"
        oom_flag = " ⚠" if status == "OOMKilled" else ""
        print(f"  {name:<48} {workspace:<25} {status:<12} {mem_limit:<10} {mem_used:<10} {mem_pct_str:<8} {age}{oom_flag}")

    print(f"\n  Total: {len(hc_pods)} | Running: {running_count} | OOMKilled: {oom_count}")


async def cmd_logs(server, pod: str = ""):
    if not pod:
        # Find hypha-server pod
        pods_out = _kubectl("get", "pods", "--no-headers", "-o", "custom-columns=NAME:.metadata.name")
        for line in pods_out.splitlines():
            if "hypha-server" in line:
                pod = line.strip()
                break
    if not pod:
        print("ERROR: No hypha-server pod found", file=sys.stderr)
        return
    print(f"=== Logs: {pod} (last 60 lines) ===")
    print(_kubectl("logs", pod, "--tail=60"))


async def cmd_exec(server, code: str):
    admin = await get_admin(server)
    out = await exec_py(admin, code, 30)
    print(out)


async def cmd_report(server):
    """Full structured JSON health report for programmatic consumption."""
    admin = await get_admin(server)
    ts = datetime.now(timezone.utc).isoformat()

    # Process + metrics in one call
    out = await exec_py(admin, '''
import json, psutil, asyncio
p = psutil.Process()
_children = p.children(recursive=True)
_children_rss = sum(c.memory_info().rss for c in _children if c.is_running())
_browser_count = sum(1 for c in _children if "chrome" in c.name().lower())
m = await store.get_metrics()
rpc_m = m["rpc"]["rpc_connections"]
eb_m = m["eventbus"]["eventbus"]
ws_set = store._websocket_server._websockets
server_id = store._server_id

redis = store.get_redis()
svc_total = 0
ws_counts = {}
clients = set()
async for key in redis.scan_iter(match="services:*", count=500):
    k = key.decode() if isinstance(key, bytes) else key
    svc_total += 1
    parts = k.split(":")
    if len(parts) >= 3:
        ws = parts[2].split("/")[0] if "/" in parts[2] else parts[2]
        ws_counts[ws] = ws_counts.get(ws, 0) + 1
        if "|built-in:" in k and k.endswith(":built-in@*"):
            clients.add(parts[2])

not_in_ws = sum(1 for c in clients if c not in ws_set and server_id not in c)
redis_info = await redis.info("clients")
import hypha as _hypha
top_ws = sorted(ws_counts.items(), key=lambda x: -x[1])[:8]
from collections import Counter as _Counter
_task_snap = [t for t in asyncio.all_tasks() if not t.done()]
_top_types = sorted(_Counter(getattr(t.get_coro(), "__qualname__", "?") for t in _task_snap).items(), key=lambda x: -x[1])[:8]
_proc_rss = round(p.memory_info().rss/1024/1024)
_children_mb = round(_children_rss/1024/1024)
import os as _os
_cgroup_path = "/sys/fs/cgroup/memory.current"
if _os.path.exists(_cgroup_path):
    _container_mb = round(int(open(_cgroup_path).read().strip()) / 1024 / 1024)
else:
    _container_mb = _proc_rss + _children_mb
print(json.dumps({
    "version": _hypha.__version__,
    "process": {
        "mem_mb": _proc_rss,
        "children_mem_mb": _children_mb,
        "container_mem_mb": _container_mb,
        "browser_workers": _browser_count,
        "threads": p.num_threads(),
        "fds": p.num_fds(),
    },
    "connections": {
        "active_rpc": rpc_m["active"],
        "active_ws": len(ws_set),
        "total_redis_clients": len(clients),
        "not_in_ws": not_in_ws,
        "redis_pool_connections": redis_info.get("connected_clients", 0),
    },
    "services": {"total": svc_total, "top_workspaces": top_ws},
    "eventbus": {"patterns": eb_m["patterns"]["active"]},
    "tasks": {
        "total": len(_task_snap),
        "heartbeat_total": sum(1 for t in _task_snap if "heartbeat" in getattr(t.get_coro(), "__qualname__", "")),
        "heartbeat_stuck": sum(
            1 for t in _task_snap
            if "heartbeat" in getattr(t.get_coro(), "__qualname__", "")
            and any(
                f.f_locals.get("method_task") is not None
                and hasattr(f.f_locals["method_task"], "done")
                and f.f_locals["method_task"].done()
                for f in t.get_stack()
            )
        ),
        "pending_rpc_calls": sum(1 for t in _task_snap if "Timer._job" in getattr(t.get_coro(), "__qualname__", "")),
        "rpc_object_entries": sum(
            len(v) for r in (o for o in __import__("gc").get_objects() if type(o).__name__ == "RPC" and hasattr(o, "_object_store"))
            for v in r._object_store.values() if isinstance(v, dict)
        ),
        "top_types": _top_types,
    },
    "ghost_last_seen": len(store._websocket_server._last_seen) - len(ws_set),
}))
''', 30)

    # Pod data
    pods_data = _kubectl_json("get", "pods")
    pods_summary = []
    oom_pods = []
    for item in pods_data.get("items", []):
        name = item["metadata"]["name"]
        containers = item.get("status", {}).get("containerStatuses", [{}])
        container = containers[0] if containers else {}
        restarts = container.get("restartCount", 0)
        # Check both lastState (restarted pods) and state (first-crash pods, restartCount=0)
        last_terminated = container.get("lastState", {}).get("terminated", {})
        cur_terminated = container.get("state", {}).get("terminated", {})
        exit_code = last_terminated.get("exitCode")
        oom_reason = last_terminated.get("reason") or cur_terminated.get("reason")
        if exit_code is None:
            exit_code = cur_terminated.get("exitCode")
        if oom_reason == "OOMKilled":
            oom_pods.append(name)
        pods_summary.append({"name": name, "restarts": restarts, "last_exit": exit_code})

    # hc-* pod memory usage
    def _parse_mem_mi(s: str) -> float:
        s = s.strip()
        if s.endswith("Gi"): return float(s[:-2]) * 1024
        if s.endswith("Mi"): return float(s[:-2])
        if s.endswith("G"):  return float(s[:-1]) * 1024
        if s.endswith("M"):  return float(s[:-1])
        if s.endswith("Ki"): return float(s[:-2]) / 1024
        return 0.0

    top_mem = {}
    top_out = _kubectl("top", "pods", "--no-headers")
    for line in top_out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].startswith("hc-"):
            top_mem[parts[0]] = parts[2]

    hc_pod_summary = []
    for item in pods_data.get("items", []):
        name = item["metadata"]["name"]
        if not name.startswith("hc-"):
            continue
        spec_c = item["spec"]["containers"][0] if item["spec"].get("containers") else {}
        mem_limit = spec_c.get("resources", {}).get("limits", {}).get("memory", "")
        mem_used = top_mem.get(name, "")
        mem_pct = None
        if mem_limit and mem_used:
            limit_mi = _parse_mem_mi(mem_limit)
            used_mi = _parse_mem_mi(mem_used)
            if limit_mi > 0:
                mem_pct = round(used_mi / limit_mi * 100, 1)
        cs = item.get("status", {}).get("containerStatuses", [{}])
        c = cs[0] if cs else {}
        cur_term = c.get("state", {}).get("terminated", {})
        if cur_term.get("reason") == "OOMKilled":
            status = "OOMKilled"
        elif c.get("state", {}).get("running"):
            status = "Running"
        else:
            status = item.get("status", {}).get("phase", "Unknown")
        hc_pod_summary.append({
            "name": name,
            "status": status,
            "mem_limit": mem_limit,
            "mem_used": mem_used or None,
            "mem_pct": mem_pct,
        })

    try:
        proc_data = json.loads(out.strip())
    except Exception:
        proc_data = {"error": out.strip()}

    report = {
        "timestamp": ts,
        "server": proc_data,
        "pods": {
            "total": len(pods_summary),
            "oom_killed": oom_pods,
            "high_restart": [p for p in pods_summary if p["restarts"] >= 10],
        },
        "hc_pods": {
            "total": len(hc_pod_summary),
            "running": sum(1 for p in hc_pod_summary if p["status"] == "Running"),
            "oom_killed": sum(1 for p in hc_pod_summary if p["status"] == "OOMKilled"),
            "pods": hc_pod_summary,
        },
        "alerts": [],
    }

    # Alerts
    mem = proc_data.get("process", {}).get("mem_mb", 0)
    container_mem = proc_data.get("process", {}).get("container_mem_mb", mem)
    rpc = proc_data.get("connections", {}).get("active_rpc", 0)
    svcs = proc_data.get("services", {}).get("total", 0)
    fds = proc_data.get("process", {}).get("fds", 0)
    redis_pool = proc_data.get("connections", {}).get("redis_pool_connections", 0)
    if mem > 3000: report["alerts"].append(f"HIGH MEMORY (process): {mem} MB")
    if container_mem > 5000: report["alerts"].append(f"HIGH MEMORY (container): {container_mem} MB")
    if rpc > 600: report["alerts"].append(f"HIGH RPC: {rpc}")
    if svcs > 1500: report["alerts"].append(f"HIGH SERVICES: {svcs}")
    if fds > 3000: report["alerts"].append(f"HIGH FDS: {fds}")
    if redis_pool > 900: report["alerts"].append(f"HIGH REDIS POOL: {redis_pool} connections")
    not_in_ws = proc_data.get("connections", {}).get("not_in_ws", 0)
    if not_in_ws > 5: report["alerts"].append(f"SUSPECT CLIENTS: {not_in_ws} not in WS (run 'zombies' to verify)")
    hb_stuck = proc_data.get("tasks", {}).get("heartbeat_stuck", 0)
    if hb_stuck > 10: report["alerts"].append(f"HEARTBEAT LEAK: {hb_stuck} stuck tasks (run cleanup-tasks)")
    pending_rpc = proc_data.get("tasks", {}).get("pending_rpc_calls", 0)
    if pending_rpc > 200: report["alerts"].append(f"HIGH PENDING RPC: {pending_rpc} calls in flight (Timer._job tasks)")
    rpc_obj = proc_data.get("tasks", {}).get("rpc_object_entries", 0)
    if rpc_obj > 200000: report["alerts"].append(f"HIGH RPC OBJECTS: {rpc_obj} entries in _object_store (>200k, investigate leak)")
    ghost = proc_data.get("ghost_last_seen", 0)
    if ghost > 0: report["alerts"].append(f"GHOST LAST_SEEN: {ghost} entries (clients disconnected without cleanup)")
    if oom_pods: report["alerts"].append(f"OOM PODS: {', '.join(oom_pods)}")
    near_oom = [f"{p['name']}({p['mem_pct']:.0f}%)" for p in hc_pod_summary
                if p["mem_pct"] is not None and p["mem_pct"] >= 75 and p["status"] == "Running"]
    if near_oom: report["alerts"].append(f"HC POD HIGH MEM: {', '.join(near_oom)}")
    top_ws = proc_data.get("services", {}).get("top_workspaces", [])
    # hypha-agents is expected to have many services (compute worker proxy registrations per app)
    # Only alert at >1000 for that workspace; other workspaces alert at >100
    COMPUTE_WORKSPACES = {"hypha-agents"}
    ws_anomalies = [
        f"{ws}({cnt})" for ws, cnt in top_ws
        if cnt > (1000 if ws in COMPUTE_WORKSPACES else 100)
    ]
    if ws_anomalies: report["alerts"].append(f"HIGH WS SERVICES: {', '.join(ws_anomalies)}")

    print(json.dumps(report, indent=2))





async def cmd_cleanup_tasks(server):
    """Cancel stuck heartbeat tasks where method_task is already done."""
    print("=== Cleanup Stuck Asyncio Tasks ===")
    admin = await get_admin(server)
    out = await exec_py(admin, '''
import asyncio, json
async def _cleanup():
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    hb = [t for t in tasks if "heartbeat" in str(getattr(t.get_coro(), "__qualname__", ""))]
    cancelled = 0
    for t in hb:
        for f in t.get_stack():
            mt = f.f_locals.get("method_task")
            if mt is not None and hasattr(mt, "done") and mt.done():
                t.cancel()
                cancelled += 1
                break
    return {"scanned": len(hb), "cancelled": cancelled, "remaining": len(hb) - cancelled}
data = await _cleanup()
print(json.dumps(data))
''', 30)
    try:
        data = json.loads(out.strip())
        print(f"Scanned: {data['scanned']} heartbeat tasks")
        print(f"Cancelled: {data['cancelled']} stuck tasks (method already done)")
        print(f"Remaining: {data['remaining']} legitimately running")
    except Exception:
        print(out)


async def cmd_tasks(server):
    """Show asyncio task counts by type (useful for detecting leaks like heartbeat accumulation)."""
    print("=== Asyncio Task Breakdown ===")
    admin = await get_admin(server)
    out = await exec_py(admin, '''
import asyncio, json
async def _tasks():
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    by_coro = {}
    for t in tasks:
        name = getattr(t.get_coro(), "__qualname__", str(t.get_coro()))
        by_coro[name] = by_coro.get(name, 0) + 1
    return {"total": len(tasks), "by_type": sorted(by_coro.items(), key=lambda x: -x[1])[:20]}
data = await _tasks()
print(json.dumps(data))
''', 15)
    try:
        data = json.loads(out.strip())
        print(f"Total tasks: {data['total']}")
        print(f"{'Coroutine type':<55} {'Count':>6}")
        print("-" * 64)
        for name, count in data["by_type"]:
            flag = " ⚠" if count > 200 else ""
            print(f"  {name:<53} {count:>6}{flag}")
    except Exception:
        print(out)


async def cmd_clients(server):
    """Show active clients per workspace with service and svc-per-client counts."""
    print("=== Clients Per Workspace ===")
    admin = await get_admin(server)
    out = await exec_py(admin, '''
import json

async def get_clients_by_workspace():
    redis = store.get_redis()
    ws_clients = {}
    ws_services = {}
    async for key in redis.scan_iter(match="services:*", count=500):
        k = key.decode() if isinstance(key, bytes) else key
        parts = k.split(":")
        if len(parts) >= 3:
            ws_client = parts[2]
            ws = ws_client.split("/")[0] if "/" in ws_client else ws_client
            client_id = ws_client.split("/")[1] if "/" in ws_client else "?"
            if ws not in ws_clients:
                ws_clients[ws] = set()
            ws_clients[ws].add(client_id)
            ws_services[ws] = ws_services.get(ws, 0) + 1
    result = []
    for ws in sorted(ws_clients.keys(), key=lambda w: -len(ws_clients[w])):
        result.append({
            "workspace": ws,
            "clients": len(ws_clients[ws]),
            "services": ws_services.get(ws, 0),
        })
    return result

data = await get_clients_by_workspace()
print(json.dumps(data))
''', 30)
    try:
        rows = json.loads(out.strip())
        print(f"{'Workspace':<50} {'Clients':>7} {'Services':>8} {'Svc/Client':>10}")
        print("-" * 78)
        for r in rows:
            spc = round(r["services"] / max(1, r["clients"]), 1)
            flag = " ⚠" if r["clients"] > 20 else ""
            print(f"  {r['workspace']:<48} {r['clients']:>7} {r['services']:>8} {spc:>10}{flag}")
    except Exception:
        print(out)


async def cmd_sessions(server):
    """List active app sessions in Redis with status and metadata summary."""
    print("=== Active App Sessions ===")
    admin = await get_admin(server)
    out = await exec_py(admin, '''
import json, time

async def get_sessions():
    keys = []
    async for k in redis.scan_iter("sessions:*"):
        keys.append(k.decode())

    rows = []
    for key in keys:
        data = await redis.hgetall(key)
        ttl = await redis.ttl(key)
        fields = {fk.decode(): fv.decode() for fk, fv in data.items()}
        # Extract workspace/client from key: sessions:<ws>/<client_id>
        parts = key.split(":", 1)
        ws_client = parts[1] if len(parts) > 1 else key
        ws = ws_client.split("/")[0] if "/" in ws_client else ws_client
        app_id = fields.get("app_id", fields.get("alias", "?"))
        status = fields.get("status", "?")
        worker_id = fields.get("worker_id", "")
        worker_id_short = worker_id[-12:] if worker_id else ""
        rows.append({
            "key": ws_client,
            "workspace": ws,
            "app_id": app_id,
            "status": status,
            "worker_id": worker_id_short,
            "ttl": ttl,
            "fields": len(fields),
        })
    return sorted(rows, key=lambda r: r["workspace"])

rows = await get_sessions()
print(json.dumps(rows))
''', 20)
    try:
        rows = json.loads(out.strip())
        if not rows:
            print("No active sessions in Redis.")
            return
        print(f"{'Workspace/Client':<40} {'App':<25} {'Status':<12} {'TTL':>6} {'Fields':>6}")
        print("-" * 94)
        for r in rows:
            ttl_str = f"{r['ttl']}s" if r["ttl"] > 0 else "no-ttl"
            print(f"  {r['key'][:38]:<38} {r['app_id'][:23]:<25} {r['status']:<12} {ttl_str:>6} {r['fields']:>6}")
        print(f"\nTotal: {len(rows)} sessions")
    except Exception:
        print(out)


async def cmd_status(server):
    await cmd_health(server)
    print()
    await cmd_workspaces(server)
    print()
    await cmd_quick_zombies(server)


async def cmd_kickout(server, workspace: str, client_id: str):
    admin = await get_admin(server)
    out = await exec_py(admin, f'''
import json
try:
    result = await store.kickout_client("{workspace}", "{client_id}", 1001, "Admin kickout")
    print(json.dumps({{"kicked": True, "result": str(result)}}))
except Exception as e:
    print(json.dumps({{"kicked": False, "error": str(e)}}))
''', 15)
    print(out)


async def cmd_scale(server, deployment: str, replicas: int):
    """Scale a Kubernetes deployment."""
    out = _kubectl("scale", "deployment", deployment, f"--replicas={replicas}")
    print(out)


async def cmd_version_check(server):
    """Compare live server version against latest GitHub release."""
    import urllib.request
    admin = await get_admin(server)

    # Get live version
    out = await exec_py(admin, "import hypha; print(hypha.__version__)", 10)
    live_ver = out.strip()

    # Get latest GitHub release
    url = "https://api.github.com/repos/amun-ai/hypha/releases/latest"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        latest_tag = data.get("tag_name", "?").lstrip("v")
        published = data.get("published_at", "?")[:10]
    except Exception as e:
        latest_tag = f"(error: {e})"
        published = "?"

    status = "UP TO DATE" if live_ver == latest_tag else "UPGRADE AVAILABLE"
    flag = "" if live_ver == latest_tag else " ⚠"
    print(f"=== Version Check ===")
    print(f"  Live server:     {live_ver}")
    print(f"  Latest release:  {latest_tag}  (published {published})")
    print(f"  Status:          {status}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

COMMANDS = {
    "health":         (cmd_health,         "Quick health check: process, metrics, pods, OOM"),
    "metrics":        (cmd_metrics,        "Full RPC/eventbus metrics JSON"),
    "workspaces":     (cmd_workspaces,     "Workspaces sorted by service count"),
    "services":       (cmd_services,       "List services [workspace]"),
    "quick-zombies":  (cmd_quick_zombies,  "Fast zombie check via WS dict (~1s, no ping)"),
    "zombies":        (cmd_zombies,        "Ping-verified zombie detection (~2s per suspect)"),
    "cleanup":        (cmd_cleanup_zombies,"Run orphan cleanup with before/after count"),
    "pods":           (cmd_pods,           "Pod status + resource usage"),
    "hc-pods":        (cmd_hc_pods,        "Compute (hc-*) pods: workspace, mem limit, status, age"),
    "logs":           (cmd_logs,           "Tail logs [pod-name]"),
    "exec":           (cmd_exec,           "Execute Python on server"),
    "report":         (cmd_report,         "Full JSON health report"),
    "cleanup-tasks":  (cmd_cleanup_tasks,  "Cancel stuck heartbeat tasks (method already done)"),
    "tasks":          (cmd_tasks,          "Asyncio task breakdown by type (leak detection)"),
    "clients":        (cmd_clients,        "Active clients per workspace with service count"),
    "sessions":       (cmd_sessions,       "List active app sessions in Redis with status/TTL"),
    "status":         (cmd_status,         "Full status: health + workspaces + quick-zombies"),
    "kickout":        (cmd_kickout,        "Kick client: kickout <workspace> <client_id>"),
    "scale":          (cmd_scale,          "Scale deployment: scale <name> <replicas>"),
    "version-check":  (cmd_version_check,  "Compare live server version against latest GitHub release"),
}


async def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "status"
    rest = args[1:]

    if cmd in ("-h", "--help", "help"):
        print("hypha-admin.py <command> [args...]")
        print()
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<18} {desc}")
        return

    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Run 'hypha-admin.py help' for usage", file=sys.stderr)
        sys.exit(1)

    server = await connect()
    fn = COMMANDS[cmd][0]
    try:
        if cmd == "services":
            await fn(server, rest[0] if rest else "")
        elif cmd == "logs":
            await fn(server, rest[0] if rest else "")
        elif cmd == "exec":
            await fn(server, rest[0] if rest else 'print("hello")')
        elif cmd == "kickout":
            if len(rest) < 2:
                print("Usage: hypha-admin.py kickout <workspace> <client_id>", file=sys.stderr)
                sys.exit(1)
            await fn(server, rest[0], rest[1])
        elif cmd == "scale":
            if len(rest) < 2:
                print("Usage: hypha-admin.py scale <deployment> <replicas>", file=sys.stderr)
                sys.exit(1)
            await fn(server, rest[0], int(rest[1]))
        else:
            await fn(server)
    finally:
        await server.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
