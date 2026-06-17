"""Empirical reproduction harness for the N>=2 reject-storm (Issue #2).

Launches TWO real hypha-server processes sharing one real Redis, with BOTH
servers' stderr captured, then runs the prod scenario: a stable caller
repeatedly calls a service registered by a "worker" client that connects,
serves an in-flight call, and disconnects mid-call — repeatedly, with a new
client_id each cycle (the user's reconnect-loop pattern).

After the scenario it scans both servers' captured stderr for the storm markers
("Message routing failed", "Target peer ... is not connected") so we can SEE
whether the storm reproduces and what produces it — instead of guessing.

Requires Docker (redis-stack) via the redis_server fixture + MinIO.
"""
import asyncio
import os
import signal
import subprocess
import sys
import time

import pytest
from hypha_rpc import connect_to_server

from . import (
    REDIS_PORT,
    MINIO_SERVER_URL,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
    MINIO_SERVER_URL_PUBLIC,
)

pytestmark = pytest.mark.asyncio


def find_free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(server_id, port, extra=()):
    env = os.environ.copy()
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "hypha.server",
            f"--port={port}",
            "--public-base-url=http://my-public-url.com",
            f"--server-id={server_id}",
            f"--redis-uri=redis://127.0.0.1:{REDIS_PORT}/0",
            *extra,
        ],
        env=env,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge so we capture everything in one stream
    )
    return proc


def _wait_health(url, timeout=60):
    import httpx
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{url}/health/readiness", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


async def test_repro_n2_reject_storm(redis_server, minio_server, test_user_token):
    port_a = find_free_port()
    port_b = find_free_port()
    proc_a = _start_server(
        "repro-a", port_a,
        extra=(
            "--enable-server-apps", "--enable-s3", "--reset-redis",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
            "--workspace-bucket=my-workspaces", "--s3-admin-type=minio",
        ),
    )
    url_a = f"http://127.0.0.1:{port_a}"
    assert _wait_health(url_a), "server A did not start"
    time.sleep(2)
    proc_b = _start_server("repro-b", port_b)
    url_b = f"http://127.0.0.1:{port_b}"
    assert _wait_health(url_b), "server B did not start"

    try:
        # Admin sets up a workspace + token.
        admin = await connect_to_server(
            {"client_id": "repro-admin", "server_url": url_a, "token": test_user_token}
        )
        ws = await admin.create_workspace(
            {"name": "repro-ws", "persistent": True, "owners": ["user1@imjoy.io"]},
            overwrite=True,
        )
        token = await admin.generate_token({"workspace": "repro-ws"})

        # Stable caller on pod A holds a reference to the worker's service and
        # keeps calling it across the worker's reconnect cycles.
        caller = await connect_to_server(
            {"client_id": "caller", "server_url": url_a, "workspace": "repro-ws",
             "token": token, "method_timeout": 5}
        )

        async def slow(x):
            await asyncio.sleep(0.4)
            return x * 2

        # Reconnect loop: worker connects (new client_id), registers svc, caller
        # fires an in-flight call, worker disconnects MID-CALL. Repeat.
        for i in range(12):
            worker = await connect_to_server(
                {"client_id": f"worker-{i}", "server_url": url_a, "workspace": "repro-ws",
                 "token": token, "method_timeout": 5}
            )
            await worker.register_service(
                {"id": "svc", "config": {"visibility": "public", "require_context": False},
                 "slow": slow}
            )
            svc = await caller.get_service(f"worker-{i}:svc")
            fut = asyncio.ensure_future(svc.slow(i))  # in-flight
            await asyncio.sleep(0.05)
            await worker.disconnect()  # mid-call disconnect
            try:
                await asyncio.wait_for(fut, timeout=3)
            except Exception:
                pass
            await asyncio.sleep(0.2)

        await asyncio.sleep(2)  # let any storm flush to logs
        await caller.disconnect()
        await admin.disconnect()
    finally:
        for proc in (proc_a, proc_b):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
        out_a = proc_a.communicate(timeout=15)[0].decode("utf-8", "replace")
        out_b = proc_b.communicate(timeout=15)[0].decode("utf-8", "replace")

    def _count(s, needle):
        return s.count(needle)

    a_routing = _count(out_a, "Message routing failed")
    b_routing = _count(out_b, "Message routing failed")
    a_notconn = _count(out_a, "is not connected")
    b_notconn = _count(out_b, "is not connected")
    print(f"\n=== REPRO RESULTS ===")
    print(f"pod A: 'Message routing failed'={a_routing}  'is not connected'={a_notconn}")
    print(f"pod B: 'Message routing failed'={b_routing}  'is not connected'={b_notconn}")
    # Dump a sample of the storm lines for inspection.
    for label, out in (("A", out_a), ("B", out_b)):
        lines = [ln for ln in out.splitlines() if "not connected" in ln or "Message routing failed" in ln]
        if lines:
            print(f"--- pod {label} sample ({len(lines)} lines) ---")
            for ln in lines[:6]:
                print(ln[-300:])
    # Assertion: with the fix, there should be no storm. (Initially expected to
    # FAIL, reproducing the bug, so we can see the counts and the messages.)
    assert a_routing + b_routing + a_notconn + b_notconn == 0, (
        f"reject-storm reproduced: routing_failed A={a_routing} B={b_routing}, "
        f"not_connected A={a_notconn} B={b_notconn}"
    )
