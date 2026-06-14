# F6 — Multi-Replica Hypha-Server: Design & Scoping

> Status: **Draft / scoping**. Authored from a code audit of the data plane and
> control plane. No code changes proposed here — this document scopes the work
> and sequences it. Related: [`TODO-scalability.md`](TODO-scalability.md),
> [`autoscaling.md`](autoscaling.md).

## 1. Motivation

Today `hypha-server` runs as a **single replica** (no HPA) — a shared
single-point-of-failure across many products. The svamp reliability audit
established the concrete driver:

- **F4** (graceful WS + HTTP-stream drain on shutdown, shipped in 0.21.86) makes
  clients reconnect *cleanly and immediately* on rollout. But on a **single
  replica** there is nowhere warm to land: the reconnect races the new pod's
  cold-start + service re-registration window (~90 s observed in prod), which
  dominates the blip. F4's benefit is therefore **structurally unobservable**
  until a surviving/warm replica exists.
- The only reconnect "storms" in the 7-day svamp logs were **rollout-induced**;
  a multi-replica deployment with a rolling update keeps ≥1 warm replica in the
  Service endpoints at all times, so clients drain off the terminating pod (F4 +
  the F5 preStop drain) and immediately re-attach to a peer that is already
  serving.

**Goal:** run `hypha-server` with N ≥ 2 replicas (and eventually an HPA) safely,
so that rollouts and single-pod failures cause **zero or near-zero** client-visible
disruption, and so the per-call auth/resolution overhead (the real per-replica
ceiling noted in the audit) is spread across replicas.

**Non-goals (this doc):** the per-call auth/service-resolution caching
optimization (separate item), and worker-side autoscaling policy redesign.

## 2. Architecture summary: what is already shared vs per-instance

Hypha was built on a Redis event bus precisely to scale horizontally, and the
**data plane is already multi-replica-correct**. The gaps are in the **control
plane** — singleton-ish background loops and a non-idempotent startup that each
replica would run independently.

### 2.1 Data plane — already safe ✅

| Concern | Mechanism | Evidence |
| --- | --- | --- |
| Targeted client→client messaging across instances | `targeted:{ws}/{client}:*` Redis pub/sub; each instance `PSUBSCRIBE`s only for its **local** clients; sender skips Redis for local clients (fast path) | `core/__init__.py` `RedisEventBus.emit` ~1492–1559, `subscribe_to_client_events` ~1259–1286, `is_local_client` ~1315 |
| Connection affinity (socket lives on one instance) | Other instances publish to Redis; the owning instance receives and forwards to its local socket | `core/__init__.py` ~1505–1514, 1668–1711; `websocket.py` handler ~761; `http_rpc.py` ~138 |
| Service discovery | Services stored in Redis by key pattern; any instance can resolve any service | `core/workspace.py` register/lookup ~2282, ~2427 |
| Reconnection across instances | `reconnection_token` is a self-contained JWT; validated on **any** instance; client_id/workspace re-bound from token scope | `websocket.py` ~373–407; `http_rpc.py` `_authenticate_with_reconnection_token` ~185–224 |
| Duplicate-client takeover | `check_client` pings the existing client **through the event bus**, so it works even if the old socket is on another instance | `websocket.py` ~353–371; `core/workspace.py` `ping_client` ~1416–1456 |
| Teardown cleanup scope | `_clear_all_server_services()` deletes only **this** server's keys (scoped by `server_id` / `manager-{server_id}`) | `core/store.py` ~679–723 |
| App/session state | Persisted in Redis (`sessions:{ws}/{client}`); any instance can stop/manage an app started elsewhere | `apps.py` ~556–590, `_get_all_sessions` ~622, `_stop` ~2830 |
| Client-disconnect cleanup | Idempotent (Redis HDEL); safe if multiple instances react | `apps.py` ~437–505, 604 |

**Hard requirement:** N > 1 **requires a real Redis** broker. `fakeredis` is
in-process only — two server processes each get their own in-memory Redis and
cannot see each other's pub/sub. (Confirmed; single-replica may keep using
fakeredis.)

**Sticky affinity is the recommended Phase-1 LB strategy.** Reconnection tokens
validate on any replica, so round-robin is correct *for connection setup*.
However, the local-client fast-path (P1b) and per-instance manager (P1c) create
correctness edges when a client **migrates** between replicas. Sticky / consistent-hash
affinity (pin each client to one replica) makes both moot and is the pragmatic
Phase-1 lever; true affinity-free round-robin needs the Phase-2 distributed client
registry. See §5.

### 2.2 Control plane — needs work ⚠️

| # | Problem | Severity | Evidence | Root issue |
| --- | --- | --- | --- | --- |
| P0 | **`HYPHA_RESET_REDIS=true` flushes the SHARED Redis on every replica boot.** With N replicas, each cold start wipes the others' state. | **HIGH** | `store.py` ~816–818 | Config assumes single owner of Redis |
| P1 | **Startup double-registration.** Every replica's `store.init()` registers built-in/public services (login, queue, workspace-manager) and creates the public/root/anonymous workspaces & their clients. Not idempotent; concurrent `init()` races (check-then-register). | **HIGH** | `store.py` `init()` ~813–966 (esp. login ~942–948, queue ~950–957, public-services loop ~916–935); `workspace.py` register ~2284–2316 | No leader-election / idempotent register |
| P1b | **Local-client fast-path misroutes on replica migration.** Targeted messages to a client in a replica's in-memory `_local_clients` set skip Redis. If a client moves to another replica, the *old* replica still has it locally and delivers to a dead socket instead of publishing to Redis for the new owner. (Sticky affinity makes this moot — see §5.) | **MED** (HIGH without affinity) | `core/__init__.py` ~1505–1514; `_local_clients` ~1210 | Per-instance registry, not Redis-backed with TTL |
| P1c | **Manager affinity.** Each replica runs `manager-{server_id}`; clients cache a `manager_id` and route manager RPC to it. Reconnect to a different replica → calls target the old manager. (hypha-rpc 0.21.41/F1 retargets `wm`/getService to the new `manager_id` on reconnect, mitigating client-side; sticky affinity removes it entirely.) | **MED** | `store.py` 241–242; `websocket.py` ~504 / `http_rpc.py` ~360 (manager_id in connection_info) | Per-instance manager + client-side pinning |
| P2 | **Autoscaling runs N times.** Each replica runs its own `AutoscalingManager` monitor loop per app, guarded only by a **local** `asyncio.Lock`. N replicas independently decide to scale the same app every 10 s → over-scaling / thrashing. Cooldown timers are per-instance. | **HIGH** | `apps.py` `AutoscalingManager` ~206–263, `_monitor_app_load` ~248–263, `_check_and_scale`, `_last_scale_time`/`_scaling_locks` ~206–212 | No distributed lock / single owner |
| P3 | **Workspace activity-cleanup race.** Each replica runs a `WorkspaceActivityManager`; N trackers may concurrently unload/delete the same inactive workspace. HDEL is idempotent, but the **unload callbacks** (S3 / artifact cleanup) can race. | **MED** | `workspace.py` `WorkspaceActivityManager` ~139–296, `_cleanup_inactive_workspace` ~256–282 (plain `hexists`→`hdel`, no WATCH/revision) | No leader / no transactional guard |
| P4 | **App inactivity-stop fires N times.** Each replica registers its own inactivity tracker per session; multiple replicas may each call `_stop_after_inactive` → stop the app repeatedly. | **MED** | `apps.py` ~2635–2657 | Per-instance tracker; no Redis recheck |
| P5 | **Worker sessions orphaned on instance death.** Workers hold session state **in-memory**; Redis keeps only metadata. If the instance that launched a worker dies, its sessions are unrecoverable by peers (`worker.stop` raises `SessionNotFound`). | **MED** | `apps.py` worker cache ~587–588; `workers/browser.py` `_sessions` ~88–89, stop ~467 | No lease/heartbeat; stateful workers |
| P6 | **Per-instance quotas & rate limits.** Per-user / per-workspace connection limits and the WS token-bucket rate limiter are in-memory per replica → effective limits become ≈ ×N; a re-routed client gets a fresh bucket. | **LOW** | `websocket.py` semaphore ~87, rate limiter ~43–63/528–540; `resource_limits.py` counters ~38–172 | Not aggregated across replicas |
| P7 | **Slower cross-instance dead-peer detection.** Fast-path "recently disconnected" set is local; cross-instance dead clients fall back to a 120 s TTL. | **LOW** | `core/__init__.py` `_recently_disconnected` ~1214–1215 | Local-only optimization |

## 3. Proposed approach

### 3.1 One primitive solves most of the control-plane gaps: a Redis leader lease

P1, P2, P3, and (a future) worker health monitor are all **"exactly-one-runner"**
problems. Introduce a small **distributed leader lease** (Redis `SET key val NX PX=ttl`
renewed on an interval; value = `server_id`) and gate the singleton work behind
"am I the leader?":

- **Leader-only loops:** autoscaling monitor (P2), workspace activity cleanup
  (P3), housekeeping (currently commented out in `store.py`), and any future
  worker health monitor.
- **Followers** still serve all client traffic, RPC, and service registration —
  they just don't run the singleton background controllers.
- On leader loss/crash the lease expires (PX TTL) and another replica acquires
  it; loops resume there. This is eventually-consistent and crash-safe.

This is deliberately *not* full Raft — a single Redis lease is adequate because
the protected work is idempotent-ish and tolerant of a brief gap during failover.

### 3.2 Idempotent startup (P1)

Two complementary fixes (do both):
1. **Idempotent built-in registration** — registering an already-present
   built-in service should be a no-op upsert keyed deterministically (not a
   blind replace racing peers). Workspace creation already uses `overwrite=False`;
   extend the same "create-if-absent, tolerate-exists" discipline to the login /
   queue / manager services, and make the check-then-register atomic (Redis
   `SETNX`-style guard or accept the upsert as authoritative).
2. **Leader-gated one-time setup** — the genuinely once-only bits (e.g. default
   data seeding) run only on the leader.

The workspace-manager service itself is per-replica by design (each replica must
serve management RPC for its local clients) — that one is **not** a singleton and
should remain per-instance. The audit must distinguish "must be per-replica"
(manager service, connection handlers) from "must be one" (autoscaler, cleanup).

### 3.3 App lifecycle hardening (P4, P5)

- **P4:** before `_stop_after_inactive` actually stops an app, re-check Redis for
  recent activity / existence, and/or gate inactivity-driven stops behind the
  leader. Cheapest correct fix: recheck-then-stop.
- **P5:** add a **worker lease/heartbeat**. Each worker (or the instance that owns
  it) heartbeats into Redis; on missed heartbeats a (leader-run) reaper marks the
  session dead and cleans Redis metadata, instead of calling a dead worker. Longer
  term, prefer **stateless / externally-recoverable** worker sessions. This is the
  largest sub-effort and can be phased last.

### 3.4 Quotas & limits (P6) — optional

For correctness most deployments tolerate per-replica limits. If global fairness
matters, move per-user / per-workspace counters to Redis (atomic `INCR`/`DECR`
with the connection lifecycle). Treat as a follow-up, not a blocker.

## 4. Deployment / Kubernetes changes

- **Replicas:** `replicas: 2+` (start at 2; add an HPA on CPU/connection count
  later). Real Redis is already part of the `hypha-server-kit`.
- **Rolling update:** `maxUnavailable: 0`, `maxSurge: 1` so a warm replica always
  remains in the Service endpoints during a rollout (this is what makes F4 pay off).
- **preStop drain (F5):** already validated in prod (kth-k8s chart 0.21.87:
  `terminationGracePeriodSeconds: 45` + `preStop: sleep 15`). Keep it — combined
  with F4 it gives clean drain *and* clean reconnect.
- **Redis:** mandatory and must be HA itself for the cluster to be truly resilient
  (Redis becomes the new shared dependency; a managed/replicated Redis is assumed).
- **Probes:** ensure readiness gates on `_ready` so a cold-starting replica is not
  added to endpoints until built-in services are registered (reduces the
  re-registration blip for clients that land early).

## 5. Load-balancing & affinity recommendation

**Phase 1: sticky / consistent-hash affinity (recommended).** Pin each client
(`client_id` hash or cookie) to one replica. This keeps the local-client fast-path
(P1b) and per-instance manager (P1c) valid by construction — a client never
migrates mid-session — which lets us ship multi-replica *without* first building a
distributed client registry. It also keeps per-client rate-limit/quota state on one
replica and avoids cross-instance `check_client` pings on reconnect.

**Phase 2: affinity-free round-robin** becomes safe once the distributed client
registry (Redis-backed register/deregister + TTL) replaces the per-instance
`_local_clients` set and the manager is made stateless/distributed. Until then,
do **not** run affinity-free.

> Infra note (resolved, peer-owned — see hypha-cloud PR #4): `hypha.aicell.io` runs
> on **classic NGINX Ingress** (not NGF Gateway); annotations are honored (WebSocket
> connections hold >60s), so **session affinity is feasible today with no ingress
> swap**. Recommended: **consistent-hash `upstream-hash-by` on `client_id`** (more
> robust than cookie affinity for long-lived daemon clients).

## 6. Phased plan

1. **Phase 0 — De-risk & gate (no behavior change at N=1):**
   add the Redis leader-lease primitive + an `is_leader()` check; unit-test it.
2. **Phase 1 — Gate the singletons:** autoscaling (P2) and workspace activity
   cleanup (P3) become leader-only. Make startup registration idempotent (P1).
   This unblocks running N=2 safely for the common case (no app-autoscaling churn,
   no double cleanup).
3. **Phase 2 — App lifecycle:** P4 recheck-before-stop; begin P5 worker
   lease/heartbeat + leader-run reaper.
4. **Phase 3 — K8s rollout:** bump replicas to 2 with `maxUnavailable: 0`, add
   readiness gating, observe a rollout — **this is where F4 finally validates**
   (terminating 0.21.86 pod emits close-frames, clients land on the warm peer; the
   residual probe-400/timeout should disappear).
5. **Phase 4 — Scale-out:** add HPA; optionally Redis-backed quotas (P6) and faster
   cross-instance dead-peer detection (P7).

## 7. Testing strategy

- **Leader lease:** unit tests for acquire/renew/expire/failover (real fakeredis
  for single-process; a 2-instance test sharing one Redis to assert only one
  acquires).
- **Startup idempotency:** spin up two `RedisStore.init()` against one Redis;
  assert exactly one login/queue/manager service and one set of system workspaces;
  no errors on the second init.
- **Autoscaling single-owner:** two controllers + one app over threshold → assert
  scale actions originate from one instance only.
- **Activity-cleanup race:** two activity managers + one idle workspace → assert a
  single unload, callbacks fire once.
- **Cross-instance messaging / reconnect:** two server processes on one real Redis;
  client connected to A receives a message emitted from B; client reconnects via
  token to B after A drops it.
- **Rollout integration:** the F4/F5 validation the svamp audit wants — a 2-replica
  rolling restart with the daemon log showing immediate clean reconnect and no
  10–20 s heartbeat gap.

## 8. Open questions & risks

- **Redis as new SPOF/throughput ceiling.** Multi-replica trades a stateless-pod
  SPOF for a Redis dependency. Need HA Redis and a view on pub/sub throughput at
  the target client count.
- **Per-call overhead still applies per replica.** Multi-replica raises the
  ceiling but doesn't remove the per-RPC JWT parse + service lookup; pair with the
  auth/resolution caching item for real headroom.
- **Worker statefulness (P5)** is the deepest change; Phases 0–3 deliver most of
  the rollout/HA benefit without it, so it can trail.
- **Leader failover gap.** Brief windows with no active autoscaler/cleanup during
  failover are acceptable (work is idempotent and resumes); confirm no business
  logic requires sub-second continuity.

## 9. Summary

The data plane is already horizontally scalable; **F6 is mostly a control-plane
hardening effort**, and a single Redis leader-lease primitive resolves the two
HIGH and one MED issues that block a safe N=2. Worker-session recovery (P5) is the
one substantial deeper change and can be phased last. Multi-replica is also the
prerequisite that finally makes the shipped F4/F5 reconnection work observably
pay off on rollout.

## 10. Reconciliation with infra-side scoping (svamp peer ff2018f0)

The svamp peer independently audited the codebase. The two reads agree and merge
into a sharper plan. **Key combined insight: sticky affinity and a leader-lease are
orthogonal and BOTH needed in Phase 1** — affinity fixes the *data-plane* migration
edges (P1b/P1c), the leader-lease fixes the *control-plane* singletons (P1/P2/P3).
Neither alone is sufficient.

**Merged blocker set:** infra side surfaced P0 (`HYPHA_RESET_REDIS` flush) and
sharpened P1b/P1c (routing/manager affinity); this audit surfaced the control-plane
singletons P2 (autoscaling runs N×) and P3 (activity-cleanup race) that affinity
does **not** address. Together they are the complete Phase-1 list.

**Agreed phasing (merged):**
- **Phase 1 — zero-blip rollouts (quick win):** 2+ replicas + **sticky affinity** +
  `HYPHA_RESET_REDIS=false` + **first-replica/leader-gated init** + **leader-gated
  autoscaling & activity-cleanup loops** + HPA. A warm replica during rollouts kills
  the ~90 s cold-start blip and finally validates F4/F5.
- **Phase 2 — elastic / affinity-free (~6–12 wk):** distributed client registry
  (replaces `_local_clients`, fixes the local-only fast-path), distributed/stateless
  manager, worker session lease/recovery (P5).

**Division of labor:**
- **Peer (infra / kth-k8s):** ingress sticky/consistent-hash routing on NGINX Gateway
  Fabric (confirm support), HPA + `replicas` + `maxUnavailable: 0` in the chart,
  `HYPHA_RESET_REDIS=false` in deploy config.
- **This side (hypha):**
  - (a) **Empirically validate** that `targeted:` Redis routing actually delivers an
    *RPC* (not just a message) cross-replica today — 2-replica + real Redis.
  - (b) **First-replica-only init behind a Redis lock**, and extend it to a renewable
    leader-lease gating the ongoing singleton loops (autoscaling P2, activity-cleanup
    P3). Make built-in registration idempotent (P1).
  - (c) **2-replica local integration test** (docker-compose) proving cross-replica
    `get_service` + reconnect work.
  - Scope the Phase-2 distributed client registry.
