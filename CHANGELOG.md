# Hypha Change Log

### 0.21.102

 - **Support shallow git clone (`git clone --depth=N`)** over the smart-HTTP transport (#986). The upload-pack handler parsed the client's `deepen` request but ignored it — it replied with `NAK` where the client expected the shallow/unshallow section, so `git clone --depth=1` failed with `fatal: expected shallow/unshallow, got NAK; the remote end hung up unexpectedly` (full clone always worked). Now `_parse_upload_pack_request` extracts `deepen`/`shallow`/`done`; `resolve_pack_shas_shallow` does a breadth-first, depth-bounded walk (each want is depth 1) that collects each included commit's full tree + blobs but cuts off parents beyond the limit, marking the boundary commits; and the response emits the `shallow <sha>` section. The stateless-RPC two-round dance is handled: the negotiation round (wants+`deepen`, no `done`) returns **only** the shallow-update (shallow lines + flush, no pack), and the round carrying `done` returns shallow-update + NAK + pack. **All shallow logic is gated on `depth`, so the full-clone path is byte-for-byte unchanged.** Tests: `tests/test_git.py::test_git_shallow_clone_depth` (`--depth=1`→1 commit, `--depth=2`→2, full clone→3, content verified).
 - **Validate artifact alias length** (#987). The `create` handler validated alias *format* but not *length*, so a 104-char alias was accepted (HTTP 200) and flowed into URLs, breadcrumb UIs and S3/file-system paths (a slug-injection surface). Aliases longer than `MAX_ARTIFACT_ALIAS_LENGTH` (64) are now rejected with a clear error; the raw user-provided alias is checked so `{uuid}`/`{timestamp}` template placeholders (which expand server-side) are unaffected. Test: `tests/test_artifact.py::test_artifact_alias_length_validation` (104→rejected, 64→accepted, 65→rejected).

### 0.21.101

 - Fix Git LFS **download** failing anonymously on public git-storage artifacts. `git clone` of a public repo fetched the git pack fine but the LFS smudge failed — `batch response: Repository or object not found` — and a direct POST to `…/<alias>.git/info/lfs/objects/batch` returned `404 {"detail":"Repository not found"}` for anonymous requests, while the smart-HTTP git protocol (`info/refs`) and the REST file endpoint served the same object anonymously. Root cause: `hypha/git/lfs.py::create_lfs_router.lfs_auth` returned **`None`** for unauthenticated requests, whereas the smart-HTTP git route's `git_optional_auth` falls back to `login_optional` (yielding an **anonymous `UserInfo`**). With `user_info=None`, `get_lfs_handler_callback` → `_get_artifact_with_permission(None, …)` could not resolve even a `"*": "r"` public repo → 404. git-lfs discovers the LFS endpoint as `auth=none` and POSTs the batch anonymously, so the smudge always failed for anonymous clones. Fix: `lfs_auth` now falls back to `login_optional` for the anonymous / no-token / bad-token cases (mirroring `git_optional_auth`), so public LFS objects download anonymously while private ones still get a clean 401/403. Upload-side LFS and authenticated download were unaffected. Test: `tests/test_git.py::test_git_lfs_batch_anonymous_public_repo` (public repo + anonymous batch → 200; fails as 404 without the fix).

### 0.21.100

 - Fix the MCP-driven server OOM completely — **two** distinct per-request leaks on the JSON-RPC POST path, both proven (and the fix verified) on dev with gc **object counts** (RSS is masked by warm-allocator reuse and misled the initial triage). Validation: 2600 mixed requests → all four object families bounded at ~1/service (vs +1/request before).

   **(1) Per-request adapter leak.** `MCPRoutingMiddleware._handle_mcp_request` built a **new** `HyphaMCPAdapter` (+`StreamableHTTPSessionManager`+`RedisEventStore`) for **every** request. Combined with `0.21.98`'s run-once, each request's adapter started a `run()` background task that awaited `_mgr_stop` forever (only set by `shutdown()`, only on cache eviction, which per-request adapters never undergo) → the event loop pinned every adapter. Dev: 400 requests → +400 `HyphaMCPAdapter`/`RedisEventStore`/`StreamableHTTPSessionManager`. Fix: route `_handle_mcp_request` through the per-service `_get_or_create_mcp_app` cache (one adapter per service, reused), cache-miss creation **lock-guarded** so concurrent first requests can't orphan extras. The cached entry keeps its workspace-interface **context manager** alive (closed on eviction) so the reused adapter's `service` proxy stays valid — also fixes a latent SSE bug where `_cleanup_mcp_app` called `__aexit__` on the returned Munch interface (no `__aexit__`) instead of the context manager, so SSE interfaces never actually closed on eviction.

   **(2) Per-request `ServerSession` leak (run-once was the wrong model).** With the adapter cached, a *second* leak surfaced: +1 `ServerSession` per request (2600 reqs → +2600, non-recovering). Root cause: the SDK's stateless `StreamableHTTPSessionManager` spawns a `run_stateless_server` task into its `run()` task group per request, and that task is **only reaped when the task group is cancelled — i.e. when `run()` exits** (`mcp/server/streamable_http_manager.py`). `0.21.98`'s held-open run-once therefore never reaped them. Fix: revert to entering **and exiting** `run()` **per request** — but with a fresh, **lightweight** `StreamableHTTPSessionManager` created per request in `HyphaMCPAdapter.handle_request`, while the heavy/stateful parts (service proxy, MCP server handlers, event store, workspace interface) are reused from the cached adapter. Entering+exiting `run()` tears the task group down and reaps the `ServerSession`; the cached adapter means no per-request heavy work. The `0.21.98` run-once machinery (`_run_session_manager`/`_ensure_session_manager_running`/`_mgr_*`) is removed; `shutdown()` is now a no-op hook.

   Tests: `tests/test_mcp.py::test_mcp_middleware_caches_adapter_per_service` (adapter built once per service, interface kept alive) and `::test_mcp_handle_request_per_request_manager_enters_and_exits` (fresh manager per request, `run()` entered **and** exited each time → session reaped); full `tests/test_mcp.py` (HTTP + SSE) green; dev object-count PASS bar met on both happy (valid `type:mcp`) and bare-echo paths.

### 0.21.99

 - Durable, Postgres-backed admin blocklist. `block_user`/`block_ip` previously lived only in Redis, so on prod (which runs `HYPHA_RESET_REDIS=true`) every pod restart wiped the blocklist — blocks didn't survive restarts/OOMs/deploys. The blocklist is now persisted in a `blocklist` table (Postgres, `hypha/resource_limits.py::BlockEntry`) as the **source of truth**, with Redis kept as a fast cache for the per-request `check_blocked` hot path. On startup (`RedisStore.init`, right after `_maybe_reset_redis`) the cache is **repopulated from Postgres** (`sync_blocklist_to_redis`), so blocks survive a Redis reset/flush/data-loss — not just pod restarts; expired rows are pruned. `block`/`unblock` write through to both stores; `list_blocked` reads the durable store (`ttl_seconds=-1` = permanent). Degrades to Redis-only when no SQL engine is present (tests/dev). Tests: `tests/test_resource_limits.py::TestDurableBlocklist` (incl. a simulated Redis flush). Complements the prod config change `HYPHA_RESET_REDIS=false`.

### 0.21.98

 - Fix an MCP-driven server OOM. The MCP adapter is cached and shared across all requests for a service, and `StreamableHTTPSessionManager.run()` owns an internal anyio task group designed to be entered **once per instance**. `handle_request` entered `run()` **per request** on the shared manager, so under an MCP request flood (a safe-colab security-agent client stuck in a code-review retry loop generated ~700k requests, ~15/s bursts) task groups accumulated → a transient ~400 MB→8 G spike → `OOMKilled` (the event loop blocked during the burst — `admin-utils` `execute_command` timed out, a tell). Now `run()` is started lazily **once** and held open by a dedicated background task; every request dispatches via `session_manager.handle_request` into that single running manager, with concurrent first-requests lock-guarded and `_cleanup_mcp_app` calling `adapter.shutdown()` to tear down the task group on eviction. Test: `tests/test_mcp.py::test_mcp_session_manager_run_entered_once` (25 concurrent + 25 sequential requests → `run()` entered exactly once, clean shutdown).
 - Enforce the user/IP blocklist on HTTP + MCP requests, not just WebSocket connect. `block_user`/`block_ip` were only consulted in `WebsocketServer.check_connection_allowed` (WS connect), so a blocked principal could keep hammering stateless HTTP `/rpc` + `/mcp` endpoints unchecked — i.e. blocking an abusive MCP client had no effect on its actual vector. The blocklist check is factored into `ResourceLimitsManager.check_blocked(user_id, ip)` and now also runs in `RedisStore.login_optional` (the common HTTP + MCP auth boundary), returning HTTP 403 for a blocked user/IP (honoring `X-Forwarded-For`). Fails open if Redis is unavailable. Tests: `tests/test_resource_limits.py::TestCheckBlocked`.

### 0.21.97

 - Fix dedicated browser-worker routing (unblocks taking headless-Chromium memory off the API pod). `hypha/workers/browser.py`'s `__main__` set the `--visibility`/`HYPHA_VISIBILITY` override at the **top level** of the service config (`service_config["visibility"]`) instead of inside `config` (`service_config["config"]["visibility"]`), which is where `register_service` reads it. The override was silently dropped, so a dedicated browser-worker pod always registered with the base default `config.visibility="protected"` and its `compile()` was un-invokable cross-workspace by the `public` server-apps controller ("Permission denied … workspace mismatch: ws-user-root != public") — blocking the dedicated-worker cutover. browser.py now matches the k8s/conda/terminal workers (which already nest it correctly). Regression guard: `tests/test_worker.py::test_worker_main_sets_visibility_in_config_not_toplevel`.
 - Consolidates the git smart-HTTP memory work that shipped in the 0.21.95/0.21.96 images: streaming clone + disk-spill push + per-op concurrency cap (#977), and range-read S3 pack for O(1) clone/fetch memory independent of pack size (#979).

### 0.21.90

 - Fix a frontend crash on the Applications page (`TypeError: Cannot read properties of null (reading 'startsWith')`). `server-apps.list_apps` returned the stored manifest verbatim and trusted it to carry an `id`; apps created directly via the artifact manager (bypassing `apps.install`, where the manifest id is persisted) — or installed before that id was persisted — had `manifest.id == None`, so `app.id.startsWith('public/')` threw. `list_apps` now derives `id` from the authoritative artifact alias (matching the value `start`/`uninstall`/`edit_app` expect) instead of trusting the manifest blob. Test: `tests/test_server_apps.py::test_list_apps_heals_null_manifest_id`.

### 0.21.89

 - Resilient embedding-model load: `load_fastembed_model` retries the fastembed model download (`BAAI/bge-small-en-v1.5`) on transient HuggingFace/CDN failures ("Could not load model … from any source") with linear backoff, instead of hard-failing a server startup (or a CI run) on a momentary blip. Applied to all vector backends (pgvector, s3vector, vectors).
 - Fix the multi-replica reject-storm (F6, second N≥2 facet). When a client with in-flight RPCs disconnects, the server cleans up its pending promises by routing reject/result callbacks back to the now-gone client; each hit the dead-peer check and raised "Target peer is not connected" — a storm (~100/2min on the busy pod) that churned the client into a reconnect loop at N≥2. The dead-peer check now fast-fails ONLY primary calls awaiting a response (which carry a `session`); fire-and-forget results/rejects/callbacks to a gone peer are dropped silently. With the cross-pod migration fix in 0.21.88 (#969), hypha-server is safe at N≥2. Tests: `tests/test_dead_peer_reject_storm.py`; the two-server suite now also simulates mid-RPC-disconnect churn (`tests/test_multi_replica_integration.py`).

### 0.21.88

 - Fix multi-replica cross-pod RPC false-reject (F6). With hypha-server at N≥2, the dead-peer check (`RedisRPCConnection.emit_message`, "Target peer is not connected") rejected RPCs to a peer that had re-pinned to a sibling pod: the `_recently_disconnected` cache is per-pod/in-memory (120s TTL), so a client moving pods (rollout / HPA / restart / `client_id`-hash re-pin) left the old pod wrongly rejecting it for the full TTL — dropping live RPCs. Now, on (re)connect, a pod broadcasts `client-reconnected` and every pod clears that client from its `_recently_disconnected` cache (`RedisEventBus.notify_client_connected`), wired into both the WebSocket and HTTP-streaming connect paths. Collapses the false-reject window from 120s to sub-second with no per-message cost; the dead-peer fast-fail for genuinely-gone peers is unchanged. This was the gap that forced the first N≥2 rollout back to N=1; multi-replica is safe again with this release. Tests: `tests/test_cross_pod_reconnect.py`, `tests/test_multi_replica_integration.py::test_cross_pod_repin_no_false_reject`.

### 0.21.87

 - Multi-replica safety — control-plane hardening (F6 Phase 1). Hypha's data plane (Redis event-bus routing, service discovery, reconnection tokens) was already horizontally scalable; this release makes the **control plane** safe to run with N≥2 `hypha-server` replicas sharing one Redis:
   - **Leader election** (`hypha/core/leader.py`): a best-effort Redis lease (`SET NX PX` + `WATCH`/`MULTI` renew) elects a single leader; `RedisStore.is_leader()` defaults to `True` for single-replica so behavior is unchanged.
   - **Autoscaling** is leader-gated — only the leader scales an app, preventing N replicas from independently over-scaling.
   - **Workspace activity-cleanup** and **app inactivity-stop** use short per-resource Redis locks so exactly one replica acts (a per-resource lock rather than a leader-gate, which would leak resources owned by a non-leader replica under sticky affinity).
   - **Reset guard**: `--reset-redis` / `HYPHA_RESET_REDIS=true` no longer flushes shared Redis if other servers are already registered — a restarting replica cannot wipe a live cluster.
 - Proven with real multi-replica integration tests (two live servers + one real Redis): leader election, reset-guard, and cross-replica RPC. (`tests/test_multi_replica_integration.py`.)
 - CI: fixed the `test.yml` Release job failing on every `main` push (conda env missing `pip`); hardened a flaky vector-collection test (embedding generation exceeding the default RPC timeout under CI load).

### 0.21.86

 - Graceful connection draining on server shutdown (F4). On a rollout/redeploy the old pod previously cut long-lived connections abruptly, so clients only detected the dead pod via their heartbeat/read timeout (10–20 s) and all reconnected in the same window — a thundering-herd reconnection storm. `RedisStore.teardown()` now proactively drains both transports: WebSocket clients receive an explicit `1001 GOING_AWAY` close frame, and HTTP-streaming `/rpc` clients receive the `None` stream-close sentinel (clean EOF) so they reconnect immediately. Previously only WebSockets were closed (with a bare code 1000); the HTTP-streaming transport had no graceful drain at all.
 - Bump bundled `hypha-rpc` to 0.21.42 (from 0.21.38). Includes the getService/`wm` proxy retarget to the new `manager_id` after reconnection (avoids a spurious 400 on getService following a server restart) and an event-loop initialization fix.

### 0.21.85

 - Fix `/zip-files/` perf cliff on very large ZIP64 archives (236 GB / 3.6M entries observed at 28–55 s per request, even on cache hits). The Redis cache stored the raw central-directory bytes but `zipfile.ZipFile()` re-parsed all 3.6M CDH entries on every request (~20–30 s). Adds an in-process `{filename: ZipInfo}` memo keyed on `(bucket, key, content_length)` so sequential requests on the same replica share a single parse; subsequent requests now resolve in milliseconds. Bumps the Redis CD-bytes cache TTL from 60 s → 1 h (committed artifacts are immutable). Drops the eager full-CD validation parse in `fetch_zip_tail` that doubled the cost on cache misses.

### 0.21.84

 - Fix `/zip-files/` endpoint failing with "Corrupt zip64 end of central directory locator" on ZIP64 archives (>4 GB or >65535 entries). `fetch_zip_tail` now parses the ZIP64 EOCD locator/record to determine the central directory's actual offset and size, ensures the fetched tail covers the entire central directory, and exposes a sparse file-like view (`_SparseZipReader`) that lets `zipfile.ZipFile` resolve absolute offsets in EOCD64/CD records correctly.

### 0.21.80

 - Add connection admission control to prevent reconnection storms from overwhelming the server (configurable via `HYPHA_MAX_CONCURRENT_CONNECTIONS`, default 10)
 - Reconnecting clients receive random jitter (0–1s) to spread reconnection load after server restarts

### 0.21.79

 - Add per-client WebSocket rate limiting to prevent RPC message spam (configurable via `HYPHA_WS_MSG_RATE_LIMIT`, `HYPHA_WS_MSG_BURST_LIMIT`)
 - Add per-IP HTTP rate limiting middleware for all HTTP endpoints (configurable via `HYPHA_HTTP_RATE_LIMIT`, `HYPHA_HTTP_BURST_LIMIT`)
 - Add per-client service registration quota (configurable via `HYPHA_MAX_SERVICES_PER_CLIENT`, default 1000)
 - Add per-user workspace creation quota (configurable via `HYPHA_MAX_WORKSPACES_PER_USER`, default 100)
 - Health check endpoints (`/health/*`) are exempt from HTTP rate limiting

### 0.20.55

 - Add `get_secret` and `set_secret` to the artifact manager to allow storing and retrieving secret values in the artifact.
 - Add `set_env` and `get_env` to workspace to allow setting and retrieving environment variables in the workspace.
 - Provide environment variables in the web ui.

### 0.20.41

 - Add `stop_after_inactive` option for server apps to stop the server app after a period of inactivity.
 - Support launching server apps worker in a separate service

### 0.20.40
 
 - Add vector store service to support vector search and retrieval.
 - Fix zenodo file upload issue
 - Speed up server by removing the `asyncio.sleep(0.01)` throttling and support concurrent handling of events in the redis event bus.

### 0.20.39

 - Revise artifact manager to use artifact id as the primary key, remove `prefix` based keys.
 - Support versioning and custom config (e.g. artifact specific s3 credentials) for the artifact manager.
 - Use SQLModel and support database migration using `alembic`.

### 0.20.38

 - Support event logging in the workspace, use `log_event` to log events in the workspace and use `get_events` to get the events in the workspace. The events will be persists in the SQL database.
 - Allow passing workspace and expires_in to the `login` function to generate workspace specific token.
 - When using http endpoint to access the service, you can now pass workspace specific token to the http header `Authorization` to access the service. (Previously, all the services are assumed to be accessed from the same service provider workspace)
 - Breaking Change: Remove `info`, `warning`, `error`, `critical`, `debug` from the `hypha` module, use `log` or `log_event` instead.
 - Support basic observability for the workspace, including workspace status, event bus and websocket connection status.
 - Support download statistics for the artifacts in the artifact manager.
 - Change http endpoint from `/{workspace}/artifact/{artifact_id}` to `/{workspace}/artifacts/{artifact_id}` to make it consistent with the other endpoints.

### 0.20.37
 - Add s3-proxy to allow accessing s3 presigned url in case the s3 server is not directly accessible. Use `--enable-s3-proxy` to enable the s3 proxy when starting Hypha.
 - Add `artifact-manager` service to provide comprehensive artifact management, used for creating gallery-like service portal. The artifact manager service is backed by s3 storage and supports presigned url for direct access to the artifacts. This is a replacement of the previous `card` service.

### 0.20.36

 - Upgrade hypha-rpc to support updating reconnection token (otherwise it generate token expired error after some time)

### 0.20.35

 - Upgrade hypha-rpc to fix reset timer

### 0.20.34
 
 - Fix persistent workspace unloaded issue when s3 is not available.
 - Improve ASGI support for streaming response.

### 0.20.33

 - Add `delete_workspace` to the workspace api.
 - Add workspaces panel to the web ui.

### 0.20.31

 - Upgrade hypha-rpc to fix ssl issue with the hypha-rpc client.

### 0.20.30

 - Fix server crashing bug when websocket.send is called after the connection is closed.

### 0.20.20

 - Fix static files not included in the package

### 0.20.19

 - Support invoke token
 - Add basic web ui for the workspace
 - BREAKING Change: Change the signature, now you need to pass a dictionary as options for `get_service`, `get_service_info`, `register_service` etc.

### 0.20.15

 - Add `revoke_token` to the workspace api.
 - Simplify http endpoints to a fixed pattern such as "{workspace}/services/*" and "{workspace}/apps/*".
 - To avoid naming convension, workspace names now must contain at least one hyphens, and only lowercase letters, numbers and hyphens are allowed.

### 0.20.14

 - Make `get_service` more restricted to support only service id string, see [migration guide](./docs/migration-guide.md) for more details.
 - Clean up http endpoints for the services.
 - Remove local cache of the server apps, we now always use s3 as the primary storage.

### 0.20.12

 - New Feature: In order to support large language models' function calling feature, hypha support built-in type annotation. With `hypha-rpc>=0.20.12`, we also support type annotation for the service functions in JSON Schema format. In Python, you can use `Pydantic` or simple python type hint, or directly write the json schema for Javascript service functions. This allows you to specify the inputs spec for functions.
 - Add type support for the `hypha` module. It allows you to register a type in the workspace using `register_service_type`, `get_service_type`, `list_service_types`. When registering a new service, you can specify the type and enable type check for the service. The type check will be performed when calling the service function. The type check is only available in Python.
 - Fix reconnecton issue in the client.
 - Support case conversion, which allows converting the service functions to snake_case or camelCase in `get_service` (Python) or `getService` (JavaScript).
 - **Breaking Changes**: In Python, all the function names uses snake case, and in JavaScript, all the function names uses camel case. For example, you should call `server.getService` instead of `server.get_service` in JavaScript, and `server.get_service` instead of `server.getService` in Python.
 - **Breaking Changes**: The new version of Hypha (0.20.0+) improves the RPC connection to make it more stable and secure, most importantly it supports automatic reconnection when the connection is lost. This also means breaking changes to the previous version. In the new version you will need a new library called `hypha-rpc` (instead of the hypha submodule in the `imjoy-rpc` module) to connect to the server.

