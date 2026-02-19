# Logging, Interception, and Billing

This guide covers Hypha's unified event pipeline for logging, interception, and billing hooks.

Scope:
- Extensible application events.
- Limited system defaults for baseline observability.
- Interceptor policies (`allow`, `stop`, `recover`).
- Billing usage hooks (`record_usage`, `get_usage_summary`).

Out of scope:
- Full monitoring dashboards.
- Alerting/incident management products.
- Full rating/invoicing engines.

## Event Schema and Categories

Hypha persists events with a shared schema:

- `event_type`: canonical event name (for example `system.app.started`, `billing.usage.recorded`)
- `category`: `system`, `application`, `billing`, `audit`
- `level`: `debug`, `info`, `warning`, `error`
- `workspace`
- `user_id`
- `app_id` (optional)
- `session_id` (optional)
- `timestamp`
- `data` (JSON payload)
- `idempotency_key` (required for billing usage writes)
- `intercepted` (`false` by default)
- `interceptor_action` (`stop|recover`, nullable)
- `interceptor_reason` (`policy|security|limit`, nullable)
- `interceptor_code` (`INTERCEPT_STOP_*`, nullable)
- `interceptor_id` (nullable)
- `interceptor_name` (nullable)
- `intercepted_at` (nullable UTC timestamp)

### System Default Events (Baseline)

Hypha emits baseline system events at key lifecycle points:

- Workspace lifecycle:
  - `system.workspace.created`
  - `system.workspace.deleted`
- Service lifecycle:
  - `system.service.registered`
  - `system.service.updated`
  - `system.service.removed`
- App lifecycle:
  - `system.app.started`
  - `system.app.stopped`
  - `system.app.failed`

These defaults are intentionally limited and are not intended to be a complete observability product.

## Interception API and Behavior

Interceptors are evaluated in deterministic order after event persistence.

1. Validate event.
2. Persist event.
3. Evaluate matching interceptors.
4. Apply action.

Actions:
- `allow`: continue normal flow.
- `stop`: stop the current call only.
- `recover`: run recovery behavior and return a controlled response.

Important behavior:
- `stop` does not auto-stop app sessions.
- when `stop` matches, the already-persisted source event is flagged in `event_logs`
  with `intercepted=true` and interceptor metadata.
- if stop-flag persistence fails, the caller gets `INTERNAL_ERROR` instead of a stop success.

Workspace APIs:
- `register_interceptor(...)`
- `list_interceptors(...)`
- `remove_interceptor(...)`

Minimal interceptor example:

```python
await api.register_interceptor({
    "name": "block-risky-op",
    "enabled": True,
    "scope": {"workspace": api.config.workspace},
    "event_selector": {"event_types": ["app.task.execute"]},
    "condition": {
        "field": "data.risk_level",
        "op": "eq",
        "value": "high",
    },
    "action": {"type": "stop", "reason": "policy"},
    "priority": 100,
})
```

### Stop Error Mapping

When action is `stop`, Hypha returns structured errors with stable codes.

Recommended HTTP mapping:
- `INTERCEPT_STOP_POLICY` -> `422` (or `409` when conflict semantics are clearer)
- `INTERCEPT_STOP_SECURITY` -> `403`
- `INTERCEPT_STOP_LIMIT` -> `429`

For RPC, clients should branch on `error.code` instead of parsing message text.

## Billing Hooks

Billing hooks are implemented as specialized event ingestion and aggregation APIs.

- `record_usage(usage)`
- `get_usage_summary(query)`

Usage payload fields:
- `billing_point`
- `amount`
- `unit`
- `idempotency_key` (required)
- optional dimensions (`model`, `region`, `operation`, etc.)

Example:

```python
await api.record_usage({
    "billing_point": "tokens.prompt",
    "amount": 1024,
    "unit": "tokens",
    "idempotency_key": "req-8f73f0a1",
    "app_id": "assistant-app",
    "session_id": "ws-user-abc/client-123",
    "dimensions": {"model": "gpt-4.1"},
})
```

Summary query example:

```python
summary = await api.get_usage_summary({
    "start_time": "2026-02-01T00:00:00Z",
    "end_time": "2026-02-29T23:59:59Z",
    "group_by": ["billing_point", "app_id"],
})
```

## Idempotency Semantics and Retry Pattern

`idempotency_key` prevents double counting when retries happen.

Rules:
- Reuse the same key for the same logical billing event.
- Use a different key for different logical events.

Expected behavior:
- First request with a new key is persisted.
- Retries with the same key return deduped behavior and do not double count.
- Aggregation excludes intercepted billing rows (`intercepted=true`) so blocked usage
  never contributes to billable totals.

Retry example (client-side):

```python
idempotency_key = f"usage:{request_id}"

for _ in range(3):
    try:
        result = await api.record_usage({
            "billing_point": "requests.api",
            "amount": 1,
            "unit": "request",
            "idempotency_key": idempotency_key,
        })
        break
    except TimeoutError:
        # Safe to retry with the same idempotency key.
        continue
```

## Billing Model Assumptions (Launch)

Launch assumptions:
- Prepaid-first billing model.
- Subscription plans carry access + allowance + feature entitlements.
- Hard stop on limit exhaustion (no grace period).
- HTTP quota/allowance exhaustion should surface as `429`.

Billable dimensions at launch:
- tokens
- requests
- storage
- runtime

## Payer Account Model

Hypha keeps an internal payer identity separate from Stripe customer identity.

- `billing_account_id`: internal payer/account ID used by Hypha for metering and enforcement.
- `stripe_customer_id`: mapped Stripe customer reference used for payment lifecycle synchronization.

One payer account can map to multiple workspaces.

## Stripe Ownership and Mirroring Model

- Stripe is the source of truth for payment/subscription financial lifecycle state.
- Hypha is the source of truth for metering ledger, entitlement enforcement, and workspace/payer mapping.
- Hypha mirrors Stripe identifiers and key status snapshots for runtime enforcement and traceability.

V1 Stripe integration scope:
- customer mapping
- subscription lifecycle
- invoices
- payment intents
- webhook ingestion with idempotent replay handling

## Retention, Reporting, and Export

Retention is configurable per customer/tier.

Recommended baselines:
- billable usage ledger: 13 months
- non-billing operational events: 90 days

Reporting/export should be available for both customer-facing and internal operations.

## PostgreSQL / Cloud SQL Storage and Migrations

Hypha persists event and billing data in PostgreSQL (including GCP Cloud SQL for PostgreSQL).

Key index patterns:
- `(workspace, timestamp)`
- `(workspace, category, timestamp)`
- `(workspace, event_type, timestamp)`
- billing idempotency dedupe index for `(workspace, idempotency_key)` scoped to billing events

Migration workflow:

```bash
alembic upgrade head
```

For schema changes, add Alembic revisions under `hypha/migrations/versions/` and validate upgrade paths in development/integration environments.

## Local Live/E2E Execution

Phase 6 local execution is fully automatable and does not require manual UI clicking.

One-command run:

```bash
./scripts/run_logging_billing_live_e2e.sh
```

What this command does:
- starts Postgres + Redis with Docker Compose
- starts Hypha from local source on the host (default mode) against those real services
- waits for readiness
- runs live E2E scenarios in `tests/live/test_logging_interception_billing_live_e2e.py`
- tears down the stack unless `HYPHA_LIVE_E2E_KEEP_STACK=1` is set.

Phase 6 scenarios covered:
- duplicate billing submissions with retry-like behavior and dedupe/no double-billing verification
- interceptor stop and recover with realistic payloads
- usage summary correctness against persisted events
- allowance exhaustion limit enforcement (`TOO_MANY_REQUESTS` / limit-stop semantics)
- multi-workspace usage using one payer account (`billing_account_id`) created via Stripe webhook path.

Useful environment overrides:
- `HYPHA_LIVE_E2E_SERVER_URL` (default `http://127.0.0.1:9527`)
- `HYPHA_LIVE_E2E_ROOT_TOKEN` (default `09zDo-WV2_ZLwVfTA9Gj-pGKs2X403nio-StS2e-JihUBAiPW3hXsQ`)
- `HYPHA_LIVE_E2E_WEBHOOK_SECRET` (default `whsec_contract_fixture_secret`)
- `HYPHA_LIVE_E2E_HYPHA_MODE` (`host` by default, optional `compose`)
- `HYPHA_LIVE_E2E_KEEP_STACK=1` to keep services running after tests.
