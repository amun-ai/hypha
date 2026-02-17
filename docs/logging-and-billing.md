# Logging, Interception, and Billing

This guide describes the planned and recommended design for Hypha logging extensibility, interception, and billing hooks.

Audience:
- application developers building on Hypha
- platform operators deploying/configuring Hypha for customers/teams

Scope:
- Extend logging to support application-defined events.
- Add limited system-level default logging for baseline observability.
- Add interception policies (`allow`, `stop`, `recover`).
- Add billing usage hooks and summaries.

Non-goals:
- Full monitoring dashboards.
- Alerting and incident management products.
- Full rating/invoicing engine.

## Event Model

Use a single event pipeline for system, application, audit, and billing events.

Recommended event fields:
- `event_type`: Canonical event name (for example `app.task.failed`, `billing.tokens.used`).
- `category`: One of `system`, `application`, `billing`, `audit`.
- `level`: Severity or intent (`debug`, `info`, `warning`, `error`).
- `workspace`
- `user_id`
- `app_id` (optional)
- `session_id` (optional)
- `timestamp`
- `data` (JSON payload)
- `idempotency_key` (required for billing events)

## Interception

Interception is a server-side policy layer in the event pipeline:

1. Validate event.
2. Persist event.
3. Evaluate matching interceptors in deterministic order.
4. Apply action (`allow`, `stop`, `recover`).

### Why interception exists

It allows workspace/application owners to enforce runtime rules without duplicating guard logic in every service function.

Typical use cases:
- Block specific operations when policy conditions are met.
- Stop risky or invalid calls.
- Trigger recovery handlers/fallback logic.

### Interceptor structure (recommended)

- `id`
- `name`
- `enabled`
- `scope`:
  - `workspace` (required)
  - `app_id` (optional)
  - `session_id` (optional)
- `event_selector` (event type/category match)
- `condition` (simple operators such as `eq`, `ne`, `gt`, `lt`, `contains`)
- `action` (`allow`, `stop`, `recover`)
- `priority` (lower number runs first)
- `recovery_config` (only for `recover`)

### Action semantics

- `allow`: Continue request flow.
- `stop`: Stop only the current call.
- `recover`: Invoke configured recovery behavior and return controlled outcome.

Confirmed behavior:
- `stop` must not auto-stop app sessions.

### Stop reasons, codes, and status mapping

When action is `stop`, return structured, reason-specific errors.

Recommended error payload fields:
- `code` (machine-readable)
- `message` (human-readable)
- `reason` (policy reason identifier)
- `interceptor_id`
- `interceptor_name`
- `details` (optional safe metadata)

Recommended code/status mapping for HTTP entry points:
- Policy/business rule stop: `INTERCEPT_STOP_POLICY` -> `422` (or `409` when conflict semantics are clearer).
- Permission/security stop: `INTERCEPT_STOP_SECURITY` -> `403`.
- Quota/rate stop: `INTERCEPT_STOP_LIMIT` -> `429`.

For RPC calls, return the same structured error payload through a typed RPC exception so clients can branch on `error.code` instead of parsing text.

## Billing Hooks

Billing hooks should be implemented as specialized logging events.

Recommended APIs:
- `record_usage(...)`
- `get_usage_summary(...)`

Recommended billing fields:
- `billing_point`
- `amount`
- `unit`
- `idempotency_key`
- optional dimensions (for example `model`, `region`, `operation`)

## Current Billing Capabilities

Current billing foundation is prepaid-first, with forward compatibility for postpaid and hybrid models.

What you can rely on:
- Payer model supports workspace/org or app-owner account style billing.
- A single payer account can be associated with multiple workspaces.
- Subscription plans can express:
  - access permissions
  - usage allowances
  - feature entitlements.
- Billable usage dimensions currently targeted:
  - tokens
  - requests
  - storage
  - runtime.
- Allowance exhaustion behavior is a hard stop (no grace period), typically surfaced as HTTP `429` for quota/limit exhaustion.

### Top-ups/Credits

Top-up credits are not in initial scope, but API/data contracts should keep extension points for future credit buckets and hybrid charging behavior.

## Stripe Integration

For deployments using Stripe, the initial integration scope includes:
- customer creation/mapping
- subscription lifecycle
- invoices
- payment intents
- webhook ingestion.

Recommended operating model:
- Treat Stripe as authoritative for payment/subscription financial lifecycle state.
- Keep mirrored Stripe identifiers/status snapshots in Hypha for runtime enforcement and traceability.
- Keep Hypha as source of truth for usage/metering ledger, entitlement checks, and payer/workspace mapping.

Reliability target for this foundation:
- webhook idempotency and replay handling
- lightweight sync/consistency checks
- not finance-grade accounting or full reconciliation workflow.

## Idempotency for billing events

`idempotency_key` prevents double charging during retries, network failures, and duplicate submissions.

Rules:
- Same logical billing event must reuse the same key.
- Different billing event must use a different key.
- Enforce uniqueness at storage level (for example unique `(workspace, idempotency_key)` index/constraint).

Expected behavior:
- First submission with new key: persisted and counted.
- Retry with same key: treated as duplicate and not double-counted.

## Storage and deployment notes

- Use PostgreSQL (including GCP Cloud SQL for PostgreSQL).
- Add indexes for query and billing usage paths:
  - `(workspace, timestamp)`
  - `(workspace, category, timestamp)`
  - `(workspace, event_type, timestamp)`
  - unique `(workspace, idempotency_key)` for billing dedupe.
- Retention should be configurable per customer/tier:
  - recommended default minimum for billable usage ledger: 13 months
  - recommended default for non-billing operational events: 90 days
  - higher tiers can offer longer retention.
- Audit/export/reporting should be available for both customer-facing and internal operations.

## Testing strategy

Use three layers:

1. Unit tests:
- condition evaluation and action mapping.
- payload validation and idempotency key enforcement.

2. Integration tests:
- API behavior, permission checks, workspace isolation, dedupe behavior.
- stop reason code and status/exception mapping verification.

3. Live/E2E tests:
- Hypha + Redis + PostgreSQL stack.
- retry and duplicate simulation under real services.
