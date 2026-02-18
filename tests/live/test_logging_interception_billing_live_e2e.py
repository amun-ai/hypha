"""Live E2E scenarios for logging/interception/billing against a running stack."""

import asyncio
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
import requests
from hypha_rpc import connect_to_server

pytestmark = [pytest.mark.asyncio, pytest.mark.live_e2e]


def _enabled() -> bool:
    return os.environ.get("HYPHA_LIVE_E2E", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


if not _enabled():
    pytest.skip(
        "Live E2E disabled. Set HYPHA_LIVE_E2E=1 to run these tests.",
        allow_module_level=True,
    )


SERVER_URL = os.environ.get("HYPHA_LIVE_E2E_SERVER_URL", "http://127.0.0.1:9527")
ROOT_TOKEN = os.environ.get("HYPHA_LIVE_E2E_ROOT_TOKEN")
WEBHOOK_SECRET = os.environ.get(
    "HYPHA_LIVE_E2E_WEBHOOK_SECRET",
    "whsec_contract_fixture_secret",
)
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "stripe"
WEBHOOK_PATHS = [
    "/api/v1/billing/stripe/webhook",
    "/api/billing/stripe/webhook",
    "/billing/stripe/webhook",
    "/stripe/webhook",
]

if not ROOT_TOKEN:
    pytest.skip(
        "Missing HYPHA_LIVE_E2E_ROOT_TOKEN for live E2E scenarios.",
        allow_module_level=True,
    )


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


def _utc_now():
    return datetime.now(timezone.utc)


def _window():
    now = _utc_now()
    start = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    end = (now + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    return start, end


def _stripe_signature(payload: str, secret: str, timestamp: int) -> str:
    signed = f"{timestamp}.{payload}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def _extract_error_code(exc: Exception):
    for attr in ("code", "error_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            return value

    payload = getattr(exc, "error", None)
    if isinstance(payload, dict) and isinstance(payload.get("code"), str):
        return payload["code"]

    text = str(exc)
    known_codes = (
        "INVALID_ARGUMENT",
        "UNAUTHORIZED",
        "FORBIDDEN",
        "NOT_FOUND",
        "CONFLICT",
        "UNPROCESSABLE_ENTITY",
        "TOO_MANY_REQUESTS",
        "INTERCEPT_STOP_POLICY",
        "INTERCEPT_STOP_SECURITY",
        "INTERCEPT_STOP_LIMIT",
        "IDEMPOTENCY_CONFLICT",
        "INTERNAL_ERROR",
    )
    for code in known_codes:
        if code in text:
            return code
    return None


def _extract_rows(summary):
    if isinstance(summary, dict):
        rows = summary.get("rows")
        if isinstance(rows, list):
            return rows
        data = summary.get("data")
        if isinstance(data, list):
            return data
    if isinstance(summary, list):
        return summary
    return []


async def _connect_client(*, token=None, workspace=None, client_id=None):
    config = {
        "server_url": SERVER_URL,
        "name": client_id or "logging-billing-live-e2e-client",
    }
    if token:
        config["token"] = token
    if workspace:
        config["workspace"] = workspace
    if client_id:
        config["client_id"] = client_id
    return await connect_to_server(config)


async def _create_workspace_client(root_api, *, prefix: str):
    workspace = _unique(prefix)
    await root_api.create_workspace(
        {
            "name": workspace,
            "description": f"Live E2E workspace: {prefix}",
        },
        overwrite=True,
    )
    token = await root_api.generate_token(
        {
            "workspace": workspace,
            "permission": "read_write",
            "expires_in": 3600,
        }
    )
    api = await _connect_client(
        token=token,
        workspace=workspace,
        client_id=f"{prefix}-client-{uuid4().hex[:6]}",
    )
    return workspace, api


def _post_signed_webhook(payload: dict):
    payload_text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    timestamp = int(_utc_now().timestamp())
    signature = _stripe_signature(payload_text, WEBHOOK_SECRET, timestamp=timestamp)
    headers = {
        "Content-Type": "application/json",
        "Stripe-Signature": signature,
    }
    for path in WEBHOOK_PATHS:
        response = requests.post(
            f"{SERVER_URL}{path}",
            data=payload_text,
            headers=headers,
            timeout=20,
        )
        if response.status_code not in (404, 405):
            return response, path
    pytest.fail("No Stripe webhook endpoint was found for live E2E execution.")


def _load_subscription_fixture():
    with open(FIXTURE_DIR / "subscription_updated.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload


@pytest_asyncio.fixture
async def root_api():
    api = await _connect_client(
        token=ROOT_TOKEN, client_id="logging-billing-live-e2e-root"
    )
    try:
        yield api
    finally:
        await api.disconnect()


async def test_e2e_duplicate_billing_retries_and_no_double_count(root_api):
    workspace, api = await _create_workspace_client(root_api, prefix="live-e2e-retry")
    try:
        billing_point = f"e2e.live.usage.retry.{workspace}"
        idempotency_key = _unique("retry-key")
        occurred_at = _utc_now().isoformat().replace("+00:00", "Z")
        payload = {
            "billing_point": billing_point,
            "amount": 11,
            "unit": "request",
            "idempotency_key": idempotency_key,
            "occurred_at": occurred_at,
            "dimensions": {"source": "live-e2e"},
        }

        try:
            await asyncio.wait_for(api.record_usage(payload), timeout=0.001)
        except asyncio.TimeoutError:
            pass

        retry_result = await api.record_usage(payload)
        duplicate_results = await asyncio.gather(
            *[api.record_usage(payload) for _ in range(5)]
        )

        all_results = [retry_result, *duplicate_results]
        usage_event_ids = {
            item.get("usage_event_id") for item in all_results if isinstance(item, dict)
        }
        assert len(usage_event_ids) == 1
        assert all(item.get("ok") is True for item in all_results)

        start_time, end_time = _window()
        summary = await api.get_usage_summary(
            {
                "start_time": start_time,
                "end_time": end_time,
                "group_by": ["billing_point"],
            }
        )
        rows = _extract_rows(summary)
        matched = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("billing_point") == billing_point
        ]
        assert len(matched) == 1
        assert matched[0]["amount"] == pytest.approx(11.0)
        assert matched[0]["count"] == 1
    finally:
        await api.disconnect()


async def test_e2e_interceptor_stop_and_recover_with_realistic_payload(root_api):
    _, api = await _create_workspace_client(
        root_api, prefix="live-e2e-interceptor"
    )
    interceptor_ids = []
    try:
        stop_event_type = _unique("e2e.live.stop.payment")
        recover_event_type = _unique("e2e.live.recover.job")
        recovery_log_type = _unique("e2e.live.recover.logged")

        stop_interceptor = await api.register_interceptor(
            {
                "name": "live-e2e-stop-security",
                "scope": {"workspace": api.config.workspace},
                "enabled": True,
                "priority": 10,
                "event_selector": {"event_types": [stop_event_type], "categories": ["application"]},
                "condition": {"field": "data.operation", "op": "eq", "value": "charge"},
                "action": {"type": "stop", "reason": "security"},
            }
        )
        interceptor_ids.append(stop_interceptor["id"])

        with pytest.raises(Exception) as stop_error:
            await api.log_event(
                stop_event_type,
                {
                    "operation": "charge",
                    "amount": 1999,
                    "currency": "usd",
                    "payment_method": "card",
                    "request_id": _unique("charge"),
                },
                category="application",
            )
        assert _extract_error_code(stop_error.value) == "INTERCEPT_STOP_SECURITY"

        await api.remove_interceptor(stop_interceptor["id"])
        interceptor_ids.clear()

        recover_interceptor = await api.register_interceptor(
            {
                "name": "live-e2e-recover-pipeline",
                "scope": {"workspace": api.config.workspace},
                "enabled": True,
                "priority": 20,
                "event_selector": {
                    "event_types": [recover_event_type],
                    "categories": ["application"],
                },
                "condition": {"field": "data.status", "op": "eq", "value": "failed"},
                "action": {
                    "type": "recover",
                    "reason": "policy",
                    "recovery": {"mode": "log_only", "event_type": recovery_log_type},
                },
            }
        )
        interceptor_ids.append(recover_interceptor["id"])

        recover_result = await api.log_event(
            recover_event_type,
            {
                "status": "failed",
                "component": "worker.queue",
                "attempt": 2,
                "error": "timeout while calling upstream dependency",
            },
            category="application",
        )
        assert recover_result["ok"] is True
        assert recover_result["action"] == "recover"
        assert recover_result.get("recovery_event_id")

        recovered_events = await api.get_events(event_type=recovery_log_type)
        recovered_ids = {
            item.get("id") for item in recovered_events if isinstance(item, dict)
        }
        assert recover_result["recovery_event_id"] in recovered_ids
    finally:
        for interceptor_id in interceptor_ids:
            try:
                await api.remove_interceptor(interceptor_id)
            except Exception:
                pass
        await api.disconnect()


async def test_e2e_usage_summary_correctness_on_persisted_events(root_api):
    _, api = await _create_workspace_client(root_api, prefix="live-e2e-summary")
    try:
        point_input = _unique("e2e.live.usage.input")
        point_output = _unique("e2e.live.usage.output")
        occurred_at = _utc_now().isoformat().replace("+00:00", "Z")
        events = [
            (point_input, 2),
            (point_input, 3),
            (point_output, 5),
        ]
        for billing_point, amount in events:
            await api.record_usage(
                {
                    "billing_point": billing_point,
                    "amount": amount,
                    "unit": "token",
                    "idempotency_key": _unique("summary-key"),
                    "occurred_at": occurred_at,
                }
            )

        start_time, end_time = _window()
        summary = await api.get_usage_summary(
            {
                "start_time": start_time,
                "end_time": end_time,
                "group_by": ["billing_point", "unit"],
            }
        )
        rows = _extract_rows(summary)
        amounts = {
            row["billing_point"]: row["amount"]
            for row in rows
            if isinstance(row, dict) and row.get("billing_point") in {point_input, point_output}
        }
        counts = {
            row["billing_point"]: row["count"]
            for row in rows
            if isinstance(row, dict) and row.get("billing_point") in {point_input, point_output}
        }

        assert amounts[point_input] == pytest.approx(5.0)
        assert amounts[point_output] == pytest.approx(5.0)
        assert counts[point_input] == 2
        assert counts[point_output] == 1
        assert summary["totals"]["amount"] >= 10.0
    finally:
        await api.disconnect()


async def test_e2e_allowance_exhaustion_returns_limit_error(root_api):
    _, api = await _create_workspace_client(root_api, prefix="live-e2e-limit")
    try:
        entitlement_key = _unique("e2e.live.entitlement")
        common = {
            "billing_point": _unique("e2e.live.usage.limit"),
            "unit": "request",
            "entitlement": {"key": entitlement_key, "limit": 5},
            "occurred_at": _utc_now().isoformat().replace("+00:00", "Z"),
        }

        first = await api.record_usage(
            {
                **common,
                "amount": 3,
                "idempotency_key": _unique("limit-key"),
            }
        )
        assert first["ok"] is True and first["deduped"] is False

        with pytest.raises(Exception) as exc_info:
            await api.record_usage(
                {
                    **common,
                    "amount": 3,
                    "idempotency_key": _unique("limit-key"),
                }
            )
        code = _extract_error_code(exc_info.value)
        assert code in {"TOO_MANY_REQUESTS", "INTERCEPT_STOP_LIMIT"}
    finally:
        await api.disconnect()


async def test_e2e_multi_workspace_with_one_payer_account(root_api):
    ws1, api_ws1 = await _create_workspace_client(root_api, prefix="live-e2e-payer-a")
    ws2, api_ws2 = await _create_workspace_client(root_api, prefix="live-e2e-payer-b")
    try:
        webhook_payload = _load_subscription_fixture()
        webhook_payload["id"] = _unique("evt-live-e2e")
        webhook_payload["created"] = int(_utc_now().timestamp())
        webhook_payload.setdefault("data", {}).setdefault("object", {})
        webhook_payload["data"]["object"]["customer"] = _unique("cus-live-e2e")
        webhook_payload["data"]["object"]["id"] = _unique("sub-live-e2e")

        webhook_response, path = _post_signed_webhook(webhook_payload)
        assert webhook_response.status_code in (200, 201, 202), (
            f"Expected webhook success on {path}, got {webhook_response.status_code}: "
            f"{webhook_response.text}"
        )
        body = webhook_response.json()
        billing_account_id = body.get("billing_account_id")
        assert isinstance(billing_account_id, str) and billing_account_id

        occurred_at = _utc_now().isoformat().replace("+00:00", "Z")
        billing_point = _unique("e2e.live.usage.payer")
        await api_ws1.record_usage(
            {
                "billing_point": billing_point,
                "amount": 2,
                "unit": "token",
                "idempotency_key": _unique("payer-key"),
                "billing_account_id": billing_account_id,
                "occurred_at": occurred_at,
            }
        )
        await api_ws2.record_usage(
            {
                "billing_point": billing_point,
                "amount": 5,
                "unit": "token",
                "idempotency_key": _unique("payer-key"),
                "billing_account_id": billing_account_id,
                "occurred_at": occurred_at,
            }
        )

        start_time, end_time = _window()
        query = {
            "start_time": start_time,
            "end_time": end_time,
            "group_by": ["workspace", "billing_point"],
            "billing_account_id": billing_account_id,
            "billing_point": billing_point,
        }
        summary_ws1 = await api_ws1.get_usage_summary(query)
        summary_ws2 = await api_ws2.get_usage_summary(query)

        rows_ws1 = _extract_rows(summary_ws1)
        rows_ws2 = _extract_rows(summary_ws2)
        assert len(rows_ws1) == 1 and len(rows_ws2) == 1
        assert rows_ws1[0]["workspace"] == ws1
        assert rows_ws2[0]["workspace"] == ws2
        assert rows_ws1[0]["amount"] == pytest.approx(2.0)
        assert rows_ws2[0]["amount"] == pytest.approx(5.0)
    finally:
        await api_ws1.disconnect()
        await api_ws2.disconnect()
