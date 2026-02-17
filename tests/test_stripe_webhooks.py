"""Contract-first tests for Stripe webhook ingestion behavior."""

from pathlib import Path
import hashlib
import hmac
import json

import pytest
import requests

from . import SERVER_URL

pytestmark = [pytest.mark.logging_billing]

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stripe"
WEBHOOK_SECRET = "whsec_contract_fixture_secret"
WEBHOOK_PATH_CANDIDATES = [
    "/api/v1/billing/stripe/webhook",
    "/api/billing/stripe/webhook",
    "/billing/stripe/webhook",
    "/stripe/webhook",
]


def _load_fixture_payload(filename: str) -> str:
    with open(FIXTURE_DIR / filename, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _stripe_signature(payload: str, secret: str, timestamp: int = 1760001000) -> str:
    signed = f"{timestamp}.{payload}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def _post_webhook(payload: str, signature: str):
    headers = {
        "Content-Type": "application/json",
        "Stripe-Signature": signature,
    }
    for path in WEBHOOK_PATH_CANDIDATES:
        response = requests.post(
            f"{SERVER_URL}{path}",
            data=payload,
            headers=headers,
            timeout=10,
        )
        if response.status_code not in (404, 405):
            return response, path

    pytest.xfail(
        "No Stripe webhook endpoint found yet at known contract paths "
        "(Phase 4 integration pending)"
    )


def _json_or_none(response):
    try:
        return response.json()
    except Exception:
        return None


def test_stripe_webhook_rejects_invalid_signature_contract(fastapi_server):
    """Contract: webhook signature must be validated and bad signatures rejected."""
    payload = _load_fixture_payload("invoice_payment_succeeded.json")
    invalid_signature = _stripe_signature(payload, "whsec_wrong_secret")

    response, path = _post_webhook(payload, invalid_signature)

    assert response.status_code in (400, 401, 403, 422), (
        f"Expected invalid signature rejection on {path}, got {response.status_code}: "
        f"{response.text}"
    )


def test_stripe_webhook_duplicate_delivery_contract(fastapi_server):
    """Contract: duplicate delivery/replay is idempotent and deterministic."""
    payload = _load_fixture_payload("invoice_payment_succeeded.json")
    signature = _stripe_signature(payload, WEBHOOK_SECRET)

    first, path = _post_webhook(payload, signature)
    second, _ = _post_webhook(payload, signature)

    assert first.status_code in (200, 201, 202), (
        f"Expected first webhook delivery acceptance on {path}, got "
        f"{first.status_code}: {first.text}"
    )
    assert second.status_code in (200, 201, 202, 409), (
        f"Expected deterministic duplicate handling on {path}, got "
        f"{second.status_code}: {second.text}"
    )

    second_payload = _json_or_none(second)
    if isinstance(second_payload, dict) and "deduped" in second_payload:
        assert second_payload["deduped"] is True

    if second.status_code == 409 and isinstance(second_payload, dict):
        serialized = json.dumps(second_payload).lower()
        assert "idempot" in serialized or "conflict" in serialized


def test_subscription_snapshot_contract_xfail_until_exposed(fastapi_server):
    """Contract skeleton: subscription updates should refresh mirrored snapshot state."""
    payload = _load_fixture_payload("subscription_updated.json")
    signature = _stripe_signature(payload, WEBHOOK_SECRET)

    response, path = _post_webhook(payload, signature)

    if response.status_code not in (200, 201, 202):
        pytest.fail(
            f"Expected webhook acceptance on {path}, got {response.status_code}: {response.text}"
        )

    body = _json_or_none(response)
    if body is None:
        pytest.xfail(
            "Webhook accepted but response does not expose mirrored snapshot metadata yet"
        )

    serialized = json.dumps(body).lower()
    if not any(
        token in serialized
        for token in ("subscription", "snapshot", "entitlement", "status")
    ):
        pytest.xfail(
            "Subscription snapshot reflection contract not surfaced in response yet"
        )
