"""Billing and Stripe webhook integration endpoints."""

import hashlib
import hmac
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hypha.core.workspace import BillingContractError

logger = logging.getLogger("billing")

_STRIPE_WEBHOOK_PATHS = [
    "/api/v1/billing/stripe/webhook",
    "/api/billing/stripe/webhook",
    "/billing/stripe/webhook",
    "/stripe/webhook",
]

_ERROR_STATUS_BY_CODE = {
    "INVALID_ARGUMENT": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "CONFLICT": 409,
    "IDEMPOTENCY_CONFLICT": 409,
    "UNPROCESSABLE_ENTITY": 422,
    "INTERCEPT_STOP_POLICY": 422,
    "INTERCEPT_STOP_SECURITY": 403,
    "INTERCEPT_STOP_LIMIT": 429,
    "TOO_MANY_REQUESTS": 429,
    "INTERNAL_ERROR": 500,
}


def _parse_stripe_signature(signature_header: str) -> tuple[Optional[str], list[str]]:
    timestamp = None
    signatures = []
    for part in (signature_header or "").split(","):
        key, sep, value = part.strip().partition("=")
        if not sep:
            continue
        if key == "t":
            timestamp = value
        elif key == "v1":
            signatures.append(value)
    return timestamp, signatures


def _verify_stripe_signature(
    payload: bytes, signature_header: str, secret: str
) -> bool:
    timestamp, signatures = _parse_stripe_signature(signature_header)
    if not timestamp or not signatures:
        return False
    try:
        payload_text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    signed_payload = f"{timestamp}.{payload_text}".encode("utf-8")
    digest = hmac.new(
        secret.encode("utf-8"), signed_payload, hashlib.sha256
    ).hexdigest()
    return any(hmac.compare_digest(digest, sig) for sig in signatures)


class BillingWebhookController:
    """Register Stripe webhook routes and map them into workspace billing logic."""

    def __init__(self, store):
        self._store = store
        self._secret = os.environ.get(
            "HYPHA_STRIPE_WEBHOOK_SECRET",
            "whsec_contract_fixture_secret",
        )
        router = APIRouter()

        async def stripe_webhook(request: Request):
            signature = request.headers.get("Stripe-Signature", "")
            raw_body = await request.body()
            if not _verify_stripe_signature(raw_body, signature, self._secret):
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": {
                            "code": "UNAUTHORIZED",
                            "message": "Invalid Stripe webhook signature",
                        },
                    },
                )
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": {
                            "code": "INVALID_ARGUMENT",
                            "message": "Webhook body must be valid JSON",
                        },
                    },
                )

            wm = self._store._workspace_manager
            if wm is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "ok": False,
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "Workspace manager is not initialized",
                        },
                    },
                )
            try:
                result = await wm.process_stripe_webhook_event(payload)
                return JSONResponse(status_code=200, content=result)
            except BillingContractError as exp:
                code = getattr(exp, "code", "INTERNAL_ERROR")
                status = _ERROR_STATUS_BY_CODE.get(code, 500)
                return JSONResponse(
                    status_code=status,
                    content={
                        "ok": False,
                        "error": {
                            "code": code,
                            "message": str(exp),
                            "details": getattr(exp, "details", {}),
                        },
                    },
                )
            except Exception as exp:
                logger.exception("Unhandled Stripe webhook processing error")
                return JSONResponse(
                    status_code=500,
                    content={
                        "ok": False,
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": str(exp),
                        },
                    },
                )

        for webhook_path in _STRIPE_WEBHOOK_PATHS:
            router.add_api_route(webhook_path, stripe_webhook, methods=["POST"])

        store.register_router(router)
