"""Contract-first tests for billing hook APIs."""

import pytest
from hypha_rpc import connect_to_server

from . import SERVER_URL

pytestmark = [pytest.mark.asyncio, pytest.mark.logging_billing]


def _is_missing_method_error(exc: Exception, method_name: str) -> bool:
    text = str(exc).lower()
    method = method_name.lower()
    normalized = text.strip().strip("'\"")
    if normalized == method:
        return True
    return (
        ("method" in text and "not found" in text and method in text)
        or ("no attribute" in text and method in text)
        or ("attributeerror" in text and method in text)
        or ("unknown method" in text and method in text)
        or (method in text and "remote" in text and "service" in text)
    )


def _extract_error_code(exc: Exception):
    for attr in ("code", "error_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            return value

    payload = getattr(exc, "error", None)
    if isinstance(payload, dict) and isinstance(payload.get("code"), str):
        return payload["code"]

    text = str(exc)
    for code in (
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
    ):
        if code in text:
            return code
    return None


async def _connect_client(*, token=None, workspace=None, client_id=None):
    config = {
        "server_url": SERVER_URL,
        "name": client_id or "billing-contract-client",
    }
    if client_id:
        config["client_id"] = client_id
    if token:
        config["token"] = token
    if workspace:
        config["workspace"] = workspace
    return await connect_to_server(config)


def _usage_payload(*, key: str, point: str, amount: float, unit: str = "token"):
    return {
        "billing_point": point,
        "amount": amount,
        "unit": unit,
        "idempotency_key": key,
        "dimensions": {
            "model": "contract-model-v1",
            "region": "test-us",
        },
        "occurred_at": "2026-01-02T03:04:05Z",
    }


async def _record_or_xfail(api, payload):
    try:
        return await api.record_usage(payload)
    except Exception as exc:
        if _is_missing_method_error(exc, "record_usage"):
            pytest.xfail("record_usage API is not implemented yet")
        raise


async def _summary_or_xfail(api, query):
    try:
        return await api.get_usage_summary(query)
    except Exception as exc:
        if _is_missing_method_error(exc, "get_usage_summary"):
            pytest.xfail("get_usage_summary API is not implemented yet")
        raise


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
    return None


def _row_amount(row):
    for key in ("amount", "total_amount", "sum_amount", "value"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _sum_amount_for_billing_point(rows, billing_point):
    total = 0.0
    matched = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("billing_point") != billing_point:
            continue
        amount = _row_amount(row)
        if amount is None:
            continue
        total += amount
        matched += 1
    return matched, total


async def test_record_usage_validation_contract(fastapi_server, test_user_token):
    """Contract: idempotency_key required and amount must be strictly positive."""
    api = await _connect_client(
        token=test_user_token, client_id="contract-billing-validation"
    )

    try:
        missing_key = {
            "billing_point": "contract.billing.validation",
            "amount": 1,
            "unit": "token",
        }
        with pytest.raises(Exception) as exc_missing_key:
            await _record_or_xfail(api, missing_key)

        code_missing = _extract_error_code(exc_missing_key.value)
        message_missing = str(exc_missing_key.value).lower()
        assert (
            code_missing == "INVALID_ARGUMENT"
            or "idempotency" in message_missing
            or code_missing == "UNPROCESSABLE_ENTITY"
        )

        non_positive_amount = _usage_payload(
            key="contract-billing-validation-amount",
            point="contract.billing.validation",
            amount=0,
        )
        with pytest.raises(Exception) as exc_non_positive:
            await _record_or_xfail(api, non_positive_amount)

        code_amount = _extract_error_code(exc_non_positive.value)
        message_amount = str(exc_non_positive.value).lower()
        assert (
            code_amount == "INVALID_ARGUMENT"
            or "amount" in message_amount
            or code_amount == "UNPROCESSABLE_ENTITY"
        )
    finally:
        await api.disconnect()


async def test_get_usage_summary_requires_time_window_contract(
    fastapi_server,
    test_user_token,
):
    """Contract: get_usage_summary requires both start_time and end_time."""
    api = await _connect_client(
        token=test_user_token, client_id="contract-billing-summary-window"
    )

    try:
        with pytest.raises(Exception) as exc_info:
            await _summary_or_xfail(api, {"group_by": ["billing_point"]})

        code = _extract_error_code(exc_info.value)
        msg = str(exc_info.value).lower()
        assert (
            code == "INVALID_ARGUMENT"
            or code == "UNPROCESSABLE_ENTITY"
            or "start_time" in msg
            or "end_time" in msg
        )
    finally:
        await api.disconnect()


async def test_record_usage_idempotency_replay_contract(
    fastapi_server, test_user_token
):
    """Contract: duplicate usage submissions must not double-count."""
    api = await _connect_client(
        token=test_user_token, client_id="contract-billing-replay"
    )

    try:
        billing_point = "contract.billing.replay"
        payload = _usage_payload(
            key="contract-billing-replay-key",
            point=billing_point,
            amount=11,
        )

        first = await _record_or_xfail(api, payload)
        second = await _record_or_xfail(api, payload)

        if not isinstance(first, dict) or not isinstance(second, dict):
            pytest.xfail("record_usage response envelope is not implemented yet")

        if "deduped" not in first or "deduped" not in second:
            pytest.xfail("record_usage deduped response field is not implemented yet")

        assert first["deduped"] is False
        assert second["deduped"] is True

        first_id = first.get("usage_event_id")
        second_id = second.get("usage_event_id")
        if first_id and second_id:
            assert first_id == second_id

        summary = await _summary_or_xfail(
            api,
            {
                "start_time": "2026-01-01T00:00:00Z",
                "end_time": "2026-12-31T23:59:59Z",
                "group_by": ["billing_point"],
            },
        )

        rows = _extract_rows(summary)
        if not rows:
            pytest.xfail("get_usage_summary row contract is not implemented yet")

        matched, total = _sum_amount_for_billing_point(rows, billing_point)
        if matched == 0:
            pytest.xfail(
                "get_usage_summary billing_point grouping is not implemented yet"
            )
        assert total == pytest.approx(11.0)
    finally:
        await api.disconnect()


async def test_record_usage_idempotency_conflict_contract(
    fastapi_server,
    test_user_token,
):
    """Contract: same key with different payload must raise IDEMPOTENCY_CONFLICT."""
    api = await _connect_client(
        token=test_user_token, client_id="contract-billing-conflict"
    )

    try:
        key = "contract-billing-conflict-key"
        first = _usage_payload(
            key=key,
            point="contract.billing.conflict",
            amount=5,
        )
        second = _usage_payload(
            key=key,
            point="contract.billing.conflict",
            amount=9,
        )

        await _record_or_xfail(api, first)

        with pytest.raises(Exception) as exc_info:
            await _record_or_xfail(api, second)

        code = _extract_error_code(exc_info.value)
        if code is None and "idempotency" in str(exc_info.value).lower():
            pytest.xfail("IDEMPOTENCY_CONFLICT code mapping not implemented yet")
        assert code == "IDEMPOTENCY_CONFLICT" or "IDEMPOTENCY_CONFLICT" in str(
            exc_info.value
        )
    finally:
        await api.disconnect()


async def test_usage_summary_workspace_isolation_contract(
    fastapi_server,
    test_user_token,
):
    """Contract: usage summaries are isolated by workspace context."""
    api_ws1 = await _connect_client(
        token=test_user_token, client_id="contract-billing-ws1"
    )
    api_ws2 = None

    try:
        ws1 = api_ws1.config.workspace
        ws2_name = "contract-billing-ws2"

        await api_ws1.create_workspace(
            {
                "name": ws2_name,
                "description": "Workspace for billing summary isolation contract",
            },
            overwrite=True,
        )
        ws2_token = await api_ws1.generate_token(
            {
                "workspace": ws2_name,
                "permission": "read_write",
            }
        )
        api_ws2 = await _connect_client(
            token=ws2_token,
            workspace=ws2_name,
            client_id="contract-billing-ws2-client",
        )

        await _record_or_xfail(
            api_ws1,
            _usage_payload(
                key="contract-billing-ws1-key",
                point="contract.billing.workspace",
                amount=3,
            ),
        )
        await _record_or_xfail(
            api_ws2,
            _usage_payload(
                key="contract-billing-ws2-key",
                point="contract.billing.workspace",
                amount=7,
            ),
        )

        query = {
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-12-31T23:59:59Z",
            "group_by": ["workspace", "billing_point"],
        }
        summary_ws1 = await _summary_or_xfail(api_ws1, query)
        summary_ws2 = await _summary_or_xfail(api_ws2, query)

        rows_ws1 = _extract_rows(summary_ws1)
        rows_ws2 = _extract_rows(summary_ws2)
        if not rows_ws1 or not rows_ws2:
            pytest.xfail("get_usage_summary rows not implemented yet")

        ws_values_1 = {
            row.get("workspace")
            for row in rows_ws1
            if isinstance(row, dict) and "workspace" in row
        }
        ws_values_2 = {
            row.get("workspace")
            for row in rows_ws2
            if isinstance(row, dict) and "workspace" in row
        }

        if not ws_values_1 or not ws_values_2:
            pytest.xfail("workspace grouping in usage summary not implemented yet")

        assert ws_values_1 == {ws1}
        assert ws_values_2 == {ws2_name}
    finally:
        if api_ws2 is not None:
            await api_ws2.disconnect()
        await api_ws1.disconnect()
