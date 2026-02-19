"""Contract-first tests for interceptor APIs and behavior."""

import datetime

import pytest
from hypha_rpc import connect_to_server
from hypha.core.workspace import (
    BillingContractError,
    EventInterceptor,
    EventLog,
    WorkspaceManager,
)

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


def _assert_error_code(exc: Exception, expected_code: str):
    code = _extract_error_code(exc)
    if code == expected_code:
        return
    assert expected_code in str(exc), (
        f"Expected error code {expected_code}, got {code}. " f"Error: {exc}"
    )


def _latest_event(events):
    if not events:
        return None
    return sorted(events, key=lambda item: item.get("id", 0))[-1]


async def _connect_client(*, token=None, workspace=None, client_id=None):
    config = {
        "server_url": SERVER_URL,
        "name": client_id or "interceptor-contract-client",
    }
    if client_id:
        config["client_id"] = client_id
    if token:
        config["token"] = token
    if workspace:
        config["workspace"] = workspace
    return await connect_to_server(config)


def _build_interceptor(
    *,
    workspace: str,
    event_type: str,
    name: str,
    field: str,
    op: str,
    value,
    action_type: str,
    reason: str = "policy",
    priority: int = 100,
):
    interceptor = {
        "name": name,
        "scope": {"workspace": workspace},
        "enabled": True,
        "priority": priority,
        "event_selector": {
            "event_types": [event_type],
            "categories": ["application"],
        },
        "condition": {
            "field": field,
            "op": op,
            "value": value,
        },
        "action": {
            "type": action_type,
            "reason": reason,
        },
    }
    if action_type == "recover":
        interceptor["action"]["recovery"] = {
            "mode": "log_only",
            "event_type": "interceptor.recovered",
        }
    return interceptor


async def _register_or_xfail(api, interceptor):
    try:
        return await api.register_interceptor(interceptor)
    except Exception as exc:
        if _is_missing_method_error(exc, "register_interceptor"):
            pytest.xfail("register_interceptor API is not implemented yet")
        raise


async def _list_or_xfail(api):
    try:
        return await api.list_interceptors()
    except Exception as exc:
        if _is_missing_method_error(exc, "list_interceptors"):
            pytest.xfail("list_interceptors API is not implemented yet")
        raise


async def _remove_or_xfail(api, interceptor_id):
    try:
        return await api.remove_interceptor(interceptor_id)
    except Exception as exc:
        if _is_missing_method_error(exc, "remove_interceptor"):
            pytest.xfail("remove_interceptor API is not implemented yet")
        raise


async def test_register_list_remove_interceptor_contract(
    fastapi_server,
    test_user_token,
):
    """Contract: register/list/remove APIs exist and preserve priority ordering."""
    api = await _connect_client(
        token=test_user_token, client_id="contract-int-lifecycle"
    )
    created_ids = []

    try:
        workspace = api.config.workspace
        event_type = "contract.interceptor.lifecycle"

        first = await _register_or_xfail(
            api,
            _build_interceptor(
                workspace=workspace,
                event_type=event_type,
                name="lifecycle-priority-20",
                field="event_type",
                op="eq",
                value=event_type,
                action_type="allow",
                priority=20,
            ),
        )
        second = await _register_or_xfail(
            api,
            _build_interceptor(
                workspace=workspace,
                event_type=event_type,
                name="lifecycle-priority-10",
                field="event_type",
                op="eq",
                value=event_type,
                action_type="allow",
                priority=10,
            ),
        )

        first_id = first.get("id") if isinstance(first, dict) else None
        second_id = second.get("id") if isinstance(second, dict) else None
        assert (
            first_id and second_id
        ), "register_interceptor must return created interceptor IDs"
        created_ids.extend([first_id, second_id])

        items = await _list_or_xfail(api)
        assert isinstance(items, list)

        indexed = {
            item.get("id"): idx
            for idx, item in enumerate(items)
            if isinstance(item, dict)
        }
        assert first_id in indexed and second_id in indexed
        assert (
            indexed[second_id] < indexed[first_id]
        ), "Interceptors should be ordered by priority asc"

        for interceptor_id in [first_id, second_id]:
            removal = await _remove_or_xfail(api, interceptor_id)
            if isinstance(removal, dict) and "ok" in removal:
                assert removal["ok"] is True

        created_ids.clear()
        remaining = await _list_or_xfail(api)
        remaining_ids = {item.get("id") for item in remaining if isinstance(item, dict)}
        assert first_id not in remaining_ids and second_id not in remaining_ids
    finally:
        for interceptor_id in created_ids:
            try:
                await _remove_or_xfail(api, interceptor_id)
            except Exception:
                pass
        await api.disconnect()


async def test_interceptor_permission_contract_admin_only(
    fastapi_server,
    test_user_token,
):
    """Contract: only workspace admins can manage interceptors."""
    admin_api = await _connect_client(
        token=test_user_token, client_id="contract-int-admin"
    )
    read_api = None

    try:
        workspace = admin_api.config.workspace
        readonly_token = await admin_api.generate_token(
            {
                "workspace": workspace,
                "permission": "read",
                "expires_in": 3600,
            }
        )

        read_api = await _connect_client(
            token=readonly_token,
            workspace=workspace,
            client_id="contract-int-readonly",
        )

        interceptor = _build_interceptor(
            workspace=workspace,
            event_type="contract.interceptor.permission",
            name="readonly-should-fail",
            field="event_type",
            op="eq",
            value="contract.interceptor.permission",
            action_type="allow",
        )

        with pytest.raises(Exception) as exc_info:
            await _register_or_xfail(read_api, interceptor)

        err = str(exc_info.value).lower()
        code = _extract_error_code(exc_info.value)
        assert "permission" in err or code in {"FORBIDDEN", "UNAUTHORIZED"}
    finally:
        if read_api is not None:
            await read_api.disconnect()
        await admin_api.disconnect()


async def test_interceptor_condition_ops_contract(
    fastapi_server,
    test_user_token,
):
    """Contract: interceptor evaluator supports eq/ne/gt/lt/contains."""
    api = await _connect_client(token=test_user_token, client_id="contract-int-ops")
    created_ids = []

    try:
        workspace = api.config.workspace
        event_type = "contract.interceptor.ops"

        cases = [
            ("eq", "data.value", 10, {"value": 10, "text": "alpha gold"}, True),
            ("ne", "data.value", 5, {"value": 10, "text": "alpha gold"}, True),
            ("gt", "data.value", 9, {"value": 10, "text": "alpha gold"}, True),
            ("lt", "data.value", 11, {"value": 10, "text": "alpha gold"}, True),
            (
                "contains",
                "data.text",
                "gold",
                {"value": 10, "text": "alpha gold"},
                True,
            ),
            ("eq", "data.value", 999, {"value": 10, "text": "alpha gold"}, False),
        ]

        for idx, (op, field, value, payload, should_stop) in enumerate(cases):
            interceptor = _build_interceptor(
                workspace=workspace,
                event_type=event_type,
                name=f"ops-{op}-{idx}",
                field=field,
                op=op,
                value=value,
                action_type="stop",
                reason="policy",
                priority=10,
            )
            created = await _register_or_xfail(api, interceptor)
            interceptor_id = created.get("id") if isinstance(created, dict) else None
            if not interceptor_id:
                pytest.fail("register_interceptor must return interceptor id")
            created_ids.append(interceptor_id)

            if should_stop:
                with pytest.raises(Exception) as exc_info:
                    await api.log_event(event_type, payload)
                _assert_error_code(exc_info.value, "INTERCEPT_STOP_POLICY")
            else:
                await api.log_event(event_type, payload)

            await _remove_or_xfail(api, interceptor_id)
            created_ids.remove(interceptor_id)
    finally:
        for interceptor_id in created_ids:
            try:
                await _remove_or_xfail(api, interceptor_id)
            except Exception:
                pass
        await api.disconnect()


async def test_interceptor_stop_only_current_call_contract(
    fastapi_server,
    test_user_token,
):
    """Contract: stop blocks current call only and does not kill session."""
    api = await _connect_client(token=test_user_token, client_id="contract-int-stop")
    interceptor_id = None

    try:
        workspace = api.config.workspace
        blocked_type = "contract.interceptor.blocked"
        allowed_type = "contract.interceptor.allowed"

        created = await _register_or_xfail(
            api,
            _build_interceptor(
                workspace=workspace,
                event_type=blocked_type,
                name="stop-current-call-only",
                field="event_type",
                op="eq",
                value=blocked_type,
                action_type="stop",
                reason="policy",
            ),
        )
        interceptor_id = created.get("id") if isinstance(created, dict) else None

        with pytest.raises(Exception) as exc_info:
            await api.log_event(blocked_type, {"blocked": True})
        _assert_error_code(exc_info.value, "INTERCEPT_STOP_POLICY")

        services = await api.list_services()
        assert isinstance(services, list), "Session should remain healthy after stop"

        await api.log_event(allowed_type, {"blocked": False})
    finally:
        if interceptor_id is not None:
            try:
                await _remove_or_xfail(api, interceptor_id)
            except Exception:
                pass
        await api.disconnect()


@pytest.mark.parametrize(
    "reason, expected_code",
    [
        ("policy", "INTERCEPT_STOP_POLICY"),
        ("security", "INTERCEPT_STOP_SECURITY"),
        ("limit", "INTERCEPT_STOP_LIMIT"),
    ],
)
async def test_interceptor_reason_to_error_code_contract(
    fastapi_server,
    test_user_token,
    reason,
    expected_code,
):
    """Contract: reason-specific stop actions map to stable codes and persisted flags."""
    api = await _connect_client(
        token=test_user_token, client_id=f"contract-int-code-{reason}"
    )
    interceptor_id = None

    try:
        workspace = api.config.workspace
        event_type = f"contract.interceptor.reason.{reason}"

        created = await _register_or_xfail(
            api,
            _build_interceptor(
                workspace=workspace,
                event_type=event_type,
                name=f"reason-{reason}",
                field="event_type",
                op="eq",
                value=event_type,
                action_type="stop",
                reason=reason,
            ),
        )
        interceptor_id = created.get("id") if isinstance(created, dict) else None

        with pytest.raises(Exception) as exc_info:
            await api.log_event(event_type, {"reason": reason})
        _assert_error_code(exc_info.value, expected_code)

        events = await api.get_events(event_type=event_type)
        event = _latest_event(events)
        assert event is not None, "Expected blocked event row to remain persisted"

        required_fields = [
            "intercepted",
            "interceptor_action",
            "interceptor_reason",
            "interceptor_code",
            "interceptor_id",
            "interceptor_name",
            "intercepted_at",
        ]
        missing_fields = [field for field in required_fields if field not in event]
        if missing_fields:
            pytest.xfail(
                "Intercepted-stop event fields are not persisted yet "
                f"(missing: {', '.join(missing_fields)})"
            )

        assert event.get("intercepted") is True
        assert event.get("interceptor_action") == "stop"
        assert event.get("interceptor_reason") == reason
        assert event.get("interceptor_code") == expected_code
        assert event.get("interceptor_id") == interceptor_id
        assert event.get("interceptor_name") == f"reason-{reason}"
        assert event.get("intercepted_at"), "Expected intercepted_at timestamp"
    finally:
        if interceptor_id is not None:
            try:
                await _remove_or_xfail(api, interceptor_id)
            except Exception:
                pass
        await api.disconnect()


async def test_interceptor_stop_flag_write_failure_returns_internal_error_unit_contract(
    monkeypatch,
):
    """Contract: _mark_event_intercepted_stop failure maps to INTERNAL_ERROR."""

    class _DummyScalarResult:
        def __init__(self, items):
            self._items = items

        def scalars(self):
            return self

        def all(self):
            return self._items

    class _DummyBegin:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummySession:
        def __init__(self, interceptors):
            self._interceptors = interceptors

        def begin(self):
            return _DummyBegin()

        async def execute(self, query):
            return _DummyScalarResult(self._interceptors)

        async def close(self):
            return None

    workspace = "contract-unit-interceptor-workspace"
    event_type = "contract.interceptor.stop-flag-write-failure"
    interceptor = EventInterceptor(
        id="int-stop-write-failure",
        name="stop-write-failure",
        workspace=workspace,
        app_id=None,
        session_id=None,
        enabled=True,
        priority=100,
        event_types=[event_type],
        categories=["application"],
        condition_field="event_type",
        condition_op="eq",
        condition_value=event_type,
        action_type="stop",
        action_reason="policy",
        action_recovery=None,
        created_by="contract-unit",
        created_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
        updated_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
    )
    event_log = EventLog(
        id=123,
        event_type=event_type,
        category="application",
        level="info",
        workspace=workspace,
        user_id="contract-unit-user",
        data={"case": "stop-flag-write-failure"},
    )

    manager = object.__new__(WorkspaceManager)

    async def _fake_get_sql_session():
        return _DummySession([interceptor])

    async def _raise_write_failure(*args, **kwargs):
        raise RuntimeError("forced intercepted-stop flag write failure")

    monkeypatch.setattr(manager, "_get_sql_session", _fake_get_sql_session)
    monkeypatch.setattr(
        manager, "_mark_event_intercepted_stop", _raise_write_failure
    )

    with pytest.raises(BillingContractError) as exc_info:
        await manager._evaluate_interceptors(event_log)

    assert exc_info.value.code == "INTERNAL_ERROR"
    assert exc_info.value.reason == "internal"
    details = exc_info.value.details
    assert details.get("event_id") == 123
    assert details.get("workspace") == workspace
    assert details.get("interceptor_id") == interceptor.id
    assert details.get("interceptor_code") == "INTERCEPT_STOP_POLICY"


async def test_interceptor_recover_action_contract_xfail_until_implemented(
    fastapi_server,
    test_user_token,
):
    """Contract skeleton: recover action should return controlled outcome."""
    api = await _connect_client(token=test_user_token, client_id="contract-int-recover")
    interceptor_id = None

    try:
        workspace = api.config.workspace
        event_type = "contract.interceptor.recover"
        created = await _register_or_xfail(
            api,
            _build_interceptor(
                workspace=workspace,
                event_type=event_type,
                name="recover-action",
                field="event_type",
                op="eq",
                value=event_type,
                action_type="recover",
                reason="policy",
            ),
        )
        interceptor_id = created.get("id") if isinstance(created, dict) else None

        try:
            result = await api.log_event(event_type, {"mode": "recover"})
        except Exception as exc:
            msg = str(exc).lower()
            if "recover" in msg and (
                "not implemented" in msg or "unsupported" in msg or "invalid" in msg
            ):
                pytest.xfail("recover interceptor action is not implemented yet")
            raise

        if result is None:
            pytest.xfail(
                "recover action contract return shape is not implemented yet "
                "(log_event returned no structured recovery result)"
            )
    finally:
        if interceptor_id is not None:
            try:
                await _remove_or_xfail(api, interceptor_id)
            except Exception:
                pass
        await api.disconnect()
