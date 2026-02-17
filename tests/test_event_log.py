"""Contract-first tests for event logging APIs."""

import pytest
from hypha_rpc import connect_to_server

from . import SERVER_URL

pytestmark = [pytest.mark.asyncio, pytest.mark.logging_billing]


async def _connect_client(*, token=None, workspace=None, client_id=None):
    """Create a test client connection with predictable options."""
    config = {
        "server_url": SERVER_URL,
        "name": client_id or "event-log-contract-client",
    }
    if client_id:
        config["client_id"] = client_id
    if token:
        config["token"] = token
    if workspace:
        config["workspace"] = workspace
    return await connect_to_server(config)


async def test_log_event_legacy_contract_still_works(fastapi_server, test_user_token):
    """Existing positional log_event contract must remain compatible."""
    api = await _connect_client(
        token=test_user_token, client_id="contract-event-legacy"
    )

    try:
        event_type = "contract.event.legacy"
        event_data = {"case": "legacy", "value": 1}
        await api.log_event(event_type, event_data)

        events = await api.get_events(event_type=event_type)
        assert events, "Expected at least one persisted event"
        assert any(e.get("event_type") == event_type for e in events)
        assert any(e.get("data") == event_data for e in events)
    finally:
        await api.disconnect()


async def test_log_event_rejects_event_type_with_spaces(
    fastapi_server, test_user_token
):
    """Contract: event_type must reject spaces."""
    api = await _connect_client(
        token=test_user_token,
        client_id="contract-event-invalid-event-type",
    )

    try:
        with pytest.raises(Exception) as exc_info:
            await api.log_event("contract event invalid", {"case": "invalid"})

        assert "space" in str(exc_info.value).lower()
    finally:
        await api.disconnect()


async def test_get_event_stats_filters_by_event_type(fastapi_server, test_user_token):
    """Contract: get_event_stats(event_type=...) counts only selected type."""
    api = await _connect_client(token=test_user_token, client_id="contract-event-stats")

    try:
        await api.log_event("contract.event.stats.target", {"k": 1})
        await api.log_event("contract.event.stats.target", {"k": 2})
        await api.log_event("contract.event.stats.other", {"k": 3})

        stats = await api.get_event_stats(event_type="contract.event.stats.target")
        assert stats, "Expected non-empty stats for target event type"

        target_stats = [
            s for s in stats if s.get("event_type") == "contract.event.stats.target"
        ]
        assert target_stats, "Expected target event type in stats output"
        assert target_stats[0]["count"] >= 2
    finally:
        await api.disconnect()


async def test_event_log_workspace_isolation_contract(fastapi_server, test_user_token):
    """Contract: workspace context isolates event query visibility."""
    api_ws1 = await _connect_client(
        token=test_user_token, client_id="contract-event-ws1"
    )
    api_ws2 = None

    try:
        ws1 = api_ws1.config.workspace
        ws2_name = "contract-event-log-ws2"

        await api_ws1.create_workspace(
            {
                "name": ws2_name,
                "description": "Workspace for event log isolation contract",
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
            client_id="contract-event-ws2",
        )

        event_type = "contract.event.isolation"
        await api_ws1.log_event(event_type, {"origin": ws1})
        await api_ws2.log_event(event_type, {"origin": ws2_name})

        ws1_events = await api_ws1.get_events(event_type=event_type)
        ws2_events = await api_ws2.get_events(event_type=event_type)

        assert ws1_events and ws2_events
        assert all(e.get("workspace") == ws1 for e in ws1_events)
        assert all(e.get("workspace") == ws2_name for e in ws2_events)
        assert all(e.get("data", {}).get("origin") == ws1 for e in ws1_events)
        assert all(e.get("data", {}).get("origin") == ws2_name for e in ws2_events)
    finally:
        if api_ws2 is not None:
            await api_ws2.disconnect()
        await api_ws1.disconnect()


async def test_enriched_event_schema_contract_xfail_until_implemented(
    fastapi_server,
    test_user_token,
):
    """Contract: enriched event fields should be persisted once Phase 2 lands."""
    api = await _connect_client(
        token=test_user_token, client_id="contract-event-enriched"
    )

    try:
        event_type = "contract.event.enriched"
        await api.log_event(event_type, {"contract": "enriched-schema"})

        events = await api.get_events(event_type=event_type)
        assert events, "Expected persisted event for enriched schema contract"
        event = sorted(events, key=lambda item: item.get("id", 0))[-1]

        required_fields = {
            "category": "application",
            "level": "info",
            "app_id": None,
            "session_id": None,
            "idempotency_key": None,
        }
        missing = [field for field in required_fields if field not in event]
        if missing:
            pytest.xfail(
                "Enriched EventV1 fields are not persisted yet "
                f"(missing: {', '.join(missing)})"
            )

        for field, expected in required_fields.items():
            assert event.get(field) == expected
    finally:
        await api.disconnect()
