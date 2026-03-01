"""Tests for null value preservation in service metadata during Redis serialization.

Reproduces issue #901: Null values silently stripped from service metadata
during Redis serialization via ServiceInfo.to_redis_dict().
"""

import pytest

from hypha.core import ServiceInfo, ServiceConfig


class TestServiceInfoNullSerialization:
    """Unit tests for ServiceInfo.to_redis_dict / from_redis_dict null handling."""

    def test_null_fields_present_in_redis_dict(self):
        """Explicitly null fields should appear in the Redis dict (not be skipped).

        This is the core of issue #901: to_redis_dict() currently skips None
        values, so they are absent from Redis entirely. After the fix, null
        fields should be stored with a sentinel so they can be distinguished
        from "never set".
        """
        svc = ServiceInfo(
            id="test/svc-1",
            name=None,  # Explicitly null
            type="generic",
            description=None,  # Explicitly null
        )

        redis_data = svc.to_redis_dict()

        # After fix: null fields should be present in redis_data
        assert "name" in redis_data, (
            "Null 'name' field was silently dropped from redis_data. "
            "This is the bug described in issue #901."
        )
        assert "description" in redis_data, (
            "Null 'description' field was silently dropped from redis_data."
        )

    def test_null_round_trip_preserves_none(self):
        """Null fields should round-trip through to_redis_dict/from_redis_dict as None."""
        svc = ServiceInfo(
            id="test/svc-2",
            name=None,
            type="generic",
            description=None,
            docs=None,
            app_id=None,
            service_schema=None,
        )

        redis_data = svc.to_redis_dict()
        restored = ServiceInfo.from_redis_dict(redis_data, in_bytes=False)

        assert restored.name is None
        assert restored.description is None
        assert restored.docs is None
        assert restored.app_id is None
        assert restored.service_schema is None

    def test_null_config_extra_fields_preserved(self):
        """Null values in ServiceConfig extra fields (like metadata) should survive."""
        config = ServiceConfig(visibility="public", require_context=False)
        config_dict = config.model_dump()
        config_dict["metadata"] = {"ralphLoop": None, "title": "test"}
        config = ServiceConfig.model_validate(config_dict)

        svc = ServiceInfo(id="test/svc-3", name="Test Service", config=config)

        redis_data = svc.to_redis_dict()
        restored = ServiceInfo.from_redis_dict(redis_data, in_bytes=False)

        metadata = restored.config.model_dump().get("metadata", {})
        assert "ralphLoop" in metadata
        assert metadata["ralphLoop"] is None
        assert metadata["title"] == "test"

    def test_mix_of_null_and_non_null_fields(self):
        """A service with some null and some non-null fields should preserve both."""
        svc = ServiceInfo(
            id="test/svc-4",
            name="Calculator",
            type="generic",
            description=None,
            docs="Some docs",
            app_id=None,
            service_schema={"methods": ["add"]},
        )

        redis_data = svc.to_redis_dict()
        restored = ServiceInfo.from_redis_dict(redis_data, in_bytes=False)

        assert restored.id == "test/svc-4"
        assert restored.name == "Calculator"
        assert restored.type == "generic"
        assert restored.description is None
        assert restored.docs == "Some docs"
        assert restored.app_id is None
        assert restored.service_schema == {"methods": ["add"]}

    def test_bytes_mode_null_round_trip(self):
        """Null fields should survive round-trip when Redis returns bytes."""
        svc = ServiceInfo(
            id="test/svc-5",
            name=None,
            description="A service",
            app_id=None,
        )

        redis_data = svc.to_redis_dict()

        # Simulate Redis returning bytes
        bytes_data = {}
        for k, v in redis_data.items():
            bytes_data[k.encode("utf-8")] = (
                v.encode("utf-8") if isinstance(v, str) else v
            )

        restored = ServiceInfo.from_redis_dict(bytes_data, in_bytes=True)

        assert restored.id == "test/svc-5"
        assert restored.name is None
        assert restored.description == "A service"
        assert restored.app_id is None

    def test_sentinel_not_leaked_in_model_dump(self):
        """The Redis null sentinel should never appear in model_dump() output."""
        svc = ServiceInfo(
            id="test/svc-6",
            name=None,
            description=None,
        )

        redis_data = svc.to_redis_dict()
        restored = ServiceInfo.from_redis_dict(redis_data, in_bytes=False)
        dumped = restored.model_dump()

        # Sentinel values must not leak into the model dump
        assert dumped["name"] is None
        assert dumped["description"] is None
        for key, value in dumped.items():
            if isinstance(value, str):
                assert value != "__null__", (
                    f"Sentinel leaked into model_dump for field '{key}'"
                )


@pytest.mark.asyncio
async def test_null_metadata_via_register_service(minio_server, fastapi_server):
    """Integration test: register a service with null metadata fields,
    retrieve it, and verify nulls are preserved through Redis.
    """
    from hypha_rpc import connect_to_server
    from tests import SERVER_URL, find_item

    api = await connect_to_server(
        {"name": "null-test-client", "server_url": SERVER_URL}
    )

    svc_info = await api.register_service(
        {
            "id": "null-test-svc",
            "name": "Null Test Service",
            "type": "generic",
            "description": None,  # Explicitly null
            "config": {
                "visibility": "protected",
                "metadata": {"ralphLoop": None, "title": "test session"},
            },
            "echo": lambda x: x,
        }
    )

    # Retrieve the service info via get_service
    svc = await api.get_service("null-test-svc")
    svc_info_dict = svc.to_dict() if hasattr(svc, 'to_dict') else dict(svc)

    # Also verify via list_services
    wm = await api.get_service("~")
    services = await wm.list_services()
    # Service IDs in list_services include the client prefix
    svc_from_list = find_item(
        services, lambda s: s.get("id", "").endswith(":null-test-svc")
    )
    assert svc_from_list is not None, (
        f"Service null-test-svc not found in list. "
        f"Available IDs: {[s.get('id') for s in services]}"
    )

    # Verify null top-level field preserved
    assert svc_from_list["description"] is None, (
        f"description should be None but got: {svc_from_list['description']}"
    )
    # Check metadata within config
    config = svc_from_list.get("config", {})
    metadata = config.get("metadata", {})
    assert "ralphLoop" in metadata, (
        f"ralphLoop key missing from config.metadata. Got: {metadata}"
    )
    assert metadata["ralphLoop"] is None
    assert metadata["title"] == "test session"
