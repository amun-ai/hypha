"""Tests for ServiceInfo old-format to new-format migration."""

import json

import pytest

from hypha.core import ServiceConfig, ServiceInfo


class TestServiceInfoOldFormatMigration:
    """Test that from_redis_dict handles old-format flat config keys."""

    def _make_old_format_data(self, in_bytes=True):
        """Create old-format Redis data with flat config keys (no 'config' key).

        In the old format, to_redis_dict() stored:
        - ServiceInfo str fields: as plain strings (id, name, type)
        - ServiceInfo non-str fields: JSON-encoded (description via json.dumps, service_schema)
        - ServiceConfig str fields: as plain strings (visibility, workspace)
        - ServiceConfig non-str fields: JSON-encoded (require_context, singleton, created_by)
        - ServiceConfig list fields: comma-separated (flags, authorized_workspaces)
        """
        data = {
            "id": "test-ws/client-id:my-service",
            "name": "My Service",
            "type": "generic",
            # description was JSON-encoded in old format (constr != Optional[str])
            "description": json.dumps("A test service"),
            # Old format: config fields stored as flat hash keys
            "visibility": "protected",  # str field -> plain string
            "require_context": "false",  # Union type -> JSON-encoded
            "singleton": "false",  # Optional[bool] -> JSON-encoded
            "workspace": "test-ws",  # Optional[str] -> plain string
            "flags": "",  # List[str] -> comma-separated (empty)
            "created_by": json.dumps({"id": "user@example.com"}),  # Dict -> JSON
        }
        if in_bytes:
            return {k.encode(): v.encode() for k, v in data.items()}
        return data

    def _make_new_format_data(self, in_bytes=True):
        """Create new-format Redis data with config as JSON key."""
        config = {
            "visibility": "protected",
            "require_context": False,
            "singleton": False,
            "workspace": "test-ws",
            "flags": [],
            "created_by": {"id": "user@example.com"},
        }
        data = {
            "id": "test-ws/client-id:my-service",
            "name": "My Service",
            "type": "generic",
            # description is JSON-encoded by to_redis_dict (constr != Optional[str])
            "description": json.dumps("A test service"),
            "config": json.dumps(config),
        }
        if in_bytes:
            return {k.encode(): v.encode() for k, v in data.items()}
        return data

    def test_new_format_roundtrip(self):
        """New format data round-trips correctly."""
        data = self._make_new_format_data()
        info = ServiceInfo.from_redis_dict(data)
        assert info.id == "test-ws/client-id:my-service"
        assert info.name == "My Service"
        assert info.config is not None
        assert info.config.visibility.value == "protected"
        assert info.config.singleton is False
        assert info.config.workspace == "test-ws"

    def test_old_format_deserializes(self):
        """Old format data (flat config keys) deserializes correctly."""
        data = self._make_old_format_data()
        info = ServiceInfo.from_redis_dict(data)
        assert info.id == "test-ws/client-id:my-service"
        assert info.name == "My Service"
        assert info.type == "generic"
        assert info.config is not None
        assert info.config.visibility.value == "protected"
        assert info.config.singleton is False
        assert info.config.workspace == "test-ws"
        assert info.config.created_by == {"id": "user@example.com"}

    def test_old_format_reserializes_to_new_format(self):
        """Old format data, after deserialization, serializes to new format."""
        data = self._make_old_format_data()
        info = ServiceInfo.from_redis_dict(data)
        new_data = info.to_redis_dict()
        # New format should have 'config' as a JSON string
        assert "config" in new_data
        config = json.loads(new_data["config"])
        assert config["visibility"] == "protected"
        assert config["singleton"] is False
        # Should NOT have flat config keys
        assert "visibility" not in new_data
        assert "singleton" not in new_data
        assert "require_context" not in new_data

    def test_old_format_non_bytes(self):
        """Old format works with string keys (non-bytes)."""
        data = self._make_old_format_data(in_bytes=False)
        info = ServiceInfo.from_redis_dict(data, in_bytes=False)
        assert info.config is not None
        assert info.config.visibility.value == "protected"

    def test_old_format_with_authorized_workspaces(self):
        """Old format with comma-separated authorized_workspaces."""
        data = self._make_old_format_data()
        data[b"authorized_workspaces"] = b"ws1,ws2,ws3"
        info = ServiceInfo.from_redis_dict(data)
        assert info.config.authorized_workspaces == ["ws1", "ws2", "ws3"]

    def test_old_format_minimal(self):
        """Old format with only visibility (minimal flat config)."""
        data = {
            b"id": b"test-ws/client:svc",
            b"name": b"Svc",
            b"type": b"generic",
            b"visibility": b"public",
        }
        info = ServiceInfo.from_redis_dict(data)
        assert info.config is not None
        assert info.config.visibility.value == "public"

    def test_no_config_no_flat_keys(self):
        """Data with no config key and no flat config keys still works."""
        data = {
            b"id": b"test-ws/client:svc",
            b"name": b"Svc",
            b"type": b"generic",
        }
        info = ServiceInfo.from_redis_dict(data)
        assert info.id == "test-ws/client:svc"
        # No config data at all
        assert info.config is None

    def test_migration_roundtrip(self):
        """Full migration simulation: old data -> from_redis_dict -> to_redis_dict -> from_redis_dict."""
        old_data = self._make_old_format_data()
        # Step 1: Parse old format
        info1 = ServiceInfo.from_redis_dict(old_data)
        # Step 2: Serialize to new format
        new_data = info1.to_redis_dict()
        # Step 3: Simulate reading back from Redis (bytes keys)
        bytes_data = {k.encode(): v.encode() for k, v in new_data.items()}
        info2 = ServiceInfo.from_redis_dict(bytes_data)
        # Verify data is preserved
        assert info2.id == info1.id
        assert info2.name == info1.name
        assert info2.config.visibility == info1.config.visibility
        assert info2.config.singleton == info1.config.singleton
        assert info2.config.workspace == info1.config.workspace
        assert info2.config.created_by == info1.config.created_by
