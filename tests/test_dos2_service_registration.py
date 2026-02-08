"""Tests for DOS2: Service registration flooding protection.

This test suite validates DOS protection for service registration:
- Rate limiting per workspace
- Service quota enforcement
- Metrics tracking
"""

import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from hypha.core.workspace import WorkspaceManager
from hypha.core.metrics import (
    DOS_SERVICE_REGISTRATIONS_TOTAL,
)


class TestServiceRegistrationDoSConfiguration:
    """Test service registration DoS protection configuration."""

    def test_service_rate_limit_initialization(self):
        """Test WorkspaceManager initializes with DOS2 protection config."""
        # Create mocks
        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        # Create WorkspaceManager
        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        # Verify DoS config attributes exist
        assert hasattr(wm, "_rate_limiter")
        assert hasattr(wm, "_service_rate_limit_enabled")
        assert hasattr(wm, "_max_services_per_workspace")
        assert hasattr(wm, "_service_registration_rate")
        assert hasattr(wm, "_service_registration_window")

    def test_service_default_limits(self):
        """Test default service registration limits are set."""
        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        # Check defaults are sensible
        assert wm._max_services_per_workspace == 1000
        assert wm._service_registration_rate == 100
        assert wm._service_registration_window == 60

    def test_service_env_override(self, monkeypatch):
        """Test service rate limit configuration from environment variables."""
        # Set custom environment variables
        monkeypatch.setenv("HYPHA_SERVICE_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("HYPHA_MAX_SERVICES_PER_WORKSPACE", "500")
        monkeypatch.setenv("HYPHA_SERVICE_REGISTRATION_RATE", "50")
        monkeypatch.setenv("HYPHA_SERVICE_REGISTRATION_WINDOW", "30")

        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        # Verify custom values
        assert wm._max_services_per_workspace == 500
        assert wm._service_registration_rate == 50
        assert wm._service_registration_window == 30
        assert wm._service_rate_limit_enabled is True

    def test_service_rate_limit_disabled(self, monkeypatch):
        """Test service rate limiting can be disabled via environment."""
        monkeypatch.setenv("HYPHA_SERVICE_RATE_LIMIT_ENABLED", "false")

        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        assert wm._service_rate_limit_enabled is False


class TestServiceRegistrationMetrics:
    """Test service registration metrics are properly tracked."""

    def test_service_registration_metrics_exist(self):
        """Test that service registration metrics are defined."""
        assert DOS_SERVICE_REGISTRATIONS_TOTAL is not None

    def test_service_metrics_have_labels(self):
        """Test service registration metrics have correct labels."""
        # Test registration metric can be labeled
        metric = DOS_SERVICE_REGISTRATIONS_TOTAL.labels(
            workspace="test", status="accepted"
        )
        assert metric is not None

        # Test rejected status
        rejected = DOS_SERVICE_REGISTRATIONS_TOTAL.labels(
            workspace="test", status="rejected_quota"
        )
        assert rejected is not None

    def test_service_metrics_can_be_incremented(self):
        """Test service registration metrics can be updated."""
        workspace = "svc-metrics-test"

        # Get initial value
        initial = DOS_SERVICE_REGISTRATIONS_TOTAL.labels(
            workspace=workspace, status="accepted"
        )._value._value

        # Increment
        DOS_SERVICE_REGISTRATIONS_TOTAL.labels(
            workspace=workspace, status="accepted"
        ).inc()

        # Verify increment
        after = DOS_SERVICE_REGISTRATIONS_TOTAL.labels(
            workspace=workspace, status="accepted"
        )._value._value

        assert after > initial


class TestServiceRateLimiterIntegration:
    """Test service registration integrates with rate limiter."""

    def test_rate_limiter_created_with_redis(self):
        """Test rate limiter is created when Redis is available."""
        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        # Rate limiter should be created with Redis
        assert wm._rate_limiter is not None

    def test_rate_limiter_none_without_redis(self):
        """Test rate limiter is None when Redis is unavailable."""
        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=None,  # No Redis
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        # Rate limiter should be None without Redis
        assert wm._rate_limiter is None


class TestServiceRegistrationQuota:
    """Test service registration quota limits."""

    def test_quota_configuration(self):
        """Test service quota can be configured."""
        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        # Default quota should be set
        assert wm._max_services_per_workspace > 0

    def test_quota_custom_value(self, monkeypatch):
        """Test custom quota value from environment."""
        monkeypatch.setenv("HYPHA_MAX_SERVICES_PER_WORKSPACE", "100")

        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        assert wm._max_services_per_workspace == 100


class TestServiceRegistrationRateWindow:
    """Test service registration rate window configuration."""

    def test_default_rate_window(self):
        """Test default rate window is set."""
        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        # Default window should be 60 seconds
        assert wm._service_registration_window == 60

    def test_custom_rate_window(self, monkeypatch):
        """Test custom rate window from environment."""
        monkeypatch.setenv("HYPHA_SERVICE_REGISTRATION_WINDOW", "120")

        mock_store = MagicMock()
        mock_redis = AsyncMock()
        mock_event_bus = MagicMock()
        mock_root_user = MagicMock()
        mock_server_info = {}

        wm = WorkspaceManager(
            store=mock_store,
            redis=mock_redis,
            root_user=mock_root_user,
            event_bus=mock_event_bus,
            server_info=mock_server_info,
            client_id="test-manager",
        )

        assert wm._service_registration_window == 120
