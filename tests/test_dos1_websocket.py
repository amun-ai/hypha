"""Tests for DOS1: WebSocket connection flooding protection.

This test suite validates DOS protection for WebSocket connections:
- Rate limiting configuration
- Metrics tracking
- Integration with rate limiter
"""

import pytest
import os
from unittest.mock import MagicMock
from hypha.websocket import WebsocketServer
from hypha.core.metrics import (
    DOS_WEBSOCKET_CONNECTIONS_TOTAL,
    DOS_WEBSOCKET_ACTIVE_CONNECTIONS,
)


class TestWebSocketDoSConfiguration:
    """Test WebSocket DoS protection configuration."""

    def test_websocket_rate_limit_initialization(self):
        """Test WebSocket server initializes with DoS protection config."""
        # Create mock store
        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = MagicMock()  # Mock Redis client
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        # Create WebSocket server
        ws_server = WebsocketServer(mock_store)

        # Verify DoS config attributes exist
        assert hasattr(ws_server, "_rate_limiter")
        assert hasattr(ws_server, "_ws_rate_limit_enabled")
        assert hasattr(ws_server, "_ws_max_connections_per_workspace")
        assert hasattr(ws_server, "_ws_max_connections_per_user")
        assert hasattr(ws_server, "_ws_connection_rate_per_workspace")
        assert hasattr(ws_server, "_ws_connection_rate_window")

    def test_websocket_default_limits(self):
        """Test default rate limit values are set."""
        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = MagicMock()
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        # Check defaults are sensible
        assert ws_server._ws_max_connections_per_workspace == 1000
        assert ws_server._ws_max_connections_per_user == 100
        assert ws_server._ws_connection_rate_per_workspace == 50
        assert ws_server._ws_connection_rate_window == 60

    def test_websocket_env_override(self, monkeypatch):
        """Test rate limit configuration from environment variables."""
        # Set custom environment variables
        monkeypatch.setenv("HYPHA_WS_MAX_CONNECTIONS_PER_WORKSPACE", "500")
        monkeypatch.setenv("HYPHA_WS_MAX_CONNECTIONS_PER_USER", "50")
        monkeypatch.setenv("HYPHA_WS_CONNECTION_RATE_PER_WORKSPACE", "25")
        monkeypatch.setenv("HYPHA_WS_CONNECTION_RATE_WINDOW", "30")
        monkeypatch.setenv("HYPHA_WS_RATE_LIMIT_ENABLED", "true")

        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = MagicMock()
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        # Verify custom values
        assert ws_server._ws_max_connections_per_workspace == 500
        assert ws_server._ws_max_connections_per_user == 50
        assert ws_server._ws_connection_rate_per_workspace == 25
        assert ws_server._ws_connection_rate_window == 30
        assert ws_server._ws_rate_limit_enabled is True

    def test_websocket_rate_limit_disabled(self, monkeypatch):
        """Test rate limiting can be disabled via environment."""
        monkeypatch.setenv("HYPHA_WS_RATE_LIMIT_ENABLED", "false")

        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = MagicMock()
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        assert ws_server._ws_rate_limit_enabled is False


class TestWebSocketMetricsTracking:
    """Test WebSocket metrics are properly tracked."""

    def test_websocket_connection_metrics_exist(self):
        """Test that WebSocket connection metrics are defined."""
        # These metrics should be importable and defined
        assert DOS_WEBSOCKET_CONNECTIONS_TOTAL is not None
        assert DOS_WEBSOCKET_ACTIVE_CONNECTIONS is not None

    def test_websocket_metrics_have_labels(self):
        """Test WebSocket metrics have correct labels."""
        # Test connection metric can be labeled
        metric = DOS_WEBSOCKET_CONNECTIONS_TOTAL.labels(
            workspace="test", status="accepted"
        )
        assert metric is not None

        # Test active connections metric
        active_metric = DOS_WEBSOCKET_ACTIVE_CONNECTIONS.labels(workspace="test")
        assert active_metric is not None

    def test_websocket_metrics_can_be_incremented(self):
        """Test WebSocket metrics can be updated."""
        workspace = "metrics-test-ws"

        # Get initial value
        initial = DOS_WEBSOCKET_CONNECTIONS_TOTAL.labels(
            workspace=workspace, status="accepted"
        )._value._value

        # Increment
        DOS_WEBSOCKET_CONNECTIONS_TOTAL.labels(
            workspace=workspace, status="accepted"
        ).inc()

        # Verify increment
        after = DOS_WEBSOCKET_CONNECTIONS_TOTAL.labels(
            workspace=workspace, status="accepted"
        )._value._value

        assert after > initial

    def test_websocket_active_connections_gauge(self):
        """Test active connections gauge can be set."""
        workspace = "gauge-test-ws"

        # Set gauge value
        DOS_WEBSOCKET_ACTIVE_CONNECTIONS.labels(workspace=workspace).set(5)

        # Read value
        value = DOS_WEBSOCKET_ACTIVE_CONNECTIONS.labels(workspace=workspace)._value._value

        assert value == 5


class TestWebSocketRateLimiterIntegration:
    """Test WebSocket server integrates with rate limiter."""

    def test_rate_limiter_created_with_redis(self):
        """Test rate limiter is created when Redis is available."""
        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_redis = MagicMock()  # Mock Redis client
        mock_event_bus._redis = mock_redis
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        # Rate limiter should be created with Redis
        assert ws_server._rate_limiter is not None

    def test_rate_limiter_none_without_redis(self):
        """Test rate limiter is None when Redis is unavailable."""
        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = None  # No Redis
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        # Rate limiter should be None without Redis
        assert ws_server._rate_limiter is None


class TestWebSocketIdleTimeout:
    """Test WebSocket idle timeout configuration."""

    def test_default_idle_timeout(self):
        """Test default idle timeout is set."""
        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = None
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        # Default idle timeout should be 600 seconds (10 minutes)
        assert ws_server._idle_timeout == 600

    def test_custom_idle_timeout(self, monkeypatch):
        """Test custom idle timeout from environment."""
        monkeypatch.setenv("HYPHA_WS_IDLE_TIMEOUT", "300")

        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = None
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        assert ws_server._idle_timeout == 300


class TestWebSocketConnectionTracking:
    """Test WebSocket connection tracking."""

    def test_websocket_tracking_dict_initialized(self):
        """Test WebSocket tracking dictionary is initialized."""
        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = None
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        assert hasattr(ws_server, "_websockets")
        assert isinstance(ws_server._websockets, dict)
        assert len(ws_server._websockets) == 0

    def test_last_seen_tracking_initialized(self):
        """Test last seen tracking is initialized."""
        mock_store = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus._redis = None
        mock_store.get_event_bus.return_value = mock_event_bus
        mock_store._app = MagicMock()
        mock_store.set_websocket_server = MagicMock()

        ws_server = WebsocketServer(mock_store)

        assert hasattr(ws_server, "_last_seen")
        assert isinstance(ws_server._last_seen, dict)
        assert len(ws_server._last_seen) == 0
