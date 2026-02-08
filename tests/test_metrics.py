"""Tests for Prometheus metrics module.

This test suite validates the centralized metrics infrastructure for:
- Rate limiting metrics
- DoS protection metrics
- Resource usage tracking
- System health monitoring
- Attack detection
- Quota management
"""

import pytest
from prometheus_client import REGISTRY, CollectorRegistry
from hypha.core.metrics import (
    # Rate limiting
    RATE_LIMIT_REQUESTS_TOTAL,
    RATE_LIMIT_REJECTIONS_TOTAL,
    RATE_LIMIT_CURRENT_RATE,
    RATE_LIMIT_TOKEN_BUCKET,
    # DoS protection
    DOS_WEBSOCKET_CONNECTIONS_TOTAL,
    DOS_WEBSOCKET_ACTIVE_CONNECTIONS,
    DOS_SERVICE_REGISTRATIONS_TOTAL,
    DOS_ARTIFACT_UPLOADS_TOTAL,
    DOS_ARTIFACT_UPLOAD_BYTES,
    DOS_DATABASE_QUERIES_TOTAL,
    DOS_RPC_MESSAGES_TOTAL,
    DOS_HTTP_REQUESTS_TOTAL,
    # Resource usage
    RESOURCE_MEMORY_BYTES,
    RESOURCE_CPU_USAGE,
    RESOURCE_ACTIVE_TASKS,
    RESOURCE_QUEUE_DEPTH,
    RESOURCE_BANDWIDTH_BYTES,
    # System health
    HEALTH_ERRORS_TOTAL,
    HEALTH_LATENCY_SECONDS,
    HEALTH_PROCESSING_TIME_SECONDS,
    HEALTH_CIRCUIT_BREAKER_STATE,
    # Attack detection
    ATTACK_DETECTION_EVENTS,
    ATTACK_BLOCKED_IPS,
    ATTACK_SUSPICIOUS_PATTERNS,
    # Quota management
    QUOTA_USAGE_PERCENTAGE,
    QUOTA_LIMIT_VALUE,
    QUOTA_VIOLATIONS_TOTAL,
    # Helper functions
    record_rate_limit_request,
    record_rate_limit_rejection,
    update_current_rate,
    record_websocket_connection,
    update_websocket_connections,
    record_service_registration,
    record_artifact_upload,
    record_database_query,
    record_rpc_message,
    record_http_request,
    record_attack_detection,
    update_quota_usage,
    record_quota_violation,
)


class TestRateLimitMetrics:
    """Test rate limiting metrics."""

    def test_rate_limit_requests_total(self):
        """Test rate limit request counter."""
        workspace = "test-ws"
        limit_type = "websocket"

        # Record allowed request
        record_rate_limit_request(workspace, limit_type, "allowed")

        # Get metric value
        metric = RATE_LIMIT_REQUESTS_TOTAL.labels(
            workspace=workspace,
            limit_type=limit_type,
            status="allowed"
        )
        assert metric._value._value >= 1

    def test_rate_limit_rejections_total(self):
        """Test rate limit rejection counter."""
        workspace = "test-ws"
        limit_type = "rpc"

        # Record rejection
        record_rate_limit_rejection(workspace, limit_type, "rate_exceeded")

        # Get metric value
        metric = RATE_LIMIT_REJECTIONS_TOTAL.labels(
            workspace=workspace,
            limit_type=limit_type,
            reason="rate_exceeded"
        )
        assert metric._value._value >= 1

    def test_rate_limit_current_rate(self):
        """Test current rate gauge."""
        workspace = "test-ws"
        limit_type = "artifact"

        # Update rate
        update_current_rate(workspace, limit_type, 42.5)

        # Get metric value
        metric = RATE_LIMIT_CURRENT_RATE.labels(
            workspace=workspace,
            limit_type=limit_type
        )
        assert metric._value._value == 42.5

    def test_rate_limit_multiple_workspaces(self):
        """Test rate limiting metrics for multiple workspaces."""
        workspaces = ["ws-1", "ws-2", "ws-3"]

        for ws in workspaces:
            record_rate_limit_request(ws, "websocket", "allowed")
            record_rate_limit_request(ws, "websocket", "allowed")

        # Each workspace should have independent counters
        for ws in workspaces:
            metric = RATE_LIMIT_REQUESTS_TOTAL.labels(
                workspace=ws,
                limit_type="websocket",
                status="allowed"
            )
            assert metric._value._value >= 2


class TestDoSProtectionMetrics:
    """Test DoS protection metrics."""

    def test_websocket_connections(self):
        """Test WebSocket connection metrics."""
        workspace = "test-ws"

        # Record connection attempts
        record_websocket_connection(workspace, "accepted")
        record_websocket_connection(workspace, "accepted")
        record_websocket_connection(workspace, "rejected_rate_limit")

        # Update active connections
        update_websocket_connections(workspace, 2)

        # Check counters
        accepted = DOS_WEBSOCKET_CONNECTIONS_TOTAL.labels(
            workspace=workspace,
            status="accepted"
        )
        assert accepted._value._value >= 2

        rejected = DOS_WEBSOCKET_CONNECTIONS_TOTAL.labels(
            workspace=workspace,
            status="rejected_rate_limit"
        )
        assert rejected._value._value >= 1

        # Check gauge
        active = DOS_WEBSOCKET_ACTIVE_CONNECTIONS.labels(workspace=workspace)
        assert active._value._value == 2

    def test_service_registrations(self):
        """Test service registration metrics."""
        workspace = "test-ws"

        # Record registrations
        record_service_registration(workspace, "accepted")
        record_service_registration(workspace, "rejected_quota")

        # Check metrics
        accepted = DOS_SERVICE_REGISTRATIONS_TOTAL.labels(
            workspace=workspace,
            status="accepted"
        )
        assert accepted._value._value >= 1

        rejected = DOS_SERVICE_REGISTRATIONS_TOTAL.labels(
            workspace=workspace,
            status="rejected_quota"
        )
        assert rejected._value._value >= 1

    def test_artifact_uploads(self):
        """Test artifact upload metrics."""
        workspace = "test-ws"

        # Record uploads with sizes
        record_artifact_upload(workspace, "accepted", 1024)
        record_artifact_upload(workspace, "accepted", 2048)
        record_artifact_upload(workspace, "rejected_size")

        # Check upload counter
        accepted = DOS_ARTIFACT_UPLOADS_TOTAL.labels(
            workspace=workspace,
            status="accepted"
        )
        assert accepted._value._value >= 2

        # Check bytes counter
        bytes_counter = DOS_ARTIFACT_UPLOAD_BYTES.labels(workspace=workspace)
        assert bytes_counter._value._value >= 3072  # 1024 + 2048

    def test_database_queries(self):
        """Test database query metrics."""
        workspace = "test-ws"

        # Record queries
        record_database_query(workspace, "select", "accepted")
        record_database_query(workspace, "insert", "accepted")
        record_database_query(workspace, "select", "rejected_rate_limit")

        # Check metrics
        select_accepted = DOS_DATABASE_QUERIES_TOTAL.labels(
            workspace=workspace,
            query_type="select",
            status="accepted"
        )
        assert select_accepted._value._value >= 1

        select_rejected = DOS_DATABASE_QUERIES_TOTAL.labels(
            workspace=workspace,
            query_type="select",
            status="rejected_rate_limit"
        )
        assert select_rejected._value._value >= 1

    def test_rpc_messages(self):
        """Test RPC message metrics."""
        workspace = "test-ws"

        # Record messages
        record_rpc_message(workspace, "accepted")
        record_rpc_message(workspace, "rejected_rate_limit")
        record_rpc_message(workspace, "rejected_size")

        # Check metrics
        accepted = DOS_RPC_MESSAGES_TOTAL.labels(
            workspace=workspace,
            status="accepted"
        )
        assert accepted._value._value >= 1

    def test_http_requests(self):
        """Test HTTP request metrics."""
        workspace = "test-ws"

        # Record requests
        record_http_request(workspace, "/api/services", "accepted")
        record_http_request(workspace, "/api/artifacts", "rejected_rate_limit")

        # Check metrics
        accepted = DOS_HTTP_REQUESTS_TOTAL.labels(
            workspace=workspace,
            endpoint="/api/services",
            status="accepted"
        )
        assert accepted._value._value >= 1


class TestResourceUsageMetrics:
    """Test resource usage metrics."""

    def test_memory_usage(self):
        """Test memory usage gauge."""
        RESOURCE_MEMORY_BYTES.labels(component="websocket").set(1024 * 1024)
        RESOURCE_MEMORY_BYTES.labels(component="rpc").set(512 * 1024)

        # Check values
        ws_mem = RESOURCE_MEMORY_BYTES.labels(component="websocket")
        assert ws_mem._value._value == 1024 * 1024

        rpc_mem = RESOURCE_MEMORY_BYTES.labels(component="rpc")
        assert rpc_mem._value._value == 512 * 1024

    def test_cpu_usage(self):
        """Test CPU usage gauge."""
        RESOURCE_CPU_USAGE.labels(component="artifact").set(45.5)

        metric = RESOURCE_CPU_USAGE.labels(component="artifact")
        assert metric._value._value == 45.5

    def test_queue_depth(self):
        """Test queue depth gauge."""
        RESOURCE_QUEUE_DEPTH.labels(queue_type="rpc").set(10)
        RESOURCE_QUEUE_DEPTH.labels(queue_type="artifact").set(5)

        rpc_queue = RESOURCE_QUEUE_DEPTH.labels(queue_type="rpc")
        assert rpc_queue._value._value == 10

    def test_bandwidth_tracking(self):
        """Test bandwidth counter."""
        RESOURCE_BANDWIDTH_BYTES.labels(
            direction="inbound",
            protocol="websocket"
        ).inc(1024)

        RESOURCE_BANDWIDTH_BYTES.labels(
            direction="outbound",
            protocol="http"
        ).inc(2048)

        # Check counters
        inbound = RESOURCE_BANDWIDTH_BYTES.labels(
            direction="inbound",
            protocol="websocket"
        )
        assert inbound._value._value >= 1024


class TestSystemHealthMetrics:
    """Test system health metrics."""

    def test_error_counter(self):
        """Test error counter."""
        HEALTH_ERRORS_TOTAL.labels(
            component="websocket",
            error_type="connection_failed"
        ).inc()

        metric = HEALTH_ERRORS_TOTAL.labels(
            component="websocket",
            error_type="connection_failed"
        )
        assert metric._value._value >= 1

    def test_latency_histogram(self):
        """Test latency histogram."""
        # Record latencies
        HEALTH_LATENCY_SECONDS.labels(
            component="rpc",
            operation="method_call"
        ).observe(0.05)

        HEALTH_LATENCY_SECONDS.labels(
            component="rpc",
            operation="method_call"
        ).observe(0.1)

        # Check histogram has samples
        metric = HEALTH_LATENCY_SECONDS.labels(
            component="rpc",
            operation="method_call"
        )
        assert metric._sum._value > 0

    def test_processing_time_summary(self):
        """Test processing time summary."""
        HEALTH_PROCESSING_TIME_SECONDS.labels(
            component="artifact",
            operation="upload"
        ).observe(2.5)

        metric = HEALTH_PROCESSING_TIME_SECONDS.labels(
            component="artifact",
            operation="upload"
        )
        assert metric._sum._value >= 2.5

    def test_circuit_breaker_state(self):
        """Test circuit breaker state gauge."""
        # States: 0=closed, 1=half-open, 2=open
        HEALTH_CIRCUIT_BREAKER_STATE.labels(component="database").set(0)
        HEALTH_CIRCUIT_BREAKER_STATE.labels(component="redis").set(2)

        db_state = HEALTH_CIRCUIT_BREAKER_STATE.labels(component="database")
        assert db_state._value._value == 0

        redis_state = HEALTH_CIRCUIT_BREAKER_STATE.labels(component="redis")
        assert redis_state._value._value == 2


class TestAttackDetectionMetrics:
    """Test attack detection metrics."""

    def test_attack_detection_events(self):
        """Test attack detection counter."""
        record_attack_detection("rate_limit", "medium")
        record_attack_detection("size_limit", "high")
        record_attack_detection("pattern", "critical")

        # Check metrics
        rate_limit = ATTACK_DETECTION_EVENTS.labels(
            attack_type="rate_limit",
            severity="medium"
        )
        assert rate_limit._value._value >= 1

    def test_blocked_ips(self):
        """Test blocked IPs gauge."""
        ATTACK_BLOCKED_IPS.set(5)
        assert ATTACK_BLOCKED_IPS._value._value == 5

    def test_suspicious_patterns(self):
        """Test suspicious pattern counter."""
        ATTACK_SUSPICIOUS_PATTERNS.labels(pattern_type="burst").inc()
        ATTACK_SUSPICIOUS_PATTERNS.labels(pattern_type="scan").inc()

        burst = ATTACK_SUSPICIOUS_PATTERNS.labels(pattern_type="burst")
        assert burst._value._value >= 1


class TestQuotaMetrics:
    """Test quota management metrics."""

    def test_quota_usage_percentage(self):
        """Test quota usage gauge."""
        workspace = "test-ws"

        update_quota_usage(workspace, "connections", 75.5, 100)
        update_quota_usage(workspace, "storage", 90.2, 1000)

        # Check usage
        conn_usage = QUOTA_USAGE_PERCENTAGE.labels(
            workspace=workspace,
            quota_type="connections"
        )
        assert conn_usage._value._value == 75.5

        # Check limit
        conn_limit = QUOTA_LIMIT_VALUE.labels(
            workspace=workspace,
            quota_type="connections"
        )
        assert conn_limit._value._value == 100

    def test_quota_violations(self):
        """Test quota violation counter."""
        workspace = "test-ws"

        record_quota_violation(workspace, "services")
        record_quota_violation(workspace, "services")
        record_quota_violation(workspace, "bandwidth")

        # Check violations
        services = QUOTA_VIOLATIONS_TOTAL.labels(
            workspace=workspace,
            quota_type="services"
        )
        assert services._value._value >= 2

    def test_quota_multi_workspace(self):
        """Test quota metrics across multiple workspaces."""
        workspaces = ["ws-a", "ws-b", "ws-c"]

        for idx, ws in enumerate(workspaces):
            usage = (idx + 1) * 25.0
            update_quota_usage(ws, "connections", usage, 100)

        # Each workspace should have independent metrics
        usage_a = QUOTA_USAGE_PERCENTAGE.labels(
            workspace="ws-a",
            quota_type="connections"
        )
        assert usage_a._value._value == 25.0

        usage_c = QUOTA_USAGE_PERCENTAGE.labels(
            workspace="ws-c",
            quota_type="connections"
        )
        assert usage_c._value._value == 75.0


class TestMetricsIntegration:
    """Test metrics integration scenarios."""

    def test_full_request_flow(self):
        """Test complete request flow with metrics."""
        workspace = "integration-ws"

        # 1. WebSocket connection
        record_websocket_connection(workspace, "accepted")
        update_websocket_connections(workspace, 1)

        # 2. Service registration
        record_service_registration(workspace, "accepted")

        # 3. RPC messages
        record_rpc_message(workspace, "accepted")
        record_rpc_message(workspace, "accepted")

        # 4. HTTP request
        record_http_request(workspace, "/api/services", "accepted")

        # 5. Update quota
        update_quota_usage(workspace, "connections", 10.0, 100)

        # Verify all metrics recorded
        ws_conn = DOS_WEBSOCKET_CONNECTIONS_TOTAL.labels(
            workspace=workspace,
            status="accepted"
        )
        assert ws_conn._value._value >= 1

        rpc_msg = DOS_RPC_MESSAGES_TOTAL.labels(
            workspace=workspace,
            status="accepted"
        )
        assert rpc_msg._value._value >= 2

    def test_attack_scenario(self):
        """Test attack detection and blocking scenario."""
        workspace = "attack-ws"

        # Simulate attack
        for _ in range(100):
            record_rate_limit_rejection(workspace, "websocket", "rate_exceeded")

        # Detect attack
        record_attack_detection("rate_limit", "high")

        # Block connections
        for _ in range(50):
            record_websocket_connection(workspace, "rejected_rate_limit")

        # Update blocked IPs
        ATTACK_BLOCKED_IPS.set(1)

        # Verify metrics
        rejections = RATE_LIMIT_REJECTIONS_TOTAL.labels(
            workspace=workspace,
            limit_type="websocket",
            reason="rate_exceeded"
        )
        assert rejections._value._value >= 100

        attacks = ATTACK_DETECTION_EVENTS.labels(
            attack_type="rate_limit",
            severity="high"
        )
        assert attacks._value._value >= 1

    def test_quota_exhaustion_scenario(self):
        """Test quota exhaustion scenario."""
        workspace = "quota-ws"

        # Gradually increase usage
        for usage in [25, 50, 75, 90, 100]:
            update_quota_usage(workspace, "connections", usage, 100)

        # Record violations
        record_quota_violation(workspace, "connections")
        record_quota_violation(workspace, "connections")

        # Reject new connections
        record_websocket_connection(workspace, "rejected_quota")

        # Verify final state
        usage = QUOTA_USAGE_PERCENTAGE.labels(
            workspace=workspace,
            quota_type="connections"
        )
        assert usage._value._value == 100

        violations = QUOTA_VIOLATIONS_TOTAL.labels(
            workspace=workspace,
            quota_type="connections"
        )
        assert violations._value._value >= 2


class TestMetricsExport:
    """Test metrics export and Prometheus compatibility."""

    def test_metrics_registered(self):
        """Test that all metrics are registered with Prometheus."""
        # Get all registered metrics
        metric_names = set()
        for collector in REGISTRY.collect():
            metric_names.add(collector.name)

        # Check key metrics are present
        # Note: Prometheus may convert underscores to dashes in metric names
        expected_metrics = [
            "hypha_rate_limit_requests",
            "hypha_dos_websocket_connections",
            "hypha_resource_memory_bytes",
            "hypha_health_errors",
            "hypha_attack_detection_events",
            "hypha_quota_usage_percentage",
        ]

        for metric in expected_metrics:
            # Check if metric exists (with or without _total suffix)
            found = any(
                name == metric or
                name == f"{metric}_total" or
                name.startswith(f"{metric}_")
                for name in metric_names
            )
            assert found, f"Metric {metric} not registered. Available: {sorted(metric_names)}"

    def test_label_consistency(self):
        """Test that metrics maintain consistent label sets."""
        workspace = "label-test-ws"

        # Record metrics with same labels
        record_rate_limit_request(workspace, "websocket", "allowed")
        record_rate_limit_rejection(workspace, "websocket", "rate_exceeded")

        # Both should use same workspace label
        requests = RATE_LIMIT_REQUESTS_TOTAL.labels(
            workspace=workspace,
            limit_type="websocket",
            status="allowed"
        )
        rejections = RATE_LIMIT_REJECTIONS_TOTAL.labels(
            workspace=workspace,
            limit_type="websocket",
            reason="rate_exceeded"
        )

        assert requests._value._value >= 1
        assert rejections._value._value >= 1
