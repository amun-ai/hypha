"""Centralized Prometheus metrics for Hypha server.

This module provides a unified metrics interface for monitoring DoS protection,
rate limiting, resource usage, and system health across all Hypha components.

Metrics Categories:
- Rate Limiting: Request rate tracking per workspace/client
- DoS Protection: Attack detection and mitigation
- Resource Usage: Memory, CPU, connections, bandwidth
- System Health: Error rates, latency, queue depths
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Optional


# ==============================================================================
# Rate Limiting Metrics
# ==============================================================================

RATE_LIMIT_REQUESTS_TOTAL = Counter(
    "hypha_rate_limit_requests_total",
    "Total requests processed by rate limiter",
    ["workspace", "limit_type", "status"]
)

RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    "hypha_rate_limit_rejections_total",
    "Total requests rejected by rate limiter",
    ["workspace", "limit_type", "reason"]
)

RATE_LIMIT_CURRENT_RATE = Gauge(
    "hypha_rate_limit_current_rate",
    "Current request rate per workspace",
    ["workspace", "limit_type"]
)

RATE_LIMIT_TOKEN_BUCKET = Gauge(
    "hypha_rate_limit_token_bucket",
    "Current tokens in bucket for rate limiter",
    ["workspace", "limit_type"]
)


# ==============================================================================
# DoS Protection Metrics
# ==============================================================================

DOS_WEBSOCKET_CONNECTIONS_TOTAL = Counter(
    "hypha_dos_websocket_connections_total",
    "WebSocket connection attempts",
    ["workspace", "status"]  # status: accepted, rejected_rate_limit, rejected_quota
)

DOS_WEBSOCKET_ACTIVE_CONNECTIONS = Gauge(
    "hypha_dos_websocket_active_connections",
    "Current active WebSocket connections",
    ["workspace"]
)

DOS_SERVICE_REGISTRATIONS_TOTAL = Counter(
    "hypha_dos_service_registrations_total",
    "Service registration attempts",
    ["workspace", "status"]  # status: accepted, rejected_quota, rejected_rate_limit
)

DOS_ARTIFACT_UPLOADS_TOTAL = Counter(
    "hypha_dos_artifact_uploads_total",
    "Artifact upload attempts",
    ["workspace", "status"]  # status: accepted, rejected_size, rejected_rate_limit
)

DOS_ARTIFACT_UPLOAD_BYTES = Counter(
    "hypha_dos_artifact_upload_bytes_total",
    "Total bytes uploaded via artifacts",
    ["workspace"]
)

DOS_DATABASE_QUERIES_TOTAL = Counter(
    "hypha_dos_database_queries_total",
    "Database query attempts",
    ["workspace", "query_type", "status"]  # status: accepted, rejected_rate_limit
)

DOS_RPC_MESSAGES_TOTAL = Counter(
    "hypha_dos_rpc_messages_total",
    "RPC message processing attempts",
    ["workspace", "status"]  # status: accepted, rejected_rate_limit, rejected_size
)

DOS_HTTP_REQUESTS_TOTAL = Counter(
    "hypha_dos_http_requests_total",
    "HTTP request attempts",
    ["workspace", "endpoint", "status"]  # status: accepted, rejected_rate_limit
)


# ==============================================================================
# Resource Usage Metrics
# ==============================================================================

RESOURCE_MEMORY_BYTES = Gauge(
    "hypha_resource_memory_bytes",
    "Memory usage in bytes",
    ["component"]  # component: websocket, rpc, artifact, database
)

RESOURCE_CPU_USAGE = Gauge(
    "hypha_resource_cpu_usage",
    "CPU usage percentage",
    ["component"]
)

RESOURCE_ACTIVE_TASKS = Gauge(
    "hypha_resource_active_tasks",
    "Number of active async tasks",
    ["component"]
)

RESOURCE_QUEUE_DEPTH = Gauge(
    "hypha_resource_queue_depth",
    "Depth of processing queues",
    ["queue_type"]  # queue_type: rpc, artifact, database
)

RESOURCE_BANDWIDTH_BYTES = Counter(
    "hypha_resource_bandwidth_bytes_total",
    "Total bandwidth consumed",
    ["direction", "protocol"]  # direction: inbound/outbound, protocol: websocket/http
)


# ==============================================================================
# System Health Metrics
# ==============================================================================

HEALTH_ERRORS_TOTAL = Counter(
    "hypha_health_errors_total",
    "Total errors encountered",
    ["component", "error_type"]
)

HEALTH_LATENCY_SECONDS = Histogram(
    "hypha_health_latency_seconds",
    "Request latency in seconds",
    ["component", "operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

HEALTH_PROCESSING_TIME_SECONDS = Summary(
    "hypha_health_processing_time_seconds",
    "Processing time summary",
    ["component", "operation"]
)

HEALTH_CIRCUIT_BREAKER_STATE = Gauge(
    "hypha_health_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["component"]
)


# ==============================================================================
# Attack Detection Metrics
# ==============================================================================

ATTACK_DETECTION_EVENTS = Counter(
    "hypha_attack_detection_events_total",
    "Detected attack events",
    ["attack_type", "severity"]  # attack_type: rate_limit, size_limit, pattern
)

ATTACK_BLOCKED_IPS = Gauge(
    "hypha_attack_blocked_ips",
    "Number of currently blocked IP addresses",
    []
)

ATTACK_SUSPICIOUS_PATTERNS = Counter(
    "hypha_attack_suspicious_patterns_total",
    "Suspicious activity patterns detected",
    ["pattern_type"]
)


# ==============================================================================
# Workspace Quota Metrics
# ==============================================================================

QUOTA_USAGE_PERCENTAGE = Gauge(
    "hypha_quota_usage_percentage",
    "Quota usage as percentage",
    ["workspace", "quota_type"]  # quota_type: connections, services, storage, bandwidth
)

QUOTA_LIMIT_VALUE = Gauge(
    "hypha_quota_limit_value",
    "Configured quota limit",
    ["workspace", "quota_type"]
)

QUOTA_VIOLATIONS_TOTAL = Counter(
    "hypha_quota_violations_total",
    "Total quota violations",
    ["workspace", "quota_type"]
)


# ==============================================================================
# Helper Functions
# ==============================================================================

def record_rate_limit_request(
    workspace: str,
    limit_type: str,
    status: str
):
    """Record a rate limit request.

    Args:
        workspace: Workspace identifier
        limit_type: Type of rate limit (e.g., 'websocket', 'rpc', 'artifact')
        status: Request status ('allowed', 'rejected')
    """
    RATE_LIMIT_REQUESTS_TOTAL.labels(
        workspace=workspace,
        limit_type=limit_type,
        status=status
    ).inc()


def record_rate_limit_rejection(
    workspace: str,
    limit_type: str,
    reason: str
):
    """Record a rate limit rejection.

    Args:
        workspace: Workspace identifier
        limit_type: Type of rate limit
        reason: Rejection reason ('rate_exceeded', 'quota_exceeded', 'burst_exceeded')
    """
    RATE_LIMIT_REJECTIONS_TOTAL.labels(
        workspace=workspace,
        limit_type=limit_type,
        reason=reason
    ).inc()


def update_current_rate(
    workspace: str,
    limit_type: str,
    rate: float
):
    """Update current request rate gauge.

    Args:
        workspace: Workspace identifier
        limit_type: Type of rate limit
        rate: Current rate (requests per second)
    """
    RATE_LIMIT_CURRENT_RATE.labels(
        workspace=workspace,
        limit_type=limit_type
    ).set(rate)


def record_websocket_connection(
    workspace: str,
    status: str
):
    """Record WebSocket connection attempt.

    Args:
        workspace: Workspace identifier
        status: Connection status ('accepted', 'rejected_rate_limit', 'rejected_quota')
    """
    DOS_WEBSOCKET_CONNECTIONS_TOTAL.labels(
        workspace=workspace,
        status=status
    ).inc()


def update_websocket_connections(
    workspace: str,
    count: int
):
    """Update active WebSocket connections gauge.

    Args:
        workspace: Workspace identifier
        count: Number of active connections
    """
    DOS_WEBSOCKET_ACTIVE_CONNECTIONS.labels(
        workspace=workspace
    ).set(count)


def record_service_registration(
    workspace: str,
    status: str
):
    """Record service registration attempt.

    Args:
        workspace: Workspace identifier
        status: Registration status ('accepted', 'rejected_quota', 'rejected_rate_limit')
    """
    DOS_SERVICE_REGISTRATIONS_TOTAL.labels(
        workspace=workspace,
        status=status
    ).inc()


def record_artifact_upload(
    workspace: str,
    status: str,
    size_bytes: Optional[int] = None
):
    """Record artifact upload attempt.

    Args:
        workspace: Workspace identifier
        status: Upload status ('accepted', 'rejected_size', 'rejected_rate_limit')
        size_bytes: Size of uploaded artifact in bytes (if accepted)
    """
    DOS_ARTIFACT_UPLOADS_TOTAL.labels(
        workspace=workspace,
        status=status
    ).inc()

    if size_bytes is not None and status == "accepted":
        DOS_ARTIFACT_UPLOAD_BYTES.labels(
            workspace=workspace
        ).inc(size_bytes)


def record_database_query(
    workspace: str,
    query_type: str,
    status: str
):
    """Record database query attempt.

    Args:
        workspace: Workspace identifier
        query_type: Type of query ('select', 'insert', 'update', 'delete')
        status: Query status ('accepted', 'rejected_rate_limit')
    """
    DOS_DATABASE_QUERIES_TOTAL.labels(
        workspace=workspace,
        query_type=query_type,
        status=status
    ).inc()


def record_rpc_message(
    workspace: str,
    status: str
):
    """Record RPC message attempt.

    Args:
        workspace: Workspace identifier
        status: Message status ('accepted', 'rejected_rate_limit', 'rejected_size')
    """
    DOS_RPC_MESSAGES_TOTAL.labels(
        workspace=workspace,
        status=status
    ).inc()


def record_http_request(
    workspace: str,
    endpoint: str,
    status: str
):
    """Record HTTP request attempt.

    Args:
        workspace: Workspace identifier
        endpoint: Request endpoint path
        status: Request status ('accepted', 'rejected_rate_limit')
    """
    DOS_HTTP_REQUESTS_TOTAL.labels(
        workspace=workspace,
        endpoint=endpoint,
        status=status
    ).inc()


def record_attack_detection(
    attack_type: str,
    severity: str
):
    """Record detected attack event.

    Args:
        attack_type: Type of attack detected
        severity: Severity level ('low', 'medium', 'high', 'critical')
    """
    ATTACK_DETECTION_EVENTS.labels(
        attack_type=attack_type,
        severity=severity
    ).inc()


def update_quota_usage(
    workspace: str,
    quota_type: str,
    usage_percentage: float,
    limit_value: Optional[float] = None
):
    """Update workspace quota usage metrics.

    Args:
        workspace: Workspace identifier
        quota_type: Type of quota
        usage_percentage: Current usage as percentage (0-100)
        limit_value: Configured limit value (optional)
    """
    QUOTA_USAGE_PERCENTAGE.labels(
        workspace=workspace,
        quota_type=quota_type
    ).set(usage_percentage)

    if limit_value is not None:
        QUOTA_LIMIT_VALUE.labels(
            workspace=workspace,
            quota_type=quota_type
        ).set(limit_value)


def record_quota_violation(
    workspace: str,
    quota_type: str
):
    """Record quota violation.

    Args:
        workspace: Workspace identifier
        quota_type: Type of quota violated
    """
    QUOTA_VIOLATIONS_TOTAL.labels(
        workspace=workspace,
        quota_type=quota_type
    ).inc()
