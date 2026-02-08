# Protocol Adapter Security Vulnerabilities Report

## Executive Summary

This report documents critical security vulnerabilities discovered in the MCP (Model Context Protocol) and A2A (Agent-to-Agent) protocol adapters in Hypha. These adapters convert Hypha services to external protocol formats and expose them via HTTP endpoints.

**Total Vulnerabilities Found:** 8 (4 HIGH, 3 MEDIUM, 1 LOW severity)

---

## Vulnerability Details

### V16: MCP Session Cache Poisoning (HIGH SEVERITY)

**Location:** `hypha/mcp.py:1171-1268` (MCPRoutingMiddleware._get_or_create_mcp_app)

**Description:**
The MCP adapter caches created MCP apps using a simple key of `f"{workspace}/{service_id}"` without including user context or validating permissions on cache hits. This allows:
1. User A registers an MCP service in workspace A
2. Service is cached with User A's workspace interface
3. User B accesses the cached app (cache hit)
4. User B may inherit User A's workspace interface and permissions

**Code:**
```python
# Line 1181
cache_key = f"{workspace}/{service_id}"

if cache_key in self.mcp_app_cache:
    logger.debug(f"Using cached MCP app for {cache_key}")
    cache_entry = self.mcp_app_cache[cache_key]
    return cache_entry["app"] if isinstance(cache_entry, dict) else cache_entry
```

**Impact:**
- Permission elevation through cached workspace interfaces
- Cross-user session leakage
- Unauthorized access to workspace resources

**Recommendation:**
1. Include user ID/session ID in cache key
2. Validate user permissions before returning cached app
3. Invalidate cache when permissions change

---

### V17: MCP Workspace Interface Leak (MEDIUM SEVERITY)

**Location:** `hypha/mcp.py:1190-1229` (SSE connection caching)

**Description:**
When handling SSE (Server-Sent Events) connections (`is_sse=True`), the code keeps workspace interfaces alive by storing them in the cache:

```python
# Line 1225-1229
# Cache the app AND the API connection for SSE
self.mcp_app_cache[cache_key] = {
    "app": mcp_app,
    "api": api  # Keep the API connection alive for SSE
}
```

These interfaces are never cleaned up automatically and persist even after:
- Service is unregistered
- User permissions are revoked
- Session expires

**Impact:**
- Resource leakage (workspace interfaces stay open indefinitely)
- Unauthorized access after permission revocation
- Long-lived connections maintain stale permissions

**Recommendation:**
1. Implement automatic timeout for cached workspace interfaces
2. Clean up cache when services are unregistered
3. Validate workspace interface is still valid before reuse
4. Add periodic cleanup of stale cache entries

---

### V18: MCP Cross-Workspace Service Access (HIGH SEVERITY)

**Location:** `hypha/mcp.py:1206-1255` (service ID construction)

**Description:**
The MCP routing middleware constructs `full_service_id` using the workspace from the URL path without validating that the current user has permission to access services in that workspace:

```python
# Extract workspace from URL
workspace = path_params["workspace"]

# Construct service ID with URL workspace (not user's workspace!)
if "/" in service_id_without_app:
    full_service_id = service_id_without_app + app_id_suffix
elif ":" in service_id_without_app:
    full_service_id = f"{workspace}/{service_id_without_app}{app_id_suffix}"
else:
    # Use wildcard for client ID
    full_service_id = f"{workspace}/*:{service_id_without_app}{app_id_suffix}"
```

An attacker can craft URLs to access services in other workspaces:
- URL: `/{victim_workspace}/mcp/{service_id}/mcp`
- The code trusts `victim_workspace` from URL
- Permission check happens later and may be insufficient

**Impact:**
- Cross-workspace information disclosure
- Unauthorized service invocation
- Workspace isolation bypass

**Recommendation:**
1. Validate URL workspace matches user's current workspace or permissions
2. Don't construct service IDs from untrusted URL parameters
3. Explicit permission check at routing level before service access
4. Log security events for cross-workspace access attempts

---

### V19: A2A Context Filtering Bypass (HIGH SEVERITY)

**Location:** `hypha/a2a.py:145-153` (HyphaAgentExecutor._filter_context_for_hypha)

**Description:**
The A2A adapter unconditionally returns `None` for the context when calling Hypha services, completely losing all security context:

```python
def _filter_context_for_hypha(self, context: RequestContext) -> Optional[Dict]:
    """Filter RequestContext to only include serializable data for Hypha."""
    try:
        # For now, return None to avoid serialization issues
        # In the future, we could extract specific fields from context
        return None  # ← ALL SECURITY CONTEXT LOST!
    except Exception as e:
        logger.exception(f"Error filtering context: {e}")
        return None
```

This means services called through A2A receive no information about:
- Who is calling them (user identity)
- What workspace the call originated from
- What permissions the caller has

**Impact:**
- Complete loss of authentication/authorization context
- Services cannot validate caller permissions
- No audit trail for who invoked the service
- Workspace isolation bypass

**Recommendation:**
1. Extract and pass user info, workspace, and permissions from RequestContext
2. Create minimal security context dictionary with required fields
3. Handle serialization properly without losing security context
4. NEVER return None for security-critical context

---

### V20: A2A Agent Card Execution Without Context (MEDIUM SEVERITY)

**Location:** `hypha/a2a.py:186-202, 586-604` (agent_card callable handling)

**Description:**
If a service registers with a callable `agent_card` function instead of static data, the function is executed without any context validation:

```python
# Line 191-202
if callable(agent_card_data):
    logger.info("agent_card is callable, calling it...")
    if inspect.iscoroutinefunction(agent_card_data):
        agent_card_data = await agent_card_data()  # Executed without context!
    else:
        agent_card_data = agent_card_data()
        if asyncio.isfuture(agent_card_data) or inspect.iscoroutine(agent_card_data):
            agent_card_data = await agent_card_data
```

A malicious service can:
1. Register with a callable agent_card
2. Have arbitrary code execute when A2A app is created
3. Access workspace interfaces or server internals
4. Leak data or perform unauthorized actions

**Impact:**
- Arbitrary code execution with server privileges
- Workspace interface access without permission checks
- Data exfiltration through callable side effects

**Recommendation:**
1. Disallow callable agent_card - require static data only
2. If callables are necessary, execute in sandboxed environment
3. Pass and validate context before executing callables
4. Limit what callables can access

---

### V21: A2A Workspace Extraction Bypass (HIGH SEVERITY)

**Location:** `hypha/a2a.py:398-410` (A2A request routing)

**Description:**
Similar to V18, the A2A routing middleware extracts workspace from the URL path and trusts it without validation:

```python
# Line 398-399
workspace = path_params["workspace"]
service_id_path = path_params["service_id"]

# Line 458
full_service_id = f"{workspace}/{service_id}"
```

Attacker can specify any workspace in the URL to access services across workspace boundaries.

**Impact:**
- Cross-workspace unauthorized access
- Workspace isolation violation
- Service enumeration across workspaces

**Recommendation:**
1. Validate URL workspace against user permissions
2. Reject cross-workspace requests to protected services
3. Use user's authenticated workspace instead of URL parameter
4. Log cross-workspace access attempts for security monitoring

---

### V27: MCP Service Info Information Leakage (LOW SEVERITY)

**Location:** MCP info route handling (implied from middleware structure)

**Description:**
The MCP info route may leak service information even for services the user shouldn't access, including:
- Service existence (enables enumeration)
- Service configuration and capabilities
- Tool/resource schemas

**Impact:**
- Information disclosure about protected services
- Service enumeration for reconnaissance
- Metadata leakage

**Recommendation:**
1. Enforce same permissions on info route as service access
2. Return 404 (not 403) for unauthorized access to prevent enumeration
3. Don't leak service details for protected services

---

### V28: A2A Info Information Leakage (LOW SEVERITY)

**Location:** `hypha/a2a.py:529-742` (_send_agent_info)

**Description:**
The A2A info route (`/{workspace}/a2a-info/{service_id}`) may leak agent card information for services the user shouldn't access:

```python
# Line 584-594
if "agent_card" in service:
    agent_card_data = service["agent_card"]
    # Handle callable agent_card
    if callable(agent_card_data):
        if inspect.iscoroutinefunction(agent_card_data):
            agent_card_data = await agent_card_data()
        # ... returns agent card without permission validation
```

**Impact:**
- Agent capabilities and skills disclosure
- Service metadata leakage
- Protected service enumeration

**Recommendation:**
1. Validate user has permission to access service before returning info
2. Return 404 for unauthorized access
3. Don't execute callable agent_card without permission check

---

## Summary of Recommendations

### Immediate Actions (HIGH Priority):

1. **Fix Context Loss (V19):** Restore security context in A2A adapter
2. **Validate Workspace from URL (V18, V21):** Don't trust workspace parameter
3. **User-Specific Caching (V16):** Include user context in cache key

### Important Actions (MEDIUM Priority):

4. **Workspace Interface Cleanup (V17):** Implement automatic cleanup
5. **Restrict Callable Execution (V20):** Validate or disallow callable agent_card

### Good Practice (LOW Priority):

6. **Info Route Permissions (V22, V23):** Enforce permissions on info endpoints

---

## Testing Strategy

Tests have been created in `tests/test_protocol_adapter_security.py` but require running infrastructure (PostgreSQL, Redis). The tests document:
- Exploitation scenarios for each vulnerability
- Expected vs actual behavior
- Security impact

To run tests:
```bash
pytest tests/test_protocol_adapter_security.py -v
```

---

## Related Security Patterns

These vulnerabilities follow similar patterns to previously identified issues:

- **V10-V14:** Cross-workspace access by trusting user-provided IDs
  - **Lesson:** Always validate workspace from authenticated context, not parameters

- **V1:** Service ID workspace injection
  - **Lesson:** Reconstruct IDs from trusted sources, don't trust user input

The MCP and A2A adapters should apply the same security principles as the core Hypha services.
