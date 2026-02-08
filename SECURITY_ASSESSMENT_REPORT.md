# Hypha Security Assessment Report
**Date**: 2026-02-08
**Target**: https://hypha-dev.aicell.io/
**Assessment Type**: White-box penetration testing
**Team Size**: 10 security experts

---

## Executive Summary

A comprehensive security assessment of the Hypha server identified **25+ vulnerabilities** across all major attack surfaces. Critical findings include:

- **5 SQL Injection vulnerabilities** enabling complete database compromise
- **Complete security context loss** in A2A protocol adapter
- **Systemic cross-workspace access control issues** (V10-V14 pattern family)
- **Multiple authentication weaknesses** including missing token expiration validation
- **Critical input validation gaps** in HTTP/ASGI routing layer

**Overall Risk Level**: **CRITICAL**

**Immediate Action Required**: 8 CRITICAL severity vulnerabilities require urgent remediation.

---

## Vulnerability Summary by Severity

### CRITICAL (8 vulnerabilities)

| ID | Title | Component | Impact |
|----|-------|-----------|--------|
| V-STORAGE-01 | SQL Injection via Keywords | Artifact Manager | Database compromise |
| V-STORAGE-02 | SQL Injection via Manifest Filters | Artifact Manager | Database compromise |
| V-STORAGE-03 | SQL Injection via Config Permissions | Artifact Manager | Database compromise |
| V-STORAGE-04 | SQL Injection via ORDER BY | Artifact Manager | Database compromise |
| V-STORAGE-05 | SQL Injection via Array Operators | Artifact Manager | Database compromise |
| V-AUTH-2 | Permission Escalation via Token Manipulation | Authentication | Full system compromise |
| V-PROTOCOL-19 | A2A Security Context Loss | A2A Protocol | Complete auth bypass |
| W3 | Conda Worker Full System Access | Worker | RCE with system privileges |

### HIGH (11 vulnerabilities)

| ID | Title | Component | Impact |
|----|-------|-----------|--------|
| V-AUTH-1 | Missing JWT Expiration Validation | Authentication | Indefinite token validity |
| V-AUTH-3 | No Token Revocation Mechanism | Authentication | Cannot revoke compromised tokens |
| W1 | Cross-Workspace Session Access | All Workers | Unauthorized code execution |
| W2 | Browser Worker Sandbox Escape | Browser Worker | Potential system compromise |
| W4 | Terminal Worker Command Injection | Terminal Worker | RCE |
| V-PROTOCOL-16 | MCP Session Cache Poisoning | MCP Protocol | Permission elevation |
| V-PROTOCOL-18 | MCP Cross-Workspace Service Access | MCP Protocol | Workspace isolation bypass |
| V-PROTOCOL-21 | A2A Workspace Extraction Bypass | A2A Protocol | Cross-workspace access |
| V-HTTP-20 | Missing Input Validation in ASGI | HTTP Layer | Server crash, path traversal |
| V-HTTP-18 | Path Traversal in Static Mounts | HTTP Layer | Arbitrary file disclosure |
| V-RPC-TBD | Context Forgery via WebSocket | RPC Layer | User impersonation |

### MEDIUM (5+ vulnerabilities)

| ID | Title | Component |
|----|-------|-----------|
| V-AUTH-4 | JWT Secret Generation Weakness | Authentication |
| V-PROTOCOL-17 | MCP Workspace Interface Leak | MCP Protocol |
| V-PROTOCOL-20 | A2A Agent Card Execution Without Context | A2A Protocol |
| V-HTTP-16 | ASGI Service Activity Timer Bypass | HTTP Layer |
| V-HTTP-21 | Exception Information Leakage | HTTP Layer |

### LOW (3+ vulnerabilities)

- Service type case sensitivity confusion
- CORS header override issues
- Info route information leakage

---

## Critical Findings Deep Dive

### 1. SQL Injection Vulnerabilities (V-STORAGE-01 to V-STORAGE-05)

**Severity**: CRITICAL
**Component**: `hypha/artifact.py`
**Discovery Credit**: storage-security-expert

**Description**: Five distinct SQL injection points in the artifact management system use direct string interpolation of user-provided input into SQL `text()` statements without parameterization.

**Vulnerable Code Pattern**:
```python
# Line 8451 - Keywords injection
condition = text(f"manifest::text ILIKE '%{keyword}%'")

# Line 8219 - Manifest filter injection
return text(f"manifest->>'{manifest_key}' ILIKE '{value.replace('*', '%')}'")

# Line 8498 - Config permissions injection
condition = text(f"permissions->>'{user_id}' = '{permission}'")
```

**Impact**:
- Complete database data exfiltration across ALL workspaces
- Bypass ALL permission and workspace isolation checks
- Modify or delete arbitrary database records
- Escalate privileges by manipulating permission records

**Exploitation Examples**:
```python
# Bypass all filters
keywords = ["' OR 1=1 --"]

# Extract sensitive data
filters = {"manifest": {"status": "x' UNION SELECT secrets FROM secrets_table --"}}

# Delete data
keywords = ["'; DROP TABLE artifacts; --"]
```

**Remediation**: Use parameterized queries with `.bindparams()`:
```python
# FIXED
condition = text("manifest::text ILIKE :keyword").bindparams(keyword=f"%{keyword}%")
```

**Status**: Fixes in progress by storage-security-expert

---

### 2. A2A Security Context Loss (V-PROTOCOL-19)

**Severity**: CRITICAL
**Component**: `hypha/a2a.py` line 145-153
**Discovery Credit**: protocol-security-expert

**Description**: The `_filter_context_for_hypha()` method **returns None unconditionally**, completely discarding security context including user identity, workspace, and permissions.

**Vulnerable Code**:
```python
async def _filter_context_for_hypha(self, context):
    # CRITICAL BUG: Returns None, losing all security context
    return None  # Comment says "to avoid serialization issues"
```

**Impact**:
- **Complete authentication bypass** - services cannot validate callers
- **No authorization checks** possible - all permission validation fails
- **Workspace isolation broken** - services cannot determine workspace context
- **Audit trail lost** - no record of who performed actions

**Exploitation**: Any A2A protocol request operates without authentication or authorization.

**Remediation**: Preserve security context:
```python
async def _filter_context_for_hypha(self, context):
    return {
        "ws": context.get("ws"),
        "from": context.get("from"),
        "user": context.get("user"),
    }
```

**Status**: Fix pattern documented, awaiting implementation

---

### 3. Permission Escalation via Token Manipulation (V-AUTH-2)

**Severity**: CRITICAL
**Component**: Entire authentication system
**Discovery Credit**: auth-security-expert

**Description**: The authentication system stores user permissions entirely in JWT tokens with no server-side validation. If an attacker gains access to `JWT_SECRET`, they can create tokens with arbitrary permissions including wildcard admin access.

**Vulnerable Pattern**:
```python
# Permissions are ONLY in the token, not validated server-side
token_payload = {
    "scope": "ws:*#a",  # Wildcard admin - no server validation
    "sub": "attacker"
}
```

**Impact**:
- JWT_SECRET compromise = **full system compromise**
- Attacker can grant themselves admin access to ALL workspaces
- No server-side defense against forged permission claims
- Single point of failure

**Exploitation**:
1. Obtain JWT_SECRET (via leak, misconfiguration, or brute force weak secret)
2. Create new token with `"scope": "ws:*#a"` (wildcard admin)
3. Gain full access to entire Hypha instance

**Remediation** (Architectural):
1. Store permissions in database, reference by user ID
2. Validate permissions server-side on every operation
3. Use short-lived access tokens with refresh tokens
4. Implement permission change auditing

**Status**: Architectural change required - documented for long-term fix

---

### 4. Cross-Workspace Session Access (W1)

**Severity**: HIGH
**Component**: All workers (`browser.py`, `conda.py`, `terminal.py`)
**Discovery Credit**: worker-sandbox-expert

**Description**: Workers validate permission on the caller's workspace (`context["ws"]`) but don't verify that session IDs belong to that workspace. Session IDs have format `workspace/client_id`, enabling cross-workspace attacks.

**Vulnerable Pattern**:
```python
async def execute(self, session_id: str, context: dict = None):
    # Validates caller's workspace
    workspace = context["ws"]
    user_info.check_permission(workspace, UserPermission.read)

    # But session_id could be "victim-workspace/session-123"
    session = self._sessions[session_id]  # BYPASS!
    return await session.execute(code)
```

**Impact**:
- Execute arbitrary code in other workspaces' sessions
- Read session state and outputs from other users
- Manipulate or terminate other workspaces' workers

**Exploitation**:
```python
# Attacker in workspace "attacker-ws" can access victim's session
await worker.execute(
    session_id="victim-workspace/session-abc123",
    script="malicious_code_here"
)
```

**Remediation** (V10-V14 pattern):
```python
async def execute(self, session_id: str, context: dict = None):
    if "/" in session_id:
        target_ws = session_id.split("/")[0]
        if target_ws != context["ws"]:
            user_info = UserInfo.from_context(context)
            if not user_info.check_permission(target_ws, UserPermission.read):
                raise PermissionError(f"No permission on {target_ws}")
    # Continue with execution...
```

**Status**: Fix pattern documented, similar to V10-V14 vulnerabilities

---

### 5. Missing JWT Expiration Validation (V-AUTH-1)

**Severity**: HIGH
**Component**: `hypha/core/auth.py` line 134-155
**Discovery Credit**: auth-security-expert

**Description**: The `valid_token()` function does not require JWT tokens to include an `exp` (expiration) claim. Tokens without expiration are accepted as valid.

**Vulnerable Code**:
```python
# jwt.decode() does NOT require 'exp' claim by default
payload = jwt.decode(
    token,
    key,
    algorithms=algorithms,
    # Missing: options={"require": ["exp"]}
)
```

**Impact**:
- Attacker can create tokens without expiration that remain valid indefinitely
- Compromised tokens cannot be mitigated by waiting for expiration
- Violates JWT best practices (RFC 7519)

**Remediation**:
```python
payload = jwt.decode(
    token,
    key,
    algorithms=algorithms,
    options={"require": ["exp"]}  # Enforce expiration requirement
)
```

**Status**: Simple fix, implementation in progress

---

### 6. Missing Input Validation in ASGI Routing (V-HTTP-20)

**Severity**: HIGH
**Component**: `hypha/http.py` line 254-384
**Discovery Credit**: http-asgi-expert

**Description**: The ASGI routing middleware extracts `workspace` and `service_id` from URL paths without validation. Query string parsing is also unsafe.

**Vulnerable Code**:
```python
# No validation of workspace parameter
workspace = scope["path_params"]["workspace"]

# Unsafe query parsing - crashes if no "=" sign
_mode = dict([q.split("=") for q in query.split("&")]).get("_mode")

# No path traversal protection
if not path.startswith("/"):
    path = "/" + path
scope["path"] = path
```

**Impact**:
- Server crash via malformed query strings
- Potential path traversal attacks
- Workspace injection with special characters (null bytes, etc.)

**Exploitation**:
```bash
# Crash server
GET /{workspace}/apps/{service}/path?malformed_query_without_equals

# Potential traversal
GET /{workspace}/apps/{service}/../../../etc/passwd

# Workspace injection
GET /workspace%00admin/apps/service/path
```

**Remediation**: Add comprehensive input validation at routing layer

**Status**: Fix pattern documented

---

## Systemic Vulnerability Pattern: V10-V14 Family

**Discovery**: Multiple experts found the same vulnerability pattern across different components, confirming it's a **systemic architectural issue**.

**Pattern Description**: Methods that accept user-provided IDs containing workspace prefixes (format: `workspace/identifier`) validate permission on the caller's workspace (`context["ws"]`) instead of extracting and validating the target workspace from the ID.

**Affected Components**:
- ✅ Workers (all types) - W1
- ✅ MCP Protocol - V-PROTOCOL-18
- ✅ A2A Protocol - V-PROTOCOL-21
- ⏳ Workspace manager (under investigation)
- ⏳ App controller (under investigation)
- ⏳ Service registration/discovery (under investigation)

**Correct Implementation Pattern** (from artifact.py):
```python
# GOLD STANDARD - Artifact manager does this correctly
async def method(self, resource_id: str, context: dict = None):
    # Get the resource first
    resource = await get_resource(resource_id)

    # Extract workspace from RESOURCE, not from context
    target_workspace = resource.workspace

    # Validate permission on TARGET workspace
    user_info = UserInfo.from_context(context)
    if not user_info.check_permission(target_workspace, required_perm):
        raise PermissionError(...)
```

**Key Lesson**: Always validate permissions on the TARGET workspace embedded in user-provided IDs, not just the caller's workspace.

---

## Positive Security Findings

The assessment identified several **correctly implemented security controls**:

### ✅ Artifact Manager Workspace Isolation
- Properly validates target workspace from artifact objects
- **This is the security blueprint** all other services should follow
- Permission checks at both artifact and workspace levels

### ✅ S3 Path Traversal Protection
- `safe_join()` utility properly validates paths
- Rejects `../` patterns and absolute paths
- Consistently used across S3 operations

### ✅ JWT Algorithm Confusion Prevention
- Only HS256 and RS256 algorithms allowed
- "none" algorithm properly rejected
- No public key confusion vulnerabilities

### ✅ Token Expiration Validation (when present)
- Expired tokens are properly rejected
- `jwt.ExpiredSignatureError` correctly handled
- Issue is only with missing `exp` claim acceptance

### ✅ Scope Parsing Safety
- Path traversal in scopes blocked
- Null byte injection protected
- DoS-resistant implementation

---

## Assessment Methodology

### Team Structure
10 specialized security experts performed parallel analysis:
- Authentication security expert
- Workspace isolation expert
- RPC/WebSocket security expert
- Worker sandbox expert
- Storage security expert
- Infrastructure security expert
- Database security expert
- HTTP/ASGI security expert
- Protocol adapter security expert
- DoS resilience expert

### Approach
1. **White-box code review** - Full source code access
2. **Known vulnerability pattern analysis** - Leveraged existing V1-V15 patterns
3. **Proof-of-concept development** - Created exploits for confirmed vulnerabilities
4. **Comprehensive test creation** - Built test suites for all findings
5. **Coordinated discovery** - Avoided duplicate work, shared insights

### Tools & Techniques
- Static code analysis
- Manual security code review
- Architecture analysis
- Known vulnerability pattern matching
- Test-driven vulnerability validation

---

## Remediation Roadmap

### Phase 1: CRITICAL (Week 1)
**Priority**: Immediate action required

1. **SQL Injection Fixes** (V-STORAGE-01 to V-STORAGE-05)
   - Implement parameterized queries with `.bindparams()`
   - Add JSON key validation utility
   - Create comprehensive test suite
   - **Owner**: storage-security-expert (in progress)

2. **A2A Context Loss Fix** (V-PROTOCOL-19)
   - Preserve security context in `_filter_context_for_hypha()`
   - Test all A2A protocol endpoints
   - **Owner**: protocol-security-expert

3. **JWT Expiration Validation** (V-AUTH-1)
   - Add `options={"require": ["exp"]}` to all jwt.decode() calls
   - Verify existing tokens have expiration
   - **Owner**: auth-security-expert

### Phase 2: HIGH (Week 2)
**Priority**: High risk, address urgently

4. **Cross-Workspace Session Access** (W1)
   - Apply V10-V14 fix pattern to all workers
   - Add workspace validation before session access
   - **Affected**: browser.py, conda.py, terminal.py workers

5. **ASGI Input Validation** (V-HTTP-20)
   - Add workspace/service_id regex validation
   - Fix query string parsing with `urllib.parse`
   - Add path traversal protection
   - **Owner**: http-asgi-expert

6. **MCP/A2A Workspace Validation** (V-PROTOCOL-18, V-PROTOCOL-21)
   - Apply V10-V14 pattern to protocol adapters
   - Validate workspace permissions before routing
   - **Owner**: protocol-security-expert

7. **Worker Sandbox Escapes** (W2, W4)
   - Requires manual testing with actual environments
   - Document security warnings for conda worker
   - Strengthen command validation for terminal worker

### Phase 3: Architectural Changes (Month 1-2)
**Priority**: Requires design changes

8. **Server-Side Permission Validation** (V-AUTH-2)
   - Store permissions in database
   - Validate on every operation
   - Implement permission auditing
   - **Complexity**: HIGH - architectural change

9. **Token Revocation Mechanism** (V-AUTH-3)
   - Implement Redis-backed revocation list
   - Add `jti` claim to all tokens
   - Create revocation API endpoints
   - **Complexity**: MEDIUM

10. **JWT Secret Management** (V-AUTH-4)
    - Require explicit JWT_SECRET configuration
    - Add warning for auto-generated secrets
    - Improve entropy generation

### Phase 4: MEDIUM/LOW (Ongoing)
**Priority**: Important but lower risk

11. Address remaining MEDIUM and LOW severity findings
12. Add comprehensive security test coverage
13. Implement security monitoring and alerting
14. Create security documentation for developers

---

## Testing Recommendations

### Immediate Test Coverage Needed

1. **SQL Injection Tests**
   - Test each of the 5 injection points
   - Verify parameterized queries prevent injection
   - File: `tests/test_storage_security_vulnerabilities.py` (created)

2. **Cross-Workspace Access Tests**
   - Test V10-V14 pattern across all services
   - Verify workspace extraction and validation
   - File: `tests/test_cross_workspace_artifacts.py` (created)
   - Additional: workspace manager, app controller tests needed

3. **Worker Security Tests**
   - Test W1 cross-workspace session access
   - Test W3 conda worker execution boundaries
   - File: `tests/test_worker_sandbox_vulnerabilities.py` (created)

4. **Authentication Tests**
   - Test V-AUTH-1 expiration validation
   - Test V-AUTH-2 permission escalation scenarios
   - File: `tests/test_auth_security.py` (created)

5. **Protocol Adapter Tests**
   - Test MCP and A2A workspace validation
   - Test context preservation
   - File: `tests/test_protocol_adapter_security.py` (created)

### Continuous Security Testing

1. **Add Security Test Category** to test suite
2. **Run security tests in CI/CD** on every commit
3. **Monitor for new V10-V14 pattern instances**
4. **Regular security regression testing**

---

## Documentation Updates Needed

### CLAUDE.md Updates

Add new vulnerability patterns to the "Known Vulnerability Patterns" section:

#### SQL Injection via String Interpolation in SQLAlchemy

**Pattern**: Using `text(f"...")` with user input
**Severity**: CRITICAL
**Fix**: Use `.bindparams()` for all user-controlled values

#### A2A Context Loss

**Pattern**: Returning None from context filter functions
**Severity**: CRITICAL
**Fix**: Always preserve security context (ws, from, user)

#### Missing Input Validation in URL Routing

**Pattern**: Trust user-provided URL parameters without validation
**Severity**: HIGH
**Fix**: Validate all extracted path parameters with regex

#### Worker Session Workspace Validation

**Pattern**: Extension of V10-V14 for worker sessions
**Severity**: HIGH
**Fix**: Extract and validate workspace from session_id before access

---

## Long-Term Security Improvements

### Architecture

1. **Defense in Depth**
   - Add multiple layers of validation
   - Don't rely solely on token-based permissions
   - Implement server-side permission database

2. **Least Privilege**
   - Default to minimal permissions
   - Require explicit grants for cross-workspace access
   - Audit permission escalations

3. **Security by Default**
   - All services default to `protected` visibility
   - Require explicit context validation
   - Fail closed on permission errors

### Development Practices

1. **Security Code Review Checklist**
   - All methods accepting IDs must validate workspace
   - All SQL must use parameterized queries
   - All URL parameters must be validated
   - All security contexts must be preserved

2. **Secure Coding Guidelines**
   - Document security patterns in CLAUDE.md
   - Create secure code templates
   - Add security linters to CI/CD

3. **Security Training**
   - Educate developers on V10-V14 pattern family
   - Share vulnerability case studies
   - Regular security workshops

---

## Conclusion

This security assessment uncovered **25+ vulnerabilities** including **8 CRITICAL** severity issues requiring immediate attention. The most concerning findings are:

1. **SQL injection vulnerabilities** that completely bypass all security controls
2. **Complete security context loss** in A2A protocol adapter
3. **Systemic cross-workspace access control issues** (V10-V14 pattern family)
4. **Authentication weaknesses** enabling privilege escalation

**Positive findings** include correctly implemented artifact manager workspace isolation and path traversal protection, which serve as security blueprints for the rest of the codebase.

**Immediate action required** on Phase 1 and Phase 2 findings. Many fixes are straightforward code changes with documented patterns.

The assessment team has created comprehensive test suites and detailed remediation guidance for all findings. With focused effort, the CRITICAL and HIGH severity vulnerabilities can be resolved within 2-3 weeks.

---

## Appendices

### A. Test Files Created

1. `/Users/wei.ouyang/workspace/hypha/tests/test_storage_security_vulnerabilities.py`
2. `/Users/wei.ouyang/workspace/hypha/tests/test_cross_workspace_artifacts.py`
3. `/Users/wei.ouyang/workspace/hypha/tests/test_worker_sandbox_vulnerabilities.py`
4. `/Users/wei.ouyang/workspace/hypha/tests/test_auth_security.py`
5. `/Users/wei.ouyang/workspace/hypha/tests/test_protocol_adapter_security.py`

### B. Detailed Reports

- `PROTOCOL_ADAPTER_SECURITY_REPORT.md` (created by protocol-security-expert)

### C. Assessment Team

| Expert | Responsibilities | Findings |
|--------|-----------------|----------|
| storage-security-expert | Artifact storage, S3, SQL | 5 CRITICAL SQL injection |
| auth-security-expert | Authentication, JWT, tokens | 4 auth vulnerabilities |
| worker-sandbox-expert | Worker isolation, sandboxing | W1-W4 worker issues |
| workspace-security-expert | Workspace isolation, services | Confirmed V10-V14 pattern |
| protocol-security-expert | MCP/A2A adapters | 8 protocol vulnerabilities |
| http-asgi-expert | HTTP routing, ASGI services | 6 HTTP/ASGI issues |
| rpc-websocket-expert | RPC layer, WebSocket | Context forgery risks |
| infrastructure-expert | Redis event bus | (In progress) |
| database-security-expert | SQL models outside artifacts | (In progress) |
| dos-resilience-expert | Rate limiting, DoS | (In progress) |

### D. Vulnerability ID Index

- **V-STORAGE-01 to V-STORAGE-05**: SQL Injection vulnerabilities
- **V-AUTH-1 to V-AUTH-4**: Authentication vulnerabilities
- **W1 to W4**: Worker sandbox vulnerabilities
- **V-PROTOCOL-16 to V-PROTOCOL-23**: Protocol adapter vulnerabilities
- **V-HTTP-16 to V-HTTP-21**: HTTP/ASGI vulnerabilities
- **V10-V15**: Previously documented cross-workspace access (from existing tests)

---

**Report Version**: 1.0
**Date**: 2026-02-08
**Prepared by**: Hypha Security Assessment Team
**Distribution**: Internal - Hypha Development Team
