"""Local authentication provider for Hypha.

This module implements a self-contained email/password authentication provider
for Hypha deployments that do not use an external identity provider (Auth0).

Email verification
------------------
Signup is a two-step flow:

1. ``signup`` validates the input, checks the email is not already registered,
   generates a short-lived numeric verification code, stashes the *pending*
   account (name + password hash + code) in a bounded, TTL-evicting in-memory
   store, and emails the code via an injectable :class:`EmailTransport`.
   The account is **not** created until the code is verified.
2. ``verify_email`` checks the submitted code (with attempt counting, TTL, and
   max-attempts lockout) and, on success, materialises the real user artifact.

Email transport
---------------
The email transport is *injectable* so it can be exercised in tests without a
real provider (see :class:`CapturingEmailTransport`) and so production can use
Resend (:class:`ResendEmailTransport`, driven by the ``RESEND_API_KEY`` env var).

Dev fallback
------------
If ``RESEND_API_KEY`` is unset, the provider falls back to
:class:`LoggingEmailTransport`, which logs the code instead of sending it and
marks itself as a dev fallback. In that mode ONLY, ``signup`` returns the code
in its response (``dev_code``) so developers can complete the flow locally
without an email provider. When a real transport is configured the code is
never returned to the client.
"""

import hashlib
import secrets
import os
import time
import asyncio
import logging
from typing import Any, Callable, Optional

from hypha.core import UserInfo, UserPermission
from hypha.core.auth import _parse_token, create_scope
from hypha.utils import random_id
import shortuuid

logger = logging.getLogger(__name__)

# In-memory storage for login sessions (production should use Redis)
LOGIN_SESSIONS = {}

# ---------------------------------------------------------------------------
# Email verification configuration
# ---------------------------------------------------------------------------

# Length of the numeric verification code.
VERIFICATION_CODE_LENGTH = 6
# How long a verification code / pending-signup session is valid, in seconds.
VERIFICATION_CODE_TTL = int(os.environ.get("HYPHA_VERIFICATION_CODE_TTL", "600"))
# Maximum number of wrong-code attempts before the pending session is discarded.
MAX_VERIFICATION_ATTEMPTS = int(os.environ.get("HYPHA_VERIFICATION_MAX_ATTEMPTS", "5"))
# Minimum seconds between successive "resend code" requests for one email.
RESEND_THROTTLE_SECONDS = int(os.environ.get("HYPHA_VERIFICATION_RESEND_THROTTLE", "30"))
# Upper bound on how many pending-verification sessions we keep in memory. This
# caps the memory used by abandoned signups (each also expires via TTL).
MAX_PENDING_VERIFICATIONS = int(
    os.environ.get("HYPHA_VERIFICATION_MAX_PENDING", "10000")
)
# From-address used for verification emails.
VERIFICATION_FROM_EMAIL = os.environ.get(
    "HYPHA_VERIFICATION_FROM_EMAIL", "Hypha <onboarding@resend.dev>"
)


def generate_verification_code() -> str:
    """Generate a cryptographically-random numeric verification code."""
    upper = 10 ** VERIFICATION_CODE_LENGTH
    return str(secrets.randbelow(upper)).zfill(VERIFICATION_CODE_LENGTH)


# ---------------------------------------------------------------------------
# Email transport abstraction (injectable)
# ---------------------------------------------------------------------------


class EmailTransport:
    """Interface for sending verification emails.

    Subclasses must implement :meth:`send_verification_email`. The
    ``is_dev_fallback`` attribute indicates whether this transport actually
    delivers email (``False``) or is a non-delivering dev stand-in (``True``);
    in the latter case the signup handler may expose the code to the caller.
    """

    is_dev_fallback: bool = False

    async def send_verification_email(self, to: str, code: str) -> None:
        raise NotImplementedError


def _verification_email_html(code: str) -> str:
    return (
        "<div style=\"font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"
        "sans-serif;max-width:480px;margin:0 auto;padding:32px 24px;color:#1f2937\">"
        "<h1 style=\"font-size:20px;margin:0 0 16px\">Verify your email</h1>"
        "<p style=\"font-size:14px;line-height:1.6;margin:0 0 24px\">"
        "Use the following code to finish creating your Hypha account. "
        "This code expires in "
        f"{VERIFICATION_CODE_TTL // 60} minutes.</p>"
        "<div style=\"font-size:32px;font-weight:700;letter-spacing:8px;"
        "background:#f3f4f6;border-radius:8px;padding:16px;text-align:center;"
        f"margin:0 0 24px\">{code}</div>"
        "<p style=\"font-size:12px;color:#6b7280;margin:0\">"
        "If you did not request this, you can safely ignore this email.</p></div>"
    )


def _verification_email_text(code: str) -> str:
    return (
        f"Your Hypha verification code is: {code}\n\n"
        f"It expires in {VERIFICATION_CODE_TTL // 60} minutes. "
        "If you did not request this, you can ignore this email."
    )


class ResendEmailTransport(EmailTransport):
    """Send verification emails through the Resend HTTP API.

    ``http_post`` is injectable for testing (defaults to an ``httpx.AsyncClient``
    POST). In production it is left unset and a real HTTPS call is made.
    """

    is_dev_fallback = False
    API_URL = "https://api.resend.com/emails"

    def __init__(
        self,
        api_key: str,
        from_email: str,
        http_post: Optional[Callable] = None,
    ):
        assert api_key, "Resend API key is required"
        self.api_key = api_key
        self.from_email = from_email
        self._http_post = http_post

    async def _default_post(self, url, json=None, headers=None):
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            return await client.post(url, json=json, headers=headers)

    async def send_verification_email(self, to: str, code: str) -> None:
        post = self._http_post or self._default_post
        payload = {
            "from": self.from_email,
            "to": [to],
            "subject": "Your Hypha verification code",
            "html": _verification_email_html(code),
            "text": _verification_email_text(code),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = await post(self.API_URL, json=payload, headers=headers)
        status = getattr(response, "status_code", 200)
        if status >= 400:
            # Surface a sanitized error; do NOT leak the code or API key.
            body = getattr(response, "text", "")
            logger.error("Resend email send failed (status %s): %s", status, body)
            raise RuntimeError(f"Failed to send verification email (status {status})")


class LoggingEmailTransport(EmailTransport):
    """Dev fallback transport: logs the code instead of sending it.

    Used when ``RESEND_API_KEY`` is not configured so that signup remains usable
    in development. Marked as a dev fallback so the signup handler exposes the
    code to the caller (never done when a real transport is configured).
    """

    is_dev_fallback = True

    def __init__(self, from_email: str = VERIFICATION_FROM_EMAIL):
        self.from_email = from_email

    async def send_verification_email(self, to: str, code: str) -> None:
        logger.warning(
            "[local-auth dev] RESEND_API_KEY not set; verification code for %s is %s",
            to,
            code,
        )


class CapturingEmailTransport(EmailTransport):
    """In-process transport that CAPTURES sent emails (for tests).

    This is a REAL object (not a mock): it records every send so tests can assert
    on the recipient and code without any external service.
    """

    is_dev_fallback = False

    def __init__(self):
        self.sent = []

    async def send_verification_email(self, to: str, code: str) -> None:
        self.sent.append({"to": to, "code": code})


def build_email_transport(
    api_key: Optional[str] = None,
    from_email: str = VERIFICATION_FROM_EMAIL,
) -> EmailTransport:
    """Construct the appropriate transport based on configuration.

    Uses Resend when an API key is available, otherwise the logging dev fallback.
    """
    if api_key is None:
        api_key = os.environ.get("RESEND_API_KEY")
    if api_key:
        return ResendEmailTransport(api_key=api_key, from_email=from_email)
    return LoggingEmailTransport(from_email=from_email)


# The process-wide email transport. Lazily built on first use from the
# environment; overridable via set_email_transport (used by tests to inject a
# real capturing transport, and could be used by deployments to customise).
_email_transport: Optional[EmailTransport] = None


def get_email_transport() -> EmailTransport:
    """Return the active email transport, building the default lazily."""
    global _email_transport
    if _email_transport is None:
        _email_transport = build_email_transport()
    return _email_transport


def set_email_transport(transport: EmailTransport) -> None:
    """Override the active email transport (dependency injection)."""
    global _email_transport
    _email_transport = transport


def reset_email_transport() -> None:
    """Reset to the default (env-derived) transport, rebuilt on next use."""
    global _email_transport
    _email_transport = None


# ---------------------------------------------------------------------------
# Pending-verification store (bounded, TTL-evicting)
# ---------------------------------------------------------------------------


class PendingVerificationStore:
    """A bounded, TTL-evicting store for pending email-verification sessions.

    Keyed by (lowercased) email. Each entry holds the verification code, the
    pending account data (name + password hash + salt), an attempt counter, and
    timestamps. Expired entries are evicted on read; the store never grows beyond
    ``max_entries`` (oldest entries are dropped first) so abandoned signups can
    never leak unbounded memory.
    """

    def __init__(self, ttl: int, max_entries: int):
        self.ttl = ttl
        self.max_entries = max_entries
        self._entries = {}

    def _is_expired(self, entry: dict) -> bool:
        return (time.time() - entry["created_at"]) > self.ttl

    def put(self, email: str, data: dict) -> None:
        now = time.time()
        entry = dict(data)
        entry.setdefault("attempts", 0)
        entry.setdefault("created_at", now)
        entry.setdefault("last_sent_at", now)
        self._entries[email] = entry
        # Enforce the bound: drop oldest entries (by created_at) beyond the cap.
        while len(self._entries) > self.max_entries:
            oldest_key = min(
                self._entries, key=lambda k: self._entries[k]["created_at"]
            )
            del self._entries[oldest_key]

    def get(self, email: str):
        entry = self._entries.get(email)
        if entry is None:
            return None
        if self._is_expired(entry):
            del self._entries[email]
            return None
        return entry

    def delete(self, email: str) -> None:
        self._entries.pop(email, None)

    def sweep_expired(self) -> int:
        """Remove all expired entries; return the number removed."""
        expired = [k for k, e in self._entries.items() if self._is_expired(e)]
        for k in expired:
            del self._entries[k]
        return len(expired)


_PENDING_VERIFICATIONS = PendingVerificationStore(
    ttl=VERIFICATION_CODE_TTL, max_entries=MAX_PENDING_VERIFICATIONS
)

# Background sweeper handle (started in hypha_startup).
_sweeper_task = None


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str, salt: str) -> str:
    """Hash a password with salt using SHA-256."""
    return hashlib.sha256((password + salt).encode()).hexdigest()


def verify_password(password: str, salt: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password, salt) == hashed


async def local_parse_token(token: str) -> UserInfo:
    """Parse a local authentication token or fall back to JWT."""
    # For local auth, we just use the standard JWT tokens
    # This will handle both authenticated users and anonymous users properly
    # Note: anonymous users won't have a token, so this shouldn't be called for them
    # But if it is, we need to handle it properly
    if not token:
        # This shouldn't happen - anonymous users shouldn't trigger parse_token
        # But if it does, generate an anonymous user
        from hypha.core.auth import generate_anonymous_user
        return generate_anonymous_user()
    result = _parse_token(token)
    return result


async def local_generate_token(user_info: UserInfo, expires_in: int) -> str:
    """Generate a JWT token for local authentication."""
    # Use the default JWT token generation
    # We must NOT call generate_auth_token as that would cause infinite recursion
    from hypha.core.auth import _generate_presigned_token
    
    # The token should include email and other user info
    # Make sure the UserInfo has all necessary fields
    token = _generate_presigned_token(user_info, expires_in)
    
    # Log for debugging (can be removed later)
    logger.debug(f"Generated token for user: {user_info.id}, email: {user_info.email}")
    
    return token


async def profile_page_handler(event):
    """Serve the profile management page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Profile - Hypha</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <h1 class="text-2xl font-bold text-center text-gray-800 mb-6">My Profile</h1>
        
        <div id="profileInfo" class="mb-6">
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700">Email</label>
                <p id="userEmail" class="mt-1 text-gray-900">Loading...</p>
            </div>
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700">Name</label>
                <p id="userName" class="mt-1 text-gray-900">Loading...</p>
            </div>
            <div class="mb-4">
                <label class="block text-sm font-medium text-gray-700">User ID</label>
                <p id="userId" class="mt-1 text-gray-900 font-mono text-sm">Loading...</p>
            </div>
        </div>

        <!-- Update Profile Form -->
        <div id="updateForm" class="space-y-4 border-t pt-4">
            <h2 class="text-lg font-semibold text-gray-800">Update Profile</h2>
            
            <div>
                <label for="newName" class="block text-sm font-medium text-gray-700">Name</label>
                <input type="text" id="newName" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            
            <div class="border-t pt-4">
                <h3 class="text-md font-semibold text-gray-800 mb-2">Change Password</h3>
                <div class="space-y-2">
                    <div>
                        <label for="currentPassword" class="block text-sm font-medium text-gray-700">Current Password</label>
                        <input type="password" id="currentPassword" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                    </div>
                    <div>
                        <label for="newPassword" class="block text-sm font-medium text-gray-700">New Password</label>
                        <input type="password" id="newPassword" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                    </div>
                    <div>
                        <label for="confirmPassword" class="block text-sm font-medium text-gray-700">Confirm New Password</label>
                        <input type="password" id="confirmPassword" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                    </div>
                </div>
            </div>
            
            <button onclick="updateProfile()" class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                Update Profile
            </button>
        </div>

        <!-- Back to Home -->
        <div class="mt-6 text-center">
            <a href="/" class="text-blue-600 hover:text-blue-800">← Back to Home</a>
        </div>

        <!-- Messages -->
        <div id="message" class="mt-4 text-center text-sm"></div>
    </div>

    <script>
        // Get token from localStorage or cookie
        function getToken() {
            // Try localStorage first
            const token = localStorage.getItem('hypha_token');
            if (token) return token;
            
            // Try cookie
            const cookies = document.cookie.split(';');
            for (let cookie of cookies) {
                const [name, value] = cookie.trim().split('=');
                if (name === 'access_token') {
                    return value;
                }
            }
            return null;
        }

        // Parse JWT token
        function parseJwt(token) {
            try {
                const base64Url = token.split('.')[1];
                const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
                const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)).join(''));
                return JSON.parse(jsonPayload);
            } catch (e) {
                return null;
            }
        }

        // Load user info
        function loadUserInfo() {
            const token = getToken();
            if (!token) {
                window.location.href = '/public/apps/hypha-login';
                return;
            }

            const userInfo = parseJwt(token);
            if (!userInfo) {
                window.location.href = '/public/apps/hypha-login';
                return;
            }

            // Display user info - check for namespaced fields
            const namespace = 'https://amun.ai/';
            const email = userInfo.email || userInfo[namespace + 'email'] || 'Not available';
            const roles = userInfo.roles || userInfo[namespace + 'roles'] || [];
            const name = userInfo.name || email.split('@')[0] || 'Not available';
            const userId = userInfo.id || userInfo.sub || 'Not available';
            
            document.getElementById('userEmail').textContent = email;
            document.getElementById('userName').textContent = name;
            document.getElementById('userId').textContent = userId;
            
            // Pre-fill the name field
            document.getElementById('newName').value = name || '';
        }

        function showMessage(msg, isError = false) {
            const messageEl = document.getElementById('message');
            messageEl.textContent = msg;
            messageEl.className = `mt-4 text-center text-sm ${isError ? 'text-red-600' : 'text-green-600'}`;
        }

        async function updateProfile() {
            const token = getToken();
            if (!token) {
                showMessage('Not authenticated', true);
                return;
            }

            const userInfo = parseJwt(token);
            const namespace = 'https://amun.ai/';
            const userEmail = userInfo.email || userInfo[namespace + 'email'];
            const newName = document.getElementById('newName').value;
            const currentPassword = document.getElementById('currentPassword').value;
            const newPassword = document.getElementById('newPassword').value;
            const confirmPassword = document.getElementById('confirmPassword').value;

            // Validate passwords if changing
            if (newPassword) {
                if (!currentPassword) {
                    showMessage('Current password is required to change password', true);
                    return;
                }
                if (newPassword !== confirmPassword) {
                    showMessage('New passwords do not match', true);
                    return;
                }
                if (newPassword.length < 8) {
                    showMessage('New password must be at least 8 characters', true);
                    return;
                }
            }

            try {
                const response = await fetch('/public/services/hypha-login/update_profile', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                        user_email: userEmail,
                        name: newName || undefined,
                        current_password: currentPassword || undefined,
                        new_password: newPassword || undefined
                    })
                });

                const result = await response.json();
                if (result.success) {
                    showMessage('Profile updated successfully!');
                    // Clear password fields
                    document.getElementById('currentPassword').value = '';
                    document.getElementById('newPassword').value = '';
                    document.getElementById('confirmPassword').value = '';
                    
                    // Update displayed name
                    document.getElementById('userName').textContent = newName || document.getElementById('userName').textContent;
                } else {
                    showMessage(result.error || 'Failed to update profile', true);
                }
            } catch (error) {
                showMessage('An error occurred: ' + error.message, true);
            }
        }

        // Load user info on page load
        loadUserInfo();
        
        // Also store token in cookie when updating profile if we have it
        window.addEventListener('DOMContentLoaded', function() {
            const token = getToken();
            if (token) {
                try {
                    const tokenData = parseJwt(token);
                    if (tokenData && tokenData.exp) {
                        const maxAge = tokenData.exp - Math.floor(Date.now() / 1000);
                        if (maxAge > 0) {
                            document.cookie = `access_token=${token}; path=/; max-age=${maxAge}; samesite=lax`;
                        }
                    }
                } catch(e) {
                    // Ignore errors
                }
            }
        });
    </script>
</body>
</html>
    """
    return {
        "status": 200,
        "headers": {"Content-Type": "text/html"},
        "body": html_content,
    }


async def index_handler(event):
    """Serve the login/signup page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hypha Local Authentication</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <!-- Logged In View -->
        <div id="loggedInView" class="hidden">
            <h1 class="text-2xl font-bold text-center text-gray-800 mb-6">Welcome Back!</h1>
            <div class="bg-gray-50 rounded-lg p-6 mb-6">
                <div class="flex items-center mb-4">
                    <div id="userAvatar" class="w-16 h-16 rounded-full bg-blue-600 text-white flex items-center justify-center text-2xl font-bold mr-4">
                        U
                    </div>
                    <div>
                        <h2 id="userName" class="text-xl font-semibold text-gray-800">User</h2>
                        <p id="userEmail" class="text-sm text-gray-600">email@example.com</p>
                    </div>
                </div>
                <div class="border-t pt-4 mt-4">
                    <p class="text-sm text-gray-600 mb-1">User ID</p>
                    <p id="userId" class="text-xs font-mono text-gray-700">-</p>
                </div>
                <div class="mt-3">
                    <p class="text-sm text-gray-600 mb-1">Session Status</p>
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        <span class="w-2 h-2 bg-green-400 rounded-full mr-1.5"></span>
                        Active
                    </span>
                </div>
            </div>
            
            <div class="space-y-3">
                <a href="/" class="block w-full bg-blue-600 text-white text-center py-2 px-4 rounded-md hover:bg-blue-700 transition duration-300">
                    <i class="fas fa-home mr-2"></i> Go to Dashboard
                </a>
                <a href="/public/apps/hypha-login/profile" class="block w-full bg-gray-600 text-white text-center py-2 px-4 rounded-md hover:bg-gray-700 transition duration-300">
                    <i class="fas fa-user mr-2"></i> Manage Profile
                </a>
                <button onclick="handleLogout()" class="w-full bg-red-600 text-white py-2 px-4 rounded-md hover:bg-red-700 transition duration-300">
                    <i class="fas fa-sign-out-alt mr-2"></i> Logout
                </button>
                <button onclick="switchToLogin()" class="w-full bg-gray-200 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-300 transition duration-300">
                    <i class="fas fa-user-plus mr-2"></i> Login as Different User
                </button>
            </div>
        </div>

        <!-- Login/Signup View -->
        <div id="authView">
            <div class="mb-6">
                <h1 class="text-2xl font-bold text-center text-gray-800 mb-2">Hypha Authentication</h1>
                <div class="flex justify-center space-x-4 mb-6">
                    <button onclick="showTab('login')" id="loginTab" class="px-4 py-2 font-semibold text-blue-600 border-b-2 border-blue-600">Login</button>
                    <button onclick="showTab('signup')" id="signupTab" class="px-4 py-2 font-semibold text-gray-600">Sign Up</button>
                </div>
            </div>

            <!-- Login Form -->
            <div id="loginForm" class="space-y-4">
                <div>
                    <label for="loginEmail" class="block text-sm font-medium text-gray-700">Email</label>
                    <input type="email" id="loginEmail" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500" required>
                </div>
                <div>
                    <label for="loginPassword" class="block text-sm font-medium text-gray-700">Password</label>
                    <input type="password" id="loginPassword" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500" required>
                </div>
                <button onclick="handleLogin()" class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                    Login
                </button>
            </div>

            <!-- Signup Form -->
            <div id="signupForm" class="space-y-4 hidden">
                <div>
                    <label for="signupName" class="block text-sm font-medium text-gray-700">Name</label>
                    <input type="text" id="signupName" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500" required>
                </div>
                <div>
                    <label for="signupEmail" class="block text-sm font-medium text-gray-700">Email</label>
                    <input type="email" id="signupEmail" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500" required>
                </div>
                <div>
                    <label for="signupPassword" class="block text-sm font-medium text-gray-700">Password</label>
                    <input type="password" id="signupPassword" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500" required>
                </div>
                <div>
                    <label for="signupPasswordConfirm" class="block text-sm font-medium text-gray-700">Confirm Password</label>
                    <input type="password" id="signupPasswordConfirm" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500" required>
                </div>
                <button onclick="handleSignup()" class="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2">
                    Sign Up
                </button>
            </div>

            <!-- Email Verification Form -->
            <div id="verifyForm" class="space-y-4 hidden">
                <div class="text-center">
                    <div class="mx-auto w-12 h-12 rounded-full bg-green-100 flex items-center justify-center mb-3">
                        <i class="fas fa-envelope-open-text text-green-600 text-xl"></i>
                    </div>
                    <h2 class="text-lg font-semibold text-gray-800">Check your email</h2>
                    <p class="text-sm text-gray-600 mt-1">
                        We sent a 6-digit code to <span id="verifyEmailLabel" class="font-medium text-gray-900"></span>.
                    </p>
                </div>
                <div>
                    <label for="verifyCode" class="block text-sm font-medium text-gray-700">Verification Code</label>
                    <input type="text" id="verifyCode" inputmode="numeric" autocomplete="one-time-code" maxlength="6"
                        class="mt-1 block w-full px-3 py-2 text-center text-2xl tracking-[0.5em] font-mono border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                        placeholder="______">
                </div>
                <button onclick="handleVerify()" class="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                    Verify &amp; Create Account
                </button>
                <div class="flex items-center justify-between text-sm">
                    <button onclick="handleResend()" id="resendBtn"
                        class="text-blue-600 hover:text-blue-800 disabled:text-gray-400 disabled:cursor-not-allowed disabled:no-underline">
                        Resend code
                    </button>
                    <span id="resendCountdown" class="text-gray-500"></span>
                </div>
                <button onclick="cancelVerify()" class="w-full text-gray-500 text-sm hover:text-gray-700">
                    ← Use a different email
                </button>
            </div>

            <!-- Messages -->
            <div id="message" class="mt-4 text-center text-sm"></div>
        </div>
    </div>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const loginKey = urlParams.get('key');
        const workspace = urlParams.get('workspace');
        const redirect = urlParams.get('redirect');
        const expires_in = urlParams.get('expires_in');

        // Add Font Awesome for icons
        const fontAwesome = document.createElement('link');
        fontAwesome.rel = 'stylesheet';
        fontAwesome.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css';
        document.head.appendChild(fontAwesome);

        // Check for existing token in cookies or localStorage on page load
        window.addEventListener('DOMContentLoaded', function() {
            // Try to get token from cookie
            let token = null;
            try {
                const cookieToken = document.cookie.split('; ').find(row => row.startsWith('access_token'));
                if (cookieToken) {
                    token = cookieToken.split('=')[1];
                }
            } catch(e) {
                // Ignore cookie errors
            }
            
            // If no cookie token, try localStorage
            if (!token) {
                token = localStorage.getItem('hypha_token');
            }
            
            // If we have a token and it's still valid, show logged in view
            if (token) {
                try {
                    // Parse JWT to check expiration
                    const base64Url = token.split('.')[1];
                    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
                    const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)).join(''));
                    const tokenData = JSON.parse(jsonPayload);
                    
                    // Check if token is still valid
                    if (tokenData.exp * 1000 > Date.now()) {
                        // Token is valid - show logged in view
                        showLoggedInView(tokenData);
                        
                        // If this is a popup login flow, report the token
                        if (loginKey) {
                            fetch('/public/services/hypha-login/report', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    key: loginKey,
                                    token: token,
                                    user_id: tokenData.id || tokenData.sub,
                                    email: tokenData.email || tokenData['https://amun.ai/email'],
                                    workspace: workspace || tokenData.scope?.current_workspace,
                                    expires_in: expires_in || 3600
                                })
                            }).then(() => {
                                // Handle redirect or close window
                                if (redirect) {
                                    setTimeout(() => {
                                        window.location.href = redirect;
                                    }, 500);
                                } else {
                                    // Add a notice that login was successful
                                    const notice = document.createElement('div');
                                    notice.className = 'mt-4 p-3 bg-green-100 border border-green-400 text-green-700 rounded text-center';
                                    notice.innerHTML = '<i class="fas fa-check-circle mr-2"></i>Login successful! This window will close automatically.';
                                    document.getElementById('loggedInView').appendChild(notice);
                                    setTimeout(() => {
                                        window.close();
                                    }, 2000);
                                }
                            });
                        } else if (redirect) {
                            // If we have a redirect but no key, just redirect
                            setTimeout(() => {
                                window.location.href = redirect;
                            }, 500);
                        }
                    } else {
                        // Token expired, clear it
                        localStorage.removeItem('hypha_token');
                        // Also clear cookie
                        document.cookie = 'access_token=; path=/; max-age=0; samesite=lax';
                    }
                } catch(e) {
                    // Invalid token, clear it
                    console.error('Invalid token:', e);
                    localStorage.removeItem('hypha_token');
                    document.cookie = 'access_token=; path=/; max-age=0; samesite=lax';
                }
            }
        });

        function showLoggedInView(tokenData) {
            // Hide auth view, show logged in view
            document.getElementById('authView').classList.add('hidden');
            document.getElementById('loggedInView').classList.remove('hidden');
            
            // Update user info - check for namespaced fields
            const namespace = 'https://amun.ai/';
            const email = tokenData.email || tokenData[namespace + 'email'] || 'Not available';
            const name = tokenData.name || (email !== 'Not available' ? email.split('@')[0] : 'User');
            const userId = tokenData.id || tokenData.sub || 'Not available';
            
            document.getElementById('userName').textContent = name;
            document.getElementById('userEmail').textContent = email;
            document.getElementById('userId').textContent = userId;
            
            // Update avatar with first letter of name
            const firstLetter = (name[0] || 'U').toUpperCase();
            document.getElementById('userAvatar').textContent = firstLetter;
        }

        function switchToLogin() {
            // Clear tokens but stay on the page
            localStorage.removeItem('hypha_token');
            document.cookie = 'access_token=; path=/; max-age=0; samesite=lax';
            
            // Show auth view, hide logged in view
            document.getElementById('loggedInView').classList.add('hidden');
            document.getElementById('authView').classList.remove('hidden');
            
            // Show a message
            showMessage('You can now login with a different account', false);
        }

        function handleLogout() {
            // Clear tokens
            localStorage.removeItem('hypha_token');
            document.cookie = 'access_token=; path=/; max-age=0; samesite=lax';
            
            // If this is a popup, close it
            if (loginKey) {
                window.close();
            } else {
                // Show auth view
                document.getElementById('loggedInView').classList.add('hidden');
                document.getElementById('authView').classList.remove('hidden');
                showMessage('Logged out successfully', false);
            }
        }

        function showTab(tab) {
            const loginForm = document.getElementById('loginForm');
            const signupForm = document.getElementById('signupForm');
            const loginTab = document.getElementById('loginTab');
            const signupTab = document.getElementById('signupTab');

            // Always leave the verification step when switching tabs.
            const verifyForm = document.getElementById('verifyForm');
            if (verifyForm) verifyForm.classList.add('hidden');

            if (tab === 'login') {
                loginForm.classList.remove('hidden');
                signupForm.classList.add('hidden');
                loginTab.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');
                loginTab.classList.remove('text-gray-600');
                signupTab.classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
                signupTab.classList.add('text-gray-600');
            } else {
                signupForm.classList.remove('hidden');
                loginForm.classList.add('hidden');
                signupTab.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');
                signupTab.classList.remove('text-gray-600');
                loginTab.classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
                loginTab.classList.add('text-gray-600');
            }
        }

        function showMessage(msg, isError = false) {
            const messageEl = document.getElementById('message');
            messageEl.textContent = msg;
            messageEl.className = `mt-4 text-center text-sm ${isError ? 'text-red-600' : 'text-green-600'}`;
        }

        async function handleLogin() {
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;

            if (!email || !password) {
                showMessage('Please fill in all fields', true);
                return;
            }

            try {
                // Call the login handler
                const response = await fetch('/public/services/hypha-login/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        email,
                        password,
                        workspace: workspace || null,
                        key: loginKey
                    })
                });

                const result = await response.json();
                if (result.success) {
                    showMessage('Login successful!');
                    
                    // Store token in localStorage for persistence
                    localStorage.setItem('hypha_token', result.token);
                    
                    // Also set cookie for server-side authentication
                    const tokenData = JSON.parse(atob(result.token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
                    const maxAge = tokenData.exp - Math.floor(Date.now() / 1000);
                    document.cookie = `access_token=${result.token}; path=/; max-age=${maxAge}; samesite=lax`;
                    
                    // Add user info from result to tokenData if not present
                    if (!tokenData.name && result.user_id) {
                        tokenData.name = result.email?.split('@')[0];
                    }
                    if (!tokenData.email && result.email) {
                        tokenData.email = result.email;
                    }
                    if (!tokenData.id && result.user_id) {
                        tokenData.id = result.user_id;
                    }
                    
                    // Show logged in view
                    showLoggedInView(tokenData);
                    
                    // If we have a login key, report it
                    if (loginKey) {
                        await fetch('/public/services/hypha-login/report', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                key: loginKey,
                                token: result.token,
                                user_id: result.user_id,
                                email: result.email,
                                workspace: result.workspace || workspace,
                                expires_in: expires_in || 3600
                            })
                        });
                        
                        // Handle redirect or close window
                        if (redirect) {
                            setTimeout(() => {
                                window.location.href = redirect;
                            }, 500);
                        } else {
                            // Add a notice that login was successful
                            const notice = document.createElement('div');
                            notice.className = 'mt-4 p-3 bg-green-100 border border-green-400 text-green-700 rounded text-center';
                            notice.innerHTML = '<i class="fas fa-check-circle mr-2"></i>Login successful! This window will close automatically.';
                            document.getElementById('loggedInView').appendChild(notice);
                            setTimeout(() => {
                                window.close();
                            }, 2000);
                        }
                    } else if (redirect) {
                        // If we have a redirect but no key, just redirect
                        setTimeout(() => {
                            window.location.href = redirect;
                        }, 500);
                    }
                } else {
                    showMessage(result.error || 'Login failed', true);
                }
            } catch (error) {
                showMessage('An error occurred: ' + error.message, true);
            }
        }

        async function handleSignup() {
            const name = document.getElementById('signupName').value;
            const email = document.getElementById('signupEmail').value;
            const password = document.getElementById('signupPassword').value;
            const passwordConfirm = document.getElementById('signupPasswordConfirm').value;

            if (!name || !email || !password || !passwordConfirm) {
                showMessage('Please fill in all fields', true);
                return;
            }

            if (password !== passwordConfirm) {
                showMessage('Passwords do not match', true);
                return;
            }

            if (password.length < 8) {
                showMessage('Password must be at least 8 characters', true);
                return;
            }

            try {
                const response = await fetch('/public/services/hypha-login/signup', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        name,
                        email,
                        password
                    })
                });

                const result = await response.json();
                if (result.success) {
                    // A verification code has been emailed. Move to the verify step.
                    pendingEmail = result.email || email;
                    document.getElementById('verifyEmailLabel').textContent = pendingEmail;
                    showVerifyForm();
                    if (result.dev_code) {
                        // Dev fallback (no RESEND_API_KEY): prefill the code to ease local testing.
                        document.getElementById('verifyCode').value = result.dev_code;
                        showMessage('Dev mode: code auto-filled (RESEND_API_KEY not set).', false);
                    } else {
                        showMessage('We emailed you a 6-digit verification code.', false);
                    }
                    startResendCountdown(30);
                } else {
                    showMessage(result.error || 'Signup failed', true);
                }
            } catch (error) {
                showMessage('An error occurred: ' + error.message, true);
            }
        }

        // ---- Email verification step ----
        let pendingEmail = null;
        let resendTimer = null;

        function showVerifyForm() {
            document.getElementById('loginForm').classList.add('hidden');
            document.getElementById('signupForm').classList.add('hidden');
            document.getElementById('verifyForm').classList.remove('hidden');
            document.getElementById('verifyCode').focus();
        }

        function cancelVerify() {
            if (resendTimer) { clearInterval(resendTimer); resendTimer = null; }
            pendingEmail = null;
            document.getElementById('verifyForm').classList.add('hidden');
            document.getElementById('verifyCode').value = '';
            showTab('signup');
            showMessage('', false);
        }

        function startResendCountdown(seconds) {
            const btn = document.getElementById('resendBtn');
            const label = document.getElementById('resendCountdown');
            btn.disabled = true;
            let remaining = seconds;
            label.textContent = `Resend available in ${remaining}s`;
            if (resendTimer) clearInterval(resendTimer);
            resendTimer = setInterval(() => {
                remaining -= 1;
                if (remaining <= 0) {
                    clearInterval(resendTimer);
                    resendTimer = null;
                    btn.disabled = false;
                    label.textContent = '';
                } else {
                    label.textContent = `Resend available in ${remaining}s`;
                }
            }, 1000);
        }

        async function handleVerify() {
            const code = document.getElementById('verifyCode').value.trim();
            if (!pendingEmail) { showMessage('No pending signup. Please sign up again.', true); return; }
            if (!code || code.length < 6) { showMessage('Please enter the 6-digit code', true); return; }

            try {
                const response = await fetch('/public/services/hypha-login/verify_email', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ email: pendingEmail, code })
                });
                const result = await response.json();
                if (result.success) {
                    if (resendTimer) { clearInterval(resendTimer); resendTimer = null; }
                    document.getElementById('verifyForm').classList.add('hidden');
                    document.getElementById('verifyCode').value = '';
                    showMessage('Email verified! Account created. You can now log in.', false);
                    setTimeout(() => showTab('login'), 1500);
                } else {
                    showMessage(result.error || 'Verification failed', true);
                }
            } catch (error) {
                showMessage('An error occurred: ' + error.message, true);
            }
        }

        async function handleResend() {
            if (!pendingEmail) { showMessage('No pending signup. Please sign up again.', true); return; }
            try {
                const response = await fetch('/public/services/hypha-login/resend_code', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ email: pendingEmail })
                });
                const result = await response.json();
                if (result.success) {
                    showMessage('A new code has been sent.', false);
                    if (result.dev_code) {
                        document.getElementById('verifyCode').value = result.dev_code;
                    }
                    startResendCountdown(30);
                } else {
                    showMessage(result.error || 'Could not resend code', true);
                }
            } catch (error) {
                showMessage('An error occurred: ' + error.message, true);
            }
        }
    </script>
</body>
</html>
    """
    return {
        "status": 200,
        "headers": {"Content-Type": "text/html"},
        "body": html_content,
    }


async def start_login_handler(workspace: str = None, expires_in: int = None):
    """Start a login session."""
    key = shortuuid.uuid()
    LOGIN_SESSIONS[key] = {
        "status": "pending",
        "workspace": workspace,
        "expires_in": expires_in or 3600,
        "created_at": time.time()
    }
    return {
        "login_url": f"/public/apps/hypha-login/?key={key}",
        "key": key,
        "report_url": "/public/services/hypha-login/report",
        "check_url": "/public/services/hypha-login/check",
    }


async def check_login_handler(key, timeout=180, profile=False):
    """Check login status."""
    if key not in LOGIN_SESSIONS:
        raise ValueError("Invalid login key")
    
    session = LOGIN_SESSIONS[key]
    
    # If timeout is 0, check immediately without waiting
    if timeout == 0:
        if session["status"] == "completed":
            token = session.get("token")
            if profile:
                return {
                    "token": token,
                    "user_id": session.get("user_id"),
                    "email": session.get("email"),
                    "email_verified": session.get("email_verified", True),
                    "name": session.get("name"),
                    "nickname": session.get("nickname"),
                    "picture": session.get("picture"),
                    "workspace": session.get("workspace")
                }
            else:
                return token
        else:
            return None
    
    # Otherwise, wait for the specified timeout
    start_time = time.time()
    while time.time() - start_time < timeout:
        if session["status"] == "completed":
            token = session.get("token")
            if profile:
                return {
                    "token": token,
                    "user_id": session.get("user_id"),
                    "email": session.get("email"),
                    "email_verified": session.get("email_verified", True),
                    "name": session.get("name"),
                    "nickname": session.get("nickname"),
                    "picture": session.get("picture"),
                    "workspace": session.get("workspace")
                }
            else:
                return token
        await asyncio.sleep(1)
    
    raise TimeoutError(f"Login timeout after {timeout} seconds")


async def report_login_handler(
    key,
    token=None,
    workspace=None,
    expires_in=None,
    email=None,
    user_id=None,
    **kwargs
):
    """Report login completion."""
    if key not in LOGIN_SESSIONS:
        raise ValueError("Invalid login key")
    
    session = LOGIN_SESSIONS[key]
    session["status"] = "completed"
    session["token"] = token
    session["user_id"] = user_id
    session["email"] = email
    session["workspace"] = workspace or session.get("workspace")
    
    return {"success": True}


async def _get_users_collection_id(artifact_manager):
    """Return the id of the local-auth users collection, creating it if absent."""
    collection_alias = "ws-user-root/local-auth-users"
    try:
        collection = await artifact_manager.read(collection_alias)
        return collection["id"]
    except Exception:
        collection = await artifact_manager.create(
            workspace="ws-user-root",
            alias="local-auth-users",
            type="collection",
            manifest={
                "name": "Local Authentication Users",
                "description": "User accounts for local authentication",
            },
            # here we can add {"admin-user-id": "*"} for enable user management
            config={"permissions": {}},
        )
        return collection.id if hasattr(collection, "id") else collection["id"]


async def _find_user_by_email(artifact_manager, collection_id, email):
    """Return the user manifest dict for ``email`` (case-insensitive) or None."""
    target = _normalize_email(email)
    all_users = await artifact_manager.list(collection_id)
    for user in all_users:
        manifest = user.get("manifest", {})
        stored = manifest.get("email")
        if stored and _normalize_email(stored) == target:
            return user
    return None


async def _create_user_artifact(artifact_manager, collection_id, name, email, password_hash, salt):
    """Materialise a verified user account and return its user_id."""
    user_id = random_id(readable=True)
    user_data = {
        "id": user_id,
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "salt": salt,
        "created_at": time.time(),
        "email_verified": True,  # only created after email verification succeeds
        "roles": ["user"],
        "workspaces": {},  # User workspaces
    }
    await artifact_manager.create(
        parent_id=collection_id,
        alias=user_id,
        type="user",
        manifest=user_data,
        stage=False,  # Directly commit the user
    )
    return user_id


async def signup_handler(server, context=None, name: str = None, email: str = None, password: str = None):
    """Handle user signup (step 1 of 2: send verification code).

    Validates the input, ensures the email is not already registered, then stores
    a *pending* signup (name + password hash + verification code) and emails the
    code. The account is created only after ``verify_email_handler`` succeeds.

    Returns a dict with ``verification_required: True`` on success. In the dev
    fallback (no RESEND_API_KEY), the response additionally contains ``dev_code``.
    """
    if not all([name, email, password]):
        return {"success": False, "error": "Name, email and password are required"}

    if len(password) < 8:
        return {"success": False, "error": "Password must be at least 8 characters"}

    email = _normalize_email(email)

    try:
        artifact_manager = await server.get_service("public/artifact-manager")
        if not artifact_manager:
            return {"success": False, "error": "Artifact manager not available"}

        collection_id = await _get_users_collection_id(artifact_manager)

        # Reject duplicates BEFORE generating/sending a code.
        existing = await _find_user_by_email(artifact_manager, collection_id, email)
        if existing is not None:
            return {"success": False, "error": "Email already registered"}

        # Generate salt and hash password; store only the hash in the pending entry.
        salt = secrets.token_hex(32)
        password_hash = hash_password(password, salt)
        code = generate_verification_code()

        _PENDING_VERIFICATIONS.put(
            email,
            {
                "name": name,
                "email": email,
                "password_hash": password_hash,
                "salt": salt,
                "code": code,
                "attempts": 0,
            },
        )

        # Send the verification email via the injectable transport.
        transport = get_email_transport()
        await transport.send_verification_email(email, code)

        result = {"success": True, "verification_required": True, "email": email}
        # In the dev fallback ONLY, expose the code so local dev can proceed
        # without an email provider. Never exposed when a real transport is set.
        if getattr(transport, "is_dev_fallback", False):
            result["dev_code"] = code
        return result
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return {"success": False, "error": f"Signup failed: {str(e)}"}


async def verify_email_handler(server, context=None, email: str = None, code: str = None):
    """Handle email verification (step 2 of 2: create the account).

    Checks the submitted code against the pending signup with attempt counting,
    TTL expiry, and max-attempts lockout. On success, materialises the real user
    artifact and clears the pending session.
    """
    if not email or not code:
        return {"success": False, "error": "Email and verification code are required"}

    email = _normalize_email(email)
    entry = _PENDING_VERIFICATIONS.get(email)
    if entry is None:
        # Either never started, already consumed, expired, or evicted.
        return {
            "success": False,
            "error": "No pending verification for this email, or the code has expired. Please sign up again.",
        }

    # Enforce max attempts (lockout): discard the pending session.
    if entry["attempts"] >= MAX_VERIFICATION_ATTEMPTS:
        _PENDING_VERIFICATIONS.delete(email)
        return {
            "success": False,
            "error": "Too many incorrect attempts. Please sign up again.",
        }

    # Constant-time compare of the code.
    if not secrets.compare_digest(str(entry["code"]), str(code)):
        entry["attempts"] += 1
        remaining = MAX_VERIFICATION_ATTEMPTS - entry["attempts"]
        if remaining <= 0:
            _PENDING_VERIFICATIONS.delete(email)
            return {
                "success": False,
                "error": "Too many incorrect attempts. Please sign up again.",
            }
        return {
            "success": False,
            "error": f"Incorrect code. {remaining} attempt(s) remaining.",
        }

    # Code is correct — create the account.
    try:
        artifact_manager = await server.get_service("public/artifact-manager")
        if not artifact_manager:
            return {"success": False, "error": "Artifact manager not available"}

        collection_id = await _get_users_collection_id(artifact_manager)

        # Guard against a race where the email was registered meanwhile.
        existing = await _find_user_by_email(artifact_manager, collection_id, email)
        if existing is not None:
            _PENDING_VERIFICATIONS.delete(email)
            return {"success": False, "error": "Email already registered"}

        user_id = await _create_user_artifact(
            artifact_manager,
            collection_id,
            entry["name"],
            entry["email"],
            entry["password_hash"],
            entry["salt"],
        )
        _PENDING_VERIFICATIONS.delete(email)
        return {"success": True, "user_id": user_id, "email_verified": True}
    except Exception as e:
        logger.error(f"Email verification error: {str(e)}")
        return {"success": False, "error": f"Verification failed: {str(e)}"}


async def resend_code_handler(server, context=None, email: str = None):
    """Resend a verification code for a pending signup, subject to throttling.

    Generates a fresh code (invalidating the previous one) and re-emails it,
    provided the throttle window since the last send has elapsed.
    """
    if not email:
        return {"success": False, "error": "Email is required"}

    email = _normalize_email(email)
    entry = _PENDING_VERIFICATIONS.get(email)
    if entry is None:
        return {
            "success": False,
            "error": "No pending verification for this email. Please sign up again.",
        }

    elapsed = time.time() - entry.get("last_sent_at", 0)
    if elapsed < RESEND_THROTTLE_SECONDS:
        wait = int(RESEND_THROTTLE_SECONDS - elapsed) + 1
        return {
            "success": False,
            "error": f"Please wait {wait}s before requesting another code.",
        }

    # Issue a fresh code and reset the attempt counter.
    code = generate_verification_code()
    entry["code"] = code
    entry["attempts"] = 0
    entry["last_sent_at"] = time.time()

    transport = get_email_transport()
    await transport.send_verification_email(email, code)

    result = {"success": True, "email": email}
    if getattr(transport, "is_dev_fallback", False):
        result["dev_code"] = code
    return result


async def _pending_verification_sweeper(interval: int = 60):
    """Periodically evict expired pending-verification sessions."""
    while True:
        await asyncio.sleep(interval)
        try:
            removed = _PENDING_VERIFICATIONS.sweep_expired()
            if removed:
                logger.debug("Swept %s expired pending verification(s)", removed)
        except Exception as e:  # pragma: no cover - defensive log, never fatal
            logger.error("Pending-verification sweep error: %s", e)


async def login_handler(server, context=None, email: str = None, password: str = None, workspace: str = None, key: str = None):
    """Handle user login."""
    if not email or not password:
        return {"success": False, "error": "Email and password are required"}
    
    try:
        # Get the artifact manager
        artifact_manager = await server.get_service("public/artifact-manager")
        if not artifact_manager:
            return {"success": False, "error": "Artifact manager not available"}
        
        # Get users collection from root workspace
        collection_alias = "ws-user-root/local-auth-users"
        try:
            collection = await artifact_manager.read(collection_alias)
            collection_id = collection["id"]
        except Exception:
            # Collection doesn't exist yet (no users registered)
            return {"success": False, "error": "No users registered yet. Please sign up first."}

        # Find the user with matching email (case-insensitive)
        user = await _find_user_by_email(artifact_manager, collection_id, email)
        if not user:
            return {"success": False, "error": "Invalid email or password"}

        # Get the full user artifact
        user_artifact = await artifact_manager.read(user["id"])
        user_data = user_artifact["manifest"]
        
        # Verify password
        if not verify_password(password, user_data["salt"], user_data["password_hash"]):
            return {"success": False, "error": "Invalid email or password"}
        
        # Create user info and generate token
        user_workspace = workspace or f"ws-user-{user_data['id']}"
        user_info = UserInfo(
            id=user_data["id"],
            is_anonymous=False,
            email=user_data["email"],
            parent=None,
            roles=user_data.get("roles", ["user"]),
            scope=create_scope(
                workspaces={
                    user_workspace: UserPermission.admin,
                    f"ws-user-{user_data['id']}": UserPermission.admin
                },
                current_workspace=user_workspace
            ),
            expires_at=None,
        )
        
        # Generate token
        token = await local_generate_token(user_info, 3600 * 24)  # 24 hours
        
        # If login key provided, update session
        if key and key in LOGIN_SESSIONS:
            LOGIN_SESSIONS[key]["status"] = "completed"
            LOGIN_SESSIONS[key]["token"] = token
            LOGIN_SESSIONS[key]["user_id"] = user_data["id"]
            LOGIN_SESSIONS[key]["email"] = user_data["email"]
            # email_verified reflects the REAL verification state recorded at
            # account creation. Accounts are only created after the emailed code
            # is verified, so this is normally True; legacy accounts predating
            # verification default to True for backward compatibility.
            LOGIN_SESSIONS[key]["email_verified"] = user_data.get("email_verified", True)
            LOGIN_SESSIONS[key]["name"] = user_data.get("name", user_data["email"].split("@")[0])
            LOGIN_SESSIONS[key]["nickname"] = user_data.get("nickname", user_data.get("name", user_data["email"].split("@")[0]))
            LOGIN_SESSIONS[key]["picture"] = user_data.get("picture")  # May be None
            LOGIN_SESSIONS[key]["workspace"] = user_workspace
        
        return {
            "success": True,
            "token": token,
            "user_id": user_data["id"],
            "email": user_data["email"],
            "workspace": user_workspace
        }
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        # Provide more specific error message
        error_msg = str(e)
        if "does not exist" in error_msg:
            return {"success": False, "error": "No users registered yet. Please sign up first."}
        elif "Invalid email or password" in error_msg:
            return {"success": False, "error": "Invalid email or password"}
        else:
            return {"success": False, "error": f"Login failed: {error_msg}"}


async def update_profile_handler(server, context=None, name: str = None, current_password: str = None, new_password: str = None, user_email: str = None):
    """Update user profile."""
    # Since context is not being passed properly in public services,
    # we'll require the user to provide their email for verification
    if not user_email:
        return {"success": False, "error": "Email is required for profile updates"}
    
    try:
        # Get the artifact manager
        artifact_manager = await server.get_service("public/artifact-manager")
        if not artifact_manager:
            return {"success": False, "error": "Artifact manager not available"}
        
        # Get users collection and user artifact from root workspace
        collection_alias = "ws-user-root/local-auth-users"
        try:
            collection = await artifact_manager.read(collection_alias)
            collection_id = collection["id"]
        except:
            # Collection doesn't exist yet (no users registered)
            return {"success": False, "error": "User not found"}
        
        # Find user artifact by email (case-insensitive)
        user = await _find_user_by_email(artifact_manager, collection_id, user_email)
        if not user:
            return {"success": False, "error": "User not found"}

        user_artifact = await artifact_manager.read(user["id"])
        user_data = user_artifact["manifest"]
        
        # Update name if provided
        if name:
            user_data["name"] = name
        
        # Update password if provided
        if new_password:
            if not current_password:
                return {"success": False, "error": "Current password required to change password"}
            
            # Verify current password
            if not verify_password(current_password, user_data["salt"], user_data["password_hash"]):
                return {"success": False, "error": "Current password is incorrect"}
            
            if len(new_password) < 8:
                return {"success": False, "error": "New password must be at least 8 characters"}
            
            # Generate new salt and hash
            user_data["salt"] = secrets.token_hex(32)
            user_data["password_hash"] = hash_password(new_password, user_data["salt"])
        
        # Update the artifact
        await artifact_manager.edit(
            artifact_id=user_artifact["id"],
            manifest=user_data
        )
        
        return {"success": True, "message": "Profile updated successfully"}
        
    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        return {"success": False, "error": "Failed to update profile"}


def local_get_token(scope):
    """Extract token from custom locations for local authentication.
    
    This function allows tokens to be extracted from various sources:
    - X-Local-Auth header
    - hypha_local_token cookie
    - Standard Authorization header (fallback)
    - access_token cookie (fallback)
    
    Args:
        scope: The ASGI scope object containing request information
        
    Returns:
        The extracted token string or None to fall back to default extraction
    """
    headers = scope.get("headers", [])
    
    for key, value in headers:
        # Decode bytes if necessary
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        # Check for custom local auth header
        if key.lower() == "x-local-auth":
            return value
        
        # Check for cookies
        if key.lower() == "cookie":
            # Parse cookies
            cookie_dict = {}
            for cookie in value.split(";"):
                if "=" in cookie:
                    k, v = cookie.split("=", 1)
                    cookie_dict[k.strip()] = v.strip()
            
            # Check for local auth specific cookie
            if "hypha_local_token" in cookie_dict:
                return cookie_dict["hypha_local_token"]
    
    # Return None to fall back to default extraction (Authorization header, access_token cookie)
    return None


async def hypha_startup(server):
    """Startup function for local authentication."""
    logger.info("Initializing local authentication provider")

    # Initialise the email transport from the environment. Logs which mode is
    # active so operators know whether real emails will be sent.
    transport = get_email_transport()
    if getattr(transport, "is_dev_fallback", False):
        logger.warning(
            "Local auth email verification is in DEV FALLBACK mode "
            "(RESEND_API_KEY not set): verification codes are logged, not emailed, "
            "and returned in the signup response. Do NOT use this in production."
        )
    else:
        logger.info("Local auth email verification using Resend transport")

    # Start the background sweeper that evicts abandoned pending verifications.
    global _sweeper_task
    if _sweeper_task is None or _sweeper_task.done():
        _sweeper_task = asyncio.ensure_future(_pending_verification_sweeper())

    # Register the authentication handlers with additional methods
    await server.register_auth_service(
        parse_token=local_parse_token,
        generate_token=local_generate_token,
        get_token=local_get_token,  # Add custom token extraction
        index_handler=index_handler,
        start_handler=start_login_handler,
        check_handler=check_login_handler,
        report_handler=report_login_handler,
        # Profile page handler
        profile=profile_page_handler,
        # Additional handlers for user management - these will be added to the hypha-login service
        # The RPC framework passes context as a parameter
        signup=lambda context=None, **kwargs: signup_handler(server, context, **kwargs),
        verify_email=lambda context=None, **kwargs: verify_email_handler(server, context, **kwargs),
        resend_code=lambda context=None, **kwargs: resend_code_handler(server, context, **kwargs),
        login=lambda context=None, **kwargs: login_handler(server, context, **kwargs),
        update_profile=lambda context=None, **kwargs: update_profile_handler(server, context, **kwargs),
    )

    logger.info("Local authentication provider initialized")