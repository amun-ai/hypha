"""Local authentication provider for Hypha."""

import hashlib
import secrets
import time
import asyncio
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import _parse_token, create_scope
from hypha.utils import random_id
import shortuuid
import logging

logger = logging.getLogger(__name__)

# In-memory storage for login sessions (production should use Redis)
LOGIN_SESSIONS = {}


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
                    showMessage('Account created successfully! You can now login.');
                    setTimeout(() => showTab('login'), 2000);
                } else {
                    showMessage(result.error || 'Signup failed', true);
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


async def signup_handler(server, context=None, name: str = None, email: str = None, password: str = None):
    """Handle user signup."""
    if not all([name, email, password]):
        return {"success": False, "error": "Name, email and password are required"}
    
    if len(password) < 8:
        return {"success": False, "error": "Password must be at least 8 characters"}
    
    try:
        # Get the artifact manager to store user data
        artifact_manager = await server.get_service("public/artifact-manager")
        if not artifact_manager:
            return {"success": False, "error": "Artifact manager not available"}
        
        # Check if user collection exists, create if not
        # Store in the root user's workspace (ws-user-root)
        collection_alias = "ws-user-root/local-auth-users"
        collection_id = None
        try:
            # Try to read the collection first
            collection = await artifact_manager.read(collection_alias)
            collection_id = collection["id"]
        except:
            # Create the collection if it doesn't exist in root workspace
            collection = await artifact_manager.create(
                workspace="ws-user-root",
                alias="local-auth-users",
                type="collection",
                manifest={
                    "name": "Local Authentication Users",
                    "description": "User accounts for local authentication",
                },
                # here we can add {"admin-user-id": "*"} for enable user managment
                config={"permissions": {}}
            )
            collection_id = collection.id if hasattr(collection, 'id') else collection["id"]
        
        # Check if email already exists by listing all users and filtering
        # Note: artifact manager doesn't support nested key filtering
        all_users = await artifact_manager.list(collection_id)
        
        # Check if email already exists
        for user in all_users:
            if user.get("manifest", {}).get("email") == email:
                return {"success": False, "error": "Email already registered"}
        
        # Generate salt and hash password
        salt = secrets.token_hex(32)
        password_hash = hash_password(password, salt)
        
        # Create user artifact
        # Generate a readable user ID using random_id
        user_id = random_id(readable=True)
        user_data = {
            "id": user_id,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "salt": salt,
            "created_at": time.time(),
            "roles": ["user"],
            "workspaces": {}  # User workspaces
        }
        
        # Store user in artifact manager
        user_artifact = await artifact_manager.create(
            parent_id=collection_id,
            alias=user_id,
            type="user", 
            manifest=user_data,
            stage=False  # Directly commit the user
        )
        
        return {"success": True, "user_id": user_id}
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return {"success": False, "error": f"Signup failed: {str(e)}"}


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
        except:
            # Collection doesn't exist yet (no users registered)
            return {"success": False, "error": "No users registered yet. Please sign up first."}
        
        # Find user by email
        all_users = await artifact_manager.list(collection_id)
        
        # Find the user with matching email
        user_artifact_id = None
        for user in all_users:
            if user.get("manifest", {}).get("email") == email:
                user_artifact_id = user["id"]
                break
        
        if not user_artifact_id:
            return {"success": False, "error": "Invalid email or password"}
        
        # Get the full user artifact
        user_artifact = await artifact_manager.read(user_artifact_id)
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
            LOGIN_SESSIONS[key]["email_verified"] = True  # Local auth emails are always verified
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
        
        # Find user artifact by email
        all_users = await artifact_manager.list(collection_id)
        
        # Find the user with matching email
        user_artifact_id = None
        user_id = None
        for user in all_users:
            if user.get("manifest", {}).get("email") == user_email:
                user_artifact_id = user["id"]
                user_id = user.get("manifest", {}).get("id")
                break
        
        if not user_artifact_id:
            return {"success": False, "error": "User not found"}
        
        user_artifact = await artifact_manager.read(user_artifact_id)
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
        login=lambda context=None, **kwargs: login_handler(server, context, **kwargs),
        update_profile=lambda context=None, **kwargs: update_profile_handler(server, context, **kwargs),
    )
    
    logger.info("Local authentication provider initialized")