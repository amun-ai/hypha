"""Local authentication provider for Hypha."""

import hashlib
import secrets
import time
import asyncio
import json
from typing import Optional, Dict, Any
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import generate_auth_token, _parse_token, create_scope
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
    return _generate_presigned_token(user_info, expires_in)


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

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const loginKey = urlParams.get('key');
        const workspace = urlParams.get('workspace');

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
                    showMessage('Login successful! Redirecting...');
                    
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
                                workspace: result.workspace
                            })
                        });
                        setTimeout(() => {
                            window.close();
                        }, 1000);
                    } else {
                        // Store token and redirect
                        localStorage.setItem('hypha_token', result.token);
                        setTimeout(() => {
                            window.location.href = '/';
                        }, 1000);
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
        collection_id = None
        try:
            # Try to read the collection first
            collection = await artifact_manager.read("local-auth-users")
            collection_id = collection["id"]
        except:
            # Create the collection if it doesn't exist
            collection = await artifact_manager.create(
                alias="local-auth-users",
                type="collection",
                manifest={
                    "name": "Local Authentication Users",
                    "description": "User accounts for local authentication",
                },
                config={"permissions": {"*": "", "@": "rw+"}}  # Only authenticated users can access
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
        # Generate a user ID that will work with workspace naming (lowercase only)
        user_id = f"u{shortuuid.uuid().lower()}"
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
        
        # Get users collection
        collection = await artifact_manager.read("local-auth-users")
        collection_id = collection["id"]
        
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
        return {"success": False, "error": "Login failed"}


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
        
        # Get users collection and user artifact
        collection = await artifact_manager.read("local-auth-users")
        collection_id = collection["id"]
        
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


async def hypha_startup(server):
    """Startup function for local authentication."""
    logger.info("Initializing local authentication provider")
    
    # Register the authentication handlers with additional methods
    await server["register_auth_service"](
        parse_token=local_parse_token,
        generate_token=local_generate_token,
        index_handler=index_handler,
        start_handler=start_login_handler,
        check_handler=check_login_handler,
        report_handler=report_login_handler,
        # Additional handlers for user management - these will be added to the hypha-login service
        # The RPC framework passes context as a parameter
        signup=lambda context=None, **kwargs: signup_handler(server, context, **kwargs),
        login=lambda context=None, **kwargs: login_handler(server, context, **kwargs),
        update_profile=lambda context=None, **kwargs: update_profile_handler(server, context, **kwargs),
    )
    
    logger.info("Local authentication provider initialized")