"""Custom authentication startup function for testing."""
import time
import json
import asyncio
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope

# Mock session storage for login
LOGIN_SESSIONS = {}
USER_TOKENS = {}

def custom_generate_token(user_info: UserInfo, expires_in: int) -> str:
    """Generate a custom token for testing."""
    # Create a simple custom token format: CUSTOM:user_id:workspace:expires
    # For anonymous users, preserve the original workspace name
    if user_info.scope.workspaces:
        workspace = list(user_info.scope.workspaces.keys())[0]
    elif user_info.scope.current_workspace:
        workspace = user_info.scope.current_workspace
    else:
        workspace = "default"
    expires_at = int(time.time()) + expires_in
    token = f"CUSTOM:{user_info.id}:{workspace}:{expires_at}"
    
    # Store the token for login service validation
    USER_TOKENS[token] = user_info
    
    return token


async def custom_parse_token(token: str) -> UserInfo:
    """Parse a custom token for testing."""
    # Check if it's a login token with stored user info
    if token.startswith("CUSTOM_LOGIN:"):
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError("Invalid custom login token format")
        
        user_id = parts[1]
        expires_at = int(parts[2])
        
        if time.time() > expires_at:
            raise ValueError("Token has expired")
        
        # Return stored user info or create default
        if token in USER_TOKENS:
            return USER_TOKENS[token]
        else:
            # Create default user info for the token
            # Use the workspace from the token if available
            workspace = "custom-login-workspace"
            # Try to extract workspace from stored sessions
            for session in LOGIN_SESSIONS.values():
                if session.get("token") == token:
                    workspace = session.get("workspace", workspace)
                    break
            
            return UserInfo(
                id=user_id,
                is_anonymous=False,
                email=f"{user_id}@custom-login.com",
                parent=None,
                roles=["user"],
                scope=create_scope(
                    workspaces={workspace: UserPermission.admin}
                ),
                expires_at=None,
            )
    
    if not token.startswith("CUSTOM:"):
        # Fall back to default JWT parsing for non-custom tokens
        from hypha.core.auth import _parse_token
        return _parse_token(token)
    
    # Parse our custom token format
    parts = token.split(":")
    if len(parts) != 4:
        raise ValueError("Invalid custom token format")
    
    _, user_id, workspace, expires_at = parts
    
    # Check expiration
    if int(expires_at) < time.time():
        raise ValueError("Token has expired")
    
    # Handle different workspace formats
    # If workspace contains the user ID already (ws-user-xxx), extract the owner
    # Otherwise use the user_id from the token
    if workspace.startswith("ws-user-"):
        # For workspaces like ws-user-anonymouz-xxx, the user should have admin access
        # regardless of the token's user_id (which might be a child token)
        workspace_permissions = {workspace: UserPermission.admin}
    else:
        # For other workspaces, use standard permissions
        workspace_permissions = {workspace: UserPermission.admin}
    
    user_info = UserInfo(
        id=user_id,
        is_anonymous=False,
        email=f"{user_id}@example.com",
        parent=None,
        roles=[],
        scope=create_scope(
            workspaces=workspace_permissions,
            current_workspace=workspace
        ),
        expires_at=None,
    )
    return user_info


async def custom_index(event):
    """Serve custom login page."""
    html_content = """
    <html>
    <head><title>Custom Login</title></head>
    <body>
        <h1>Custom Authentication Login</h1>
        <p>This is a custom login page for testing.</p>
        <form id="loginForm">
            <input type="text" id="username" placeholder="Username" required>
            <input type="password" id="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
    </body>
    </html>
    """
    return {
        "status": 200,
        "headers": {"Content-Type": "text/html"},
        "body": html_content,
    }

async def custom_start_login(workspace: str = None, expires_in: int = None):
    """Start a custom login session."""
    import shortuuid
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

async def custom_check_login(key, timeout=180, profile=False):
    """Check custom login status."""
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

async def custom_report_login(
    key,
    token=None,
    workspace=None,
    expires_in=None,
    email=None,
    user_id=None,
    **kwargs
):
    """Report custom login completion."""
    if key not in LOGIN_SESSIONS:
        raise ValueError("Invalid login key")
    
    session = LOGIN_SESSIONS[key]
    
    # If no token provided, generate one using a special login token format
    if not token:
        final_workspace = workspace or session.get("workspace") or "custom-workspace"
        user_info = UserInfo(
            id=user_id or "custom-user",
            is_anonymous=False,
            email=email or "user@custom-login.com",
            parent=None,
            roles=["user"],
            scope=create_scope(
                workspaces={
                    final_workspace: UserPermission.admin,
                    f"ws-user-{user_id or 'custom-user'}": UserPermission.admin  # Also give access to user's default workspace
                },
                current_workspace=final_workspace
            ),
            expires_at=None,
        )
        # Generate a special login token
        expires_at = int(time.time() + (expires_in or session["expires_in"]))
        token = f"CUSTOM_LOGIN:{user_info.id}:{expires_at}"
        USER_TOKENS[token] = user_info
    
    session["status"] = "completed"
    session["token"] = token
    session["user_id"] = user_id
    session["email"] = email
    session["workspace"] = workspace or session["workspace"]
    
    return {"success": True}

def custom_get_token(scope):
    """Extract token from custom headers or cookies."""
    headers = scope.get("headers", [])
    
    for key, value in headers:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        # Check for custom CF_Authorization header (e.g., from Cloudflare)
        if key.lower() == "cf_authorization":
            return value
        
        # Check for X-Hypha-Token header
        if key.lower() == "x-hypha-token":
            return value
        
        # Check for custom cookie
        if key.lower() == "cookie":
            cookies = value
            # Parse cookies
            cookie_dict = {}
            for cookie in cookies.split(";"):
                if "=" in cookie:
                    k, v = cookie.split("=", 1)
                    cookie_dict[k.strip()] = v.strip()
            
            # Check for hypha_token cookie
            if "hypha_token" in cookie_dict:
                return cookie_dict["hypha_token"]
            
            # Check for cf_token cookie (Cloudflare)
            if "cf_token" in cookie_dict:
                return cookie_dict["cf_token"]
    
    # Fall back to default extraction (Authorization header or access_token cookie)
    return None


async def hypha_startup(server):
    """Startup function with complete custom authentication and login service."""
    # Register complete custom authentication with login service
    await server.register_auth_service(
        parse_token=custom_parse_token,
        generate_token=custom_generate_token,
        get_token=custom_get_token,  # Add custom token extraction
        index_handler=custom_index,
        start_handler=custom_start_login,
        check_handler=custom_check_login,
        report_handler=custom_report_login,
    )
    
    # Register a test service to verify the custom auth is working
    await server.register_service(
        {
            "id": "custom-auth-test",
            "name": "Custom Auth Test Service",
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "echo": lambda msg: f"Custom auth server says: {msg}",
        }
    )
    
    print("Custom authentication with login service configured")