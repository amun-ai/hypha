"""Custom authentication with login service test."""
import time
import json
import asyncio
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope

# Mock session storage for login
LOGIN_SESSIONS = {}
USER_TOKENS = {}

def custom_generate_token(user_info: UserInfo, expires_in: int) -> str:
    """Generate a custom token."""
    token = f"CUSTOM_LOGIN:{user_info.id}:{int(time.time() + expires_in)}"
    USER_TOKENS[token] = user_info
    return token

async def custom_parse_token(token: str) -> UserInfo:
    """Parse a custom token."""
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
            return UserInfo(
                id=user_id,
                is_anonymous=False,
                email=f"{user_id}@custom-login.com",
                parent=None,
                roles=["user"],
                scope=create_scope(
                    workspaces={"custom-login-workspace": UserPermission.admin}
                ),
                expires_at=None,
            )
    else:
        # Fall back to default JWT parsing
        from hypha.core.auth import _parse_token
        return _parse_token(token)

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
    
    # If no token provided, generate one
    if not token:
        user_info = UserInfo(
            id=user_id or "custom-user",
            email=email or "user@custom-login.com",
            roles=["user"],
            scope=create_scope(
                workspaces={workspace or "custom-workspace": UserPermission.admin}
            ),
        )
        token = custom_generate_token(user_info, expires_in or session["expires_in"])
    
    session["status"] = "completed"
    session["token"] = token
    session["user_id"] = user_id
    session["email"] = email
    session["workspace"] = workspace or session["workspace"]
    
    return {"success": True}

async def hypha_startup(server):
    """Startup function with custom login service."""
    # Register complete custom authentication with login service
    await server.register_auth_service(
        parse_token=custom_parse_token,
        generate_token=custom_generate_token,
        index_handler=custom_index,
        start_handler=custom_start_login,
        check_handler=custom_check_login,
        report_handler=custom_report_login,
    )
    
    # Register a test service
    await server.register_service(
        {
            "id": "custom-login-test",
            "name": "Custom Login Test Service",
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "test": lambda x: f"Custom login test: {x}",
        }
    )
    
    print("Custom authentication with login service configured")