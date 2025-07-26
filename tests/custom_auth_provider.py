"""Custom authentication provider for testing."""
import sys
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope
from datetime import datetime, timedelta

logger = logging.getLogger("custom-auth-provider")


async def hypha_startup(server):
    """Startup function to register custom auth provider."""
    import sys
    print(f"[CUSTOM AUTH] Starting custom auth provider registration", file=sys.stderr)
    logger.info("Registering custom Cloudflare auth provider")
    
    # Create a simple ASGI app for login
    login_app = FastAPI()
    
    @login_app.get("/")
    async def login_page():
        """Custom login page."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Custom Cloudflare Login</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                .container { max-width: 600px; margin: 0 auto; }
                .login-box { border: 1px solid #ccc; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Custom Cloudflare Login</h1>
                <div class="login-box">
                    <p>This is a custom login page for Cloudflare authentication.</p>
                    <p>Enter your Cloudflare token:</p>
                    <input type="text" id="token" placeholder="CF Token" style="width: 300px; padding: 5px;">
                    <button onclick="login()">Login</button>
                    <div id="result"></div>
                </div>
            </div>
            <script>
                async function login() {
                    const token = document.getElementById('token').value;
                    const response = await fetch('/public/apps/custom-auth-login/api/login', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({token: token})
                    });
                    const result = await response.json();
                    document.getElementById('result').innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @login_app.post("/api/login")
    async def api_login(request: Request):
        """Handle login API requests."""
        data = await request.json()
        token = data.get("token", "")
        
        # Validate token
        if token.startswith("test-cloudflare-token-"):
            user_id = "cf-user-123"
            return JSONResponse({
                "success": True,
                "user": {
                    "id": user_id,
                    "email": "user@cloudflare.com",
                    "name": "Cloudflare User"
                }
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Invalid token"
            }, status_code=401)
    
    async def serve_login_app(args):
        """ASGI wrapper for the login app."""
        await login_app(args["scope"], args["receive"], args["send"])
    
    # Register login app as ASGI service
    await server.register_service({
        "id": "custom-auth-login",
        "name": "Custom Auth Login",
        "type": "asgi",
        "config": {
            "visibility": "public",
        },
        "serve": serve_login_app,
    })
    
    # Define auth provider methods
    async def extract_token_from_headers(headers):
        """Extract token from custom headers."""
        print(f"[CUSTOM AUTH] extract_token_from_headers called", file=sys.stderr)
        print(f"[CUSTOM AUTH] Headers keys: {list(headers.keys())}", file=sys.stderr)
        # Check for Cloudflare token - case insensitive
        for key, value in headers.items():
            print(f"[CUSTOM AUTH] Header {key}: {value[:20]}...", file=sys.stderr)
            if key.lower() == "cf-token":
                print(f"[CUSTOM AUTH] Found CF-Token: {value}", file=sys.stderr)
                return value
        
        # Check for standard authorization but with CF prefix
        auth = headers.get("authorization", "")
        if auth.startswith("Bearer CF:"):
            return auth.replace("Bearer CF:", "")
        
        return None
    
    async def validate_token(token):
        """Validate the token and return UserInfo."""
        # Handle tokens that start with "CF:"
        if token.startswith("CF:"):
            token = token[3:]
        
        # Simple validation for demo
        if token.startswith("test-cloudflare-token-"):
            # Extract user info from token (in real implementation, validate with CF API)
            user_id = "cf-user-123"
            
            user_info = UserInfo(
                id=user_id,
                is_anonymous=False,
                email="user@cloudflare.com",
                parent=None,
                roles=["cloudflare-user"],
                scope=create_scope(
                    workspaces={
                        f"ws-{user_id}": UserPermission.admin,
                        "public": UserPermission.read,
                    },
                    current_workspace=f"ws-{user_id}",
                ),
                expires_at=None,  # No expiration for this demo
            )
            return user_info
        else:
            # Invalid token
            raise ValueError(f"Invalid Cloudflare token: {token}")
    
    async def get_login_redirect():
        """Get custom login redirect URL."""
        return "https://cloudflare.example.com/login"
    
    # Register auth provider service
    auth_service = await server.register_service({
        "id": "cloudflare-auth-provider",
        "name": "Cloudflare Auth Provider",
        "type": "auth-provider",  # Special type that requires startup context
        "config": {
            "visibility": "public",
            "singleton": True,  # Only one auth provider allowed
        },
        "extract_token_from_headers": extract_token_from_headers,
        "validate_token": validate_token,
        "get_login_redirect": get_login_redirect,
    })
    
    logger.info(f"Custom auth provider registered: {auth_service['id']}")
    print(f"[CUSTOM AUTH] Registration completed: {auth_service['id']}", file=sys.stderr)
    return auth_service