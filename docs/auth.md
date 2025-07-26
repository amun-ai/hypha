# Authentication Service for Hypha

## Setup Auth0 Authentication

By default, Hypha uses auth0 to manage authentication. This allows us to use a variety of authentication providers, including Google, GitHub.

The default setting in hypha uses common auth0 setting managed by us, but you can also setup your own auth0 account and use it.

To set up your own account, follow these steps:
 - go to https://auth0.com/ and create an account, or re-use an existing Github or Google Account.
 - For the first time, you will be asked to create a "Tenant Domain" and choose a "Region", choose any name for the domain (e.g. hypha), and choose a suitable for the region (e.g. US or EU). Then click "Create".
 - After that you should be logged in to the auth0 dashboard. Click on "Applications" on the left menu, and then click on "Create Application".
 - Give your application a name (e.g. hypha), and choose "Single Page Web Applications" as the application type. Then click "Create".
 - Now go to the "Settings" tab of your application, and copy the "Domain" and "Client ID" values to create environment variables for running Hypha:
 ```
 AUTH0_CLIENT_ID=paEagfNXPBVw8Ss80U5RAmAV4pjCPsD2 # replace with your own value from the "Settings" tab
 AUTH0_DOMAIN=amun-ai.eu.auth0.com # replace with your own value from the "Settings" tab
 AUTH0_AUDIENCE=https://amun-ai.eu.auth0.com/api/v2/ # replace 'amun-ai.eu.auth0.com' to your own auth0 domain
 AUTH0_ISSUER=https://amun.ai/ # keep it or replace 'amun.ai' to any website you want to use as the issuer
 AUTH0_NAMESPACE=https://amun.ai/ # keep it or replace 'amun.ai' to any identifier you want to use as the namespace
 ```
 
 You can either set the environment variables in your system, or create a `.env` file in the root directory of Hypha, and add the above lines to the file.
 - Importantly, you also need to configure your own hypha server domain so Auth0 will allow it to login from your own domain. 
 For example, if you want to serve hypha server at https://my-org.com, you need to set the following in "Settings" tab:
    * scroll down to the "Allowed Callback URLs" section, and add the following URLs: https://my-org.com
    * scroll down to the "Allowed Logout URLs" section, and add the following URLs: https://my-org.com/public/apps/hypha-login/
    * scroll down to the "Allowed Web Origins" section, and add the following URLs: https://my-org.com
    * scroll down to the "Allowed Origins (CORS)" section, and add the following URLs: https://my-org.com
 For local development, you can also add `http://127.0.0.1:9527` to the above URLs, separated by comma. For example, "Allowed Callback URLs" can be `https://my-org.com,http://http://127.0.0.1:9527`.
 - Now you can start the hypha server (with the AUTH0 environment variables, via `python3 -m hypha.server --host=0.0.0.0 --port=9527`), and you should be able to test it by going to https://my-org.com/public/apps/hypha-login/ (replace with your own domain) or http://127.0.0.1:9527/public/apps/hypha-login.
 - By default, auth0 will provide a basic username-password-authentication which will store user information at auth0. You can also add other authentication providers (e.g. Google, Github) in the "Authenticaiton" tab of your application in Auth0 dashboard.
    * In order to add Google, click "Social", click "Create Connection", find Google/Gmail, and click "Continue", you will need to obtain the Client ID by following the instructions in the "How to obtain a Client ID" below the "Client ID" field.
    * Similarily, you can add Github by clicking "Social", click "Create Connection", find Github, and click "Continue", you will need to obtain the Client ID by following the instructions in the "How to obtain a Client ID" below the "Client ID" field. In the permissions section, it is recommended to check "Email address" so that Hypha can get the email address of the user.
    * Feel free to also customize the login page, and other settings in Auth0 dashboard.
 - Hypha also dependent on custom `roles` and `email` added in the JWT token by Auth0. You can add custom claims by installing a custom action in the login flow. 
    * Go to "Actions" in the Auth0 dashboard, and click "Create Action".
    * Choose "Create a new action", and choose "Login" as the trigger, then click "Create".
    * Give it a name, e.g. "Add Roles" and then in the code editor, replace the code with the following code:
    ```javascript
    exports.onExecutePostLogin = async (event, api) => {
      const namespace = 'https://amun.ai'; // replace with your own namespace, i.e. same as the AUTH0_NAMESPACE you set in the environment variables
      if (event.authorization) {
         api.idToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
         api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
         api.accessToken.setCustomClaim(`${namespace}/email`, event.user.email);
         if(!event.user.email_verified){
         api.access.deny(`Access to ${event.client.name} is not allowed, please verify your email.`);
         }
      }
   };
    ```
    * Click "Deploy", and you can then drag and drop the action to the middle of the "Login" flow in the "Actions" tab. Make sure you have "Start" -> "Add Roles" -> "Complete" in the flow.
    * Now you should be able to see the `roles` and `email` in the JWT token when you login to Hypha.

## Custom Authentication

Hypha supports custom authentication providers through its extensible authentication system. This allows you to integrate your own authentication mechanisms (SAML, SSO, OAuth, Cloudflare Access, etc.) seamlessly.

The custom auth provider system allows you to:
- Extract authentication tokens from custom headers (e.g., Cloudflare's CF-Token)
- Validate tokens using your own logic
- Convert external user identities to Hypha's UserInfo format
- Provide custom login pages/flows
- Maintain compatibility with existing Auth0-based authentication

### Security

For security reasons, auth providers can **only** be registered during server startup via startup functions. This prevents malicious code from hijacking the authentication system at runtime.

### Auth Provider Interface

An auth provider is a service with type `auth-provider` that implements the following methods:

**Important**: The authentication system supports multiple methods:
- **Headers**: Custom headers like `CF-Token`, `X-API-Key`
- **Cookies**: Via the standard `access_token` cookie
- **Bearer tokens**: Standard `Authorization: Bearer <token>` header
- **Query parameters**: For WebSocket connections
- **Session tokens**: Can be implemented with cookie storage

#### Required Methods

##### `extract_token_from_headers(headers: dict) -> str | None`
Extract authentication token from HTTP headers, cookies, or other sources.

**Note**: The headers dictionary includes all HTTP headers including cookies. Cookie values can be extracted from the `cookie` header.

```python
async def extract_token_from_headers(headers):
    """Extract token from various sources."""
    # 1. Check for custom headers (case-insensitive)
    for key, value in headers.items():
        if key.lower() == "cf-token":
            return value
        if key.lower() == "x-api-key":
            return f"api-key:{value}"
    
    # 2. Check cookies
    cookie_header = headers.get("cookie", "")
    if cookie_header:
        cookies = dict(c.split("=") for c in cookie_header.split("; ") if "=" in c)
        if "session_token" in cookies:
            return f"session:{cookies['session_token']}"
    
    # 3. Check for custom Bearer token format
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer CUSTOM:"):
        return auth.replace("Bearer CUSTOM:", "")
    
    return None
```

##### `validate_token(token: str) -> UserInfo`
Validate the token and return a UserInfo object.

```python
async def validate_token(token):
    """Validate the token and return UserInfo."""
    # Validate token with your auth system
    if not is_valid_token(token):
        raise ValueError("Invalid token")
    
    # Extract user information
    user_data = decode_token(token)
    
    # Create UserInfo object
    user_info = UserInfo(
        id=user_data["user_id"],
        is_anonymous=False,
        email=user_data["email"],
        parent=None,
        roles=user_data.get("roles", []),
        scope=create_scope(
            workspaces={
                f"ws-{user_data['user_id']}": UserPermission.admin,
                "public": UserPermission.read,
            },
            current_workspace=f"ws-{user_data['user_id']}",
        ),
        expires_at=user_data.get("expires_at"),
    )
    return user_info
```

#### Optional Methods

##### `get_login_redirect() -> str | None`
Return a custom login URL to redirect users to.

```python
async def get_login_redirect():
    """Get custom login redirect URL."""
    return "https://your-auth-system.com/login"
```

### Complete Examples

#### API Key Authentication

```python
# api_key_auth_provider.py
import logging
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import hashlib
import json

logger = logging.getLogger("api-key-auth")

# In production, store these securely
API_KEYS = {
    "test-api-key-123": {
        "user_id": "api-user-1",
        "email": "api1@example.com",
        "permissions": ["read", "write"]
    },
    "readonly-key-456": {
        "user_id": "api-user-2",
        "email": "api2@example.com",
        "permissions": ["read"]
    }
}

async def hypha_startup(server):
    """Register API key authentication provider."""
    logger.info("Registering API key auth provider")
    
    async def extract_token_from_headers(headers):
        """Extract API key from headers."""
        # Check X-API-Key header
        api_key = headers.get("x-api-key")
        if api_key:
            return api_key
        
        # Check Authorization header for API key
        auth = headers.get("authorization", "")
        if auth.startswith("ApiKey "):
            return auth.replace("ApiKey ", "")
        
        return None
    
    async def validate_token(token):
        """Validate API key and return UserInfo."""
        if token not in API_KEYS:
            raise ValueError("Invalid API key")
        
        key_info = API_KEYS[token]
        
        # Determine permissions based on key permissions
        workspace_perms = {}
        if "write" in key_info["permissions"]:
            workspace_perms[f"ws-{key_info['user_id']}"] = UserPermission.admin
        else:
            workspace_perms[f"ws-{key_info['user_id']}"] = UserPermission.read
        
        workspace_perms["public"] = UserPermission.read
        
        return UserInfo(
            id=key_info["user_id"],
            is_anonymous=False,
            email=key_info["email"],
            parent=None,
            roles=["api-user"],
            scope=create_scope(
                workspaces=workspace_perms,
                current_workspace=f"ws-{key_info['user_id']}",
            ),
            expires_at=None,
        )
    
    # Register the auth provider
    await server.register_service({
        "id": "api-key-auth-provider",
        "name": "API Key Auth Provider",
        "type": "auth-provider",
        "config": {"visibility": "public", "singleton": True},
        "extract_token_from_headers": extract_token_from_headers,
        "validate_token": validate_token,
    })
    
    logger.info("API key auth provider registered")
```

#### Session-Based Authentication

```python
# session_auth_provider.py
import logging
import secrets
import time
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope
from fastapi import FastAPI, Request, Response, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

logger = logging.getLogger("session-auth")

# In-memory session store (use Redis in production)
SESSIONS = {}

# Simple user database (use proper database in production)
USERS = {
    "admin": {"password": "admin123", "email": "admin@example.com", "role": "admin"},
    "user": {"password": "user123", "email": "user@example.com", "role": "user"},
}

async def hypha_startup(server):
    """Register session-based authentication provider."""
    logger.info("Registering session auth provider")
    
    # Create login app
    login_app = FastAPI()
    
    @login_app.get("/")
    async def login_page():
        """Session login page."""
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                .login-form { max-width: 300px; margin: 0 auto; }
                input { width: 100%; padding: 8px; margin: 5px 0; }
                button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; }
            </style>
        </head>
        <body>
            <div class="login-form">
                <h2>Login</h2>
                <form method="post" action="/public/apps/session-login/login">
                    <input type="text" name="username" placeholder="Username" required>
                    <input type="password" name="password" placeholder="Password" required>
                    <button type="submit">Login</button>
                </form>
            </div>
        </body>
        </html>
        """)
    
    @login_app.post("/login")
    async def login(response: Response, username: str = Form(), password: str = Form()):
        """Handle login and create session."""
        user = USERS.get(username)
        if not user or user["password"] != password:
            return JSONResponse({"error": "Invalid credentials"}, status_code=401)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        SESSIONS[session_id] = {
            "user_id": username,
            "email": user["email"],
            "role": user["role"],
            "created_at": time.time()
        }
        
        # Set session cookie
        response.set_cookie(
            key="session_token",
            value=session_id,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=86400  # 24 hours
        )
        
        return RedirectResponse(url="/", status_code=303)
    
    @login_app.get("/logout")
    async def logout(response: Response):
        """Handle logout."""
        response.delete_cookie("session_token")
        return RedirectResponse(url="/public/apps/session-login/")
    
    async def serve_login_app(args):
        """ASGI wrapper for the login app."""
        await login_app(args["scope"], args["receive"], args["send"])
    
    # Register login app
    await server.register_service({
        "id": "session-login",
        "name": "Session Login",
        "type": "asgi",
        "config": {"visibility": "public"},
        "serve": serve_login_app,
    })
    
    async def extract_token_from_headers(headers):
        """Extract session token from cookies."""
        cookie_header = headers.get("cookie", "")
        if not cookie_header:
            return None
        
        # Parse cookies
        cookies = {}
        for cookie in cookie_header.split("; "):
            if "=" in cookie:
                key, value = cookie.split("=", 1)
                cookies[key] = value
        
        return cookies.get("session_token")
    
    async def validate_token(token):
        """Validate session token and return UserInfo."""
        session = SESSIONS.get(token)
        if not session:
            raise ValueError("Invalid or expired session")
        
        # Check session expiry (24 hours)
        if time.time() - session["created_at"] > 86400:
            del SESSIONS[token]
            raise ValueError("Session expired")
        
        # Create UserInfo based on role
        is_admin = session["role"] == "admin"
        workspace_id = f"ws-{session['user_id']}"
        
        return UserInfo(
            id=session["user_id"],
            is_anonymous=False,
            email=session["email"],
            parent=None,
            roles=[session["role"]],
            scope=create_scope(
                workspaces={
                    workspace_id: UserPermission.admin,
                    "public": UserPermission.read,
                    "*": UserPermission.admin if is_admin else None,
                },
                current_workspace=workspace_id,
            ),
            expires_at=None,
        )
    
    async def get_login_redirect():
        """Redirect to session login page."""
        return "/public/apps/session-login/"
    
    # Register auth provider
    await server.register_service({
        "id": "session-auth-provider",
        "name": "Session Auth Provider",
        "type": "auth-provider",
        "config": {"visibility": "public", "singleton": True},
        "extract_token_from_headers": extract_token_from_headers,
        "validate_token": validate_token,
        "get_login_redirect": get_login_redirect,
    })
    
    logger.info("Session auth provider registered")
```

### Complete Example - Cloudflare Access

Here's a complete example of a Cloudflare Access auth provider:

```python
# custom_auth_provider.py
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from hypha.core import UserInfo, create_scope, UserPermission
from datetime import datetime, timedelta
import httpx

logger = logging.getLogger("custom-auth-provider")

async def hypha_startup(api):
    """Startup function to register custom auth provider."""
    logger.info("Registering Cloudflare Access auth provider")
    # Create login app
    login_app = FastAPI()
    
    @login_app.get("/")
    async def login_page():
        """Custom login page."""
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cloudflare Access Login</title>
        </head>
        <body>
            <h1>Cloudflare Access Login</h1>
            <p>You will be redirected to Cloudflare Access...</p>
            <script>
                window.location.href = 'https://yourdomain.cloudflareaccess.com/login';
            </script>
        </body>
        </html>
        """)
    
    async def serve_login_app(args):
        """ASGI wrapper for the login app."""
        await login_app(args["scope"], args["receive"], args["send"])
    
    # Register login app
    await api.register_service({
        "id": "cf-access-login",
        "name": "Cloudflare Access Login",
        "type": "asgi",
        "config": {"visibility": "public"},
        "serve": serve_login_app,
    })
    
    # Auth provider methods
    async def extract_token_from_headers(headers):
        """Extract Cloudflare Access JWT from headers."""
        # Cloudflare Access puts JWT in Cf-Access-Jwt-Assertion header
        return headers.get("cf-access-jwt-assertion")
    
    async def validate_token(token):
        """Validate Cloudflare Access JWT."""
        # In production, validate JWT signature with CF public keys
        # This is a simplified example
        try:
            # Decode JWT (verify signature in production!)
            import jwt
            payload = jwt.decode(token, options={"verify_signature": False})
            
            # Extract user info from CF Access JWT
            user_id = payload.get("sub", "unknown")
            email = payload.get("email", "")
            
            user_info = UserInfo(
                id=f"cf-{user_id}",
                is_anonymous=False,
                email=email,
                parent=None,
                roles=["cloudflare-user"],
                scope=create_scope(
                    workspaces={
                        f"ws-cf-{user_id}": UserPermission.admin,
                        "public": UserPermission.read,
                    },
                    current_workspace=f"ws-cf-{user_id}",
                ),
                expires_at=payload.get("exp"),
            )
            return user_info
            
        except Exception as e:
            logger.error(f"Failed to validate CF Access token: {e}")
            raise ValueError(f"Invalid Cloudflare Access token: {e}")
    
    async def get_login_redirect():
        """Redirect to Cloudflare Access login."""
        return "https://yourdomain.cloudflareaccess.com/login"
    
    # Register auth provider
    auth_service = await api.register_service({
        "id": "cloudflare-access-provider",
        "name": "Cloudflare Access Auth Provider",
        "type": "auth-provider",  # Special type - requires startup context
        "config": {
            "visibility": "public",
            "singleton": True,  # Only one auth provider allowed
        },
        "extract_token_from_headers": extract_token_from_headers,
        "validate_token": validate_token,
        "get_login_redirect": get_login_redirect,
    })
    
    logger.info(f"Cloudflare Access auth provider registered: {auth_service['id']}")
```

### Starting Hypha with Custom Auth

To use your custom auth provider, start Hypha with the `--startup-functions` parameter:

```bash
python -m hypha.server \
    --port=9527 \
    --startup-functions=./custom_auth_provider.py:hypha_startup
```

### Authentication Flow

#### HTTP Requests

1. Client makes HTTP request with custom headers
2. Hypha middleware calls `extract_token_from_headers()` to find token
3. If token found, calls `validate_token()` to get UserInfo
4. Request proceeds with authenticated user context
5. If validation fails, falls back to default auth or anonymous

#### WebSocket Connections

1. Client connects with token in query parameter or first message
2. If token starts with your custom prefix (e.g., "CF:"), auth provider handles it
3. Otherwise, default auth is used
4. Connection established with authenticated user context

#### Login Flow

1. User visits `/login`
2. If auth provider defines `get_login_redirect()`, redirects there
3. Otherwise, shows default Hypha login page
4. Custom login page can be provided as an ASGI service

### Best Practices

1. **Token Validation**: Always validate tokens properly, including signature verification
2. **Error Handling**: Return clear error messages for debugging
3. **Fallback**: Allow fallback to default auth when your provider doesn't handle a token
4. **Workspace Mapping**: Map external users to appropriate Hypha workspaces
5. **Permissions**: Set appropriate permissions based on user roles
6. **Caching**: Cache validated tokens to reduce auth system load
7. **Logging**: Log authentication events for security monitoring

### Advanced Integration Examples

#### OAuth2/OIDC with Refresh Tokens

```python
# oauth2_auth_provider.py
import logging
import httpx
import jwt
from datetime import datetime, timedelta
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse

logger = logging.getLogger("oauth2-auth")

# OAuth2 configuration
OAUTH2_CLIENT_ID = "your-client-id"
OAUTH2_CLIENT_SECRET = "your-client-secret"
OAUTH2_AUTHORIZE_URL = "https://oauth.provider.com/authorize"
OAUTH2_TOKEN_URL = "https://oauth.provider.com/token"
OAUTH2_USERINFO_URL = "https://oauth.provider.com/userinfo"
OAUTH2_REDIRECT_URI = "https://your-hypha-server.com/public/apps/oauth2-callback/callback"

# Token cache (use Redis in production)
TOKEN_CACHE = {}

async def hypha_startup(server):
    """Register OAuth2 authentication provider."""
    logger.info("Registering OAuth2 auth provider")
    
    # Create OAuth2 app
    oauth_app = FastAPI()
    
    @oauth_app.get("/")
    async def oauth_login():
        """Redirect to OAuth2 provider."""
        auth_url = (
            f"{OAUTH2_AUTHORIZE_URL}?"
            f"client_id={OAUTH2_CLIENT_ID}&"
            f"redirect_uri={OAUTH2_REDIRECT_URI}&"
            f"response_type=code&"
            f"scope=openid email profile"
        )
        return RedirectResponse(url=auth_url)
    
    @oauth_app.get("/callback")
    async def oauth_callback(code: str, state: str = None):
        """Handle OAuth2 callback."""
        async with httpx.AsyncClient() as client:
            # Exchange code for tokens
            token_response = await client.post(
                OAUTH2_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": OAUTH2_REDIRECT_URI,
                    "client_id": OAUTH2_CLIENT_ID,
                    "client_secret": OAUTH2_CLIENT_SECRET,
                }
            )
            
            if token_response.status_code != 200:
                return JSONResponse({"error": "Failed to get tokens"}, status_code=400)
            
            tokens = token_response.json()
            access_token = tokens["access_token"]
            refresh_token = tokens.get("refresh_token")
            
            # Get user info
            user_response = await client.get(
                OAUTH2_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if user_response.status_code != 200:
                return JSONResponse({"error": "Failed to get user info"}, status_code=400)
            
            user_info = user_response.json()
            
            # Store tokens
            user_id = user_info["sub"]
            TOKEN_CACHE[f"oauth-{user_id}"] = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "user_info": user_info,
                "expires_at": datetime.utcnow() + timedelta(hours=1)
            }
            
            # Create a session token
            session_token = jwt.encode(
                {"sub": user_id, "type": "oauth2"},
                "your-secret-key",
                algorithm="HS256"
            )
            
            # Redirect with session token
            response = RedirectResponse(url="/")
            response.set_cookie(
                key="oauth_session",
                value=session_token,
                httponly=True,
                secure=True,
                samesite="lax"
            )
            return response
    
    async def serve_oauth_app(args):
        """ASGI wrapper for the OAuth app."""
        await oauth_app(args["scope"], args["receive"], args["send"])
    
    # Register OAuth callback app
    await server.register_service({
        "id": "oauth2-callback",
        "name": "OAuth2 Callback",
        "type": "asgi",
        "config": {"visibility": "public"},
        "serve": serve_oauth_app,
    })
    
    async def extract_token_from_headers(headers):
        """Extract OAuth session token."""
        # Check for direct bearer token
        auth = headers.get("authorization", "")
        if auth.startswith("Bearer oauth:"):
            return auth.replace("Bearer oauth:", "")
        
        # Check cookies
        cookie_header = headers.get("cookie", "")
        if cookie_header:
            cookies = dict(c.split("=") for c in cookie_header.split("; ") if "=" in c)
            return cookies.get("oauth_session")
        
        return None
    
    async def validate_token(token):
        """Validate OAuth token and refresh if needed."""
        try:
            # Decode session token
            payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
            user_id = payload["sub"]
            
            # Get cached tokens
            cache_key = f"oauth-{user_id}"
            cached = TOKEN_CACHE.get(cache_key)
            
            if not cached:
                raise ValueError("No cached OAuth tokens")
            
            # Check if access token expired
            if datetime.utcnow() > cached["expires_at"]:
                # Refresh the token
                async with httpx.AsyncClient() as client:
                    refresh_response = await client.post(
                        OAUTH2_TOKEN_URL,
                        data={
                            "grant_type": "refresh_token",
                            "refresh_token": cached["refresh_token"],
                            "client_id": OAUTH2_CLIENT_ID,
                            "client_secret": OAUTH2_CLIENT_SECRET,
                        }
                    )
                    
                    if refresh_response.status_code == 200:
                        new_tokens = refresh_response.json()
                        cached["access_token"] = new_tokens["access_token"]
                        cached["expires_at"] = datetime.utcnow() + timedelta(hours=1)
                        if "refresh_token" in new_tokens:
                            cached["refresh_token"] = new_tokens["refresh_token"]
                    else:
                        raise ValueError("Failed to refresh token")
            
            # Create UserInfo
            user_info_data = cached["user_info"]
            return UserInfo(
                id=f"oauth-{user_id}",
                is_anonymous=False,
                email=user_info_data.get("email", ""),
                parent=None,
                roles=["oauth-user"],
                scope=create_scope(
                    workspaces={
                        f"ws-oauth-{user_id}": UserPermission.admin,
                        "public": UserPermission.read,
                    },
                    current_workspace=f"ws-oauth-{user_id}",
                ),
                expires_at=None,
            )
            
        except Exception as e:
            logger.error(f"OAuth validation failed: {e}")
            raise ValueError(f"Invalid OAuth session: {e}")
    
    async def get_login_redirect():
        """Redirect to OAuth2 login."""
        return "/public/apps/oauth2-callback/"
    
    # Register auth provider
    await server.register_service({
        "id": "oauth2-auth-provider",
        "name": "OAuth2 Auth Provider",
        "type": "auth-provider",
        "config": {"visibility": "public", "singleton": True},
        "extract_token_from_headers": extract_token_from_headers,
        "validate_token": validate_token,
        "get_login_redirect": get_login_redirect,
    })
    
    logger.info("OAuth2 auth provider registered")
```

#### SAML Integration

```python
async def validate_token(token):
    """Validate SAML assertion."""
    import onelogin.saml2
    
    # Parse SAML response
    saml_response = onelogin.saml2.Response(token)
    if not saml_response.is_valid():
        raise ValueError("Invalid SAML assertion")
    
    # Extract attributes
    attributes = saml_response.get_attributes()
    user_id = attributes['uid'][0]
    email = attributes['email'][0]
    groups = attributes.get('groups', [])
    
    # Map to UserInfo
    return UserInfo(
        id=f"saml-{user_id}",
        email=email,
        roles=groups,
        # ... rest of UserInfo
    )
```

#### OAuth2/OIDC Integration

```python
async def validate_token(token):
    """Validate OAuth2 access token."""
    async with httpx.AsyncClient() as client:
        # Introspect token
        response = await client.post(
            "https://oauth.provider.com/introspect",
            data={"token": token, "client_id": CLIENT_ID},
            auth=(CLIENT_ID, CLIENT_SECRET)
        )
        
        if not response.json().get("active"):
            raise ValueError("Token is not active")
        
        # Get user info
        user_response = await client.get(
            "https://oauth.provider.com/userinfo",
            headers={"Authorization": f"Bearer {token}"}
        )
        user_data = user_response.json()
        
        return UserInfo(
            id=f"oauth-{user_data['sub']}",
            email=user_data['email'],
            # ... rest of UserInfo
        )
```

### Client Usage Examples

#### Using API Keys

```python
# Python client
from hypha_rpc import connect_to_server

# Connect with API key
api = await connect_to_server({
    "server_url": "https://your-server.com",
    "token": "test-api-key-123",  # Will be sent as X-API-Key header
})
```

```javascript
// JavaScript client
import { connectToServer } from 'hypha-rpc';

// Connect with API key
const api = await connectToServer({
    serverUrl: 'https://your-server.com',
    token: 'test-api-key-123',  // Will be sent as X-API-Key header
});
```

```bash
# HTTP API with API key
curl -H "X-API-Key: test-api-key-123" \
     https://your-server.com/workspace/services/my-service/method
```

#### Using Session Authentication

```javascript
// Browser-based session authentication
// After login, the session cookie is automatically included
const response = await fetch('https://your-server.com/workspace/services/my-service/method', {
    method: 'POST',
    credentials: 'include',  // Include cookies
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({ param: 'value' })
});
```

### Troubleshooting

#### Auth provider not being called
- Ensure service type is exactly `"auth-provider"`
- Check that the service is registered in the public workspace
- Verify startup function is executing without errors

#### Token validation failures
- Check token format and encoding
- Verify signature validation logic
- Ensure UserInfo object has all required fields

#### WebSocket authentication issues
- Token must be passed in query parameter or first message
- Check for proper token prefix handling
- Verify scope and workspace settings in UserInfo

### Security Considerations

1. **Startup-only Registration**: Auth providers can only be registered during startup to prevent runtime hijacking
2. **Token Validation**: Always validate token signatures and expiration
3. **Secure Communication**: Use HTTPS/WSS in production
4. **Rate Limiting**: Implement rate limiting for auth endpoints
5. **Audit Logging**: Log all authentication events
6. **Token Rotation**: Support token refresh/rotation mechanisms