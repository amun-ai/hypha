# Authentication

Hypha provides flexible authentication options to secure your services and control access to resources. You can use the built-in Auth0 integration or implement custom authentication providers.

## Native Auth0 Support

By default, Hypha uses [Auth0](https://auth0.com) for authentication, allowing integration with various identity providers including Google, GitHub, and traditional username/password authentication.

### Using the Default Configuration

Hypha comes with a default Auth0 configuration that works out of the box:

```python
from hypha_rpc import login, connect_to_server

# Login to get a token
token = await login({"server_url": "https://ai.imjoy.io"})

# Connect with the token
async with connect_to_server({
    "server_url": "https://ai.imjoy.io",
    "token": token
}) as server:
    # Use authenticated services
    pass
```

### Logging Out

Hypha provides a `logout` function to properly end user sessions. This clears the authentication state on both the client and server side (including Auth0 session).

#### Python Client

```python
from hypha_rpc import logout

# Logout with a callback to handle the logout URL
async def logout_callback(context):
    # The context contains the logout_url for server-side logout
    print(f"Please visit to complete logout: {context['logout_url']}")
    # In a GUI application, you might open this URL in a browser
    import webbrowser
    webbrowser.open(context['logout_url'])

await logout({
    "server_url": "https://ai.imjoy.io",
    "logout_callback": logout_callback
})
```

#### JavaScript Client (Browser)

```javascript
import { hyphaWebsocketClient } from 'hypha-rpc';

// Logout with a callback to handle the logout URL
const logoutCallback = async (context) => {
    // Open logout URL in a popup to complete server-side logout
    if (context.logout_url) {
        window.open(context.logout_url, '_blank', 'width=600,height=700');
    }
};

await hyphaWebsocketClient.logout({
    server_url: window.location.origin,
    logout_callback: logoutCallback,
});

// Also clear local storage and cookies
localStorage.removeItem('hypha_token');
localStorage.removeItem('hypha_user_profile');
document.cookie = 'access_token=; path=/; max-age=0; samesite=lax';
```

#### Complete Logout Flow

For a complete logout that clears both local state and server-side sessions:

```javascript
async function performLogout() {
    try {
        // 1. Call server-side logout
        await hyphaWebsocketClient.logout({
            server_url: window.location.origin,
            logout_callback: async (context) => {
                if (context.logout_url) {
                    // Opens Auth0 logout which clears the IdP session
                    window.open(context.logout_url, '_blank', 'width=600,height=700');
                }
            },
        });
    } catch (e) {
        console.log("Error during server logout:", e);
    }

    // 2. Clear local cookies
    document.cookie = 'access_token=; path=/; max-age=0; samesite=lax';

    // 3. Clear local storage
    localStorage.removeItem('hypha_token');
    localStorage.removeItem('hypha_user_profile');

    // 4. Disconnect from server (optional)
    if (server) {
        await server.disconnect();
    }
}
```

### Setting Up Your Own Auth0 Account

For production deployments, you should configure your own Auth0 account:

#### 1. Create an Auth0 Account

1. Go to [https://auth0.com/](https://auth0.com/) and create an account
2. Create a "Tenant Domain" (e.g., `hypha`) and choose a region
3. Click "Applications" → "Create Application"
4. Name your application (e.g., `hypha`) and choose "Single Page Web Applications"

#### 2. Configure Environment Variables

Copy the following values from your Auth0 application's "Settings" tab:

```bash
# Using environment variables
export AUTH0_CLIENT_ID=paEagfNXPBVw8Ss80U5RAmAV4pjCPsD2  # Your Client ID
export AUTH0_DOMAIN=your-tenant.auth0.com                 # Your Domain
export AUTH0_AUDIENCE=https://your-tenant.auth0.com/api/v2/
export AUTH0_ISSUER=https://your-organization.com/        # Your issuer URL
export AUTH0_NAMESPACE=https://your-organization.com/     # Your namespace

# Or create a .env file in the Hypha root directory
```

#### 3. Configure Allowed URLs

In the Auth0 application settings, configure these URLs for your domain (e.g., `https://my-org.com`):

- **Allowed Callback URLs**: `https://my-org.com`
- **Allowed Logout URLs**: `https://my-org.com/public/apps/hypha-login/`
- **Allowed Web Origins**: `https://my-org.com`
- **Allowed Origins (CORS)**: `https://my-org.com`

For local development, add `http://127.0.0.1:9527` to each field (comma-separated).

#### 4. Add Custom Claims Action

Hypha requires custom `roles` and `email` claims in JWT tokens. Add them via Auth0 Actions:

1. Go to "Actions" → "Flows" → "Login"
2. Create a new custom action named "Add Roles"
3. Replace the code with:

```javascript
exports.onExecutePostLogin = async (event, api) => {
  const namespace = 'https://your-organization.com'; // Same as AUTH0_NAMESPACE
  if (event.authorization) {
    api.idToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
    api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
    api.accessToken.setCustomClaim(`${namespace}/email`, event.user.email);
    
    if(!event.user.email_verified){
      api.access.deny(`Please verify your email to access ${event.client.name}.`);
    }
  }
};
```

4. Deploy and add the action to your Login flow

#### 5. Add Authentication Providers

In the Auth0 dashboard under "Authentication":

- **Google**: Social → Create Connection → Google/Gmail → Follow the Client ID instructions
- **GitHub**: Social → Create Connection → GitHub → Enable "Email address" permission
- **Username/Password**: Enabled by default

#### 6. Start Hypha with Auth0

```bash
python -m hypha.server --host=0.0.0.0 --port=9527
```

Test the login at: `https://my-org.com/public/apps/hypha-login/`

## Custom Authentication Providers

Hypha supports custom authentication through startup functions, allowing you to replace or extend the default JWT authentication with your own implementation. This section will guide you through creating a complete custom authentication service.

### Overview

Custom authentication in Hypha is implemented using a startup function that registers your authentication handlers when the server starts. The startup function uses the `register_auth_service` API to customize:

1. **Token Parsing**: Custom token validation and user extraction
2. **Token Generation**: Custom token creation logic
3. **Token Extraction**: Custom token extraction from requests (headers, cookies, etc.)
4. **Login Service**: Custom login UI and authentication flow with multiple handlers

### Creating a Custom Authentication Service

#### Step 1: Create Your Authentication Module

Create a Python file (e.g., `my_auth.py`) with a `hypha_startup` function:

```python
# my_auth.py
import logging

logger = logging.getLogger(__name__)

async def hypha_startup(server):
    """Startup function called when Hypha server starts."""
    logger.info("Initializing custom authentication")
    
    # Register your authentication service
    await server.register_auth_service(
        # Your authentication handlers go here
    )
    
    logger.info("Custom authentication initialized")
```

#### Step 2: Start Hypha with Your Module

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 \
    --startup-functions=my_auth.py:hypha_startup
```

### Custom Token Extraction (get_token)

By default, Hypha looks for tokens in the `Authorization` header or `access_token` cookie. You can customize this behavior by providing a `get_token` function that extracts tokens from custom locations in the request:

```python
# custom_token_extraction.py

def custom_get_token(scope):
    """Extract token from custom headers or cookies.
    
    Args:
        scope: The ASGI scope object containing request information
        
    Returns:
        The extracted token string or None if no token found
    """
    headers = scope.get("headers", [])
    
    for key, value in headers:
        # Decode bytes if necessary
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        # Check for Cloudflare Access token
        if key.lower() == "cf-authorization":
            return value
        
        # Check for custom X-API-Key header
        if key.lower() == "x-api-key":
            return value
        
        # Check for custom cookies
        if key.lower() == "cookie":
            # Parse cookies
            cookie_dict = {}
            for cookie in value.split(";"):
                if "=" in cookie:
                    k, v = cookie.split("=", 1)
                    cookie_dict[k.strip()] = v.strip()
            
            # Check for custom token cookie
            if "app_token" in cookie_dict:
                return cookie_dict["app_token"]
            
            # Check for Cloudflare token cookie
            if "CF_Authorization" in cookie_dict:
                return cookie_dict["CF_Authorization"]
    
    # Return None to fall back to default extraction
    return None

async def hypha_startup(server):
    """Register custom token extraction."""
    await server.register_auth_service(
        get_token=custom_get_token,
        # Optional: also customize parsing if needed
        parse_token=custom_parse_token,
    )
```

#### Common Use Cases for Custom Token Extraction

1. **Cloudflare Access Integration**:
```python
def cloudflare_get_token(scope):
    """Extract Cloudflare Access JWT token."""
    headers = scope.get("headers", [])
    for key, value in headers:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        # Cloudflare Access uses CF-Authorization header
        if key.lower() == "cf-authorization":
            return value
        
        # Also check cookie
        if key.lower() == "cookie" and "CF_Authorization" in value:
            for cookie in value.split(";"):
                if cookie.strip().startswith("CF_Authorization="):
                    return cookie.split("=", 1)[1].strip()
    return None
```

2. **API Key in Custom Header**:
```python
def api_key_get_token(scope):
    """Extract API key from custom header."""
    headers = scope.get("headers", [])
    for key, value in headers:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        # Check multiple possible API key headers
        if key.lower() in ["x-api-key", "api-key", "x-auth-token"]:
            return value
    return None
```

3. **Session-based Authentication**:
```python
def session_get_token(scope):
    """Extract session ID from cookie."""
    headers = scope.get("headers", [])
    for key, value in headers:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        if key.lower() == "cookie":
            # Parse cookies
            for cookie in value.split(";"):
                if "=" in cookie:
                    k, v = cookie.split("=", 1)
                    if k.strip() == "session_id":
                        # Return session ID prefixed to identify it
                        return f"session:{v.strip()}"
    return None
```

4. **Query Parameter Token (for WebSocket connections)**:
```python
def query_param_get_token(scope):
    """Extract token from query parameters (useful for WebSocket)."""
    # First try headers
    headers = scope.get("headers", [])
    for key, value in headers:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if key.lower() == "authorization":
            return value
    
    # Then check query string for WebSocket connections
    query_string = scope.get("query_string", b"")
    if query_string:
        params = {}
        for param in query_string.decode("utf-8").split("&"):
            if "=" in param:
                k, v = param.split("=", 1)
                params[k] = v
        
        # Check for token in query params
        if "token" in params:
            return params["token"]
        if "access_token" in params:
            return params["access_token"]
    
    return None
```

### Complete Example: Local Authentication Implementation

Here's a complete example based on Hypha's built-in local authentication provider that demonstrates all the components needed for a full authentication system, including custom token extraction:

```python
# local_auth_example.py
import hashlib
import secrets
import time
import asyncio
from typing import Optional
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import _parse_token, create_scope, _generate_presigned_token
from hypha.utils import random_id
import logging

logger = logging.getLogger(__name__)

# In-memory storage for demo (use database in production)
USER_DATABASE = {}
LOGIN_SESSIONS = {}

def hash_password(password: str, salt: str) -> str:
    """Hash a password with salt."""
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_password(password: str, salt: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password, salt) == hashed

# 1. Token Parsing Function
async def custom_parse_token(token: str) -> UserInfo:
    """Parse and validate tokens."""
    if not token:
        from hypha.core.auth import generate_anonymous_user
        return generate_anonymous_user()
    
    # For this example, we use standard JWT tokens
    # You could implement your own token format here
    return _parse_token(token)

# 2. Token Generation Function  
async def custom_generate_token(user_info: UserInfo, expires_in: int) -> str:
    """Generate a token for a user."""
    # Use Hypha's built-in JWT generation
    # You could implement your own token generation here
    token = _generate_presigned_token(user_info, expires_in)
    logger.debug(f"Generated token for user: {user_info.id}")
    return token

# 3. Login Page Handler
async def login_page_handler(event):
    """Serve the login/signup page."""
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Custom Authentication</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <h1 class="text-2xl font-bold text-center mb-6">Custom Login</h1>
        
        <!-- Login Form -->
        <div class="space-y-4">
            <input type="email" id="email" placeholder="Email" 
                class="w-full px-3 py-2 border rounded-md">
            <input type="password" id="password" placeholder="Password" 
                class="w-full px-3 py-2 border rounded-md">
            <button onclick="login()" 
                class="w-full bg-blue-600 text-white py-2 rounded-md">
                Login
            </button>
        </div>
        
        <div id="message" class="mt-4 text-center"></div>
    </div>
    
    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const loginKey = urlParams.get('key');
        
        async function login() {
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            const response = await fetch('/public/services/hypha-login/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({email, password, key: loginKey})
            });
            
            const result = await response.json();
            if (result.success) {
                if (loginKey) {
                    // Report success and close popup
                    await fetch('/public/services/hypha-login/report', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            key: loginKey,
                            token: result.token,
                            email: result.email
                        })
                    });
                    window.close();
                }
                document.getElementById('message').textContent = 'Login successful!';
            } else {
                document.getElementById('message').textContent = result.error;
            }
        }
    </script>
</body>
</html>
    '''
    return {
        "status": 200,
        "headers": {"Content-Type": "text/html"},
        "body": html_content
    }

# 4. Login Flow Handlers
async def start_login_handler(workspace: str = None, expires_in: int = None):
    """Start a login session (called by hypha-rpc login)."""
    import shortuuid
    key = shortuuid.uuid()
    LOGIN_SESSIONS[key] = {
        "status": "pending",
        "workspace": workspace,
        "expires_in": expires_in or 3600
    }
    return {
        "login_url": f"/public/apps/hypha-login/?key={key}",
        "key": key,
        "report_url": "/public/services/hypha-login/report",
        "check_url": "/public/services/hypha-login/check"
    }

async def check_login_handler(key, timeout=180, profile=False):
    """Check if login is complete (polled by hypha-rpc)."""
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
                    "email": session.get("email"),
                    "workspace": session.get("workspace")
                }
            return token
        await asyncio.sleep(1)
    
    raise TimeoutError(f"Login timeout after {timeout} seconds")

async def report_login_handler(key, token=None, email=None, **kwargs):
    """Report login completion (called by login page)."""
    if key not in LOGIN_SESSIONS:
        raise ValueError("Invalid login key")

    LOGIN_SESSIONS[key]["status"] = "completed"
    LOGIN_SESSIONS[key]["token"] = token
    LOGIN_SESSIONS[key]["email"] = email
    return {"success": True}

async def custom_logout_handler(config=None):
    """Handle logout requests.

    This handler is called when a client requests to logout.
    It should return a dictionary with a 'logout_url' that the client
    will open to complete the server-side logout (e.g., Auth0 logout).

    Args:
        config: Optional configuration passed from the client

    Returns:
        Dictionary with 'logout_url' for completing the logout flow
    """
    # For custom auth, you might redirect to your own logout page
    # or to an identity provider's logout endpoint
    logout_page_url = "/public/apps/hypha-login/?logout=true&close=true"
    return {
        "logout_url": logout_page_url
    }

# 5. User Management Functions
async def signup_handler(server, context=None, name=None, email=None, password=None):
    """Handle user registration."""
    if not all([name, email, password]):
        return {"success": False, "error": "Missing required fields"}
    
    if email in USER_DATABASE:
        return {"success": False, "error": "Email already registered"}
    
    # Generate user ID and hash password
    user_id = random_id(readable=True)
    salt = secrets.token_hex(32)
    password_hash = hash_password(password, salt)
    
    # Store user
    USER_DATABASE[email] = {
        "id": user_id,
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "salt": salt,
        "roles": ["user"]
    }
    
    return {"success": True, "user_id": user_id}

async def login_handler(server, context=None, email=None, password=None, key=None, **kwargs):
    """Handle user login."""
    if not email or not password:
        return {"success": False, "error": "Email and password required"}
    
    user = USER_DATABASE.get(email)
    if not user:
        return {"success": False, "error": "Invalid credentials"}
    
    if not verify_password(password, user["salt"], user["password_hash"]):
        return {"success": False, "error": "Invalid credentials"}
    
    # Create UserInfo and generate token
    user_info = UserInfo(
        id=user["id"],
        email=user["email"],
        roles=user["roles"],
        is_anonymous=False,
        scope=create_scope(
            workspaces={f"ws-user-{user['id']}": UserPermission.admin},
            current_workspace=f"ws-user-{user['id']}"
        )
    )
    
    token = await custom_generate_token(user_info, 3600 * 24)
    
    # Update login session if key provided
    if key and key in LOGIN_SESSIONS:
        LOGIN_SESSIONS[key]["status"] = "completed"
        LOGIN_SESSIONS[key]["token"] = token
        LOGIN_SESSIONS[key]["email"] = user["email"]
    
    return {
        "success": True,
        "token": token,
        "email": user["email"],
        "user_id": user["id"]
    }

# 6. Custom Token Extraction (optional)
def custom_get_token(scope):
    """Extract token from custom locations."""
    headers = scope.get("headers", [])
    
    for key, value in headers:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        # Check for custom header
        if key.lower() == "x-custom-token":
            return value
        
        # Check for custom cookie
        if key.lower() == "cookie":
            for cookie in value.split(";"):
                if "=" in cookie:
                    k, v = cookie.split("=", 1)
                    if k.strip() == "auth_token":
                        return v.strip()
    
    # Fall back to default extraction
    return None

# 7. Main Startup Function
async def hypha_startup(server):
    """Register the complete authentication service."""
    logger.info("Initializing custom authentication")
    
    await server.register_auth_service(
        # Core authentication functions
        parse_token=custom_parse_token,
        generate_token=custom_generate_token,
        get_token=custom_get_token,  # Optional custom extraction

        # Login service handlers
        index_handler=login_page_handler,
        start_handler=start_login_handler,
        check_handler=check_login_handler,
        report_handler=report_login_handler,
        logout_handler=custom_logout_handler,  # Custom logout handler

        # Additional service methods
        signup=lambda context=None, **kwargs: signup_handler(server, context, **kwargs),
        login=lambda context=None, **kwargs: login_handler(server, context, **kwargs)
    )

    logger.info("Custom authentication service registered")
```

### Key Components Explained

#### 1. The Startup Function
The `hypha_startup` function is called when Hypha starts. It receives the server object which provides access to the `register_auth_service` function.

#### 2. Token Management
- **parse_token**: Validates tokens and returns a `UserInfo` object
- **generate_token**: Creates tokens from `UserInfo` and expiration time
- **get_token** (optional): Extracts tokens from custom locations in the request scope

#### 3. Login Service Handlers
The login service implements the OAuth-like flow used by hypha-rpc's `login()` function:
- **index_handler**: Serves the login page HTML
- **start_handler**: Initiates a login session, returns a key and URLs
- **check_handler**: Polls for login completion (used by hypha-rpc)
- **report_handler**: Reports successful login (called by login page)
- **logout_handler**: Returns logout URL for ending sessions (called by hypha-rpc's `logout()`)

#### 4. Additional Methods
Any extra keyword arguments to `register_auth_service` are added as methods to the login service. These can be called via `/public/services/hypha-login/<method_name>`.

### Using Your Custom Authentication

Once implemented, clients can use your authentication seamlessly:

```python
from hypha_rpc import login, connect_to_server

# The login() function will use your custom authentication
token = await login({"server_url": "http://localhost:9527"})

# Connect with the token
async with connect_to_server({
    "server_url": "http://localhost:9527",
    "token": token
}) as server:
    # Use authenticated services
    pass
```

### Production Considerations

When implementing custom authentication for production use:

#### 1. Persistent Storage
Replace in-memory storage with a database:

```python
# Use Redis for session storage
import redis.asyncio as redis

class AuthStorage:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    async def store_session(self, key: str, data: dict, ttl: int = 3600):
        await self.redis.setex(f"session:{key}", ttl, json.dumps(data))
    
    async def get_session(self, key: str):
        data = await self.redis.get(f"session:{key}")
        return json.loads(data) if data else None

# Use PostgreSQL or MongoDB for user data
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String)
    password_hash = Column(String)
    salt = Column(String)
    roles = Column(String)  # JSON string

# Initialize database
engine = create_engine('postgresql://user:password@localhost/hypha_auth')
SessionLocal = sessionmaker(bind=engine)
```

#### 2. Security Best Practices

- **Password Requirements**: Enforce strong password policies
- **Rate Limiting**: Prevent brute force attacks
- **Token Rotation**: Implement refresh tokens
- **Secure Communication**: Always use HTTPS in production
- **Input Validation**: Sanitize all user inputs
- **Audit Logging**: Log authentication events

```python
import time
from functools import lru_cache

# Rate limiting example
LOGIN_ATTEMPTS = {}

def check_rate_limit(email: str, max_attempts: int = 5, window: int = 300):
    """Check if user has exceeded login attempts."""
    now = time.time()
    if email not in LOGIN_ATTEMPTS:
        LOGIN_ATTEMPTS[email] = []
    
    # Remove old attempts outside the window
    LOGIN_ATTEMPTS[email] = [t for t in LOGIN_ATTEMPTS[email] if now - t < window]
    
    if len(LOGIN_ATTEMPTS[email]) >= max_attempts:
        raise Exception(f"Too many login attempts. Try again in {window} seconds.")
    
    LOGIN_ATTEMPTS[email].append(now)
```

#### 3. Using Hypha's Artifact Manager for Storage

For better integration with Hypha, you can use the Artifact Manager to store user data:

```python
async def store_user_with_artifacts(server, user_data):
    """Store user data using Hypha's Artifact Manager."""
    artifact_manager = await server.get_service("public/artifact-manager")
    
    # Create or get users collection
    try:
        collection = await artifact_manager.read("ws-user-root/auth-users")
    except:
        collection = await artifact_manager.create(
            workspace="ws-user-root",
            alias="auth-users",
            type="collection",
            manifest={"name": "Authentication Users"}
        )
    
    # Store user as artifact
    await artifact_manager.create(
        parent_id=collection["id"],
        alias=user_data["id"],
        type="user",
        manifest=user_data
    )
```

### Advanced Authentication Patterns

#### Multi-Factor Authentication (MFA)

Add an additional verification step:

```python
import pyotp

async def setup_mfa(user_id: str) -> str:
    """Generate MFA secret for user."""
    secret = pyotp.random_base32()
    # Store secret with user data
    return pyotp.totp.TOTP(secret).provisioning_uri(
        name=user_id,
        issuer_name="Hypha"
    )

async def verify_mfa(user_id: str, code: str) -> bool:
    """Verify MFA code."""
    secret = await get_user_mfa_secret(user_id)
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)
```

#### Session Management

Implement proper session handling:

```python
async def create_session(user_info: UserInfo, device_info: dict = None):
    """Create a user session with device tracking."""
    session_id = secrets.token_urlsafe(32)
    session_data = {
        "user_id": user_info.id,
        "created_at": time.time(),
        "last_activity": time.time(),
        "device_info": device_info,
        "ip_address": request.client.host
    }
    await store_session(session_id, session_data)
    return session_id

async def invalidate_all_sessions(user_id: str):
    """Logout user from all devices."""
    # Implementation depends on your storage backend
    pass
```

### Combining Custom Token Extraction with Authentication Providers

You can combine custom token extraction with various authentication methods to support multiple token sources:

```python
# multi_source_auth.py

def flexible_get_token(scope):
    """Extract tokens from multiple possible sources."""
    headers = scope.get("headers", [])
    
    # Priority order for token extraction
    for key, value in headers:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        # 1. Check Cloudflare Access header (highest priority)
        if key.lower() == "cf-authorization":
            return value
        
        # 2. Check API key headers
        if key.lower() in ["x-api-key", "api-key"]:
            return f"api:{value}"  # Prefix to identify API keys
        
        # 3. Check Authorization header
        if key.lower() == "authorization":
            return value
        
        # 4. Check cookies as fallback
        if key.lower() == "cookie":
            cookies = {}
            for cookie in value.split(";"):
                if "=" in cookie:
                    k, v = cookie.split("=", 1)
                    cookies[k.strip()] = v.strip()
            
            # Check various cookie names
            for cookie_name in ["CF_Authorization", "auth_token", "session_id", "access_token"]:
                if cookie_name in cookies:
                    if cookie_name == "session_id":
                        return f"session:{cookies[cookie_name]}"
                    return cookies[cookie_name]
    
    return None

async def flexible_parse_token(token: str) -> UserInfo:
    """Parse tokens from different sources."""
    if not token:
        from hypha.core.auth import generate_anonymous_user
        return generate_anonymous_user()
    
    # Handle API keys
    if token.startswith("api:"):
        api_key = token[4:]
        # Validate API key and return user info
        return validate_api_key(api_key)
    
    # Handle session tokens
    if token.startswith("session:"):
        session_id = token[8:]
        # Validate session and return user info
        return validate_session(session_id)
    
    # Handle JWT tokens (default)
    from hypha.core.auth import _parse_token
    return _parse_token(token)

async def hypha_startup(server):
    """Register flexible authentication."""
    await server.register_auth_service(
        get_token=flexible_get_token,
        parse_token=flexible_parse_token,
        # ... other handlers
    )
```

### Integration with External Providers

You can integrate multiple authentication providers in your custom auth:

```python
async def hybrid_auth_startup(server):
    """Support multiple authentication methods."""
    
    async def hybrid_parse_token(token: str) -> UserInfo:
        """Parse tokens from multiple sources."""
        # Check token type
        if token.startswith("local_"):
            return await parse_local_token(token)
        elif token.startswith("oauth_"):
            return await parse_oauth_token(token)
        elif token.startswith("ldap_"):
            return await parse_ldap_token(token)
        else:
            # Fall back to JWT
            return _parse_token(token)
    
    await server.register_auth_service(
        parse_token=hybrid_parse_token,
        # ... other handlers
    )
```

### Basic Custom Authentication

Here's a simple example implementing token-based authentication:

```python
# custom_auth.py
import time
import secrets
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope

# In-memory token storage (use a database in production)
TOKEN_STORE = {}

def custom_generate_token(user_info: UserInfo, expires_in: int) -> str:
    """Generate a custom token."""
    token = secrets.token_urlsafe(32)
    TOKEN_STORE[token] = {
        "user_info": user_info,
        "expires_at": time.time() + expires_in
    }
    return token

async def custom_parse_token(token: str) -> UserInfo:
    """Parse and validate a custom token."""
    if token not in TOKEN_STORE:
        raise ValueError("Invalid token")
    
    token_data = TOKEN_STORE[token]
    if time.time() > token_data["expires_at"]:
        del TOKEN_STORE[token]
        raise ValueError("Token expired")
    
    return token_data["user_info"]

async def hypha_startup(server):
    """Register custom authentication on startup."""
    # Use the unified register_auth_service function
    await server.register_auth_service(
        parse_token=custom_parse_token,
        generate_token=custom_generate_token
    )
    print("Custom authentication configured")
```

Start Hypha with your custom authentication:

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 \
    --startup-functions=custom_auth.py:hypha_startup
```

### Custom Login Service

You can customize the login interface and flow by providing custom handlers:

```python
# custom_login.py
import asyncio
import shortuuid

# Store login sessions
LOGIN_SESSIONS = {}

async def custom_login_page(event):
    """Serve custom login page."""
    return {
        "status": 200,
        "headers": {"Content-Type": "text/html"},
        "body": "<html><body><h1>Custom Login</h1>...</body></html>"
    }

async def start_login(workspace=None, expires_in=None):
    """Start a login session."""
    key = shortuuid.uuid()
    LOGIN_SESSIONS[key] = {"status": "pending", "workspace": workspace}
    return {
        "login_url": f"/public/apps/hypha-login/?key={key}",
        "key": key,
    }

async def check_login(key, timeout=180, profile=False):
    """Check login status."""
    # Wait for login completion
    for _ in range(timeout):
        if key in LOGIN_SESSIONS and LOGIN_SESSIONS[key]["status"] == "complete":
            return LOGIN_SESSIONS[key]["token"]
        await asyncio.sleep(1)
    raise TimeoutError("Login timeout")

async def report_login(key, token, **kwargs):
    """Report login completion."""
    if key in LOGIN_SESSIONS:
        LOGIN_SESSIONS[key]["status"] = "complete"
        LOGIN_SESSIONS[key]["token"] = token

async def hypha_startup(server):
    """Register custom login service."""
    await server.register_auth_service(
        index_handler=custom_login_page,
        start_handler=start_login,
        check_handler=check_login,
        report_handler=report_login,
    )
```

### API Key Authentication

Implement API key authentication for programmatic access:

```python
# api_key_auth.py
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope

# API key database (use secure storage in production)
API_KEYS = {
    "sk_live_abc123": {
        "user_id": "api_user_1",
        "email": "api@example.com",
        "permissions": ["read", "write"],
        "workspace": "api-workspace"
    }
}

async def parse_api_key_token(token: str) -> UserInfo:
    """Parse API key format tokens."""
    if token.startswith("sk_"):
        # API key authentication
        if token not in API_KEYS:
            raise ValueError("Invalid API key")
        
        key_data = API_KEYS[token]
        return UserInfo(
            id=key_data["user_id"],
            email=key_data["email"],
            roles=key_data["permissions"],
            scope=create_scope(
                workspaces={key_data["workspace"]: UserPermission.admin}
            )
        )
    else:
        # Fall back to default JWT parsing
        from hypha.core.auth import _parse_token
        return _parse_token(token)

async def hypha_startup(server):
    """Configure API key authentication."""
    await server.register_auth_service(
        parse_token=parse_api_key_token
        # Keep default token generation and login service
    )
```

### SAML Authentication

Integrate with enterprise SAML identity providers:

```python
# saml_auth.py
import uuid
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope

class SAMLAuth:
    def __init__(self, idp_metadata_url: str):
        self.idp_metadata_url = idp_metadata_url
        self.sessions = {}
    
    async def parse_saml_response(self, saml_response: str) -> dict:
        """Parse and validate SAML response."""
        # In production, use python-saml2 or similar library
        # This is a simplified example
        return {
            "name_id": "user@example.com",
            "attributes": {
                "email": ["user@example.com"],
                "groups": ["developers", "admins"]
            }
        }
    
    def create_session_token(self, saml_data: dict) -> str:
        """Create a session token from SAML data."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": saml_data["name_id"].split("@")[0],
            "email": saml_data["attributes"]["email"][0],
            "groups": saml_data["attributes"].get("groups", []),
            "expires_at": time.time() + 3600  # 1 hour
        }
        return f"saml_{session_id}"
    
    async def parse_token(self, token: str) -> UserInfo:
        """Parse SAML session tokens."""
        if token.startswith("saml_"):
            session_id = token[5:]
            if session_id not in self.sessions:
                raise ValueError("Invalid SAML session")
            
            session = self.sessions[session_id]
            if time.time() > session["expires_at"]:
                del self.sessions[session_id]
                raise ValueError("Session expired")
            
            return UserInfo(
                id=session["user_id"],
                email=session["email"],
                roles=session["groups"],
                scope=create_scope(
                    workspaces={"saml-workspace": UserPermission.admin}
                )
            )
        else:
            from hypha.core.auth import _parse_token
            return _parse_token(token)

# Create ASGI login service
async def create_saml_login_service(server, saml_auth):
    """Create ASGI service for SAML login flow."""
    
    async def saml_asgi_app(scope, receive, send):
        """ASGI app handling SAML endpoints."""
        path = scope["path"]
        
        if path == "/saml/login":
            # Redirect to IdP
            redirect_url = f"{saml_auth.idp_metadata_url}/sso?..."
            await send({
                "type": "http.response.start",
                "status": 302,
                "headers": [[b"location", redirect_url.encode()]],
            })
            await send({"type": "http.response.body", "body": b""})
            
        elif path == "/saml/acs":
            # Assertion Consumer Service
            # Parse SAML response from request body
            body = b""
            while True:
                message = await receive()
                if message["type"] == "http.request":
                    body += message.get("body", b"")
                    if not message.get("more_body"):
                        break
            
            # Process SAML response
            saml_response = body.decode()  # Parse actual SAML XML
            saml_data = await saml_auth.parse_saml_response(saml_response)
            token = saml_auth.create_session_token(saml_data)
            
            # Return token to user
            response = f"Login successful. Token: {token}".encode()
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            })
            await send({"type": "http.response.body", "body": response})
    
    # Register as login service
    await server["set_login_service"](saml_asgi_app)
    
    # Also register as a regular ASGI service for additional endpoints
    await server.register_service({
        "id": "saml-auth",
        "type": "asgi",
        "serve": lambda args: saml_asgi_app(
            args["scope"], args["receive"], args["send"]
        )
    })

async def hypha_startup(server):
    """Configure SAML authentication."""
    saml_auth = SAMLAuth("https://idp.example.com/metadata")
    await server.register_auth_service(
        parse_token=saml_auth.parse_token
    )
    # Optionally register SAML endpoints as additional services
    await create_saml_login_service(server, saml_auth)
    print("SAML authentication configured")
```

### OAuth2/OIDC Authentication

Integrate with OAuth2/OpenID Connect providers:

```python
# oauth2_auth.py
import httpx
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope

class OAuth2Auth:
    def __init__(self, client_id: str, client_secret: str, 
                 auth_url: str, token_url: str, userinfo_url: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self.token_url = token_url
        self.userinfo_url = userinfo_url
        self.access_tokens = {}
    
    async def exchange_code_for_token(self, code: str) -> str:
        """Exchange authorization code for access token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_url, data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": "http://localhost:9527/oauth/callback"
            })
            data = response.json()
            access_token = data["access_token"]
            
            # Get user info
            user_response = await client.get(
                self.userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            user_data = user_response.json()
            
            # Store access token with user info
            token_id = str(uuid.uuid4())
            self.access_tokens[token_id] = {
                "access_token": access_token,
                "user_data": user_data,
                "expires_at": time.time() + data.get("expires_in", 3600)
            }
            
            return f"oauth2_{token_id}"
    
    async def parse_token(self, token: str) -> UserInfo:
        """Parse OAuth2 tokens."""
        if token.startswith("oauth2_"):
            token_id = token[7:]
            if token_id not in self.access_tokens:
                raise ValueError("Invalid OAuth2 token")
            
            token_data = self.access_tokens[token_id]
            if time.time() > token_data["expires_at"]:
                del self.access_tokens[token_id]
                raise ValueError("Token expired")
            
            user_data = token_data["user_data"]
            return UserInfo(
                id=user_data.get("sub", user_data.get("id")),
                email=user_data.get("email"),
                roles=user_data.get("roles", []),
                scope=create_scope(
                    workspaces={"oauth-workspace": UserPermission.admin}
                )
            )
        else:
            from hypha.core.auth import _parse_token
            return _parse_token(token)

async def hypha_startup(server):
    """Configure OAuth2 authentication."""
    oauth_auth = OAuth2Auth(
        client_id="your-client-id",
        client_secret="your-client-secret",
        auth_url="https://provider.com/authorize",
        token_url="https://provider.com/token",
        userinfo_url="https://provider.com/userinfo"
    )
    await server.register_auth_service(
        parse_token=oauth_auth.parse_token
    )
    print("OAuth2 authentication configured")
```

### Multi-Provider Authentication

Support multiple authentication methods simultaneously:

```python
# multi_auth.py
from hypha.core import UserInfo
from hypha.core.auth import _parse_token

class MultiProviderAuth:
    def __init__(self):
        self.providers = {}
    
    def register_provider(self, prefix: str, parser):
        """Register an authentication provider."""
        self.providers[prefix] = parser
    
    async def parse_token(self, token: str) -> UserInfo:
        """Route token to appropriate provider."""
        # Check each provider by token prefix
        for prefix, parser in self.providers.items():
            if token.startswith(prefix):
                return await parser(token)
        
        # Fall back to default JWT
        return _parse_token(token)

async def hypha_startup(server):
    """Configure multi-provider authentication."""
    multi_auth = MultiProviderAuth()
    
    # Register API key provider
    multi_auth.register_provider("sk_", parse_api_key)
    
    # Register SAML provider
    multi_auth.register_provider("saml_", parse_saml_token)
    
    # Register OAuth2 provider
    multi_auth.register_provider("oauth2_", parse_oauth_token)
    
    # Register custom provider
    multi_auth.register_provider("custom_", parse_custom_token)
    
    await server.register_auth_service(
        parse_token=multi_auth.parse_token
    )
    print("Multi-provider authentication configured")
```

## Service Authorization

Once users are authenticated, you can implement fine-grained authorization within your services:

```python
async def start_service(server):
    """Service with authorization checks."""
    
    # Define authorized users/roles
    AUTHORIZED_USERS = ["admin@example.com", "user@example.com"]
    AUTHORIZED_ROLES = ["admin", "developer"]
    
    def protected_function(data, context=None):
        """Function requiring authorization."""
        if not context:
            raise Exception("Authentication required")
        
        user = context["user"]
        
        # Check user email
        if user.get("email") not in AUTHORIZED_USERS:
            # Check user roles
            user_roles = user.get("roles", [])
            if not any(role in AUTHORIZED_ROLES for role in user_roles):
                raise Exception(f"User {user.get('email', 'unknown')} is not authorized")
        
        # Process authorized request
        return f"Protected data for {user['email']}: {data}"
    
    await server.register_service({
        "id": "protected-service",
        "config": {
            "visibility": "public",
            "require_context": True  # Enable user context
        },
        "protected_function": protected_function
    })
```

## HTTP API Authentication

For non-WebSocket clients, authenticate via HTTP headers. With custom `get_token` functions, you can support various authentication methods:

### Using Standard Authorization Header

```bash
# Get token via login service
curl -X POST "https://ai.imjoy.io/public/services/hypha-login/start" \
  -H "Content-Type: application/json" -d '{}'

# Use token in Authorization header
curl -X POST "https://ai.imjoy.io/public/services/my-service/method" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"param": "value"}'
```

### Using Custom Headers (with get_token)

```bash
# Use custom X-API-Key header
curl -X POST "https://ai.imjoy.io/public/services/my-service/method" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"param": "value"}'

# Use Cloudflare Access token
curl -X POST "https://ai.imjoy.io/public/services/my-service/method" \
  -H "Content-Type: application/json" \
  -H "CF-Authorization: YOUR_CF_TOKEN" \
  -d '{"param": "value"}'
```

### Using Cookies (with get_token)

```bash
# Use cookie authentication
curl -X POST "https://ai.imjoy.io/public/services/my-service/method" \
  -H "Content-Type: application/json" \
  -b "auth_token=YOUR_TOKEN" \
  -d '{"param": "value"}'

# Use Cloudflare Access cookie
curl -X POST "https://ai.imjoy.io/public/services/my-service/method" \
  -H "Content-Type: application/json" \
  -b "CF_Authorization=YOUR_CF_TOKEN" \
  -d '{"param": "value"}'
```

## Best Practices

### Security Considerations

1. **Token Storage**: Never store tokens in plain text. Use secure storage mechanisms.
2. **Token Expiration**: Always implement token expiration and refresh mechanisms.
3. **HTTPS Only**: Always use HTTPS in production to protect tokens in transit.
4. **Rate Limiting**: Implement rate limiting to prevent brute force attacks.
5. **Audit Logging**: Log authentication events for security monitoring.

### Implementation Tips

1. **Fallback to Default**: Always provide fallback to default JWT parsing for compatibility
2. **Async Functions**: Use async functions for authentication to support I/O operations
3. **Error Handling**: Provide clear error messages for authentication failures
4. **Testing**: Test authentication with various token types and edge cases
5. **Documentation**: Document your authentication requirements for API users

### Production Deployment

1. **Environment Variables**: Use environment variables for sensitive configuration
2. **Database Storage**: Use persistent storage (Redis, PostgreSQL) for tokens in production
3. **Load Balancing**: Ensure token validation works across multiple server instances
4. **Monitoring**: Monitor authentication failures and suspicious patterns
5. **Backup Auth**: Consider implementing backup authentication methods

## Troubleshooting

### Common Issues

**Token Validation Fails**
- Check token expiration
- Verify JWT_SECRET is consistent across services
- Ensure custom parser returns proper UserInfo object

**CORS Issues**
- Configure allowed origins in Auth0 or server settings
- Check browser console for specific CORS errors

**Login Redirect Issues**
- Verify callback URLs in Auth0 configuration
- Check that redirect URIs match exactly

**Custom Auth Not Working**
- Ensure startup function is properly registered
- Check server logs for startup errors
- Verify function signatures match expected format

## Examples

Complete examples are available in the repository:

- [Basic Custom Auth](https://github.com/amun-ai/hypha/blob/main/examples/custom-auth-basic.py)
- [API Key Authentication](https://github.com/amun-ai/hypha/blob/main/examples/custom-auth-api-key.py)
- [SAML Integration](https://github.com/amun-ai/hypha/blob/main/examples/custom-auth-saml.py)
- [Multi-Provider Setup](https://github.com/amun-ai/hypha/blob/main/examples/custom-auth-multi.py)

## Next Steps

- Learn about [Service Configuration](configurations.md#advanced-authentication-and-authorization) for authorization
- Explore [Workspace Management](configurations.md#workspace-management) for team collaboration
- Set up [Production Deployment](configurations.md#production-deployment) with proper security