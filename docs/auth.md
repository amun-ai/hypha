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

Hypha supports custom authentication through startup functions, allowing you to replace or extend the default JWT authentication with your own implementation.

### Overview

Custom authentication in Hypha is configured through the unified `register_auth_service` function, which allows you to customize:

1. **Token Parsing**: Custom token validation and user extraction
2. **Token Generation**: Custom token creation logic
3. **Login Service**: Custom login UI and authentication flow

The `register_auth_service` function provides a simplified interface that handles all authentication aspects in one place.

### register_auth_service Function

```python
await server.register_auth_service(
    parse_token=None,           # Custom token parsing function
    generate_token=None,        # Custom token generation function  
    login_service=None,         # Complete login service dict (advanced)
    index_handler=None,         # Custom login page handler
    start_handler=None,         # Custom login start handler
    check_handler=None,         # Custom login check handler
    report_handler=None,        # Custom login report handler
    **extra_handlers            # Additional service methods
)
```

**Parameters:**
- **parse_token**: Function to parse and validate tokens, returning `UserInfo`
- **generate_token**: Function to generate tokens from `UserInfo` and expiration
- **login_service**: Complete service dictionary (overrides individual handlers)
- **index_handler**: Serves the login page (`async def index(event)`)
- **start_handler**: Starts login session (`async def start(workspace=None, expires_in=None)`)
- **check_handler**: Checks login status (`async def check(key, timeout=180, profile=False)`)
- **report_handler**: Reports login completion (`async def report(key, token, **kwargs)`)
- **extra_handlers**: Additional methods to add to the login service

The login service is automatically registered with ID `"hypha-login"` to replace the default.

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

For non-WebSocket clients, authenticate via HTTP headers:

### Using Tokens

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

### Using API Keys

```bash
# Use API key directly
curl -X POST "https://ai.imjoy.io/public/services/my-service/method" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk_live_your_api_key" \
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