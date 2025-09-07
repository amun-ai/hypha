"""Provide authentication."""

import asyncio
import json
import logging
import ssl
import inspect
import os
import sys
from calendar import timegm
import datetime
from os import environ as env
from typing import List, Union, Dict, Callable
from urllib.request import urlopen

import shortuuid
from dotenv import find_dotenv, load_dotenv
from fastapi import HTTPException
from jinja2 import Environment, PackageLoader, select_autoescape
from jose import jwt

from hypha.core import UserInfo, UserTokenInfo, ScopeInfo, UserPermission, WorkspaceInfo
from hypha.utils import random_id
from hypha import __version__

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("auth")
logger.setLevel(LOGLEVEL)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

MAXIMUM_LOGIN_TIME = env.get("MAXIMUM_LOGIN_TIME", "180")  # 3 minutes
AUTH0_CLIENT_ID = env.get("AUTH0_CLIENT_ID", "paEagfNXPBVw8Ss80U5RAmAV4pjCPsD2")
AUTH0_DOMAIN = env.get("AUTH0_DOMAIN", "amun-ai.eu.auth0.com")
AUTH0_AUDIENCE = env.get("AUTH0_AUDIENCE", "https://amun-ai.eu.auth0.com/api/v2/")
AUTH0_ISSUER = env.get("AUTH0_ISSUER", "https://amun.ai/")
AUTH0_NAMESPACE = env.get("AUTH0_NAMESPACE", "https://amun.ai/")
def _get_jwt_secret():
    """Get JWT secret, ensuring consistency during testing and runtime."""
    # Always check environment variables first (important for testing)
    secret = env.get("HYPHA_JWT_SECRET") or env.get("JWT_SECRET")
    if not secret:
        logger.info(
            "Neither HYPHA_JWT_SECRET nor JWT_SECRET is defined, using a random JWT_SECRET"
        )
        secret = shortuuid.ShortUUID().random(length=22)
        # Set the environment variable to ensure consistency across module reloads
        env["JWT_SECRET"] = secret
        env["HYPHA_JWT_SECRET"] = secret
    return secret

JWT_SECRET = _get_jwt_secret()

def set_jwt_secret(secret: str):
    """Set JWT secret explicitly (mainly for testing)."""
    global JWT_SECRET
    env["JWT_SECRET"] = secret
    env["HYPHA_JWT_SECRET"] = secret
    JWT_SECRET = secret
LOGIN_SERVICE_URL = "/public/services/hypha-login"
LOGIN_KEY_PREFIX = "login_key:"


def get_user_email(token):
    """Return the user email from the token."""
    return token.credentials.get(AUTH0_NAMESPACE + "email")


def get_user_id(token):
    """Return the user id from the token."""
    return token.credentials.get("sub")


def get_user_info(credentials):
    """Return the user info from the token."""
    expires_at = credentials["exp"]
    scope = parse_scope(credentials.get("scope"))
    roles = credentials.get(AUTH0_NAMESPACE + "roles", [])
    info = UserInfo(
        id=credentials.get("sub"),
        is_anonymous="anonymous" in roles,
        email=credentials.get(AUTH0_NAMESPACE + "email"),
        roles=roles,
        scope=scope,
        expires_at=expires_at,
        current_workspace=scope.current_workspace,
    )
    # make sure the user has admin permission to their own workspace
    user_workspace = info.get_workspace()
    info.scope.workspaces[user_workspace] = UserPermission.admin
    return info


JWKS = None


def get_rsa_key(kid, refresh=False):
    """Return an rsa key."""
    global JWKS  # pylint: disable=global-statement
    if JWKS is None or refresh:
        with urlopen(
            f"https://{AUTH0_DOMAIN}/.well-known/jwks.json",
            context=ssl._create_default_https_context(),
        ) as jsonurl:
            JWKS = json.loads(jsonurl.read())
    rsa_key = {}
    for key in JWKS["keys"]:
        if key["kid"] == kid:
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"],
            }
            break
    return rsa_key


def valid_token(authorization: str):
    """Validate token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is expected")

    try:
        unverified_header = jwt.get_unverified_header(authorization)
        alg = unverified_header.get("alg")

        if alg == "HS256":
            payload = jwt.decode(
                authorization,
                JWT_SECRET,
                algorithms=["HS256"],
                audience=AUTH0_AUDIENCE,
                issuer=AUTH0_ISSUER,
            )
        elif alg == "RS256":
            # Get RSA key
            rsa_key = get_rsa_key(unverified_header["kid"], refresh=False)
            # Try to refresh jwks if failed
            if not rsa_key:
                rsa_key = get_rsa_key(unverified_header["kid"], refresh=True)

            # Decode token
            payload = jwt.decode(
                authorization,
                rsa_key,
                algorithms=["RS256"],
                audience=AUTH0_AUDIENCE,
                issuer=f"https://{AUTH0_DOMAIN}/",
            )
        else:
            raise HTTPException(status_code=401, detail="Invalid algorithm: " + alg)
        return payload

    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=401, detail="The token has expired. Please fetch a new one"
        ) from err
    except jwt.JWTError as err:
        raise HTTPException(status_code=401, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=401, detail=str(err)) from err


def generate_anonymous_user(scope=None) -> UserInfo:
    """Generate user info for an anonymous user."""
    current_time = datetime.datetime.now(datetime.timezone.utc)
    user_id = "anonymouz-" + random_id(
        readable=True
    )  # Use hyphen instead of underscore
    expires_at = current_time + datetime.timedelta(seconds=600)
    expires_at = timegm(expires_at.utctimetuple())
    return UserInfo(
        id=user_id,
        is_anonymous=True,
        email=None,
        roles=["anonymous"],
        scope=scope,
        expires_at=expires_at,
    )


def _parse_token(authorization: str, expected_workspace: str = None):
    """Parse the token with optional workspace validation.
    
    Args:
        authorization: The authorization token string
        expected_workspace: If provided, will validate that the token's current_workspace matches
                          this value before performing full token validation. This prevents
                          validating tokens for unauthorized workspaces.
    
    Returns:
        UserInfo object if token is valid and workspace matches (if specified)
    
    Raises:
        HTTPException: If token is invalid or workspace doesn't match
    """
    assert authorization, "Authorization is required"
    if authorization.startswith("Bearer ") or authorization.startswith("bearer "):
        parts = authorization.split()
        if parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=401, detail="Authorization header must start with" " Bearer"
            )
        if len(parts) == 1:
            raise HTTPException(status_code=401, detail="Token not found")
        if len(parts) > 2:
            raise HTTPException(
                status_code=401, detail="Authorization header must be 'Bearer' token"
            )

        token = parts[1]
    else:
        token = authorization
    
    if not token:
        raise ValueError("Token is empty")
    
    # Check if this is the root token FIRST, before any JWT decoding
    global _current_root_token

    # Try stripping whitespace in case the browser adds some
    if _current_root_token and (token == _current_root_token or (isinstance(token, str) and token.strip() == _current_root_token)):
        logger.info("Root token authenticated successfully")
        # Return root user info
        return UserInfo(
            id="root",
            is_anonymous=False,
            email=None,
            parent=None,
            roles=["admin"],
            scope=create_scope("*#a", current_workspace=expected_workspace or "ws-user-root"),
            expires_at=None,
        )

    # If expected_workspace is provided, extract and verify workspace before full validation
    if expected_workspace:
        try:
            # Decode without verification to get the payload
            unverified_payload = jwt.get_unverified_claims(token)
            
            # Extract the current workspace from the scope
            scope_str = unverified_payload.get("scope", "")
            scope_info = parse_scope(scope_str)
            
            # Check if current workspace matches expected
            # Only validate if the token has a current_workspace set
            if scope_info.current_workspace and scope_info.current_workspace != expected_workspace:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Token is not authorized for workspace '{expected_workspace}'"
                )
        except jwt.JWTError as err:
            # If we can't even decode the token structure, it's invalid
            raise HTTPException(status_code=401, detail=f"Invalid token structure: {str(err)}") from err

    # Now perform full validation
    payload = valid_token(token)
    return get_user_info(payload)


_current_auth_function = None
_current_root_token = None
_current_get_token_function = None


async def set_parse_token_function(auth_function: Callable):
    """Set the auth provider."""
    global _current_auth_function
    _current_auth_function = auth_function
    logger.info("Custom parse_token function has been set")


def set_root_token(root_token: str):
    """Set the root token for authentication."""
    global _current_root_token
    _current_root_token = root_token
    if root_token:
        logger.info("Root token has been configured")


async def set_get_token_function(get_token_function: Callable):
    """Set the custom token extraction function.
    
    The function should accept a scope object (from websocket.scope or request.scope)
    and return the token string to be used for authentication.
    """
    global _current_get_token_function
    _current_get_token_function = get_token_function
    logger.info("Custom get_token function has been set")

async def extract_token_from_scope(scope: dict) -> str:
    """Extract token from a scope object using custom or default logic.
    
    Args:
        scope: The scope object from websocket or request
        
    Returns:
        The extracted token string or None if no token found
    """
    if _current_get_token_function:
        token = _current_get_token_function(scope)
        if inspect.isawaitable(token):
            token = await token
        return token
    
    # Default extraction logic - look in headers for Authorization
    headers = scope.get("headers", [])
    for key, value in headers:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if key.lower() == "authorization":
            return value
        # Also check for access_token cookie
        if key.lower() == "cookie":
            cookies = value
            # Parse cookies
            cookie_dict = {}
            for cookie in cookies.split(";"):
                if "=" in cookie:
                    k, v = cookie.split("=", 1)
                    cookie_dict[k.strip()] = v.strip()
            if "access_token" in cookie_dict:
                return cookie_dict["access_token"]
    
    return None

async def parse_auth_token(token: str, expected_workspace: str = None):
    """Parse auth token with optional workspace validation.
    
    Args:
        token: The authorization token string
        expected_workspace: If provided, will validate that the token's current_workspace matches
                          this value (only applies to default parser)
    
    Returns:
        UserInfo object if token is valid and workspace matches (if specified)
    """
    if _current_auth_function is None:
        auth_function = lambda t: _parse_token(t, expected_workspace)
    else:
        # Custom auth functions may not support workspace validation
        # They should implement their own logic if needed
        auth_function = _current_auth_function
    user_info = auth_function(token)
    if inspect.isawaitable(user_info):
        user_info = await user_info
    return user_info

def _generate_presigned_token(
    user_info: UserInfo,
    expires_in: int,
):
    """Generate presigned tokens.

    This will generate a token which will be connected as a child user.
    Child user may generate more child user token if it has admin permission.
    """

    email = user_info.email
    # Inherit roles from parent
    roles = user_info.roles
    assert expires_in > 0, "expires_in should be greater than 0"
    current_time = datetime.datetime.now(datetime.timezone.utc)
    expires_at = current_time + datetime.timedelta(seconds=expires_in)

    token = jwt.encode(
        {
            "iss": AUTH0_ISSUER,
            "sub": user_info.id,  # user_id
            "aud": AUTH0_AUDIENCE,
            "iat": current_time,
            "exp": expires_at,
            "scope": generate_jwt_scope(user_info.scope),
            "gty": "client-credentials",
            AUTH0_NAMESPACE + "roles": roles,
            AUTH0_NAMESPACE + "email": email,
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return token

_generate_token_function = _generate_presigned_token

async def set_generate_token_function(generate_token_function: Callable):
    """Set the generate token function."""
    global _generate_token_function
    _generate_token_function = generate_token_function
    logger.info("Custom generate_token function has been set")

async def generate_auth_token(user_info: UserInfo, expires_in: int):
    """Generate a presigned token."""
    if _generate_token_function is None:
        generate = _generate_presigned_token
    else:
        generate = _generate_token_function
    
    result = generate(user_info, expires_in)
    if inspect.isawaitable(result):
        return await result
    else:
        return result

def parse_scope(scope: str) -> ScopeInfo:
    """Parse the scope."""
    parsed = ScopeInfo(extra_scopes=[])
    scopes = scope.split(" ")
    for scope in scopes:
        if scope.startswith("ws:"):
            name, mode = scope[3:].split("#")
            parsed.workspaces[name] = UserPermission(mode)
        elif scope.startswith("cid:"):
            parsed.client_id = scope[4:]
        elif scope.startswith("wid:"):
            parsed.current_workspace = scope[4:]
        elif scope.strip():
            parsed.extra_scopes.append(scope.strip())
    return parsed


def create_scope(
    workspaces: Union[str, Dict[str, UserPermission]] = None,
    client_id: str = None,
    current_workspace: str = None,
    extra_scopes: List[str] = None,
) -> ScopeInfo:
    """Create a scope."""
    # workspace is a quick shortcut to create a scope the format is workspace#mode with comma separated for multiple workspaces
    if isinstance(workspaces, str):
        workspaces = workspaces.split(",")
        # parse mode by #
        workspaces = {
            w.split("#")[0]: UserPermission(w.split("#")[1]) for w in workspaces
        }
    else:
        assert isinstance(
            workspaces, dict
        ), "Invalid workspaces, it should be a string or a dict"
        for w in list(workspaces.keys()):
            m = workspaces[w]
            # it should be either a string or a UserPermission
            if isinstance(m, str):
                assert m in UserPermission.__members__.values(), f"Invalid mode {m}"
                m = UserPermission(m)
            workspaces[w] = m

    return ScopeInfo(
        current_workspace=current_workspace,
        workspaces=workspaces,
        client_id=client_id,
        extra_scopes=extra_scopes,
    )


def update_user_scope(
    user_info: UserInfo, workspace_info: WorkspaceInfo, client_id: str = None
):
    """Update the user scope for a workspace."""
    user_info.scope = user_info.scope or ScopeInfo()
    permission = user_info.get_permission(workspace_info.id)
    ws_scopes = user_info.scope.workspaces.copy()
    if not permission:
        # infer permission from workspace
        if user_info.get_workspace() == workspace_info.id or workspace_info.owned_by(user_info):
            permission = UserPermission.admin

    if permission:
        ws_scopes[workspace_info.id] = permission

    if "admin" in user_info.roles:
        ws_scopes["*"] = UserPermission.admin

    return create_scope(
        workspaces=ws_scopes,
        client_id=client_id,
        current_workspace=workspace_info.id,
        extra_scopes=user_info.scope.extra_scopes,
    )


def generate_jwt_scope(scope: ScopeInfo) -> str:
    """Generate scope."""
    ps = " ".join([f"ws:{w}#{m.value}" for w, m in scope.workspaces.items()])

    if scope.client_id:
        ps += f" cid:{scope.client_id}"

    if scope.current_workspace:
        ps += f" wid:{scope.current_workspace}"

    if scope.extra_scopes:
        ps += " " + " ".join(scope.extra_scopes)
    return ps


def create_login_service(store):
    """Hypha startup function for registering additional services."""
    redis = store.get_redis()
    server_url = store.public_base_url
    login_service_url = f"{server_url}{LOGIN_SERVICE_URL}"
    generate_token_url = f"{server_url}/public/services/ws/generate_token"
    jinja_env = Environment(
        loader=PackageLoader("hypha"), autoescape=select_autoescape()
    )
    temp = jinja_env.get_template("apps/login_template.html")
    login_page = temp.render(
        hypha_version=__version__,
        login_service_url=login_service_url,
        generate_token_url=generate_token_url,
        auth0_client_id=AUTH0_CLIENT_ID,
        auth0_domain=AUTH0_DOMAIN,
        auth0_audience=AUTH0_AUDIENCE,
        auth0_issuer=AUTH0_ISSUER,
    )

    async def start_login(workspace: str = None, expires_in: int = None):
        """Start the login process."""
        key = str(random_id(readable=False))
        # set the key and with expire time
        await redis.setex(LOGIN_KEY_PREFIX + key, MAXIMUM_LOGIN_TIME, "")
        return {
            "login_url": f"{login_service_url.replace('/services/', '/apps/')}/?key={key}"
            + (
                f"&workspace={workspace}"
                if workspace
                else "" + f"&expires_in={expires_in}" if expires_in else ""
            ),
            "key": key,
            "report_url": f"{login_service_url}/report",
            "check_url": f"{login_service_url}/check",
            "generate_url": f"{login_service_url}/generate",
        }

    async def index(event):
        """Index function to serve the login page."""
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "body": login_page,
        }

    async def check_login(key, timeout=MAXIMUM_LOGIN_TIME, profile=False):
        """Check the status of a login session."""
        assert await redis.exists(
            LOGIN_KEY_PREFIX + key
        ), "Invalid key, key does not exist"
        if timeout <= 0:
            user_info = await redis.get(LOGIN_KEY_PREFIX + key)
            if user_info == b"":
                return None
            user_info = json.loads(user_info)
            user_info = UserTokenInfo.model_validate(user_info)
            if user_info:
                await redis.delete(LOGIN_KEY_PREFIX + key)
            return (
                user_info.model_dump(mode="json")
                if profile
                else (user_info and user_info.token)
            )
        count = 0
        while True:
            user_info = await redis.get(LOGIN_KEY_PREFIX + key)
            if user_info != b"":
                user_info = json.loads(user_info)
                user_info = UserTokenInfo.model_validate(user_info)
                if user_info is None:
                    raise Exception(
                        f"Login session expired, the maximum login time is {MAXIMUM_LOGIN_TIME} seconds"
                    )
                if user_info:
                    await redis.delete(LOGIN_KEY_PREFIX + key)
                    return (
                        user_info.model_dump(mode="json")
                        if profile
                        else user_info.token
                    )
            await asyncio.sleep(1)
            count += 1
            if count > timeout:
                raise Exception(f"Login timeout, waited for {timeout} seconds")

    async def report_login(
        key,
        token,
        workspace=None,
        expires_in=None,
        email=None,
        email_verified=None,
        name=None,
        nickname=None,
        user_id=None,
        picture=None,
    ):
        """Report a token associated with a login session."""
        assert await redis.exists(
            LOGIN_KEY_PREFIX + key
        ), "Invalid key, key does not exist or expired"
        # workspace = workspace or ("ws-user-" + user_id)
        kwargs = {
            "token": token,
            "workspace": workspace,
            "expires_in": expires_in or None,
            "email": email,
            "email_verified": email_verified,
            "name": name,
            "nickname": nickname,
            "user_id": user_id,
            "picture": picture,
        }

        user_token_info = UserTokenInfo.model_validate(kwargs)
        if workspace:
            user_info = await parse_auth_token(token)
            # based on the user token, create a scoped token
            workspace = workspace or user_info.get_workspace()
            # generate scoped token
            workspace_info = await store.load_or_create_workspace(user_info, workspace)
            user_info.scope = update_user_scope(user_info, workspace_info)
            if not user_info.check_permission(workspace, UserPermission.read):
                raise Exception(f"Invalid permission for the workspace {workspace}")

            token = await generate_auth_token(user_info, int(expires_in or 3600))
            # replace the token
            user_token_info.token = token
        
        await redis.setex(
            LOGIN_KEY_PREFIX + key,
            MAXIMUM_LOGIN_TIME,
            user_token_info.model_dump_json(),
        )

    async def profile(event):
        """Redirect to the Auth0 login page for profile management."""
        # For Auth0, we redirect to the login template which handles profile display
        return {
            "status": 302,
            "headers": {"Location": f"{login_service_url.replace('/services/', '/apps/')}"},
            "body": "Redirecting to profile page..."
        }

    logger.info(
        f"To preview the login page, visit: {login_service_url.replace('/services/', '/apps/')}"
    )
    return {
        "name": "Hypha Login",
        "id": "hypha-login",
        "type": "functions",
        "description": "Login service for Hypha",
        "config": {"visibility": "public"},
        "index": index,
        "start": start_login,
        "check": check_login,
        "report": report_login,
        "profile": profile,
    }

