"""Provide authentication."""
import asyncio
import json
import logging
import ssl
import sys
from calendar import timegm
import datetime
from os import environ as env
from typing import List, Union, Dict
from urllib.request import urlopen

import shortuuid
from dotenv import find_dotenv, load_dotenv
from fastapi import HTTPException
from jinja2 import Environment, PackageLoader, select_autoescape
from jose import jwt

from hypha.core import UserInfo, UserTokenInfo, ScopeInfo, UserPermission, WorkspaceInfo
from hypha.utils import random_id

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("auth")
logger.setLevel(logging.INFO)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

MAXIMUM_LOGIN_TIME = env.get("MAXIMUM_LOGIN_TIME", "180")  # 3 minutes
AUTH0_CLIENT_ID = env.get("AUTH0_CLIENT_ID", "paEagfNXPBVw8Ss80U5RAmAV4pjCPsD2")
AUTH0_DOMAIN = env.get("AUTH0_DOMAIN", "amun-ai.eu.auth0.com")
AUTH0_AUDIENCE = env.get("AUTH0_AUDIENCE", "https://amun-ai.eu.auth0.com/api/v2/")
AUTH0_ISSUER = env.get("AUTH0_ISSUER", "https://amun.ai/")
AUTH0_NAMESPACE = env.get("AUTH0_NAMESPACE", "https://amun.ai/")
JWT_SECRET = env.get("JWT_SECRET")
LOGIN_SERVICE_URL = "/public/services/hypha-login"
LOGIN_KEY_PREFIX = "login_key:"

if not JWT_SECRET:
    logger.warning(
        "JWT_SECRET is not defined, you will need a fixed JWT_SECRET for clients to reconnect"
    )
    JWT_SECRET = shortuuid.ShortUUID().random(length=22)


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
    user_id = random_id(readable=True)
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


def parse_token(authorization: str):
    """Parse the token."""
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

    payload = valid_token(token)
    return get_user_info(payload)


def generate_presigned_token(
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


def generate_reconnection_token(user_info: UserInfo, expires_in: int = 60):
    """Generate a token for reconnection."""
    current_time = datetime.datetime.now(datetime.timezone.utc)
    expires_at = current_time + datetime.timedelta(seconds=expires_in)

    ret = jwt.encode(
        {
            "iss": AUTH0_ISSUER,
            "sub": user_info.id,
            "aud": AUTH0_AUDIENCE,
            "iat": current_time,
            "exp": expires_at,
            "gty": "client-credentials",
            AUTH0_NAMESPACE + "email": user_info.email,
            AUTH0_NAMESPACE + "roles": user_info.roles,
            "scope": generate_jwt_scope(user_info.scope),
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return ret


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
        if (
            user_info.get_workspace() == workspace_info.id
            or user_info.email in workspace_info.owners
            or user_info.id in workspace_info.owners
        ):
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
                else "" + f"&expires_in={expires_in}"
                if expires_in
                else ""
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
            user_info = parse_token(token)
            # based on the user token, create a scoped token
            workspace = workspace or user_info.get_workspace()
            # generate scoped token
            workspace_info = await store.load_or_create_workspace(user_info, workspace)
            user_info.scope = update_user_scope(user_info, workspace_info)
            if not user_info.check_permission(workspace, UserPermission.read):
                raise Exception(f"Invalid permission for the workspace {workspace}")

            token = generate_presigned_token(user_info, int(expires_in or 3600))
            # replace the token
            user_token_info.token = token
        await redis.setex(
            LOGIN_KEY_PREFIX + key,
            MAXIMUM_LOGIN_TIME,
            user_token_info.model_dump_json(),
        )

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
    }
