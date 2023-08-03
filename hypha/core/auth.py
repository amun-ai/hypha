"""Provide authentication."""
import asyncio
import json
import logging
import ssl
import sys
import time
import traceback
from os import environ as env
from typing import List
from urllib.request import urlopen

import shortuuid
from dotenv import find_dotenv, load_dotenv
from fastapi import Header, HTTPException
from jinja2 import Environment, PackageLoader, select_autoescape
from jose import jwt
from pydantic import BaseModel  # pylint: disable=no-name-in-module

from hypha.core import TokenConfig, UserInfo
from hypha.utils import AsyncTTLCache

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("auth")
logger.setLevel(logging.INFO)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

MAXIMUM_LOGIN_TIME = env.get("MAXIMUM_LOGIN_TIME", "180")  # 3 minutes
AUTH0_CLIENT_ID = env.get("AUTH0_CLIENT_ID", "ofsvx6A7LdMhG0hklr5JCAEawLv4Pyse")
AUTH0_DOMAIN = env.get("AUTH0_DOMAIN", "imjoy.eu.auth0.com")
AUTH0_AUDIENCE = env.get("AUTH0_AUDIENCE", "https://imjoy.eu.auth0.com/api/v2/")
AUTH0_ISSUER = env.get("AUTH0_ISSUER", "https://imjoy.io/")
AUTH0_NAMESPACE = env.get("AUTH0_NAMESPACE", "https://api.imjoy.io/")
JWT_SECRET = env.get("JWT_SECRET")

if not JWT_SECRET:
    logger.warning("JWT_SECRET is not defined")
    JWT_SECRET = str(shortuuid.uuid())


class AuthError(Exception):
    """Represent an authentication error."""

    def __init__(self, error, status_code):
        """Set up instance."""
        super().__init__()
        self.error = error
        self.status_code = status_code


class ValidToken(BaseModel):
    """Represent a valid token."""

    credentials: dict
    scopes: List[str] = []

    def has_scope(self, checked_token):
        """Return True if the token has the correct scope."""
        if checked_token in self.scopes:
            return True

        raise HTTPException(
            status_code=403, detail="Not authorized to perform this action"
        )


def login_optional(authorization: str = Header(None)):
    """Return user info or create an anonymouse user.

    If authorization code is valid the user info is returned,
    If the code is invalid an an anonymouse user is created.
    """
    return parse_token(authorization, allow_anonymouse=True)


def login_required(authorization: str = Header(None)):
    """Return user info if authorization code is valid."""
    return parse_token(authorization)


def admin_required(authorization: str = Header(None)):
    """Return user info if the authorization code has an admin role."""
    token = parse_token(authorization)
    roles = token.credentials.get(AUTH0_NAMESPACE + "roles", [])
    if "admin" not in roles:
        raise HTTPException(status_code=401, detail="Admin required")
    return token


def is_admin(token):
    """Check if token has an admin role."""
    roles = token.credentials.get(AUTH0_NAMESPACE + "roles", [])
    if "admin" not in roles:
        return False
    return True


def get_user_email(token):
    """Return the user email from the token."""
    return token.credentials.get(AUTH0_NAMESPACE + "email")


def get_user_id(token):
    """Return the user id from the token."""
    return token.credentials.get("sub")


def get_user_info(token):
    """Return the user info from the token."""
    credentials = token.credentials
    expires_at = credentials["exp"]
    info = UserInfo(
        id=credentials.get("sub"),
        is_anonymous=not credentials.get(AUTH0_NAMESPACE + "email"),
        email=credentials.get(AUTH0_NAMESPACE + "email"),
        parent=credentials.get("parent", None),
        roles=credentials.get(AUTH0_NAMESPACE + "roles", []),
        scopes=token.scopes,
        expires_at=expires_at,
    )
    if credentials.get("pc"):
        info.set_metadata("parent_client", credentials.get("pc"))
    return info


JWKS = None


def get_rsa_key(kid, refresh=False):
    """Return an rsa key."""
    global JWKS  # pylint: disable=global-statement
    if JWKS is None or refresh:
        with urlopen(
            f"https://{AUTH0_DOMAIN}/.well-known/jwks.json",
            # pylint: disable=protected-access
            context=ssl._create_unverified_context(),
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


def simulate_user_token(returned_token, request):
    """Allow admin all_users to simulate another user."""
    if "user_id" in request.query_params:
        returned_token.credentials["sub"] = request.query_params["user_id"]
    if "email" in request.query_params:
        returned_token.credentials[AUTH0_NAMESPACE + "email"] = request.query_params[
            "email"
        ]
    if "roles" in request.query_params:
        returned_token.credentials[AUTH0_NAMESPACE + "roles"] = request.query_params[
            "roles"
        ].split(",")


def valid_token(authorization: str):
    """Validate token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is expected")

    try:
        unverified_header = jwt.get_unverified_header(authorization)

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

        returned_token = ValidToken(
            credentials=payload, scopes=payload["scope"].split(" ")
        )

        return returned_token

    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=401, detail="The token has expired. Please fetch a new one"
        ) from err
    except jwt.JWTError as err:
        raise HTTPException(status_code=401, detail=traceback.format_exc()) from err


def generate_anonymouse_user():
    """Generate user info for a anonymouse user."""
    iat = time.time()
    return ValidToken(
        credentials={
            "iss": AUTH0_ISSUER,
            "sub": shortuuid.uuid(),  # user_id
            "aud": AUTH0_AUDIENCE,
            "iat": iat,
            "exp": iat + 600,
            "azp": "aormkFV0l7T0shrIwjdeQIUmNLt09DmA",
            "scope": "",
            "gty": "client-credentials",
            AUTH0_NAMESPACE + "roles": [],
            AUTH0_NAMESPACE + "email": None,
        },
        scopes=[],
    )


def parse_token(authorization: str, allow_anonymouse=False):
    """Parse the token."""
    if not authorization:
        if allow_anonymouse:
            info = generate_anonymouse_user()
            return get_user_info(info)
        raise HTTPException(status_code=401, detail="Authorization header is expected")

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

    if "@imjoy@" not in token:
        # auth0 token
        info = valid_token(token)
    else:
        # generated token
        token = token.split("@imjoy@")[1]
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            audience=AUTH0_AUDIENCE,
            issuer=AUTH0_ISSUER,
        )
        info = ValidToken(credentials=payload, scopes=payload["scope"].split(" "))
    return get_user_info(info)


def generate_presigned_token(
    user_info: UserInfo, config: TokenConfig, child: bool = True
):
    """Generate presigned tokens.

    This will generate a token which will be connected as a child user.
    Child user may generate more child user token if it has admin permission.
    """
    scopes = config.scopes

    if child:
        # always generate a new user id
        uid = shortuuid.uuid()
        parent = user_info.parent if user_info.parent else user_info.id
        email = config.email
    else:
        uid = user_info.id
        parent = user_info.parent
        email = user_info.email

    # Inherit roles from parent
    roles = user_info.roles

    expires_in = config.expires_in or 10800
    current_time = time.time()
    expires_at = current_time + expires_in
    token = jwt.encode(
        {
            "iss": AUTH0_ISSUER,
            "sub": uid,  # user_id
            "aud": AUTH0_AUDIENCE,
            "iat": current_time,
            "exp": expires_at,
            "scope": " ".join(scopes),
            "parent": parent,
            "pc": config.parent_client,
            "gty": "client-credentials",
            AUTH0_NAMESPACE + "roles": roles,
            AUTH0_NAMESPACE + "email": email,
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return uid + "@imjoy@" + token


def generate_reconnection_token(
    user_info: UserInfo, client_id: str, workspace: str, expires_in: int = 60
):
    """Generate a token for reconnection."""
    current_time = time.time()
    expires_at = current_time + expires_in
    ret = jwt.encode(
        {
            "iss": AUTH0_ISSUER,
            "sub": user_info.id,
            "aud": AUTH0_AUDIENCE,
            "iat": current_time,
            "exp": expires_at,
            "gty": "client-credentials",
            "cid": client_id,
            "ws": workspace,
            AUTH0_NAMESPACE + "email": user_info.email,
            AUTH0_NAMESPACE + "roles": user_info.roles,
            "parent": user_info.parent,
            "scope": " ".join(user_info.scopes),
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return ret


def parse_reconnection_token(token):
    """Parse a reconnection token."""
    payload = jwt.decode(
        token,
        JWT_SECRET,
        algorithms=["HS256"],
        audience=AUTH0_AUDIENCE,
        issuer=AUTH0_ISSUER,
    )
    info = ValidToken(credentials=payload, scopes=payload["scope"].split(" "))
    return get_user_info(info), payload["ws"], payload["cid"]


def parse_user(token):
    """Parse user info from a token."""
    if token:
        user_info = parse_token(token)
        uid = user_info.id
        logger.info("User connected: %s", uid)
    else:
        uid = shortuuid.uuid()
        user_info = UserInfo(
            id=uid,
            is_anonymous=True,
            email=None,
            parent=None,
            roles=[],
            scopes=[],
            expires_at=None,
        )
        logger.info("Anonymized User connected: %s", uid)

    if uid == "root":
        logger.error("Root user is not allowed to connect remotely")
        raise Exception("Root user is not allowed to connect remotely")

    return user_info


async def register_login_service(server):
    """Hypha startup function for registering additional services."""
    cache = AsyncTTLCache(ttl=int(MAXIMUM_LOGIN_TIME))
    server_url = server.config["public_base_url"]
    login_url = f"{server_url}/{server.config['workspace']}/apps/hypha-login/"
    report_url = (
        f"{server_url}/{server.config['workspace']}/services/hypha-login/report"
    )

    jinja_env = Environment(
        loader=PackageLoader("hypha"), autoescape=select_autoescape()
    )
    temp = jinja_env.get_template("login_template.html")
    login_page = temp.render(
        report_url=report_url,
        auth0_client_id=AUTH0_CLIENT_ID,
        auth0_domain=AUTH0_DOMAIN,
        auth0_audience=AUTH0_AUDIENCE,
        auth0_issuer=AUTH0_ISSUER,
    )

    login_page = login_page.replace("{{ TOKEN_REPORT_URL }}", report_url)

    async def start_login():
        """Start the login process."""
        key = str(shortuuid.uuid())
        await cache.add(key, False)
        return {
            "login_url": f"{login_url}?key={key}",
            "key": key,
            "report_url": report_url,
        }

    async def index(event):
        """Index function to serve the login page."""
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "body": login_page,
        }

    async def check_login(key, timeout=MAXIMUM_LOGIN_TIME):
        """Check the status of a login session."""
        assert key in cache, "Invalid key, key does not exist"
        if timeout <= 0:
            return await cache.get(key)
        count = 0
        while True:
            token = await cache.get(key)
            if token is None:
                raise Exception(
                    f"Login session expired, the maximum login time is {MAXIMUM_LOGIN_TIME} seconds"
                )
            if token:
                del cache[key]
                return token
            await asyncio.sleep(1)
            count += 1
            if count > timeout:
                raise Exception("Login timeout")

    async def report_token(key, token):
        """Report a token associated with a login session."""
        await cache.update(key, token)

    await server.register_service(
        {
            "name": "Hypha Login",
            "id": "hypha-login",
            "type": "functions",
            "description": "Login service for Hypha",
            "config": {"visibility": "public"},
            "index": index,
            "start": start_login,
            "check": check_login,
            "report": report_token,
        }
    )

    logger.info("Login service is ready.")
    logger.info(f"To preview the login page, visit: {login_url}")
