"""Provide authentication."""
import json
import logging
import ssl
import sys
import time
import traceback
import uuid
from os import environ as env
from typing import List
from urllib.request import urlopen
import shortuuid

from dotenv import find_dotenv, load_dotenv
from fastapi import Header, HTTPException
from jose import jwt
from pydantic import BaseModel  # pylint: disable=no-name-in-module

from hypha.core import UserInfo, TokenConfig

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("auth")
logger.setLevel(logging.INFO)

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

AUTH0_DOMAIN = env.get("AUTH0_DOMAIN", "imjoy.eu.auth0.com")
AUTH0_AUDIENCE = env.get("AUTH0_AUDIENCE", "https://imjoy.eu.auth0.com/api/v2/")
JWT_SECRET = env.get("JWT_SECRET")
if not JWT_SECRET:
    logger.warning("JWT_SECRET is not defined")
    JWT_SECRET = str(uuid.uuid4())


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
    roles = token.credentials.get("https://api.imjoy.io/roles", [])
    if "admin" not in roles:
        raise HTTPException(status_code=401, detail="Admin required")
    return token


def is_admin(token):
    """Check if token has an admin role."""
    roles = token.credentials.get("https://api.imjoy.io/roles", [])
    if "admin" not in roles:
        return False
    return True


def get_user_email(token):
    """Return the user email from the token."""
    return token.credentials.get("https://api.imjoy.io/email")


def get_user_id(token):
    """Return the user id from the token."""
    return token.credentials.get("sub")


def get_user_info(token):
    """Return the user info from the token."""
    credentials = token.credentials
    expires_at = credentials["exp"]
    return UserInfo(
        id=credentials.get("sub"),
        is_anonymous=False,
        email=credentials.get("https://api.imjoy.io/email"),
        parent=credentials.get("parent", None),
        roles=credentials.get("https://api.imjoy.io/roles", []),
        scopes=token.scopes,
        expires_at=expires_at,
    )


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
        returned_token.credentials["https://api.imjoy.io/email"] = request.query_params[
            "email"
        ]
    if "roles" in request.query_params:
        returned_token.credentials["https://api.imjoy.io/roles"] = request.query_params[
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
            "iss": "https://imjoy.eu.auth0.com/",
            "sub": shortuuid.uuid(),  # user_id
            "aud": "https://imjoy.eu.auth0.com/api/v2/",
            "iat": iat,
            "exp": iat + 600,
            "azp": "aormkFV0l7T0shrIwjdeQIUmNLt09DmA",
            "scope": "",
            "gty": "client-credentials",
            "https://api.imjoy.io/roles": [],
            "https://api.imjoy.io/email": "anonymous@imjoy.io",
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
            issuer="https://imjoy.io/",
        )
        info = ValidToken(credentials=payload, scopes=payload["scope"].split(" "))
    return get_user_info(info)


def generate_presigned_token(user_info: UserInfo, config: TokenConfig):
    """Generate presigned tokens.

    This will generate a token which will be connected as a child user.
    Child user may generate more child user token if it has admin permission.
    """
    scopes = config.scopes

    # always generate a new user id
    uid = shortuuid.uuid()
    expires_in = config.expires_in or 10800
    expires_at = time.time() + expires_in

    parent = user_info.parent if user_info.parent else user_info.id

    token = jwt.encode(
        {
            "iss": "https://imjoy.io/",
            "sub": uid,  # user_id
            "aud": "https://imjoy.eu.auth0.com/api/v2/",
            "iat": expires_in,
            "exp": expires_at,
            "scope": " ".join(scopes),
            "parent": parent,
            "gty": "client-credentials",
            "https://api.imjoy.io/roles": [],
            "https://api.imjoy.io/email": config.email,
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return uid + "@imjoy@" + token
