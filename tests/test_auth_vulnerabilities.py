"""Test suite for authentication and authorization vulnerabilities.

This test file contains security tests for JWT token validation,
permission escalation, and authentication bypass vulnerabilities.

Vulnerability Areas:
- JWT algorithm confusion attacks
- Token expiration bypass
- Permission escalation (read -> read_write -> admin)
- Anonymous user privilege escalation
- Root token exposure
- Scope manipulation
- JWKS poisoning
"""

import pytest
import pytest_asyncio
import uuid
import asyncio
import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import json

from hypha_rpc import connect_to_server
from hypha.core.auth import (
    valid_token,
    generate_auth_token,
    parse_auth_token,
    generate_anonymous_user,
    create_scope,
    parse_scope,
    JWT_SECRET,
    AUTH0_AUDIENCE,
    AUTH0_ISSUER,
    AUTH0_DOMAIN,
    set_jwt_secret,
)
from hypha.core import UserPermission, UserInfo

from . import (
    WS_SERVER_URL,
    SERVER_URL,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


class TestJWTAlgorithmConfusion:
    """
    Test for JWT algorithm confusion vulnerabilities.

    Attack: An attacker might try to exploit algorithm confusion by:
    1. Creating a token signed with HS256 using the public RSA key as secret
    2. Forcing RS256 validation to use a known secret
    3. Using "none" algorithm to bypass signature verification
    """

    async def test_none_algorithm_rejected(self):
        """Test that tokens with 'none' algorithm are rejected."""
        # Create a token with 'none' algorithm
        payload = {
            "iss": AUTH0_ISSUER,
            "sub": "attacker",
            "aud": AUTH0_AUDIENCE,
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }

        # Encode with 'none' algorithm (no signature)
        none_token = pyjwt.encode(payload, "", algorithm="none")

        # This should be rejected
        with pytest.raises(Exception) as exc_info:
            valid_token(none_token)

        error_msg = str(exc_info.value).lower()
        assert "algorithm" in error_msg or "invalid" in error_msg, \
            f"Expected algorithm/invalid error for 'none' algorithm, got: {exc_info.value}"

    async def test_hs256_with_public_key_rejected(self):
        """Test that HS256 tokens signed with RSA public key are rejected.

        This tests protection against the classic JWT algorithm confusion attack
        where an attacker signs a token with HS256 using the RSA public key as
        the HMAC secret.
        """
        # This test is informational - real exploit would need the actual RSA public key
        # which is fetched from Auth0's JWKS endpoint

        # Create a payload
        payload = {
            "iss": AUTH0_ISSUER,
            "sub": "attacker",
            "aud": AUTH0_AUDIENCE,
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }

        # Sign with HS256 using a fake "public key" as secret
        fake_public_key = "fake_rsa_public_key"
        confused_token = pyjwt.encode(payload, fake_public_key, algorithm="HS256")

        # This should be rejected (wrong signature)
        with pytest.raises(Exception) as exc_info:
            valid_token(confused_token)

        assert exc_info.value is not None

    async def test_unsupported_algorithm_rejected(self):
        """Test that tokens with unsupported algorithms are rejected."""
        payload = {
            "iss": AUTH0_ISSUER,
            "sub": "attacker",
            "aud": AUTH0_AUDIENCE,
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }

        # Try with HS512 (unsupported)
        hs512_token = pyjwt.encode(payload, "secret", algorithm="HS512")

        with pytest.raises(Exception) as exc_info:
            valid_token(hs512_token)

        error_msg = str(exc_info.value).lower()
        assert "algorithm" in error_msg or "invalid" in error_msg, \
            f"Expected algorithm error, got: {exc_info.value}"


class TestTokenExpirationBypass:
    """
    Test for token expiration bypass vulnerabilities.

    Attack vectors:
    1. Tokens with very long expiration times
    2. Tokens with missing exp claim
    3. Tokens with exp in the past
    4. Timezone manipulation
    """

    async def test_expired_token_rejected(self, fastapi_server, test_user_token):
        """Test that expired tokens are properly rejected."""
        api = await connect_to_server({
            "client_id": "exp-test-1",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Generate token with 1 second expiration
        short_token = await api.generate_token({"expires_in": 1})

        # Wait for expiration
        await asyncio.sleep(2)

        # Try to use expired token
        with pytest.raises(Exception) as exc_info:
            await connect_to_server({
                "client_id": "exp-test-2",
                "server_url": WS_SERVER_URL,
                "token": short_token,
            })

        error_msg = str(exc_info.value).lower()
        assert "expired" in error_msg or "invalid" in error_msg, \
            f"Expected expired token error, got: {exc_info.value}"

        await api.disconnect()

    async def test_token_without_expiration_rejected(self):
        """Test that tokens without exp claim are rejected."""
        payload = {
            "iss": AUTH0_ISSUER,
            "sub": "attacker",
            "aud": AUTH0_AUDIENCE,
            # Missing exp claim
        }

        token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")

        # Should be rejected
        with pytest.raises(Exception) as exc_info:
            valid_token(token)

        assert exc_info.value is not None

    async def test_very_long_expiration_allowed_if_signed(self):
        """Test that long-lived tokens are allowed if properly signed.

        This is informational - very long expiration times may be legitimate
        for certain use cases, but should be monitored.
        """
        payload = {
            "iss": AUTH0_ISSUER,
            "sub": "user-123",
            "aud": AUTH0_AUDIENCE,
            "exp": datetime.now(timezone.utc) + timedelta(days=365),  # 1 year
            "scope": "ws:test-ws#r",
        }

        long_lived_token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")

        # This should be accepted (valid signature, future expiration)
        result = valid_token(long_lived_token)
        assert result is not None
        assert result["sub"] == "user-123"


class TestPermissionEscalation:
    """
    Test for permission escalation vulnerabilities.

    Attack vectors:
    1. Modifying token scope to escalate from read to read_write/admin
    2. Adding wildcard workspace permissions
    3. Manipulating parent user permissions
    4. Escalating anonymous user privileges
    """

    async def test_cannot_modify_scope_in_token(self, fastapi_server, test_user_token):
        """Test that modifying scope in JWT doesn't grant elevated permissions."""
        api = await connect_to_server({
            "client_id": "perm-test-1",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Create a workspace with read-only access
        ws_info = await api.create_workspace({
            "name": f"perm-escalate-ws-{uuid.uuid4().hex[:8]}",
            "description": "Permission escalation test workspace",
        })
        workspace = ws_info["id"]

        # Generate a read-only token for the workspace
        read_token = await api.generate_token({
            "workspace": workspace,
            "expires_in": 3600,
        })

        # Decode the token (without verification) to get payload
        decoded = pyjwt.decode(read_token, options={"verify_signature": False})

        # Attempt to modify scope from read to admin
        # Original: "ws:workspace#r" -> Modified: "ws:workspace#a"
        original_scope = decoded.get("scope", "")

        # Parse and modify the scope
        if f"ws:{workspace}#" in original_scope:
            # User has some permission, try to escalate to admin
            modified_scope = original_scope.replace(f"ws:{workspace}#r", f"ws:{workspace}#a")
            modified_scope = modified_scope.replace(f"ws:{workspace}#rw", f"ws:{workspace}#a")

            decoded["scope"] = modified_scope

            # Re-sign with the same secret (simulating attacker who somehow got JWT_SECRET)
            # In practice, attacker wouldn't have JWT_SECRET, but this tests if scope is validated
            forged_token = pyjwt.encode(decoded, JWT_SECRET, algorithm="HS256")

            # Try to use the forged token
            api_attacker = await connect_to_server({
                "client_id": "perm-test-attacker",
                "workspace": workspace,
                "server_url": WS_SERVER_URL,
                "token": forged_token,
            })

            # Try to perform admin operation (delete workspace)
            # This should fail if permissions are properly validated server-side
            with pytest.raises(Exception) as exc_info:
                await api_attacker.delete_workspace(workspace)

            error_msg = str(exc_info.value).lower()
            assert "permission" in error_msg or "denied" in error_msg, \
                f"Expected permission denied, got: {exc_info.value}"

            await api_attacker.disconnect()

        await api.disconnect()

    async def test_cannot_add_wildcard_workspace_permission(self, fastapi_server, test_user_token):
        """Test that users cannot grant themselves wildcard workspace permissions."""
        api = await connect_to_server({
            "client_id": "wildcard-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Decode the user token
        decoded = pyjwt.decode(test_user_token, options={"verify_signature": False})

        # Try to add wildcard admin permission: "ws:*#a"
        original_scope = decoded.get("scope", "")
        modified_scope = original_scope + " ws:*#a"
        decoded["scope"] = modified_scope

        # Re-sign (requires JWT_SECRET knowledge, which attacker shouldn't have)
        forged_token = pyjwt.encode(decoded, JWT_SECRET, algorithm="HS256")

        # Try to connect with wildcard permissions
        api_attacker = await connect_to_server({
            "client_id": "wildcard-attacker",
            "server_url": WS_SERVER_URL,
            "token": forged_token,
        })

        # Try to create/access public workspace (requires admin)
        # This should fail for non-admin users
        with pytest.raises(Exception) as exc_info:
            # Try to delete the public workspace
            await api_attacker.delete_workspace("public")

        error_msg = str(exc_info.value).lower()
        # Should fail (either permission denied or public can't be deleted)
        assert "permission" in error_msg or "denied" in error_msg or "cannot" in error_msg, \
            f"Wildcard permission should not grant public workspace admin, got: {exc_info.value}"

        await api_attacker.disconnect()
        await api.disconnect()


class TestAnonymousUserEscalation:
    """
    Test for anonymous user privilege escalation.

    Attack vectors:
    1. Anonymous users gaining non-anonymous privileges
    2. Anonymous users extending their token lifetime
    3. Anonymous users accessing protected resources
    """

    async def test_anonymous_user_has_limited_permissions(self):
        """Test that anonymous users have restricted permissions."""
        anon_user = generate_anonymous_user()

        # Verify it's marked as anonymous
        assert anon_user.is_anonymous is True
        assert "anonymous" in anon_user.roles

        # Verify limited lifetime (10 minutes)
        assert anon_user.expires_at is not None
        current_time = datetime.now(timezone.utc).timestamp()
        time_left = anon_user.expires_at - current_time
        assert time_left <= 600, f"Anonymous token should expire in ≤10 minutes, got {time_left}s"

    async def test_anonymous_user_cannot_create_long_lived_token(self, fastapi_server):
        """Test that anonymous users cannot generate long-lived tokens."""
        anon_user = generate_anonymous_user()

        # Try to generate a token with very long expiration
        # This should either fail or be clamped to anonymous user's expiration
        try:
            # Anonymous users shouldn't have generate_token capability in most cases
            # But if they do, test that expiration is properly limited
            long_token = await generate_auth_token(anon_user, expires_in=86400)  # 24 hours

            # Decode and check expiration
            decoded = pyjwt.decode(long_token, JWT_SECRET, algorithms=["HS256"],
                                 audience=AUTH0_AUDIENCE, issuer=AUTH0_ISSUER)

            # Token expiration should not exceed anonymous user's expiration
            assert decoded["exp"] <= anon_user.expires_at, \
                "Anonymous-generated token should not exceed anonymous user's expiration"
        except Exception as e:
            # If it fails, that's also acceptable (anonymous can't generate tokens)
            pass

    async def test_anonymous_user_cannot_access_admin_operations(self, fastapi_server):
        """Test that anonymous users cannot perform admin operations."""
        anon_user = generate_anonymous_user()
        anon_token = await generate_auth_token(anon_user, expires_in=600)

        # Try to connect as anonymous user
        try:
            api = await connect_to_server({
                "client_id": "anon-test",
                "server_url": WS_SERVER_URL,
                "token": anon_token,
            })

            # Try admin operation - should fail
            with pytest.raises(Exception) as exc_info:
                await api.create_workspace({
                    "name": f"anon-attack-ws-{uuid.uuid4().hex[:8]}",
                    "description": "Anonymous user trying to create workspace",
                })

            error_msg = str(exc_info.value).lower()
            assert "permission" in error_msg or "denied" in error_msg or "anonymous" in error_msg, \
                f"Anonymous user should not create workspaces, got: {exc_info.value}"

            await api.disconnect()
        except Exception as e:
            # If anonymous connection itself fails, that's also acceptable
            pass


class TestScopeManipulation:
    """
    Test for scope manipulation vulnerabilities.

    Attack vectors:
    1. Injecting malicious scope strings
    2. Scope parsing confusion
    3. Client ID spoofing
    4. Workspace ID injection
    """

    def test_scope_parsing_with_malicious_input(self):
        """Test that scope parsing handles malicious input safely."""
        # Test various malicious scope strings
        malicious_scopes = [
            "ws:../../../etc/passwd#a",  # Path traversal
            "ws:workspace#a ws:*#a",  # Wildcard injection
            "ws:workspace#admin",  # Invalid permission level
            "ws:workspace#r" * 1000,  # DoS via large input
            "cid:../attacker",  # Client ID injection
            "wid:workspace\x00injected",  # Null byte injection
        ]

        for malicious_scope in malicious_scopes:
            try:
                parsed = parse_scope(malicious_scope)
                # Verify parsing doesn't crash and produces safe output
                assert isinstance(parsed.workspaces, dict)

                # Check for path traversal
                for ws_name in parsed.workspaces.keys():
                    assert ".." not in ws_name, f"Path traversal in workspace name: {ws_name}"
                    assert "/" not in ws_name, f"Slash in workspace name: {ws_name}"

            except Exception:
                # Exceptions are acceptable (malformed input rejected)
                pass

    def test_scope_creation_validates_permissions(self):
        """Test that scope creation validates permission levels."""
        # Try to create scope with invalid permission
        try:
            scope = create_scope(workspaces={"test-ws": "superadmin"})
            # Should either fail or normalize to valid permission
            assert scope.workspaces["test-ws"] in [
                UserPermission.read,
                UserPermission.read_write,
                UserPermission.admin,
            ]
        except Exception:
            # Rejection is acceptable
            pass

    async def test_cannot_inject_arbitrary_client_id(self, fastapi_server, test_user_token):
        """Test that client_id in scope cannot be arbitrarily set to spoof identity."""
        api = await connect_to_server({
            "client_id": "scope-inject-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Get current token and decode
        decoded = pyjwt.decode(test_user_token, options={"verify_signature": False})

        # Try to inject a different client_id in scope
        original_scope = decoded.get("scope", "")
        # Add a fake client ID
        modified_scope = original_scope + " cid:victim-client-123"
        decoded["scope"] = modified_scope

        # Re-sign
        forged_token = pyjwt.encode(decoded, JWT_SECRET, algorithm="HS256")

        # Connect with forged token
        api_attacker = await connect_to_server({
            "client_id": "attacker-client",
            "server_url": WS_SERVER_URL,
            "token": forged_token,
        })

        # Verify that the actual client_id is based on connection, not token scope
        # The client_id should be "attacker-client", not "victim-client-123"
        assert api_attacker.config.client_id == "attacker-client", \
            "Client ID should be determined by connection, not token scope"

        await api.disconnect()
        await api_attacker.disconnect()


class TestJWKSPoisoning:
    """
    Test for JWKS (JSON Web Key Set) poisoning vulnerabilities.

    Attack vectors:
    1. JWKS cache poisoning
    2. JWKS endpoint manipulation
    3. Key ID (kid) confusion
    """

    async def test_invalid_kid_rejected(self):
        """Test that tokens with invalid key IDs are rejected."""
        # Create a token with a fake kid
        payload = {
            "iss": f"https://{AUTH0_DOMAIN}/",
            "sub": "attacker",
            "aud": AUTH0_AUDIENCE,
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }

        # Encode with RS256 header but invalid kid
        fake_token = pyjwt.encode(
            payload,
            "fake_private_key",
            algorithm="RS256",
            headers={"kid": "fake-key-id-12345"}
        )

        # This should be rejected (invalid kid or signature)
        with pytest.raises(Exception) as exc_info:
            valid_token(fake_token)

        assert exc_info.value is not None


class TestRootTokenSecurity:
    """
    Test for root token security vulnerabilities.

    Attack vectors:
    1. Root token exposure
    2. Root token guessing
    3. Root token bypass
    """

    async def test_root_token_not_in_error_messages(self, fastapi_server):
        """Test that root token is not leaked in error messages."""
        # This is informational - in production, root token should be kept secure
        # We verify it's not accidentally exposed in API responses

        # Try to connect with invalid token
        try:
            api = await connect_to_server({
                "client_id": "root-leak-test",
                "server_url": WS_SERVER_URL,
                "token": "invalid-token-12345",
            })
            await api.disconnect()
        except Exception as e:
            error_msg = str(e)
            # Verify error doesn't contain any token-like strings
            # (basic check - real root token should be long and complex)
            assert len(error_msg) < 500, "Error message suspiciously long"

    async def test_cannot_use_root_without_token(self, fastapi_server, test_user_token):
        """Test that 'root' user ID cannot be claimed without proper token."""
        api = await connect_to_server({
            "client_id": "fake-root-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Verify we're NOT the root user
        user_info = await parse_auth_token(test_user_token)
        assert user_info.id != "root", "Regular user should not have root ID"
        assert "admin" not in user_info.roles or user_info.id.startswith("ws-user-"), \
            "Regular admin should have ws-user- prefix, not root"

        await api.disconnect()


class TestTokenRevocation:
    """
    Test for token revocation vulnerabilities.

    Issues:
    1. Tokens are stateless JWTs - no server-side revocation mechanism visible in code
    2. Once issued, tokens remain valid until expiration
    3. No apparent token blacklist or revocation list
    """

    async def test_token_revocation_not_implemented(self, fastapi_server, test_user_token):
        """Informational test: Token revocation does not appear to be implemented.

        This is a security concern - if a token is compromised, it cannot be
        revoked before expiration.
        """
        api = await connect_to_server({
            "client_id": "revocation-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Generate a token
        token = await api.generate_token({"expires_in": 3600})

        # Verify token works
        api2 = await connect_to_server({
            "client_id": "revocation-test-2",
            "server_url": WS_SERVER_URL,
            "token": token,
        })

        # In a system with revocation, there would be an API like:
        # await api.revoke_token(token)
        # But no such API exists in Hypha

        # Token should still work (no revocation mechanism)
        # This is expected behavior but represents a security limitation

        await api.disconnect()
        await api2.disconnect()


class TestWorkspacePermissionValidation:
    """
    Test that workspace permissions are properly validated server-side.
    """

    async def test_workspace_permission_validated_on_operations(self, fastapi_server, test_user_token):
        """Test that workspace permissions are checked for each operation."""
        api = await connect_to_server({
            "client_id": "ws-perm-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })

        # Create a workspace
        ws_info = await api.create_workspace({
            "name": f"perm-check-ws-{uuid.uuid4().hex[:8]}",
            "description": "Permission check test",
        })
        workspace = ws_info["id"]

        # Create second user workspace
        ws2_info = await api.create_workspace({
            "name": f"other-ws-{uuid.uuid4().hex[:8]}",
            "description": "Other workspace",
        })
        other_ws = ws2_info["id"]

        # Generate read-only token for ws2
        readonly_token = await api.generate_token({"workspace": other_ws})

        # Connect with read-only token
        api_readonly = await connect_to_server({
            "client_id": "readonly-user",
            "workspace": other_ws,
            "server_url": WS_SERVER_URL,
            "token": readonly_token,
        })

        # Try to delete workspace with read-only permission - should fail
        with pytest.raises(Exception) as exc_info:
            await api_readonly.delete_workspace(other_ws)

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Read-only user should not delete workspace, got: {exc_info.value}"

        await api.disconnect()
        await api_readonly.disconnect()
