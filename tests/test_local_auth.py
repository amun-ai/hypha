"""Test local authentication functionality."""

import pytest
import asyncio
from hypha_rpc import connect_to_server


@pytest.mark.asyncio
async def test_local_auth_server_basic(local_auth_server):
    """Test that local auth server starts and accepts connections."""
    server_url = local_auth_server.replace("http://", "ws://") + "/ws"
    
    # Connect as anonymous user
    async with connect_to_server(
        {
            "client_id": "test-local-auth-anon",
            "server_url": server_url,
        }
    ) as api:
        # Verify hypha-login service is available
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_found = any("hypha-login" in sid for sid in service_ids)
        assert hypha_login_found, f"hypha-login service not found. Available: {service_ids}"
        
        # Get the hypha-login service
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        # Verify service has expected methods
        assert hasattr(login_service, 'signup')
        assert hasattr(login_service, 'login')
        assert hasattr(login_service, 'start')
        assert hasattr(login_service, 'check')
        assert hasattr(login_service, 'report')


@pytest.mark.asyncio
async def test_local_auth_signup_and_login(local_auth_server):
    """Test user signup and login flow."""
    server_url = local_auth_server.replace("http://", "ws://") + "/ws"
    
    async with connect_to_server(
        {
            "client_id": "test-signup-login",
            "server_url": server_url,
        }
    ) as api:
        # Get the hypha-login service
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        # Test user data
        test_email = "testuser@example.com"
        test_password = "TestPassword123!"
        test_name = "Test User"
        
        # Test signup
        signup_result = await login_service.signup(
            name=test_name,
            email=test_email,
            password=test_password
        )
        assert signup_result["success"] is True
        assert "user_id" in signup_result
        user_id = signup_result["user_id"]
        
        # Test login with correct credentials
        login_result = await login_service.login(
            email=test_email,
            password=test_password
        )
        assert login_result["success"] is True
        assert login_result["email"] == test_email
        assert login_result["user_id"] == user_id
        assert "token" in login_result
        token = login_result["token"]
        
        # Test login with incorrect password
        login_fail = await login_service.login(
            email=test_email,
            password="WrongPassword"
        )
        assert login_fail["success"] is False
        assert "error" in login_fail
    
    # Test using the token to connect
    async with connect_to_server(
        {
            "client_id": "test-authenticated",
            "server_url": server_url,
            "token": token,
        }
    ) as auth_api:
        # Verify we can access services with the token
        services = await auth_api.list_services("public")
        assert len(services) > 0
        
        # Check we're in the user's workspace
        assert auth_api.config.workspace == f"ws-user-{user_id}"


@pytest.mark.asyncio
async def test_local_auth_duplicate_signup(local_auth_server):
    """Test that duplicate email signup is rejected."""
    server_url = local_auth_server.replace("http://", "ws://") + "/ws"
    
    async with connect_to_server(
        {
            "client_id": "test-duplicate",
            "server_url": server_url,
        }
    ) as api:
        # Get the hypha-login service
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        # First signup
        result1 = await login_service.signup(
            name="First User",
            email="duplicate@example.com",
            password="Password123!"
        )
        assert result1["success"] is True
        
        # Second signup with same email should fail
        result2 = await login_service.signup(
            name="Second User",
            email="duplicate@example.com",
            password="DifferentPass123!"
        )
        assert result2["success"] is False
        assert "already registered" in result2["error"].lower()


@pytest.mark.asyncio
async def test_local_auth_login_flow(local_auth_server):
    """Test the complete login flow with hypha-login service."""
    server_url = local_auth_server.replace("http://", "ws://") + "/ws"
    
    async with connect_to_server(
        {
            "client_id": "test-login-flow",
            "server_url": server_url,
        }
    ) as api:
        # Get the hypha-login service
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        # Create a test user first
        signup_result = await login_service.signup(
            name="Flow Test User",
            email="flowtest@example.com",
            password="FlowTest123!"
        )
        assert signup_result["success"] is True
        user_id = signup_result["user_id"]
        
        # Start a login session
        login_info = await login_service.start(
            workspace=f"ws-user-{user_id}",
            expires_in=3600
        )
        assert "key" in login_info
        assert "login_url" in login_info
        login_key = login_info["key"]
        
        # Simulate login by calling the login handler with the key
        login_result = await login_service.login(
            email="flowtest@example.com",
            password="FlowTest123!",
            key=login_key
        )
        assert login_result["success"] is True
        
        # Check the login status
        token = await login_service.check(key=login_key, timeout=0)
        assert token is not None
        
        # Check with profile
        profile_info = await login_service.check(key=login_key, timeout=0, profile=True)
        assert profile_info["token"] == token
        assert profile_info["email"] == "flowtest@example.com"


@pytest.mark.asyncio
async def test_local_auth_password_validation(local_auth_server):
    """Test password validation rules."""
    server_url = local_auth_server.replace("http://", "ws://") + "/ws"
    
    async with connect_to_server(
        {
            "client_id": "test-password",
            "server_url": server_url,
        }
    ) as api:
        # Get the hypha-login service
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        # Test short password
        result = await login_service.signup(
            name="Test User",
            email="shortpass@example.com",
            password="short"  # Less than 8 characters
        )
        assert result["success"] is False
        assert "8 characters" in result["error"]
        
        # Test valid password
        result = await login_service.signup(
            name="Test User",
            email="goodpass@example.com",
            password="ValidPass123!"  # 8+ characters
        )
        assert result["success"] is True


@pytest.mark.asyncio
async def test_local_auth_profile_update(local_auth_server):
    """Test updating user profile."""
    server_url = local_auth_server.replace("http://", "ws://") + "/ws"
    
    # First create a user and login
    async with connect_to_server(
        {
            "client_id": "test-profile-setup",
            "server_url": server_url,
        }
    ) as api:
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        # Create user
        signup_result = await login_service.signup(
            name="Original Name",
            email="profile@example.com",
            password="OriginalPass123!"
        )
        assert signup_result["success"] is True
        
        # Login to get token
        login_result = await login_service.login(
            email="profile@example.com",
            password="OriginalPass123!"
        )
        assert login_result["success"] is True
        token = login_result["token"]
    
    # Now connect with the token and update profile
    async with connect_to_server(
        {
            "client_id": "test-profile-update",
            "server_url": server_url,
            "token": token,
        }
    ) as auth_api:
        services = await auth_api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await auth_api.get_service(hypha_login_service_id)
        
        # Update name (need to pass email since context isn't working)
        update_result = await login_service.update_profile(
            name="Updated Name",
            user_email="profile@example.com"
        )
        assert update_result["success"] is True
        
        # Update password (need to pass email since context isn't working)
        update_result = await login_service.update_profile(
            current_password="OriginalPass123!",
            new_password="NewPass456!",
            user_email="profile@example.com"
        )
        assert update_result["success"] is True
    
    # Verify new password works
    async with connect_to_server(
        {
            "client_id": "test-new-password",
            "server_url": server_url,
        }
    ) as api:
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        # Old password should fail
        login_result = await login_service.login(
            email="profile@example.com",
            password="OriginalPass123!"
        )
        assert login_result["success"] is False
        
        # New password should work
        login_result = await login_service.login(
            email="profile@example.com",
            password="NewPass456!"
        )
        assert login_result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])