"""Custom authentication startup function for testing."""
import time
from hypha.core import UserInfo, UserPermission


def custom_generate_token(user_info: UserInfo, expires_in: int) -> str:
    """Generate a custom token for testing."""
    # Create a simple custom token format: CUSTOM:user_id:workspace:expires
    # For anonymous users, preserve the original workspace name
    if user_info.scope.workspaces:
        workspace = list(user_info.scope.workspaces.keys())[0]
    elif user_info.scope.current_workspace:
        workspace = user_info.scope.current_workspace
    else:
        workspace = "default"
    expires_at = int(time.time()) + expires_in
    token = f"CUSTOM:{user_info.id}:{workspace}:{expires_at}"
    return token


async def custom_parse_token(token: str) -> UserInfo:
    """Parse a custom token for testing."""
    if not token.startswith("CUSTOM:"):
        # Fall back to default JWT parsing for non-custom tokens
        from hypha.core.auth import _parse_token
        return _parse_token(token)
    
    # Parse our custom token format
    parts = token.split(":")
    if len(parts) != 4:
        raise ValueError("Invalid custom token format")
    
    _, user_id, workspace, expires_at = parts
    
    # Check expiration
    if int(expires_at) < time.time():
        raise ValueError("Token has expired")
    
    # Create UserInfo from token
    from hypha.core.auth import create_scope
    
    # Handle different workspace formats
    # If workspace contains the user ID already (ws-user-xxx), extract the owner
    # Otherwise use the user_id from the token
    if workspace.startswith("ws-user-"):
        # For workspaces like ws-user-anonymouz-xxx, the user should have admin access
        # regardless of the token's user_id (which might be a child token)
        workspace_permissions = {workspace: UserPermission.admin}
    else:
        # For other workspaces, use standard permissions
        workspace_permissions = {workspace: UserPermission.admin}
    
    user_info = UserInfo(
        id=user_id,
        is_anonymous=False,
        email=f"{user_id}@example.com",
        parent=None,
        roles=[],
        scope=create_scope(
            workspaces=workspace_permissions,
            current_workspace=workspace
        ),
        expires_at=None,
    )
    return user_info


async def hypha_startup(server):
    """Startup function to register custom authentication."""
    # Use the new simplified register_auth_service function
    await server.register_auth_service(
        parse_token=custom_parse_token,
        generate_token=custom_generate_token
    )
    
    # Register a test service to verify the custom auth is working
    await server.register_service(
        {
            "id": "custom-auth-test",
            "name": "Custom Auth Test Service",
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "echo": lambda msg: f"Custom auth server says: {msg}",
        }
    )
    
    print("Custom authentication startup complete")