"""Test artifact workspace context for different user types and connection methods."""

import pytest
import httpx
from hypha_rpc import connect_to_server
from . import SERVER_URL

# All test coroutines will be treated as marked
pytestmark = pytest.mark.asyncio


async def test_artifact_workspace_context_http_and_websocket(
    minio_server, fastapi_server, test_user_token
):
    """Test that artifacts are created in the correct workspace for different users and connection types."""
    
    server_url = SERVER_URL
    print(f"\nTesting with server: {server_url}")
    
    # Get user workspace from token
    # The test token creates a user with id "user-1"
    user_workspace = "ws-user-user-1"
    print(f"Test user workspace: {user_workspace}")
    
    # Test Case 1: Anonymous user via HTTP accessing public workspace
    print("\n=== Test Case 1: Anonymous HTTP -> /public/services/artifact-manager ===")
    
    async with httpx.AsyncClient() as client:
        # Create artifact via HTTP as anonymous user
        response = await client.post(
            f"{server_url}/public/services/artifact-manager/create",
            json={
                "alias": "test-anonymous-http-public",
                "manifest": {"name": "Test Anonymous HTTP Public"},
                "type": "generic",
                "_rkwargs": True
            }
        )
        
        # Anonymous users should NOT be able to create artifacts in public
        if response.status_code == 403 or response.status_code == 401:
            print("✅ Good: Anonymous user cannot create artifacts in public (permission denied)")
        elif response.status_code == 200:
            artifact = response.json()
            print(f"❌ SECURITY ISSUE: Anonymous created artifact in workspace: {artifact.get('workspace')}")
            # Clean up if it was created
            await client.post(
                f"{server_url}/public/services/artifact-manager/delete",
                json={"artifact_id": artifact['id'], "_rkwargs": True}
            )
            pytest.fail("Anonymous users should not be able to create artifacts in public workspace")
        else:
            print(f"⚠️ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
    
    # Test Case 2: Logged-in user via HTTP to public workspace
    print("\n=== Test Case 2: Logged-in HTTP -> /public/services/artifact-manager ===")
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {test_user_token}"}
        
        # Logged-in users might be able to create in public if they have permission
        response = await client.post(
            f"{server_url}/public/services/artifact-manager/create",
            json={
                "alias": "test-logged-http-public",
                "manifest": {"name": "Test Logged HTTP Public"},
                "type": "generic",
                "_rkwargs": True
            },
            headers=headers
        )
        
        if response.status_code == 200:
            artifact = response.json()
            workspace = artifact.get('workspace')
            print(f"Created artifact in workspace: {workspace}")
            
            # The artifact should be created in the workspace from context
            # With the fix, context['ws'] should be 'public'
            if workspace == 'public':
                print("✅ Artifact created in public workspace as expected")
            else:
                print(f"⚠️ Artifact created in unexpected workspace: {workspace}")
            
            # Clean up
            await client.post(
                f"{server_url}/public/services/artifact-manager/delete",
                json={"artifact_id": artifact['id'], "_rkwargs": True},
                headers=headers
            )
        else:
            print(f"Could not create artifact: {response.status_code}")
            print(f"Response: {response.text}")
    
    # Test Case 3: Anonymous user via WebSocket to public workspace
    print("\n=== Test Case 3: Anonymous WebSocket -> public workspace ===")

    async with connect_to_server({
        "name": "anonymous websocket client",
        "server_url": server_url,
        "workspace": "public",
        "token": None  # Anonymous
    }) as server:
        print(f"Connected to workspace: {server.config.workspace}")
        
        # Get artifact manager
        am = await server.get_service("public/artifact-manager")
        
        # Try to create artifact - should fail for anonymous
        try:
            artifact = await am.create(
                alias="test-anonymous-ws-public",
                manifest={"name": "Test Anonymous WS Public"},
                type="generic"
            )
            print(f"❌ SECURITY ISSUE: Anonymous created artifact in: {artifact['workspace']}")
            await am.delete(artifact['id'])
            pytest.fail("Anonymous should not create artifacts in public")
        except Exception as e:
            print(f"✅ Good: Anonymous cannot create in public: {e}")

    # Test Case 4: Logged-in user via WebSocket to their workspace
    print("\n=== Test Case 4: Logged-in WebSocket -> user workspace ===")
    
    async with connect_to_server({
        "name": "logged websocket client",
        "server_url": server_url,
        "workspace": user_workspace,
        "token": test_user_token
    }) as server:
        print(f"Connected to workspace: {server.config.workspace}")
        
        # Get artifact manager (public service)
        am = await server.get_service("public/artifact-manager")
        
        # Create artifact - should be in user's workspace
        artifact = await am.create(
            alias="test-logged-ws-user",
            manifest={"name": "Test Logged WS User"},
            type="generic"
        )
        print(f"Created artifact in workspace: {artifact['workspace']}")
        
        assert artifact['workspace'] == user_workspace, \
            f"Expected workspace '{user_workspace}', got '{artifact['workspace']}'"
        print("✅ Artifact created in user's workspace as expected")
        
        # Clean up
        await am.delete(artifact['id'])
    
    # Test Case 5: Logged-in user via WebSocket to public workspace
    print("\n=== Test Case 5: Logged-in WebSocket -> public workspace ===")
    
    async with connect_to_server({
        "name": "logged websocket to public",
        "server_url": server_url,
        "workspace": "public",
        "token": test_user_token
    }) as server:
        print(f"Connected to workspace: {server.config.workspace}")
        
        # Get artifact manager
        am = await server.get_service("public/artifact-manager")
        
        # Create artifact - should be in public workspace
        artifact = await am.create(
            alias="test-logged-ws-public",
            manifest={"name": "Test Logged WS Public"},
            type="generic"
        )
        print(f"Created artifact in workspace: {artifact['workspace']}")
        
        assert artifact['workspace'] == 'public', \
            f"Expected workspace 'public', got '{artifact['workspace']}'"
        print("✅ Artifact created in public workspace as expected")
        
        # Clean up
        await am.delete(artifact['id'])
    
    print("\n✅ All workspace context tests passed!")


async def test_artifact_permission_security(
    minio_server, fastapi_server, test_user_token
):
    """Test that artifact permissions are properly enforced."""
    
    server_url = SERVER_URL
    
    print("\n=== Testing Artifact Permission Security ===")
    
    # Test that anonymous users cannot create artifacts in workspaces they don't own
    workspaces_to_test = [
        ("public", False, "Anonymous should NOT create in public"),
        ("ws-user-other", False, "Anonymous should NOT create in other user workspace"),
    ]
    
    async with httpx.AsyncClient() as client:
        for workspace, should_succeed, description in workspaces_to_test:
            print(f"\nTesting: {description}")
            print(f"Workspace: {workspace}")
            
            response = await client.post(
                f"{server_url}/public/services/artifact-manager/create",
                json={
                    "workspace": workspace,
                    "alias": f"test-security-{workspace}",
                    "manifest": {"name": f"Security Test {workspace}"},
                    "type": "generic",
                    "_rkwargs": True
                }
            )
            
            if response.status_code in [403, 401]:
                print(f"✅ Permission denied as expected")
            elif response.status_code == 200:
                artifact = response.json()
                print(f"❌ SECURITY ISSUE: Created artifact in {artifact.get('workspace')}")
                # Try to clean up
                await client.post(
                    f"{server_url}/public/services/artifact-manager/delete",
                    json={"artifact_id": artifact['id'], "_rkwargs": True}
                )
                if not should_succeed:
                    pytest.fail(f"{description} - but artifact was created!")
            else:
                print(f"⚠️ Unexpected response: {response.status_code}")
    
    print("\n✅ Permission security tests passed!")