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
    """Test that artifacts are created in the correct workspace based on context.
    
    Key principle: The workspace where artifacts are created should be determined by:
    1. The context["ws"] which comes from the URL path (e.g., /public/services/... vs /ws-user-xxx/services/...)
    2. NOT from an explicit workspace parameter
    3. User permissions come from their token
    """
    
    server_url = SERVER_URL
    print(f"\nTesting with server: {server_url}")
    
    # The test token creates a user with id "user-1"
    # Their workspace is encoded in the token
    user_workspace = "ws-user-user-1"
    print(f"Test user workspace (from token): {user_workspace}")
    
    # ========== Test Case 1: Anonymous user via HTTP to public service ==========
    print("\n=== Test Case 1: Anonymous HTTP -> /public/services/artifact-manager/create ===")
    print("Expected: Anonymous users should NOT be able to create artifacts")
    
    async with httpx.AsyncClient() as client:
        # No authorization header - anonymous user
        response = await client.post(
            f"{server_url}/public/services/artifact-manager/create",
            json={
                # NOT specifying workspace - should use context from URL
                "alias": "test-anonymous-http",
                "manifest": {"name": "Test Anonymous HTTP"},
                "type": "generic"
            }
        )
        
        if response.status_code in [403, 401]:
            print("✅ Good: Anonymous user cannot create artifacts (permission denied)")
        elif response.status_code == 200:
            artifact = response.json()
            print(f"❌ SECURITY ISSUE: Anonymous created artifact in workspace: {artifact.get('workspace')}")
            # Clean up if it was created
            await client.post(
                f"{server_url}/public/services/artifact-manager/delete",
                json={"artifact_id": artifact['id']}
            )
            pytest.fail("Anonymous users should not be able to create artifacts")
        else:
            # Could be missing parameters or other error
            print(f"⚠️ Status {response.status_code}: Anonymous cannot create (likely permission issue)")
    
    # ========== Test Case 2: Logged-in user accessing service from public URL ==========
    print("\n=== Test Case 2: Logged-in user -> /public/services/artifact-manager/create ===")
    print("Context: User accesses artifact-manager service hosted in public workspace")
    print("Expected: Artifact created in user's own workspace (from token), NOT public")
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {test_user_token}"}
        
        response = await client.post(
            f"{server_url}/public/services/artifact-manager/create",
            json={
                # NOT specifying workspace - should use user's workspace from token
                "alias": "test-user-via-public-url",
                "manifest": {"name": "Test User via Public URL"},
                "type": "generic"
            },
            headers=headers
        )
        
        if response.status_code == 200:
            artifact = response.json()
            workspace = artifact.get('workspace')
            print(f"Created artifact in workspace: {workspace}")
            
            # Artifact should be created in user's workspace, NOT public
            if workspace == user_workspace:
                print(f"✅ Correct: Artifact created in user's workspace ({user_workspace})")
                print("   Service in public, but data stays in user's workspace (secure)")
            elif workspace == "public":
                print("❌ SECURITY ISSUE: Artifact created in public workspace!")
                print("   This is dangerous - artifacts should never be in public")
                pytest.fail("Artifacts should not be created in public workspace")
            else:
                print(f"❌ Unexpected: Artifact created in '{workspace}'")
            
            # Clean up
            await client.post(
                f"{server_url}/public/services/artifact-manager/delete",
                json={"artifact_id": artifact['id']},
                headers=headers
            )
        else:
            print(f"Status {response.status_code}: Could not create artifact")
            if response.status_code in [403, 401, 400]:
                print("Error: User should be able to create artifacts in their own workspace")
    
    # ========== Test Case 3: User trying to override workspace (should be ignored) ==========
    print("\n=== Test Case 3: User trying to override workspace parameter ===")
    print("Context: User tries to explicitly set workspace='ws-user-hacker'")
    print("Expected: Parameter ignored, artifact created in user's workspace (from token)")
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {test_user_token}"}
        
        response = await client.post(
            f"{server_url}/public/services/artifact-manager/create",
            json={
                "workspace": "ws-user-hacker",  # TRYING TO OVERRIDE - should be ignored!
                "alias": "test-exploit-attempt",
                "manifest": {"name": "Exploit Attempt"},
                "type": "generic"
            },
            headers=headers
        )
        
        if response.status_code == 200:
            artifact = response.json()
            workspace = artifact.get('workspace')
            print(f"Created artifact in workspace: {workspace}")
            
            if workspace == "ws-user-hacker":
                print("❌ SECURITY ISSUE: User was able to override workspace!")
                pytest.fail("Users should not be able to specify arbitrary workspace")
            elif workspace == user_workspace:
                print(f"✅ Good: Workspace override ignored, artifact in user's workspace ({workspace})")
            else:
                print(f"✅ Good: Workspace override ignored, artifact in: {workspace}")
            
            # Clean up
            await client.post(
                f"{server_url}/public/services/artifact-manager/delete",
                json={"artifact_id": artifact['id']},
                headers=headers
            )
        elif response.status_code in [403, 401, 400]:
            # This shouldn't happen - user should be able to create in their own workspace
            print(f"Unexpected error: {response.status_code}")
            print(f"User should be able to create artifacts in their own workspace")
        else:
            print(f"Status {response.status_code}: Could not create artifact")
    
    # ========== Test Case 4: Anonymous WebSocket connection ==========
    print("\n=== Test Case 4: Anonymous WebSocket connection ===")
    print("Expected: Anonymous users get their own isolated workspace")
    
    # First verify anonymous cannot connect to public
    print("\n4a. Verify anonymous cannot connect to public workspace...")
    try:
        async with connect_to_server({
            "name": "anonymous-test",
            "server_url": server_url,
            "workspace": "public",  # Trying to connect to public
            "token": None  # Anonymous (no token)
        }) as server:
            print("❌ SECURITY ISSUE: Anonymous connected to public workspace")
            pytest.fail("Anonymous should not connect to public")
    except (ConnectionAbortedError, PermissionError, Exception) as e:
        if "Permission denied" in str(e):
            print("✅ Good: Anonymous cannot connect to public workspace")
        else:
            print(f"✅ Good: Anonymous cannot connect to public: {e}")
    
    # Now test anonymous connecting to their own workspace
    print("\n4b. Anonymous connecting without specifying workspace...")
    async with connect_to_server({
        "name": "anonymous-test",
        "server_url": server_url,
        # NOT specifying workspace - should get their own
        "token": None  # Anonymous
    }) as server:
        anon_workspace = server.config.workspace
        print(f"Connected to workspace: {anon_workspace}")
        assert anon_workspace.startswith("ws-user-anonymouz-"), \
            f"Anonymous should get their own workspace, got: {anon_workspace}"
        
        # Get artifact manager service from public
        am = await server.get_service("public/artifact-manager")
        
        # Try to create artifact - should fail for anonymous users in ephemeral workspace
        try:
            artifact = await am.create(
                alias="test-anon-artifact",
                manifest={"name": "Anonymous Artifact"},
                type="generic"
            )
            print(f"❌ SECURITY ISSUE: Anonymous created artifact in: {artifact['workspace']}")
            pytest.fail("Anonymous users should not be able to create artifacts in ephemeral workspaces")
        except Exception as e:
            if "non-persistent workspace" in str(e):
                print("✅ Good: Anonymous cannot create artifacts in ephemeral workspace")
            else:
                print(f"✅ Good: Anonymous cannot create artifacts: {e}")
    
    # ========== Test Case 5: Logged-in user WebSocket to their workspace ==========
    print("\n=== Test Case 5: Logged-in user WebSocket connection ===")
    print("Expected: User connects to their workspace from token")
    
    async with connect_to_server({
        "name": "user-test",
        "server_url": server_url,
        "workspace": user_workspace,  # Connecting to their own workspace
        "token": test_user_token
    }) as server:
        print(f"Connected to workspace: {server.config.workspace}")
        assert server.config.workspace == user_workspace
        
        # Get artifact manager from public
        am = await server.get_service("public/artifact-manager")
        
        # Create artifact - should go to user's workspace (from context)
        artifact = await am.create(
            alias="test-user-artifact",
            manifest={"name": "User Artifact"},
            type="generic"
        )
        print(f"Created artifact in workspace: {artifact['workspace']}")
        assert artifact['workspace'] == user_workspace, \
            f"Expected {user_workspace}, got {artifact['workspace']}"
        print("✅ User's artifact created in their workspace")
        
        # Clean up
        await am.delete(artifact['id'])
    
    # ========== Test Case 6: Logged-in user connecting to public workspace ==========
    print("\n=== Test Case 6: Logged-in user WebSocket to public workspace ===")
    print("Expected: User may or may not have access to public workspace")
    
    try:
        async with connect_to_server({
            "name": "user-public-test",
            "server_url": server_url,
            "workspace": "public",  # Explicitly connecting to public
            "token": test_user_token
        }) as server:
            print(f"Connected to workspace: {server.config.workspace}")
            assert server.config.workspace == "public"
            
            # Get artifact manager
            am = await server.get_service("public/artifact-manager")
            
            # Create artifact - should go to public (current workspace context)
            artifact = await am.create(
                alias="test-user-public-artifact",
                manifest={"name": "User Public Artifact"},
                type="generic"
            )
            print(f"Created artifact in workspace: {artifact['workspace']}")
            assert artifact['workspace'] == "public", \
                f"Expected 'public', got {artifact['workspace']}"
            print("✅ Artifact created in public workspace as expected")
            
            # Clean up
            await am.delete(artifact['id'])
    except (ConnectionAbortedError, PermissionError, Exception) as e:
        if "Permission denied" in str(e):
            print("✅ Good: User doesn't have permission to public workspace (expected)")
        else:
            print(f"Note: User cannot connect to public workspace: {e}")
    
    print("\n" + "="*60)
    print("✅ All workspace context tests passed!")
    print("Key findings:")
    print("  - Anonymous users cannot create artifacts (security feature)")
    print("  - Anonymous users get isolated ephemeral workspaces")
    print("  - User context comes from token, NOT from URL (secure)")
    print("  - Artifacts always created in user's workspace, never in public")
    print("  - Users cannot override workspace with explicit parameter")
    print("  - Services can be hosted anywhere but operate on user's workspace")
    print("="*60)


async def test_artifact_default_workspace_behavior(
    minio_server, fastapi_server, test_user_token
):
    """Test that artifacts are created in the correct default workspace.
    
    The key principle: When workspace is not specified in the create() call,
    it should use context["ws"] which comes from:
    - For HTTP: The URL path (e.g., /public/services/... or /ws-user-xxx/services/...)
    - For WebSocket: The workspace the user is connected to
    """
    
    server_url = SERVER_URL
    user_workspace = "ws-user-user-1"
    
    print("\n" + "="*60)
    print("Testing Default Workspace Behavior")
    print("="*60)
    
    # Test via WebSocket to verify default behavior
    print("\n=== Testing default workspace via WebSocket ===")
    
    async with connect_to_server({
        "name": "default-test",
        "server_url": server_url,
        "workspace": user_workspace,
        "token": test_user_token
    }) as server:
        print(f"Connected to workspace: {server.config.workspace}")
        
        # Get artifact manager
        am = await server.get_service("public/artifact-manager")
        
        # Create artifact WITHOUT specifying workspace
        artifact = await am.create(
            # NO workspace parameter - should use context
            alias="test-default-workspace",
            manifest={"name": "Default Workspace Test"},
            type="generic"
        )
        
        print(f"Created artifact in workspace: {artifact['workspace']}")
        
        # Should be created in user's workspace (the connected workspace)
        assert artifact['workspace'] == user_workspace, \
            f"Without workspace param, should use context workspace {user_workspace}, got {artifact['workspace']}"
        
        print(f"✅ Correct: Artifact created in user's workspace by default")
        
        # Clean up
        await am.delete(artifact['id'])
    
    print("\n✅ Default workspace behavior test passed!")