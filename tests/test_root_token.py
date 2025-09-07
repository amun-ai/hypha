#!/usr/bin/env python
"""Comprehensive test script for root token functionality."""

import asyncio
import sys
import os
import json
import httpx
import subprocess
import time
import secrets
from pathlib import Path
from hypha_rpc import connect_to_server

async def test_root_token_with_server():
    """Test root token with running server."""
    print("\nTesting root token with server...")

    # Check if server is running
    base_url = "http://127.0.0.1:9527"
    root_token = "Klo2SYaX7DmL3M7O7pcM4Scp4CIqLrXzCzOzew_Qm8s"

    try:
        async with httpx.AsyncClient() as client:
            # Test without token
            response = await client.get(f"{base_url}/assets/config.json")
            print(f"✓ Server is running, config response: {response.status_code}")

            # Test with root token accessing root workspace info
            headers = {"Authorization": f"Bearer {root_token}"}
            response = await client.get(
                f"{base_url}/ws-user-root/info",
                headers=headers
            )

            if response.status_code == 200:
                print(f"✓ Root workspace access successful!")
                data = response.json()
                print(f"  Workspace: {data.get('name')}")
                print(f"  ID: {data.get('id')}")
                print(f"  Owners: {data.get('owners')}")
                return True
            elif response.status_code == 403:
                print(f"✗ Access denied to root workspace")
                return False
            else:
                print(f"✗ Unexpected response: {response.status_code} - {response.text}")
                return False

    except httpx.ConnectError:
        print("Server is not running.")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


async def test_root_services():
    """Test accessing services with root token."""
    print("\nTesting root services access...")

    base_url = "http://127.0.0.1:9527"
    root_token = "Klo2SYaX7DmL3M7O7pcM4Scp4CIqLrXzCzOzew_Qm8s"

    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {root_token}"}

            # Test accessing public workspace services
            response = await client.get(
                f"{base_url}/public/services/",
                headers=headers
            )

            if response.status_code == 200:
                services = response.json()
                print(f"✓ Public workspace services access successful!")
                print(f"  Found {len(services)} services")
                if services:
                    print(f"  First service: {services[0].get('id', 'N/A')}")
                return True
            else:
                print(f"✗ Failed to access services: {response.status_code}")
                return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False


async def test_websocket_connection():
    """Test WebSocket connection with root token."""
    print("\nTesting WebSocket connection with root token...")

    root_token = "Klo2SYaX7DmL3M7O7pcM4Scp4CIqLrXzCzOzew_Qm8s"
    ws_url = "ws://127.0.0.1:9527/ws"

    try:
        # Connect using hypha-rpc with root token
        api = await connect_to_server({
            "server_url": ws_url,
            "token": root_token,
            "client_id": "root-test-client"
        })

        # Check user info
        user_info = api.config.get("user", {})
        print(f"✓ Connected via WebSocket!")
        print(f"  User ID: {user_info.get('id')}")
        print(f"  Roles: {user_info.get('roles')}")
        print(f"  Workspace: {api.config.get('workspace')}")

        # Test admin capabilities
        workspaces = await api.list_workspaces()
        print(f"✓ Listed {len(workspaces)} workspaces")

        # Create a test workspace
        test_ws = await api.create_workspace({
            "name": "test-ws-from-root",
            "description": "Test workspace created by root",
            "owners": ["root"]
        })
        print(f"✓ Created workspace: {test_ws['name']}")

        # Delete the test workspace
        await api.delete_workspace(test_ws["name"])
        print(f"✓ Deleted workspace: {test_ws['name']}")

        await api.disconnect()
        return True

    except Exception as e:
        print(f"✗ WebSocket connection failed: {e}")
        return False


async def test_weak_token_rejection():
    """Test that weak tokens are rejected."""
    print("\nTesting weak token rejection...")

    # Start server with weak token (should fail)
    weak_token = "test-root-token"

    print(f"Attempting to start server with weak token: '{weak_token}'")

    proc = subprocess.Popen(
        [sys.executable, "-m", "hypha.server", f"--root-token={weak_token}", "--port=9528"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Give it a moment to start
    time.sleep(2)

    # Check if process is still running
    poll = proc.poll()

    if poll is not None:
        # Process exited (as expected for weak token)
        stdout, stderr = proc.communicate()
        output = stderr.decode() + stdout.decode()

        if "Root token must be at least 32 characters" in output or "Root token appears to be too simple" in output:
            print("✓ Weak token was properly rejected!")
            return True
        else:
            print(f"✗ Process exited but with unexpected error: {output[:500]}")
            return False
    else:
        # Process is still running (shouldn't happen with weak token)
        proc.terminate()
        proc.wait()
        print("✗ Server started with weak token (should have been rejected)")
        return False


async def test_strong_token_acceptance():
    """Test that strong tokens are accepted."""
    print("\nTesting strong token acceptance...")

    # Generate a strong token
    
    strong_token = secrets.token_urlsafe(32)

    print(f"Testing with strong token: {strong_token[:10]}...")

    proc = subprocess.Popen(
        [sys.executable, "-m", "hypha.server", f"--root-token={strong_token}", "--port=9529"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to start
    time.sleep(3)

    # Check if process is still running
    poll = proc.poll()

    if poll is None:
        # Process is running (expected for strong token)
        print("✓ Server started successfully with strong token!")

        # Test connection
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:9529/health/liveness")
                if response.status_code == 200:
                    print("✓ Server is healthy and responding")
                    result = True
                else:
                    print(f"✗ Server health check failed: {response.status_code}")
                    result = False
        except Exception as e:
            print(f"✗ Failed to connect to server: {e}")
            result = False

        # Clean up
        proc.terminate()
        proc.wait()
        return result
    else:
        # Process exited (shouldn't happen with strong token)
        stdout, stderr = proc.communicate()
        print(f"✗ Server failed to start with strong token: {stderr.decode()[:500]}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Comprehensive Root Token Functionality Tests")
    print("=" * 60)

    # Test with existing server (if running)
    server_test = await test_root_token_with_server()
    services_test = await test_root_services()
    websocket_test = False

    if server_test:
        # Only test WebSocket if server is running with correct token
        websocket_test = await test_websocket_connection()

    # Test token validation
    weak_rejection = await test_weak_token_rejection()
    strong_acceptance = await test_strong_token_acceptance()

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Server HTTP:        {'✓ PASS' if server_test else '✗ FAIL (server may not be running)'}")
    print(f"  Services Access:    {'✓ PASS' if services_test else '✗ FAIL (server may not be running)'}")
    print(f"  WebSocket:          {'✓ PASS' if websocket_test else '✗ SKIP (requires running server)'}")
    print(f"  Weak Token Reject:  {'✓ PASS' if weak_rejection else '✗ FAIL'}")
    print(f"  Strong Token Accept: {'✓ PASS' if strong_acceptance else '✗ FAIL'}")
    print("=" * 60)

    critical_tests = weak_rejection and strong_acceptance

    if critical_tests:
        print("\n✅ Critical tests passed! Root token validation is working correctly.")
        if not server_test:
            print("\nℹ️  To test with a running server, start it with:")
            print("    python -m hypha.server --root-token 'YOUR_STRONG_TOKEN_HERE'")
    else:
        print("\n❌ Some critical tests failed. Please check the implementation.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)