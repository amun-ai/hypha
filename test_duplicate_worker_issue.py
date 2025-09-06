import subprocess
import sys
import time
import asyncio
import requests
from redis import Redis

REDIS_PORT = 6379

async def test_rolling_update_cleanup():
    """Test the actual cleanup behavior during rolling update."""
    
    # Get Redis client for verification
    redis_client = Redis(host="localhost", port=REDIS_PORT, decode_responses=True)
    
    # Clear any existing data
    redis_client.flushall()
    print("Cleared Redis data")
    
    # Start first server
    print("\n1. Starting server-1...")
    proc1 = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--port=19527",
            f"--redis-uri=redis://localhost:{REDIS_PORT}",
            "--server-id=test-server-1",
            "--public-base-url=http://127.0.0.1:19527",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for first server to be ready
    timeout = 20
    while timeout > 0:
        try:
            response = requests.get("http://127.0.0.1:19527/health/readiness")
            if response.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    
    if timeout <= 0:
        proc1.terminate()
        proc1.wait()
        raise Exception("Server 1 did not start")
    
    print("Server-1 is ready")
    
    # Check services registered by server 1
    await asyncio.sleep(1)
    all_services = redis_client.keys("services:*")
    print(f"\n2. Services after server-1 started: {len(all_services)} total")
    
    # Check specifically for hypha-login
    login_services = redis_client.keys("services:*:public/*:hypha-login@*")
    print(f"   hypha-login services: {len(login_services)}")
    for svc in login_services[:2]:
        print(f"     - {svc}")
    
    # Start second server
    print("\n3. Starting server-2...")
    proc2 = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--port=19528",
            f"--redis-uri=redis://localhost:{REDIS_PORT}",
            "--server-id=test-server-2",
            "--public-base-url=http://127.0.0.1:19528",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for second server to be ready
    timeout = 20
    while timeout > 0:
        try:
            response = requests.get("http://127.0.0.1:19528/health/readiness")
            if response.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    
    if timeout <= 0:
        proc1.terminate()
        proc2.terminate()
        proc1.wait()
        proc2.wait()
        raise Exception("Server 2 did not start")
    
    print("Server-2 is ready")
    
    # Check services after server 2 starts
    await asyncio.sleep(1)
    all_services = redis_client.keys("services:*")
    print(f"\n4. Services after server-2 started: {len(all_services)} total")
    
    login_services = redis_client.keys("services:*:public/*:hypha-login@*")
    print(f"   hypha-login services: {len(login_services)}")
    for svc in login_services[:3]:
        print(f"     - {svc}")
    
    # Now terminate server 1
    print("\n5. Terminating server-1...")
    proc1.terminate()
    
    # Wait for graceful shutdown
    try:
        proc1.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc1.kill()
        proc1.wait()
    
    print("Server-1 terminated")
    
    # Give time for cleanup
    await asyncio.sleep(3)
    
    # Check what's left
    all_services = redis_client.keys("services:*")
    print(f"\n6. Services after server-1 terminated: {len(all_services)} total")
    
    # Check server-1 specific services
    server1_services = redis_client.keys("services:*:*/test-server-1*")
    print(f"   Server-1 services remaining: {len(server1_services)}")
    for svc in server1_services[:3]:
        print(f"     - {svc}")
    
    # Check hypha-login services
    login_services = redis_client.keys("services:*:public/*:hypha-login@*")
    print(f"   hypha-login services: {len(login_services)}")
    for svc in login_services[:3]:
        print(f"     - {svc}")
    
    # Clean up
    print("\n7. Cleaning up server-2...")
    proc2.terminate()
    try:
        proc2.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc2.kill()
        proc2.wait()
    
    print("\nTest complete!")
    
    # Return results for analysis
    return {
        "login_services_remaining": len(login_services),
        "server1_services_remaining": len(server1_services)
    }

# Run the test
if __name__ == "__main__":
    result = asyncio.run(test_rolling_update_cleanup())
    print(f"\nResults: {result}")
    
    if result["server1_services_remaining"] > 0:
        print("❌ PROBLEM: Server-1 services were not cleaned up properly")
    else:
        print("✓ Server-1 services were cleaned up")
    
    if result["login_services_remaining"] == 0:
        print("❌ PROBLEM: hypha-login services were incorrectly removed")
    else:
        print("✓ hypha-login services from server-2 are still present")
