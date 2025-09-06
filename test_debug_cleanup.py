import subprocess
import sys
import time
import asyncio
import requests
from redis import Redis
import urllib.parse

# Hardcode for local test
redis_server = "redis://127.0.0.1:6379"

async def test_server_disconnection_cleanup():
    """Test that dead servers are properly cleaned up during rolling updates."""
    
    # Parse the redis URL to get the port
    parsed_url = urllib.parse.urlparse(redis_server)
    redis_port = parsed_url.port or 6379
    
    # Start Redis server locally
    print("Starting Redis server...")
    redis_proc = subprocess.Popen(
        ["docker", "run", "--rm", "-p", f"{redis_port}:6379", "redis:latest"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    
    # Get Redis client for verification
    redis_client = Redis(host="localhost", port=redis_port, decode_responses=True)
    
    # Clear any existing data
    redis_client.flushall()
    
    # Start first server
    print("Starting server-1...")
    proc1 = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--port=19527",
            f"--redis-uri={redis_server}",
            "--server-id=rolling-update-server-1",
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
    
    assert timeout > 0, "First server did not start in time"
    print("Server-1 is ready")
    
    # Verify server 1 registered services
    await asyncio.sleep(1)  # Give time for service registration
    server1_services = redis_client.keys("services:*:*/rolling-update-server-1:*")
    print(f"Server-1 services: {len(server1_services)}")
    for s in server1_services[:3]:
        print(f"  - {s}")
    
    # Start second server (simulating rolling update)
    print("\nStarting server-2...")
    proc2 = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--port=19528",
            f"--redis-uri={redis_server}",
            "--server-id=rolling-update-server-2",
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
    
    assert timeout > 0, "Second server did not start in time"
    print("Server-2 is ready")
    
    # Verify server 2 registered services
    await asyncio.sleep(1)
    server2_services = redis_client.keys("services:*:*/rolling-update-server-2:*")
    print(f"Server-2 services: {len(server2_services)}")
    for s in server2_services[:3]:
        print(f"  - {s}")
    
    if len(server2_services) == 0:
        print("\nERROR: Server-2 didn't register services")
        print("Checking server-2 stderr output:")
        stderr = proc2.stderr.read(2000).decode()
        print(stderr)
    
    # Cleanup
    proc1.terminate()
    proc2.terminate()
    proc1.wait()
    proc2.wait()
    redis_proc.terminate()
    redis_proc.wait()

asyncio.run(test_server_disconnection_cleanup())
