import subprocess
import sys
import time
import asyncio
import requests
from redis import Redis
import urllib.parse

async def debug_test():
    """Debug the server disconnection test."""
    
    # Use test Redis port
    redis_uri = "redis://127.0.0.1:6338"
    redis_port = 6338
    
    # Start Redis using docker
    print("Starting Redis on port 6338...")
    redis_proc = subprocess.Popen(
        ["docker", "run", "--rm", "-p", f"{redis_port}:6379", "redis:latest"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    
    # Get Redis client
    redis_client = Redis(host="localhost", port=redis_port, decode_responses=True)
    redis_client.flushall()
    
    # Start first server
    print("\nStarting server-1...")
    proc1 = subprocess.Popen(
        [
            sys.executable, "-m", "hypha.server",
            "--port=19527",
            f"--redis-uri={redis_uri}",
            "--server-id=rolling-update-server-1",
            "--public-base-url=http://127.0.0.1:19527",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait and check output
    for _ in range(50):
        try:
            if requests.get("http://127.0.0.1:19527/health/readiness").ok:
                print("Server-1 is ready")
                break
        except:
            pass
        time.sleep(0.1)
    
    await asyncio.sleep(1)
    
    # Check what keys exist
    all_keys = redis_client.keys("*")
    print(f"\nAll Redis keys after server-1 start ({len(all_keys)}):")
    for key in sorted(all_keys)[:20]:
        print(f"  {key}")
    
    # Check services
    services = redis_client.keys("services:*")
    print(f"\nService keys ({len(services)}):")
    for key in sorted(services)[:10]:
        print(f"  {key}")
    
    # Check pattern used in test
    pattern = "services:*:*/rolling-update-server-1:*"
    matching = redis_client.keys(pattern)
    print(f"\nKeys matching '{pattern}' ({len(matching)}):")
    for key in matching[:5]:
        print(f"  {key}")
    
    # Start second server
    print("\nStarting server-2...")
    proc2 = subprocess.Popen(
        [
            sys.executable, "-m", "hypha.server",
            "--port=19528",
            f"--redis-uri={redis_uri}",
            "--server-id=rolling-update-server-2",
            "--public-base-url=http://127.0.0.1:19528",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server-2
    for _ in range(50):
        try:
            if requests.get("http://127.0.0.1:19528/health/readiness").ok:
                print("Server-2 is ready")
                break
        except:
            pass
        time.sleep(0.1)
    
    await asyncio.sleep(1)
    
    # Check services after server-2
    services2 = redis_client.keys("services:*:*/rolling-update-server-2:*")
    print(f"\nServer-2 services ({len(services2)}):")
    for key in services2[:5]:
        print(f"  {key}")
    
    # Get some output from server-2 to debug
    if len(services2) == 0:
        print("\nServer-2 stderr (first 1000 chars):")
        for _ in range(10):
            line = proc2.stdout.readline()
            if line:
                print(f"  {line.strip()}")
    
    # Cleanup
    proc1.terminate()
    proc2.terminate()
    proc1.wait()
    proc2.wait()
    redis_proc.terminate()
    redis_proc.wait()

asyncio.run(debug_test())
