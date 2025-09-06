import subprocess
import sys
import time
import asyncio
import requests
from redis import Redis
import urllib.parse

async def test():
    # Start redis using Docker
    print("Starting Redis...")
    redis_proc = subprocess.Popen(
        ["docker", "run", "--rm", "-p", "6379:6379", "redis:latest"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    
    redis_client = Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.flushall()
    
    # Start a server
    print("Starting server...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "hypha.server",
            "--port=19527",
            "--redis-uri=redis://localhost:6379",
            "--server-id=test-server-1",
            "--public-base-url=http://127.0.0.1:19527",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server
    timeout = 20
    while timeout > 0:
        try:
            if requests.get("http://127.0.0.1:19527/health/readiness").ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    
    if timeout <= 0:
        print("Server didn't start")
        proc.terminate()
        redis_proc.terminate()
        return
    
    print("Server ready")
    await asyncio.sleep(1)
    
    # Check ALL service keys
    all_services = redis_client.keys("services:*")
    print(f"\nAll service keys ({len(all_services)}):")
    for key in sorted(all_services):
        print(f"  {key}")
    
    # Check specifically for server-1 pattern
    pattern = "services:*:*/test-server-1:*"
    matching = redis_client.keys(pattern)
    print(f"\nKeys matching '{pattern}' ({len(matching)}):")
    for key in sorted(matching):
        print(f"  {key}")
    
    # Cleanup
    proc.terminate()
    proc.wait()
    redis_proc.terminate()
    redis_proc.wait()

asyncio.run(test())
