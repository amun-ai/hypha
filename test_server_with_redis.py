import subprocess
import sys
import time
import asyncio
import requests
from redis import Redis

async def test():
    # Start redis 
    print("Starting Redis...")
    redis_proc = subprocess.Popen(
        ["docker", "run", "--rm", "-p", "6379:6379", "redis:latest"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    
    redis_client = Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.flushall()
    
    # Start a server with output
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
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Read some output
    for i in range(20):
        line = proc.stdout.readline()
        if line:
            print(f"Server: {line.strip()}")
        if "Ready to accept" in line or "Server ready" in line:
            break
        time.sleep(0.1)
    
    # Check if server is responsive
    try:
        response = requests.get("http://127.0.0.1:19527/health/readiness")
        print(f"Server health check: {response.status_code}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    await asyncio.sleep(1)
    
    # Check service keys
    all_keys = redis_client.keys("*")
    print(f"\nAll Redis keys ({len(all_keys)}):")
    for key in sorted(all_keys)[:10]:
        print(f"  {key}")
    
    # Cleanup
    proc.terminate()
    proc.wait()
    redis_proc.terminate()
    redis_proc.wait()

asyncio.run(test())
