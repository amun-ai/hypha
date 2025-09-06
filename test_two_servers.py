import subprocess
import sys
import time
import asyncio
import requests
from redis import Redis
import urllib.parse

async def test():
    redis_uri = "redis://localhost:6379"
    
    # Start Redis if not running
    try:
        redis_client = Redis(host="localhost", port=6379, decode_responses=True)
        redis_client.ping()
    except:
        print("Starting Redis server...")
        redis_proc = subprocess.Popen(
            ["redis-server", "--port", "6379"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1)
    
    redis_client = Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.flushall()
    
    # Start server 1
    print("Starting server 1...")
    proc1 = subprocess.Popen(
        [sys.executable, "-m", "hypha.server",
         "--port=19527",
         f"--redis-uri={redis_uri}",
         "--server-id=test-1",
         "--public-base-url=http://127.0.0.1:19527"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Wait for server 1
    for i in range(50):
        try:
            if requests.get("http://127.0.0.1:19527/health/readiness").ok:
                break
        except:
            pass
        time.sleep(0.1)
    
    print("Server 1 ready")
    await asyncio.sleep(1)
    
    # Check services
    services1 = redis_client.keys("services:*:*/test-1:*")
    print(f"Server 1 services: {len(services1)}")
    
    # Start server 2
    print("\nStarting server 2...")
    proc2 = subprocess.Popen(
        [sys.executable, "-m", "hypha.server",
         "--port=19528",
         f"--redis-uri={redis_uri}",
         "--server-id=test-2",
         "--public-base-url=http://127.0.0.1:19528"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Wait for server 2
    for i in range(50):
        try:
            if requests.get("http://127.0.0.1:19528/health/readiness").ok:
                break
        except:
            pass
        time.sleep(0.1)
    
    print("Server 2 ready")
    await asyncio.sleep(1)
    
    # Check services
    services2 = redis_client.keys("services:*:*/test-2:*")
    print(f"Server 2 services: {len(services2)}")
    
    # Check if server 2 mistakenly cleaned up server 1
    services1_after = redis_client.keys("services:*:*/test-1:*")
    print(f"Server 1 services after server 2 started: {len(services1_after)}")
    
    # Check server 2 output for errors
    if len(services2) == 0:
        print("\nServer 2 output (first 1000 chars):")
        output = proc2.stdout.read(1000).decode()
        print(output)
    
    # Cleanup
    proc1.terminate()
    proc2.terminate()
    proc1.wait()
    proc2.wait()

asyncio.run(test())
