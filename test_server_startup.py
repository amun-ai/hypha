import subprocess
import sys
import time
import requests
from redis import Redis

REDIS_PORT = 6379

# Get Redis client for verification
redis_client = Redis(host="localhost", port=REDIS_PORT, decode_responses=True)

# Clear any existing data
redis_client.flushall()
print("Cleared Redis data")

# Start first server
print("\nStarting server-1...")
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
    stderr=subprocess.STDOUT,
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
    print("Server 1 did not start, checking output:")
    output = proc1.stdout.read(1000).decode()
    print(output)
    proc1.terminate()
    proc1.wait()
    sys.exit(1)

print("Server-1 is ready")

# Check services
time.sleep(1)
server1_services = redis_client.keys("services:*:*/test-server-1:*")
print(f"Server-1 services: {len(server1_services)}")

# Start second server
print("\nStarting server-2...")
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
    stderr=subprocess.STDOUT,
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
    print("Server 2 did not start, checking output:")
    output = proc2.stdout.read(1000).decode()
    print(output)
    proc1.terminate()
    proc2.terminate()
    proc1.wait()
    proc2.wait()
    sys.exit(1)

print("Server-2 is ready")

# Check services
time.sleep(1)
server2_services = redis_client.keys("services:*:*/test-server-2:*")
print(f"Server-2 services: {len(server2_services)}")

# List all services
all_services = redis_client.keys("services:*")
print(f"\nTotal services in Redis: {len(all_services)}")
for svc in all_services[:5]:
    print(f"  - {svc}")

# Clean up
proc1.terminate()
proc2.terminate()
proc1.wait()
proc2.wait()

print("\nTest complete")
