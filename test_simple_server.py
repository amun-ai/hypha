import subprocess
import sys
import time
import requests

# Just test that a single server starts and registers services
print("Starting server...")
proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "hypha.server",
        "--port=19527",
        "--server-id=test-server",
        "--public-base-url=http://127.0.0.1:19527",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Wait for server to be ready
timeout = 20
while timeout > 0:
    try:
        response = requests.get("http://127.0.0.1:19527/health/readiness")
        if response.ok:
            print("Server is ready")
            break
    except:
        pass
    timeout -= 0.1
    time.sleep(0.1)

if timeout <= 0:
    print("Server did not start in time")
    output = proc.stdout.read(2000).decode()
    print("Server output:", output)
else:
    print("Server started successfully")
    time.sleep(1)
    
proc.terminate()
proc.wait()
