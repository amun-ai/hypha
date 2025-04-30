import subprocess
import re
import asyncio
import httpx
import time
from typing import Optional, Tuple
from urllib.parse import urljoin

async def start_and_verify_service() -> Tuple[subprocess.Popen, str]:
    """Start a single Deno service and verify it's working"""
    print("Starting Deno process...")
    
    # Start the Deno process
    process = subprocess.Popen(
        ["deno", "run", "--allow-net", "--allow-read", "--allow-env", "scripts/deno-demo-asgi-app.js"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # This makes stdout and stderr return strings instead of bytes
    )
    
    # Regular expression to match the service URL
    url_pattern = re.compile(r"Access your app at: (https://[^\s]+)")
    
    # Wait for the service to start and get the URL
    service_url = None
    start_time = time.time()
    while service_url is None and time.time() - start_time < 30:  # 30 seconds timeout
        if process.poll() is not None:  # Process ended
            raise Exception("Process ended unexpectedly")
            
        line = process.stdout.readline()
        if line:
            print("Server output:", line.strip())
            match = url_pattern.search(line)
            if match:
                service_url = match.group(1)
                # Ensure URL ends with trailing slash
                if not service_url.endswith('/'):
                    service_url += '/'
                break
    
    if not service_url:
        process.terminate()
        raise Exception("Failed to get service URL within timeout")

    print(f"Service URL: {service_url}")
    
    # Test the service
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Try to connect to the service (with retries)
        for _ in range(5):
            try:
                # Test the root endpoint
                response = await client.get(service_url)
                response.raise_for_status()
                print("Root endpoint test successful!")
                
                # Test the API endpoint
                api_url = urljoin(service_url, 'api/v1/test')
                api_response = await client.get(api_url)
                api_response.raise_for_status()
                api_data = api_response.json()
                assert api_data["message"] == "Hello, it works!"
                print("API endpoint test successful!")
                
                return process, service_url
                
            except Exception as e:
                print(f"Connection attempt failed: {e}")
                await asyncio.sleep(2)
        
        # If we get here, all retries failed
        process.terminate()
        raise Exception("Failed to verify service is working after multiple attempts")

async def main():
    try:
        process, url = await start_and_verify_service()
        print("\nService is up and running!")
        print(f"URL: {url}")
        
        # Keep running for a bit to verify stability
        await asyncio.sleep(10)
        
        # Cleanup
        process.terminate()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 