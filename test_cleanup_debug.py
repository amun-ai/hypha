import asyncio
from redis import Redis

async def test():
    redis_client = Redis(host="localhost", port=6338, decode_responses=True)
    
    # Check what's in Redis after test failure
    all_keys = redis_client.keys("*")
    print(f"All keys in Redis ({len(all_keys)}):")
    for key in sorted(all_keys)[:30]:
        print(f"  {key}")
    
    # Check services
    services = redis_client.keys("services:*")
    print(f"\nService keys ({len(services)}):")
    for key in sorted(services):
        print(f"  {key}")
    
    # Check what would match the cleanup pattern for server-1
    pattern1 = "services:*|*:*/rolling-update-server-1:*@*"
    pattern2 = "services:*|*:*/rolling-update-server-1-*:*@*"
    
    matches1 = redis_client.keys(pattern1)
    matches2 = redis_client.keys(pattern2)
    
    print(f"\nWould be cleaned by pattern '{pattern1}': {len(matches1)}")
    for key in matches1:
        print(f"  {key}")
    
    print(f"\nWould be cleaned by pattern '{pattern2}': {len(matches2)}")
    for key in matches2:
        print(f"  {key}")

asyncio.run(test())
