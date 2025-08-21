import asyncio
import sys
sys.path.insert(0, '.')

from hypha_rpc import connect_to_server

async def test():
    WS_SERVER_URL = "ws://127.0.0.1:38283/ws"
    
    # Use a simple connection without token to see if it's an auth issue
    api = await connect_to_server({
        "client_id": "test-tilde-debug",
        "server_url": WS_SERVER_URL,
    })
    
    # Try to list services in the * workspace
    print("Getting workspace manager...")
    try:
        ws_manager = await api.get_service("*:default")
        print(f"Got workspace manager: {ws_manager}")
        
        # List services to see what's available
        services = await ws_manager.list_services("*")
        print(f"Services in * workspace: {services}")
    except Exception as e:
        print(f"Error: {e}")
        
        # Try a different approach - list all services
        try:
            print("Trying built-in service...")
            built_in = await api.get_service("*:built-in")
            services = await built_in.list_services("*")
            print(f"Services via built-in: {services}")
        except Exception as e2:
            print(f"Also failed: {e2}")
    
    await api.disconnect()

asyncio.run(test())
