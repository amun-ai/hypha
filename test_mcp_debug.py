import asyncio
import sys
sys.path.insert(0, '.')

from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

async def test():
    WS_SERVER_URL = "ws://127.0.0.1:38283/ws"
    SERVER_URL = "http://127.0.0.1:38283"
    
    # Connect to server
    api = await connect_to_server({
        "client_id": "test-mcp-debug",
        "server_url": WS_SERVER_URL,
    })
    
    workspace = api.config.workspace
    
    docs_content = """
    # Service Documentation
    
    This service processes data according to specific rules.
    """
    
    @schema_function
    def process(x: int) -> int:
        """Process a value."""
        return x * 2
    
    # Register service
    service_info = await api.register_service({
        "id": "docs-service",
        "name": "Service with Documentation",
        "docs": docs_content,
        "description": "A service with documentation field",
        "config": {"visibility": "public"},
        "process": process,
    })
    
    print(f"Registered service: {service_info['id']}")
    
    # Try to access it via MCP
    service_id = service_info['id'].split('/')[-1]
    base_url = f"{SERVER_URL}/{workspace}/mcp/{service_id}/mcp"
    print(f"MCP URL: {base_url}")
    
    # Test with curl first
    import subprocess
    result = subprocess.run([
        "curl", "-X", "POST",
        base_url,
        "-H", "Content-Type: application/json",
        "-d", '{"jsonrpc": "2.0", "method": "resources/list", "id": 1}'
    ], capture_output=True, text=True)
    
    print(f"List resources response: {result.stdout}")
    
    # Now try read_resource
    result = subprocess.run([
        "curl", "-X", "POST",
        base_url,
        "-H", "Content-Type: application/json",
        "-d", '{"jsonrpc": "2.0", "method": "resources/read", "params": {"uri": "resource://docs"}, "id": 2}'
    ], capture_output=True, text=True)
    
    print(f"Read resource response: {result.stdout}")
    
    await api.disconnect()

asyncio.run(test())
