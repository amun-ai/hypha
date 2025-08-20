#!/usr/bin/env python
"""Test script for GitHub CI MCP server."""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test the GitHub CI MCP server."""
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=["scripts/github_ci_mcp.py", "stdio"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test check_ci_status
            print("\nTesting check_ci_status for main branch...")
            result = await session.call_tool(
                "check_ci_status",
                arguments={"branch": "main"}
            )
            
            if result.content:
                import json
                data = json.loads(result.content[0].text)
                print(f"  Status: {data.get('status')}")
                print(f"  Conclusion: {data.get('conclusion')}")
                if 'html_url' in data:
                    print(f"  URL: {data.get('html_url')}")
            
            print("\nMCP server test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())