"""Unit test to validate MCP resource fixes."""
import asyncio
from unittest.mock import MagicMock, AsyncMock
import mcp.types as types
from hypha.mcp import HyphaMCPAdapter

async def test_docs_and_string_resources():
    """Test that docs field and nested strings are exposed as MCP resources."""
    
    # Mock service with docs and nested string fields
    service = {
        "id": "test-service",
        "name": "Test Service",
        "type": "generic",
        "docs": "# Documentation\nThis is the service documentation.",
        "metadata": {
            "version": "1.0.0",
            "author": "Test Author",
            "nested": {
                "info": "Some nested information"
            }
        },
        "settings": {
            "theme": "dark",
            "language": "en"
        },
        "process": lambda x: x * 2  # This should be skipped
    }
    
    # Mock service info
    service_info = MagicMock()
    service_info.id = "test-service"
    service_info.name = "Test Service"
    service_info.type = "generic"
    
    # Mock redis client
    redis_client = MagicMock()
    
    # Create adapter
    adapter = HyphaMCPAdapter(service, service_info, redis_client)
    
    # Check that the adapter has registered resources
    assert hasattr(adapter, 'string_resources'), "Adapter should have string_resources"
    assert 'docs' in adapter.string_resources, "docs should be in string_resources"
    
    # Get the list_resources handler
    list_resources_handler = None
    read_resource_handler = None
    
    # Access the handlers through request_handlers
    for handler in adapter.server.request_handlers:
        if handler.method == "resources/list":
            list_resources_handler = handler.handler
        elif handler.method == "resources/read":
            read_resource_handler = handler.handler
    
    assert list_resources_handler is not None, "list_resources handler should be registered"
    assert read_resource_handler is not None, "read_resource handler should be registered"
    
    # Test list_resources
    resources = await list_resources_handler()
    assert isinstance(resources, list), "Resources should be a list"
    
    # Check for docs resource
    docs_resource = None
    for resource in resources:
        if str(resource.uri) == "resource://docs":
            docs_resource = resource
            break
    
    assert docs_resource is not None, "docs resource should be present"
    assert "Documentation for Test Service" in docs_resource.name
    assert "documentation content" in docs_resource.description.lower()
    
    # Check for nested string resources
    expected_uris = {
        "resource://metadata/version",
        "resource://metadata/author", 
        "resource://metadata/nested/info",
        "resource://settings/theme",
        "resource://settings/language"
    }
    
    found_uris = {str(r.uri) for r in resources}
    for expected_uri in expected_uris:
        assert expected_uri in found_uris, f"Expected resource {expected_uri} not found"
    
    # Test read_resource for docs
    from mcp.shared.urls import AnyUrl
    docs_result = await read_resource_handler(AnyUrl("resource://docs"))
    assert docs_result.contents[0].text == service["docs"]
    
    # Test read_resource for nested field
    nested_result = await read_resource_handler(AnyUrl("resource://metadata/version"))
    assert nested_result.contents[0].text == "1.0.0"
    
    print("✅ All tests passed!")
    print(f"  - Found {len(resources)} resources")
    print(f"  - docs resource: ✓")
    print(f"  - nested string resources: ✓")
    print(f"  - read_resource works: ✓")

if __name__ == "__main__":
    asyncio.run(test_docs_and_string_resources())
