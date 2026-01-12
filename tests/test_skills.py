"""Test the agent skills module."""

import pytest
import requests

from . import SERVER_URL


@pytest.mark.asyncio
async def test_skills_endpoint_index(fastapi_server):
    """Test the agent skills index endpoint."""
    # Test the skills endpoint for a workspace
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/")
    assert response.status_code == 200
    data = response.json()

    # Verify the response structure
    assert "name" in data
    assert data["name"] == "hypha"
    assert "version" in data
    assert "description" in data
    assert "files" in data

    # Verify expected files are listed
    expected_files = ["SKILL.md", "REFERENCE.md", "EXAMPLES.md", "WORKSPACE_CONTEXT.md"]
    for f in expected_files:
        assert f in data["files"], f"Expected file {f} not in files list"


@pytest.mark.asyncio
async def test_skills_skill_md(fastapi_server):
    """Test the SKILL.md endpoint."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/markdown")

    content = response.text

    # Verify YAML frontmatter
    assert content.startswith("---")
    assert "name: hypha" in content
    assert "description:" in content

    # Verify key sections exist
    assert "# Hypha Workspace Manager" in content
    assert "## Workspace Context" in content
    assert "## Quick Start" in content
    assert "## Core Capabilities" in content
    assert "## Instructions for AI Agents" in content


@pytest.mark.asyncio
async def test_skills_reference_md(fastapi_server):
    """Test the REFERENCE.md endpoint with dynamic API documentation."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/REFERENCE.md")
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/markdown")

    content = response.text

    # Verify the reference contains dynamic API documentation
    assert "# Hypha API Reference" in content
    assert "## Connection Methods" in content

    # Verify core services are documented
    assert "## Workspace Manager" in content
    assert "## Artifact Manager" in content
    assert "## Server Apps" in content

    # Verify at least some methods are documented (from __schema__)
    # These should be dynamically extracted from the actual service classes
    assert "### list_services" in content or "list_services" in content
    assert "### create" in content or "create" in content

    # Verify HTTP endpoints reference
    assert "## HTTP Endpoints Reference" in content


@pytest.mark.asyncio
async def test_skills_examples_md(fastapi_server):
    """Test the EXAMPLES.md endpoint."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/EXAMPLES.md")
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/markdown")

    content = response.text

    # Verify key sections
    assert "# Hypha Code Examples" in content
    assert "## Connection Examples" in content
    assert "## Service Examples" in content
    assert "## Artifact Examples" in content
    assert "## Server App Examples" in content

    # Verify code examples are present
    assert "```python" in content
    assert "connect_to_server" in content
    assert "register_service" in content

    # Verify Server Apps examples include comprehensive coverage
    assert "Deploy a FastAPI/ASGI Application" in content or "ASGI" in content
    assert "Serverless HTTP Functions" in content or "functions" in content
    assert "Manage App Lifecycle" in content
    assert "controller.install" in content
    assert "controller.start" in content
    assert "controller.list_apps" in content or "list_running" in content


@pytest.mark.asyncio
async def test_skills_workspace_context_md(fastapi_server):
    """Test the WORKSPACE_CONTEXT.md endpoint."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/WORKSPACE_CONTEXT.md")
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/markdown")

    content = response.text

    # Verify workspace-specific content
    assert "# Workspace Context" in content
    assert "## Workspace Information" in content
    assert "## Capabilities" in content
    assert "## Quick Access URLs" in content

    # The workspace should be "public" since we're accessing from public path
    assert "public" in content


@pytest.mark.asyncio
async def test_skills_404_for_unknown_file(fastapi_server):
    """Test that unknown files return 404."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/UNKNOWN.md")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_skills_cors_headers(fastapi_server):
    """Test that CORS headers are properly set."""
    response = requests.get(
        f"{SERVER_URL}/public/apps/agent-skills/SKILL.md",
        headers={"Origin": "http://example.com"}
    )
    assert response.status_code == 200

    # Check CORS headers are present
    assert "access-control-allow-origin" in response.headers or \
           "Access-Control-Allow-Origin" in response.headers


@pytest.mark.asyncio
async def test_skills_different_workspaces(fastapi_server):
    """Test that skills endpoint works for public workspace.

    Note: The agent-skills service is registered as a public service in the 'public'
    workspace, so it's accessed via /public/apps/agent-skills/...
    The workspace context in the documentation is determined by the URL workspace.
    """
    # Test with public workspace - the service is registered in public workspace
    response_public = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response_public.status_code == 200
    assert "public" in response_public.text

    # Test WORKSPACE_CONTEXT.md shows public workspace info
    response_ctx = requests.get(f"{SERVER_URL}/public/apps/agent-skills/WORKSPACE_CONTEXT.md")
    assert response_ctx.status_code == 200
    assert "public" in response_ctx.text


@pytest.mark.asyncio
async def test_skills_dynamic_schema_extraction(fastapi_server):
    """Test that API documentation is dynamically extracted from service schemas."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/REFERENCE.md")
    assert response.status_code == 200

    content = response.text

    # Verify that the documentation includes parameter descriptions
    # These should be extracted from Pydantic Field descriptions in the actual code

    # Check for parameter documentation format
    assert "**Parameters:**" in content

    # Check that we have descriptions for parameters (from Field descriptions)
    # The actual text will come from the service implementations
    assert "`" in content  # Parameter names are in backticks
    assert ":" in content  # Descriptions follow colons


def test_skills_module_imports():
    """Test that the skills module can be imported without errors."""
    from hypha.skills import (
        AgentSkillsService,
        AgentSkillsMiddleware,
        DynamicDocGenerator,
        setup_agent_skills,
        create_agent_skills_service,
        get_skill_md,
        get_examples_md,
        get_workspace_context_md,
        extract_schema_from_callable,
        format_schema_as_markdown,
    )

    # Verify the classes and functions exist
    assert AgentSkillsService is not None
    assert AgentSkillsMiddleware is not None
    assert DynamicDocGenerator is not None
    assert callable(setup_agent_skills)
    assert callable(create_agent_skills_service)
    assert callable(get_skill_md)
    assert callable(get_examples_md)
    assert callable(get_workspace_context_md)
    assert callable(extract_schema_from_callable)
    assert callable(format_schema_as_markdown)


def test_skills_schema_extraction():
    """Test the schema extraction function."""
    from hypha.skills import extract_schema_from_callable, extract_docstring
    from hypha_rpc.utils.schema import schema_function
    from pydantic import Field

    @schema_function
    def test_function(
        name: str = Field(..., description="The name parameter"),
        value: int = Field(10, description="The value parameter")
    ) -> str:
        """A test function with docstring."""
        return f"{name}: {value}"

    # Test schema extraction
    schema = extract_schema_from_callable(test_function)
    assert schema is not None
    assert "parameters" in schema
    assert "properties" in schema["parameters"]
    assert "name" in schema["parameters"]["properties"]
    assert "value" in schema["parameters"]["properties"]

    # Test docstring extraction
    docstring = extract_docstring(test_function)
    assert "A test function with docstring" in docstring


def test_skills_format_schema_as_markdown():
    """Test the markdown formatting function."""
    from hypha.skills import format_schema_as_markdown

    schema = {
        "name": "test_method",
        "description": "Test method description",
        "parameters": {
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter", "default": 10}
            },
            "required": ["param1"]
        }
    }

    markdown = format_schema_as_markdown(schema, "test_method", "Test docstring")

    # Verify the markdown output
    assert "### test_method" in markdown
    assert "**Parameters:**" in markdown
    assert "`param1`" in markdown
    assert "`param2`" in markdown
    assert "*(required)*" in markdown  # param1 is required
    assert "Default:" in markdown  # param2 has default


def test_skills_load_documentation_file():
    """Test loading documentation files from the docs directory."""
    from hypha.skills import load_documentation_file

    # Test loading an existing file
    content = load_documentation_file("quick-start.md")
    if content:  # File exists
        assert "Hypha" in content

    # Test loading a non-existent file
    content = load_documentation_file("nonexistent-file.md")
    assert content is None


@pytest.mark.asyncio
async def test_skills_index_enabled_services(fastapi_server):
    """Test that index endpoint includes enabled services list."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/")
    assert response.status_code == 200
    data = response.json()

    # Verify enabled services and source endpoints are included
    assert "enabled_services" in data
    assert "source_endpoints" in data
    assert isinstance(data["enabled_services"], list)
    assert isinstance(data["source_endpoints"], list)

    # Workspace manager should always be enabled
    assert "workspace-manager" in data["enabled_services"]

    # Should have corresponding source endpoints
    for svc_id in data["enabled_services"]:
        assert f"SOURCE/{svc_id}" in data["source_endpoints"]


@pytest.mark.asyncio
async def test_skills_reference_shows_enabled_services(fastapi_server):
    """Test that REFERENCE.md shows which services are enabled."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/REFERENCE.md")
    assert response.status_code == 200

    content = response.text

    # Verify enabled services section
    assert "## Enabled Services" in content
    assert "Workspace Manager" in content
    assert "workspace-manager" in content


@pytest.mark.asyncio
async def test_skills_source_endpoint_service_list(fastapi_server):
    """Test the source code endpoint for listing service methods."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SOURCE/workspace-manager")
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/markdown")

    content = response.text

    # Verify the response structure
    assert "# Source Code Reference" in content
    assert "WorkspaceManager" in content
    assert "## Available Methods" in content
    # Should list some common methods
    assert "list_services" in content or "register_service" in content


@pytest.mark.asyncio
async def test_skills_source_endpoint_specific_method(fastapi_server):
    """Test getting source code for a specific method."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SOURCE/workspace-manager/list_services")
    assert response.status_code == 200

    content = response.text

    # Verify it contains actual source code
    assert "```python" in content
    assert "def" in content or "async def" in content


@pytest.mark.asyncio
async def test_skills_source_endpoint_unknown_service(fastapi_server):
    """Test that unknown services return 404 with helpful message."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SOURCE/unknown-service")
    assert response.status_code == 404

    content = response.text

    # Should list available services
    assert "Unknown service" in content
    assert "workspace-manager" in content


@pytest.mark.asyncio
async def test_skills_source_endpoint_unknown_method(fastapi_server):
    """Test that unknown methods return 404."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SOURCE/workspace-manager/nonexistent_method_xyz")
    assert response.status_code == 404
    assert "Method not found" in response.text


def test_skills_dynamic_doc_generator_methods():
    """Test DynamicDocGenerator helper methods."""
    from hypha.skills import DynamicDocGenerator

    # Create a mock store with minimal attributes
    class MockStore:
        _public_services = []
        _artifact_manager = None
        _s3_controller = None

    mock_store = MockStore()
    generator = DynamicDocGenerator(mock_store, "http://localhost:9527")

    # Test enabled services detection
    enabled = generator.get_enabled_services()
    assert isinstance(enabled, list)
    assert "workspace-manager" in enabled  # Always enabled

    # Test source code retrieval (should work for hypha modules)
    source = generator.get_source_code("hypha.skills", "DynamicDocGenerator")
    assert source is not None
    assert "class DynamicDocGenerator" in source

    # Test source code security - should reject non-hypha modules
    source = generator.get_source_code("os", "path")
    assert source is None

    # Test method source retrieval
    method_source = generator.get_method_source("WorkspaceManager", "list_services")
    assert method_source is not None
    assert "list_services" in method_source or "def" in method_source


@pytest.mark.asyncio
async def test_skills_dynamic_doc_generator_service_apis():
    """Test DynamicDocGenerator service API extraction."""
    from hypha.skills import DynamicDocGenerator

    class MockStore:
        _public_services = []
        _artifact_manager = True  # Simulate artifact manager enabled
        _s3_controller = None

        def get_server_info(self):
            return {"public_base_url": "http://localhost:9527"}

    mock_store = MockStore()
    generator = DynamicDocGenerator(mock_store, "http://localhost:9527")

    # Test workspace manager API (always available)
    ws_api = await generator.get_service_api("workspace-manager")
    assert ws_api is not None
    assert ws_api["id"] == "workspace-manager"
    assert "methods" in ws_api
    assert len(ws_api["methods"]) > 0

    # Test artifact manager API (simulated as enabled)
    artifact_api = await generator.get_service_api("artifact-manager")
    assert artifact_api is not None
    assert artifact_api["id"] == "artifact-manager"

    # Test get all enabled services
    all_services = await generator.get_all_enabled_services()
    assert isinstance(all_services, list)
    service_ids = [s["id"] for s in all_services]
    assert "workspace-manager" in service_ids


@pytest.mark.asyncio
async def test_skills_create_zip_file(fastapi_server):
    """Test the create-zip-file endpoint returns a valid zip archive."""
    import zipfile
    import io

    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/create-zip-file")
    assert response.status_code == 200
    assert response.headers.get("content-type") == "application/zip"
    assert "attachment" in response.headers.get("content-disposition", "")

    # Verify it's a valid zip file
    zip_buffer = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_buffer, 'r') as zf:
        # Check required files exist
        file_names = zf.namelist()
        assert "SKILL.md" in file_names
        assert "REFERENCE.md" in file_names
        assert "EXAMPLES.md" in file_names
        assert "WORKSPACE_CONTEXT.md" in file_names

        # Check SKILL.md content
        skill_content = zf.read("SKILL.md").decode("utf-8")
        assert "name: hypha" in skill_content
        assert "# Hypha Workspace Manager" in skill_content

        # Check SOURCE directory exists for workspace-manager
        source_files = [f for f in file_names if f.startswith("SOURCE/")]
        assert len(source_files) > 0
        assert any("workspace-manager" in f for f in source_files)


@pytest.mark.asyncio
async def test_skills_index_includes_download_info(fastapi_server):
    """Test that index endpoint includes download information."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/")
    assert response.status_code == 200
    data = response.json()

    # Verify download section is included
    assert "download" in data
    assert "zip" in data["download"]
    assert data["download"]["zip"] == "create-zip-file"
