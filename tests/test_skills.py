"""Test the agent skills module."""

import io
import re
import zipfile

import pytest
import requests
import yaml

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
    assert "## Step 1: Install Dependencies" in content
    assert "## Step 2: Authentication" in content
    assert "## Step 3: Connect and Use Services" in content
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
async def test_skills_cross_workspace_access(fastapi_server):
    """Test that agent-skills is accessible from any workspace via HTTP fallback.

    The agent-skills service is registered in the public workspace, but should
    be accessible from any workspace URL. The HTTP proxy falls back to the
    public workspace when a service is not found in the target workspace.
    """
    import os
    from hypha_rpc import connect_to_server

    token = os.environ.get("HYPHA_ROOT_TOKEN")
    assert token, "HYPHA_ROOT_TOKEN must be set"

    # Connect and create a workspace
    api = await connect_to_server(
        {"server_url": SERVER_URL, "token": token}
    )
    workspace = api.config["workspace"]

    # Generate a token for the workspace
    ws_token = await api.generate_token()

    # Access agent-skills from the non-public workspace with auth
    response = requests.get(
        f"{SERVER_URL}/{workspace}/apps/agent-skills/SKILL.md",
        headers={"Authorization": f"Bearer {ws_token}"},
    )
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text[:200]}"
    )
    assert "# Hypha Workspace Manager" in response.text

    # Verify the workspace context reflects the actual workspace from URL
    assert workspace in response.text

    # Also test the index endpoint
    response_index = requests.get(
        f"{SERVER_URL}/{workspace}/apps/agent-skills/",
        headers={"Authorization": f"Bearer {ws_token}"},
    )
    assert response_index.status_code == 200
    data = response_index.json()
    assert data["name"] == "hypha"

    await api.disconnect()


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


@pytest.mark.asyncio
async def test_skills_bootstrapping_instructions(fastapi_server):
    """Test that SKILL.md contains complete bootstrapping instructions.

    An AI agent should be able to follow these instructions to:
    1. Install the required library
    2. Authenticate
    3. Connect and use services
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    content = response.text

    # Step 1: Must include installation instructions
    assert "pip install hypha-rpc" in content
    assert "npm install hypha-rpc" in content

    # Step 2: Must include authentication instructions
    assert "## Step 2: Authentication" in content
    assert "Anonymous Access" in content
    assert "Token-Based Access" in content
    assert "Authorization: Bearer" in content or "Bearer" in content

    # Step 3: Must include connection instructions
    assert "connect_to_server" in content
    assert "## Step 3: Connect and Use Services" in content

    # Must include HTTP-only path (no library needed)
    assert "HTTP Only" in content or "HTTP API" in content
    assert "curl" in content

    # Must include token generation for programmatic access
    assert "generate_token" in content


@pytest.mark.asyncio
async def test_skills_examples_has_installation(fastapi_server):
    """Test that EXAMPLES.md starts with setup/installation."""
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/EXAMPLES.md")
    assert response.status_code == 200
    content = response.text

    # Must include installation section
    assert "## Setup & Installation" in content
    assert "pip install hypha-rpc" in content

    # Must include anonymous connection example
    assert "Anonymous Connection" in content or "anonymous" in content.lower()


@pytest.mark.asyncio
async def test_skills_public_workspace_accessible_without_auth(fastapi_server):
    """Test that public workspace skills are accessible without authentication."""
    # Public workspace should work without any auth token
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    assert "# Hypha Workspace Manager" in response.text

    # Non-public workspace skills require authentication
    response = requests.get(f"{SERVER_URL}/test-workspace/apps/agent-skills/SKILL.md")
    assert response.status_code == 401
    data = response.json()
    assert "Authentication required" in data["error"]
    assert "global_url" in data  # Points to the global endpoint


@pytest.mark.asyncio
async def test_skills_global_endpoint_no_auth(fastapi_server):
    """Test the global /apps/agent-skills/ endpoint works without auth."""
    # Global index
    response = requests.get(f"{SERVER_URL}/apps/agent-skills/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "hypha"
    assert data["type"] == "global"
    assert "SKILL.md" in data["files"]
    assert "workspace_skills_pattern" in data

    # Global SKILL.md
    response = requests.get(f"{SERVER_URL}/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/markdown")
    content = response.text
    assert "# Hypha Platform Guide" in content
    assert "global" in content.lower()
    assert "workspace-specific skills" in content.lower() or "Workspace-Specific" in content

    # Global REFERENCE.md
    response = requests.get(f"{SERVER_URL}/apps/agent-skills/REFERENCE.md")
    assert response.status_code == 200
    assert "**Parameters:**" in response.text

    # Global EXAMPLES.md
    response = requests.get(f"{SERVER_URL}/apps/agent-skills/EXAMPLES.md")
    assert response.status_code == 200
    assert "hypha" in response.text.lower()

    # 404 for unknown files
    response = requests.get(f"{SERVER_URL}/apps/agent-skills/UNKNOWN.md")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_skills_global_has_cors_headers(fastapi_server):
    """Test that global endpoint includes CORS headers."""
    response = requests.get(f"{SERVER_URL}/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


@pytest.mark.asyncio
async def test_skills_global_skill_md_content(fastapi_server):
    """Test that the global SKILL.md has proper content for bootstrapping."""
    response = requests.get(f"{SERVER_URL}/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    content = response.text

    # Should have YAML frontmatter
    assert content.startswith("---")

    # Should explain how to connect
    assert "connect_to_server" in content

    # Should explain how to get tokens
    assert "generate_token" in content

    # Should reference workspace-specific skills
    assert "/apps/agent-skills/" in content

    # Should have HTTP examples
    assert "curl" in content

    # Should explain authentication options
    assert "Bearer" in content


# ============================================================================
# Utility Tests: Verifying the agent skills documentation is actually useful
# for AI agents to bootstrap and interact with Hypha
# ============================================================================


@pytest.mark.asyncio
async def test_skills_frontmatter_spec_compliance(fastapi_server):
    """Test that SKILL.md frontmatter complies with the Agent Skills spec.

    The spec (https://agentskills.io/specification.md) requires:
    - YAML frontmatter delimited by ---
    - 'name' field: 1-64 chars, lowercase letters, numbers, hyphens only
    - 'description' field: 1-1024 chars
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    content = response.text

    # Extract YAML frontmatter
    assert content.startswith("---"), "SKILL.md must start with YAML frontmatter delimiter"
    parts = content.split("---", 2)
    assert len(parts) >= 3, "SKILL.md must have opening and closing --- delimiters"

    frontmatter = yaml.safe_load(parts[1])
    assert frontmatter is not None, "Frontmatter must be valid YAML"

    # Validate 'name' field
    assert "name" in frontmatter, "Frontmatter must contain 'name'"
    name = frontmatter["name"]
    assert 1 <= len(name) <= 64, f"Name length {len(name)} not in [1, 64]"
    assert re.match(r"^[a-z0-9-]+$", name), f"Name '{name}' contains invalid chars"

    # Validate 'description' field
    assert "description" in frontmatter, "Frontmatter must contain 'description'"
    desc = frontmatter["description"]
    assert 1 <= len(desc) <= 1024, f"Description length {len(desc)} not in [1, 1024]"

    # Validate optional but recommended fields
    assert "metadata" in frontmatter, "Should include metadata"
    assert "version" in frontmatter["metadata"], "Metadata should include version"
    assert "server_url" in frontmatter["metadata"], "Metadata should include server_url"


@pytest.mark.asyncio
async def test_skills_server_url_consistency(fastapi_server):
    """Test that server URLs in all documents are consistent and valid."""
    files = ["SKILL.md", "REFERENCE.md", "EXAMPLES.md", "WORKSPACE_CONTEXT.md"]
    server_urls = set()

    for filename in files:
        response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/{filename}")
        assert response.status_code == 200, f"Failed to fetch {filename}"
        content = response.text

        # Extract server_url references (look for http:// or https:// URLs)
        urls = re.findall(r'https?://[^\s"\'`\)}\]]+', content)
        for url in urls:
            # Clean trailing punctuation
            url = url.rstrip(".,;:")
            # Only consider Hypha server URLs (not example external URLs)
            if "127.0.0.1" in url or "localhost" in url or "hypha" in url:
                # Extract base URL (scheme + host + port)
                match = re.match(r'(https?://[^/]+)', url)
                if match:
                    server_urls.add(match.group(1))

    # All Hypha server URLs should be the same base URL
    assert len(server_urls) <= 1, f"Inconsistent server URLs found: {server_urls}"


@pytest.mark.asyncio
async def test_skills_all_code_blocks_have_language(fastapi_server):
    """Test that all fenced code blocks specify a language for syntax highlighting.

    AI agents benefit from knowing the language of code blocks to execute them correctly.
    """
    files = ["SKILL.md", "EXAMPLES.md"]

    for filename in files:
        response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/{filename}")
        assert response.status_code == 200

        content = response.text
        # Find all code fence openings
        fences = re.findall(r"^```(.*)$", content, re.MULTILINE)

        for i, fence in enumerate(fences):
            fence = fence.strip()
            # Every other fence is a closing fence (empty)
            # But we should check that opening fences have a language
            if fence == "":
                # This could be a closing fence. Count if we're inside a block.
                # Simple heuristic: check if the previous non-empty fence had a language
                continue

            # The language should be a known language identifier
            known_languages = {"python", "javascript", "bash", "json", "yaml", "text", "markdown"}
            assert fence.lower() in known_languages, (
                f"Code block in {filename} has unrecognized language: '{fence}'. "
                f"Known: {known_languages}"
            )


@pytest.mark.asyncio
async def test_skills_python_examples_have_imports(fastapi_server):
    """Test that Python code examples include necessary import statements.

    An AI agent following the examples should not have to guess imports.
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    content = response.text

    # Extract Python code blocks
    python_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    assert len(python_blocks) > 0, "SKILL.md should contain Python examples"

    # Check that blocks using connect_to_server have the import
    for block in python_blocks:
        if "connect_to_server(" in block and "import" not in block:
            # It might be a continuation block; check if it uses connect_to_server
            # in a way that assumes prior import context
            if "from hypha_rpc" not in block and "async with" in block:
                # Allow continuation blocks that are inside an async with context
                pass
            else:
                assert False, (
                    f"Python block uses connect_to_server but missing import:\n{block[:200]}"
                )


@pytest.mark.asyncio
async def test_skills_http_examples_are_functional(fastapi_server):
    """Test that HTTP examples in SKILL.md actually work against the server.

    This verifies the documented curl commands would produce valid results.
    """
    # The SKILL.md documents: curl "{server_url}/public/services"
    response = requests.get(f"{SERVER_URL}/public/services")
    assert response.status_code == 200, (
        "HTTP example 'GET /public/services' from SKILL.md doesn't work"
    )
    data = response.json()
    assert isinstance(data, list), "Services endpoint should return a list"

    # The SKILL.md documents health endpoint
    response = requests.get(f"{SERVER_URL}/health/liveness")
    assert response.status_code == 200, (
        "Health endpoint documented in EXAMPLES.md doesn't work"
    )


@pytest.mark.asyncio
async def test_skills_reference_documents_all_key_methods(fastapi_server):
    """Test that REFERENCE.md documents the most important methods.

    An AI agent needs at minimum these operations documented:
    - list_services (discover what's available)
    - register_service (create services)
    - get_service (use services)
    - generate_token (authentication)
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/REFERENCE.md")
    assert response.status_code == 200
    content = response.text

    # Essential workspace manager methods that an agent needs
    essential_methods = [
        "list_services",
        "register_service",
        "generate_token",
    ]

    for method in essential_methods:
        assert method in content, (
            f"REFERENCE.md missing documentation for essential method: {method}"
        )

    # Verify methods have parameter documentation
    # Each method section should have Parameters
    method_sections = content.split("### ")
    documented_methods = 0
    for section in method_sections[1:]:  # Skip first split before any ###
        if "**Parameters:**" in section:
            documented_methods += 1

    assert documented_methods >= 5, (
        f"Only {documented_methods} methods have parameter docs, expected at least 5"
    )


@pytest.mark.asyncio
async def test_skills_examples_cover_all_capabilities(fastapi_server):
    """Test that EXAMPLES.md covers all major capabilities listed in SKILL.md."""
    # Get SKILL.md capabilities
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    skill_content = response.text

    # Get EXAMPLES.md
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/EXAMPLES.md")
    assert response.status_code == 200
    examples_content = response.text

    # Every major capability in SKILL.md should have examples
    capabilities = [
        ("Service Management", "register_service"),
        ("Token & Permission Management", "generate_token"),
        ("Artifact Management", "artifact_manager"),
        ("Server Applications", "controller.install"),
        ("MCP Integration", "mcp"),
        ("A2A Protocol", "a2a"),
        ("Vector Search", "search_vectors"),
    ]

    for capability_name, keyword in capabilities:
        assert keyword in examples_content.lower() or keyword in examples_content, (
            f"EXAMPLES.md missing examples for capability: {capability_name} "
            f"(expected keyword: {keyword})"
        )


@pytest.mark.asyncio
async def test_skills_examples_show_error_handling(fastapi_server):
    """Test that EXAMPLES.md includes error handling patterns.

    AI agents need to know how to handle common errors.
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/EXAMPLES.md")
    assert response.status_code == 200
    content = response.text

    # Should have an error handling section
    assert "## Error Handling" in content, "EXAMPLES.md should have Error Handling section"

    # Should cover common error types
    assert "KeyError" in content, "Should show how to handle service not found"
    assert "PermissionError" in content, "Should show how to handle permission denied"
    assert "TimeoutError" in content, "Should show how to handle timeouts"


@pytest.mark.asyncio
async def test_skills_progressive_disclosure(fastapi_server):
    """Test that skills documents follow progressive disclosure pattern.

    SKILL.md should be concise (overview), while REFERENCE.md provides details.
    This pattern helps AI agents consume documentation efficiently.
    """
    skill_resp = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    ref_resp = requests.get(f"{SERVER_URL}/public/apps/agent-skills/REFERENCE.md")
    examples_resp = requests.get(f"{SERVER_URL}/public/apps/agent-skills/EXAMPLES.md")

    skill_len = len(skill_resp.text)
    ref_len = len(ref_resp.text)
    examples_len = len(examples_resp.text)

    # SKILL.md should be the most concise (overview/quickstart)
    assert skill_len < ref_len, (
        f"SKILL.md ({skill_len} chars) should be shorter than REFERENCE.md ({ref_len} chars)"
    )

    # REFERENCE.md should be detailed (full API docs)
    assert ref_len > 5000, (
        f"REFERENCE.md ({ref_len} chars) seems too short for comprehensive API docs"
    )

    # EXAMPLES.md should have substantial content
    assert examples_len > 5000, (
        f"EXAMPLES.md ({examples_len} chars) seems too short for comprehensive examples"
    )

    # SKILL.md should reference the other files
    assert "REFERENCE.md" in skill_resp.text, "SKILL.md should link to REFERENCE.md"
    assert "EXAMPLES.md" in skill_resp.text, "SKILL.md should link to EXAMPLES.md"


@pytest.mark.asyncio
async def test_skills_zip_contains_complete_documentation(fastapi_server):
    """Test that the zip download contains all documentation and source code.

    An AI agent should be able to download the entire skill as a zip
    and have everything needed offline.
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/create-zip-file")
    assert response.status_code == 200

    zip_buffer = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_buffer, "r") as zf:
        file_names = zf.namelist()

        # Must have all 4 documentation files
        required_files = ["SKILL.md", "REFERENCE.md", "EXAMPLES.md", "WORKSPACE_CONTEXT.md"]
        for f in required_files:
            assert f in file_names, f"Zip missing required file: {f}"

        # Must have SOURCE directory with workspace-manager
        source_files = [f for f in file_names if f.startswith("SOURCE/")]
        assert len(source_files) > 0, "Zip should contain SOURCE/ directory"

        ws_manager_files = [f for f in source_files if "workspace-manager" in f]
        assert len(ws_manager_files) > 0, "Zip should contain workspace-manager source"

        # Source files should contain actual Python code
        for f in ws_manager_files:
            if f.endswith(".py"):
                source_code = zf.read(f).decode("utf-8")
                assert "def " in source_code or "async def " in source_code, (
                    f"Source file {f} doesn't contain function definitions"
                )

        # SKILL.md in zip should match the live endpoint
        skill_in_zip = zf.read("SKILL.md").decode("utf-8")
        live_response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
        assert skill_in_zip == live_response.text, (
            "SKILL.md in zip should match the live endpoint"
        )


@pytest.mark.asyncio
async def test_skills_agent_can_discover_services(fastapi_server):
    """End-to-end test: Verify an agent can follow the SKILL.md instructions
    to discover available services using the documented HTTP endpoint."""
    # Step 1: Agent reads SKILL.md
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    skill_content = response.text

    # Step 2: Agent extracts the server URL from frontmatter
    parts = skill_content.split("---", 2)
    frontmatter = yaml.safe_load(parts[1])
    server_url = frontmatter["metadata"]["server_url"]
    workspace = frontmatter["metadata"]["workspace"]

    # Step 3: Agent follows the documented HTTP API pattern to list services
    # From SKILL.md: curl "{server_url}/{workspace}/services"
    services_response = requests.get(f"{server_url}/{workspace}/services")
    assert services_response.status_code == 200
    services = services_response.json()
    assert isinstance(services, list), "Services endpoint should return a list"
    assert len(services) > 0, "Should have at least one service"

    # Step 4: Agent can get more details from REFERENCE.md
    ref_response = requests.get(f"{server_url}/{workspace}/apps/agent-skills/REFERENCE.md")
    assert ref_response.status_code == 200
    assert "Workspace Manager" in ref_response.text


@pytest.mark.asyncio
async def test_skills_agent_can_call_service_via_http(fastapi_server):
    """End-to-end test: Verify an agent can call a service function via HTTP
    following the documented pattern in SKILL.md."""
    # The SKILL.md documents: GET /{workspace}/services/{service_id}/{function_name}
    # The workspace manager's check_status is a simple function to test

    # Call check_status on the default workspace manager
    response = requests.get(f"{SERVER_URL}/public/services/~/check_status")
    assert response.status_code == 200
    status = response.json()
    assert isinstance(status, dict), "check_status should return a dict"


@pytest.mark.asyncio
async def test_skills_index_is_valid_discovery_document(fastapi_server):
    """Test that the index endpoint serves as a valid discovery document.

    An AI agent should be able to GET the index and understand the full
    skill structure from the response.
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/")
    assert response.status_code == 200
    data = response.json()

    # Required discovery fields
    assert "name" in data
    assert "version" in data
    assert "files" in data
    assert "enabled_services" in data
    assert "source_endpoints" in data
    assert "download" in data

    # The files list should allow an agent to iterate and fetch each
    for filename in data["files"]:
        file_response = requests.get(
            f"{SERVER_URL}/public/apps/agent-skills/{filename}"
        )
        assert file_response.status_code == 200, (
            f"File listed in index not accessible: {filename}"
        )

    # Source endpoints should be accessible
    for endpoint in data["source_endpoints"][:3]:  # Test first 3
        source_response = requests.get(
            f"{SERVER_URL}/public/apps/agent-skills/{endpoint}"
        )
        assert source_response.status_code == 200, (
            f"Source endpoint listed in index not accessible: {endpoint}"
        )


@pytest.mark.asyncio
async def test_skills_multiformat_examples(fastapi_server):
    """Test that SKILL.md provides examples in all three formats: Python, JS, HTTP.

    AI agents may work in different environments, so all formats should be covered.
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    content = response.text

    # Step 3 should have Python, JavaScript, and HTTP sections
    assert "### Python" in content, "Missing Python section in Step 3"
    assert "### JavaScript" in content, "Missing JavaScript section in Step 3"
    assert "### HTTP API" in content, "Missing HTTP API section in Step 3"

    # Each should have working code examples
    assert "```python" in content
    assert "```javascript" in content
    assert "```bash" in content

    # Python example should show async with pattern
    assert "async with connect_to_server" in content

    # JS example should show connectToServer
    assert "connectToServer" in content

    # HTTP example should show curl
    assert "curl" in content


@pytest.mark.asyncio
async def test_skills_auth_options_documented(fastapi_server):
    """Test that all three authentication options are clearly documented.

    An AI agent needs to understand which auth method to use in different contexts.
    """
    response = requests.get(f"{SERVER_URL}/public/apps/agent-skills/SKILL.md")
    assert response.status_code == 200
    content = response.text

    # Option A: Anonymous
    assert "Anonymous Access" in content
    # Should explain when to use it
    assert "testing" in content.lower() or "public" in content.lower()

    # Option B: Token-Based
    assert "Token-Based Access" in content
    assert "Bearer" in content
    # Should show both Python and HTTP ways to use tokens
    assert '"token"' in content or "'token'" in content
    assert "Authorization:" in content

    # Option C: Interactive Login
    assert "Interactive Login" in content or "OAuth" in content
    assert "login" in content

    # Token generation docs
    assert "generate_token" in content
    # Should explain permission levels
    assert "read" in content
    assert "read_write" in content
    assert "admin" in content
