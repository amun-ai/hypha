"""Test module for app parser."""

from pathlib import Path
from hypha.plugin_parser import parse_imjoy_plugin, extract_files_from_source
from hypha_rpc.utils import ObjectProxy


def test_python_plugin():
    """Test loading python app."""
    source = (
        (Path(__file__).parent / "testWebPythonPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = parse_imjoy_plugin(source)
    assert [] == config["requirements"]
    assert config["api_version"] == "0.1.7"
    assert config["type"] == "web-python"


def test_js_plugin():
    """Test loading javascript app."""
    source = (
        (Path(__file__).parent / "testWebWorkerPlugin.imjoy.html")
        .open(encoding="utf-8")
        .read()
    )
    config = parse_imjoy_plugin(source)
    assert [] == config["requirements"]
    assert config["api_version"] == "0.1.7"
    assert config["type"] == "web-worker"


def test_script_tag_extraction():
    """Test that script content is extracted without script tags."""
    source = """
<config lang="json">
{
    "name": "Test Script App",
    "type": "web-python",
    "version": "0.1.0"
}
</config>

<script lang="python">
import sys
import os
from hypha_rpc import api

def hello():
    return "Hello World"

api.export({"hello": hello})
</script>
"""
    
    # Test extract_files_from_source
    extracted_files, remaining_source = extract_files_from_source(source)
    
    # Should extract script as main.py
    script_files = [f for f in extracted_files if f.get("source_type") == "script"]
    assert len(script_files) == 1
    assert script_files[0]["name"] == "main.py"
    
    script_content = script_files[0]["content"]
    assert script_content.strip().startswith("import sys")
    assert "def hello():" in script_content
    assert "api.export" in script_content
    # Ensure no script tags in content
    assert "<script" not in script_content
    assert "</script>" not in script_content
    
    # Remaining source should not contain script tags
    assert "<script" not in remaining_source
    
    # Test parse_imjoy_plugin
    config = parse_imjoy_plugin(source)
    assert config["name"] == "Test Script App"
    assert config["type"] == "web-python"
    assert "import sys" in config["script"]
    assert "<script" not in config["script"]


def test_file_tag_extraction():
    """Test that file content is extracted without file tags and with proper whitespace handling."""
    source = """
<config lang="json">
{
    "name": "Test File App",
    "type": "web-python"
}
</config>

<file name="requirements.txt">
fastapi==0.112.1
numpy==1.24.0
pandas==2.0.0
</file>

<file name="utils.py" format="text">
def helper_function():
    return "Hello from utils!"

class DataProcessor:
    def process(self, data):
        return data * 2
</file>

<script lang="python">
from utils import helper_function
print(helper_function())
</script>
"""
    
    extracted_files, remaining_source = extract_files_from_source(source)
    
    # Should extract 4 files: manifest.json, requirements.txt, utils.py, and main.py
    assert len(extracted_files) == 4
    
    # Check manifest.json (config)
    manifest_files = [f for f in extracted_files if f["name"] == "manifest.json"]
    assert len(manifest_files) == 1
    manifest_content = manifest_files[0]["content"]
    assert manifest_content["name"] == "Test File App"
    assert manifest_content["type"] == "web-python"
    
    # Check requirements.txt
    req_files = [f for f in extracted_files if f["name"] == "requirements.txt"]
    assert len(req_files) == 1
    req_content = req_files[0]["content"]
    assert req_content.startswith("fastapi==0.112.1")
    assert "numpy==1.24.0" in req_content
    assert "pandas==2.0.0" in req_content
    # Should not have leading/trailing whitespace
    assert not req_content.startswith("\n")
    assert not req_content.endswith("\n")
    
    # Check utils.py
    utils_files = [f for f in extracted_files if f["name"] == "utils.py"]
    assert len(utils_files) == 1
    utils_content = utils_files[0]["content"]
    assert "def helper_function():" in utils_content
    assert "class DataProcessor:" in utils_content
    assert "def process(self, data):" in utils_content
    # Should not have leading/trailing whitespace
    assert not utils_content.startswith("\n")
    assert not utils_content.endswith("\n")
    
    # Check main.py (script)
    script_files = [f for f in extracted_files if f.get("source_type") == "script"]
    assert len(script_files) == 1
    script_content = script_files[0]["content"]
    assert "from utils import helper_function" in script_content
    assert "print(helper_function())" in script_content
    assert script_files[0]["name"] == "main.py"
    
    # Remaining source should not contain file or script tags
    assert "<file" not in remaining_source
    assert "<script" not in remaining_source


def test_file_tag_with_nested_html():
    """Test file tag extraction with nested HTML content."""
    source = """
<config lang="json">
{
    "name": "Test Nested Content",
    "type": "window"
}
</config>

<file name="template.html">
<div class="container">
    <h1>Welcome to <strong>My App</strong></h1>
    <p>This is a <em>complex</em> template with nested tags.</p>
    <ul>
        <li>Feature 1</li>
        <li>Feature 2</li>
    </ul>
    <script>
        console.log("Embedded JavaScript");
    </script>
</div>
</file>

<file name="styles.css">
.container {
    padding: 20px;
}

h1 {
    color: #333;
}
</file>
"""
    
    extracted_files, remaining_source = extract_files_from_source(source)
    
    # Should extract 2 or 3 files (might extract script from within file content)
    assert len(extracted_files) >= 2
    
    # Check template.html - should capture all nested content
    html_files = [f for f in extracted_files if f["name"] == "template.html"]
    assert len(html_files) == 1
    html_content = html_files[0]["content"]
    
    # All text content should be preserved
    assert "Welcome to" in html_content
    assert "My App" in html_content
    assert "complex" in html_content
    assert "Feature 1" in html_content
    assert "Feature 2" in html_content
    assert "Embedded JavaScript" in html_content
    
    # Check styles.css
    css_files = [f for f in extracted_files if f["name"] == "styles.css"]
    assert len(css_files) == 1
    css_content = css_files[0]["content"]
    assert ".container" in css_content
    assert "padding: 20px;" in css_content
    assert "color: #333;" in css_content


def test_manifest_tag_extraction():
    """Test manifest tag extraction and JSON parsing."""
    source = """
<manifest lang="json">
{
    "name": "Manifest Test App",
    "type": "conda-jupyter-kernel",
    "version": "1.0.0",
    "dependencies": ["numpy", "pandas"],
    "entry_point": "main.py"
}
</manifest>

<script lang="python">
import numpy as np
print("Hello from manifest app")
</script>
"""
    
    extracted_files, remaining_source = extract_files_from_source(source)
    
    # Should extract 2 files: manifest.json and main.py
    assert len(extracted_files) == 2
    
    # Check that manifest.json was extracted
    manifest_files = [f for f in extracted_files if f["name"] == "manifest.json"]
    assert len(manifest_files) == 1
    
    manifest_content = manifest_files[0]["content"]
    assert isinstance(manifest_content, dict)  # Should be parsed JSON
    assert manifest_content["name"] == "Manifest Test App"
    assert manifest_content["type"] == "conda-jupyter-kernel"
    assert manifest_content["dependencies"] == ["numpy", "pandas"]
    
    # Check script extraction
    script_files = [f for f in extracted_files if f.get("source_type") == "script"]
    assert len(script_files) == 1
    assert script_files[0]["name"] == "main.py"
    assert "import numpy as np" in script_files[0]["content"]
    
    # Remaining source should be clean
    assert "<manifest" not in remaining_source
    assert "<script" not in remaining_source


def test_multiple_script_tags():
    """Test handling of multiple script tags with different languages."""
    source = """
<config lang="json">
{
    "name": "Multi Script App",
    "type": "web-worker"
}
</config>

<script lang="javascript">
class MyApp {
    setup() {
        console.log("JavaScript setup");
    }
}
api.export(new MyApp());
</script>

<script lang="python">
from hypha_rpc import api
print("Python script")
api.export({"test": lambda: "python"})
</script>
"""
    
    extracted_files, remaining_source = extract_files_from_source(source)
    
    # Should extract 2 script files
    script_files = [f for f in extracted_files if f.get("source_type") == "script"]
    assert len(script_files) == 2
    
    # Check JavaScript file
    js_files = [f for f in script_files if f["name"] == "main.js"]
    assert len(js_files) == 1
    js_content = js_files[0]["content"]
    assert "class MyApp" in js_content
    assert "console.log" in js_content
    assert "api.export(new MyApp())" in js_content
    
    # Check Python file
    py_files = [f for f in script_files if f["name"] == "main.py"]
    assert len(py_files) == 1
    py_content = py_files[0]["content"]
    assert "from hypha_rpc import api" in py_content
    assert "print(\"Python script\")" in py_content
    
    # Remaining source should be clean
    assert "<script" not in remaining_source


def test_empty_and_whitespace_content():
    """Test handling of empty and whitespace-only content."""
    source = """
<config lang="json">
{
    "name": "Whitespace Test",
    "type": "web-python"
}
</config>

<file name="empty.txt">
</file>

<file name="whitespace.txt">


</file>

<file name="normal.txt">
Some content here
</file>

<script lang="python">

print("Script with leading whitespace")

</script>
"""
    
    extracted_files, remaining_source = extract_files_from_source(source)
    
    # Should extract 5 files: manifest.json, empty.txt, whitespace.txt, normal.txt, main.py
    assert len(extracted_files) == 5
    
    # Check empty file
    empty_files = [f for f in extracted_files if f["name"] == "empty.txt"]
    assert len(empty_files) == 1
    assert empty_files[0]["content"] == ""
    
    # Check whitespace-only file
    ws_files = [f for f in extracted_files if f["name"] == "whitespace.txt"]
    assert len(ws_files) == 1
    assert ws_files[0]["content"] == ""  # Should be stripped
    
    # Check normal file
    normal_files = [f for f in extracted_files if f["name"] == "normal.txt"]
    assert len(normal_files) == 1
    assert normal_files[0]["content"] == "Some content here"
    
    # Check script with whitespace
    script_files = [f for f in extracted_files if f.get("source_type") == "script"]
    assert len(script_files) == 1
    assert script_files[0]["name"] == "main.py"
    script_content = script_files[0]["content"]
    assert script_content == "print(\"Script with leading whitespace\")"  # Should be stripped


def test_config_parsing_variations():
    """Test different config tag variations."""
    # Test YAML config
    yaml_source = """
<config lang="yaml">
name: YAML Config App
type: web-python
version: 1.0.0
dependencies:
  - numpy
  - pandas
</config>

<script lang="python">
print("YAML config test")
</script>
"""
    
    config = parse_imjoy_plugin(yaml_source)
    assert config["name"] == "YAML Config App"
    assert config["type"] == "web-python"
    assert config["version"] == "1.0.0"
    assert "numpy" in config["dependencies"]
    assert "pandas" in config["dependencies"]
    
    # Test JSON config
    json_source = """
<config lang="json">
{
    "name": "JSON Config App",
    "type": "web-worker",
    "version": "2.0.0"
}
</config>

<script lang="javascript">
console.log("JSON config test");
</script>
"""
    
    config = parse_imjoy_plugin(json_source)
    assert config["name"] == "JSON Config App"
    assert config["type"] == "web-worker"
    assert config["version"] == "2.0.0"


def test_raw_html_content():
    """Test that raw HTML content is not parsed as ImJoy/Hypha format."""
    raw_html = """<!DOCTYPE html>
<html>
<head>
    <title>Raw HTML</title>
</head>
<body>
    <h1>This is raw HTML</h1>
    <p>It should not be parsed as ImJoy format</p>
</body>
</html>
"""
    
    extracted_files, remaining_source = extract_files_from_source(raw_html)
    
    # Should not extract any files
    assert len(extracted_files) == 0
    # Should return original source unchanged
    assert remaining_source == raw_html


def test_error_handling():
    """Test error handling for malformed content."""
    # Test invalid JSON in config
    try:
        invalid_json_source = """
<config lang="json">
{
    "name": "Invalid JSON"
    "missing_comma": true
}
</config>
"""
        config = parse_imjoy_plugin(invalid_json_source)
        assert False, "Should have raised an exception for invalid JSON"
    except Exception as e:
        # Check for JSON-related error messages
        error_msg = str(e).lower()
        assert ("json" in error_msg or "expecting" in error_msg or "delimiter" in error_msg)
    
    # Test unsupported config language
    try:
        unsupported_source = """
<config lang="xml">
<name>Unsupported XML</name>
</config>
"""
        config = parse_imjoy_plugin(unsupported_source)
        assert False, "Should have raised an exception for unsupported config language"
    except Exception as e:
        assert "Unsupported config language" in str(e)


def test_object_proxy_functionality():
    """Test ObjectProxy functionality used in parsing."""
    # Create an ObjectProxy instance
    proxy = ObjectProxy()
    
    # Test basic attribute setting and getting
    proxy.name = "Test"
    proxy.version = "1.0.0"
    assert proxy.name == "Test"
    assert proxy.version == "1.0.0"
    
    # Test dictionary-style access
    proxy["type"] = "web-python"
    assert proxy["type"] == "web-python"
    
    # Test fromDict class method
    data = {"name": "FromDict Test", "version": "2.0.0"}
    proxy2 = ObjectProxy.fromDict(data)
    assert proxy2.name == "FromDict Test"
    assert proxy2.version == "2.0.0"
