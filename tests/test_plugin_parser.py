"""Test module for app parser."""
from pathlib import Path
from hypha.plugin_parser import parse_imjoy_plugin


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
