"""imjoy plugin parser module."""

import json
import uuid
import re
import base64

import yaml
from lxml import etree

from hypha_rpc.utils import DefaultObjectProxy
from hypha.core import ApplicationManifest

tag_types = [
    "config",
    "script",
    "link",
    "window",
    "style",
    "docs",
    "attachment",
    "file",
    "manifest",
]

CONFIGURABLE_FIELDS = [
    "env",
    "requirements",
    "dependencies",
    "icon",
    "ui",
    "type",
    "flags",
    "labels",
    "cover",
    "base_frame",
    "base_worker",
    "passive",
]


def parse_imjoy_plugin(source, overwrite_config=None):
    """Parse imjoy plugin file and return a dict with all the fields."""
    root = etree.HTML("<html>" + source + "</html>")
    plugin_comp = DefaultObjectProxy()
    for tag_type in tag_types:
        elms = root.xpath(f".//{tag_type}")
        values = []
        for elm in elms:
            # Get the complete inner HTML content
            inner_content = "".join(elm.itertext()) if elm.text or list(elm) else elm.text
            # Clean up the content by removing the outer tag wrapper
            if inner_content:
                inner_content = inner_content.strip()
            values.append(
                DefaultObjectProxy(
                    attrs=DefaultObjectProxy.fromDict(elm.attrib),
                    content=inner_content,
                )
            )
        plugin_comp[tag_type] = values
    if (
        plugin_comp.config
        and len(plugin_comp.config) > 0
        and plugin_comp.config[0].attrs.lang == "yaml"
    ):
        config = yaml.safe_load(plugin_comp.config[0].content)
    elif (
        plugin_comp.config
        and len(plugin_comp.config) > 0
        and plugin_comp.config[0].attrs.lang == "json"
    ):
        config = json.loads(plugin_comp.config[0].content)
    elif plugin_comp.config and len(plugin_comp.config) > 0:
        raise Exception(
            "Unsupported config language: " + plugin_comp.config[0].attrs.lang
        )
    else:
        # No config section found, return empty config
        config = {}

    overwrite_config = overwrite_config or {}
    config["tag"] = overwrite_config.get("tag") or (
        config.get("tags") and config.get("tags")[0]
    )
    config["hot_reloading"] = overwrite_config.get("hot_reloading")
    config["scripts"] = []
    # try to match the script with current tag
    for elm in plugin_comp.script:
        if elm.attrs.tag == config["tag"]:
            config["script"] = elm.content
        # exclude script with mismatched tag
        if not elm.attrs.tag or elm.attrs.tag == config["tag"]:
            config["scripts"].append(elm)
    if not config.get("script") and len(plugin_comp.script) > 0:
        config["script"] = plugin_comp.script[0].content
        config["lang"] = plugin_comp.script[0].attrs.lang
    config["links"] = plugin_comp.link or None
    config["windows"] = plugin_comp.window or None
    config["styles"] = plugin_comp.style or None
    config["docs"] = plugin_comp.docs[0] if plugin_comp.docs else config.get("docs")
    config["attachments"] = plugin_comp.attachment or None

    # config["_id"] = overwrite_config.get("_id") or config.get("name").replace(" ", "_")
    config["uri"] = overwrite_config.get("uri")
    config["origin"] = overwrite_config.get("origin")
    config["namespace"] = overwrite_config.get("namespace")
    config["code"] = source
    # config["id"] = (
    #     config.get("name").strip().replace(" ", "_") + "_" + str(uuid.uuid4())
    # )
    # config["runnable"] = config.get("runnable", True)
    config["requirements"] = config.get("requirements") or []

    for field in CONFIGURABLE_FIELDS:
        obj = config.get(field)
        if obj and isinstance(obj, dict) and not isinstance(obj, list):
            if config.get("tag"):
                config[field] = obj.get(config.get("tag"))
                if not obj.get(config.get("tag")):
                    print(
                        "WARNING: "
                        + field
                        + " do not contain a tag named: "
                        + config.get("tag")
                    )
            else:
                raise Exception("You must use 'tags' with configurable fields.")
    config["lang"] = config.get("lang") or "javascript"
    return config


def convert_config_to_artifact(plugin_config, plugin_id, source_url=None):
    """Convert imjoy plugin config to Artifact format."""
    artifact = DefaultObjectProxy(
        {
            "type": None,
            "id": plugin_id,
        }
    )

    fields = [
        field
        for field in ApplicationManifest.model_fields.keys()
        if field not in ["id", "config"]
    ]
    for field in fields:
        if field in plugin_config:
            artifact[field] = plugin_config[field]
    if source_url:
        artifact["source"] = source_url
    tags = plugin_config.get("labels", []) + plugin_config.get("flags", [])
    artifact["tags"] = tags

    # Store the original config as a nested dict to preserve app type info
    artifact["config"] = plugin_config

    docs = plugin_config.get("docs")
    if docs:
        artifact["documentation"] = docs.get("content")
    artifact["covers"] = plugin_config.get("cover")
    # make sure we have a list
    if not artifact["covers"]:
        artifact["covers"] = []
    elif not isinstance(artifact["covers"], list):
        artifact["covers"] = [artifact["covers"]]

    artifact["badges"] = plugin_config.get("badge")
    if not artifact["badges"]:
        artifact["badges"] = []
    elif not isinstance(artifact["badges"], list):
        artifact["badges"] = [artifact["badges"]]

    artifact["authors"] = plugin_config.get("author")
    if not artifact["authors"]:
        artifact["authors"] = []
    elif not isinstance(artifact["authors"], list):
        artifact["authors"] = [artifact["authors"]]

    artifact["attachments"] = {}
    return artifact


def extract_files_from_source(source):
    """Extract files from Hypha XML source format.

    Converts <manifest> <file> tags to individual files.

    Returns:
        tuple: (extracted_files, remaining_source)
        - extracted_files: List of file dictionaries with 'name', 'content', 'type'
        - remaining_source: Source with extracted tags removed
    """

    # Check if it's likely raw HTML content that shouldn't be parsed as ImJoy/Hypha XML
    source_lower = source.lower().strip()
    if source_lower.startswith(("<!doctype html", "<html")):
        # Don't try to parse raw HTML as XML - just return as-is
        return [], source

    try:
        root = etree.HTML("<html>" + source + "</html>")
    except Exception as e:
        # If XML parsing fails, return original source as-is
        return [], source

    extracted_files = []

    # Parse all tag types
    plugin_comp = DefaultObjectProxy()
    for tag_type in tag_types:
        elms = root.xpath(f".//{tag_type}")
        values = []
        for elm in elms:
            # For file tags, get complete inner content including nested elements
            if tag_type == "file":
                # Get the complete inner HTML content
                inner_content = "".join(elm.itertext()) if elm.text or list(elm) else elm.text
                # Clean up the content by removing the outer tag wrapper
                if inner_content:
                    inner_content = inner_content.strip()
            else:
                # For other tags, use the text content directly
                inner_content = elm.text
                
            values.append(
                DefaultObjectProxy(
                    attrs=DefaultObjectProxy.fromDict(elm.attrib),
                    content=inner_content,
                )
            )
        plugin_comp[tag_type] = values

    # Extract manifest files from both <manifest> and <config> tags
    for manifest_elm in plugin_comp.manifest or []:
        lang = manifest_elm.attrs.get("lang", "json").lower()
        content = manifest_elm.content

        if lang == "yaml":
            filename = "config.yaml"
            # Validate YAML syntax
            try:
                content = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise Exception(f"Invalid YAML in config: {e}")
        elif lang == "json":
            filename = "config.json"
            # Validate JSON syntax
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON in config: {e}")
        else:
            raise Exception(f"Unsupported config language: {lang}")

        extracted_files.append(
            {"name": "manifest.json", "content": content, "type": "json"}
        )

    # Also extract config files from <config> tags (ImJoy format)
    for config_elm in plugin_comp.config or []:
        lang = config_elm.attrs.get("lang", "json").lower()
        content = config_elm.content

        if lang == "yaml":
            # Validate YAML syntax
            try:
                content = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise Exception(f"Invalid YAML in config: {e}")
        elif lang == "json":
            # Validate JSON syntax
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON in config: {e}")
        else:
            raise Exception(f"Unsupported config language: {lang}")

        extracted_files.append(
            {"name": "manifest.json", "content": content, "type": "json"}
        )

    # Extract file elements
    for file_elm in plugin_comp.file or []:
        name = file_elm.attrs.get("name")
        format = file_elm.attrs.get("format", "text").lower()
        content = file_elm.content
        
        # Clean up content by stripping leading/trailing whitespace for better file content
        if content:
            content = content.strip()
        
        extracted_files.append({"name": name, "content": content, "type": format})

    # Extract script elements as files (without the script tags)
    for script_elm in plugin_comp.script or []:
        lang = script_elm.attrs.get("lang", "javascript").lower()
        content = script_elm.content
        if content and content.strip():
            # Clean up content by stripping leading/trailing whitespace
            content = content.strip()
            
            # Determine file extension based on language
            if lang == "python":
                file_ext = "py"
            elif lang in ["javascript", "js"]:
                file_ext = "js"
            else:
                file_ext = "txt"
            
            # Use a default filename if not specified
            script_name = f"main.{file_ext}"
            extracted_files.append({
                "name": script_name, 
                "content": content, 
                "type": "text",
                "source_type": "script"  # Mark this as coming from a script tag
            })

    # Create remaining source by removing extracted tags
    remaining_source = source

    # Remove manifest tags
    for manifest_elm in plugin_comp.manifest or []:
        lang = manifest_elm.attrs.get("lang", "json")
        content = manifest_elm.content or ""
        # Create regex pattern to match the tag
        pattern = rf'<manifest[^>]*lang=["\']{re.escape(lang)}["\'][^>]*>.*?</manifest>'
        remaining_source = re.sub(
            pattern, "", remaining_source, flags=re.DOTALL | re.IGNORECASE
        )

        # Also try without lang attribute
        pattern = rf"<manifest[^>]*>.*?{re.escape(content)}.*?</manifest>"
        remaining_source = re.sub(
            pattern, "", remaining_source, flags=re.DOTALL | re.IGNORECASE
        )

    # Remove config tags (ImJoy format)
    for config_elm in plugin_comp.config or []:
        lang = config_elm.attrs.get("lang", "json")
        content = config_elm.content or ""
        # Create regex pattern to match the tag
        pattern = rf'<config[^>]*lang=["\']{re.escape(lang)}["\'][^>]*>.*?</config>'
        remaining_source = re.sub(
            pattern, "", remaining_source, flags=re.DOTALL | re.IGNORECASE
        )

        # Also try without lang attribute
        pattern = rf"<config[^>]*>.*?{re.escape(content)}.*?</config>"
        remaining_source = re.sub(
            pattern, "", remaining_source, flags=re.DOTALL | re.IGNORECASE
        )

    # Remove file tags
    for file_elm in plugin_comp.file or []:
        name = file_elm.attrs.get("name", "")
        content = file_elm.content or ""
        # Create regex pattern to match the tag
        pattern = rf'<file[^>]*name=["\']{re.escape(name)}["\'][^>]*>.*?</file>'
        remaining_source = re.sub(
            pattern, "", remaining_source, flags=re.DOTALL | re.IGNORECASE
        )

    # Remove script tags
    for script_elm in plugin_comp.script or []:
        lang = script_elm.attrs.get("lang", "javascript")
        content = script_elm.content or ""
        if content.strip():
            # Create regex pattern to match the script tag
            pattern = rf'<script[^>]*lang=["\']{re.escape(lang)}["\'][^>]*>.*?</script>'
            remaining_source = re.sub(
                pattern, "", remaining_source, flags=re.DOTALL | re.IGNORECASE
            )
            
            # Also try without lang attribute or with different quote styles
            pattern = rf'<script[^>]*>.*?{re.escape(content)}.*?</script>'
            remaining_source = re.sub(
                pattern, "", remaining_source, flags=re.DOTALL | re.IGNORECASE
            )

    # Clean up extra whitespace
    remaining_source = re.sub(r"\n\s*\n", "\n", remaining_source.strip())

    return extracted_files, remaining_source
