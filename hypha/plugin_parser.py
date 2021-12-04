import json
import uuid

import yaml
from lxml import etree

from hypha.utils import dotdict

tag_types = ["config", "script", "link", "window", "style", "docs", "attachment"]

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
    """Parse ImJoy plugin file and return a dict with all the fields."""
    root = etree.HTML("<html>" + source + "</html>")
    pluginComp = dotdict()
    for tag_type in tag_types:
        elms = root.xpath(f".//{tag_type}")
        values = []
        for elm in elms:
            values.append(
                dotdict(
                    attrs=dotdict(elm.attrib),
                    content=elm.text,
                )
            )
        pluginComp[tag_type] = values
    if pluginComp.config[0].attrs.lang == "yaml":
        config = yaml.safe_load(pluginComp.config[0].content)
    elif pluginComp.config[0].attrs.lang == "json":
        config = json.loads(pluginComp.config[0].content)
    else:
        raise Exception(
            "Unsupported config language: " + pluginComp.config[0].attrs.lang
        )

    overwrite_config = overwrite_config or {}
    config["tag"] = overwrite_config.get("tag") or (
        config.get("tags") and config.get("tags")[0]
    )
    config["hot_reloading"] = overwrite_config.get("hot_reloading")
    config["scripts"] = []
    # try to match the script with current tag
    for i in range(len(pluginComp.script)):
        if pluginComp.script[i].attrs.tag == config["tag"]:
            config["script"] = pluginComp.script[i].content
        # exclude script with mismatched tag
        if (
            not pluginComp.script[i].attrs.tag
            or pluginComp.script[i].attrs.tag == config["tag"]
        ):
            config["scripts"].append(pluginComp.script[i])
    if not config.get("script") and len(pluginComp.script) > 0:
        config["script"] = pluginComp.script[0].content
        config["lang"] = pluginComp.script[0].attrs.lang
    config["links"] = pluginComp.link or None
    config["windows"] = pluginComp.window or None
    config["styles"] = pluginComp.style or None
    config["docs"] = pluginComp.docs and pluginComp.docs[0] or config.get("docs")
    config["attachments"] = pluginComp.attachment or None

    config["_id"] = overwrite_config.get("_id") or config.get("name").replace(" ", "_")
    config["uri"] = overwrite_config.get("uri")
    config["origin"] = overwrite_config.get("origin")
    config["namespace"] = overwrite_config.get("namespace")
    config["code"] = source
    config["id"] = (
        config.get("name").strip().replace(" ", "_") + "_" + str(uuid.uuid4())
    )
    config["runnable"] = config.get("runnable", True)
    config["requirements"] = config.get("requirements") or []

    for i in range(len(CONFIGURABLE_FIELDS)):
        obj = config.get(CONFIGURABLE_FIELDS[i])
        if obj and type(obj) == dict and not isinstance(obj, list):
            if config.get("tag"):
                config[CONFIGURABLE_FIELDS[i]] = obj.get(config.get("tag"))
                if not obj.get(config.get("tag")):
                    print(
                        "WARNING: "
                        + CONFIGURABLE_FIELDS[i]
                        + " do not contain a tag named: "
                        + config.get("tag")
                    )
            else:
                raise "You must use 'tags' with configurable fields."
    config["lang"] = config.get("lang") or "javascript"
    return config
