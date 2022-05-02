"""Provide main entrypoint."""
import asyncio
import inspect
import json
import logging
import os
import re
import sys
import urllib.request
from types import ModuleType

import aiofiles
import yaml
from imjoy_rpc.hypha.websocket_client import connect_to_server

from hypha.utils import dotdict

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("plugin-runner")
logger.setLevel(logging.INFO)


async def export_service(plugin_api, config, imjoy_rpc):
    try:
        wm = await connect_to_server(config)
        imjoy_rpc.api.update(wm)  # make the api available to the plugin
        rpc = wm.rpc
        if not isinstance(plugin_api, dict) and inspect.isclass(type(plugin_api)):
            plugin_api = {a: getattr(plugin_api, a) for a in dir(plugin_api)}
        # Copy the plugin name as the default name
        plugin_api["id"] = "default"
        plugin_api["name"] = config.get("name", "default")
        await rpc.register_service(plugin_api, overwrite=True)
        svc = await rpc.get_remote_service(rpc._client_id + ":default")
        if svc.setup:
            await svc.setup()
    except Exception as exp:
        logger.exception(exp)
        loop = asyncio.get_event_loop()
        loop.stop()
        sys.exit(1)


async def patch_imjoy_rpc(default_config):
    def export(api, config=None):
        default_config.update(config or {})
        imjoy_rpc.ready = asyncio.ensure_future(
            export_service(api, default_config, imjoy_rpc)
        )

    # create a fake imjoy_rpc to patch hypha rpc
    imjoy_rpc = ModuleType("imjoy_rpc")
    sys.modules["imjoy_rpc"] = imjoy_rpc
    imjoy_rpc.api = dotdict(export=export)
    return imjoy_rpc


async def run_plugin(plugin_file, default_config, quit_on_ready=False):
    """Load plugin file."""
    loop = asyncio.get_event_loop()
    if os.path.isfile(plugin_file):
        async with aiofiles.open(plugin_file, "r", encoding="utf-8") as fil:
            content = await fil.read()
    elif plugin_file.startswith("http"):
        with urllib.request.urlopen(plugin_file) as response:
            content = response.read().decode("utf-8")
        # remove query string
        plugin_file = plugin_file.split("?")[0]
    else:
        raise Exception(f"Invalid input plugin file path: {plugin_file}")

    if plugin_file.endswith(".py"):
        filename, _ = os.path.splitext(os.path.basename(plugin_file))
        default_config["name"] = filename[:32]
        imjoy_rpc = await patch_imjoy_rpc(default_config)
        exec(content, globals())  # pylint: disable=exec-used
        logger.info("Plugin executed")
        if quit_on_ready:
            imjoy_rpc.ready.add_done_callback(lambda fut: loop.stop())

    elif plugin_file.endswith(".imjoy.html"):
        # load config
        found = re.findall("<config (.*)>\n(.*)</config>", content, re.DOTALL)[0]
        if "json" in found[0]:
            plugin_config = json.loads(found[1])
        elif "yaml" in found[0]:
            plugin_config = yaml.safe_load(found[1])
        default_config.update(plugin_config)
        imjoy_rpc = await patch_imjoy_rpc(default_config)
        # load script
        found = re.findall("<script (.*)>\n(.*)</script>", content, re.DOTALL)[0]
        if "python" in found[0]:
            exec(found[1], globals())  # pylint: disable=exec-used
            logger.info("Plugin executed")
            if quit_on_ready:
                imjoy_rpc.ready.add_done_callback(lambda fut: loop.stop())
        else:
            raise RuntimeError(
                f"Invalid script type ({found[0]}) in file {plugin_file}"
            )
    else:
        raise RuntimeError(f"Invalid script file type ({plugin_file})")


async def start(args):
    """Run the plugin."""
    try:
        default_config = {
            "server_url": args.server_url,
            "workspace": args.workspace,
            "token": args.token,
        }
        await run_plugin(args.file, default_config, quit_on_ready=args.quit_on_ready)
    except Exception:  # pylint: disable=broad-except
        logger.exception("Failed to run plugin.")
        loop = asyncio.get_event_loop()
        loop.stop()
        sys.exit(1)


def start_runner(args):
    """Start the plugin runner."""
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(start(args))
    loop.run_forever()
