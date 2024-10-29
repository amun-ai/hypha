"""Provide main entrypoint."""
import asyncio
import inspect
import json
import logging
import os
import re
import sys
import urllib.request

import aiofiles
import yaml
from hypha_rpc.utils import ObjectProxy
from hypha_rpc import connect_to_server


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("app-runner")
logger.setLevel(logging.INFO)


async def export_service(app_api, config, hypha_rpc):
    try:
        wm = await connect_to_server(config)
        hypha_rpc.api.update(wm)  # make the api available to the app
        rpc = wm.rpc
        if not isinstance(app_api, dict) and inspect.isclass(type(app_api)):
            app_api = {a: getattr(app_api, a) for a in dir(app_api)}
        # Copy the app name as the default name
        app_api["id"] = "default"
        app_api["name"] = config.get("name", "default")
        svc = await rpc.register_service(app_api, {"overwrite": True, "notify": True})
        svc = await rpc.get_remote_service(svc["id"])
        if svc.setup:
            await svc.setup()
    except Exception as exp:
        logger.exception(exp)
        loop = asyncio.get_event_loop()
        loop.stop()
        sys.exit(1)


async def patch_hypha_rpc(default_config):
    import hypha_rpc

    def export(api, config=None):
        default_config.update(config or {})
        hypha_rpc.ready = asyncio.ensure_future(
            export_service(api, default_config, hypha_rpc)
        )

    hypha_rpc.api = ObjectProxy(export=export)
    return hypha_rpc


async def run_app(app_file, default_config, quit_on_ready=False):
    """Load app file."""
    loop = asyncio.get_event_loop()
    if os.path.isfile(app_file):
        async with aiofiles.open(app_file, "r", encoding="utf-8") as fil:
            content = await fil.read()
    elif app_file.startswith("http"):
        with urllib.request.urlopen(app_file) as response:
            content = response.read().decode("utf-8")
        # remove query string
        app_file = app_file.split("?")[0]
    else:
        raise Exception(f"Invalid input app file path: {app_file}")

    if app_file.endswith(".py"):
        filename, _ = os.path.splitext(os.path.basename(app_file))
        default_config["name"] = filename[:32]
        hypha_rpc = await patch_hypha_rpc(default_config)
        exec(content, globals())  # pylint: disable=exec-used
        logger.info("app executed")

        if quit_on_ready:

            def done_callback(fut):
                if fut.done():
                    if fut.exception():
                        logger.error(fut.exception())
                loop.stop()

            hypha_rpc.ready.add_done_callback(done_callback)

    elif app_file.endswith(".imjoy.html"):
        # load config
        found = re.findall("<config (.*)>\n(.*)</config>", content, re.DOTALL)[0]
        if "json" in found[0]:
            app_config = json.loads(found[1])
        elif "yaml" in found[0]:
            app_config = yaml.safe_load(found[1])
        default_config.update(app_config)
        hypha_rpc = await patch_hypha_rpc(default_config)
        # load script
        found = re.findall("<script (.*)>\n(.*)</script>", content, re.DOTALL)[0]
        if "python" in found[0]:
            exec(found[1], globals())  # pylint: disable=exec-used
            logger.info("app executed")
            if quit_on_ready:
                hypha_rpc.ready.add_done_callback(lambda fut: loop.stop())
        else:
            raise RuntimeError(f"Invalid script type ({found[0]}) in file {app_file}")
    else:
        raise RuntimeError(f"Invalid script file type ({app_file})")


async def start(args):
    """Run the app."""
    try:
        default_config = {
            "server_url": args.server_url,
            "workspace": args.workspace,
            "token": args.token,
        }
        await run_app(args.file, default_config, quit_on_ready=args.quit_on_ready)
    except Exception:  # pylint: disable=broad-except
        logger.exception("Failed to run app, exiting.")
        loop = asyncio.get_event_loop()
        loop.stop()
        sys.exit(1)


def start_runner(args):
    """Start the app runner."""
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(start(args))
    loop.run_forever()
