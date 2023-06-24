import asyncio
import logging
import os
import sys

import httpx
from simpervisor import SupervisedProcess

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("services")
logger.setLevel(logging.INFO)
import importlib.util


def _load_function(module_path, function_name):
    spec = importlib.util.spec_from_file_location(
        f"config_function_{function_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    return function


async def _http_ready_func(url, p):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(url)
            # We only care if we get back *any* response, not just 200
            # If there's an error response, that can be shown directly to the user
            logger.debug(f"Got code {resp.status} back from {url}")
            return True
        except httpx.RequestError as exc:
            logger.debug(f"Connection to {url} failed: {exc}")
            return False


async def load_services(store: any, services_config: str, set_ready: callable):
    """Load external services."""
    services = services_config["services"]
    count = 0
    server = store.get_public_workspace_interface()
    for service in services:
        if "command" in service:
            cmd = service["command"]
            workspace = service.get("workspace")

            if workspace and not await store.workspace_exists(workspace):
                ws = await server.create_workspace(
                    dict(
                        name=workspace,
                        owners=[],
                        visibility="protected",
                        persistent=True,
                        read_only=False,
                    ),
                    overwrite=False,
                )
                workspace_name = ws.config["workspace"]
                token = await ws.generate_token()
            else:
                ws = store.get_public_workspace_interface()
                workspace_name = ws.config["workspace"]
                token = await ws.generate_token()
            # format the cmd so we fill in the {server_url} placeholder
            cmd = cmd.format(
                server_url=store.local_base_url, workspace=workspace_name, token=token
            )
            cmd = [c.strip() for c in cmd.split() if c.strip()]
            assert len(cmd) > 0, f"Invalid command: {service['command']}"
            name = service.get("name", cmd[0])
            # split command into list, strip off spaces and filter out empty strings
            server_env = os.environ.copy()
            server_env.update(service.get("env", {}))

            check_services = service.get("check_services")
            check_url = service.get("check_url")

            async def ready_function(p):
                if check_services:
                    for service_id in check_services:
                        try:
                            await ws.get_service(service_id)
                        except Exception as e:
                            logger.info(f"Service {service_id} not ready: {e}")
                            return False
                    return True
                if check_url:
                    return await _http_ready_func(check_url, p)
                return False

            proc = SupervisedProcess(
                name,
                *cmd,
                env=server_env,
                always_restart=False,
                ready_func=ready_function,
                ready_timeout=service.get("timeout", 5),
                log=logger,
            )

            try:
                await proc.start()

                is_ready = await proc.ready()

                if not is_ready:
                    await proc.kill()
                    raise Exception(f"Service {name} failed to start")
            except:
                logger.exception(f"Service {name} failed to start")
                raise
        elif "module" in service and "entrypoint" in service:
            module_path = service["module"]
            entrypoint = service["entrypoint"]
            kwargs = service.get("kwargs", {})
            # load the python module and get the entrypoint
            load_func = _load_function(module_path, entrypoint)
            # make sure the load_func is a coroutine
            assert asyncio.iscoroutinefunction(
                load_func
            ), f"Entrypoint {entrypoint} is not a coroutine"
            await load_func(server, **kwargs)
        else:
            logger.warning(f"Skipping service {service}")
            continue
        count += 1
    logger.info(f"Loaded {count} services")
    set_ready(True)
    return count
