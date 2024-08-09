"""Provide the triton proxy."""
import random
from typing import Any, List

import httpx
from fastapi import APIRouter, Depends, Request, Response
from pyotritonclient import execute, get_config

from hypha.core.store import RedisStore


class TritonProxy:
    """A proxy for accessing triton inference servers."""

    def __init__(
        self, store: RedisStore, triton_servers: str, allow_origins: str
    ) -> None:
        """Initialize the triton proxy."""
        # pylint: disable=broad-except
        router = APIRouter()
        self.store = store
        self.servers = list(filter(lambda x: x.strip(), triton_servers))
        self.servers = list(map(lambda x: x.rstrip("/"), self.servers))

        @router.get("/triton/{path:path}")
        @router.post("/triton/{path:path}")
        async def triton_proxy(
            path: str,
            request: Request,
            response: Response,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Route for listing all the models."""
            headers = dict(request.headers.items())
            # with the host header, the server will return 404
            del headers["host"]
            params = request.query_params.multi_items()
            server = random.choice(self.servers)
            url = f"{server}/{path}"
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    if request.method == "GET":
                        proxy = await client.get(url, params=params, headers=headers)
                        response.headers.update(proxy.headers)
                        response.body = proxy.content
                        response.status_code = proxy.status_code
                        return response

                    if request.method == "POST":

                        async def request_streamer():
                            async for chunk in request.stream():
                                yield chunk

                        # Use a stream to access raw request body (with compression)
                        async with client.stream(
                            "POST",
                            url,
                            data=request_streamer(),
                            params=params,
                            headers=headers,
                        ) as proxy:
                            response.headers.update(proxy.headers)
                            body = b""
                            async for chunk in proxy.aiter_raw():
                                body += chunk
                            response.body = body
                            response.status_code = proxy.status_code
                            return response
                except httpx.RequestError as exc:
                    response.status_code = 500
                    response.body = (
                        "An error occurred while " + f"requesting {exc.request.url!r}."
                    ).encode("utf-8")
                    return response

        store.register_router(router)
        store.register_public_service(self.get_triton_service())

    async def execute(
        self, model_name: str, inputs: List[Any], server_url=None, **kwargs
    ):
        """Execute the triton inference."""
        if server_url is None:
            server_url = random.choice(self.servers)
        results = await execute(
            inputs,
            server_url=server_url,
            model_name=model_name,
            cache_config=False,
            **kwargs,
        )
        return results

    async def get_config(self, model_name: str, server_url=None, **kwargs):
        """Return the config for the triton model."""
        if server_url is None:
            server_url = random.choice(self.servers)
        return await get_config(server_url=server_url, model_name=model_name, **kwargs)

    def get_triton_service(self):
        """Return the triton service."""
        return {
            "id": "triton-client",
            "name": "Triton Client",
            "description": "Triton Inference Server Client",
            "config": {"visibility": "public"},
            "execute": self.execute,
            "get_config": self.get_config,
        }
