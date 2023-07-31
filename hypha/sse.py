"""Provide the server-side event proxy."""
import asyncio
import base64
import logging
import sys

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from imjoy_rpc.hypha import RPC
from sse_starlette.sse import EventSourceResponse

from hypha.core import ClientInfo
from hypha.core.auth import login_optional
from hypha.core.store import RedisStore
from hypha.utils import GzipRoute

parent_client = None

from hypha.core.store import RedisRPCConnection

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("sse")
logger.setLevel(logging.INFO)


class SSERPCConnection:
    def __init__(self, redis_connection, user_info, workspace, on_disconnected=None):
        self.queue = asyncio.Queue()
        self.redis_connection = redis_connection
        self.user_info = user_info
        self.workspace = workspace
        self.on_disconnected = on_disconnected

    async def event_stream(self):
        try:
            while True:
                data = await self.queue.get()
                self.queue.task_done()
                data_base64 = base64.b64encode(data).decode("utf-8")
                yield dict(data=data_base64)
        except asyncio.CancelledError as e:
            if self.on_disconnected:
                await self.on_disconnected()
            else:
                logger.info("Client disconnected")


class SSEProxy:
    """A proxy for accessing services from Server-Side Events."""

    def __init__(self, store: RedisStore) -> None:
        """Initialize the http proxy."""
        # pylint: disable=broad-except
        router = APIRouter()
        router.route_class = GzipRoute
        self.store = store
        sessions = {}

        @router.get("/sse")
        @router.post("/sse")
        async def sse(
            request: Request,
            session_id: str = Query(None),
            workspace: str = Query(None),
            client_id: str = Query(None),
            reconnection_token: str = Query(None),
            user_info: login_optional = Depends(login_optional),
        ):
            """Route for server-side events."""
            if session_id is None:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "detail": f"Session id is required",
                    },
                )
            if request.method == "POST":
                if session_id not in sessions:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "success": False,
                            "detail": f"Session not found: {session_id}",
                        },
                    )
                sse = sessions[session_id]
                if workspace and workspace != sse.workspace:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": f"Permission denied to session: {session_id}",
                        },
                    )
                data = await request.body()
                await sse.redis_connection.emit_message(data)
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                    },
                )

            assert request.method == "GET"
            if workspace is None:
                workspace = user_info.id
                # Anonymous and Temporary users are not allowed to create persistant workspaces
                persistent = (
                    not user_info.is_anonymous and "temporary" not in user_info.roles
                )
                # If the user disconnected unexpectedly, the workspace will be preserved
                if not await store.get_user_workspace(user_info.id):
                    try:
                        await store.register_workspace(
                            dict(
                                name=user_info.id,
                                owners=[user_info.id],
                                visibility="protected",
                                persistent=persistent,
                                read_only=user_info.is_anonymous,
                            ),
                            overwrite=False,
                        )
                    except Exception as exp:
                        logger.error("Failed to create user workspace: %s", exp)
                        return JSONResponse(
                            status_code=500,
                            content={
                                "success": False,
                                "detail": f"Failed to create user workspace: {exp}",
                            },
                        )
            try:
                workspace_manager = await store.get_workspace_manager(
                    workspace, setup=True
                )
            except Exception as exp:
                logger.error(
                    "Failed to get workspace manager %s, error: %s", workspace, exp
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "detail": f"Failed to get workspace manager {workspace}, error: {exp}",
                    },
                )
            if not await workspace_manager.check_permission(user_info):
                logger.error(
                    "Permission denied (workspace: %s,"
                    " user: %s, client: %s, scopes: %s)",
                    workspace,
                    user_info.id,
                    client_id,
                    user_info.scopes,
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "detail": f"Permission denied (workspace: {workspace} user: {user_info.id}, client: {client_id}, scopes: {user_info.scopes})",
                    },
                )

            if not reconnection_token and await workspace_manager.check_client_exists(
                client_id
            ):
                logger.error(
                    "Another client with the same id %s"
                    " already connected to workspace: %s",
                    client_id,
                    workspace,
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "detail": f"Another client with the same id {client_id}  already connected to workspace: {workspace}",
                    },
                )

            try:
                manager = await store.get_workspace_manager(workspace)
                # TODO: secure the sessions
                redis_connection = RedisRPCConnection(
                    manager._redis,
                    manager._workspace,
                    client_id,
                    user_info,
                )

                async def on_sse_disconnected():
                    try:
                        await workspace_manager.delete_client(client_id)
                    except KeyError:
                        logger.info(
                            "Client already deleted: %s/%s",
                            workspace_manager._workspace,
                            client_id,
                        )
                    try:
                        # Clean up if the client is disconnected normally
                        await workspace_manager.delete()
                    except KeyError:
                        logger.info("Workspace already deleted: %s", workspace)

                sse = SSERPCConnection(
                    redis_connection, user_info, workspace, on_sse_disconnected
                )
                # await sse.queue.put(b'ssdfsdf')
                redis_connection.on_message(sse.queue.put)
                if not reconnection_token:
                    await workspace_manager.register_client(
                        ClientInfo(
                            id=client_id,
                            parent=parent_client,
                            workspace=workspace_manager._workspace,
                            user_info=user_info,
                        )
                    )

                sessions[session_id] = sse
                return EventSourceResponse(sse.event_stream())
            except Exception as exp:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": str(exp)},
                )

        store.register_router(router)
