"""Provide an s3 interface."""
import logging
import os
import sys
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from hypha.utils import (
    safe_join,
    list_objects_sync,
)
from hypha.s3 import FSFileResponse

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("card")
logger.setLevel(logging.INFO)


class CardController:
    """Represent a card controller."""

    # pylint: disable=too-many-statements

    def __init__(
        self,
        store,
        s3_controller=None,
        workspace_bucket="hypha-workspaces",
    ):
        """Set up controller."""
        self.event_bus = store.get_event_bus()
        self.s3_controller = s3_controller
        self.store = store
        self.workspace_bucket = workspace_bucket

        self.s3client = self.s3_controller.create_client_sync()

        try:
            self.s3client.create_bucket(Bucket=self.workspace_bucket)
            logger.info("Bucket created: %s", self.workspace_bucket)
        except self.s3client.exceptions.BucketAlreadyExists:
            pass
        except self.s3client.exceptions.BucketAlreadyOwnedByYou:
            pass

        router = APIRouter()

        @router.get("/{workspace}/cards/{path:path}")
        async def get_card_file(
            workspace: str,
            path: str,
            request: Request,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            try:
                path = safe_join(workspace, path)
                return FSFileResponse(
                    self.s3_controller.create_client_async(),
                    self.workspace_bucket,
                    path,
                )
            except ClientError:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"File does not exists: {path}",
                    },
                )

        store.register_router(router)
        store.register_public_service(self.get_card_service())

    def save(self, name, source, context: dict = None):
        """Save a card."""
        ws = context["ws"]
        response = self.s3client.put_object(
            ACL="public-read",
            Body=source,
            Bucket=self.workspace_bucket,
            Key=f"{ws}/{name}",
        )
        assert (
            "ResponseMetadata" in response
            and response["ResponseMetadata"]["HTTPStatusCode"] == 200
        ), f"Failed to deploy app: {name}"
        self.event_bus.emit("card_added", f"{ws}/{name}")

    def remove(self, name, context: dict = None):
        """Remove a card."""
        ws = context["ws"]
        response = self.s3client.delete_object(
            Bucket=self.workspace_bucket,
            Key=f"{ws}/{name}",
        )
        assert (
            "ResponseMetadata" in response
            and response["ResponseMetadata"]["HTTPStatusCode"] == 204
        ), f"Failed to undeploy app: {name}"
        self.event_bus.emit("card_removed", f"{ws}/{name}")

    def list(self, prefix: str = "", context: dict = None):
        """List all the cards."""
        ws = context["ws"]
        prefix = safe_join(ws, prefix)
        items = list_objects_sync(
            self.s3client, self.workspace_bucket, prefix=prefix, delimeter=""
        )
        ret = []
        for item in items:
            if item["type"] == "directory":
                continue
            parts = os.path.split(item["name"])
            ws = parts[0]
            name = "/".join(parts[1:])
            ret.append({"name": name, "url": f"{ws}/cards/{name}"})
        return ret

    def get_card_service(self):
        """Get card controller."""
        return {
            "id": "card",
            "config": {"visibility": "public", "require_context": True},
            "name": "Card",
            "save": self.save,
            "remove": self.remove,
            "list": self.list,
        }
