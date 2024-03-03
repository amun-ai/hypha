"""Provide an s3 interface."""
import logging
import os
import sys
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from hypha.core import UserInfo
from hypha.core.auth import login_optional
from hypha.utils import (
    safe_join,
    list_objects_sync,
)
from hypha.s3 import FSFileResponse

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("rdf")
logger.setLevel(logging.INFO)


class RDFController:
    """Represent an RDF controller."""

    # pylint: disable=too-many-statements

    def __init__(
        self,
        store,
        s3_controller=None,
        rdf_bucket="hypha-apps",
    ):
        """Set up controller."""
        self.event_bus = store.get_event_bus()
        self.s3_controller = s3_controller
        self.store = store
        self.rdf_bucket = rdf_bucket

        self.s3client = self.s3_controller.create_client_sync()

        try:
            self.s3client.create_bucket(Bucket=self.rdf_bucket)
            logger.info("Bucket created: %s", self.rdf_bucket)
        except self.s3client.exceptions.BucketAlreadyExists:
            pass
        except self.s3client.exceptions.BucketAlreadyOwnedByYou:
            pass

        router = APIRouter()

        @router.get("/public/rdfs/{path:path}")
        async def get_app_file(
            path: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            try:
                path = safe_join(path)
                return FSFileResponse(
                    self.s3_controller.create_client_async(), self.rdf_bucket, path
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
        store.register_public_service(self.get_rdf_service())

    def save(self, name, source, context: dict = None):
        """Save an RDF."""
        user_info = UserInfo.model_validate(context["user"])
        response = self.s3client.put_object(
            ACL="public-read",
            Body=source,
            Bucket=self.rdf_bucket,
            Key=f"{user_info.id}/{name}",
        )
        assert (
            "ResponseMetadata" in response
            and response["ResponseMetadata"]["HTTPStatusCode"] == 200
        ), f"Failed to deploy app: {name}"
        self.event_bus.emit("rdf_added", f"{user_info.id}/{name}")

    def remove(self, name, context: dict = None):
        """Remove an RDF."""
        user_info = UserInfo.model_validate(context["user"])
        response = self.s3client.delete_object(
            Bucket=self.rdf_bucket,
            Key=f"{user_info.id}/{name}",
        )
        assert (
            "ResponseMetadata" in response
            and response["ResponseMetadata"]["HTTPStatusCode"] == 204
        ), f"Failed to undeploy app: {name}"
        self.event_bus.emit("rdf_removed", f"{user_info.id}/{name}")

    def list(self, user: str = None, context: dict = None):
        """List all the RDFs."""
        items = list_objects_sync(
            self.s3client, self.rdf_bucket, prefix=user, delimeter=""
        )
        ret = []
        for item in items:
            if item["type"] == "directory":
                continue
            parts = os.path.split(item["name"])
            user = parts[0]
            name = "/".join(parts[1:])
            ret.append(
                {"name": name, "user": user, "url": f"public/rdfs/{user}/{name}"}
            )
        return ret

    def get_rdf_service(self):
        """Get rdf controller."""
        return {
            "id": "rdf",
            "config": {"visibility": "public", "require_context": True},
            "name": "RDF",
            "type": "rdf",
            "save": self.save,
            "remove": self.remove,
            "list": self.list,
        }
