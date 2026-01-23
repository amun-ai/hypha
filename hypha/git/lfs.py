"""Git LFS (Large File Storage) HTTP API implementation.

This module implements the Git LFS Batch API specification for handling
large file uploads and downloads via S3 presigned URLs.

Git LFS Protocol:
1. Client sends Batch API request to /{workspace}/git/{alias}.git/info/lfs/objects/batch
2. Server responds with presigned S3 URLs for upload/download
3. Client uploads/downloads directly to S3 (streaming, no server memory usage)
4. Optionally, client calls verify endpoint after upload

Uses S3 client factory pattern for reliable async operations:
Each S3 operation creates a fresh client context to avoid connection
pool exhaustion and hanging issues with aiobotocore.

Reference: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md
"""

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Git LFS content types
LFS_CONTENT_TYPE = "application/vnd.git-lfs+json"

# Default URL expiration time (1 hour)
DEFAULT_EXPIRES_IN = 3600

# LFS pointer version URLs
LFS_VERSION_URLS = [
    "https://git-lfs.github.com/spec/v1",
    "https://hawser.github.com/spec/v1",  # Pre-release version
]

# Maximum size for LFS pointer files (per spec: 1024 bytes)
LFS_POINTER_MAX_SIZE = 1024


class LFSPointer:
    """Parsed Git LFS pointer file."""

    def __init__(self, oid: str, size: int, version: str = "https://git-lfs.github.com/spec/v1"):
        self.oid = oid
        self.size = size
        self.version = version

    @classmethod
    def parse(cls, content: bytes) -> Optional["LFSPointer"]:
        """Parse an LFS pointer from file content.

        Args:
            content: Raw file content (bytes)

        Returns:
            LFSPointer if content is a valid LFS pointer, None otherwise
        """
        # LFS pointers must be less than 1024 bytes
        if len(content) > LFS_POINTER_MAX_SIZE:
            return None

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return None

        lines = text.strip().split("\n")
        if not lines:
            return None

        # Parse key-value pairs
        data = {}
        for line in lines:
            parts = line.split(" ", 1)
            if len(parts) != 2:
                return None
            key, value = parts
            data[key] = value

        # Check required fields
        if "version" not in data:
            return None

        # Verify version URL
        if data["version"] not in LFS_VERSION_URLS:
            return None

        if "oid" not in data or "size" not in data:
            return None

        # Parse OID (format: sha256:hexdigest)
        oid_parts = data["oid"].split(":", 1)
        if len(oid_parts) != 2 or oid_parts[0] != "sha256":
            return None

        oid = oid_parts[1]
        if not all(c in "0123456789abcdef" for c in oid) or len(oid) != 64:
            return None

        try:
            size = int(data["size"])
        except ValueError:
            return None

        return cls(oid=oid, size=size, version=data["version"])

    def get_s3_path(self, base_path: str) -> str:
        """Get the S3 path for this LFS object.

        Args:
            base_path: Base path in S3 for the repository (e.g., workspace/artifact)

        Returns:
            Full S3 key for the LFS object
        """
        # LFS objects are stored at: {base_path}/git/lfs/objects/{oid[0:2]}/{oid[2:4]}/{oid}
        return f"{base_path}/git/lfs/objects/{self.oid[0:2]}/{self.oid[2:4]}/{self.oid}"


def is_lfs_pointer(content: bytes) -> bool:
    """Check if content looks like an LFS pointer.

    This is a quick check that doesn't fully parse the pointer.

    Args:
        content: File content

    Returns:
        True if content appears to be an LFS pointer
    """
    if len(content) > LFS_POINTER_MAX_SIZE:
        return False

    try:
        text = content.decode("utf-8")
        return text.startswith("version https://git-lfs.github.com/spec/v1") or \
               text.startswith("version https://hawser.github.com/spec/v1")
    except UnicodeDecodeError:
        return False


class LFSObjectRequest(BaseModel):
    """A single object in an LFS batch request."""

    oid: str = Field(..., description="The object ID (SHA-256 hash)")
    size: int = Field(..., ge=0, description="Size in bytes")


class LFSRef(BaseModel):
    """Git reference information."""

    name: str = Field(..., description="Fully-qualified Git ref (e.g., refs/heads/main)")


class LFSBatchRequest(BaseModel):
    """Git LFS Batch API request body."""

    operation: str = Field(..., description="Either 'download' or 'upload'")
    objects: List[LFSObjectRequest] = Field(..., description="List of objects")
    transfers: Optional[List[str]] = Field(
        default=["basic"], description="Transfer adapters (default: basic)"
    )
    ref: Optional[LFSRef] = Field(default=None, description="Git ref context")
    hash_algo: Optional[str] = Field(
        default="sha256", description="Hash algorithm (default: sha256)"
    )


class LFSAction(BaseModel):
    """An action (upload/download/verify) for an LFS object."""

    href: str = Field(..., description="URL for the action")
    header: Optional[Dict[str, str]] = Field(
        default=None, description="HTTP headers to include"
    )
    expires_in: Optional[int] = Field(
        default=None, description="Seconds until URL expires"
    )


class LFSObjectResponse(BaseModel):
    """Response for a single LFS object in batch response."""

    oid: str
    size: int
    authenticated: Optional[bool] = True
    actions: Optional[Dict[str, LFSAction]] = None
    error: Optional[Dict[str, Any]] = None


class LFSBatchResponse(BaseModel):
    """Git LFS Batch API response body."""

    transfer: str = "basic"
    objects: List[LFSObjectResponse]
    hash_algo: str = "sha256"


class LFSVerifyRequest(BaseModel):
    """Git LFS verify request body."""

    oid: str = Field(..., description="Object ID to verify")
    size: int = Field(..., ge=0, description="Expected size in bytes")


class GitLFSHandler:
    """Handler for Git LFS operations.

    This handler manages LFS objects stored in S3. LFS objects are stored
    in the repository's git/lfs/objects directory with the path structure:
    {workspace}/{artifact_prefix}/git/lfs/objects/{oid[0:2]}/{oid[2:4]}/{oid}

    Uses S3 client factory pattern for reliable async operations:
    Each S3 operation creates a fresh client context to avoid connection
    pool exhaustion and hanging issues with aiobotocore.
    """

    def __init__(
        self,
        s3_client_factory: Callable,
        bucket: str,
        base_path: str,
        public_base_url: str,
        enable_s3_proxy: bool = False,
    ):
        """Initialize LFS handler.

        Args:
            s3_client_factory: Factory function that returns an async context manager
                               for S3 client
            bucket: S3 bucket name
            base_path: Base path in S3 for this repository (e.g., workspace/artifact)
            public_base_url: Public base URL for the Hypha server
            enable_s3_proxy: Whether to use S3 proxy URLs
        """
        self.s3_client_factory = s3_client_factory
        self.bucket = bucket
        self.base_path = base_path
        self.public_base_url = public_base_url
        self.enable_s3_proxy = enable_s3_proxy

    def _get_object_path(self, oid: str) -> str:
        """Get the S3 path for an LFS object.

        Objects are stored in a sharded directory structure:
        {base_path}/git/lfs/objects/{oid[0:2]}/{oid[2:4]}/{oid}
        """
        return f"{self.base_path}/git/lfs/objects/{oid[0:2]}/{oid[2:4]}/{oid}"

    async def object_exists(self, oid: str, expected_size: int = None) -> bool:
        """Check if an LFS object exists in S3.

        Args:
            oid: Object ID (SHA-256 hash)
            expected_size: If provided, also verify the size matches

        Returns:
            True if object exists (and size matches if provided)
        """
        path = self._get_object_path(oid)
        async with self.s3_client_factory() as s3_client:
            try:
                response = await s3_client.head_object(Bucket=self.bucket, Key=path)
                if expected_size is not None:
                    return response["ContentLength"] == expected_size
                return True
            except Exception:
                return False

    async def get_object_size(self, oid: str) -> Optional[int]:
        """Get the size of an LFS object.

        Returns:
            Size in bytes, or None if object doesn't exist
        """
        path = self._get_object_path(oid)
        async with self.s3_client_factory() as s3_client:
            try:
                response = await s3_client.head_object(Bucket=self.bucket, Key=path)
                return response["ContentLength"]
            except Exception:
                return None

    async def generate_download_url(self, oid: str, expires_in: int = DEFAULT_EXPIRES_IN) -> str:
        """Generate a presigned URL for downloading an LFS object."""
        path = self._get_object_path(oid)
        async with self.s3_client_factory() as s3_client:
            url = await s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": path},
                ExpiresIn=expires_in,
            )
        if self.enable_s3_proxy:
            url = f"{self.public_base_url}/s3/{self.bucket}/{path}?{url.split('?')[1]}"
        return url

    async def generate_upload_url(self, oid: str, size: int, expires_in: int = DEFAULT_EXPIRES_IN) -> str:
        """Generate a presigned URL for uploading an LFS object."""
        path = self._get_object_path(oid)
        async with self.s3_client_factory() as s3_client:
            url = await s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": path,
                    "ContentLength": size,
                    "ContentType": "application/octet-stream",
                },
                ExpiresIn=expires_in,
            )
        if self.enable_s3_proxy:
            url = f"{self.public_base_url}/s3/{self.bucket}/{path}?{url.split('?')[1]}"
        return url

    async def upload_object(self, oid: str, body: bytes) -> None:
        """Upload an LFS object directly to S3.

        Args:
            oid: Object ID (SHA-256 hash)
            body: Object content
        """
        path = self._get_object_path(oid)
        async with self.s3_client_factory() as s3_client:
            await s3_client.put_object(
                Bucket=self.bucket,
                Key=path,
                Body=body,
                ContentType="application/octet-stream",
            )

    async def handle_batch_request(
        self,
        request: LFSBatchRequest,
        verify_url_base: str,
        expires_in: int = DEFAULT_EXPIRES_IN,
        authorization: Optional[str] = None,
    ) -> LFSBatchResponse:
        """Handle a Git LFS Batch API request.

        Args:
            request: The batch request
            verify_url_base: Base URL for verify endpoints
            expires_in: URL expiration time in seconds
            authorization: Authorization header to include in verify actions

        Returns:
            LFSBatchResponse with actions for each object
        """
        objects = []

        for obj in request.objects:
            try:
                if request.operation == "download":
                    response = await self._handle_download(obj, expires_in)
                elif request.operation == "upload":
                    response = await self._handle_upload(
                        obj, verify_url_base, expires_in, authorization
                    )
                else:
                    response = LFSObjectResponse(
                        oid=obj.oid,
                        size=obj.size,
                        error={"code": 422, "message": f"Unknown operation: {request.operation}"},
                    )
                objects.append(response)
            except Exception as e:
                logger.error(f"Error handling LFS object {obj.oid}: {e}", exc_info=True)
                objects.append(
                    LFSObjectResponse(
                        oid=obj.oid,
                        size=obj.size,
                        error={"code": 500, "message": str(e)},
                    )
                )

        return LFSBatchResponse(
            transfer="basic",
            objects=objects,
            hash_algo=request.hash_algo or "sha256",
        )

    async def _handle_download(
        self, obj: LFSObjectRequest, expires_in: int
    ) -> LFSObjectResponse:
        """Handle download request for a single object."""
        # Check if object exists
        exists = await self.object_exists(obj.oid)
        if not exists:
            return LFSObjectResponse(
                oid=obj.oid,
                size=obj.size,
                error={"code": 404, "message": "Object not found"},
            )

        # Generate download URL
        download_url = await self.generate_download_url(obj.oid, expires_in)

        return LFSObjectResponse(
            oid=obj.oid,
            size=obj.size,
            authenticated=True,
            actions={
                "download": LFSAction(
                    href=download_url,
                    expires_in=expires_in,
                )
            },
        )

    async def _handle_upload(
        self,
        obj: LFSObjectRequest,
        verify_url_base: str,
        expires_in: int,
        authorization: Optional[str] = None,
    ) -> LFSObjectResponse:
        """Handle upload request for a single object."""
        # Check if object already exists with correct size
        if await self.object_exists(obj.oid, obj.size):
            # Object already exists, no need to upload
            logger.info(f"LFS object {obj.oid} already exists, skipping upload")
            return LFSObjectResponse(
                oid=obj.oid,
                size=obj.size,
                authenticated=True,
                # No actions = object already exists
            )

        # Generate upload URL
        upload_url = await self.generate_upload_url(obj.oid, obj.size, expires_in)

        # Generate verify URL
        verify_url = f"{verify_url_base}/{obj.oid}"

        # Build verify headers - include authorization if provided
        verify_headers = {}
        if authorization:
            verify_headers["Authorization"] = authorization

        return LFSObjectResponse(
            oid=obj.oid,
            size=obj.size,
            authenticated=True,
            actions={
                "upload": LFSAction(
                    href=upload_url,
                    header={"Content-Type": "application/octet-stream"},
                    expires_in=expires_in,
                ),
                "verify": LFSAction(
                    href=verify_url,
                    header=verify_headers if verify_headers else None,
                    expires_in=expires_in,
                ),
            },
        )


def create_lfs_router(
    get_lfs_handler_callback: Callable,
    login_optional_dep,
    login_required_dep,
    parse_user_token: Callable = None,
) -> APIRouter:
    """Create a FastAPI router for Git LFS endpoints.

    Args:
        get_lfs_handler_callback: Async function (workspace, alias, user_info, write) -> GitLFSHandler
        login_optional_dep: FastAPI dependency for optional authentication
        login_required_dep: FastAPI dependency for required authentication
        parse_user_token: Async function to parse token string into UserInfo

    Returns:
        FastAPI router with Git LFS endpoints
    """
    router = APIRouter()

    async def lfs_auth(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        """Handle Git LFS authentication.

        Git LFS uses the same Basic Auth as Git itself.
        Username must be 'git', token is the password.
        """
        from hypha.git.http import extract_credentials_from_basic_auth

        logger.debug(f"LFS auth request: method={request.method}, path={request.url.path}")

        if not authorization:
            return None

        provided_username, token = extract_credentials_from_basic_auth(authorization)

        # Validate username is 'git'
        if provided_username is not None and provided_username != "git":
            logger.warning(f"LFS auth: invalid username '{provided_username}', expected 'git'")
            raise HTTPException(
                status_code=401,
                detail="Username must be 'git'. Use 'git' as username and your token as password.",
                headers={
                    "WWW-Authenticate": 'Basic realm="Git LFS"',
                    "LFS-Authenticate": 'Basic realm="Git LFS"',
                },
            )

        if not token:
            return None

        if parse_user_token:
            try:
                user_info = await parse_user_token(token)
                if user_info.scope.current_workspace is None:
                    user_info.scope.current_workspace = user_info.get_workspace()
                return user_info
            except Exception as e:
                logger.warning(f"LFS auth failed: {e}")
                return None

        return None

    async def lfs_auth_required(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        """Require authentication for LFS operations."""
        user_info = await lfs_auth(request, authorization)
        if not user_info:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={
                    "WWW-Authenticate": 'Basic realm="Git LFS"',
                    "LFS-Authenticate": 'Basic realm="Git LFS"',
                },
            )
        return user_info

    @router.post("/{workspace}/git/{alias}/info/lfs/objects/batch")
    async def lfs_batch(
        workspace: str,
        alias: str,
        request: Request,
        authorization: Optional[str] = Header(None),
        user_info=Depends(lfs_auth),
    ):
        """Git LFS Batch API endpoint.

        This is the main entry point for Git LFS operations.
        """
        # Strip .git suffix if present (Git LFS clients may add it)
        if alias.endswith(".git"):
            alias = alias[:-4]

        # Validate content type
        content_type = request.headers.get("content-type", "")
        if "application/vnd.git-lfs+json" not in content_type:
            logger.warning(f"LFS batch: invalid content-type: {content_type}")
            # Be lenient - some clients don't set the correct content type
            pass

        # Parse request body
        try:
            body = await request.json()
            batch_request = LFSBatchRequest(**body)
        except Exception as e:
            logger.error(f"LFS batch: failed to parse request: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request body: {e}",
            )

        logger.info(
            f"LFS batch: workspace={workspace}, alias={alias}, "
            f"operation={batch_request.operation}, objects={len(batch_request.objects)}"
        )

        # Check authentication for uploads
        write_required = batch_request.operation == "upload"
        if write_required and not user_info:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for upload",
                headers={
                    "WWW-Authenticate": 'Basic realm="Git LFS"',
                    "LFS-Authenticate": 'Basic realm="Git LFS"',
                },
            )

        try:
            handler = await get_lfs_handler_callback(
                workspace, alias, user_info, write=write_required
            )
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except KeyError:
            raise HTTPException(status_code=404, detail="Repository not found")

        # Build verify URL base
        verify_url_base = str(request.url).rsplit("/objects/batch", 1)[0] + "/objects/verify"

        # Handle the batch request
        # Pass authorization header so it can be included in verify action
        response = await handler.handle_batch_request(
            batch_request,
            verify_url_base=verify_url_base,
            expires_in=DEFAULT_EXPIRES_IN,
            authorization=authorization,
        )

        return JSONResponse(
            content=response.model_dump(exclude_none=True),
            media_type=LFS_CONTENT_TYPE,
        )

    @router.post("/{workspace}/git/{alias}/info/lfs/objects/verify/{oid}")
    async def lfs_verify(
        workspace: str,
        alias: str,
        oid: str,
        request: Request,
        user_info=Depends(lfs_auth_required),
    ):
        """Git LFS verify endpoint.

        Called by the client after a successful upload to confirm the object exists.
        """
        # Strip .git suffix if present (Git LFS clients may add it)
        if alias.endswith(".git"):
            alias = alias[:-4]

        # Parse request body
        try:
            body = await request.json()
            verify_request = LFSVerifyRequest(**body)
        except Exception as e:
            logger.error(f"LFS verify: failed to parse request: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid request body: {e}",
            )

        # Validate OID matches URL
        if verify_request.oid != oid:
            raise HTTPException(
                status_code=422,
                detail=f"OID mismatch: URL has {oid}, body has {verify_request.oid}",
            )

        logger.info(f"LFS verify: workspace={workspace}, alias={alias}, oid={oid}")

        try:
            handler = await get_lfs_handler_callback(workspace, alias, user_info, write=False)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except KeyError:
            raise HTTPException(status_code=404, detail="Repository not found")

        # Check if object exists with correct size
        if not await handler.object_exists(oid, verify_request.size):
            actual_size = await handler.get_object_size(oid)
            if actual_size is None:
                raise HTTPException(status_code=404, detail="Object not found")
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Size mismatch: expected {verify_request.size}, got {actual_size}",
                )

        return Response(status_code=200)

    @router.get("/{workspace}/git/{alias}/info/lfs/objects/{oid}")
    async def lfs_download_object(
        workspace: str,
        alias: str,
        oid: str,
        user_info=Depends(lfs_auth),
    ):
        """Direct download endpoint for LFS objects.

        This is a fallback endpoint that redirects to S3.
        Most clients will use the presigned URLs from the batch API.
        """
        # Strip .git suffix if present (Git LFS clients may add it)
        if alias.endswith(".git"):
            alias = alias[:-4]

        logger.info(f"LFS download: workspace={workspace}, alias={alias}, oid={oid}")

        try:
            handler = await get_lfs_handler_callback(workspace, alias, user_info, write=False)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except KeyError:
            raise HTTPException(status_code=404, detail="Repository not found")

        if not await handler.object_exists(oid):
            raise HTTPException(status_code=404, detail="Object not found")

        download_url = await handler.generate_download_url(oid)
        return Response(
            status_code=307,
            headers={"Location": download_url},
        )

    @router.put("/{workspace}/git/{alias}/info/lfs/objects/{oid}")
    async def lfs_upload_object(
        workspace: str,
        alias: str,
        oid: str,
        request: Request,
        user_info=Depends(lfs_auth_required),
    ):
        """Direct upload endpoint for LFS objects.

        This is a fallback endpoint that streams to S3.
        Most clients will use the presigned URLs from the batch API.

        WARNING: This endpoint loads the entire body into memory.
        For large files, clients should use the presigned URL approach.
        """
        # Strip .git suffix if present (Git LFS clients may add it)
        if alias.endswith(".git"):
            alias = alias[:-4]

        logger.info(f"LFS upload: workspace={workspace}, alias={alias}, oid={oid}")

        content_length = request.headers.get("content-length")
        if not content_length:
            raise HTTPException(status_code=411, detail="Content-Length required")

        size = int(content_length)

        try:
            handler = await get_lfs_handler_callback(workspace, alias, user_info, write=True)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except KeyError:
            raise HTTPException(status_code=404, detail="Repository not found")

        # Check if object already exists
        if await handler.object_exists(oid, size):
            return Response(status_code=200)

        # Stream upload to S3
        # Note: For very large files, this could be optimized with multipart upload
        body = await request.body()

        # Verify hash matches OID
        computed_hash = hashlib.sha256(body).hexdigest()
        if computed_hash != oid:
            raise HTTPException(
                status_code=422,
                detail=f"Hash mismatch: expected {oid}, got {computed_hash}",
            )

        # Upload to S3 using the handler method
        await handler.upload_object(oid, body)

        logger.info(f"LFS upload complete: oid={oid}, size={size}")
        return Response(status_code=200)

    return router
