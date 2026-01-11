"""Provide utilities that should not be aware of hypha."""

import copy
import asyncio
import gzip
import os
import posixpath
import secrets
import string
import time
from datetime import datetime
from typing import Callable, List, Optional, Dict
import shortuuid
import friendlywords as fw
import logging

import httpx
from fastapi.routing import APIRoute
from starlette.datastructures import MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

_os_alt_seps: List[str] = list(
    sep for sep in [os.path.sep, os.path.altsep] if sep is not None and sep != "/"
)


def random_id(readable=True):
    """Generate a random ID."""
    if readable:
        return (
            fw.generate("po", separator="-")
            + "-"
            + f"{int((time.time() % 1) * 100000000):08d}"
        )
    else:
        return shortuuid.ShortUUID().random(length=22)


PLUGIN_CONFIG_FIELDS = [
    "name",
    "source_hash",
    "version",
    "format_version",
    "type",
    "tags",
    "icon",
    "requirements",
    "env",
    "defaults",
    "flags",
    "inputs",
    "outputs",
    "dependencies",
    "app_id",
    "required_artifact_files",
]


class EventBus:
    """An event bus class for handling and listening to events asynchronously."""

    def __init__(self, logger=None):
        """Initialize the event bus."""
        self._callbacks = {}
        self._logger = logger

    def on(self, event_name, func):
        """Register an event callback."""
        self._callbacks[event_name] = self._callbacks.get(event_name, []) + [func]
        return func

    def once(self, event_name, func):
        """Register an event callback that only runs once."""

        def once_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self.off(event_name, once_wrapper)
            return result

        return self.on(event_name, once_wrapper)

    def off(self, event_name, func=None):
        """Remove an event callback."""
        if func:
            if event_name in self._callbacks and func in self._callbacks[event_name]:
                self._callbacks[event_name].remove(func)
                if not self._callbacks[event_name]:
                    del self._callbacks[event_name]
        else:
            self._callbacks.pop(event_name, None)

    def emit(self, event_name, data):
        """Trigger an event and return a task that completes when all handlers are done."""
        tasks = []
        for func in self._callbacks.get(event_name, []):
            try:
                result = func(data)
                if asyncio.iscoroutine(result):
                    tasks.append(result)
            except Exception as e:
                if self._logger:
                    self._logger.error(
                        "Error in event callback: %s, %s, error: %s",
                        event_name,
                        func,
                        e,
                    )

        if tasks:
            return asyncio.ensure_future(asyncio.gather(*tasks))
        else:
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut

    async def wait_for(self, event_name, match=None, timeout=None):
        """Wait for a specific event with an optional timeout and conditions."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def handler(args):
            if self._matches(match, args):
                if not future.done():
                    future.set_result(args)

        self.on(event_name, handler)
        try:
            result = await asyncio.wait_for(future, timeout)
            return result
        except asyncio.TimeoutError:
            raise
        finally:
            self.off(event_name, handler)

    def _matches(self, match, data):
        """Check if the event data matches the given criteria."""
        if not match:
            return True  # No match criteria provided, always match
        if not data or len(data) < 1:
            return False  # No data to match against
        return all(data.get(key) == value for key, value in match.items())


def generate_password(length=20):
    """Generate a password."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for i in range(length))


def is_safe_path(basedir: str, path: str, follow_symlinks: bool = True) -> bool:
    """Check if the file path is safe."""
    # resolves symbolic links
    if follow_symlinks:
        matchpath = os.path.realpath(path)
    else:
        matchpath = os.path.abspath(path)
    return basedir == os.path.commonpath((basedir, matchpath))


def safe_join(directory: str, *pathnames: str) -> Optional[str]:
    """Safely join zero or more untrusted path components to a base directory.

    This avoids escaping the base directory.
    :param directory: The trusted base directory.
    :param pathnames: The untrusted path components relative to the
        base directory.
    :return: A safe path, otherwise ``None``.

    This function is copied from:
    https://github.com/pallets/werkzeug/blob/fb7ddd89ae3072e4f4002701a643eb247a402b64/src/werkzeug/security.py#L222
    """
    parts = [directory]

    for filename in pathnames:
        if filename != "":
            filename = posixpath.normpath(filename)

        if (
            any(sep in filename for sep in _os_alt_seps)
            or os.path.isabs(filename)
            or filename == ".."
            or filename.startswith("../")
        ):
            raise Exception(
                f"Illegal file path: `{filename}`, "
                "you can only operate within the work directory."
            )

        parts.append(filename)

    return posixpath.join(*parts)


def parse_s3_list_response(response, delimiter):
    """Parse the s3 list object response."""
    if response.get("KeyCount") == 0:
        return []
    items = [
        {
            "type": "file",
            "name": item["Key"].split("/")[-1] if delimiter == "/" else item["Key"],
            "size": item["Size"],
            "last_modified": datetime.timestamp(item["LastModified"]),
        }
        for item in response.get("Contents", [])
    ]
    # only include when delimiter is /
    if delimiter == "/":
        items += [
            {"type": "directory", "name": item["Prefix"].rstrip("/").split("/")[-1]}
            for item in response.get("CommonPrefixes", [])
        ]
    return items


def list_objects_sync(s3_client, bucket, prefix=None, delimiter="/"):
    """List a objects sync."""
    prefix = prefix or ""
    response = s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, Delimiter=delimiter
    )

    items = parse_s3_list_response(response, delimiter)
    while response["IsTruncated"]:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter=delimiter,
            ContinuationToken=response["NextContinuationToken"],
        )
        items += parse_s3_list_response(response, delimiter)
    return items


def remove_objects_sync(s3_client, bucket, prefix, delimiter=""):
    """Remove all objects in a folder."""
    assert prefix != "" and prefix.endswith("/")
    response = s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, Delimiter=delimiter
    )
    items = response.get("Contents", [])
    if len(items) > 0:
        delete_response = s3_client.delete_objects(
            Bucket=bucket,
            Delete={
                "Objects": [
                    {
                        "Key": item["Key"],
                        # 'VersionId': 'string'
                    }
                    for item in items
                ],
                "Quiet": True,
            },
        )
        assert (
            "ResponseMetadata" in delete_response
            and delete_response["ResponseMetadata"]["HTTPStatusCode"] == 200
        )
    while response["IsTruncated"]:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter=delimiter,
            ContinuationToken=response["NextContinuationToken"],
        )
        items = response.get("Contents", [])
        if len(items) > 0:
            delete_response = s3_client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [
                        {
                            "Key": item["Key"],
                            # 'VersionId': 'string'
                        }
                        for item in items
                    ],
                    "Quiet": True,
                },
            )
            assert (
                "ResponseMetadata" in delete_response
                and delete_response["ResponseMetadata"]["HTTPStatusCode"] == 200
            )


async def list_objects_async(
    s3_client,
    bucket: str,
    prefix: Optional[str] = None,
    delimiter: str = "/",
    max_length: Optional[int] = None,
    offset: int = 0,
):
    """List objects async with pagination support.
    
    Args:
        s3_client: The S3 client instance
        bucket: S3 bucket name
        prefix: Object key prefix to filter by
        delimiter: Delimiter for grouping keys (default "/")
        max_length: Maximum number of items to return (page size)
        offset: Number of items to skip before returning results
        
    Returns:
        List of items (files/directories) after applying offset and limit
    """
    prefix = prefix or ""
    response = await s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, Delimiter=delimiter
    )
    all_items = parse_s3_list_response(response, delimiter)
    
    # Continue fetching if we need more items to satisfy offset + max_length
    while response["IsTruncated"]:
        # If we have enough items (offset + max_length), we can stop early
        if max_length and len(all_items) >= offset + max_length:
            break
            
        response = await s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter=delimiter,
            ContinuationToken=response["NextContinuationToken"],
        )
        all_items += parse_s3_list_response(response, delimiter)
    
    # Apply offset and limit
    start_index = offset
    if max_length:
        end_index = offset + max_length
        return all_items[start_index:end_index]
    else:
        return all_items[start_index:]


async def remove_objects_async(s3_client, bucket, prefix, delimiter=""):
    """Remove all objects in a folder asynchronously."""
    assert prefix != "" and prefix.endswith("/")
    response = await s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, Delimiter=delimiter
    )
    items = response.get("Contents", [])
    if len(items) > 0:
        delete_response = await s3_client.delete_objects(
            Bucket=bucket,
            Delete={
                "Objects": [
                    {
                        "Key": item["Key"],
                        # 'VersionId': 'string'
                    }
                    for item in items
                ],
                "Quiet": True,
            },
        )
        assert (
            "ResponseMetadata" in delete_response
            and delete_response["ResponseMetadata"]["HTTPStatusCode"] == 200
        )
    while response["IsTruncated"]:
        response = await s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter=delimiter,
            ContinuationToken=response["NextContinuationToken"],
        )
        items = response.get("Contents", [])
        if len(items) > 0:
            delete_response = await s3_client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [
                        {
                            "Key": item["Key"],
                            # 'VersionId': 'string'
                        }
                        for item in items
                    ],
                    "Quiet": True,
                },
            )
            assert (
                "ResponseMetadata" in delete_response
                and delete_response["ResponseMetadata"]["HTTPStatusCode"] == 200
            )


class GzipRequest(Request):
    """Gzip Request."""

    async def body(self) -> bytes:
        """Get the body."""
        if not hasattr(self, "_body"):
            body = await super().body()
            if "gzip" in self.headers.getlist("Content-Encoding"):
                body = gzip.decompress(body)
            # pylint: disable=attribute-defined-outside-init
            self._body = body
        return self._body


class GzipRoute(APIRoute):
    """Gzip Route."""

    def get_route_handler(self) -> Callable:
        """Get route handler."""
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            request = GzipRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler


class PatchedCORSMiddleware(CORSMiddleware):
    """
    A patched version of CORS middleware.

    This is required because the CORS middleware does not
    send explicit origin when allow_credentials is True.
    """

    async def send(self, message, send, request_headers) -> None:
        """Send the message."""
        if message["type"] != "http.response.start":
            await send(message)
            return

        message.setdefault("headers", [])
        headers = MutableHeaders(scope=message)
        headers.update(self.simple_headers)
        # We need to first normalize the case of the headers
        # To avoid multiple values in the headers
        # See issue: https://github.com/encode/starlette/issues/1309
        # pylint: disable=protected-access
        items = headers._list
        # pylint: disable=protected-access
        headers._list = [
            (item[0].decode("latin-1").lower().encode("latin-1").title(), item[1])
            for item in items
        ]
        headers = MutableHeaders(headers=dict(headers.items()))

        origin = request_headers["Origin"]
        has_cookie = "cookie" in request_headers
        allow_credential = (
            "Access-Control-Allow-Credentials" in self.simple_headers
            and self.simple_headers["Access-Control-Allow-Credentials"] == "true"
        )

        # If request includes any cookie headers, then we must respond
        # with the specific origin instead of '*'.
        if self.allow_all_origins and (has_cookie or allow_credential):
            self.allow_explicit_origin(headers, origin)

        # If we only allow specific origins, then we have to mirror back
        # the Origin header in the response.
        elif not self.allow_all_origins and self.is_allowed_origin(origin=origin):
            self.allow_explicit_origin(headers, origin)

        message["headers"] = headers.raw
        await send(message)


async def _example_hypha_startup(server):
    """An example hypha startup module."""
    assert server.register_codec
    assert server.rpc
    assert server.disconnect
    await server.register_service(
        {
            "id": "example-startup-service",
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "test": lambda x: x + 22,
        }
    )


def sanitize_url_for_logging(url: str) -> str:
    """Remove sensitive parameters from URLs for safe logging."""
    import re
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    try:
        # Parse the URL
        parsed = urlparse(str(url))
        if not parsed.query:
            return str(url)  # No query parameters to sanitize

        # Parse query parameters
        params = parse_qs(parsed.query, keep_blank_values=True)

        # Remove sensitive parameters (case insensitive)
        sensitive_keys = {"access_token", "token", "key", "secret", "password"}
        sanitized_params = {
            k: v for k, v in params.items() if k.lower() not in sensitive_keys
        }

        # Rebuild the URL
        new_query = urlencode(sanitized_params, doseq=True)
        sanitized_parsed = parsed._replace(query=new_query)
        return urlunparse(sanitized_parsed)

    except Exception:
        # Fallback to regex if URL parsing fails
        sensitive_params = r"[?&](?:access_token|token|key|secret|password)=[^&]*"
        sanitized = re.sub(sensitive_params, "", str(url), flags=re.IGNORECASE)
        sanitized = re.sub(r"\?&", "?", sanitized)
        sanitized = re.sub(r"&+", "&", sanitized)
        sanitized = re.sub(r"[?&]$", "", sanitized)
        return sanitized
