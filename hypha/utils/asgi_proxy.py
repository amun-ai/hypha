"""
Custom ASGI proxy implementation based on asgiproxy but with fixes for session management issues.

This implementation addresses the problems that cause asgiproxy to become unresponsive:
1. Session reuse leading to connection exhaustion
2. Missing connection limits and timeouts
3. No proper cleanup mechanisms
4. Content-encoding issues for compressed files

Portions of this code are derived from asgiproxy:
The MIT License (MIT)

Copyright (c) 2019 Valohai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import asyncio
import logging
from typing import Optional, Union
from urllib.parse import urljoin, urlparse

import aiohttp
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

Headerlike = Union[dict, Headers]


class ProxyConfig:
    """Base proxy configuration class."""

    def get_upstream_url(self, *, scope: Scope) -> str:
        """Get the upstream URL for a client request."""
        raise NotImplementedError("...")

    def get_upstream_url_with_query(self, *, scope: Scope) -> str:
        """Get the upstream URL for a client request, including query parameters."""
        url = self.get_upstream_url(scope=scope)
        query_string = scope.get("query_string")
        if query_string:
            sep = "&" if "?" in url else "?"
            url += f"{sep}{query_string.decode('utf-8')}"
        return url

    def process_client_headers(self, *, scope: Scope, headers: Headers) -> Headerlike:
        """Process client HTTP headers before they're passed upstream."""
        return headers

    def process_upstream_headers(
        self, *, scope: Scope, proxy_response: aiohttp.ClientResponse
    ) -> Headerlike:
        """Process upstream HTTP headers before they're passed to the client."""
        return proxy_response.headers

    def get_upstream_http_options(
        self, *, scope: Scope, client_request: Request, data
    ) -> dict:
        """Get request options (as passed to aiohttp.ClientSession.request)."""
        return dict(
            method=client_request.method,
            url=self.get_upstream_url_with_query(scope=scope),
            data=data,
            headers=self.process_client_headers(
                scope=scope,
                headers=client_request.headers,
            ),
            allow_redirects=False,
        )


class BaseURLProxyConfigMixin:
    """Mixin for proxy configs that proxy to a base URL."""

    upstream_base_url: str
    rewrite_host_header: Optional[str] = None

    def get_upstream_url(self, scope: Scope) -> str:
        return urljoin(self.upstream_base_url, scope["path"])

    def process_client_headers(
        self, *, scope: Scope, headers: Headerlike
    ) -> Headerlike:
        """Process client HTTP headers before they're passed upstream."""
        if self.rewrite_host_header:
            headers = headers.mutablecopy()  # type: ignore
            headers["host"] = self.rewrite_host_header
        return super().process_client_headers(scope=scope, headers=headers)  # type: ignore


class ProxyContext:
    """
    Improved proxy context that creates fresh sessions to avoid connection issues.

    Unlike the original asgiproxy, this doesn't reuse sessions which can lead to
    connection exhaustion and unresponsiveness.
    """

    def __init__(
        self,
        config: ProxyConfig,
        max_concurrency: int = 20,
    ) -> None:
        self.config = config
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def create_session(self) -> aiohttp.ClientSession:
        """Create a fresh session with proper connection management."""
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=60,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=30,  # Socket read timeout
        )

        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            cookie_jar=aiohttp.DummyCookieJar(),
            auto_decompress=False,  # We'll handle content-encoding manually
        )

    async def __aenter__(self) -> "ProxyContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # No session to close since we create fresh ones per request
        pass

    async def close(self) -> None:
        # No persistent session to close
        pass


def _should_remove_gzip_encoding(content_type: str, file_path: str) -> bool:
    """
    Determine if gzip content-encoding should be removed for already-compressed content.

    Uses a simple heuristic: if the content type indicates binary/compressed data
    or the file extension suggests a compressed format, remove gzip encoding.
    """
    import mimetypes
    from pathlib import Path

    # Normalize inputs
    content_type = (content_type or "").lower().split(";")[0].strip()
    file_extension = Path(file_path).suffix.lower()

    # Simple heuristic: remove gzip for binary content types that are typically compressed
    if content_type.startswith(("image/", "video/", "audio/", "application/")):
        # Allow gzip for text-based application types
        if content_type.startswith(
            ("application/json", "application/xml", "application/javascript")
        ):
            return False
        return True

    # Check common compressed file extensions
    compressed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".mp4",
        ".avi",
        ".mov",
        ".mp3",
        ".wav",
        ".pdf",
        ".zip",
        ".rar",
        ".gz",
        ".bz2",
        ".7z",
    }
    if file_extension in compressed_extensions:
        return True

    # Use mimetypes as fallback
    guessed_type, _ = mimetypes.guess_type(file_path)
    if guessed_type and not guessed_type.startswith("text/"):
        return True

    return False


# Streaming thresholds
INCOMING_STREAMING_THRESHOLD = 512 * 1024  # 512KB
OUTGOING_STREAMING_THRESHOLD = 1024 * 1024 * 5  # 5MB


def determine_incoming_streaming(request: Request) -> bool:
    """Determine if we should stream the incoming request."""
    if request.method in ("GET", "HEAD"):
        return False

    try:
        return int(request.headers["content-length"]) > INCOMING_STREAMING_THRESHOLD
    except (TypeError, ValueError, KeyError):
        # Malformed or missing content-length header; assume a very large payload
        return True


def determine_outgoing_streaming(proxy_response: aiohttp.ClientResponse) -> bool:
    """Determine if we should stream the outgoing response."""
    if proxy_response.status != 200:
        return False
    try:
        return (
            int(proxy_response.headers["content-length"]) > OUTGOING_STREAMING_THRESHOLD
        )
    except (TypeError, ValueError, KeyError):
        # Malformed or missing content-length header; assume a streaming payload
        return True


async def read_stream_in_chunks(stream, chunk_size: int = 64 * 1024):
    """Read stream in chunks."""
    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            break
        yield chunk


async def get_proxy_response(
    *,
    context: ProxyContext,
    scope: Scope,
    receive: Receive,
) -> tuple[aiohttp.ClientResponse, aiohttp.ClientSession]:
    """Get the proxy response from upstream, returning both response and session."""
    request = Request(scope, receive)
    should_stream_incoming = determine_incoming_streaming(request)

    async with context.semaphore:
        data = None
        if request.method not in ("GET", "HEAD"):
            if should_stream_incoming:
                data = request.stream()
            else:
                data = await request.body()

        kwargs = context.config.get_upstream_http_options(
            scope=scope, client_request=request, data=data
        )

        # Create fresh session for this request
        session = context.create_session()
        try:
            response = await session.request(**kwargs)
            return response, session
        except Exception:
            # Clean up session on error
            await session.close()
            raise


async def convert_proxy_response_to_user_response(
    *,
    context: ProxyContext,
    scope: Scope,
    proxy_response: aiohttp.ClientResponse,
    session: aiohttp.ClientSession,
) -> Response:
    """Convert the proxy response to a user response."""
    try:
        # Process headers
        headers_to_client = dict(
            context.config.process_upstream_headers(
                scope=scope, proxy_response=proxy_response
            )
        )

        # Remove headers that shouldn't be forwarded
        for header in ["connection", "transfer-encoding"]:
            headers_to_client.pop(header, None)

        # Handle content-encoding for compressed files
        original_encoding = proxy_response.headers.get("content-encoding", "").lower()
        path = scope.get("path", "")

        if original_encoding == "gzip" and _should_remove_gzip_encoding(
            proxy_response.headers.get("content-type", ""), path
        ):
            logger.debug(f"Removing incorrect gzip encoding for {path}")
            headers_to_client.pop("content-encoding", None)
        elif original_encoding:
            # Preserve other encodings
            headers_to_client["content-encoding"] = original_encoding

        status_to_client = proxy_response.status

        if determine_outgoing_streaming(proxy_response):
            # Stream large responses
            async def stream_content():
                try:
                    async for chunk in read_stream_in_chunks(proxy_response.content):
                        if chunk:
                            yield chunk
                except Exception as e:
                    logger.warning(f"Streaming error for {path}: {e}")
                finally:
                    # Clean up session after streaming
                    try:
                        await session.close()
                    except Exception as e:
                        logger.warning(f"Error closing session: {e}")

            return StreamingResponse(
                content=stream_content(),
                status_code=status_to_client,
                headers=headers_to_client,
                media_type=headers_to_client.get(
                    "content-type", "application/octet-stream"
                ),
            )
        else:
            # Buffer small responses
            try:
                content = await proxy_response.read()
                return Response(
                    content=content,
                    status_code=status_to_client,
                    headers=headers_to_client,
                    media_type=headers_to_client.get(
                        "content-type", "application/octet-stream"
                    ),
                )
            finally:
                # Clean up session after reading
                await session.close()

    except Exception:
        # Clean up session on any error
        try:
            await session.close()
        except Exception:
            pass
        raise


async def proxy_http(
    *,
    context: ProxyContext,
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    """Proxy HTTP requests."""
    try:
        proxy_response, session = await get_proxy_response(
            context=context, scope=scope, receive=receive
        )

        user_response = await convert_proxy_response_to_user_response(
            context=context, scope=scope, proxy_response=proxy_response, session=session
        )
        return await user_response(scope, receive, send)

    except asyncio.TimeoutError:
        logger.error(f"Proxy timeout for {scope.get('path', 'unknown')}")
        response = Response(
            content="Gateway Timeout",
            status_code=504,
            headers={"Content-Type": "text/plain"},
        )
        return await response(scope, receive, send)
    except aiohttp.ClientError as e:
        logger.error(f"Proxy client error for {scope.get('path', 'unknown')}: {e}")
        response = Response(
            content="Bad Gateway",
            status_code=502,
            headers={"Content-Type": "text/plain"},
        )
        return await response(scope, receive, send)
    except Exception as e:
        logger.error(f"Proxy error for {scope.get('path', 'unknown')}: {e}")
        response = Response(
            content="Internal Server Error",
            status_code=500,
            headers={"Content-Type": "text/plain"},
        )
        return await response(scope, receive, send)


def make_simple_proxy_app(
    proxy_context: ProxyContext,
    *,
    proxy_http_handler=proxy_http,
    proxy_websocket_handler=None,  # We don't need websocket support for S3
) -> ASGIApp:
    """
    Create a simple ASGI proxy application.

    This is compatible with the asgiproxy API but uses improved session management.
    """

    async def app(scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "lifespan":
            return None  # We explicitly do nothing here for this simple app.

        if scope["type"] == "http" and proxy_http_handler:
            return await proxy_http_handler(
                context=proxy_context, scope=scope, receive=receive, send=send
            )

        if scope["type"] == "websocket" and proxy_websocket_handler:
            return await proxy_websocket_handler(
                context=proxy_context, scope=scope, receive=receive, send=send
            )

        raise NotImplementedError(f"Scope {scope} is not understood")

    return app
