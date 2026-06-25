"""Git Smart HTTP Protocol endpoints for FastAPI.

This module implements the Git Smart HTTP protocol, allowing standard Git
clients to clone, fetch, and push to Hypha artifacts.

Protocol Reference:
- https://git-scm.com/docs/http-protocol
- https://git-scm.com/docs/protocol-v2
"""

import asyncio
import io
import os
import base64
import logging
import tempfile
from typing import AsyncIterator, Callable, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header, Query, Request
from fastapi.responses import Response, StreamingResponse

from dulwich.objects import DEFAULT_OBJECT_FORMAT, ObjectID, ShaFile
from dulwich.object_store import PACK_SPOOL_FILE_MAX_SIZE
from dulwich.pack import (
    Pack,
    PackData,
    UnpackedObject,
    write_pack_data,
    write_pack_objects,
)
from dulwich.protocol import (
    CAPABILITIES_REF,
    COMMAND_HAVE,
    COMMAND_WANT,
    SIDE_BAND_CHANNEL_DATA,
    SIDE_BAND_CHANNEL_PROGRESS,
    SIDE_BAND_CHANNEL_FATAL,
    Protocol,
    extract_capabilities,
    extract_want_line_capabilities,
)

from hypha.git.repo import S3GitRepo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concurrency cap for memory-heavy pack operations
# ---------------------------------------------------------------------------
# Pack generation (clone/fetch) and pack parsing/fattening (push) are the
# memory-heavy git operations. Without a cap, N concurrent large clones each
# spike memory independently and can OOM the server. We bound how many pack
# ops run at once with a lazily-created module-level semaphore (lazy so the
# loop binding the semaphore is the running event loop, not import-time).
HYPHA_GIT_MAX_CONCURRENT_PACK_OPS = int(
    os.environ.get("HYPHA_GIT_MAX_CONCURRENT_PACK_OPS", "4")
)
HYPHA_GIT_PACK_ACQUIRE_TIMEOUT = float(
    os.environ.get("HYPHA_GIT_PACK_ACQUIRE_TIMEOUT", "30")
)
# Below this estimated total object size, the upload-pack response is buffered
# (with Content-Length) for compatibility with libgit2/wasm-git/CORS proxies
# that mishandle chunked transfer encoding. Above it, the pack is streamed.
HYPHA_GIT_STREAM_THRESHOLD = int(
    os.environ.get("HYPHA_GIT_STREAM_THRESHOLD", str(8 * 1024 * 1024))
)

_pack_op_semaphore: Optional[asyncio.Semaphore] = None


def _get_pack_op_semaphore() -> asyncio.Semaphore:
    """Lazily create the module-level pack-op semaphore.

    Created on first use so it binds to the running event loop rather than at
    import time (which may run before any loop exists, or under a different
    loop in tests).
    """
    global _pack_op_semaphore
    if _pack_op_semaphore is None:
        _pack_op_semaphore = asyncio.Semaphore(HYPHA_GIT_MAX_CONCURRENT_PACK_OPS)
    return _pack_op_semaphore


async def _acquire_pack_slot() -> asyncio.Semaphore:
    """Acquire a pack-op slot, raising 503 on timeout.

    Returns the semaphore so the caller can release it in a finally block.
    """
    sem = _get_pack_op_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=HYPHA_GIT_PACK_ACQUIRE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Git server busy",
            headers={"Retry-After": "5"},
        )
    return sem


def _malloc_trim():
    """Return freed heap memory to the OS (glibc only; no-op elsewhere).

    Git clone/push buffers whole packs (request body, the generated pack
    BytesIO, the joined response) — hundreds of MB per op. Python frees those
    promptly, but glibc retains the freed memory in per-thread (secondary)
    arenas and does NOT auto-return it; only an explicit malloc_trim(0) does.
    Under repeated/concurrent large git ops this retention ratchets RSS up to
    OOM (measured on prod: malloc_trim reclaimed ~1-1.6GB). We call this after
    each git request (see cleanup_repo) to release the pack memory immediately.
    On non-glibc platforms (macOS dev, musl) malloc_trim is unavailable -> no-op.
    """
    import platform

    if platform.system() != "Linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=False)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except (OSError, AttributeError) as e:  # pragma: no cover - platform dependent
        logger.debug(f"malloc_trim unavailable: {e}")


async def cleanup_repo(repo):
    """Clean up resources associated with a Git repository.

    This cleans up the S3 client and database session that were created
    when getting the repo from artifact_integration.
    """
    try:
        # Release the loaded pack/index buffers immediately rather than waiting
        # for GC. Each git request builds a fresh per-request object store whose
        # S3Pack._ensure_loaded() reads the entire .pack and .idx into resident
        # bytes (_pack_data/_index_data) — multi-hundred-MB for a large repo.
        # Under concurrent git load these per-request spikes overlap and ratchet
        # RSS up to OOM if freeing is left to GC. object_store.close() clears the
        # _pack_cache and drops those buffers now.
        if getattr(repo, '_object_store', None) is not None:
            try:
                repo._object_store.close()
                logger.debug("Git repo: object store closed (pack buffers released)")
            except Exception as e:
                logger.warning(f"Error closing object store: {e}")

        # Return the freed pack memory to the OS now. close() frees the bytes in
        # Python, but glibc holds them in secondary (per-thread) arenas and won't
        # auto-return them — only an explicit malloc_trim(0) does. Without this,
        # repeated/concurrent large git ops ratchet RSS up to OOM. glibc-only.
        _malloc_trim()

        # Close S3 client if it was stored on the repo
        if hasattr(repo, '_s3_client_context') and repo._s3_client_context:
            try:
                await repo._s3_client_context.__aexit__(None, None, None)
                logger.debug("Git repo: S3 client closed")
            except Exception as e:
                logger.warning(f"Error closing S3 client: {e}")

        # Close database session if it was stored on the repo
        if hasattr(repo, '_session') and repo._session:
            try:
                await repo._session.close()
                logger.debug("Git repo: database session closed")
            except Exception as e:
                logger.warning(f"Error closing database session: {e}")
    except Exception as e:
        logger.error(f"Error during repo cleanup: {e}")


# Git protocol constants
UPLOAD_PACK_CAPABILITIES = [
    b"multi_ack",
    b"multi_ack_detailed",  # Required for smart HTTP (stateless-rpc)
    b"thin-pack",
    b"side-band",
    b"side-band-64k",
    b"ofs-delta",
    b"shallow",
    b"no-progress",
    b"include-tag",
]

RECEIVE_PACK_CAPABILITIES = [
    b"report-status",
    b"delete-refs",
    b"side-band-64k",
    b"ofs-delta",
    b"atomic",
]


def extract_credentials_from_basic_auth(
    authorization: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Extract username and token from HTTP Basic Auth header.

    Git uses HTTP Basic Authentication for push operations.
    The password field is used as the Hypha token.
    The username must be 'git' - authorization is determined solely
    by the token.

    Format: Authorization: Basic base64(git:token)

    Args:
        authorization: The Authorization header value

    Returns:
        Tuple of (username, token). For Bearer auth, username is None.
    """
    if not authorization:
        return None, None

    # Handle both Bearer and Basic auth
    if authorization.lower().startswith("bearer "):
        # Bearer token - extract directly, no username
        return None, authorization[7:].strip()
    elif authorization.lower().startswith("basic "):
        # Basic auth - decode and extract username:password
        encoded = authorization[6:].strip()
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
            if ":" in decoded:
                # Format: username:password - password is the token
                username, password = decoded.split(":", 1)
                return (
                    username if username else None,
                    password if password else None,
                )
        except Exception:
            return None, None
    else:
        # Might be a raw token
        token = authorization.strip() if authorization.strip() else None
        return None, token

    return None, None


def extract_token_from_basic_auth(authorization: Optional[str]) -> Optional[str]:
    """Extract the token from HTTP Basic Auth header.

    This is a backward-compatible wrapper around extract_credentials_from_basic_auth.

    Args:
        authorization: The Authorization header value

    Returns:
        The token (password) if Basic auth is provided, None otherwise
    """
    _, token = extract_credentials_from_basic_auth(authorization)
    return token


def pkt_line(data: bytes) -> bytes:
    """Format data as a pkt-line.

    A pkt-line is a 4-byte hex length prefix followed by the data.
    Length includes the 4-byte prefix itself.
    """
    if data is None:
        return b"0000"  # Flush packet
    length = len(data) + 4
    return f"{length:04x}".encode() + data


def pkt_flush() -> bytes:
    """Return a flush packet (0000)."""
    return b"0000"


class StreamingPktLineReader:
    """Streaming pkt-line reader that doesn't load entire body into memory.

    This class reads from a request stream and yields pkt-lines one at a time,
    with configurable memory limits.
    """

    def __init__(self, request: Request, max_pkt_line_size: int = 65520):
        """Initialize streaming reader.

        Args:
            request: FastAPI Request object
            max_pkt_line_size: Maximum size of a single pkt-line (default 65520, Git's max)
        """
        self._request = request
        self._max_pkt_line_size = max_pkt_line_size
        self._buffer = b""
        self._stream_exhausted = False
        self._stream_iter = None  # Will be initialized on first read

    async def _fill_buffer(self, min_bytes: int) -> bool:
        """Fill buffer with at least min_bytes from stream.

        Returns True if we have enough bytes, False if stream is exhausted.
        """
        logger.debug(f"_fill_buffer: need {min_bytes} bytes, have {len(self._buffer)}, exhausted={self._stream_exhausted}")
        if self._stream_exhausted:
            return len(self._buffer) >= min_bytes

        if len(self._buffer) >= min_bytes:
            return True

        # Initialize stream iterator on first use
        if self._stream_iter is None:
            self._stream_iter = self._request.stream().__aiter__()

        try:
            while len(self._buffer) < min_bytes:
                try:
                    chunk = await self._stream_iter.__anext__()
                    logger.debug(f"_fill_buffer: received chunk of {len(chunk)} bytes")
                    self._buffer += chunk
                except StopAsyncIteration:
                    logger.debug(f"_fill_buffer: stream exhausted, have {len(self._buffer)} bytes")
                    self._stream_exhausted = True
                    break
        except Exception as e:
            logger.error(f"_fill_buffer: error reading stream: {e}")
            raise

        return len(self._buffer) >= min_bytes

    async def read_pkt_line(self) -> tuple[bytes | None, bool]:
        """Read a single pkt-line from the stream.

        Returns:
            Tuple of (data, is_flush):
            - For flush packets: (None, True)
            - For data packets: (data, False)
            - For end of stream: (None, False)
        """
        # Need at least 4 bytes for length prefix
        if not await self._fill_buffer(4):
            return (None, False)  # End of stream

        # Check for PACK signature (end of pkt-lines, start of pack data)
        if self._buffer[:4] == b"PACK":
            return (None, False)  # Signal to caller to handle pack data

        # Parse length
        try:
            length = int(self._buffer[:4], 16)
        except ValueError:
            return (None, False)  # Invalid, end parsing

        if length == 0:
            # Flush packet
            self._buffer = self._buffer[4:]
            return (None, True)

        if length < 4 or length > self._max_pkt_line_size:
            return (None, False)  # Invalid length

        # Need full packet
        if not await self._fill_buffer(length):
            return (None, False)  # Incomplete packet

        # Extract data (excluding length prefix)
        data = self._buffer[4:length]
        self._buffer = self._buffer[length:]
        return (data, False)

    def get_remaining_buffer(self) -> bytes:
        """Get any remaining data in the buffer (e.g., pack data)."""
        return self._buffer

    async def read_remaining_stream(self) -> AsyncIterator[bytes]:
        """Yield remaining buffer and stream data."""
        if self._buffer:
            yield self._buffer
            self._buffer = b""

        if not self._stream_exhausted:
            # Use the stored iterator if available, otherwise create one
            if self._stream_iter is None:
                self._stream_iter = self._request.stream().__aiter__()

            while True:
                try:
                    chunk = await self._stream_iter.__anext__()
                    yield chunk
                except StopAsyncIteration:
                    self._stream_exhausted = True
                    break


async def read_pkt_lines(request: Request) -> AsyncIterator[bytes]:
    """Read pkt-lines from the request body.

    DEPRECATED: This function loads the entire body into memory.
    Use StreamingPktLineReader for large requests.
    """
    body = await request.body()
    offset = 0

    while offset < len(body):
        # Read length prefix (4 hex digits)
        if offset + 4 > len(body):
            break

        try:
            length = int(body[offset : offset + 4], 16)
        except ValueError:
            break

        if length == 0:
            # Flush packet
            yield None
            offset += 4
        elif length < 4:
            # Invalid length
            break
        else:
            # Data packet
            data = body[offset + 4 : offset + length]
            yield data
            offset += length


class GitHTTPHandler:
    """Handler for Git Smart HTTP protocol requests."""

    def __init__(
        self,
        repo: S3GitRepo,
        read_only: bool = False,
    ):
        """Initialize Git HTTP handler.

        Args:
            repo: S3GitRepo instance
            read_only: If True, reject push operations
        """
        self.repo = repo
        self.read_only = read_only

    async def generate_refs_advertisement(
        self, service: bytes
    ) -> AsyncIterator[bytes]:
        """Generate the refs advertisement response for info/refs.

        This is the initial response a Git client receives when discovering
        references via the smart HTTP protocol.
        """
        # First line: service announcement
        yield pkt_line(b"# service=" + service + b"\n")
        yield pkt_flush()

        # Get references
        refs = await self.repo.get_refs_async()

        # Filter out symbolic refs (like HEAD -> refs/heads/main)
        # Git clients don't want these in the refs listing
        real_refs = {
            ref_name: sha
            for ref_name, sha in refs.items()
            if not sha.startswith(b"ref: ")
        }

        # Get symbolic refs for symref capability (e.g., HEAD -> refs/heads/main)
        # This is crucial for git clients to know the default branch when cloning
        symrefs = {
            ref_name: sha[5:]  # Remove "ref: " prefix
            for ref_name, sha in refs.items()
            if sha.startswith(b"ref: ")
        }

        # Resolve HEAD to actual SHA if possible
        # Git clients expect HEAD to be in the refs list with the resolved commit SHA
        head_sha = None
        if b"HEAD" in symrefs:
            target = symrefs[b"HEAD"]
            if target in real_refs:
                head_sha = real_refs[target]

        # Determine capabilities based on service
        if service == b"git-upload-pack":
            caps_list = list(UPLOAD_PACK_CAPABILITIES)
        else:
            caps_list = list(RECEIVE_PACK_CAPABILITIES)

        # Add symref capabilities (e.g., symref=HEAD:refs/heads/main)
        # This tells the client what the default branch is
        for ref_name, target in symrefs.items():
            caps_list.append(b"symref=" + ref_name + b":" + target)

        caps = b" ".join(caps_list)

        if not real_refs:
            # Empty repository (or only symbolic refs) - advertise capabilities with zero-id
            # This is required for Git clients to know our capabilities
            yield pkt_line(b"0" * 40 + b" capabilities^{}\0" + caps + b"\n")
            yield pkt_flush()
            return

        # Build refs list with HEAD first (if resolvable), then sorted refs
        refs_to_send = []
        if head_sha:
            # HEAD with resolved SHA should be first
            refs_to_send.append((b"HEAD", head_sha))

        # Add other refs sorted
        for ref_name, sha in sorted(real_refs.items()):
            refs_to_send.append((ref_name, sha))

        first = True
        for ref_name, sha in refs_to_send:
            line = sha + b" " + ref_name
            if first:
                # First line includes capabilities
                line = line + b"\0" + caps
                first = False

            yield pkt_line(line + b"\n")

        yield pkt_flush()

    async def handle_upload_pack_from_bytes(self, body: bytes) -> AsyncIterator[bytes]:
        """Handle git-upload-pack requests (fetch/clone) from pre-read body.

        Protocol:
        1. Client sends WANT lines (objects it wants)
        2. Client sends HAVE lines (objects it already has)
        3. Server computes pack and sends it

        This method takes the already-read request body to avoid deadlock issues
        when using StreamingResponse (reading from request inside a streaming
        response generator causes deadlock).

        Args:
            body: The complete request body bytes
        """
        logger.info(f"handle_upload_pack_from_bytes: processing {len(body)} bytes")
        want_shas = []
        have_shas = []
        capabilities = set()
        shallow_shas = []
        depth = None

        # Parse pkt-lines from body
        offset = 0
        while offset < len(body):
            if offset + 4 > len(body):
                break

            length_hex = body[offset : offset + 4]
            try:
                length = int(length_hex, 16)
            except ValueError:
                break

            if length == 0:
                # Flush packet
                offset += 4
                continue
            elif length < 4:
                break

            data = body[offset + 4 : offset + length]
            offset += length

            line = data.rstrip(b"\n")

            if line.startswith(b"want "):
                # Parse want line: "want <sha> [capabilities...]"
                parts = line.split(b" ", 2)
                sha = parts[1]
                if len(parts) > 2:
                    # Extract capabilities from first want line
                    caps_str = parts[2]
                    capabilities.update(caps_str.split(b" "))
                want_shas.append(sha)
                logger.debug(f"handle_upload_pack: want {sha.decode()}")

            elif line.startswith(b"have "):
                sha = line.split(b" ", 1)[1]
                have_shas.append(sha)

            elif line.startswith(b"shallow "):
                sha = line.split(b" ", 1)[1]
                shallow_shas.append(sha)

            elif line.startswith(b"deepen "):
                depth = int(line.split(b" ", 1)[1])

            elif line == b"done":
                break

        if not want_shas:
            # No wants, nothing to send
            logger.info("handle_upload_pack: no wants, sending NAK")
            yield pkt_line(b"NAK\n")
            yield pkt_flush()
            return

        logger.info(f"Upload pack: wants={len(want_shas)}, haves={len(have_shas)}, caps={capabilities}")

        # Acknowledge haves
        common = []
        for sha in have_shas:
            if await self.repo.has_object_async(sha):
                common.append(sha)

        if b"multi_ack" in capabilities and common:
            for sha in common:
                yield pkt_line(b"ACK " + sha + b" continue\n")

        yield pkt_line(b"NAK\n")

        # Generate pack data
        use_side_band = b"side-band-64k" in capabilities or b"side-band" in capabilities
        include_tag = b"include-tag" in capabilities

        try:
            pack_data = await self._generate_pack(
                want_shas,
                have_shas,
                include_tag=include_tag,
            )

            logger.info(f"handle_upload_pack: generated pack of {len(pack_data)} bytes, use_side_band={use_side_band}")

            if use_side_band:
                # Send pack data through side-band channel
                # pkt-line format: 4 hex chars (length) + data
                # Max pkt-line length is 65520 (0xfff0)
                # For side-band: data = 1 byte (channel) + chunk
                # So max chunk = 65520 - 4 - 1 = 65515 bytes
                chunk_size = 65515 if b"side-band-64k" in capabilities else 999

                offset = 0
                while offset < len(pack_data):
                    chunk = pack_data[offset : offset + chunk_size]
                    yield pkt_line(bytes([SIDE_BAND_CHANNEL_DATA]) + chunk)
                    offset += chunk_size

                yield pkt_flush()
            else:
                # Send pack data directly
                yield pack_data

            logger.info("handle_upload_pack: done sending pack")

        except Exception as e:
            logger.error(f"Pack generation failed: {e}", exc_info=True)
            if use_side_band:
                yield pkt_line(
                    bytes([SIDE_BAND_CHANNEL_FATAL]) + f"error: {e}".encode()
                )
            yield pkt_flush()

    async def handle_upload_pack(self, request: Request) -> AsyncIterator[bytes]:
        """Handle git-upload-pack requests (fetch/clone).

        DEPRECATED: Use handle_upload_pack_from_bytes instead. Reading from
        request.stream() inside a StreamingResponse generator causes deadlock.

        Protocol:
        1. Client sends WANT lines (objects it wants)
        2. Client sends HAVE lines (objects it already has)
        3. Server computes pack and sends it
        """
        logger.warning("handle_upload_pack: DEPRECATED - use handle_upload_pack_from_bytes instead")
        # Fall back to reading body first
        body = await request.body()
        async for chunk in self.handle_upload_pack_from_bytes(body):
            yield chunk

    async def _generate_pack(
        self,
        want_shas: list[bytes],
        have_shas: list[bytes],
        include_tag: bool = False,
    ) -> bytes:
        """Generate a pack file containing requested objects.

        This implements the pack generation algorithm:
        1. Find all objects reachable from wants
        2. Exclude objects reachable from haves (common ancestors)
        3. Pack the remaining objects with delta compression
        """
        # Collect all objects needed
        objects_to_send = set()

        # Walk from wants
        for sha in want_shas:
            await self._collect_reachable_objects(sha, objects_to_send, set(have_shas))

        if not objects_to_send:
            # Return empty pack
            return self._create_empty_pack()

        # Get all objects
        object_list = []
        for sha in objects_to_send:
            try:
                obj = await self.repo.get_object_async(sha)
                object_list.append((obj, None))
            except KeyError:
                logger.warning(f"Object {sha.decode()} not found")

        # Create pack
        pack_buffer = io.BytesIO()
        # write_pack_objects needs a sequence (not generator) since it calls len()
        write_pack_objects(
            pack_buffer,
            [obj for obj, _ in object_list],
            object_format=DEFAULT_OBJECT_FORMAT,
        )
        pack_buffer.seek(0)
        return pack_buffer.read()

    async def compute_common_haves(self, have_shas: list[bytes]) -> list[bytes]:
        """Return the subset of have_shas the server actually has (ACK list)."""
        common = []
        for sha in have_shas:
            if await self.repo.has_object_async(sha):
                common.append(sha)
        return common

    async def buffered_upload_pack_response(
        self,
        shas: list[bytes],
        common: list[bytes],
        capabilities: set,
        shallow_shas: list[bytes] = None,
    ) -> bytes:
        """Build the full upload-pack response in memory (small repos).

        Used below the streaming threshold for client compatibility
        (libgit2/wasm-git/CORS proxies that mishandle chunked encoding). The
        pack is still generated with the lazy record iterator, but collected
        into a single buffer so Content-Length can be set.
        """
        parts: list[bytes] = []
        async for chunk in self.stream_upload_pack_response(
            shas, common, capabilities, shallow_shas=shallow_shas
        ):
            parts.append(chunk)
        return b"".join(parts)

    async def empty_upload_pack_response(self, capabilities: set) -> bytes:
        """Build a NAK + empty-pack response (no wants resolvable)."""
        parts = [pkt_line(b"NAK\n")]
        use_side_band = (
            b"side-band-64k" in capabilities or b"side-band" in capabilities
        )
        empty = self._create_empty_pack()
        if use_side_band:
            parts.append(pkt_line(bytes([SIDE_BAND_CHANNEL_DATA]) + empty))
            parts.append(pkt_flush())
        else:
            parts.append(empty)
        return b"".join(parts)

    def _parse_upload_pack_request(
        self, body: bytes
    ) -> tuple[list[bytes], list[bytes], set, "int | None", list[bytes]]:
        """Parse an upload-pack request body into wants, haves, capabilities, and the
        shallow-clone parameters.

        Returns:
            (want_shas, have_shas, capabilities, depth, client_shallows)
            - depth: the ``deepen <N>`` value for a shallow clone, or None for a full
              clone. (deepen-since/deepen-not are not supported and are ignored.)
            - client_shallows: ``shallow <sha>`` lines the client already has.
        """
        want_shas: list[bytes] = []
        have_shas: list[bytes] = []
        capabilities: set = set()
        depth: "int | None" = None
        client_shallows: list[bytes] = []
        has_done = False

        offset = 0
        while offset < len(body):
            if offset + 4 > len(body):
                break
            try:
                length = int(body[offset : offset + 4], 16)
            except ValueError:
                break
            if length == 0:
                offset += 4
                continue
            elif length < 4:
                break

            data = body[offset + 4 : offset + length]
            offset += length
            line = data.rstrip(b"\n")

            if line.startswith(b"want "):
                parts = line.split(b" ", 2)
                sha = parts[1]
                if len(parts) > 2:
                    capabilities.update(parts[2].split(b" "))
                want_shas.append(sha)
            elif line.startswith(b"have "):
                have_shas.append(line.split(b" ", 1)[1])
            elif line.startswith(b"shallow "):
                client_shallows.append(line.split(b" ", 1)[1])
            elif line.startswith(b"deepen "):
                try:
                    parsed = int(line.split(b" ", 1)[1])
                    # depth must be >= 1 to be meaningful
                    depth = parsed if parsed >= 1 else None
                except (ValueError, IndexError):
                    depth = None
            elif line == b"done":
                has_done = True
                break

        return want_shas, have_shas, capabilities, depth, client_shallows, has_done

    async def resolve_pack_shas(
        self, want_shas: list[bytes], have_shas: list[bytes]
    ) -> list[bytes]:
        """Resolve the ordered set of object SHAs to include in the pack.

        Walks reachability from wants (excluding haves), then pre-loads all
        packs so the subsequent (synchronous) pack generation can read each
        object from the in-memory S3Pack without further async I/O.
        """
        objects_to_send: set = set()
        for sha in want_shas:
            await self._collect_reachable_objects(
                sha, objects_to_send, set(have_shas)
            )
        # Pre-load packs once so the sync record iterator can pull objects.
        await self.repo.ensure_packs_loaded()
        return list(objects_to_send)

    async def resolve_pack_shas_shallow(
        self, want_shas: list[bytes], depth: int
    ) -> tuple[list[bytes], list[bytes]]:
        """Depth-bounded reachability for a shallow clone (``git clone --depth=N``).

        Walks at most ``depth`` commits back from each want (the want commit itself is
        depth 1), collecting each included commit's full tree + blobs but NOT commit
        parents beyond the depth limit. Commits whose parents are cut off by the limit
        become the SHALLOW BOUNDARY — the client is told ``shallow <sha>`` for each so it
        records a shallow repository. A breadth-first walk guarantees each commit is
        processed at its minimum depth, so boundary detection is correct across branches.

        Returns (object_shas, shallow_boundary_shas).

        Only invoked when the client sends ``deepen`` — resolve_pack_shas (full clone) is
        left completely untouched.
        """
        from collections import deque
        from dulwich.objects import Commit, Tag

        objects: set = set()
        shallow_boundary: set = set()
        seen_commits: set = set()
        queue: deque = deque()

        # Peel each want (it may be an annotated tag) down to a commit; include the tag
        # objects themselves in the pack.
        for want in want_shas:
            cur = want
            try:
                obj = await self.repo.get_object_async(cur)
            except KeyError:
                continue
            while isinstance(obj, Tag):
                objects.add(cur)
                cur = obj.object[1]
                try:
                    obj = await self.repo.get_object_async(cur)
                except KeyError:
                    obj = None
                    break
            if isinstance(obj, Commit):
                queue.append((cur, 1))

        while queue:
            sha, d = queue.popleft()
            if sha in seen_commits:
                continue
            seen_commits.add(sha)
            try:
                commit = await self.repo.get_object_async(sha)
            except KeyError:
                continue
            if not isinstance(commit, Commit):
                continue
            objects.add(sha)
            # Full tree (subtrees + blobs) at this commit — reuse the tree-only walk
            # (a tree has no parents, so this never traverses commit history).
            await self._collect_reachable_objects(commit.tree, objects, set())
            if d >= depth:
                if commit.parents:
                    shallow_boundary.add(sha)  # parents cut off → boundary
            else:
                for parent_sha in commit.parents:
                    if parent_sha not in seen_commits:
                        queue.append((parent_sha, d + 1))

        await self.repo.ensure_packs_loaded()
        return list(objects), list(shallow_boundary)

    def shallow_update_section(
        self, shallow_shas: list[bytes], unshallow_shas: list[bytes] = None
    ) -> bytes:
        """Build the shallow-update section: ``shallow``/``unshallow`` pkt-lines + flush.

        Sent on its own as the response to the stateless-RPC shallow NEGOTIATION round
        (wants+deepen without ``done``), and also prepended to the pack response on the
        round that carries ``done`` (handled in stream_upload_pack_response).
        """
        parts: list[bytes] = []
        for sha in shallow_shas or []:
            parts.append(pkt_line(b"shallow " + sha + b"\n"))
        for sha in unshallow_shas or []:
            parts.append(pkt_line(b"unshallow " + sha + b"\n"))
        parts.append(pkt_flush())
        return b"".join(parts)

    def estimate_pack_size(self, shas: list[bytes]) -> int:
        """Estimate total uncompressed object size for the given SHAs.

        Uses the pre-loaded in-memory packs (call resolve_pack_shas first).
        This is an upper-bound estimate of the response size; the actual
        compressed pack is smaller. Used only to choose buffered vs streamed.
        """
        store = self.repo._object_store
        total = 0
        for sha in shas:
            try:
                _type_num, data = store.get_raw(sha)
                total += len(data)
            except KeyError:
                continue
        return total

    def iter_pack_records(self, shas: list[bytes]):
        """Lazily yield UnpackedObject records, one object live at a time.

        Pulls each object's raw bytes from the pre-loaded in-memory packs as the
        consumer requests it, so the whole object_list is never materialized.
        Must be run after resolve_pack_shas (packs loaded). Synchronous — run
        inside a thread.
        """
        store = self.repo._object_store
        for sha in shas:
            try:
                type_num, data = store.get_raw(sha)
            except KeyError:
                logger.warning(f"Object {sha!r} not found during pack gen")
                continue
            # decomp_chunks holds the raw (uncompressed) object content; the
            # pack writer compresses it. sha must be the 20-byte binary digest.
            if len(sha) == 40:
                bin_sha = bytes.fromhex(sha.decode("ascii"))
            else:
                bin_sha = sha
            yield UnpackedObject(
                type_num,
                decomp_chunks=[data] if not isinstance(data, list) else data,
                sha=bin_sha,
            )

    async def stream_upload_pack_response(
        self,
        shas: list[bytes],
        common: list[bytes],
        capabilities: set,
        shallow_shas: list[bytes] = None,
    ) -> AsyncIterator[bytes]:
        """Stream an upload-pack response, generating the pack incrementally.

        The pack-generation loop runs in a worker thread (dulwich is sync and
        CPU-bound). It pushes side-band-wrapped pkt-line chunks into an
        asyncio.Queue; this async generator drains the queue and yields chunks
        to the StreamingResponse. Only ~1 object is decompressed at a time via
        the lazy iter_pack_records iterator, so peak memory stays bounded.
        """
        # Shallow-clone section: when the client requested --depth (shallow_shas is a
        # list, even empty), the server MUST send the shallow/unshallow lines + a flush
        # BEFORE the ACK/NAK — an EMPTY section (just the flush) is still required when
        # there is no boundary (single-commit repo, depth >= history), otherwise the
        # client errors "expected shallow list". `None` = full clone → no section.
        if shallow_shas is not None:
            for sha in shallow_shas:
                yield pkt_line(b"shallow " + sha + b"\n")
            yield pkt_flush()

        # Leading ACK/NAK lines (small) come first.
        if b"multi_ack" in capabilities and common:
            for sha in common:
                yield pkt_line(b"ACK " + sha + b" continue\n")
        yield pkt_line(b"NAK\n")

        use_side_band = (
            b"side-band-64k" in capabilities or b"side-band" in capabilities
        )
        # Max side-band payload: 65520 - 4 (len prefix) - 1 (channel) = 65515.
        chunk_size = 65515 if b"side-band-64k" in capabilities else 999

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        _SENTINEL = object()

        def _wrap(chunk: bytes) -> bytes:
            if use_side_band:
                return pkt_line(bytes([SIDE_BAND_CHANNEL_DATA]) + chunk)
            return chunk

        def _produce():
            """Sync producer: write pack chunks, side-band-wrap, enqueue."""
            buf = bytearray()

            def sink(data: bytes):
                # Accumulate then emit fixed-size side-band frames so each
                # pkt-line stays within the 65520 byte limit.
                buf.extend(data)
                while len(buf) >= chunk_size:
                    frame = bytes(buf[:chunk_size])
                    del buf[:chunk_size]
                    asyncio.run_coroutine_threadsafe(
                        queue.put(_wrap(frame)), loop
                    ).result()

            try:
                write_pack_data(
                    sink,
                    self.iter_pack_records(shas),
                    DEFAULT_OBJECT_FORMAT,
                    num_records=len(shas),
                )
                # Flush remaining buffered bytes.
                if buf:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(_wrap(bytes(buf))), loop
                    ).result()
                asyncio.run_coroutine_threadsafe(
                    queue.put(_SENTINEL), loop
                ).result()
            except Exception as exc:  # noqa: BLE001 - propagate to consumer
                asyncio.run_coroutine_threadsafe(
                    queue.put(("__error__", exc)), loop
                ).result()

        producer = loop.run_in_executor(None, _produce)
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, tuple) and item and item[0] == "__error__":
                    exc = item[1]
                    logger.error(f"Streaming pack generation failed: {exc}")
                    if use_side_band:
                        yield pkt_line(
                            bytes([SIDE_BAND_CHANNEL_FATAL])
                            + f"error: {exc}".encode()
                        )
                    break
                yield item
        finally:
            await producer

        if use_side_band:
            yield pkt_flush()

    async def _collect_reachable_objects(
        self,
        sha: bytes,
        result: set,
        exclude: set,
    ):
        """Collect all objects reachable from a given SHA."""
        if sha in result or sha in exclude:
            return

        try:
            obj = await self.repo.get_object_async(sha)
        except KeyError:
            return

        result.add(sha)

        # Recursively collect based on object type
        from dulwich.objects import Blob, Commit, Tag, Tree

        if isinstance(obj, Commit):
            # Add tree
            await self._collect_reachable_objects(obj.tree, result, exclude)
            # Traverse parent commits to include all reachable objects
            # This is essential for clones to work properly
            for parent_sha in obj.parents:
                await self._collect_reachable_objects(parent_sha, result, exclude)
        elif isinstance(obj, Tree):
            # Add all entries
            for entry in obj.items():
                await self._collect_reachable_objects(entry.sha, result, exclude)
        elif isinstance(obj, Tag):
            # Add tagged object
            await self._collect_reachable_objects(obj.object[1], result, exclude)
        # Blobs have no children

    def _create_empty_pack(self) -> bytes:
        """Create an empty pack file."""
        from dulwich.pack import write_pack_header
        import hashlib

        buffer = io.BytesIO()
        write_pack_header(buffer, 0)
        # Add checksum
        buffer.seek(0)
        data = buffer.read()
        checksum = hashlib.sha1(data).digest()
        return data + checksum

    def _build_receive_status_report(
        self, unpack_status: str, results: dict, capabilities: set
    ) -> bytes:
        """Build the (tiny) receive-pack status report, buffered.

        Returns the full report bytes (pkt-lines, optionally side-band-wrapped).
        """
        use_side_band = (
            b"side-band-64k" in capabilities or b"report-status" in capabilities
        )
        out: list[bytes] = []
        if b"report-status" in capabilities:
            if use_side_band:
                status_lines = [pkt_line(f"unpack {unpack_status}\n".encode())]
                for ref_name, error in results.items():
                    if error:
                        status = f"ng {ref_name.decode()} {error}\n"
                    else:
                        status = f"ok {ref_name.decode()}\n"
                    status_lines.append(pkt_line(status.encode()))
                status_lines.append(pkt_flush())
                for line in status_lines:
                    out.append(pkt_line(bytes([SIDE_BAND_CHANNEL_DATA]) + line))
                out.append(pkt_flush())
            else:
                out.append(pkt_line(f"unpack {unpack_status}\n".encode()))
                for ref_name, error in results.items():
                    if error:
                        out.append(pkt_line(f"ng {ref_name.decode()} {error}\n".encode()))
                    else:
                        out.append(pkt_line(f"ok {ref_name.decode()}\n".encode()))
                out.append(pkt_flush())
        else:
            out.append(pkt_flush())
        return b"".join(out)

    async def handle_receive_pack_streaming(self, request: Request) -> bytes:
        """Handle git-receive-pack by spilling the pack to a temp file.

        Streams the request body into a SpooledTemporaryFile (spilling to disk
        past PACK_SPOOL_FILE_MAX_SIZE) while parsing the leading pkt-line
        ref-update commands, then parses/stores the pack from the file. This
        avoids holding the whole pushed pack as one resident bytes object.

        The status report is tiny, so it is returned buffered (bytes).
        """
        if self.read_only:
            return pkt_line(b"unpack ng permission denied\n") + pkt_flush()

        reader = StreamingPktLineReader(request)
        ref_updates: dict = {}
        capabilities: set = set()
        first_line = True

        # Parse ref-update commands (pkt-lines before PACK).
        while True:
            data, is_flush = await reader.read_pkt_line()
            if is_flush:
                continue
            if data is None:
                # PACK signature or end of stream.
                break
            line = data.rstrip(b"\n")
            if not line:
                continue
            if first_line and b"\0" in line:
                line, caps_str = line.split(b"\0", 1)
                capabilities.update(c for c in caps_str.split(b" ") if c)
                first_line = False
            parts = line.split(b" ", 2)
            if len(parts) >= 3:
                old_sha, new_sha, ref_name = parts
                ref_updates[ref_name] = (old_sha, new_sha)

        # Spill remaining (pack) data to a seekable spool file.
        pack_file = tempfile.SpooledTemporaryFile(
            max_size=PACK_SPOOL_FILE_MAX_SIZE, prefix="incoming-push-"
        )
        total = 0
        async for chunk in reader.read_remaining_stream():
            pack_file.write(chunk)
            total += len(chunk)
        pack_file.seek(0)

        logger.info(
            f"Receive pack (streamed): {len(ref_updates)} refs, {total} bytes pack"
        )

        has_pack = total >= 4  # at least a PACK signature
        try:
            if has_pack:
                results = await self.repo.receive_pack_from_file_async(
                    pack_file, ref_updates
                )
            else:
                results = await self.repo.receive_pack_from_file_async(
                    None, ref_updates
                )
            unpack_status = "ok"
        except Exception as e:
            logger.error(f"Pack processing failed: {e}", exc_info=True)
            unpack_status = f"ng {e}"
            results = {ref: str(e) for ref in ref_updates}
        finally:
            pack_file.close()

        return self._build_receive_status_report(
            unpack_status, results, capabilities
        )

    async def handle_receive_pack_from_bytes(self, body: bytes) -> AsyncIterator[bytes]:
        """Handle git-receive-pack requests (push) from pre-read body.

        Protocol:
        1. Client sends ref updates (old-sha new-sha ref-name)
        2. Client sends pack data
        3. Server processes and reports status

        This method takes the already-read request body to avoid deadlock issues
        when using StreamingResponse (reading from request inside a streaming
        response generator causes deadlock).

        Args:
            body: The complete request body bytes
        """
        logger.info(f"handle_receive_pack_from_bytes: processing {len(body)} bytes")
        if self.read_only:
            yield pkt_line(b"unpack ng permission denied\n")
            yield pkt_flush()
            return

        # Parse ref updates and pack data from the body
        ref_updates = {}
        pack_start = None
        offset = 0
        capabilities = set()
        first_line = True

        while offset < len(body):
            # Check for pack data start (PACK signature)
            if body[offset : offset + 4] == b"PACK":
                pack_start = offset
                break

            # Read pkt-line
            try:
                length = int(body[offset : offset + 4], 16)
            except ValueError:
                break

            if length == 0:
                # Flush packet
                offset += 4
                continue
            elif length < 4:
                break

            line = body[offset + 4 : offset + length].rstrip(b"\n")
            offset += length
            logger.debug(f"handle_receive_pack: pkt-line: {line!r}")

            if not line:
                continue

            # Parse command line: old-sha new-sha ref-name[\0capabilities]
            if first_line and b"\0" in line:
                line, caps_str = line.split(b"\0", 1)
                logger.debug(f"handle_receive_pack: caps_str={caps_str!r}")
                # Filter out empty capabilities
                caps = [c for c in caps_str.split(b" ") if c]
                capabilities.update(caps)
                first_line = False
                logger.info(f"handle_receive_pack: capabilities={capabilities}")

            parts = line.split(b" ", 2)
            if len(parts) >= 3:
                old_sha, new_sha, ref_name = parts
                ref_updates[ref_name] = (old_sha, new_sha)
                logger.info(f"handle_receive_pack: ref update {ref_name}: {old_sha[:8]}... -> {new_sha[:8]}...")

        # Extract pack data
        pack_data = body[pack_start:] if pack_start else b""

        logger.info(
            f"Receive pack: {len(ref_updates)} refs, {len(pack_data)} bytes pack"
        )

        # Process the push
        try:
            logger.info(f"handle_receive_pack: calling receive_pack_async")
            results = await self.repo.receive_pack_async(pack_data, ref_updates)
            unpack_status = "ok"
            logger.info(f"handle_receive_pack: receive_pack_async succeeded, results={results}")
        except Exception as e:
            logger.error(f"Pack processing failed: {e}", exc_info=True)
            unpack_status = f"ng {e}"
            results = {ref: str(e) for ref in ref_updates}

        # Generate status report
        use_side_band = b"side-band-64k" in capabilities or b"report-status" in capabilities
        logger.info(f"handle_receive_pack: use_side_band={use_side_band}, report-status={b'report-status' in capabilities}")

        if b"report-status" in capabilities:
            if use_side_band:
                # Send through side-band
                logger.info("handle_receive_pack: sending status through side-band")
                status_lines = [pkt_line(f"unpack {unpack_status}\n".encode())]
                for ref_name, error in results.items():
                    if error:
                        status = f"ng {ref_name.decode()} {error}\n"
                    else:
                        status = f"ok {ref_name.decode()}\n"
                    status_lines.append(pkt_line(status.encode()))
                status_lines.append(pkt_flush())

                for line in status_lines:
                    yield pkt_line(bytes([SIDE_BAND_CHANNEL_DATA]) + line)
                yield pkt_flush()
                logger.info("handle_receive_pack: done sending side-band status")
            else:
                logger.info("handle_receive_pack: sending status without side-band")
                yield pkt_line(f"unpack {unpack_status}\n".encode())
                for ref_name, error in results.items():
                    if error:
                        yield pkt_line(f"ng {ref_name.decode()} {error}\n".encode())
                    else:
                        yield pkt_line(f"ok {ref_name.decode()}\n".encode())
                yield pkt_flush()
                logger.info("handle_receive_pack: done sending status")
        else:
            logger.info("handle_receive_pack: no report-status capability, sending flush only")
            yield pkt_flush()

    async def handle_receive_pack(self, request: Request) -> AsyncIterator[bytes]:
        """Handle git-receive-pack requests (push).

        DEPRECATED: Use handle_receive_pack_from_bytes instead. Reading from
        request.stream() inside a StreamingResponse generator causes deadlock.

        Protocol:
        1. Client sends ref updates (old-sha new-sha ref-name)
        2. Client sends pack data
        3. Server processes and reports status
        """
        logger.warning("handle_receive_pack: DEPRECATED - use handle_receive_pack_from_bytes instead")
        # Fall back to reading body first
        body = await request.body()
        async for chunk in self.handle_receive_pack_from_bytes(body):
            yield chunk
        return  # Don't continue to old implementation

        # Old implementation kept for reference but unreachable
        logger.info("handle_receive_pack: starting")
        if self.read_only:
            yield pkt_line(b"unpack ng permission denied\n")
            yield pkt_flush()
            return

        # Use streaming reader to parse ref updates without loading all into memory
        logger.info("handle_receive_pack: creating StreamingPktLineReader")
        reader = StreamingPktLineReader(request)
        ref_updates = {}
        capabilities = set()
        first_line = True

        # Parse ref update commands (pkt-lines before pack data)
        logger.info("handle_receive_pack: parsing ref update commands")
        while True:
            logger.debug("handle_receive_pack: reading pkt-line")
            data, is_flush = await reader.read_pkt_line()
            logger.debug(f"handle_receive_pack: got pkt-line: data={data[:50] if data else None}..., is_flush={is_flush}")

            if is_flush:
                # Flush packet - continue to next pkt-line
                logger.debug("handle_receive_pack: flush packet, continuing")
                continue

            if data is None:
                # Either end of stream or PACK signature encountered
                logger.info("handle_receive_pack: end of pkt-lines (PACK or end of stream)")
                break

            line = data.rstrip(b"\n")
            if not line:
                continue

            # Parse command line: old-sha new-sha ref-name[\0capabilities]
            if first_line and b"\0" in line:
                line, caps_str = line.split(b"\0", 1)
                capabilities.update(caps_str.split(b" "))
                first_line = False
                logger.info(f"handle_receive_pack: capabilities={capabilities}")

            parts = line.split(b" ", 2)
            if len(parts) >= 3:
                old_sha, new_sha, ref_name = parts
                ref_updates[ref_name] = (old_sha, new_sha)
                logger.info(f"handle_receive_pack: ref update {ref_name}: {old_sha} -> {new_sha}")

        # Collect pack data from remaining stream
        # Note: For very large packs, this could be further optimized to stream
        # directly to S3 in chunks, but that requires changes to receive_pack_async
        pack_chunks = []
        async for chunk in reader.read_remaining_stream():
            pack_chunks.append(chunk)
        pack_data = b"".join(pack_chunks)

        logger.info(
            f"Receive pack: {len(ref_updates)} refs, {len(pack_data)} bytes pack"
        )

        # Process the push
        try:
            results = await self.repo.receive_pack_async(pack_data, ref_updates)
            unpack_status = "ok"
        except Exception as e:
            logger.error(f"Pack processing failed: {e}")
            unpack_status = f"ng {e}"
            results = {ref: str(e) for ref in ref_updates}

        # Generate status report
        use_side_band = b"side-band-64k" in capabilities or b"report-status" in capabilities

        if b"report-status" in capabilities:
            if use_side_band:
                # Send through side-band
                status_lines = [pkt_line(f"unpack {unpack_status}\n".encode())]
                for ref_name, error in results.items():
                    if error:
                        status = f"ng {ref_name.decode()} {error}\n"
                    else:
                        status = f"ok {ref_name.decode()}\n"
                    status_lines.append(pkt_line(status.encode()))
                status_lines.append(pkt_flush())

                for line in status_lines:
                    yield pkt_line(bytes([SIDE_BAND_CHANNEL_DATA]) + line)
                yield pkt_flush()
            else:
                yield pkt_line(f"unpack {unpack_status}\n".encode())
                for ref_name, error in results.items():
                    if error:
                        yield pkt_line(f"ng {ref_name.decode()} {error}\n".encode())
                    else:
                        yield pkt_line(f"ok {ref_name.decode()}\n".encode())
                yield pkt_flush()
        else:
            yield pkt_flush()


def create_git_router(
    get_repo_callback,
    login_optional_dep,
    login_required_dep,
    parse_user_token: Callable = None,
) -> APIRouter:
    """Create a FastAPI router for Git HTTP protocol.

    Args:
        get_repo_callback: Async function (workspace, alias, user_info) -> S3GitRepo
        login_optional_dep: FastAPI dependency for optional authentication
        login_required_dep: FastAPI dependency for required authentication
        parse_user_token: Async function to parse token string into UserInfo

    Returns:
        FastAPI router with Git endpoints
    """
    router = APIRouter()

    def _validate_git_username(provided_username: Optional[str]) -> None:
        """Validate that username is 'git' for Basic Auth.

        Raises HTTPException if username is provided but not 'git'.
        """
        if provided_username is not None and provided_username != "git":
            logger.warning(f"Git auth: invalid username '{provided_username}', expected 'git'")
            raise HTTPException(
                status_code=401,
                detail="Username must be 'git'. Use 'git' as username and your token as password.",
                headers={"WWW-Authenticate": 'Basic realm="Git Repository"'},
            )

    async def git_optional_auth(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        """Optional auth for Git read operations (clone/fetch).

        If credentials are provided, validates username is 'git'.
        Returns user_info if authenticated, None if anonymous.
        """
        if not authorization:
            # No auth provided, fall back to optional dep
            return await login_optional_dep(request)

        # Auth provided - validate username and parse token
        provided_username, token = extract_credentials_from_basic_auth(authorization)
        _validate_git_username(provided_username)

        if not token:
            # Invalid auth format - fall back to optional dep
            return await login_optional_dep(request)

        if parse_user_token:
            try:
                user_info = await parse_user_token(token)
                if user_info.scope.current_workspace is None:
                    user_info.scope.current_workspace = user_info.get_workspace()
                return user_info
            except Exception:
                # Token parsing failed - fall back to optional dep
                return await login_optional_dep(request)
        else:
            return await login_optional_dep(request)

    async def git_basic_auth(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        """Custom auth dependency that handles HTTP Basic Auth for Git.

        Git clients send credentials as Basic Auth: base64(git:token)
        where username must be 'git' and password is the Hypha token.
        Authorization is determined solely by the token.
        """
        # Debug logging for all headers
        logger.info(f"Git auth request: method={request.method}, path={request.url.path}")
        logger.info(f"Git auth headers: {dict(request.headers)}")
        logger.info(f"Git auth: authorization header value = {authorization[:50] if authorization else None}...")

        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": 'Basic realm="Git Repository"'},
            )

        # Extract username and token from Basic Auth or Bearer token
        provided_username, token = extract_credentials_from_basic_auth(authorization)
        logger.debug(f"Git auth: extracted username={provided_username}, token type={authorization[:10] if authorization else None}...")
        if not token:
            logger.warning(f"Git auth: could not extract token from authorization header")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": 'Basic realm="Git Repository"'},
            )

        # Validate username is 'git'
        _validate_git_username(provided_username)

        # Parse the token to get user info
        if parse_user_token:
            try:
                logger.debug(f"Git auth: parsing token (length={len(token)})")
                user_info = await parse_user_token(token)
                if user_info.scope.current_workspace is None:
                    user_info.scope.current_workspace = user_info.get_workspace()

                logger.info(f"Git auth successful for user {user_info.id}")
                return user_info
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Git auth failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=401,
                    detail=f"Authentication failed: {str(e)}",
                    headers={"WWW-Authenticate": 'Basic realm="Git Repository"'},
                )
        else:
            # Fallback to login_required_dep if parse_user_token not provided
            logger.debug("Git auth: using login_required_dep fallback")
            return await login_required_dep(request)

    @router.get("/{workspace}/git/{alias}/info/refs")
    async def git_info_refs(
        workspace: str,
        alias: str,
        service: str = Query(None),
        user_info=Depends(git_optional_auth),
    ):
        """Git reference discovery endpoint.

        This is the first endpoint called by git clone/fetch/push.
        """
        # Strip .git suffix if present (Git clients may add it)
        if alias.endswith(".git"):
            alias = alias[:-4]

        if service not in ("git-upload-pack", "git-receive-pack"):
            # Dumb protocol fallback
            raise HTTPException(status_code=400, detail="Smart HTTP protocol required")

        # For git-receive-pack (push), require authentication at discovery stage
        # This provides a better user experience - git client will prompt for credentials
        # BEFORE starting the push, rather than failing mid-push with a confusing error
        if service == "git-receive-pack":
            if not user_info or getattr(user_info, 'is_anonymous', False):
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required for push. Please provide credentials.",
                    headers={"WWW-Authenticate": 'Basic realm="Git Repository - Push requires authentication"'},
                )

        try:
            # For push operations, check write permission at discovery phase
            write_access = service == "git-receive-pack"
            repo = await get_repo_callback(workspace, alias, user_info, write=write_access)
        except PermissionError:
            # If anonymous user doesn't have permission, request authentication
            # This allows git clients to retry with credentials
            if user_info and getattr(user_info, 'is_anonymous', False):
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": 'Basic realm="Git Repository"'},
                )
            raise HTTPException(status_code=403, detail="Permission denied")
        except KeyError as e:
            # If anonymous user can't find the artifact, it might be a private repo
            # Request authentication so git client can retry with credentials
            if user_info and getattr(user_info, 'is_anonymous', False):
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": 'Basic realm="Git Repository"'},
                )
            raise HTTPException(status_code=404, detail="Repository not found")

        handler = GitHTTPHandler(repo, read_only=False)  # Already checked write permission if needed

        media_type = f"application/x-{service}-advertisement"

        # Collect the full response body instead of streaming.
        # The refs advertisement is small and buffering it allows us to set
        # Content-Length, avoiding HTTP chunked transfer encoding.
        # This improves compatibility with libgit2/wasm-git clients and
        # CORS proxies that may not handle chunked encoding correctly.
        try:
            body_parts = []
            async for chunk in handler.generate_refs_advertisement(service.encode()):
                body_parts.append(chunk)
            response_body = b"".join(body_parts)
        finally:
            await cleanup_repo(repo)

        return Response(
            content=response_body,
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Expires": "Fri, 01 Jan 1980 00:00:00 GMT",
            },
        )

    @router.post("/{workspace}/git/{alias}/git-upload-pack")
    async def git_upload_pack(
        workspace: str,
        alias: str,
        request: Request,
        user_info=Depends(git_optional_auth),
    ):
        """Handle git fetch/clone requests."""
        # Strip .git suffix if present (Git clients may add it)
        if alias.endswith(".git"):
            alias = alias[:-4]

        try:
            repo = await get_repo_callback(workspace, alias, user_info)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except KeyError:
            raise HTTPException(status_code=404, detail="Repository not found")

        handler = GitHTTPHandler(repo)

        response_headers = {
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Expires": "Fri, 01 Jan 1980 00:00:00 GMT",
        }
        media_type = "application/x-git-upload-pack-result"

        # Acquire a pack-op slot (bounds concurrent memory-heavy clones).
        sem = await _acquire_pack_slot()

        # From here on, on ANY failure before we hand off to a StreamingResponse
        # we must release the slot and clean up the repo ourselves. Once we
        # return a StreamingResponse, its generator owns release/cleanup.
        slot_handed_off = False
        try:
            logger.info("git_upload_pack: reading request body")
            body = await request.body()
            logger.info(f"git_upload_pack: received {len(body)} bytes")

            (
                want_shas,
                have_shas,
                capabilities,
                depth,
                _client_shallows,
                has_done,
            ) = handler._parse_upload_pack_request(body)

            if not want_shas:
                # No wants: NAK + empty pack, buffered (tiny).
                response_body = await handler.empty_upload_pack_response(capabilities)
                return Response(
                    content=response_body,
                    media_type=media_type,
                    headers=response_headers,
                )

            common = await handler.compute_common_haves(have_shas)
            # None = full clone (no shallow section in the response); a list (even empty)
            # = the client requested `deepen`, so the response MUST carry a shallow-update
            # section even when there is no boundary (e.g. a single-commit repo, or
            # depth >= history) — an EMPTY shallow section (just a flush) is required, and
            # omitting it makes the client error with "expected shallow list".
            shallow_shas: list = None
            if depth is not None:
                # Shallow clone (git clone --depth=N): depth-bounded object set + the
                # shallow boundary the client must be told about. Full-clone path below
                # is unchanged.
                shas, shallow_shas = await handler.resolve_pack_shas_shallow(
                    want_shas, depth
                )
                if not has_done:
                    # Stateless-RPC shallow NEGOTIATION round: the client sent
                    # wants+deepen WITHOUT `done` and expects ONLY the shallow-update
                    # (shallow/unshallow lines + flush) — no acks, no pack. Sending the
                    # pack here leaves a stray NAK that corrupts the next round's read
                    # ("expected shallow list"). The client then re-sends with `done`.
                    return Response(
                        content=handler.shallow_update_section(shallow_shas),
                        media_type=media_type,
                        headers=response_headers,
                    )
            else:
                shas = await handler.resolve_pack_shas(want_shas, have_shas)

            if not shas:
                response_body = await handler.empty_upload_pack_response(capabilities)
                return Response(
                    content=response_body,
                    media_type=media_type,
                    headers=response_headers,
                )

            est_size = handler.estimate_pack_size(shas)
            logger.info(
                f"git_upload_pack: {len(shas)} objects, est ~{est_size} bytes, "
                f"threshold={HYPHA_GIT_STREAM_THRESHOLD}"
            )

            if est_size <= HYPHA_GIT_STREAM_THRESHOLD:
                # Buffered path: small/browser-compatible response w/ Content-Length.
                response_body = await handler.buffered_upload_pack_response(
                    shas, common, capabilities, shallow_shas=shallow_shas
                )
                return Response(
                    content=response_body,
                    media_type=media_type,
                    headers=response_headers,
                )

            # Streaming path: hand off the slot + repo to the generator.
            async def _stream():
                try:
                    async for chunk in handler.stream_upload_pack_response(
                        shas, common, capabilities, shallow_shas=shallow_shas
                    ):
                        yield chunk
                finally:
                    await cleanup_repo(repo)
                    sem.release()

            slot_handed_off = True
            return StreamingResponse(
                _stream(),
                media_type=media_type,
                headers=response_headers,
            )
        finally:
            if not slot_handed_off:
                # Buffered paths and errors: clean up here.
                await cleanup_repo(repo)
                sem.release()

    @router.post("/{workspace}/git/{alias}/git-receive-pack")
    async def git_receive_pack(
        workspace: str,
        alias: str,
        request: Request,
        user_info=Depends(git_basic_auth),
    ):
        """Handle git push requests."""
        # Strip .git suffix if present (Git clients may add it)
        if alias.endswith(".git"):
            alias = alias[:-4]

        if not user_info:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": 'Basic realm="Git Repository"'},
            )

        try:
            repo = await get_repo_callback(workspace, alias, user_info, write=True)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")
        except KeyError:
            raise HTTPException(status_code=404, detail="Repository not found")

        handler = GitHTTPHandler(repo)

        # Acquire a pack-op slot (bounds concurrent memory-heavy pushes).
        sem = await _acquire_pack_slot()
        try:
            # Stream the request body into a spool file (spills to disk past
            # PACK_SPOOL_FILE_MAX_SIZE) while parsing ref-update commands, then
            # parse/store the pack from the file. The status report is tiny and
            # returned buffered. Reading request.stream() here is safe because we
            # return a buffered Response (no StreamingResponse generator).
            logger.info("git_receive_pack: streaming request body to spool file")
            response_body = await handler.handle_receive_pack_streaming(request)
        finally:
            await cleanup_repo(repo)
            sem.release()

        return Response(
            content=response_body,
            media_type="application/x-git-receive-pack-result",
            headers={
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Expires": "Fri, 01 Jan 1980 00:00:00 GMT",
            },
        )

    return router
