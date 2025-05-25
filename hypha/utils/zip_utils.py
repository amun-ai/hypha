"""Utilities for working with ZIP files.

This module provides utilities for working with ZIP files stored in S3, particularly
for streaming the content of files from inside ZIP archives without having to download
the entire ZIP file first (for large archives).

The main functions are:
- fetch_zip_tail: Retrieve just the central directory of a ZIP file to read its structure
- get_zip_file_content: Stream content from a specific file inside a ZIP or list directory contents
- read_file_from_zip: Extract a single file from a ZIP archive
- stream_file_from_zip: Stream a file from a ZIP archive in chunks

The implementation uses an adaptive approach to handle ZIP files of different sizes:
- For small files (<50MB), it downloads the entire ZIP file for simplicity and reliability
- For larger files, it attempts to fetch just the central directory (tail) of the ZIP file
- The implementation supports both standard ZIP and ZIP64 formats
- For very large files, it implements streaming decompression to avoid loading the entire
  file content into memory at once
"""

import asyncio
import logging
import struct
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union, AsyncGenerator
import zipfile
from botocore.exceptions import ClientError
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.responses import Response
from datetime import datetime
import os
import zlib

# Setup logger
logger = logging.getLogger("zip_utils")


async def fetch_zip_tail(
    s3_client, bucket: str, key: str, content_length: int, cache_instance=None
) -> bytes:
    """
    Fetch the tail part of the zip file that contains the central directory.
    Uses an adaptive approach to find the central directory, supporting both standard ZIP and ZIP64 formats.

    Args:
        s3_client: S3 client to use for fetching
        bucket: S3 bucket name
        key: S3 object key
        content_length: Total size of the ZIP file
        cache_instance: Optional Redis cache instance

    Returns:
        bytes: The ZIP file tail containing the central directory
    """
    # Check cache first if provided
    if cache_instance:
        cache_key = f"zip_tail:{bucket}:{key}:{content_length}"
        cached_tail = await cache_instance.get(cache_key)
        if cached_tail:
            logger.debug(f"Using cached ZIP tail for {key}")
            return cached_tail

    # For small files (< 50MB), just download the entire file
    # This is more reliable for tests and small files
    if content_length < 50 * 1024 * 1024:  # < 50MB
        try:
            logger.debug(
                f"File is small ({content_length} bytes), downloading entire ZIP file"
            )
            response = await s3_client.get_object(Bucket=bucket, Key=key)
            complete_zip = await response["Body"].read()

            # Verify this is a valid ZIP file
            try:
                with zipfile.ZipFile(BytesIO(complete_zip)) as zf:
                    # Just checking if it opens correctly
                    num_files = len(zf.namelist())
                    logger.debug(
                        f"Successfully loaded complete ZIP file ({len(complete_zip)} bytes) with {num_files} files"
                    )

                # Cache the entire file
                if cache_instance:
                    logger.debug(f"Caching entire ZIP file for {key}")
                    await cache_instance.set(
                        cache_key, complete_zip, ttl=3600
                    )  # Cache for 1 hour

                return complete_zip
            except zipfile.BadZipFile as e:
                logger.error(f"Downloaded file is not a valid ZIP file: {str(e)}")
                # Continue with partial download approach

        except Exception as e:
            logger.error(f"Failed to download entire ZIP file: {str(e)}")
            # Continue with partial download approach

    # ZIP End of Central Directory (EOCD) signature
    EOCD_SIGNATURE = b"PK\x05\x06"
    ZIP64_EOCD_LOCATOR = (
        b"PK\x06\x07"  # ZIP64 End of Central Directory Locator signature
    )

    # Adaptive tail size based on file size
    MAX_TAIL = 128 * 1024 * 1024  # 128MB absolute maximum for very large ZIP files

    # Determine starting tail size based on file size
    # For small files (<10MB), start with 64KB
    # For medium files (10MB-100MB), start with 256KB
    # For large files (>100MB), start with 1MB
    # For very large files (>1GB), start with 4MB
    if content_length < 10 * 1024 * 1024:  # < 10MB
        min_tail = 64 * 1024  # 64KB
    elif content_length < 100 * 1024 * 1024:  # < 100MB
        min_tail = 256 * 1024  # 256KB
    elif content_length < 1024 * 1024 * 1024:  # < 1GB
        min_tail = 1024 * 1024  # 1MB
    else:
        min_tail = 4 * 1024 * 1024  # 4MB

    # Limit max tail to 10% of file size for small/medium files to avoid overhead
    if content_length < 100 * 1024 * 1024:
        MAX_TAIL = min(MAX_TAIL, max(content_length // 10, min_tail * 2))

    tail_length = min_tail
    found = False
    zip_tail = b""
    current_offset = content_length

    # Adaptive approach: start small and increase until we find the central directory
    while tail_length <= MAX_TAIL:
        central_directory_offset = max(content_length - tail_length, 0)
        # Only fetch the new part if we've already read a smaller tail
        if zip_tail and central_directory_offset < current_offset:
            # Fetch only the missing part
            range_header = f"bytes={central_directory_offset}-{current_offset - 1}"
            logger.debug(f"Fetching additional ZIP tail: {range_header}")
            response = await s3_client.get_object(
                Bucket=bucket, Key=key, Range=range_header
            )
            new_part = await response["Body"].read()
            zip_tail = new_part + zip_tail
            current_offset = central_directory_offset
        else:
            range_header = f"bytes={central_directory_offset}-{content_length - 1}"
            logger.debug(f"Fetching ZIP tail: {range_header}")
            response = await s3_client.get_object(
                Bucket=bucket, Key=key, Range=range_header
            )
            zip_tail = await response["Body"].read()
            current_offset = central_directory_offset

        # First check for regular ZIP EOCD signature
        eocd_offset = zip_tail.rfind(EOCD_SIGNATURE)

        # Also check for ZIP64 format
        zip64_locator_offset = zip_tail.rfind(ZIP64_EOCD_LOCATOR)

        if eocd_offset != -1:
            # Found standard ZIP EOCD
            found = True
            logger.debug(
                f"Found EOCD at offset {eocd_offset} in tail of size {len(zip_tail)}"
            )
            break
        elif zip64_locator_offset != -1:
            # Found ZIP64 EOCD Locator
            found = True
            logger.debug(
                f"Found ZIP64 EOCD Locator at offset {zip64_locator_offset} in tail of size {len(zip_tail)}"
            )
            break

        # Double the tail size and try again
        tail_length *= 2
        logger.debug(f"Increasing tail size to {tail_length}")

    # If we still haven't found the central directory, try one final approach: get the whole file
    # This is only for small enough files where it's reasonable to do so
    if not found and content_length <= 100 * 1024 * 1024:  # Only for files <= 100MB
        logger.warning(
            f"Central directory not found in tail, downloading entire ZIP file for {key}"
        )
        try:
            response = await s3_client.get_object(Bucket=bucket, Key=key)
            zip_tail = await response["Body"].read()

            # Verify this is a valid ZIP file
            try:
                with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
                    # Just checking if it opens correctly
                    found = True
                    logger.debug(
                        f"Successfully loaded complete ZIP file ({len(zip_tail)} bytes)"
                    )
            except zipfile.BadZipFile:
                logger.error(f"File is not a valid ZIP file: {key}")
                # Continue with what we have anyway
        except Exception as e:
            logger.error(f"Failed to download entire ZIP file: {str(e)}")
            # Continue with what we have

    if not found:
        # Fallback: return the largest tail we got, but warn
        logger.warning(
            f"ZIP directory signatures not found in last {tail_length//2} bytes of zip file {key}. Returning largest tail."
        )

    # Verify the tail can be opened as a ZIP file before returning
    try:
        with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
            num_files = len(zf.namelist())
            logger.debug(f"Validated ZIP tail contains {num_files} files")
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file structure: {str(e)}")
        # We'll still return what we have, the caller will handle the error

    # Cache result if cache instance provided and we found something valid
    if cache_instance and found:
        logger.debug(f"Caching ZIP tail for {key}")
        await cache_instance.set(cache_key, zip_tail, ttl=3600)  # Cache for 1 hour

    return zip_tail


async def get_zip_file_content(
    s3_client,
    bucket: str,
    key: str,
    path: str = "",
    zip_tail: Optional[bytes] = None,
    cache_instance=None,
) -> Response:
    """
    Extract and stream content from a ZIP file stored in S3.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        key: S3 object key for the ZIP file
        path: Path within the ZIP file to retrieve
        zip_tail: Optional pre-fetched ZIP tail data
        cache_instance: Optional Redis cache instance

    Returns:
        Response: Either a StreamingResponse with file content or a JSONResponse with directory listing
    """
    try:
        # Fetch the ZIP file metadata to get its size
        try:
            zip_file_metadata = await s3_client.head_object(Bucket=bucket, Key=key)
            content_length = zip_file_metadata["ContentLength"]
            logger.info(f"[zip] ZIP file size: {content_length} bytes")
        except ClientError as e:
            # Check if the error is due to the file not being found (404)
            if e.response["Error"]["Code"] == "404":
                logger.error(f"[zip] ZIP file not found: {key}")
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": f"ZIP file not found: {key}"},
                )
            else:
                # For other types of errors, raise a 500 error
                logger.error(f"[zip] Failed to fetch file metadata: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "detail": f"Failed to fetch file metadata: {str(e)}",
                    },
                )

        # Fetch the ZIP's central directory if not provided
        if not zip_tail:
            try:
                zip_tail = await fetch_zip_tail(
                    s3_client, bucket, key, content_length, cache_instance
                )
            except Exception as e:
                logger.error(f"[zip] Failed to fetch ZIP tail: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "detail": f"Failed to fetch ZIP file structure: {str(e)}",
                    },
                )

        # Open the in-memory ZIP tail and parse it
        try:
            with zipfile.ZipFile(BytesIO(zip_tail)) as zip_file:
                # If `path` ends with "/" or is empty, treat it as a directory listing
                if not path or path.endswith("/"):
                    logger.info(f"[zip] Listing directory: path={path}")
                    directory_contents = []

                    for zip_info in zip_file.infolist():
                        # Handle root directory or subdirectory files
                        if not path:
                            # Root directory - extract immediate children
                            relative_path = zip_info.filename.strip("/")
                            if "/" not in relative_path:
                                # Top-level file
                                directory_contents.append(
                                    {
                                        "type": (
                                            "file"
                                            if not zip_info.is_dir()
                                            else "directory"
                                        ),
                                        "name": relative_path,
                                        "size": (
                                            zip_info.file_size
                                            if not zip_info.is_dir()
                                            else None
                                        ),
                                        "last_modified": datetime(
                                            *zip_info.date_time
                                        ).timestamp(),
                                    }
                                )
                            else:
                                # Top-level directory
                                top_level_dir = relative_path.split("/")[0]
                                if not any(
                                    d["name"] == top_level_dir
                                    and d["type"] == "directory"
                                    for d in directory_contents
                                ):
                                    directory_contents.append(
                                        {"type": "directory", "name": top_level_dir}
                                    )
                        else:
                            # Subdirectory: Include only immediate children
                            if (
                                zip_info.filename.startswith(path)
                                and zip_info.filename != path
                            ):
                                relative_path = zip_info.filename[len(path) :].strip(
                                    "/"
                                )
                                if "/" in relative_path:
                                    # Subdirectory case
                                    child_name = relative_path.split("/")[0]
                                    if not any(
                                        d["name"] == child_name
                                        and d["type"] == "directory"
                                        for d in directory_contents
                                    ):
                                        directory_contents.append(
                                            {"type": "directory", "name": child_name}
                                        )
                                else:
                                    # File case
                                    directory_contents.append(
                                        {
                                            "type": "file",
                                            "name": relative_path,
                                            "size": zip_info.file_size,
                                            "last_modified": datetime(
                                                *zip_info.date_time
                                            ).timestamp(),
                                        }
                                    )

                    logger.info(
                        f"[zip] Directory listing complete, returning {len(directory_contents)} items"
                    )
                    return JSONResponse(status_code=200, content=directory_contents)

                # Otherwise, find the file inside the ZIP
                try:
                    zip_info = zip_file.getinfo(path)
                    logger.info(
                        f"[zip] Found file in ZIP: {path}, size: {zip_info.file_size}, compression: {zip_info.compress_type}"
                    )
                except KeyError:
                    logger.error(f"[zip] File not found inside ZIP: {path}")
                    return JSONResponse(
                        status_code=404,
                        content={
                            "success": False,
                            "detail": f"File not found inside ZIP: {path}",
                        },
                    )

                # Set content type based on file extension if possible
                content_type = "application/octet-stream"
                file_extension = os.path.splitext(path)[1].lower()
                if file_extension:
                    # Basic content type mapping
                    content_type_map = {
                        ".txt": "text/plain",
                        ".html": "text/html",
                        ".htm": "text/html",
                        ".json": "application/json",
                        ".xml": "application/xml",
                        ".csv": "text/csv",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".gif": "image/gif",
                        ".pdf": "application/pdf",
                        ".js": "application/javascript",
                        ".css": "text/css",
                        ".md": "text/markdown",
                    }
                    content_type = content_type_map.get(file_extension, content_type)

                # Use our streaming implementation with proper chunking
                # Check file size to decide on the best chunk size for streaming
                if zip_info.file_size > 10 * 1024 * 1024:  # > 10MB
                    # For larger files, use larger chunks to improve throughput
                    chunk_size = 256 * 1024  # 256KB
                else:
                    # For smaller files, use smaller chunks for lower latency
                    chunk_size = 64 * 1024  # 64KB

                logger.info(
                    f"[zip] Returning StreamingResponse for {path} with chunk size {chunk_size}"
                )

                # Stream the file data
                return StreamingResponse(
                    stream_file_from_zip(
                        s3_client, bucket, key, zip_info, zip_tail, chunk_size
                    ),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{os.path.basename(path)}"',
                        "Content-Length": str(
                            zip_info.file_size
                        ),  # Add file size for client progress tracking
                    },
                )
        except zipfile.BadZipFile as e:
            logger.error(f"[zip] Bad ZIP file: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "detail": f"Invalid ZIP file: {str(e)}"},
            )

    except Exception as e:
        logger.error(f"[zip] Unhandled exception in get_zip_file_content: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Internal server error: {str(e)}"},
        )


async def read_file_from_zip(s3_client, bucket, key, zip_info, zip_tail):
    """
    Extract a file from a ZIP archive efficiently by directly reading the compressed data
    and decompressing it on the fly.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        key: S3 object key for the ZIP file
        zip_info: ZipInfo object for the file to extract
        zip_tail: The ZIP central directory data

    Returns:
        bytes: The decompressed file content
    """
    try:
        # For small files, use the zipfile module directly for simplicity
        if zip_info.file_size < 10 * 1024 * 1024:  # < 10MB
            with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
                return zf.read(zip_info.filename)

        # For larger files, use streaming approach
        chunks = []
        async for chunk in stream_file_chunks_from_zip(
            s3_client, bucket, key, zip_info, zip_tail
        ):
            chunks.append(chunk)
        return b"".join(chunks)
    except Exception as e:
        logger.error(f"Error reading file from ZIP: {str(e)}")
        raise


async def stream_file_from_zip(
    s3_client, bucket, key, zip_info, zip_tail, chunk_size=64 * 1024
):
    """
    Stream a file from a ZIP archive, yielding chunks of decompressed data.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        key: S3 object key for the ZIP file
        zip_info: ZipInfo object for the file to extract
        zip_tail: The ZIP central directory data
        chunk_size: Size of decompressed chunks to yield

    Yields:
        bytes: Chunks of decompressed file data
    """
    try:
        # For small files or stored files, use the simpler approach
        if (
            zip_info.file_size < 1 * 1024 * 1024
            or zip_info.compress_type == zipfile.ZIP_STORED
        ):  # < 1MB
            with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
                file_content = zf.read(zip_info.filename)
                for i in range(0, len(file_content), chunk_size):
                    yield file_content[i : i + chunk_size]
            return

        # For larger files, use the streaming implementation
        async for chunk in stream_file_chunks_from_zip(
            s3_client, bucket, key, zip_info, zip_tail, chunk_size
        ):
            yield chunk
    except Exception as e:
        logger.error(f"Error streaming file from ZIP: {str(e)}")
        raise


async def stream_file_chunks_from_zip(
    s3_client, bucket, key, zip_info, zip_tail, chunk_size=64 * 1024
):
    """
    Stream a file from a ZIP archive using true streaming without loading the entire file into memory.
    This fetches compressed data in chunks and decompresses on-the-fly.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        key: S3 object key for the ZIP file
        zip_info: ZipInfo object for the file to extract
        zip_tail: The ZIP central directory data
        chunk_size: Size of decompressed chunks to yield

    Yields:
        bytes: Chunks of decompressed file data
    """
    # Get the local header offset from the central directory
    file_header_offset = zip_info.header_offset

    # Calculate data offset: header offset + size of the local file header + filename length + extra field length
    try:
        with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
            # Get header data to calculate size
            file_header = zf._RealGetContents()[zip_info.filename]
            local_header_size = 30  # Fixed size of the local file header
            filename_length = len(zip_info.filename)
            extra_field_length = len(zip_info.extra)

            # Calculate the offset where the actual file data begins
            data_offset = (
                file_header_offset
                + local_header_size
                + filename_length
                + extra_field_length
            )

            compression_method = zip_info.compress_type
            compressed_size = zip_info.compress_size
            uncompressed_size = zip_info.file_size

            logger.debug(
                f"Streaming file {zip_info.filename}, "
                f"compression: {compression_method}, "
                f"compressed size: {compressed_size}, "
                f"uncompressed size: {uncompressed_size}, "
                f"data offset: {data_offset}"
            )
    except Exception as e:
        logger.error(f"Error calculating data offset: {str(e)}")
        # Fall back to simple approach if we can't calculate the offset
        with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
            file_content = zf.read(zip_info.filename)
            for i in range(0, len(file_content), chunk_size):
                yield file_content[i : i + chunk_size]
        return

    # For DEFLATE compression, we need to use zlib to decompress
    if compression_method == zipfile.ZIP_DEFLATED:
        decompressor = zlib.decompressobj(-15)  # -15 for raw deflate data

        # Fetch compressed data in chunks and decompress on-the-fly
        # Optimize chunk size based on compression ratio
        fetch_size = max(
            64 * 1024,
            min(1 * 1024 * 1024, int(chunk_size * compressed_size / uncompressed_size)),
        )

        # Stream the compressed data
        bytes_read = 0
        buffer = b""  # Buffer for decompressed data

        while bytes_read < compressed_size:
            # Calculate the byte range to fetch
            start = data_offset + bytes_read
            end = min(start + fetch_size - 1, data_offset + compressed_size - 1)
            range_header = f"bytes={start}-{end}"

            try:
                # Fetch a chunk of compressed data
                response = await s3_client.get_object(
                    Bucket=bucket, Key=key, Range=range_header
                )

                # Read the data chunk
                chunk = await response["Body"].read()
                bytes_read += len(chunk)

                # Decompress the chunk
                if bytes_read >= compressed_size:
                    # Last chunk - use flush
                    decompressed = (
                        buffer + decompressor.decompress(chunk) + decompressor.flush()
                    )
                else:
                    decompressed = buffer + decompressor.decompress(chunk)

                # Yield complete chunks
                for i in range(0, len(decompressed) - chunk_size + 1, chunk_size):
                    yield decompressed[i : i + chunk_size]

                # Keep remainder in buffer
                buffer = decompressed[
                    len(decompressed) - (len(decompressed) % chunk_size) :
                ]

            except Exception as e:
                logger.error(f"Error fetching or decompressing chunk: {str(e)}")
                raise

        # Yield any remaining data in the buffer
        if buffer:
            yield buffer

    elif compression_method == zipfile.ZIP_STORED:
        # For uncompressed files, directly stream the data
        bytes_read = 0
        while bytes_read < uncompressed_size:
            # Calculate the byte range to fetch
            start = data_offset + bytes_read
            end = min(start + chunk_size - 1, data_offset + uncompressed_size - 1)
            range_header = f"bytes={start}-{end}"

            try:
                # Fetch a chunk of data
                response = await s3_client.get_object(
                    Bucket=bucket, Key=key, Range=range_header
                )

                # Read and yield the data chunk
                chunk = await response["Body"].read()
                bytes_read += len(chunk)
                yield chunk

            except Exception as e:
                logger.error(f"Error fetching chunk: {str(e)}")
                raise
    else:
        # For other compression methods, fall back to the zipfile module
        logger.info(
            f"Unsupported compression method {compression_method}, falling back to zipfile module"
        )
        with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
            file_content = zf.read(zip_info.filename)
            for i in range(0, len(file_content), chunk_size):
                yield file_content[i : i + chunk_size]
