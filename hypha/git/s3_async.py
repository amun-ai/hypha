"""Async S3 operations using aiohttp with AWS Signature V4.

This module provides async S3 upload functionality that works with MinIO,
bypassing aiobotocore's put_object which hangs with MinIO due to known
compatibility issues.

The implementation uses aiohttp directly with AWS Signature V4 signing,
providing a truly async alternative to sync boto3 operations.
"""

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urlparse

import aiohttp

logger = logging.getLogger(__name__)


def _sign(key: bytes, msg: str) -> bytes:
    """Create HMAC-SHA256 signature."""
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(
    secret_key: str, date_stamp: str, region: str, service: str
) -> bytes:
    """Create AWS Signature V4 signing key."""
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    return k_signing


def _uri_encode(s: str, encode_slash: bool = True) -> str:
    """URI-encode a string per AWS Signature V4 spec.

    AWS Signature V4 requires specific URI encoding:
    - Unreserved chars (A-Z, a-z, 0-9, '-', '.', '_', '~') are not encoded
    - All other chars are percent-encoded
    - Slashes are encoded unless encode_slash=False

    Args:
        s: String to encode
        encode_slash: Whether to encode forward slashes

    Returns:
        URI-encoded string
    """
    if encode_slash:
        # Encode everything including slashes
        return quote(s, safe="-_.~")
    else:
        # Don't encode slashes (for paths)
        return quote(s, safe="-_.~/")


def _create_authorization_header(
    method: str,
    endpoint_url: str,
    bucket: str,
    key: str,
    data: bytes,
    access_key: str,
    secret_key: str,
    region: str = "us-east-1",
) -> tuple[dict[str, str], str]:
    """Create AWS Signature V4 authorization header.

    Args:
        method: HTTP method (PUT, DELETE, etc.)
        endpoint_url: S3 endpoint URL
        bucket: S3 bucket name
        key: Object key
        data: Request body data (empty bytes for DELETE)
        access_key: AWS access key
        secret_key: AWS secret key
        region: AWS region

    Returns:
        Tuple of (headers dict, full URL)
    """
    host = urlparse(endpoint_url).netloc

    # URL-encode the key for the URL (preserve slashes for path structure)
    encoded_key = _uri_encode(key, encode_slash=False)
    url = f"{endpoint_url}/{bucket}/{encoded_key}"

    # Current time
    t = datetime.now(timezone.utc)
    amz_date = t.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = t.strftime("%Y%m%d")

    # Payload hash
    payload_hash = hashlib.sha256(data).hexdigest()

    # Create canonical request
    # The canonical URI must be URI-encoded, preserving slashes
    canonical_uri = f"/{bucket}/{encoded_key}"
    canonical_querystring = ""
    canonical_headers = (
        f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = (
        f"{method}\n{canonical_uri}\n{canonical_querystring}\n"
        f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )

    # Create string to sign
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = (
        f"{algorithm}\n{amz_date}\n{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    )

    # Create signing key and signature
    signing_key = _get_signature_key(secret_key, date_stamp, region, "s3")
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # Create authorization header
    authorization = (
        f"{algorithm} Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    headers = {
        "Authorization": authorization,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "Content-Type": "application/octet-stream",
    }

    if method != "DELETE":
        headers["Content-Length"] = str(len(data))

    return headers, url


class AsyncS3Client:
    """Async S3 client using aiohttp with AWS Signature V4.

    This client provides async S3 upload/delete operations that work
    reliably with MinIO, bypassing aiobotocore's put_object issues.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region_name: str = "us-east-1",
    ):
        """Initialize async S3 client.

        Args:
            endpoint_url: S3 endpoint URL (e.g., "http://localhost:9000")
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region_name: AWS region name
        """
        self._endpoint_url = endpoint_url.rstrip("/")
        self._access_key = access_key_id
        self._secret_key = secret_access_key
        self._region = region_name
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            # Use trust_env=False to ignore HTTP_PROXY environment variables
            # which can cause issues with S3/MinIO operations
            self._session = aiohttp.ClientSession(connector=connector, trust_env=False)
        return self._session

    async def put_object(self, bucket: str, key: str, body: bytes) -> dict:
        """Upload an object to S3.

        Args:
            bucket: S3 bucket name
            key: Object key
            body: Object data

        Returns:
            Response metadata dict

        Raises:
            Exception: If upload fails
        """
        headers, url = _create_authorization_header(
            "PUT",
            self._endpoint_url,
            bucket,
            key,
            body,
            self._access_key,
            self._secret_key,
            self._region,
        )

        session = await self._get_session()
        async with session.put(url, data=body, headers=headers) as response:
            if response.status >= 400:
                text = await response.text()
                raise Exception(f"S3 PUT failed: {response.status} - {text}")
            logger.debug(f"PUT {key} succeeded with status {response.status}")
            return {"HTTPStatusCode": response.status}

    async def delete_object(self, bucket: str, key: str) -> dict:
        """Delete an object from S3.

        Args:
            bucket: S3 bucket name
            key: Object key

        Returns:
            Response metadata dict
        """
        headers, url = _create_authorization_header(
            "DELETE",
            self._endpoint_url,
            bucket,
            key,
            b"",
            self._access_key,
            self._secret_key,
            self._region,
        )

        session = await self._get_session()
        async with session.delete(url, headers=headers) as response:
            # DELETE returns 204 No Content on success
            if response.status >= 400:
                text = await response.text()
                raise Exception(f"S3 DELETE failed: {response.status} - {text}")
            logger.debug(f"DELETE {key} succeeded with status {response.status}")
            return {"HTTPStatusCode": response.status}

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


def create_async_s3_client(s3_config: dict) -> AsyncS3Client:
    """Create an async S3 client from config dict.

    Args:
        s3_config: Dict with endpoint_url, access_key_id, secret_access_key,
                   and optionally region_name

    Returns:
        AsyncS3Client instance
    """
    return AsyncS3Client(
        endpoint_url=s3_config.get("endpoint_url", ""),
        access_key_id=s3_config.get("access_key_id", ""),
        secret_access_key=s3_config.get("secret_access_key", ""),
        region_name=s3_config.get("region_name", "us-east-1"),
    )
