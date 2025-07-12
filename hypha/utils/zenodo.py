import httpx
import asyncio
from typing import Dict, Any, List, Optional
import aiofiles
from pathlib import PurePosixPath
from hypha.utils import sanitize_url_for_logging

ZENODO_TIMEOUT = 30  # seconds


class ZenodoClient:
    def __init__(
        self, access_token: str, zenodo_server: str = "https://sandbox.zenodo.org"
    ):
        self.access_token = access_token
        self.zenodo_server = zenodo_server
        self.headers = {"Content-Type": "application/json"}
        self.params = {"access_token": self.access_token}
        self.client = httpx.AsyncClient(
            headers={"Connection": "close"}, timeout=ZENODO_TIMEOUT
        )

    def _sanitize_error(self, error: Exception) -> Exception:
        """Sanitize HTTP errors to remove access tokens from URLs."""
        if isinstance(error, httpx.HTTPStatusError):
            # Create a sanitized URL without the access token
            request = error.request
            url = str(request.url)
            sanitized_url = sanitize_url_for_logging(url)

            # Create a new error message with sanitized URL
            sanitized_message = f"Client error '{error.response.status_code} {error.response.reason_phrase}' for url '{sanitized_url}'"

            # Create a new HTTPStatusError with sanitized message
            new_error = httpx.HTTPStatusError(
                sanitized_message, request=request, response=error.response
            )
            return new_error
        return error

    async def _make_request(self, method: str, url: str, **kwargs):
        """Make HTTP request with error sanitization."""
        try:
            response = await getattr(self.client, method.lower())(url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            # Raise the sanitized error without chaining to hide original error with tokens
            raise self._sanitize_error(e) from None

    async def create_deposition(self) -> Dict[str, Any]:
        """Creates a new empty deposition and returns its info."""
        url = f"{self.zenodo_server}/api/deposit/depositions"
        response = await self._make_request(
            "POST", url, params=self.params, json={}, headers=self.headers
        )
        return response.json()

    async def load_deposition(self, deposition_id: str) -> Dict[str, Any]:
        """Loads an existing deposition by ID and returns its information."""
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}"
        response = await self._make_request("GET", url, params=self.params)
        return response.json()

    async def load_published_record(self, record_id: str) -> Optional[str]:
        """Loads an existing published record to retrieve the concept_id."""
        url = f"{self.zenodo_server}/api/records/{record_id}"
        response = await self._make_request("GET", url, follow_redirects=True)

        record_info = response.json()
        concept_id = record_info.get("conceptrecid")
        if not concept_id:
            raise ValueError(f"Record {record_id} does not have a conceptrecid.")
        return concept_id

    async def create_new_version(self, record_id: str) -> Dict[str, Any]:
        """Creates a new version of an existing published record."""
        url = f"{self.zenodo_server}/api/deposit/depositions/{record_id}/actions/newversion"
        response = await self._make_request("POST", url, params=self.params)

        draft_url = response.json()["links"]["latest_draft"]
        draft_response = await self._make_request("GET", draft_url, params=self.params)
        return draft_response.json()

    async def update_metadata(
        self, deposition_info: Dict[str, Any], metadata: Dict[str, Any]
    ) -> None:
        """Updates metadata for a specified deposition."""
        deposition_id = deposition_info["id"]
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}"
        await self._make_request(
            "PUT",
            url,
            params=self.params,
            json={"metadata": metadata},
            headers=self.headers,
        )

    async def list_files(self, deposition_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Lists all files in a specified deposition."""
        deposition_id = deposition_info["id"]
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}/files"
        response = await self._make_request("GET", url, params=self.params)
        return response.json()

    async def add_file(self, deposition_info: Dict[str, Any], file_path: str) -> None:
        """Uploads a file to the deposition using the bucket API."""
        bucket_url = deposition_info["links"]["bucket"]
        filename = PurePosixPath(file_path).name

        async def file_chunk_reader(file_path: str, chunk_size: int = 1024):
            async with aiofiles.open(file_path, "rb") as file_data:
                while True:
                    chunk = await file_data.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        try:
            await self.client.put(
                f"{bucket_url}/{filename}",
                params=self.params,
                data=file_chunk_reader(file_path),
            )
        except httpx.HTTPStatusError as e:
            raise self._sanitize_error(e) from None

    async def import_file(
        self, deposition_info, name, source_url, chunk_size: int = 8192
    ):
        """
        Import a file from a URL to Zenodo using streaming to handle large files efficiently.

        Args:
            deposition_info: Zenodo deposition information
            name: Name of the file in the deposition
            source_url: URL to stream the file from
            chunk_size: Size of chunks to stream (default 8192 bytes for better performance)
        """
        bucket_url = deposition_info["links"]["bucket"]
        target_url = f"{bucket_url}/{name}"

        # Use direct streaming approach to avoid S3 proxy issues
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("GET", source_url) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"Failed to download file from {source_url}: {response.status_code}"
                    )

                # Stream the file content to Zenodo with configurable chunk size
                await client.put(
                    target_url,
                    params=self.params,
                    content=response.aiter_bytes(chunk_size),
                    headers={"Content-Type": "application/octet-stream"},
                )

    async def delete_deposition(self, deposition_id: str) -> None:
        """Deletes a deposition. Only unpublished depositions can be deleted."""
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}"
        await self._make_request("DELETE", url, params=self.params)

    async def publish(self, deposition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Publishes the deposition."""
        deposition_id = deposition_info["id"]
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}/actions/publish"
        try:
            response = await self._make_request("POST", url, params=self.params)
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                # Get the response data but don't include the original error which might contain tokens
                try:
                    response_data = e.response.json()
                except:
                    response_data = {
                        "status": e.response.status_code,
                        "message": "Bad request",
                    }
                raise RuntimeError(
                    f"Failed to publish deposition: {response_data}, you might have forgotten to update the metadata."
                ) from None
            raise self._sanitize_error(e) from None


# Sample usage
async def main():
    import os

    # Initialize the Zenodo client
    zenodo_client = ZenodoClient(
        access_token=os.environ["SANDBOX_ZENODO_TOKEN"],
        zenodo_server="https://sandbox.zenodo.org",
    )

    # Create a new deposition
    deposition_info = await zenodo_client.create_deposition()
    print(f"Created new deposition with ID: {deposition_info['id']}")

    # Update metadata
    metadata = {
        "title": "My Dataset",
        "upload_type": "dataset",
        "description": "Description of my dataset.",
        "creators": [{"name": "Doe, John", "affiliation": "My Institution"}],
        "access_right": "open",
        "license": "cc-by",
        "keywords": ["example", "dataset"],
    }
    await zenodo_client.update_metadata(deposition_info, metadata)

    # Add a file to the deposition
    await zenodo_client.add_file(deposition_info, "setup.py")

    # Publish the deposition
    record = await zenodo_client.publish(deposition_info)
    print(f"Published deposition with DOI: {record['doi']}")

    # Create a new version of the published record
    record_id = record["id"]
    deposition_info = await zenodo_client.create_new_version(record_id)
    print(f"Created new version with ID: {deposition_info['id']}")

    # Update metadata for the new version
    await zenodo_client.update_metadata(deposition_info, metadata)

    # Add another file to the new version
    await zenodo_client.add_file(deposition_info, "requirements.txt")

    # Publish the new version
    record = await zenodo_client.publish(deposition_info)
    print(f"Published new version with DOI: {record['doi']}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
