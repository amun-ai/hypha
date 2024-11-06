import httpx
import asyncio
from typing import Dict, Any, List, Optional
import aiofiles
from pathlib import PurePosixPath


class ZenodoClient:
    def __init__(
        self, access_token: str, zenodo_server: str = "https://sandbox.zenodo.org"
    ):
        self.access_token = access_token
        self.zenodo_server = zenodo_server
        self.headers = {"Content-Type": "application/json"}
        self.params = {"access_token": self.access_token}
        self.client = httpx.AsyncClient()

    async def create_deposition(self) -> Dict[str, Any]:
        """Creates a new empty deposition and returns its info."""
        url = f"{self.zenodo_server}/api/deposit/depositions"
        response = await self.client.post(
            url, params=self.params, json={}, headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    async def load_deposition(self, deposition_id: str) -> Dict[str, Any]:
        """Loads an existing deposition by ID and returns its information."""
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}"
        response = await self.client.get(url, params=self.params)
        response.raise_for_status()
        return response.json()

    async def load_published_record(self, record_id: str) -> Optional[str]:
        """Loads an existing published record to retrieve the concept_id."""
        url = f"{self.zenodo_server}/api/records/{record_id}"
        response = await self.client.get(url, follow_redirects=True)
        response.raise_for_status()

        record_info = response.json()
        concept_id = record_info.get("conceptrecid")
        if not concept_id:
            raise ValueError(f"Record {record_id} does not have a conceptrecid.")
        return concept_id

    async def create_new_version(self, record_id: str) -> Dict[str, Any]:
        """Creates a new version of an existing published record."""
        url = f"{self.zenodo_server}/api/deposit/depositions/{record_id}/actions/newversion"
        response = await self.client.post(url, params=self.params)
        response.raise_for_status()

        draft_url = response.json()["links"]["latest_draft"]
        draft_response = await self.client.get(draft_url, params=self.params)
        draft_response.raise_for_status()
        return draft_response.json()

    async def update_metadata(
        self, deposition_info: Dict[str, Any], metadata: Dict[str, Any]
    ) -> None:
        """Updates metadata for a specified deposition."""
        deposition_id = deposition_info["id"]
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}"
        response = await self.client.put(
            url,
            params=self.params,
            json={"metadata": metadata},
            headers=self.headers,
        )
        response.raise_for_status()

    async def list_files(self, deposition_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Lists all files in a specified deposition."""
        deposition_id = deposition_info["id"]
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}/files"
        response = await self.client.get(url, params=self.params)
        response.raise_for_status()
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

        await self.client.put(
            f"{bucket_url}/{filename}",
            params=self.params,
            data=file_chunk_reader(file_path),
        )

    async def import_file(self, deposition_info, name, target_url):
        bucket_url = deposition_info["links"]["bucket"]
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", target_url) as response:

                async def s3_response_chunk_reader(response, chunk_size: int = 2048):
                    async for chunk in response.aiter_bytes(chunk_size):
                        yield chunk

                put_response = await self.client.put(
                    f"{bucket_url}/{name}",
                    params=self.params,
                    headers={"Content-Type": "application/octet-stream"},
                    data=s3_response_chunk_reader(response),
                )
                put_response.raise_for_status()

    async def delete_deposition(self, deposition_id: str) -> None:
        """Deletes a deposition. Only unpublished depositions can be deleted."""
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}"
        response = await self.client.delete(url, params=self.params)
        response.raise_for_status()

    async def publish(self, deposition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Publishes the deposition."""
        deposition_id = deposition_info["id"]
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}/actions/publish"
        response = await self.client.post(url, params=self.params)
        if response.status_code == 400:
            raise RuntimeError(
                f"Failed to publish deposition: {response.json()}, you might have forgotten to update the metadata."
            )
        response.raise_for_status()
        return response.json()


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
