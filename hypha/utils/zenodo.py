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

    async def create_deposition(self):
        """Creates a new empty deposition."""
        url = f"{self.zenodo_server}/api/deposit/depositions"
        response = await self.client.post(
            url, params=self.params, json={}, headers=self.headers
        )
        response.raise_for_status()

        deposition_info = response.json()
        return deposition_info

    async def load_deposition(self, deposition_id: str) -> "Deposition":
        """Loads an existing deposition by ID and returns a Deposition object."""
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}"
        response = await self.client.get(url, params=self.params)
        response.raise_for_status()

        deposition_info = response.json()
        return deposition_info

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

    async def create_new_version(self, record_id: str) -> "Deposition":
        """Creates a new version of an existing published record."""
        # Use the concept_id to initiate a new version
        url = f"{self.zenodo_server}/api/deposit/depositions/{record_id}/actions/newversion"
        response = await self.client.post(url, params=self.params)
        response.raise_for_status()

        # Fetch the latest draft version from the latest_draft URL
        draft_url = response.json()["links"]["latest_draft"]
        draft_response = await self.client.get(draft_url, params=self.params)
        draft_response.raise_for_status()

        draft_info = draft_response.json()
        return draft_info

    async def delete_deposition(self, deposition_id: str) -> None:
        """Deletes a deposition. Only unpublished depositions can be deleted."""
        url = f"{self.zenodo_server}/api/deposit/depositions/{deposition_id}"
        response = await self.client.delete(url, params=self.params)
        response.raise_for_status()


class Deposition:
    def __init__(self, zenodo_client: ZenodoClient, deposition_info: Dict[str, Any]):
        self.zenodo_client = zenodo_client
        self.deposition_id = str(deposition_info["id"])
        self.bucket_url = deposition_info["links"]["bucket"]
        self.doi = deposition_info["metadata"].get("prereserve_doi", {}).get("doi")
        self.concept_doi = deposition_info.get("conceptrecid")

    async def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Updates metadata for this deposition."""
        url = f"{self.zenodo_client.zenodo_server}/api/deposit/depositions/{self.deposition_id}"
        response = await self.zenodo_client.client.put(
            url,
            params=self.zenodo_client.params,
            json={"metadata": metadata},
            headers=self.zenodo_client.headers,
        )
        response.raise_for_status()

    async def list_files(self) -> List[Dict[str, Any]]:
        """Lists all files in this deposition."""
        url = f"{self.zenodo_client.zenodo_server}/api/deposit/depositions/{self.deposition_id}/files"
        response = await self.zenodo_client.client.get(
            url, params=self.zenodo_client.params
        )
        response.raise_for_status()
        return response.json()

    async def add_file(self, file_path: str) -> None:
        """Uploads a file to this deposition using the new bucket API."""
        filename = PurePosixPath(file_path).name

        async def file_chunk_reader(file_path: str, chunk_size: int = 1024):
            async with aiofiles.open(file_path, "rb") as file_data:
                while True:
                    chunk = await file_data.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        await self.zenodo_client.client.put(
            f"{self.bucket_url}/{filename}",
            params=self.zenodo_client.params,
            data=file_chunk_reader(file_path),
        )

    async def delete_file(self, file_id: str) -> None:
        """Deletes a file from this deposition."""
        url = f"{self.zenodo_client.zenodo_server}/api/deposit/depositions/{self.deposition_id}/files/{file_id}"
        response = await self.zenodo_client.client.delete(
            url, params=self.zenodo_client.params
        )
        response.raise_for_status()

    async def publish(self) -> None:
        """Publishes this deposition."""
        url = f"{self.zenodo_client.zenodo_server}/api/deposit/depositions/{self.deposition_id}/actions/publish"
        response = await self.zenodo_client.client.post(
            url, params=self.zenodo_client.params
        )
        if response.status_code == 400:
            raise RuntimeError(
                f"Failed to publish deposition: {response.json()}, you might forgot to update the metadata."
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
    deposition = Deposition(zenodo_client, deposition_info)
    print(f"Created new deposition with ID: {deposition.deposition_id}")

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
    await deposition.update_metadata(metadata)

    # Add a file to the deposition
    await deposition.add_file("setup.py")

    # Publish the deposition
    record = await deposition.publish()
    print(f"Published deposition with DOI: {deposition.doi}")

    record_id = record["id"]
    deposition_info = await zenodo_client.create_new_version(record_id)
    new_version = Deposition(zenodo_client, deposition_info)
    print(f"Created new version with ID: {new_version.deposition_id}")

    await new_version.update_metadata(metadata)

    # Add another file to the new version
    await new_version.add_file("requirements.txt")

    # Publish the new version
    await new_version.publish()
    print(f"Published new version with DOI: {new_version.doi}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
