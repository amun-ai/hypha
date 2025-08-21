#!/usr/bin/env python
"""Test script to verify the ZIP file fix works for large files."""

import asyncio
import zipfile
from io import BytesIO
import json
import sys
import os
import tempfile
import struct

sys.path.insert(0, '.')

from hypha.utils import zip_utils


async def test_large_zip_streaming():
    """Test streaming from large ZIP files that require partial loading."""
    
    # Create a test ZIP file 
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Add test files
        zf.writestr('data.zarr/.zattrs', json.dumps({
            "multiscales": [{"version": "0.4"}],
            "omero": {"channels": []},
        }))
        
        # Add many chunk files to make it larger
        chunk_data = b"x" * (50 * 1024)  # 50KB per chunk
        for i in range(100):
            zf.writestr(f'data.zarr/chunk_{i}', chunk_data)
    
    zip_buffer.seek(0)
    zip_content = zip_buffer.getvalue()
    
    print(f"Created ZIP file with size: {len(zip_content)} bytes")
    
    # Save the ZIP to a temporary file to simulate S3 storage
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
        tmp_file.write(zip_content)
        tmp_path = tmp_file.name
    
    try:
        # Create a mock S3 client that reads from the local file
        class MockS3Client:
            async def head_object(self, Bucket, Key):
                return {"ContentLength": len(zip_content)}
            
            async def get_object(self, Bucket, Key, Range=None):
                class Body:
                    def __init__(self, data):
                        self.data = data
                    
                    async def read(self):
                        return self.data
                
                if Range:
                    # Parse the range header
                    range_str = Range.replace("bytes=", "")
                    start, end = range_str.split("-")
                    start = int(start)
                    end = int(end) + 1 if end else len(zip_content)
                    
                    # Read the specified range from the file
                    with open(tmp_path, 'rb') as f:
                        f.seek(start)
                        data = f.read(end - start)
                    return {"Body": Body(data)}
                else:
                    with open(tmp_path, 'rb') as f:
                        data = f.read()
                    return {"Body": Body(data)}
        
        s3_client = MockS3Client()
        
        # Test fetching the ZIP tail
        print("\nFetching ZIP tail...")
        zip_tail = await zip_utils.fetch_zip_tail(
            s3_client, "test-bucket", "test.zip", len(zip_content)
        )
        print(f"ZIP tail size: {len(zip_tail)} bytes")
        
        # Try to read a file from the ZIP
        with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
            zip_info = zf.getinfo('data.zarr/.zattrs')
            print(f"\nTesting file: {zip_info.filename}")
            print(f"Header offset: {zip_info.header_offset}")
            print(f"Compressed size: {zip_info.compress_size}")
            
            # Test the streaming function
            print("\nTesting stream_file_chunks_from_zip...")
            chunks = []
            async for chunk in zip_utils.stream_file_chunks_from_zip(
                s3_client, "test-bucket", "test.zip", zip_info, zip_tail
            ):
                chunks.append(chunk)
            
            content = b"".join(chunks)
            data = json.loads(content)
            print(f"Successfully read file content: {data}")
            
            assert "multiscales" in data, "Missing multiscales in content"
            print("\n✅ Test passed! The fix works correctly.")
            
    finally:
        # Clean up
        os.unlink(tmp_path)


if __name__ == "__main__":
    asyncio.run(test_large_zip_streaming())