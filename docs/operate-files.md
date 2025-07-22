# Operating Files in S3 with Hypha

By default, Hypha does not include an S3 storage server. To enable S3 support, you need to provide credentials for an existing S3 storage (such as Minio) when starting Hypha. This document explains how to configure Hypha for S3 storage and operate files using different methods.

## Starting Hypha with S3 Support

To enable S3 storage in Hypha, use the following command when starting the server, specifying your S3 credentials:

```bash
python3 -m hypha.server \
  --enable-s3 \
  --access-key-id=MINIO_ROOT_USER \
  --secret-access-key=MINIO_ROOT_PASSWORD \
  --endpoint-url=MINIO_SERVER_URL \
  --endpoint-url-public=MINIO_SERVER_URL_PUBLIC \
  --s3-admin-type=minio
```

### Arguments:
- `--enable-s3`: Enables S3 object storage.
- `--access-key-id`: Access key for the S3 server.
- `--secret-access-key`: Secret key for the S3 server.
- `--endpoint-url`: The internal URL for Hypha to access the S3 server.
- `--endpoint-url-public`: Public URL for accessing the files (may be the same as `--endpoint-url`).
- `--s3-admin-type`: Specifies the S3 admin type (e.g., `minio`, `generic`). If it's a minio server (not gateway mode), use `minio`, otherwise use `generic` would be enough for most cases.
- `--enable-s3-proxy`: Enables the S3 proxy provided by Hypha, useful when the S3 server is not directly accessible.

If your `--endpoint-url-public` is not accessible from outside, you can enable the s3 proxy provided by hypha by adding `--enable-s3-proxy` to the command line. This will allow you to access the s3 files through the hypha server.

After starting Hypha with these arguments, you can verify that the S3 service is available by visiting the following URL:

```
https://hypha.aicell.io/public/services/s3-storage
```

Replace `https://hypha.aicell.io/` with your Hypha server URL. If everything is set up correctly, you should not encounter any errors on this page.

## Methods for Manipulating Files in S3

### Connecting to the Server and Getting the S3 Service

Before performing any operations, you need to connect to the Hypha server and get the S3 controller.

```python
from hypha_rpc import connect_to_server

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
workspace = api.config["workspace"]
s3controller = await api.get_service("public/s3-storage")
```

```javascript
const api = await connectToServer({"server_url": "https://hypha.aicell.io"});
const workspace = api.config.workspace;
const s3Controller = await api.getService("public/s3-storage");
```

### Recommended S3 Service Functions

**For optimal integration with Hypha's S3 service, we recommend using these functions:**

- **`put_file()`** - For single file uploads (< 100MB)
- **`get_file()`** - For file downloads  
- **`put_file_start_multipart()`** and **`put_file_complete_multipart()`** - For large file uploads (> 100MB)
- **`remove_file()`** - For deleting files or directories (replaces both `delete_file` and `delete_directory`)
- **`list_files()`** - For listing files and directories

**Deprecated Functions (still supported but not recommended):**
- ⚠️ `delete_file()` - Use `remove_file()` instead
- ⚠️ `delete_directory()` - Use `remove_file()` instead  
- ⚠️ `generate_presigned_url()` - Use `put_file()` and `get_file()` instead for better integration

### Using Recommended S3 Service Functions

```python
from hypha_rpc import connect_to_server
import httpx
import asyncio

# Connect to S3 service
api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
s3controller = await api.get_service("public/s3-storage")

# Upload a file using put_file (recommended)
upload_url = await s3controller.put_file("data/myfile.txt")
with open("local_file.txt", "rb") as f:
    async with httpx.AsyncClient() as client:
        response = await client.put(upload_url, content=f)
        print("Upload successful" if response.status_code == 200 else "Upload failed")

# Download a file using get_file (recommended)
download_url = await s3controller.get_file("data/myfile.txt")
async with httpx.AsyncClient() as client:
    response = await client.get(download_url)
    with open("downloaded_file.txt", "wb") as f:
        f.write(response.content)

# Remove a file or directory using remove_file (recommended)
result = await s3controller.remove_file("data/myfile.txt")  # Remove file
print(result["message"])

result = await s3controller.remove_file("data/")  # Remove entire directory
print(result["message"])

# List files
files = await s3controller.list_files("data/")
for file in files:
    print(f"File: {file['name']}, Size: {file.get('size', 'N/A')}")
```

```javascript
import axios from 'axios';

// Connect to S3 service
const api = await connectToServer({"server_url": "https://hypha.aicell.io"});
const s3Controller = await api.getService("public/s3-storage");

// Upload a file using putFile (recommended)
const uploadUrl = await s3Controller.putFile("data/myfile.txt");
const fileData = new FormData();
fileData.append('file', fileInput.files[0]);
const uploadResponse = await axios.put(uploadUrl, fileData);

// Download a file using getFile (recommended)
const downloadUrl = await s3Controller.getFile("data/myfile.txt");
const downloadResponse = await axios.get(downloadUrl, { responseType: 'blob' });

// Remove a file or directory using removeFile (recommended)
const result = await s3Controller.removeFile("data/myfile.txt");
console.log(result.message);

// List files
const files = await s3Controller.listFiles("data/");
files.forEach(file => console.log(`File: ${file.name}, Size: ${file.size || 'N/A'}`));
```

### 1. Using the Admin API (Minio-based S3 Servers)

For Minio-based S3 servers, Hypha can use the admin API to create user-specific credentials. Once you have the credentials, you can use any standard S3 client, such as [Cyberduck](https://cyberduck.io/), to access the S3 storage.

You can retrieve credentials using the following Hypha API calls:

```python
info = await s3controller.generate_credential()
```

The `info` object contains the required credentials and configuration to operate on the S3 bucket, such as `endpoint_url`, `aws_access_key_id`, `aws_secret_access_key`, `region_name`, and `bucket`.

<!-- tabs:start -->

#### ** aioboto3 **

```python
from hypha_rpc import connect_to_server
import aioboto3
import asyncio

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
s3controller = await api.get_service("public/s3-storage")
info = await s3controller.generate_credential()

async def operate_files_with_aioboto(info):
    async with aioboto3.Session().resource(
        "s3",
        endpoint_url=info["endpoint_url"],
        aws_access_key_id=info["access_key_id"],
        aws_secret_access_key=info["secret_access_key"],
        region_name=info["region_name"],
    ) as s3_client:
        bucket = await s3_client.Bucket(info["bucket"])

        # Attempt to list root folder (will fail due to permissions)
        async for s3_object in bucket.objects.all():
            print(s3_object)

        # Upload a file to the bucket
        obj = await s3_client.Object(info["bucket"], info["prefix"] + "hello.txt")
        with open("/tmp/hello.txt", "w", encoding="utf-8") as file:
            file.write("hello")
        await obj.upload_file("/tmp/hello.txt")

        # Generate a presigned URL and download the file
        url = await s3controller.generate_presigned_url("hello.txt")
        print(url)

# Example usage
# asyncio.run(operate_files_with_aioboto(info))
```

#### ** boto3 **

```python
from hypha_rpc import connect_to_server
import boto3

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
s3controller = await api.get_service("public/s3-storage")
info = await s3controller.generate_credential()

def operate_files_with_boto(info):
    s3_client = boto3.resource(
        "s3",
        endpoint_url=info["endpoint_url"],
        aws_access_key_id=info["access_key_id"],
        aws_secret_access_key=info["secret_access_key"],
        region_name=info["region_name"],
    )
    bucket = s3_client.Bucket(info["bucket"])

    # Upload a file to the bucket
    obj = s3_client.Object(info["bucket"], info["prefix"] + "hello.txt")
    with open("/tmp/hello.txt", "w", encoding="utf-8") as file:
        file.write("hello")
    obj.upload_file("/tmp/hello.txt")

    # List files in the bucket
    for obj in bucket.objects.filter(Prefix=info["prefix"]):
        print(obj.key)

    # Generate a presigned URL and download the file
    url = s3controller.generate_presigned_url("hello.txt")
    print(f"Presigned URL: {url}")

# Example usage
# operate_files_with_boto(info)
```

#### ** AWS SDK (JavaScript) **

```javascript
import { S3Client, PutObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";

const api = await connectToServer({ "server_url": "https://hypha.aicell.io" });
const s3Controller = await api.getService("public/s3-storage");
const info = await s3Controller.generateCredential();

const s3Client = new S3Client({
  region: info.region_name,
  endpoint: info.endpoint_url,
  credentials: {
    accessKeyId: info.access_key_id,
    secretAccessKey: info.secret_access_key,
  },
});

async function uploadFileToS3(file) {
  try {
    const uploadParams = {
      Bucket: info.bucket,
      Key: info.prefix + file.name,
      Body: file,
    };
    const data = await s3Client.send(new PutObjectCommand(uploadParams));
    console.log("File uploaded successfully:", data);
  } catch (err) {
    console.error("Error uploading file:", err);
  }
}

async function listFilesInS3() {
  try {
    const listParams = {
      Bucket: info.bucket,
      Prefix: info.prefix,
    };
    const data = await s3Client.send(new ListObjectsV2Command(listParams));
    data.Contents.forEach((obj) => console.log(obj.Key));
  } catch (err) {
    console.error("Error listing files:", err);
  }
}

// Example usage:
// const fileInput = document.getElementById("fileInput").files[0];
// uploadFileToS3(fileInput);
// listFilesInS3();
```

<!-- tabs:end -->

### 2. Using Presigned URLs

For any S3-compatible server, you can generate presigned URLs for secure file upload and download. This method allows direct interaction with the S3 server without routing file data through Hypha, reducing latency and bandwidth consumption.

```python
from hypha_rpc import connect_to_server

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
s3controller = await api.get_service("public/s3-storage")
upload_url = await s3controller.generate_presigned_url("myfolder/myfile.txt", client_method="put_object")
download_url = await s3controller.generate_presigned_url("myfolder/myfile.txt", client_method="get_object")
```

```javascript
const api = await connectToServer({"server_url": "https://hypha.aicell.io"});
const s3Controller = await api.getService("public/s3-storage");
const uploadUrl = await s3Controller.generatePresignedUrl("myfolder/myfile.txt", {client_method: "put_object"});
const downloadUrl = await s3Controller.generatePresignedUrl("myfolder/myfile.txt", {client_method: "get_object"});
```

<!-- tabs:start -->

#### ** Python (httpx) **

```python
from hypha_rpc import connect_to_server
import httpx
import asyncio

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
s3controller = await api.get_service("public/s3-storage")
upload_url = await s3controller.generate_presigned_url("myfolder/myfile.txt", client_method="put_object")

async def upload_file(presigned_url: str, file_path: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as file:
            response = await client.put(presigned_url, content=file)
            print(response.status_code, response.reason_phrase)

#

 Example usage:
# asyncio.run(upload_file(upload_url, "myfile.txt"))
```

#### ** Python (requests) **

```python
from hypha_rpc import connect_to_server
import requests

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
s3controller = await api.get_service("public/s3-storage")
upload_url = await s3controller.generate_presigned_url("myfolder/myfile.txt", client_method="put_object")

def upload_file(presigned_url, file_path):
    with open(file_path, "rb") as file:
        response = requests.put(presigned_url, data=file)
        print(response.status_code, response.reason)

# Example usage:
# upload_file(upload_url, "myfile.txt")
```

#### ** JavaScript (Axios) **

```javascript
const api = await connectToServer({"server_url": "https://hypha.aicell.io"});
const s3Controller = await api.getService("public/s3-storage");
const uploadUrl = await s3Controller.generatePresignedUrl("myfolder/myfile.txt", {client_method: "put_object"});

async function uploadFile(presignedUrl, file) {
  try {
    const response = await axios.put(presignedUrl, file, {
      headers: {
        "Content-Type": "application/octet-stream"
      }
    });
    console.log(response.status, response.statusText);
  } catch (error) {
    console.error(error.response.status, error.response.statusText);
  }
}

// Example usage:
// uploadFile(uploadUrl, file);
```

<!-- tabs:end -->

### 3. Using the Hypha Workspace `/files` Endpoint

You can also use Hypha's built-in HTTP proxy to interact with files stored in S3. Files can be uploaded, listed, or deleted via the `/files` endpoint in the Hypha workspace. You need to generate a token for these operations:

```python
from hypha_rpc import connect_to_server

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
workspace = api.config["workspace"]
token = await api.generate_token()
```

```javascript
const api = await connectToServer({"server_url": "https://hypha.aicell.io"});
const token = await api.generateToken();
```

<!-- tabs:start -->

#### ** Python (httpx) **

```python
from hypha_rpc import connect_to_server
import httpx
import asyncio

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
workspace = api.config["workspace"]
token = await api.generate_token()

async def upload_file_to_hypha(file_upload_url: str, file_path: str, token: str):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as file:
            response = await client.put(
                file_upload_url, headers={"Authorization": f"Bearer {token}"}, content=file
            )
            print(response.status_code, response.reason_phrase)

# Example usage:
# asyncio.run(upload_file_to_hypha(f"https://hypha.aicell.io/{workspace}/files/myfile.txt", "myfile.txt", token))
```

#### ** Python (requests) **

```python
from hypha_rpc import connect_to_server
import requests

api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
workspace = api.config["workspace"]
token = await api.generate_token()

def upload_file_to_hypha(upload_url, file_path, token):
    with open(file_path, "rb") as file:
        response = requests.put(upload_url, headers={"Authorization": f"Bearer {token}"}, data=file)
        print(response.status_code, response.reason)

# Example usage:
# upload_file_to_hypha(f"https://hypha.aicell.io/{workspace}/files/myfile.txt", "myfile.txt", token)
```

#### ** JavaScript (Axios) **

```javascript
const api = await connectToServer({"server_url": "https://hypha.aicell.io"});
const token = await api.generateToken();
const workspace = api.config.workspace;

async function uploadFileToHypha(file, fileUploadUrl, token) {
  
  try {
    const response = await axios.put(fileUploadUrl, file, {
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/octet-stream"
      }
    });
    console.log(response.status, response.statusText);
  } catch (error) {
    console.error(error.response.status, error.response.statusText);
  }
}

// Example usage:
// const file = document.getElementById(fileInputId).files[0];
// uploadFileToHypha(file, `https://hypha.aicell.io/${workspace}/files/${file.name}`, token);
```

<!-- tabs:end -->

## Uploading Large Files with Multipart Upload

For large files (> 100MB), Hypha supports multipart uploads that allow files to be uploaded in chunks. This provides better performance, reliability, and the ability to resume interrupted uploads.

### Using S3 Service Functions for Multipart Upload

```python
from hypha_rpc import connect_to_server
import asyncio
import os
import math
import httpx

# Connect to the S3 service
api = await connect_to_server({"server_url": "https://hypha.aicell.io"})
s3_controller = await api.get_service("public/s3-storage")
workspace = api.config["workspace"]

async def upload_large_file_multipart(file_path, s3_file_path):
    """Upload a large file using multipart upload."""
    
    # Calculate optimal chunk size (minimum 5MB, recommended 100MB)
    file_size = os.path.getsize(file_path)
    min_chunk_size = 5 * 1024 * 1024  # 5MB
    recommended_chunk_size = 100 * 1024 * 1024  # 100MB
    chunk_size = max(min_chunk_size, min(recommended_chunk_size, file_size // 10))
    part_count = math.ceil(file_size / chunk_size)
    
    print(f"File size: {file_size / (1024*1024):.1f}MB")
    print(f"Chunk size: {chunk_size / (1024*1024):.1f}MB")
    print(f"Number of parts: {part_count}")
    
    # Step 1: Start multipart upload
    multipart_info = await s3_controller.put_file_start_multipart(
        file_path=s3_file_path,
        part_count=part_count,
        expires_in=3600
    )
    
    upload_id = multipart_info["upload_id"]
    part_urls = multipart_info["parts"]
    
    # Step 2: Upload all parts in parallel
    async def upload_part(part_info):
        """Upload a single part."""
        part_number = part_info["part_number"]
        url = part_info["url"]
        
        # Calculate chunk boundaries
        start_pos = (part_number - 1) * chunk_size
        end_pos = min(start_pos + chunk_size, file_size)
        chunk_size_actual = end_pos - start_pos
        
        # Read chunk from file
        with open(file_path, "rb") as f:
            f.seek(start_pos)
            chunk_data = f.read(chunk_size_actual)
        
        # Upload chunk
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(url, data=chunk_data)
            if response.status_code != 200:
                raise Exception(f"Failed to upload part {part_number}")
                
            return {
                "part_number": part_number,
                "etag": response.headers["ETag"].strip('"').strip("'")
            }
    
    # Upload parts in parallel with controlled concurrency
    max_concurrent = min(10, part_count)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def upload_with_semaphore(part_info):
        async with semaphore:
            return await upload_part(part_info)
    
    print(f"Uploading {part_count} parts with max {max_concurrent} concurrent uploads...")
    uploaded_parts = await asyncio.gather(*[upload_with_semaphore(part) for part in part_urls])
    
    # Step 3: Complete multipart upload
    result = await s3_controller.put_file_complete_multipart(
        upload_id=upload_id,
        parts=uploaded_parts
    )
    
    print("Multipart upload completed successfully!")
    return result

# Example usage
# await upload_large_file_multipart("large_video.mp4", "uploads/large_video.mp4")
```

### Using HTTP Endpoints for Multipart Upload

You can also use the HTTP endpoints directly for more control:

```python
import asyncio
import httpx
import os
import math

SERVER_URL = "https://hypha.aicell.io"
workspace = "my-workspace"
token = "your-token"  # Get from api.generate_token()

async def upload_large_file_http(file_path, remote_path):
    """Upload large file using HTTP multipart endpoints."""
    
    file_size = os.path.getsize(file_path)
    chunk_size = 100 * 1024 * 1024  # 100MB chunks
    part_count = math.ceil(file_size / chunk_size)
    
    # Step 1: Create multipart upload
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/files/create-multipart-upload",
            params={
                "path": remote_path,
                "part_count": part_count,
                "expires_in": 3600
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to create multipart upload: {response.text}")
            
        upload_data = response.json()
        upload_id = upload_data["upload_id"]
        parts = upload_data["parts"]

    # Step 2: Upload parts in parallel
    async def upload_part(part_info):
        part_number = part_info["part_number"]
        url = part_info["url"]
        
        start_pos = (part_number - 1) * chunk_size
        end_pos = min(start_pos + chunk_size, file_size)
        
        with open(file_path, "rb") as f:
            f.seek(start_pos)
            chunk_data = f.read(end_pos - start_pos)
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(url, data=chunk_data)
            if response.status_code != 200:
                raise Exception(f"Failed to upload part {part_number}")
                
            return {
                "part_number": part_number,
                "etag": response.headers["ETag"].strip('"').strip("'")
            }
    
    uploaded_parts = await asyncio.gather(*[upload_part(part) for part in parts])

    # Step 3: Complete multipart upload
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/files/complete-multipart-upload",
            json={
                "upload_id": upload_id,
                "parts": uploaded_parts
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to complete multipart upload: {response.text}")
            
        return response.json()

# Example usage
# await upload_large_file_http("large_dataset.zip", "data/large_dataset.zip")
```

### Multipart Upload Best Practices

- **File Size Threshold**: Use multipart uploads for files larger than 100MB
- **Chunk Size**: 
  - Minimum: 5MB (S3 requirement, except for the last part)
  - Recommended: 100MB for optimal performance
  - Maximum parts: 10,000 per upload
- **Concurrency**: Limit concurrent uploads to avoid overwhelming the server (recommended: 10 concurrent parts)
- **Error Handling**: Implement retry logic for individual part uploads
- **Cleanup**: If an upload fails, the multipart upload session will automatically expire after the specified time
- **Progress Tracking**: Track upload progress by monitoring completed parts

### When to Use Each Method

1. **Single File Upload** (`/files/{path}`): 
   - Files < 100MB
   - Simple upload scenarios
   - Direct streaming from client to S3

2. **Multipart Upload** (`/files/create-multipart-upload` + `/files/complete-multipart-upload`):
   - Files > 100MB
   - Need upload resumption capability
   - Want parallel chunk uploads for faster performance
   - Network conditions are unreliable

## Summary

To enable S3 storage in Hypha, you need to provide existing credentials during server startup. Once configured, Hypha supports three ways to manipulate files in S3:
1. **Admin API (Minio-specific):** Directly create S3 credentials for users, enabling access through S3 clients.
2. **Presigned URLs:** Generate URLs for direct file upload and download, reducing overhead.
3. **Hypha HTTP Proxy:** Use the workspace `/files` endpoint to manage files through Hypha's proxy. Tokens for these operations can be generated via `await server.generate_token()`.

Choose the method that best suits your setup and network configuration for efficient file management in Hypha.
