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

## Summary

To enable S3 storage in Hypha, you need to provide existing credentials during server startup. Once configured, Hypha supports three ways to manipulate files in S3:
1. **Admin API (Minio-specific):** Directly create S3 credentials for users, enabling access through S3 clients.
2. **Presigned URLs:** Generate URLs for direct file upload and download, reducing overhead.
3. **Hypha HTTP Proxy:** Use the workspace `/files` endpoint to manage files through Hypha's proxy. Tokens for these operations can be generated via `await server.generate_token()`.

Choose the method that best suits your setup and network configuration for efficient file management in Hypha.
