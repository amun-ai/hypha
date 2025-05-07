# ZIP File Streaming in Hypha

Hypha provides powerful support for efficiently working with ZIP archives, particularly for large files. This document describes the ZIP streaming capabilities in Hypha and how to use them.

## Overview

Hypha's ZIP streaming capability allows users to:

1. Access files within ZIP archives without downloading the entire archive
2. Stream content from large ZIP files efficiently
3. List directories inside ZIP archives
4. Extract specific files from ZIP archives with minimal memory usage

This is particularly useful for large data collections where downloading an entire archive would be impractical.

## How It Works

The ZIP streaming implementation uses an adaptive approach to efficiently handle ZIP files of various sizes:

- **Small Archives (<50MB)**: For small ZIP files, the entire archive is downloaded for simplicity and reliability
- **Large Archives**: Only the central directory (ZIP file index) is downloaded to read the structure
- **Very Large Archives**: Implements streaming decompression to avoid loading entire files into memory

The implementation supports both standard ZIP and ZIP64 formats, making it suitable for ZIP files of any size (including those over 4GB).

### True Streaming Decompression

For large compressed files inside ZIP archives, Hypha implements true streaming decompression:

1. The system fetches compressed data in small chunks using byte-range requests
2. Each chunk is decompressed on-the-fly
3. Decompressed data is streamed to the client as it becomes available
4. This prevents having to load the entire file into memory at any point

This approach is especially valuable for very large files inside ZIP archives, as it:
- Minimizes memory usage on the server
- Starts delivering content to the client immediately
- Allows clients to process data before the entire file is downloaded
- Makes it feasible to work with files that are larger than available system memory

## API Endpoints

### Accessing ZIP File Content

#### List Directory Contents in a ZIP File

```
GET /{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}
GET /{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}?path={directory_path}
```

This endpoint lists the contents of a specific directory within a ZIP file. If no path is provided, it lists the root directory of the ZIP.

**Response**: JSON array of file and directory entries:

```json
[
    {"type": "file", "name": "example.txt", "size": 123, "last_modified": 1732363845.0},
    {"type": "directory", "name": "nested"}
]
```

#### Extract a File from a ZIP Archive

```
GET /{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}?path={file_path}
```

This endpoint extracts and streams a specific file from within a ZIP archive.

**Response**: The raw file content with appropriate Content-Type headers.

### Alternative URL Format

You can also use the "tilde" format for separating the ZIP path from the internal path:

```
GET /{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}/~/{path}
```

## Example Use Cases

1. **Large Datasets**: Access specific files from large data archives (GB/TB scale) without downloading entire datasets
2. **Document Collections**: Browse and extract specific documents from zipped document collections
3. **Software Distribution**: Extract specific components from large software packages
4. **Media Storage**: Stream large media files from within ZIP archives without buffering the entire file

## Performance Considerations

- ZIP file central directories are cached for improved performance
- The system automatically adapts to different ZIP file sizes
- Chunk sizes for decompression are optimized based on file size and compression ratio
- Streaming decompression minimizes memory usage for large files
- Content-type is automatically detected based on file extension for better client compatibility

## Implementation Details

Hypha's ZIP streaming is implemented in the `hypha.utils.zip_utils` module, with the following key functions:

- `fetch_zip_tail`: Retrieves just the central directory of a ZIP file
- `get_zip_file_content`: Streams content from a specific file or lists directory contents
- `read_file_from_zip`: Extracts a single file from a ZIP archive
- `stream_file_from_zip`: Streams a file from a ZIP archive in chunks
- `stream_file_chunks_from_zip`: Implements on-demand byte-range fetching and decompression 