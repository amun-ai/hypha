"""
Shared memory communication between processes.

This module provides tools for passing data between processes using shared memory,
with support for various data types including numpy arrays.
"""

import os
import mmap
import json
import struct
import uuid
import numpy as np
from typing import Any, Optional, Dict, Union
from pathlib import Path
import base64


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.tobytes()).decode("utf-8")
            return {
                "_type": "ndarray",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "data": data_b64,
            }
        return super().default(obj)


def numpy_decoder(obj):
    """Decode numpy arrays from JSON."""
    if isinstance(obj, dict) and obj.get("_type") == "ndarray":
        data = base64.b64decode(obj["data"])
        return np.frombuffer(data, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


class SharedMemoryChannel:
    """
    Handles shared memory communication between processes.

    Supports passing various data types including numpy arrays through shared memory.
    Uses a simple protocol:
    - 4 bytes: size of metadata
    - N bytes: metadata (JSON)
    - M bytes: actual data

    The metadata includes information about the data type and shape for arrays.
    """

    def __init__(self, channel_id: Optional[str] = None, size: int = 100 * 1024 * 1024):
        """
        Initialize a shared memory channel.

        Args:
            channel_id: Optional unique identifier for the channel
            size: Size of the shared memory segment in bytes (default: 100MB)
        """
        self.size = size
        self.channel_id = channel_id or str(uuid.uuid4())
        self.shm_path = f"/tmp/shm_{self.channel_id}"

        # Create a temporary file for shared memory
        self.fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
        os.truncate(self.fd, size)
        # Memory map the file
        self.memory = mmap.mmap(self.fd, size)

    def write_object(self, obj: Any) -> None:
        """
        Write a Python object to shared memory.

        Handles various data types including:
        - Basic Python types (via JSON)
        - Numpy arrays
        - Nested structures containing the above

        Args:
            obj: Object to write

        Raises:
            ValueError: If the data is too large for the shared memory channel
        """
        # Serialize with numpy support
        data = json.dumps(obj, cls=NumpyEncoder).encode("utf-8")
        size = len(data)

        if size + 4 > self.size:  # 4 bytes for size header
            raise ValueError(
                f"Data too large for shared memory channel "
                f"(size: {size}, max: {self.size-4})"
            )

        # Write to shared memory
        self.memory.seek(0)
        self.memory.write(struct.pack("I", size) + data)

    def read_object(self) -> Any:
        """
        Read a Python object from shared memory.

        Returns:
            The deserialized object
        """
        self.memory.seek(0)
        size = struct.unpack("I", self.memory.read(4))[0]
        data = self.memory.read(size)
        return json.loads(data.decode("utf-8"), object_hook=numpy_decoder)

    def close(self) -> None:
        """Clean up shared memory resources."""
        self.memory.close()
        os.close(self.fd)
        try:
            os.unlink(self.shm_path)
        except:
            pass

    @property
    def id(self) -> str:
        """Get the channel ID."""
        return self.channel_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
