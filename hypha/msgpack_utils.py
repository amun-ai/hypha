"""Provide utils for msgpack connection."""
import gzip
import math
import uuid
import sys
import logging

import msgpack

CHUNK_SIZE = 1024 * 1000
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("RPC")


async def send_as_msgpack(msg, send, accept_encoding):
    """Send the message by using msgpack encoding."""
    encoded = {
        "type": "msgpack_data",
        "msg_type": msg.pop("type"),
    }
    if msg.get("to"):
        encoded["to"] = msg.pop("to")

    packed = msgpack.packb(msg, use_bin_type=True)
    total_size = len(packed)
    if total_size > CHUNK_SIZE and "gzip" in accept_encoding:
        compressed = gzip.compress(packed)
        # Only send the compressed version
        # if the compression ratio is > 80%;
        if len(compressed) <= total_size * 0.8:
            packed = compressed
            encoded["compression"] = "gzip"

    total_size = len(packed)
    if total_size <= CHUNK_SIZE:
        encoded["data"] = packed
        await send(encoded)
    else:
        # Try to use the peer_id as key so one peer can only have one chunk store
        object_id = msg.get("to", str(uuid.uuid4()))
        chunk_num = int(math.ceil(float(total_size) / CHUNK_SIZE))
        # send chunk by chunk
        for idx in range(chunk_num):
            start_byte = idx * CHUNK_SIZE
            chunk = {
                "type": "msgpack_chunk",
                "object_id": object_id,
                "data": packed[start_byte : start_byte + CHUNK_SIZE],
                "index": idx,
                "total": chunk_num,
            }
            logger.info(
                "Sending chunk %d/%d (%d bytes)",
                idx + 1,
                chunk_num,
                total_size,
            )
            await send(chunk)

        # reference the chunked object
        encoded["chunked_object"] = object_id
        await send(encoded)


def decode_msgpack(data, chunk_store):
    """Try to decode the data as msgpack."""
    dtype = data.get("type")
    if dtype == "msgpack_chunk":
        id_ = data["object_id"]
        # the chunk object does not exist or it's a starting chunk
        if id_ not in chunk_store or data["index"] == 0:
            chunk_store[id_] = []
        assert data["index"] == len(chunk_store[id_])
        chunk_store[id_].append(data["data"])
        return

    if dtype == "msgpack_data":
        if data.get("chunked_object"):
            object_id = data["chunked_object"]
            chunks = chunk_store[object_id]
            del chunk_store[object_id]
            data["data"] = b"".join(chunks)
        if data.get("compression"):
            if data["compression"] == "gzip":
                data["data"] = gzip.decompress(data["data"])
            else:
                raise Exception(f"Unsupported compression: {data['compression']}")
        decoded = msgpack.unpackb(data["data"], use_list=False, raw=False)
        if data.get("to"):
            decoded["to"] = data.get("to")
        decoded["type"] = data["msg_type"]
        data = decoded
    elif data.get("to") in chunk_store:
        # Clear chunk store for the peer if exists
        del chunk_store[data.get("to")]

    return data
