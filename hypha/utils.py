"""Provide utilities that should not be aware of hypha."""
import copy
import os
import string
import secrets
import posixpath
from typing import List, Optional
from datetime import datetime

_os_alt_seps: List[str] = list(
    sep for sep in [os.path.sep, os.path.altsep] if sep is not None and sep != "/"
)

PLUGIN_CONFIG_FIELDS = [
    "name",
    "source_hash",
    "version",
    "format_version",
    "type",
    "tags",
    "icon",
    "requirements",
    "env",
    "defaults",
    "flags",
    "inputs",
    "outputs",
    "dependencies",
    "readiness_probe",
    "liveness_probe",
]


class EventBus:
    """An event bus class."""

    def __init__(self):
        """Initialize the event bus."""
        self._callbacks = {}

    def on(self, event_name, func):
        """Register an event callback."""
        self._callbacks[event_name] = self._callbacks.get(event_name, []) + [func]
        return func

    def once(self, event_name, func):
        """Register an event callback that only run once."""
        self._callbacks[event_name] = self._callbacks.get(event_name, []) + [func]
        # mark once callback
        self._callbacks[event_name].once = True
        return func

    def emit(self, event_name, *data):
        """Trigger an event."""
        for func in self._callbacks.get(event_name, []):
            func(*data)
            if hasattr(func, "once"):
                self.off(event_name, func)

    def off(self, event_name, func=None):
        """Remove an event callback."""
        if not func:
            del self._callbacks[event_name]
        else:
            self._callbacks.get(event_name, []).remove(func)


def generate_password(length=20):
    """Generate a password."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for i in range(length))


def safe_join(directory: str, *pathnames: str) -> Optional[str]:
    """Safely join zero or more untrusted path components to a base directory.

    This avoids escaping the base directory.
    :param directory: The trusted base directory.
    :param pathnames: The untrusted path components relative to the
        base directory.
    :return: A safe path, otherwise ``None``.

    This function is copied from:
    https://github.com/pallets/werkzeug/blob/fb7ddd89ae3072e4f4002701a643eb247a402b64/src/werkzeug/security.py#L222
    """
    parts = [directory]

    for filename in pathnames:
        if filename != "":
            filename = posixpath.normpath(filename)

        if (
            any(sep in filename for sep in _os_alt_seps)
            or os.path.isabs(filename)
            or filename == ".."
            or filename.startswith("../")
        ):
            raise Exception(
                f"Illegal file path: `{filename}`, "
                "you can only operate within the work directory."
            )

        parts.append(filename)

    return posixpath.join(*parts)


class dotdict(dict):  # pylint: disable=invalid-name
    """Access dictionary attributes with dot.notation."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        """Make a deep copy."""
        return dotdict(copy.deepcopy(dict(self), memo=memo))


def parse_s3_list_response(response, delimeter):
    """Parse the s3 list object response."""
    if response.get("KeyCount") == 0:
        return []
    items = [
        {
            "type": "file",
            "name": item["Key"].split("/")[-1] if delimeter == "/" else item["Key"],
            "size": item["Size"],
            "last_modified": datetime.timestamp(item["LastModified"]),
        }
        for item in response.get("Contents", [])
    ]
    # only include when delimeter is /
    if delimeter == "/":
        items += [
            {"type": "directory", "name": item["Prefix"].rstrip("/").split("/")[-1]}
            for item in response.get("CommonPrefixes", [])
        ]
    return items


def list_objects_sync(s3_client, bucket, prefix=None, delimeter="/"):
    """List a objects sync."""
    prefix = prefix or ""
    response = s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, Delimiter=delimeter
    )

    items = parse_s3_list_response(response, delimeter)
    while response["IsTruncated"]:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter=delimeter,
            ContinuationToken=response["NextContinuationToken"],
        )
        items += parse_s3_list_response(response, delimeter)
    return items


def remove_objects_sync(s3_client, bucket, prefix, delimeter=""):
    """Remove all objects in a folder."""
    assert prefix != "" and prefix.endswith("/")
    response = s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, Delimiter=delimeter
    )
    items = response.get("Contents", [])
    if len(items) > 0:
        delete_response = s3_client.delete_objects(
            Bucket=bucket,
            Delete={
                "Objects": [
                    {
                        "Key": item["Key"],
                        # 'VersionId': 'string'
                    }
                    for item in items
                ],
                "Quiet": True,
            },
        )
        assert (
            "ResponseMetadata" in delete_response
            and delete_response["ResponseMetadata"]["HTTPStatusCode"] == 200
        )
    while response["IsTruncated"]:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter=delimeter,
            ContinuationToken=response["NextContinuationToken"],
        )
        items = response.get("Contents", [])
        if len(items) > 0:
            delete_response = s3_client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [
                        {
                            "Key": item["Key"],
                            # 'VersionId': 'string'
                        }
                        for item in items
                    ],
                    "Quiet": True,
                },
            )
            assert (
                "ResponseMetadata" in delete_response
                and delete_response["ResponseMetadata"]["HTTPStatusCode"] == 200
            )


async def list_objects_async(s3_client, bucket, prefix=None, delimeter="/"):
    """List objects async."""
    prefix = prefix or ""
    response = await s3_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, Delimiter=delimeter
    )
    items = parse_s3_list_response(response, delimeter)
    while response["IsTruncated"]:
        response = await s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter=delimeter,
            ContinuationToken=response["NextContinuationToken"],
        )
        items += parse_s3_list_response(response, delimeter)
    return items
