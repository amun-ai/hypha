"""Main module for hypha."""
import json
from pathlib import Path

from imjoy_rpc import *  # noqa F401, F403
from imjoy_rpc import __all__ as imjoy_rpc_all

# read version information from file
VERSION_INFO = json.loads(
    (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
)
__version__ = VERSION_INFO["version"]

__all__ = imjoy_rpc_all
