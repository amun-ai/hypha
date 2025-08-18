"""Main module for hypha."""

import json
from pathlib import Path
import hypha_rpc

# read version information from file
VERSION_INFO = json.loads(
    (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
)
__version__ = VERSION_INFO["version"]
parts = __version__.split(".")

hypha_rpc_version = "0.20.78"


__all__ = ["__version__", "hypha_rpc_version"]
