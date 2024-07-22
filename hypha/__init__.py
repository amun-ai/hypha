"""Main module for hypha."""
import json
from pathlib import Path

# read version information from file
VERSION_INFO = json.loads(
    (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
)
__version__ = VERSION_INFO["version"]

__all__ = ["__version__"]
