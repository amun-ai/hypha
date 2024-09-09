"""Main module for hypha."""
import json
from pathlib import Path

# read version information from file
VERSION_INFO = json.loads(
    (Path(__file__).parent / "VERSION").read_text(encoding="utf-8").strip()
)
__version__ = VERSION_INFO["version"]
parts = __version__.split(".")
main_version = f"{parts[0]}.{parts[1]}.{parts[2]}"


__all__ = ["__version__", "main_version"]
