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
if len(parts) > 3:
    # if the version is not a release version, we need to use the previous version
    if not parts[3].startswith("post"):
        main_version = f"{parts[0]}.{parts[1]}.{int(parts[2])-1}"


__all__ = ["__version__", "main_version"]
