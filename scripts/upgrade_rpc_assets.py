#!/usr/bin/env python3
"""Script to fetch hypha-rpc static assets from CDN.

This script downloads the hypha-rpc-websocket.js and hypha-rpc-websocket.mjs
files from the jsDelivr CDN and saves them to the static_files folder.

Usage:
    python scripts/upgrade_rpc_assets.py          # Fetch/update static files
    python scripts/upgrade_rpc_assets.py --check  # Check version match (for CI)

The script will automatically use the hypha_rpc_version from hypha/__init__.py.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

import httpx


def get_hypha_rpc_version() -> str:
    """Extract hypha_rpc_version from hypha/__init__.py without importing.

    This avoids importing hypha which would require hypha_rpc to be installed.
    """
    init_file = Path(__file__).parent.parent / "hypha" / "__init__.py"
    content = init_file.read_text()
    match = re.search(r'hypha_rpc_version\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise RuntimeError(
            f"Could not find hypha_rpc_version in {init_file}"
        )
    return match.group(1)


hypha_rpc_version = get_hypha_rpc_version()

# Static files directory relative to this script
STATIC_FILES_DIR = Path(__file__).parent.parent / "hypha" / "static_files"

# Files to download
RPC_FILES = [
    ("hypha-rpc-websocket.mjs", f"https://cdn.jsdelivr.net/npm/hypha-rpc@{hypha_rpc_version}/dist/hypha-rpc-websocket.mjs"),
    ("hypha-rpc-websocket.js", f"https://cdn.jsdelivr.net/npm/hypha-rpc@{hypha_rpc_version}/dist/hypha-rpc-websocket.js"),
]


def check_version() -> bool:
    """Check if static RPC files version matches the expected version.

    Returns True if versions match, False otherwise.
    Exits with code 1 if there's a mismatch (for CI usage).
    """
    version_file = STATIC_FILES_DIR / "hypha-rpc-version.txt"

    if not version_file.exists():
        print(f"ERROR: Static hypha-rpc files not found at {STATIC_FILES_DIR}")
        print(f"Run 'python scripts/upgrade_rpc_assets.py' to fetch them.")
        return False

    static_version = version_file.read_text().strip()
    if static_version != hypha_rpc_version:
        print(f"ERROR: Static hypha-rpc files version mismatch!")
        print(f"  Found:    {static_version}")
        print(f"  Expected: {hypha_rpc_version}")
        print(f"Run 'python scripts/upgrade_rpc_assets.py' to update.")
        return False

    # Also check that the JS files exist
    for filename, _ in RPC_FILES:
        filepath = STATIC_FILES_DIR / filename
        if not filepath.exists():
            print(f"ERROR: Missing file: {filepath}")
            print(f"Run 'python scripts/upgrade_rpc_assets.py' to fetch them.")
            return False

    print(f"OK: Static hypha-rpc files are up to date (version {hypha_rpc_version})")
    return True


async def fetch_and_save(client: httpx.AsyncClient, filename: str, url: str) -> None:
    """Fetch a file from URL and save it to the static files directory."""
    print(f"Fetching {filename} from {url}...")
    response = await client.get(url)
    response.raise_for_status()

    filepath = STATIC_FILES_DIR / filename
    filepath.write_bytes(response.content)
    print(f"  Saved to {filepath} ({len(response.content)} bytes)")


async def upgrade():
    """Fetch all RPC assets from CDN."""
    print(f"Upgrading hypha-rpc assets to version {hypha_rpc_version}")
    print(f"Static files directory: {STATIC_FILES_DIR}")

    # Create static files directory if it doesn't exist
    STATIC_FILES_DIR.mkdir(parents=True, exist_ok=True)

    # Write version file for reference
    version_file = STATIC_FILES_DIR / "hypha-rpc-version.txt"
    version_file.write_text(f"{hypha_rpc_version}\n")
    print(f"Written version file: {version_file}")

    # Fetch all files
    async with httpx.AsyncClient(timeout=30) as client:
        for filename, url in RPC_FILES:
            await fetch_and_save(client, filename, url)

    print(f"\nSuccessfully upgraded hypha-rpc assets to version {hypha_rpc_version}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch hypha-rpc static assets from CDN"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if static files are up to date (exits with code 1 if not)",
    )
    args = parser.parse_args()

    if args.check:
        if not check_version():
            sys.exit(1)
    else:
        asyncio.run(upgrade())


if __name__ == "__main__":
    main()
