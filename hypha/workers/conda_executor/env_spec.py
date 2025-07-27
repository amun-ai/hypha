"""
Environment specification handling for conda environments.

This module provides tools for reading and handling conda environment specifications
from YAML files or inline specifications.
"""

import os
import re
import yaml
import tarfile
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, TextIO
from pathlib import Path


@dataclass
class EnvSpec:
    """Conda environment specification."""

    name: Optional[str] = None
    channels: List[str] = field(default_factory=lambda: ["conda-forge"])
    dependencies: List[str] = field(default_factory=list)
    prefix: Optional[str] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = ["conda-forge"]
        if self.dependencies is None:
            self.dependencies = []

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EnvSpec":
        """Create an EnvSpec from a conda environment.yaml file."""
        with open(path, "rb") as f:
            try:
                spec = yaml.safe_load(f)
                if not isinstance(spec, dict):
                    raise ValueError("Invalid environment specification format")
            except yaml.YAMLError as e:
                raise ValueError(f"Could not parse environment file: {e}")

        return cls(
            name=spec.get("name"),
            channels=spec.get("channels", ["conda-forge"]),
            dependencies=spec.get("dependencies", []),
            prefix=spec.get("prefix"),
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "EnvSpec":
        """Create an EnvSpec from a dictionary."""
        return cls(
            name=data.get("name"),
            channels=data.get("channels", []),
            dependencies=data.get("dependencies", []),
            prefix=data.get("prefix"),
        )

    def to_dict(self) -> Dict:
        """Convert the EnvSpec to a dictionary."""
        spec = {"name": self.name or "temp_env", "channels": [], "dependencies": []}

        # Ensure channels is a list of strings
        if isinstance(self.channels, list):
            spec["channels"] = [str(channel) for channel in self.channels]
        elif isinstance(self.channels, str):
            spec["channels"] = [str(self.channels)]
        else:
            spec["channels"] = ["conda-forge"]

        # Ensure dependencies is a list
        if isinstance(self.dependencies, list):
            spec["dependencies"].extend(self.dependencies)
        elif isinstance(self.dependencies, str):
            spec["dependencies"].append(self.dependencies)
        elif isinstance(self.dependencies, dict):
            # Handle pip dependencies separately
            for key, value in self.dependencies.items():
                if key == "pip":
                    if isinstance(value, list):
                        spec["dependencies"].append({"pip": value})
                    else:
                        spec["dependencies"].append({"pip": [value]})
                else:
                    spec["dependencies"].append(f"{key}={value}")

        return spec


def extract_spec_from_code(code: str) -> Optional[EnvSpec]:
    """
    Extract environment specification from code comments.

    Looks for comments in the format:
    # conda env
    # channels:
    #   - conda-forge
    # dependencies:
    #   - python=3.9
    #   - numpy
    """
    spec_lines = []
    in_spec = False

    for line in code.split("\n"):
        line = line.strip()
        if line == "# conda env":
            in_spec = True
            continue
        if in_spec:
            if not line.startswith("#"):
                break
            spec_lines.append(line.lstrip("#").strip())

    if not spec_lines:
        return None

    try:
        spec_yaml = yaml.safe_load("\n".join(spec_lines))
        if not isinstance(spec_yaml, dict):
            return None
        return EnvSpec.from_dict(spec_yaml)
    except yaml.YAMLError:
        return None


def read_env_spec(source: Union[str, Path, TextIO, Dict, "EnvSpec"]) -> EnvSpec:
    """
    Read environment specification from various sources.

    Args:
        source: Can be:
            - Path to a YAML file
            - String containing YAML content
            - File-like object containing YAML content
            - Dictionary with specification
            - String containing Python code with embedded spec
            - Path to a conda-pack file (.tar.gz)
            - EnvSpec object

    Returns:
        EnvSpec object

    Raises:
        ValueError: If the source cannot be parsed as a valid environment specification
        FileNotFoundError: If the source is a file path that doesn't exist
        TypeError: If the source type is not supported
    """
    if isinstance(source, EnvSpec):
        return source

    if isinstance(source, (str, Path)):
        source_path = Path(source)

        # If it's a conda-pack file, return default spec with empty channels
        if str(source_path).endswith((".tar.gz", ".tgz")):
            if not source_path.exists():
                raise FileNotFoundError(f"Conda-pack file not found: {source_path}")
            try:
                # Verify it's a valid tar.gz file
                with tarfile.open(source_path, "r:gz") as _:
                    return EnvSpec(channels=[])
            except Exception as e:
                raise ValueError(f"Invalid conda-pack file: {e}")

        # If it's a string with newlines, try to extract spec from code first
        if isinstance(source, str) and "\n" in source:
            spec = extract_spec_from_code(source)
            if spec is not None:
                return spec

            # Try parsing as YAML string
            try:
                spec_dict = yaml.safe_load(source)
                if isinstance(spec_dict, dict):
                    return EnvSpec.from_dict(spec_dict)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Could not parse environment specification from string: {e}"
                )

            raise ValueError("Could not parse environment specification from string")

        # If it's a string without newlines, try to parse it as YAML first
        if isinstance(source, str):
            try:
                spec_dict = yaml.safe_load(source)
                if isinstance(spec_dict, dict):
                    return EnvSpec.from_dict(spec_dict)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Could not parse environment specification from string: {e}"
                )

        # Handle file paths
        if not source_path.exists():
            raise FileNotFoundError(
                f"Environment specification file not found: {source_path}"
            )

        # Try to read as YAML file
        try:
            with open(source_path, "r") as f:
                spec_dict = yaml.safe_load(f)
                if isinstance(spec_dict, dict):
                    return EnvSpec.from_dict(spec_dict)
                raise ValueError("Invalid environment specification format")
        except yaml.YAMLError as e:
            raise ValueError(f"Could not parse environment file: {e}")
        except Exception as e:
            raise ValueError(f"Could not read environment file: {e}")

    elif isinstance(source, TextIO):
        try:
            spec_dict = yaml.safe_load(source)
            if isinstance(spec_dict, dict):
                return EnvSpec.from_dict(spec_dict)
            raise ValueError("Invalid environment specification format")
        except yaml.YAMLError as e:
            raise ValueError(
                f"Could not parse environment specification from file: {e}"
            )

    elif isinstance(source, dict):
        return EnvSpec.from_dict(source)

    raise TypeError(f"Unsupported source type: {type(source)}")
