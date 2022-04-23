"""Set up the hypha package."""
import json
from pathlib import Path

from setuptools import find_packages, setup

DESCRIPTION = (
    "A serverless application framework for large-scale"
    " data management and AI model serving."
)

REQUIREMENTS = [
    "aiobotocore>=2.1.0",
    "aiofiles",
    "fastapi>=0.70.0",
    "imjoy-rpc==0.5.7",
    "msgpack>=1.0.2",
    "numpy",
    "pydantic[email]>=1.8.2",
    "typing-extensions>=3.7.4.3",  # required by pydantic
    "jinja2>=3",
    "lxml",
    "python-dotenv>=0.19.0",
    "python-engineio==4.1.0",
    "python-jose==3.3.0",
    "pyyaml",
    "redislite>=6.0.674960",
    "aioredis>=2.0.1",
    "shortuuid>=1.0.1",
    "uvicorn>=0.13.4",
    "httpx>=0.21.1",
    "pyotritonclient>=0.2.0a1",
]

ROOT_DIR = Path(__file__).parent.resolve()
README_FILE = ROOT_DIR / "README.md"
LONG_DESCRIPTION = README_FILE.read_text(encoding="utf-8")
VERSION_FILE = ROOT_DIR / "hypha" / "VERSION"
VERSION = json.loads(VERSION_FILE.read_text(encoding="utf-8"))["version"]


setup(
    name="hypha",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="http://github.com/imjoy-team/hypha",
    author="ImJoy Team",
    author_email="imjoy.team@gmail.com",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require={
        "server-apps": [
            "requests>=2.26.0",
            "playwright>=1.18.1",
            "base58>=2.1.0",
            "pymultihash>=0.8.2",
        ],
    },
    zip_safe=False,
    entry_points={"console_scripts": ["imjoy = imjoy.__main__:main"]},
)
