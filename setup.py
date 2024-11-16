"""Set up the hypha package."""
import json
from pathlib import Path

from setuptools import find_packages, setup

DESCRIPTION = (
    "A serverless application framework for large-scale"
    " data management and AI model serving."
)

REQUIREMENTS = [
    "aiofiles",
    "fastapi>=0.70.0,<=0.106.0",
    "hypha-rpc>=0.20.39",
    "msgpack>=1.0.2",
    "numpy",
    "pydantic[email]>=2.6.1",
    "typing-extensions>=3.7.4.3",  # required by pydantic
    "jinja2>=3",
    "lxml",
    "python-dotenv>=0.19.0",
    "python-jose>=3.3.0",
    "python-multipart>=0.0.6",
    "pyyaml",
    "fakeredis>=2.14.1",
    "shortuuid>=1.0.1",
    "uvicorn>=0.23.2",
    "httpx>=0.21.1",
    "pyotritonclient>=0.2.4",
    # add email-validator for pyodide
    # see https://github.com/pyodide/pyodide/issues/3969
    "email-validator>=2.0.0;platform_system=='Emscripten'",
    "pyodide-http;platform_system=='Emscripten'",
    "ssl;platform_system=='Emscripten'",
    "friendlywords>=1.1.3",
    "aiocache>=0.12.2",
    "jsonschema>=3.2.0",
    "sqlalchemy>=2.0.35",
    "aiosqlite>=0.20.0",
    "prometheus-client>=0.21.0",
    "uuid-utils>=0.9.0",
    "sqlmodel>=0.0.22",
    "alembic>=1.14.0",
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
    url="http://github.com/amun-ai/hypha",
    author="Amun AI AB",
    author_email="info@amun.ai",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require={
        "s3": [
            "aiobotocore>=2.1.0",
        ],
        "server-apps": [
            "redis>=4.5.5",
            "aiobotocore>=2.1.0",
            "aiortc>=1.9.0",
            "requests>=2.26.0",
            "playwright>=1.18.1",
            "base58>=2.1.0",
            "pymultihash>=0.8.2",
        ],
        "postgres": ["psycopg2-binary>=2.9.10", "asyncpg>=0.30.0"],
    },
    zip_safe=False,
    entry_points={"console_scripts": ["hypha = hypha.__main__:main"]},
)
