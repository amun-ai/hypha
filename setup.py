"""Set up the hypha package."""

import json
from pathlib import Path

from setuptools import find_packages, setup

DESCRIPTION = (
    "A serverless application framework for large-scale"
    " data management and AI model serving."
)

REQUIREMENTS = [
    "websockets>=14.0",
    "aiofiles",
    "fastapi>=0.70.0,<=0.115.2",
    "hypha-rpc>=0.20.90",
    "msgpack>=1.0.2",
    "numpy",
    "pydantic[email]>=2.6.1",
    "typing-extensions>=3.7.4.3",  # required by pydantic
    "jinja2>=3",
    "lxml",
    "python-dotenv>=0.19.0",
    "python-jose>=3.3.0",
    "python-multipart>=0.0.18",
    "pyyaml>=6.0.1",
    "fakeredis>=2.14.1",
    "shortuuid>=1.0.1",
    "uvicorn>=0.29.0",
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
    "greenlet>=3.1.1",
    "aiosqlite>=0.20.0",
    "prometheus-client>=0.21.1",
    "uuid-utils>=0.9.0",
    "sqlmodel>=0.0.22",
    "alembic>=1.14.0",
    "hrid==0.3.0",
    "stream-zip>=0.0.83",
    "starlette-compress>=1.6.0",
    "prompt-toolkit>=3.0.50",
    "ptpython>=3.0.29",
    "ptyprocess==0.7.0",
    "psutil>=5.9.0",
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
            "dulwich>=0.21.7",  # For Git storage support
        ],
        "server-apps": [
            "redis==5.2.0",
            "aiobotocore>=2.1.0",
            "aiortc>=1.9.0",
            "requests>=2.26.0",
            "playwright>=1.51.0",
            "base58>=2.1.0",
            "pymultihash>=0.8.2",
            "fastuuid>=0.12.0",
        ],
        "db": [
            "psycopg2-binary>=2.9.10",
            "asyncpg>=0.30.0",
            "fastembed>=0.4.2",
            "zarr>=3.1.2; python_version >= '3.11'",
            "numcodecs>=0.16.2; python_version >= '3.11'",
        ],
        "k8s": [
            "kubernetes>=24.2.0",
        ],
        "llm-proxy": [
            "openai>=1.0.0",
            "tiktoken>=0.5.0",
            "click>=8.0.0",
            "tokenizers>=0.15.0",
            "uvicorn>=0.29.0",
            "uvloop>=0.21.0; sys_platform != 'win32'",
            "gunicorn>=23.0.0",
            "fastapi>=0.115.5",
            "backoff",
            "pyyaml>=6.0.1",
            "rq",
            "orjson>=3.9.7",
            "apscheduler>=3.10.4",
            "fastapi-sso>=0.16.0",
            "PyJWT>=2.8.0",
            "python-multipart>=0.0.18",
            "cryptography",
            "prisma==0.11.0",
            "azure-identity>=1.15.0",
            "azure-keyvault-secrets>=4.8.0",
            "azure-storage-blob>=12.25.1",
            "google-cloud-kms>=2.21.3",
            "google-cloud-iam>=2.19.1",
            "resend>=0.8.0",
            "pynacl>=1.5.0",
            "websockets>=13.1.0",
            "boto3==1.36.0",
            "redisvl>=0.4.1; python_version >= '3.9' and python_version < '3.14'",
            "mcp>=1.10.0; python_version >= '3.10'",
            "litellm-proxy-extras==0.2.19",
            "rich==13.7.1",
            "litellm-enterprise==0.1.20",
            "diskcache>=5.6.1",
            "polars>=1.31.0; python_version >= '3.10'",
            "semantic-router; python_version >= '3.9'",
            "mlflow>3.1.4; python_version >= '3.10'",
            # Optional provider-specific packages (can be installed as needed)
            "google-genai>=0.1.0",  # For Google Gemini
            # "anthropic>=0.30.0",  # For Anthropic Claude
            # "cohere>=4.0.0",  # For Cohere models
            # "replicate>=0.10.0",  # For Replicate
            # "together>=0.2.0",  # For Together AI
            # "huggingface-hub>=0.15.0",  # For HuggingFace
        ],
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "hypha = hypha.__main__:main",
            "hypha-cli = hypha.__main__:run_interactive_cli",
        ]
    },
)
