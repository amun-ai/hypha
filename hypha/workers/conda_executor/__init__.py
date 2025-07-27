"""
Conda Environment Executor

A package for executing Python code in isolated Conda environments.
"""

from .executor import CondaEnvExecutor, ExecutionResult, TimingInfo
from .env_spec import EnvSpec, read_env_spec
from .shared_memory import SharedMemoryChannel

__version__ = "0.1.0"
__all__ = [
    "CondaEnvExecutor",
    "ExecutionResult",
    "TimingInfo",
    "EnvSpec",
    "read_env_spec",
    "SharedMemoryChannel",
    "JobQueue",
    "JobStatus",
    "JobInfo",
    "JobResult",
]
