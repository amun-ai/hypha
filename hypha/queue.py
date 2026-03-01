"""Redis queue service for Hypha."""

import json
import os
import sys
from fakeredis import aioredis
from hypha.core.store import RedisStore
from hypha_rpc.utils.schema import schema_function
from pydantic import Field
from typing import Any, Dict, List, Optional
import logging

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("queue")
logger.setLevel(LOGLEVEL)


import re

_QUEUE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,127}$')

def _validate_queue_name(queue_name: str):
    """Validate queue name format."""
    if not queue_name or not _QUEUE_NAME_PATTERN.match(queue_name):
        raise ValueError(
            f"Invalid queue name: '{queue_name}'. "
            "Queue names must be 1-128 characters, start with alphanumeric, "
            "and contain only alphanumeric, hyphens, underscores, and dots."
        )

def create_queue_service(store: RedisStore):
    """Create a distributed task queue service for Hypha.

    This service provides a Redis-backed queue system for asynchronous task processing,
    enabling reliable task distribution across workers in a workspace.
    """
    redis: aioredis.FakeRedis = store.get_redis()

    @schema_function
    async def push_task(
        queue_name: str = Field(
            ...,
            description="Name of the queue to push the task to. Queue names should be descriptive (e.g., 'image-processing', 'data-export'). The queue is automatically scoped to the current workspace."
        ),
        task: Dict[str, Any] = Field(
            ...,
            description="Task data to push to the queue. Should be a JSON-serializable dictionary containing task details like type, parameters, and metadata."
        ),
        context: Optional[dict] = Field(
            None,
            description="Context containing workspace information. Usually provided automatically by the system."
        )
    ) -> None:
        """Push a task to a queue for asynchronous processing.
        
        This method adds a task to the specified queue using LPUSH (left push),
        making it available for workers to process. Tasks are processed in FIFO order.
        
        Examples:
            # Push an image processing task
            await push_task("image-processing", {
                "type": "resize",
                "image_id": "img-123",
                "dimensions": {"width": 800, "height": 600}
            })
            
            # Push a batch job
            await push_task("batch-jobs", {
                "job_id": "job-456",
                "operation": "export",
                "format": "csv"
            })
        """
        _validate_queue_name(queue_name)
        workspace = context["ws"]
        await redis.lpush(workspace + ":q:" + queue_name, json.dumps(task))

    @schema_function
    async def pop_task(
        queue_name: str = Field(
            ...,
            description="Name of the queue to pop a task from. The queue is automatically scoped to the current workspace."
        ),
        timeout: float = Field(
            30,
            description="Maximum time in seconds to wait for a task. 0 means non-blocking (return immediately if empty). Default: 30 seconds.",
            ge=0,
            le=300
        ),
        context: Optional[dict] = Field(
            None,
            description="Context containing workspace information. Usually provided automatically by the system."
        )
    ) -> Optional[Dict[str, Any]]:
        """Pop and retrieve a task from a queue.

        This method removes and returns a task from the specified queue using BRPOP
        (blocking right pop). Tasks are processed in FIFO order (first in, first out).

        If the queue is empty:
        - With timeout > 0: waits up to `timeout` seconds for a task, returns None if no task arrives.
        - With timeout = 0: returns None immediately (non-blocking).

        Returns:
            The task data as a dictionary, or None if no task is available within the timeout.

        Examples:
            # Pop a task, wait up to 30 seconds (default)
            task = await pop_task("image-processing")
            if task:
                print(f"Processing: {task['type']}")
            else:
                print("No tasks available")

            # Non-blocking pop
            task = await pop_task("image-processing", timeout=0)
        """
        _validate_queue_name(queue_name)
        workspace = context["ws"]
        task = await redis.brpop(workspace + ":q:" + queue_name, timeout=int(timeout))
        if task is None:
            return None
        return json.loads(task[1])

    @schema_function
    async def get_queue_length(
        queue_name: str = Field(
            ...,
            description="Name of the queue to check. The queue is automatically scoped to the current workspace."
        ),
        context: Optional[dict] = Field(
            None,
            description="Context containing workspace information. Usually provided automatically by the system."
        )
    ) -> int:
        """Get the current number of tasks in a queue.
        
        This method returns the length of the specified queue without modifying it.
        Useful for monitoring queue depth and system load.
        
        Returns:
            The number of tasks currently in the queue.
            
        Examples:
            # Check queue depth before adding more tasks
            length = await get_queue_length("batch-jobs")
            if length < 100:
                await push_task("batch-jobs", new_task)
        """
        _validate_queue_name(queue_name)
        workspace = context["ws"]
        length = await redis.llen(workspace + ":q:" + queue_name)
        return length

    @schema_function
    async def peek_queue(
        queue_name: str = Field(
            ...,
            description="Name of the queue to peek into. The queue is automatically scoped to the current workspace."
        ),
        n: int = Field(
            1,
            description="Number of tasks to peek at from the queue (default: 1). Tasks are not removed from the queue.",
            ge=1,
            le=100
        ),
        context: Optional[dict] = Field(
            None,
            description="Context containing workspace information. Usually provided automatically by the system."
        )
    ) -> List[Dict[str, Any]]:
        """Peek at tasks in a queue without removing them.
        
        This method returns the first N tasks from the queue without removing them.
        Useful for inspecting queue contents or implementing monitoring dashboards.
        
        Returns:
            List of task dictionaries (up to N tasks).
            
        Examples:
            # Inspect the next 5 tasks in the queue
            upcoming_tasks = await peek_queue("image-processing", n=5)
            for task in upcoming_tasks:
                print(f"Upcoming: {task['type']}")
                
        Limitations:
            - Maximum 100 tasks can be peeked at once
            - Does not modify the queue state
        """
        _validate_queue_name(queue_name)
        workspace = context["ws"]
        tasks = await redis.lrange(workspace + ":q:" + queue_name, 0, n - 1)
        # Reverse to match pop (BRPOP) order: LPUSH adds to left, BRPOP reads from right,
        # so LRANGE(0,n) returns newest-first but pop returns oldest-first.
        return [json.loads(task) for task in reversed(tasks)]

    return {
        "id": "queue",
        "name": "Task Queue Service",
        "description": "Distributed task queue system for asynchronous job processing in Hypha workspaces. Provides reliable FIFO task distribution with Redis backend.",
        "config": {
            "visibility": "public",
            "require_context": True,
        },
        "push": push_task,
        "pop": pop_task,
        "get_length": get_queue_length,
        "peek": peek_queue,
    }
