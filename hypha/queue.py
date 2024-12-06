"""Redis queue service for Hypha."""

import json
import sys
from fakeredis import aioredis
from hypha.core.store import RedisStore
import logging

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("queue")
logger.setLevel(logging.INFO)


def create_queue_service(store: RedisStore):
    """Create a queue service for Hypha."""
    redis: aioredis.FakeRedis = store.get_redis()

    async def push_task(queue_name, task: dict, context: dict = None):
        workspace = context["ws"]
        await redis.lpush(workspace + ":q:" + queue_name, json.dumps(task))

    async def pop_task(queue_name, context: dict = None):
        workspace = context["ws"]
        task = await redis.brpop(workspace + ":q:" + queue_name)
        return json.loads(task[1])

    async def get_queue_length(queue_name, context: dict = None):
        workspace = context["ws"]
        length = await redis.llen(workspace + ":q:" + queue_name)
        return length

    async def peek_queue(queue_name, n=1, context: dict = None):
        workspace = context["ws"]
        tasks = await redis.lrange(workspace + ":q:" + queue_name, 0, n - 1)
        return [json.loads(task) for task in tasks]

    return {
        "id": "queue",
        "name": "Queue service",
        "config": {
            "visibility": "public",
            "require_context": True,
        },
        "push": push_task,
        "pop": pop_task,
        "get_length": get_queue_length,
        "peek": peek_queue,
    }
