"""Redis queue service for Hypha."""

import json
from fakeredis import aioredis


def create_queue_service(redis: aioredis.FakeRedis, workspace: str):
    """Create a queue service for Hypha."""

    async def push_task(queue_name, task: dict):
        await redis.lpush(workspace + ":q:" + queue_name, json.dumps(task))

    async def pop_task(queue_name):
        task = await redis.brpop(workspace + ":q:" + queue_name)
        return json.loads(task[1])

    async def get_queue_length(queue_name):
        length = await redis.llen(workspace + ":q:" + queue_name)
        return length

    async def peek_queue(queue_name, n=1):
        tasks = await redis.lrange(workspace + ":q:" + queue_name, 0, n - 1)
        return [json.loads(task) for task in tasks]

    return {
        "id": "queue",
        "name": "Queue service",
        "type": "queue",
        "config": {
            "visibility": "protected",
        },
        "push": push_task,
        "pop": pop_task,
        "get_length": get_queue_length,
        "peek": peek_queue,
    }
