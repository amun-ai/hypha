"""Redis queue service for Hypha."""

import json
from fakeredis import aioredis
from hypha.core.store import RedisStore


def create_queue_service(store: RedisStore):
    """Create a queue service for Hypha."""
    redis: aioredis.FakeRedis = store.get_redis()
    event_bus = store.get_event_bus()
    
    async def on_workspace_removed(workspace):
        # delete all the keys that start with workspace["name"] + ":q:"
        keys_pattern = workspace["name"] + ":q:*"
        cursor = '0'

        while cursor != 0:
            cursor, keys = await redis.scan(cursor=cursor, match=keys_pattern)
            if keys:
                await redis.delete(*keys)
        
    event_bus.on("workspace_removed", on_workspace_removed)

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
        "type": "queue",
        "config": {
            "visibility": "public",
            "require_context": True,
        },
        "push": push_task,
        "pop": pop_task,
        "get_length": get_queue_length,
        "peek": peek_queue,
    }
