"""This file contains modules from the taskiq-redis package
We need to patch it so we can use fakeredis when redis is not available.
The library was created by Pavel Kirilin, released under the MIT license.
"""

from fakeredis import aioredis
from logging import getLogger
from typing import AsyncGenerator, Callable, Optional, TypeVar

from taskiq.abc.broker import AsyncBroker
from taskiq.abc.result_backend import AsyncResultBackend
from taskiq.message import BrokerMessage

_T = TypeVar("_T")

logger = getLogger("taskiq.redis_broker")


class BaseRedisBroker(AsyncBroker):
    """Base broker that works with Redis."""

    def __init__(
        self,
        redis: aioredis.FakeRedis,
        task_id_generator: Optional[Callable[[], str]] = None,
        result_backend: Optional[AsyncResultBackend[_T]] = None,
        queue_name: str = "taskiq",
    ) -> None:
        """
        Constructs a new broker.

        :param url: url to redis.
        :param task_id_generator: custom task_id generator.
        :param result_backend: custom result backend.
        :param queue_name: name for a list in redis.
        :param max_connection_pool_size: maximum number of connections in pool.
            Each worker opens its own connection. Therefore this value has to be
            at least number of workers + 1.
        :param connection_kwargs: additional arguments for redis BlockingConnectionPool.
        """
        super().__init__(
            result_backend=result_backend,
            task_id_generator=task_id_generator,
        )

        self.redis = redis
        self.queue_name = queue_name

    async def shutdown(self) -> None:
        """Closes redis connection pool."""
        await super().shutdown()


class PubSubBroker(BaseRedisBroker):
    """Broker that works with Redis and broadcasts tasks to all workers."""

    async def kick(self, message: BrokerMessage) -> None:
        """
        Publish message over PUBSUB channel.

        :param message: message to send.
        """
        queue_name = message.labels.get("queue_name") or self.queue_name
        await self.redis.publish(queue_name, message.message)

    async def listen(self) -> AsyncGenerator[bytes, None]:
        """
        Listen redis queue for new messages.

        This function listens to the pubsub channel
        and yields all messages with proper types.

        :yields: broker messages.
        """

        redis_pubsub_channel = self.redis.pubsub()
        await redis_pubsub_channel.subscribe(self.queue_name)
        async for message in redis_pubsub_channel.listen():
            if not message:
                continue
            if message["type"] != "message":
                logger.debug("Received non-message from redis: %s", message)
                continue
            yield message["data"]


class ListQueueBroker(BaseRedisBroker):
    """Broker that works with Redis and distributes tasks between workers."""

    async def kick(self, message: BrokerMessage) -> None:
        """
        Put a message in a list.

        This method appends a message to the list of all messages.

        :param message: message to append.
        """
        queue_name = message.labels.get("queue_name") or self.queue_name
        await self.redis.lpush(queue_name, message.message)

    async def listen(self) -> AsyncGenerator[bytes, None]:
        """
        Listen redis queue for new messages.

        This function listens to the queue
        and yields new messages if they have BrokerMessage type.

        :yields: broker messages.
        """
        redis_brpop_data_position = 1
        while True:
            try:
                yield (await self.redis.brpop(self.queue_name))[
                    redis_brpop_data_position
                ]
            except ConnectionError as exc:
                logger.warning("Redis connection error: %s", exc)
                continue
