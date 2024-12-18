"""This file contains modules from the taskiq-redis package
We need to patch it so we can use fakeredis when redis is not available.
The library was created by Pavel Kirilin, released under the MIT license.
"""

from fakeredis import aioredis
from typing import (
    Dict,
    Optional,
    TypeVar,
    Union,
)

from taskiq import AsyncResultBackend
from taskiq.abc.result_backend import TaskiqResult
from taskiq.abc.serializer import TaskiqSerializer
from taskiq.compat import model_dump, model_validate
from taskiq.depends.progress_tracker import TaskProgress
from taskiq.serializers import PickleSerializer

from hypha.taskiq_utils.exceptions import (
    DuplicateExpireTimeSelectedError,
    ExpireTimeMustBeMoreThanZeroError,
    ResultIsMissingError,
)


_ReturnType = TypeVar("_ReturnType")

PROGRESS_KEY_SUFFIX = "__progress"


class RedisAsyncResultBackend(AsyncResultBackend[_ReturnType]):
    """Async result based on redis."""

    def __init__(
        self,
        redis: aioredis.FakeRedis,
        keep_results: bool = True,
        result_ex_time: Optional[int] = None,
        result_px_time: Optional[int] = None,
        serializer: Optional[TaskiqSerializer] = None,
    ) -> None:
        """
        Constructs a new result backend.

        :param redis_url: url to redis.
        :param keep_results: flag to not remove results from Redis after reading.
        :param result_ex_time: expire time in seconds for result.
        :param result_px_time: expire time in milliseconds for result.
        :param max_connection_pool_size: maximum number of connections in pool.
        :param connection_kwargs: additional arguments for redis BlockingConnectionPool.

        :raises DuplicateExpireTimeSelectedError: if result_ex_time
            and result_px_time are selected.
        :raises ExpireTimeMustBeMoreThanZeroError: if result_ex_time
            and result_px_time are equal zero.
        """
        self.redis = redis
        self.serializer = serializer or PickleSerializer()
        self.keep_results = keep_results
        self.result_ex_time = result_ex_time
        self.result_px_time = result_px_time

        unavailable_conditions = any(
            (
                self.result_ex_time is not None and self.result_ex_time <= 0,
                self.result_px_time is not None and self.result_px_time <= 0,
            ),
        )
        if unavailable_conditions:
            raise ExpireTimeMustBeMoreThanZeroError(
                "You must select one expire time param and it must be more than zero.",
            )

        if self.result_ex_time and self.result_px_time:
            raise DuplicateExpireTimeSelectedError(
                "Choose either result_ex_time or result_px_time.",
            )

    async def shutdown(self) -> None:
        """Closes redis connection."""
        await super().shutdown()

    async def set_result(
        self,
        task_id: str,
        result: TaskiqResult[_ReturnType],
    ) -> None:
        """
        Sets task result in redis.

        Dumps TaskiqResult instance into the bytes and writes
        it to redis.

        :param task_id: ID of the task.
        :param result: TaskiqResult instance.
        """
        redis_set_params: Dict[str, Union[str, int, bytes]] = {
            "name": task_id,
            "value": self.serializer.dumpb(model_dump(result)),
        }
        if self.result_ex_time:
            redis_set_params["ex"] = self.result_ex_time
        elif self.result_px_time:
            redis_set_params["px"] = self.result_px_time

        await self.redis.set(**redis_set_params)  # type: ignore

    async def is_result_ready(self, task_id: str) -> bool:
        """
        Returns whether the result is ready.

        :param task_id: ID of the task.

        :returns: True if the result is ready else False.
        """
        return bool(await self.redis.exists(task_id))

    async def get_result(
        self,
        task_id: str,
        with_logs: bool = False,
    ) -> TaskiqResult[_ReturnType]:
        """
        Gets result from the task.

        :param task_id: task's id.
        :param with_logs: if True it will download task's logs.
        :raises ResultIsMissingError: if there is no result when trying to get it.
        :return: task's return value.
        """
        if self.keep_results:
            result_value = await self.redis.get(
                name=task_id,
            )
        else:
            result_value = await self.redis.getdel(
                name=task_id,
            )

        if result_value is None:
            raise ResultIsMissingError

        taskiq_result = model_validate(
            TaskiqResult[_ReturnType],
            self.serializer.loadb(result_value),
        )

        if not with_logs:
            taskiq_result.log = None

        return taskiq_result

    async def set_progress(
        self,
        task_id: str,
        progress: TaskProgress[_ReturnType],
    ) -> None:
        """
        Sets task progress in redis.

        Dumps TaskProgress instance into the bytes and writes
        it to redis with a standard suffix on the task_id as the key

        :param task_id: ID of the task.
        :param result: task's TaskProgress instance.
        """
        redis_set_params: Dict[str, Union[str, int, bytes]] = {
            "name": task_id + PROGRESS_KEY_SUFFIX,
            "value": self.serializer.dumpb(model_dump(progress)),
        }
        if self.result_ex_time:
            redis_set_params["ex"] = self.result_ex_time
        elif self.result_px_time:
            redis_set_params["px"] = self.result_px_time

        await self.redis.set(**redis_set_params)  # type: ignore

    async def get_progress(
        self,
        task_id: str,
    ) -> Union[TaskProgress[_ReturnType], None]:
        """
        Gets progress results from the task.

        :param task_id: task's id.
        :return: task's TaskProgress instance.
        """
        result_value = await self.redis.get(
            name=task_id + PROGRESS_KEY_SUFFIX,
        )

        if result_value is None:
            return None

        return model_validate(
            TaskProgress[_ReturnType],
            self.serializer.loadb(result_value),
        )
