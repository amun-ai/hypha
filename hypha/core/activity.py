import asyncio
import time
from typing import Callable, Dict, Optional
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("activity-tracker")
logger.setLevel(logging.INFO)


class ActivityTracker:
    def __init__(self, check_interval: int = 10):
        """
        :param check_interval: Time in seconds to check for active/inactive entities
        """
        self._registrations: Dict[str, Dict[str, dict]] = {}
        self._check_interval = check_interval
        self._stop = False

    def register(
        self,
        entity_id: str,
        inactive_period: int,
        on_active: Optional[Callable] = None,
        on_inactive: Optional[Callable] = None,
        entity_type: Optional[str] = "default",
    ) -> str:
        """Register an entity with activity tracking.

        :param entity_id: Unique ID of the entity (client, workspace, etc.)
        :param inactive_period: Time in seconds after which the entity is considered inactive
        :param on_active: Function to call when the entity becomes active
        :param on_inactive: Function to call when the entity becomes inactive
        :param entity_type: Type of entity being tracked (e.g., client, workspace)
        :return: A unique registration ID
        """
        full_id = f"{entity_type}:{entity_id}"
        if full_id not in self._registrations:
            self._registrations[full_id] = {}

        reg_id = f"{full_id}-{len(self._registrations[full_id])}"
        if inactive_period < self._check_interval:
            raise ValueError(
                f"Inactive period must be greater than the check interval ({self._check_interval} seconds)"
            )
        self._registrations[full_id][reg_id] = {
            "inactive_period": inactive_period,
            "on_active": on_active,
            "on_inactive": on_inactive,
            "last_activity": None,
            "is_active": None,
        }
        return reg_id

    def unregister(
        self, entity_id: str, reg_id: str, entity_type: Optional[str] = "default"
    ):
        """Unregister a specific registration for an entity.

        :param entity_id: The entity's ID
        :param reg_id: The unique registration ID to remove
        :param entity_type: Type of entity being tracked
        """
        full_id = f"{entity_type}:{entity_id}"
        if full_id in self._registrations:
            if reg_id in self._registrations[full_id]:
                del self._registrations[full_id][reg_id]
            if not self._registrations[
                full_id
            ]:  # Remove entity if no registrations left
                del self._registrations[full_id]

    def remove_entity(self, entity_id: str, entity_type: Optional[str] = "default"):
        """Remove all registrations for an entity.

        :param entity_id: The entity's ID to remove
        :param entity_type: Type of entity being tracked
        """
        full_id = f"{entity_type}:{entity_id}"
        if full_id in self._registrations:
            del self._registrations[full_id]

    async def reset_timer(self, entity_id: str, entity_type: Optional[str] = "default"):
        """Reset the activity timer for all registrations of an entity."""
        full_id = f"{entity_type}:{entity_id}"
        # logger.info(f"Resetting timer for {full_id}")
        if full_id in self._registrations:
            now = time.time()
            for reg in list(self._registrations[full_id].values()):
                reg["last_activity"] = now
                if not reg["is_active"]:
                    reg["is_active"] = True
                    if reg["on_active"]:
                        try:
                            await reg["on_active"]()
                            logger.info(f"{full_id} becomes active")
                        except Exception as e:
                            logger.error(f"Error calling on_active for {full_id}: {e}")

    async def monitor_entities(self):
        """Periodically check the activity of registered entities."""
        while not self._stop:
            now = time.time()
            try:
                # logger.info(f"Checking activity at {now}")
                for full_id in list(self._registrations.keys()):
                    for reg in list(self._registrations[full_id].values()):
                        if (
                            reg["last_activity"]
                            and (now - reg["last_activity"] > reg["inactive_period"])
                            and reg["is_active"]
                        ):
                            # Trigger on_inactive event
                            reg["is_active"] = False
                            if reg["on_inactive"]:
                                try:
                                    await reg["on_inactive"]()
                                    logger.info(f"{full_id} becomes inactive")
                                except Exception as e:
                                    logger.error(
                                        f"Error calling on_inactive for {full_id}: {e}"
                                    )
                        await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Error monitoring activity: {e}")
            await asyncio.sleep(self._check_interval)

    def stop(self):
        """Stop the monitoring loop."""
        self._stop = True
