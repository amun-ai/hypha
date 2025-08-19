import asyncio
import os
import time
from typing import Callable, Dict, Optional, List
import logging
import sys

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("activity-tracker")
logger.setLevel(LOGLEVEL)


class ActivityTracker:
    def __init__(self, check_interval: int = 10):
        """
        :param check_interval: Time in seconds to check for active/inactive entities
        """
        self._registrations: Dict[str, Dict[str, dict]] = {}
        self._check_interval = check_interval
        self._stop = False

        # Callbacks for entity removal
        self._entity_removed_callbacks: List[Callable[[str, Optional[str]], None]] = []

    def register_entity_removed_callback(
        self, callback: Callable[[str, Optional[str]], None]
    ):
        """
        Register a callback to be called when an entity is removed.

        :param callback: A function that takes `entity_id` and `entity_type` as arguments.
        """
        self._entity_removed_callbacks.append(callback)

    async def _notify_entity_removed(
        self, entity_id: str, entity_type: Optional[str] = "default"
    ):
        for callback in self._entity_removed_callbacks:
            try:
                await callback(entity_id, entity_type)
            except Exception as e:
                logger.error(f"Error in entity removed callback for {entity_id}: {e}")

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
        if inactive_period < self._check_interval:
            raise ValueError(
                f"Inactive period must be greater than the check interval ({self._check_interval} seconds)"
            )

        full_id = f"{entity_type}:{entity_id}"
        if full_id not in self._registrations:
            self._registrations[full_id] = {}

        reg_id = f"{full_id}-{len(self._registrations[full_id])}"

        # Initialize as active with current timestamp
        now = time.time()
        self._registrations[full_id][reg_id] = {
            "inactive_period": inactive_period,
            "on_active": on_active,
            "on_inactive": on_inactive,
            "last_activity": now,  # Initialize with current time
            "is_active": True,  # Start as active
        }
        return reg_id

    def is_registered(self, entity_id: str, entity_type: Optional[str] = "default") -> bool:
        """Check if an entity is registered.
        
        :param entity_id: The entity's ID
        :param entity_type: Type of entity being tracked
        :return: True if the entity has any registrations
        """
        full_id = f"{entity_type}:{entity_id}"
        return full_id in self._registrations
    
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

    async def remove_entity(
        self, entity_id: str, entity_type: Optional[str] = "default"
    ):
        """Remove all registrations for an entity.

        :param entity_id: The entity's ID to remove
        :param entity_type: Type of entity being tracked
        """
        full_id = f"{entity_type}:{entity_id}"
        if full_id in self._registrations:
            # Call on_inactive for all active registrations before removing
            for reg in self._registrations[full_id].values():
                if reg["is_active"] and reg["on_inactive"]:
                    try:
                        await reg["on_inactive"]()
                    except Exception as e:
                        logger.error(
                            f"Error calling on_inactive during removal for {full_id}: {e}"
                        )
            del self._registrations[full_id]
            # Only notify if the entity was actually registered
            await self._notify_entity_removed(entity_id, entity_type)

    async def reset_timer(self, entity_id: str, entity_type: Optional[str] = "default"):
        """Reset the activity timer for all registrations of an entity."""
        full_id = f"{entity_type}:{entity_id}"
        if full_id in self._registrations:
            now = time.time()
            for reg in self._registrations[full_id].values():
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
            try:
                now = time.time()
                for full_id in list(self._registrations.keys()):
                    for reg in list(self._registrations[full_id].values()):
                        if reg["is_active"]:  # Only check active entities
                            time_since_last_activity = now - reg["last_activity"]
                            if time_since_last_activity > reg["inactive_period"]:
                                reg["is_active"] = False
                                if reg["on_inactive"]:
                                    try:
                                        await reg["on_inactive"]()
                                        logger.info(
                                            f"{full_id} becomes inactive after {time_since_last_activity:.1f}s"
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"Error calling on_inactive for {full_id}: {e}"
                                        )
                        await asyncio.sleep(0)  # Allow other tasks to run
            except Exception as e:
                logger.error(f"Error monitoring activity: {e}")
            await asyncio.sleep(self._check_interval)

    def stop(self):
        """Stop the monitoring loop."""
        self._stop = True
