import logging
import sys
from sqlalchemy import (
    Column,
    String,
    Integer,
    JSON,
    DateTime,
    select,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import datetime
from hypha.core import UserInfo, UserPermission  # Replace with actual imports

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("event_logging")
logger.setLevel(logging.INFO)

Base = declarative_base()


# SQLAlchemy model for storing events
class EventLog(Base):
    __tablename__ = "event_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    workspace = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    message = Column(String, nullable=False)
    timestamp = Column(
        DateTime, default=datetime.datetime.now(datetime.timezone.utc), index=True
    )
    data = Column(JSON, nullable=True)  # Store any additional event metadata

    def to_dict(self):
        """Convert the SQLAlchemy model instance to a dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "workspace": self.workspace,
            "user_id": self.user_id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),  # Convert datetime to ISO string
            "data": self.data,
        }


class EventLoggingService:
    """Service for logging and querying events."""

    def __init__(self, store, database_uri=None):
        """Initialize the EventLoggingService with an SQLAlchemy async engine."""
        if database_uri is None:
            database_uri = (
                "sqlite+aiosqlite:///:memory:"  # In-memory SQLite for testing
            )
            logger.warning(
                "Using in-memory SQLite database for event logging, all data will be lost on restart!"
            )
        store.set_logging_service(self)
        self.store = store
        self.engine = create_async_engine(database_uri, echo=False)
        self.SessionLocal = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init_db(self):
        """Initialize the database and create tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully.")
        self.store.register_public_service(self.get_logging_service())

    async def _get_session(self):
        """Return an async session for the database."""
        return self.SessionLocal()

    def _check_permissions(self, context, permission_type=UserPermission.read_write):
        """Check user permissions for a given workspace and raise an error if not permitted."""
        user_info = UserInfo.model_validate(context["user"])
        workspace = context["ws"]

        if not user_info.check_permission(workspace, permission_type):
            raise PermissionError(
                f"User {user_info.id} does not have {permission_type} permission to the workspace '{workspace}'."
            )
        return workspace, user_info.id

    async def log_event(
        self, event_type: str, message: str, data: dict = None, context: dict = None
    ):
        """Log a new event, checking permissions."""
        workspace, user_id = self._check_permissions(context, UserPermission.read_write)

        session = await self._get_session()
        try:
            async with session.begin():
                event_log = EventLog(
                    event_type=event_type,
                    workspace=workspace,
                    user_id=user_id,
                    message=message,
                    data=data,
                )
                session.add(event_log)
                await session.commit()
                logger.info(f"Logged event: {event_type} by {user_id} in {workspace}")
        finally:
            await session.close()

    async def get_event_stats(
        self, event_type=None, start_time=None, end_time=None, context=None
    ):
        """Get statistics for specific event types, filtered by workspace, user, and time range."""
        workspace, user_id = self._check_permissions(context, UserPermission.read)

        session = await self._get_session()
        try:
            async with session.begin():
                query = select(
                    EventLog.event_type, func.count(EventLog.id).label("count")
                ).filter(EventLog.workspace == workspace, EventLog.user_id == user_id)

                # Apply optional filters
                if event_type:
                    query = query.filter(EventLog.event_type == event_type)
                if start_time:
                    query = query.filter(EventLog.timestamp >= start_time)
                if end_time:
                    query = query.filter(EventLog.timestamp <= end_time)

                query = query.group_by(EventLog.event_type)
                result = await session.execute(query)
                stats = result.fetchall()
                # Convert rows to dictionaries
                stats_dicts = [dict(row._mapping) for row in stats]
                return stats_dicts
        finally:
            await session.close()

    async def search_events(
        self, event_type=None, start_time=None, end_time=None, context: dict = None
    ):
        """Search for events with various filters, enforcing workspace and permission checks."""
        workspace, user_id = self._check_permissions(context, UserPermission.read)

        session = await self._get_session()
        try:
            async with session.begin():
                query = select(EventLog).filter(
                    EventLog.workspace == workspace, EventLog.user_id == user_id
                )

                # Apply optional filters
                if event_type:
                    query = query.filter(EventLog.event_type == event_type)
                if start_time:
                    query = query.filter(EventLog.timestamp >= start_time)
                if end_time:
                    query = query.filter(EventLog.timestamp <= end_time)

                result = await session.execute(query)
                # Use scalars() to get model instances, not rows
                events = result.scalars().all()

                # Convert each EventLog instance to a dictionary using to_dict()
                event_dicts = [event.to_dict() for event in events]

                return event_dicts
        finally:
            await session.close()

    async def get_histogram(self, event_type=None, time_interval="hour", context=None):
        """Generate a histogram for events based on time intervals (hour, day, month)."""
        workspace, _ = self._check_permissions(context, UserPermission.read)

        session = await self._get_session()
        try:
            async with session.begin():
                time_trunc_func = {
                    "hour": func.strftime("%Y-%m-%d %H", EventLog.timestamp),
                    "day": func.strftime("%Y-%m-%d", EventLog.timestamp),
                    "month": func.strftime("%Y-%m", EventLog.timestamp),
                }.get(
                    time_interval, func.strftime("%Y-%m-%d %H", EventLog.timestamp)
                )  # Default to hourly

                # Pass the expressions individually, without the square brackets
                query = select(
                    time_trunc_func.label("time_bucket"),
                    func.count(EventLog.id).label("count"),
                ).filter(EventLog.workspace == workspace)

                if event_type:
                    query = query.filter(EventLog.event_type == event_type)

                query = query.group_by("time_bucket").order_by("time_bucket")
                result = await session.execute(query)
                histogram = result.fetchall()
                histogram = [dict(row._mapping) for row in histogram]
                return histogram
        finally:
            await session.close()

    def get_logging_service(self):
        """Return the event logging service definition."""
        return {
            "id": "event-log",
            "config": {"visibility": "public", "require_context": True},
            "name": "Event Logging Service",
            "description": "Service for logging and querying system events.",
            "log_event": self.log_event,
            "get_stats": self.get_event_stats,
            "search": self.search_events,
            "histogram": self.get_histogram,
        }
