"""phase3_interceptor_storage_extension

Revision ID: c8f6df9a1c1e
Revises: 52f4bfc1d44b
Create Date: 2026-02-18 07:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c8f6df9a1c1e"
down_revision: Union[str, None] = "52f4bfc1d44b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "event_interceptors",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("workspace", sa.String(length=255), nullable=False),
        sa.Column("app_id", sa.String(length=255), nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("event_types", sa.JSON(), nullable=True),
        sa.Column("categories", sa.JSON(), nullable=True),
        sa.Column("condition_field", sa.String(length=255), nullable=False),
        sa.Column("condition_op", sa.String(length=32), nullable=False),
        sa.Column("condition_value", sa.JSON(), nullable=True),
        sa.Column("action_type", sa.String(length=32), nullable=False),
        sa.Column("action_reason", sa.String(length=32), nullable=True),
        sa.Column("action_recovery", sa.JSON(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_event_interceptors_workspace",
        "event_interceptors",
        ["workspace"],
        unique=False,
    )
    op.create_index(
        "ix_event_interceptors_app_id",
        "event_interceptors",
        ["app_id"],
        unique=False,
    )
    op.create_index(
        "ix_event_interceptors_session_id",
        "event_interceptors",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_event_interceptors_enabled",
        "event_interceptors",
        ["enabled"],
        unique=False,
    )
    op.create_index(
        "ix_event_interceptors_workspace_enabled_priority",
        "event_interceptors",
        ["workspace", "enabled", "priority"],
        unique=False,
    )
    op.create_index(
        "ix_event_interceptors_workspace_created_at",
        "event_interceptors",
        ["workspace", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_event_interceptors_workspace_created_at",
        table_name="event_interceptors",
    )
    op.drop_index(
        "ix_event_interceptors_workspace_enabled_priority",
        table_name="event_interceptors",
    )
    op.drop_index("ix_event_interceptors_enabled", table_name="event_interceptors")
    op.drop_index("ix_event_interceptors_session_id", table_name="event_interceptors")
    op.drop_index("ix_event_interceptors_app_id", table_name="event_interceptors")
    op.drop_index("ix_event_interceptors_workspace", table_name="event_interceptors")
    op.drop_table("event_interceptors")
