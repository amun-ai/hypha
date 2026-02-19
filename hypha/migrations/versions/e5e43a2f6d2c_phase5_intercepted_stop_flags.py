"""phase5_intercepted_stop_flags

Revision ID: e5e43a2f6d2c
Revises: d91b7f3ac4e2
Create Date: 2026-02-19 20:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e5e43a2f6d2c"
down_revision: Union[str, None] = "d91b7f3ac4e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "event_logs",
        sa.Column(
            "intercepted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "event_logs",
        sa.Column("interceptor_action", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "event_logs",
        sa.Column("interceptor_reason", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "event_logs",
        sa.Column("interceptor_code", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "event_logs",
        sa.Column("interceptor_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "event_logs",
        sa.Column("interceptor_name", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "event_logs",
        sa.Column("intercepted_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_event_logs_workspace_intercepted_timestamp",
        "event_logs",
        ["workspace", "intercepted", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_event_logs_workspace_intercepted_timestamp",
        table_name="event_logs",
    )
    op.drop_column("event_logs", "intercepted_at")
    op.drop_column("event_logs", "interceptor_name")
    op.drop_column("event_logs", "interceptor_id")
    op.drop_column("event_logs", "interceptor_code")
    op.drop_column("event_logs", "interceptor_reason")
    op.drop_column("event_logs", "interceptor_action")
    op.drop_column("event_logs", "intercepted")
