"""phase4_billing_hooks_usage_tracking

Revision ID: d91b7f3ac4e2
Revises: c8f6df9a1c1e
Create Date: 2026-02-18 08:55:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d91b7f3ac4e2"
down_revision: Union[str, None] = "c8f6df9a1c1e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "billing_accounts",
        sa.Column("stripe_subscription_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("stripe_subscription_status", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("current_period_start", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("current_period_end", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("last_webhook_event_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("last_webhook_event_ts", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "billing_accounts",
        sa.Column("stripe_snapshot", sa.JSON(), nullable=True),
    )

    op.create_index(
        "ix_billing_accounts_stripe_subscription_id",
        "billing_accounts",
        ["stripe_subscription_id"],
        unique=False,
    )
    op.create_index(
        "ix_billing_accounts_last_webhook_event_ts",
        "billing_accounts",
        ["last_webhook_event_ts"],
        unique=False,
    )

    op.create_table(
        "stripe_webhook_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("stripe_event_id", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=255), nullable=False),
        sa.Column("created_ts", sa.DateTime(), nullable=True),
        sa.Column("processed_at", sa.DateTime(), nullable=False),
        sa.Column("payload_hash", sa.String(length=255), nullable=False),
        sa.Column("billing_account_id", sa.String(length=255), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["billing_account_id"], ["billing_accounts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_stripe_webhook_events_billing_account_id",
        "stripe_webhook_events",
        ["billing_account_id"],
        unique=False,
    )
    op.create_index(
        "ux_stripe_webhook_events_stripe_event_id",
        "stripe_webhook_events",
        ["stripe_event_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ux_stripe_webhook_events_stripe_event_id",
        table_name="stripe_webhook_events",
    )
    op.drop_index(
        "ix_stripe_webhook_events_billing_account_id",
        table_name="stripe_webhook_events",
    )
    op.drop_table("stripe_webhook_events")

    op.drop_index(
        "ix_billing_accounts_last_webhook_event_ts",
        table_name="billing_accounts",
    )
    op.drop_index(
        "ix_billing_accounts_stripe_subscription_id",
        table_name="billing_accounts",
    )

    op.drop_column("billing_accounts", "stripe_snapshot")
    op.drop_column("billing_accounts", "last_webhook_event_ts")
    op.drop_column("billing_accounts", "last_webhook_event_id")
    op.drop_column("billing_accounts", "current_period_end")
    op.drop_column("billing_accounts", "current_period_start")
    op.drop_column("billing_accounts", "stripe_subscription_status")
    op.drop_column("billing_accounts", "stripe_subscription_id")
