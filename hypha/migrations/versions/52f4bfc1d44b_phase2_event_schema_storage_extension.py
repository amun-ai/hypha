"""phase2_event_schema_storage_extension

Revision ID: 52f4bfc1d44b
Revises: 9096f050eb04
Create Date: 2026-02-17 20:05:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "52f4bfc1d44b"
down_revision: Union[str, None] = "9096f050eb04"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    op.create_table(
        "billing_retention_policies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("billable_usage_retention_days", sa.Integer(), nullable=False),
        sa.Column("operational_event_retention_days", sa.Integer(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "billing_accounts",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("stripe_customer_id", sa.String(length=255), nullable=True),
        sa.Column("plan_id", sa.String(length=255), nullable=True),
        sa.Column("price_id", sa.String(length=255), nullable=True),
        sa.Column("entitlement_snapshot_ref", sa.String(length=255), nullable=True),
        sa.Column("retention_policy_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["retention_policy_id"],
            ["billing_retention_policies.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_billing_accounts_retention_policy_id",
        "billing_accounts",
        ["retention_policy_id"],
        unique=False,
    )
    op.create_index(
        "ux_billing_accounts_stripe_customer_id",
        "billing_accounts",
        ["stripe_customer_id"],
        unique=True,
        postgresql_where=sa.text("stripe_customer_id IS NOT NULL"),
        sqlite_where=sa.text("stripe_customer_id IS NOT NULL"),
    )

    op.create_table(
        "workspace_billing_accounts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workspace", sa.String(length=255), nullable=False),
        sa.Column("billing_account_id", sa.String(length=255), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["billing_account_id"], ["billing_accounts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_workspace_billing_accounts_billing_account_id",
        "workspace_billing_accounts",
        ["billing_account_id"],
        unique=False,
    )
    op.create_index(
        "ix_workspace_billing_accounts_workspace",
        "workspace_billing_accounts",
        ["workspace"],
        unique=False,
    )
    op.create_index(
        "ux_workspace_billing_accounts_workspace_active",
        "workspace_billing_accounts",
        ["workspace"],
        unique=True,
        postgresql_where=sa.text("active = true"),
        sqlite_where=sa.text("active = 1"),
    )

    op.add_column(
        "event_logs",
        sa.Column(
            "category",
            sa.String(length=32),
            nullable=False,
            server_default="application",
        ),
    )
    op.add_column(
        "event_logs",
        sa.Column(
            "level",
            sa.String(length=32),
            nullable=False,
            server_default="info",
        ),
    )
    op.add_column("event_logs", sa.Column("app_id", sa.String(length=255), nullable=True))
    op.add_column(
        "event_logs",
        sa.Column("session_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "event_logs",
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "event_logs",
        sa.Column("billing_account_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "event_logs",
        sa.Column("retention_policy_id", sa.Integer(), nullable=True),
    )
    if not is_sqlite:
        op.create_foreign_key(
            "fk_event_logs_billing_account_id",
            "event_logs",
            "billing_accounts",
            ["billing_account_id"],
            ["id"],
        )
        op.create_foreign_key(
            "fk_event_logs_retention_policy_id",
            "event_logs",
            "billing_retention_policies",
            ["retention_policy_id"],
            ["id"],
        )
    op.create_index(
        "ix_event_logs_billing_account_id",
        "event_logs",
        ["billing_account_id"],
        unique=False,
    )
    op.create_index(
        "ix_event_logs_retention_policy_id",
        "event_logs",
        ["retention_policy_id"],
        unique=False,
    )
    op.create_index(
        "ix_event_logs_workspace_timestamp",
        "event_logs",
        ["workspace", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_event_logs_workspace_category_timestamp",
        "event_logs",
        ["workspace", "category", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_event_logs_workspace_event_type_timestamp",
        "event_logs",
        ["workspace", "event_type", "timestamp"],
        unique=False,
    )
    op.create_index(
        "ux_event_logs_workspace_idempotency_key_billing",
        "event_logs",
        ["workspace", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text(
            "category = 'billing' AND idempotency_key IS NOT NULL"
        ),
        sqlite_where=sa.text("category = 'billing' AND idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"

    op.drop_index("ux_event_logs_workspace_idempotency_key_billing", table_name="event_logs")
    op.drop_index("ix_event_logs_workspace_event_type_timestamp", table_name="event_logs")
    op.drop_index("ix_event_logs_workspace_category_timestamp", table_name="event_logs")
    op.drop_index("ix_event_logs_workspace_timestamp", table_name="event_logs")
    op.drop_index("ix_event_logs_retention_policy_id", table_name="event_logs")
    op.drop_index("ix_event_logs_billing_account_id", table_name="event_logs")
    if not is_sqlite:
        op.drop_constraint(
            "fk_event_logs_retention_policy_id", "event_logs", type_="foreignkey"
        )
        op.drop_constraint(
            "fk_event_logs_billing_account_id", "event_logs", type_="foreignkey"
        )
    op.drop_column("event_logs", "retention_policy_id")
    op.drop_column("event_logs", "billing_account_id")
    op.drop_column("event_logs", "idempotency_key")
    op.drop_column("event_logs", "session_id")
    op.drop_column("event_logs", "app_id")
    op.drop_column("event_logs", "level")
    op.drop_column("event_logs", "category")

    op.drop_index(
        "ux_workspace_billing_accounts_workspace_active",
        table_name="workspace_billing_accounts",
    )
    op.drop_index(
        "ix_workspace_billing_accounts_workspace",
        table_name="workspace_billing_accounts",
    )
    op.drop_index(
        "ix_workspace_billing_accounts_billing_account_id",
        table_name="workspace_billing_accounts",
    )
    op.drop_table("workspace_billing_accounts")

    op.drop_index("ux_billing_accounts_stripe_customer_id", table_name="billing_accounts")
    op.drop_index("ix_billing_accounts_retention_policy_id", table_name="billing_accounts")
    op.drop_table("billing_accounts")

    op.drop_table("billing_retention_policies")
