"""monitoring metrics + alerts

Revision ID: 0fa7bbcd822b
Revises: c5b55ce3699d
Create Date: 2025-10-08 14:10:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0fa7bbcd822b"
down_revision: Union[str, Sequence[str], None] = "c5b55ce3699d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(bind, name: str, schema: str | None = None) -> bool:
    insp = sa.inspect(bind)
    return name in insp.get_table_names(schema=schema)


def _index_exists(bind, table: str, name: str, schema: str | None = None) -> bool:
    insp = sa.inspect(bind)
    for ix in insp.get_indexes(table_name=table, schema=schema):
        if ix.get("name") in {name, op.f(name)}:
            return True
    return False


def upgrade() -> None:
    """Create monitoring_metrics and alerts (idempotent)."""
    bind = op.get_bind()

    # ---- MONITORING METRICS ----
    if not _table_exists(bind, "monitoring_metrics", "public"):
        op.create_table(
            "monitoring_metrics",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("model_id", sa.Integer, nullable=False),
            sa.Column("feature", sa.String(length=255), nullable=True),
            sa.Column("method", sa.String(length=64), nullable=False),   # psi_baseline | psi_eval | ks_eval | dq_check ...
            sa.Column("value", sa.String(length=128), nullable=True),
            sa.Column("details", postgresql.JSON(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", postgresql.TIMESTAMP(), nullable=False, server_default=sa.text("now()")),
        )
    # indexes for metrics
    if not _index_exists(bind, "monitoring_metrics", "ix_monitoring_metrics_model_id", "public"):
        op.create_index(op.f("ix_monitoring_metrics_model_id"), "monitoring_metrics", ["model_id"], unique=False)
    if not _index_exists(bind, "monitoring_metrics", "ix_monitoring_metrics_feature", "public"):
        op.create_index(op.f("ix_monitoring_metrics_feature"), "monitoring_metrics", ["feature"], unique=False)
    if not _index_exists(bind, "monitoring_metrics", "ix_monitoring_metrics_method", "public"):
        op.create_index(op.f("ix_monitoring_metrics_method"), "monitoring_metrics", ["method"], unique=False)

    # ---- ALERTS ----
    if not _table_exists(bind, "alerts", "public"):
        op.create_table(
            "alerts",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("model_id", sa.Integer, nullable=False),
            sa.Column("type", sa.String(length=64), nullable=False),      # "drift", "dq", etc.
            sa.Column("severity", sa.String(length=16), nullable=False),  # "low"|"medium"|"high"
            sa.Column("message", sa.Text, nullable=False),
            sa.Column("details", postgresql.JSON(astext_type=sa.Text()), nullable=True),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="open"),
            sa.Column("created_at", postgresql.TIMESTAMP(), nullable=False, server_default=sa.text("now()")),
            sa.Column("resolved_at", postgresql.TIMESTAMP(), nullable=True),
        )
    # indexes for alerts
    if not _index_exists(bind, "alerts", "ix_alerts_model_id", "public"):
        op.create_index(op.f("ix_alerts_model_id"), "alerts", ["model_id"], unique=False)
    if not _index_exists(bind, "alerts", "ix_alerts_status", "public"):
        op.create_index(op.f("ix_alerts_status"), "alerts", ["status"], unique=False)
    if not _index_exists(bind, "alerts", "ix_alerts_severity", "public"):
        op.create_index(op.f("ix_alerts_severity"), "alerts", ["severity"], unique=False)


def downgrade() -> None:
    """Drop the same objects safely."""
    # Drop indexes first (guards make this safe on any state)
    op.execute("DROP INDEX IF EXISTS public.ix_alerts_severity")
    op.execute("DROP INDEX IF EXISTS public.ix_alerts_status")
    op.execute("DROP INDEX IF EXISTS public.ix_alerts_model_id")
    op.execute("DROP TABLE IF EXISTS public.alerts CASCADE")

    op.execute("DROP INDEX IF EXISTS public.ix_monitoring_metrics_method")
    op.execute("DROP INDEX IF EXISTS public.ix_monitoring_metrics_feature")
    op.execute("DROP INDEX IF EXISTS public.ix_monitoring_metrics_model_id")
    op.execute("DROP TABLE IF EXISTS public.monitoring_metrics CASCADE")
