"""approvals + audit

Revision ID: c5b55ce3699d
Revises: 7f15fb6879c0
Create Date: 2025-10-08 13:47:26.209759
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c5b55ce3699d"
down_revision: Union[str, Sequence[str], None] = "7f15fb6879c0"
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
    """Create required tables/indexes if they don't already exist."""
    bind = op.get_bind()

    # ---- MODELS ----
    if not _table_exists(bind, "models", "public"):
        op.create_table(
            "models",
            sa.Column("id", sa.INTEGER(), server_default=sa.text("nextval('models_id_seq'::regclass)"), autoincrement=True, nullable=False),
            sa.Column("name", sa.VARCHAR(length=255), nullable=False),
            sa.Column("description", sa.TEXT(), nullable=True),
            sa.Column("version", sa.VARCHAR(length=50), nullable=False),
            sa.Column("stage", sa.VARCHAR(length=50), nullable=False),
            sa.Column("status", sa.VARCHAR(length=50), nullable=False),
            sa.Column("model_metadata", postgresql.JSON(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", postgresql.TIMESTAMP(), nullable=False),
            sa.Column("updated_at", postgresql.TIMESTAMP(), nullable=False),
            sa.PrimaryKeyConstraint("id", name="models_pkey"),
            postgresql_ignore_search_path=False,
        )
    # indices
    if not _index_exists(bind, "models", "ix_models_name", "public"):
        op.create_index(op.f("ix_models_name"), "models", ["name"], unique=True)
    if not _index_exists(bind, "models", "ix_models_id", "public"):
        op.create_index(op.f("ix_models_id"), "models", ["id"], unique=False)

    # ---- AUDIT LOG ----
    if not _table_exists(bind, "audit_log", "public"):
        op.create_table(
            "audit_log",
            sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
            sa.Column("action", sa.VARCHAR(length=128), nullable=True),
            sa.Column("model_id", sa.INTEGER(), nullable=True),
            sa.Column("actor", sa.VARCHAR(length=255), nullable=True),
            sa.Column("details", postgresql.JSON(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", postgresql.TIMESTAMP(), nullable=False),
            sa.PrimaryKeyConstraint("id", name=op.f("audit_log_pkey")),
        )
    if not _index_exists(bind, "audit_log", "ix_audit_log_model_id", "public"):
        op.create_index(op.f("ix_audit_log_model_id"), "audit_log", ["model_id"], unique=False)
    if not _index_exists(bind, "audit_log", "ix_audit_log_action", "public"):
        op.create_index(op.f("ix_audit_log_action"), "audit_log", ["action"], unique=False)

    # ---- APPROVAL REQUESTS (legacy table some older code expects) ----
    if not _table_exists(bind, "approval_requests", "public"):
        op.create_table(
            "approval_requests",
            sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
            sa.Column("model_id", sa.INTEGER(), nullable=False),
            sa.Column("target", sa.VARCHAR(length=32), nullable=False),
            sa.Column("justification", sa.VARCHAR(length=1000), nullable=True),
            sa.Column("status", sa.VARCHAR(length=32), nullable=True),
            sa.Column("requested_by", sa.VARCHAR(length=255), nullable=True),
            sa.Column("created_at", postgresql.TIMESTAMP(), nullable=True),
            sa.Column("decisions", postgresql.JSON(astext_type=sa.Text()), nullable=True),
            sa.ForeignKeyConstraint(["model_id"], ["models.id"], name=op.f("approval_requests_model_id_fkey")),
            sa.PrimaryKeyConstraint("id", name=op.f("approval_requests_pkey")),
        )
    if not _index_exists(bind, "approval_requests", "ix_approval_requests_model_id", "public"):
        op.create_index(op.f("ix_approval_requests_model_id"), "approval_requests", ["model_id"], unique=False)


def downgrade() -> None:
    """Drop the same objects (guarded) to roll back this revision."""
    # Drop in reverse dependency order
    op.execute("DROP TABLE IF EXISTS public.approval_requests CASCADE")
    op.execute("DROP INDEX IF EXISTS public.ix_audit_log_action")
    op.execute("DROP INDEX IF EXISTS public.ix_audit_log_model_id")
    op.execute("DROP TABLE IF EXISTS public.audit_log CASCADE")
    op.execute("DROP INDEX IF EXISTS public.ix_models_name")
    op.execute("DROP INDEX IF EXISTS public.ix_models_id")
    op.execute("DROP TABLE IF EXISTS public.models CASCADE")
