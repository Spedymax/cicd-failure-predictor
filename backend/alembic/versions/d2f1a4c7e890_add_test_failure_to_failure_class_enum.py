"""add test_failure to failure_class enum

Revision ID: d2f1a4c7e890
Revises: 9a4b1eda2bb1
Create Date: 2026-05-17 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd2f1a4c7e890'
down_revision: Union[str, None] = '9a4b1eda2bb1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE failure_class ADD VALUE IF NOT EXISTS 'TEST_FAILURE'")


def downgrade() -> None:
    # PostgreSQL has no DROP VALUE for enums; rolling back this revision
    # is forward-only — the value remains in the type definition.
    pass
