"""add other_failure to failure_class enum

Revision ID: 9a4b1eda2bb1
Revises: af8ae9ba8ea1
Create Date: 2026-05-10 15:27:05.586648

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '9a4b1eda2bb1'
down_revision: Union[str, None] = 'af8ae9ba8ea1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE failure_class ADD VALUE IF NOT EXISTS 'OTHER_FAILURE'")


def downgrade() -> None:
    # PostgreSQL has no DROP VALUE for enums; rolling back this revision
    # is forward-only — the value remains in the type definition.
    pass
