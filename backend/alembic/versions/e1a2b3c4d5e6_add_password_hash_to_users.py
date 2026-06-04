"""add password_hash to users for JWT auth

Revision ID: e1a2b3c4d5e6
Revises: c7e5a3f9b210
Create Date: 2026-05-26 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e1a2b3c4d5e6"
down_revision: Union[str, None] = "c7e5a3f9b210"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("password_hash", sa.String(length=128), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "password_hash")
