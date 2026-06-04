"""drop dead column predicted_image_size_mb (always NULL, unused)

Revision ID: f2b3c4d5e6f7
Revises: e1a2b3c4d5e6
Create Date: 2026-05-26 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f2b3c4d5e6f7"
down_revision: Union[str, None] = "e1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("predictions", "predicted_image_size_mb")


def downgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("predicted_image_size_mb", sa.Float(), nullable=True),
    )
