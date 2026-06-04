"""add shap_explanation jsonb column to predictions

Revision ID: c7e5a3f9b210
Revises: d2f1a4c7e890
Create Date: 2026-05-20 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = 'c7e5a3f9b210'
down_revision: Union[str, None] = 'd2f1a4c7e890'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column(
            "shap_explanation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("predictions", "shap_explanation")
