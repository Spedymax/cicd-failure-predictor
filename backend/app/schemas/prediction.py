from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import FailureClass, PredictionDecision


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class ShapExplanation(BaseModel):
    target: str = Field(description="What is being explained: 'risk_failure' or 'class:<name>'")
    base_value: float = Field(description="Expected model output E[f(x)] over training data")
    predicted_value: float = Field(
        description="base_value + sum(shap_value); model output on this sample"
    )
    contributions: list[ShapContribution]


class RecommendationOut(BaseModel):
    severity: str
    category: str
    title: str
    description: str
    actions: list[str]
    estimated_impact: str | None = None


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    id: int
    repository_id: int
    repository_full_name: str
    commit_sha: str
    commit_short: str = Field(description="First 7 chars of SHA")
    author_email: str
    branch: str | None
    workflow_name: str | None = None
    workflow_run_url: str | None = None
    predicted_class: FailureClass
    decision: PredictionDecision
    risk_score: float
    confidence: float
    class_probabilities: dict[str, float]
    feature_importance: dict[str, float]
    shap_explanation: ShapExplanation | None = None
    predicted_memory_mb: float | None
    predicted_duration_min: float | None
    recommendations: list[RecommendationOut]
    inference_time_ms: int
    overridden_at: datetime | None
    actual_outcome: FailureClass | None
    created_at: datetime


class PredictionListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    id: int
    repository_full_name: str
    commit_short: str
    author_email: str
    predicted_class: FailureClass
    decision: PredictionDecision
    risk_score: float
    confidence: float
    created_at: datetime


class PredictionListResponse(BaseModel):
    items: list[PredictionListItem]
    total: int
    limit: int
    offset: int


class WebhookAck(BaseModel):
    accepted: bool = True
    delivery_id: str | None = None
    repository: str | None = None
    commit_sha: str | None = None
