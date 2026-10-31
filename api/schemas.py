"""
Pydantic schemas for the Fraud Detection API.
Defines the contract for what comes in and what goes out.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ──

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicyDecision(str, Enum):
    AUTO_APPROVE = "AUTO_APPROVE"
    AUTO_BLOCK = "AUTO_BLOCK"
    INVESTIGATE = "INVESTIGATE"


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


# ── Request schemas ──

class TransactionRequest(BaseModel):
    """Incoming transaction to be scored."""
    transaction_id: Optional[str] = None
    amount: float = Field(..., gt=0, description="Transaction amount in dollars")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    merchant_category: str = Field(
        ...,
        description="One of: grocery, restaurant, gas_station, online_retail, "
                    "electronics, travel, subscription, atm"
    )
    is_foreign_merchant: int = Field(..., ge=0, le=1)
    customer_account_age_days: int = Field(..., ge=0)
    customer_avg_amount_30d: float = Field(..., ge=0)
    transaction_velocity_1h: int = Field(..., ge=0)
    device_changed: int = Field(..., ge=0, le=1)
    distance_from_home_km: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 499.99,
                "hour_of_day": 23,
                "day_of_week": 5,
                "merchant_category": "electronics",
                "is_foreign_merchant": 1,
                "customer_account_age_days": 15,
                "customer_avg_amount_30d": 45.00,
                "transaction_velocity_1h": 8,
                "device_changed": 1,
                "distance_from_home_km": 8500.0,
            }
        }


class HumanReviewRequest(BaseModel):
    """Human reviewer's decision on an investigated transaction."""
    transaction_id: str
    decision: ReviewStatus
    reviewer_notes: str = ""
    reviewer_id: str = "analyst_01"


# ── Response schemas ──

class FeatureContribution(BaseModel):
    feature: str
    value: float
    importance: float


class MLScoreResponse(BaseModel):
    fraud_probability: float
    risk_level: RiskLevel
    optimal_threshold: float
    top_features: list[FeatureContribution]


class PolicyResponse(BaseModel):
    decision: PolicyDecision
    reason: str


class TransactionResponse(BaseModel):
    """Full response for a scored transaction."""
    transaction_id: str
    timestamp: str
    ml_score: MLScoreResponse
    policy: PolicyResponse
    review_status: ReviewStatus
    investigation_id: Optional[str] = None


class AuditRecord(BaseModel):
    """What gets stored in the audit trail."""
    transaction_id: str
    timestamp: str
    amount: float
    merchant_category: str
    fraud_probability: float
    risk_level: str
    policy_decision: str
    review_status: str
    reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None
    investigation_summary: Optional[str] = None
    agent_trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_scored: int
    decisions: dict
