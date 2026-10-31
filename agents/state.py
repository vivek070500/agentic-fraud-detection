"""
Investigation State
=====================
Shared state that flows through the LangGraph investigation pipeline.
Each agent reads from and writes to this state.

Why TypedDict?
  - LangGraph requires it for state management
  - Every field change is tracked → full observability
  - The state IS the audit trail of the investigation
"""

from typing import TypedDict, Optional, Literal
from langgraph.graph import add_messages


class TransactionData(TypedDict):
    """Raw transaction information passed into the investigation."""
    transaction_id: str
    amount: float
    hour_of_day: int
    day_of_week: int
    merchant_category: str
    is_foreign_merchant: int
    customer_account_age_days: int
    customer_avg_amount_30d: float
    amount_deviation: float
    transaction_velocity_1h: int
    device_changed: int
    distance_from_home_km: float
    fraud_probability: float
    risk_level: str


class InvestigationState(TypedDict):
    """
    State that accumulates as agents analyze the transaction.

    Flow:
      1. transaction_data is set at start
      2. transaction_analysis is set by Transaction Analyst
      3. behavioral_analysis is set by Behavioral Analyst
      4. risk_summary is set by Risk Summarizer
      5. recommendation is the final output
      6. needs_human_review flags whether to escalate
    """
    # Input
    transaction_data: TransactionData

    # Agent outputs (each agent fills their section)
    transaction_analysis: str
    behavioral_analysis: str
    risk_summary: str

    # Final decision
    recommendation: Literal["APPROVE", "BLOCK", "ESCALATE_TO_HUMAN"]
    confidence: float
    reasoning: str

    # Control flow
    needs_human_review: bool
    investigation_complete: bool
    error: Optional[str]
