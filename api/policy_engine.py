"""
Deterministic Policy Engine
=============================
Takes an ML fraud score and routes the transaction through business rules.

This layer exists BETWEEN the ML model and the agentic investigation.

Why separate from ML?
  - ML gives a probability. Policy gives a DECISION.
  - Thresholds are business decisions, not ML decisions.
  - Compliance team can change thresholds without retraining the model.
  - Audit trail shows: "model said 0.73, policy said INVESTIGATE because
    score was in the gray zone (0.2–0.8)"

Decision zones (from Experiment 2 findings):
  - score < 0.2  → AUTO_APPROVE  (96%+ of traffic, safe)
  - score > 0.8  → AUTO_BLOCK    (high confidence fraud)
  - 0.2 ≤ score ≤ 0.8 → INVESTIGATE (gray zone → agents + human)
"""

from api.schemas import PolicyDecision, PolicyResponse


class PolicyEngine:
    """Deterministic routing based on ML score + business rules."""

    def __init__(
        self,
        auto_approve_threshold: float = 0.2,
        auto_block_threshold: float = 0.8,
    ):
        self.auto_approve_threshold = auto_approve_threshold
        self.auto_block_threshold = auto_block_threshold

    def evaluate(self, fraud_probability: float, transaction: dict) -> PolicyResponse:
        """
        Route a transaction based on its fraud score.

        Additional business rules can be layered on top:
        - High-value transactions get extra scrutiny
        - New accounts have stricter thresholds
        - Certain merchant categories are higher risk
        """

        # ── Override rules (applied BEFORE threshold logic) ──

        # Rule: Very high amount always gets investigation, even if ML says low risk
        amount = transaction.get("amount", 0)
        if amount > 5000 and fraud_probability > 0.1:
            return PolicyResponse(
                decision=PolicyDecision.INVESTIGATE,
                reason=f"High-value transaction (${amount:,.2f}) with non-trivial "
                       f"risk score ({fraud_probability:.2f}). Requires investigation."
            )

        # Rule: Brand new account (< 7 days) with foreign merchant gets investigation
        account_age = transaction.get("customer_account_age_days", 365)
        is_foreign = transaction.get("is_foreign_merchant", 0)
        if account_age < 7 and is_foreign and fraud_probability > 0.15:
            return PolicyResponse(
                decision=PolicyDecision.INVESTIGATE,
                reason=f"New account ({account_age} days) + foreign merchant "
                       f"with risk score {fraud_probability:.2f}. Requires investigation."
            )

        # ── Standard threshold routing ──

        if fraud_probability < self.auto_approve_threshold:
            return PolicyResponse(
                decision=PolicyDecision.AUTO_APPROVE,
                reason=f"Fraud score {fraud_probability:.4f} is below auto-approve "
                       f"threshold ({self.auto_approve_threshold}). Transaction is low risk."
            )

        if fraud_probability > self.auto_block_threshold:
            return PolicyResponse(
                decision=PolicyDecision.AUTO_BLOCK,
                reason=f"Fraud score {fraud_probability:.4f} exceeds auto-block "
                       f"threshold ({self.auto_block_threshold}). High confidence fraud."
            )

        # Gray zone
        return PolicyResponse(
            decision=PolicyDecision.INVESTIGATE,
            reason=f"Fraud score {fraud_probability:.4f} is in the gray zone "
                   f"({self.auto_approve_threshold}–{self.auto_block_threshold}). "
                   f"ML is not confident — requires agentic investigation + human review."
        )
