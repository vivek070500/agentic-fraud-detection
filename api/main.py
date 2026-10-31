"""
Fraud Detection API
=====================
FastAPI service that ties together ML scoring, policy engine, and audit trail.

Endpoints:
  POST /score          — Score a transaction (ML → Policy → Audit)
  POST /review         — Submit human review decision
  GET  /transaction/{id} — Get full audit trail for a transaction
  GET  /pending        — List transactions awaiting human review
  GET  /stats          — Dashboard stats
  GET  /health         — Health check

Run:
    uvicorn api.main:app --reload --port 8000
"""

import time
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml.predict import FraudScorer
from api.policy_engine import PolicyEngine
from api.audit import get_audit_trail
from api.schemas import (
    TransactionRequest,
    TransactionResponse,
    HumanReviewRequest,
    MLScoreResponse,
    PolicyResponse,
    FeatureContribution,
    AuditRecord,
    ReviewStatus,
    HealthResponse,
)


# ── Global state ──
scorer: FraudScorer = None
policy: PolicyEngine = None
audit = None
start_time: float = 0
request_count: int = 0
decision_counts: dict = {"AUTO_APPROVE": 0, "AUTO_BLOCK": 0, "INVESTIGATE": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model and init services on startup."""
    global scorer, policy, audit, start_time

    print("🚀 Loading ML model...")
    scorer = FraudScorer()
    print(f"   Model loaded: ROC-AUC {scorer.metadata['metrics']['roc_auc']}")
    print(f"   Optimal threshold: {scorer.optimal_threshold}")

    policy = PolicyEngine()
    print(f"   Policy engine: approve < {policy.auto_approve_threshold}, "
          f"block > {policy.auto_block_threshold}")

    audit = get_audit_trail()
    print(f"   Audit trail: {audit.__class__.__name__}")

    start_time = time.time()
    print("✅ API ready\n")

    yield

    print("Shutting down...")


app = FastAPI(
    title="Agentic Fraud Detection API",
    description=(
        "ML-powered fraud scoring with deterministic policy routing, "
        "agentic investigation for ambiguous cases, and human-in-the-loop review."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──


@app.post("/score", response_model=TransactionResponse)
async def score_transaction(txn: TransactionRequest):
    """
    Score a transaction end-to-end:
    1. ML model predicts fraud probability
    2. Policy engine makes a routing decision
    3. Decision is recorded in audit trail
    4. If INVESTIGATE → will be picked up by agentic pipeline (Milestone 3)
    """
    global request_count

    # Generate transaction ID if not provided
    txn_id = txn.transaction_id or f"txn_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.utcnow().isoformat()

    # Step 1: ML Scoring
    txn_dict = txn.model_dump()
    ml_result = scorer.score(txn_dict)

    ml_response = MLScoreResponse(
        fraud_probability=ml_result["fraud_probability"],
        risk_level=ml_result["risk_level"],
        optimal_threshold=ml_result["optimal_threshold"],
        top_features=[FeatureContribution(**f) for f in ml_result["top_features"]],
    )

    # Step 2: Policy Decision
    policy_result = policy.evaluate(ml_result["fraud_probability"], txn_dict)

    # Step 3: Determine review status
    if policy_result.decision.value == "AUTO_APPROVE":
        review_status = ReviewStatus.APPROVED
    elif policy_result.decision.value == "AUTO_BLOCK":
        review_status = ReviewStatus.REJECTED
    else:
        review_status = ReviewStatus.PENDING

    # Step 4: Record in audit trail
    audit_record = AuditRecord(
        transaction_id=txn_id,
        timestamp=timestamp,
        amount=txn.amount,
        merchant_category=txn.merchant_category,
        fraud_probability=ml_result["fraud_probability"],
        risk_level=ml_result["risk_level"],
        policy_decision=policy_result.decision.value,
        review_status=review_status.value,
    )
    audit.record_decision(audit_record)

    # Update counters
    request_count += 1
    decision_counts[policy_result.decision.value] = (
        decision_counts.get(policy_result.decision.value, 0) + 1
    )

    return TransactionResponse(
        transaction_id=txn_id,
        timestamp=timestamp,
        ml_score=ml_response,
        policy=policy_result,
        review_status=review_status,
    )


@app.post("/review")
async def submit_review(review: HumanReviewRequest):
    """
    Submit a human reviewer's decision on a transaction.
    This is the HITL endpoint — used by the Streamlit dashboard.
    """
    txn = audit.get_transaction(review.transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    success = audit.update_review(
        transaction_id=review.transaction_id,
        status=review.decision.value,
        reviewer_id=review.reviewer_id,
        notes=review.reviewer_notes,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update review")

    return {
        "status": "ok",
        "transaction_id": review.transaction_id,
        "decision": review.decision.value,
        "message": f"Transaction {review.transaction_id} marked as {review.decision.value} "
                   f"by {review.reviewer_id}",
    }


@app.get("/transaction/{transaction_id}")
async def get_transaction(transaction_id: str):
    """Get full audit trail for a specific transaction."""
    txn = audit.get_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return txn


@app.get("/pending")
async def get_pending_reviews(limit: int = 50):
    """
    List transactions awaiting human review.
    Sorted by fraud probability (highest risk first).
    Used by the Streamlit dashboard to populate the review queue.
    """
    pending = audit.get_pending_reviews(limit=limit)
    return {
        "count": len(pending),
        "transactions": pending,
    }


@app.get("/recent")
async def get_recent_transactions(limit: int = 100):
    """Get recent transactions for the dashboard feed."""
    recent = audit.get_recent(limit=limit)
    return {
        "count": len(recent),
        "transactions": recent,
    }


@app.get("/stats")
async def get_stats():
    """Aggregate stats for the monitoring dashboard."""
    db_stats = audit.get_stats()
    return {
        **db_stats,
        "session": {
            "requests_this_session": request_count,
            "decisions_this_session": decision_counts,
            "uptime_seconds": round(time.time() - start_time, 1),
        },
    }


@app.post("/investigate/{transaction_id}")
async def investigate_transaction(transaction_id: str):
    """
    Trigger agentic investigation on a gray-zone transaction.

    This is the Milestone 3 endpoint:
    1. Looks up transaction from audit trail
    2. Runs LangGraph multi-agent investigation (3 agents)
    3. Updates audit trail with findings + recommendation
    4. If agents say ESCALATE → stays PENDING for human review
    5. If agents say APPROVE/BLOCK → updates status automatically
    6. Every step traced in Langfuse (if configured)
    """
    from agents.graph import investigate_transaction as run_investigation
    from observability.langfuse_config import create_investigation_handler

    # Get transaction from audit trail
    txn = audit.get_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    if txn.get("policy_decision") != "INVESTIGATE":
        raise HTTPException(
            status_code=400,
            detail=f"Transaction {transaction_id} has policy_decision="
                   f"'{txn.get('policy_decision')}', not INVESTIGATE. "
                   f"Only gray-zone transactions need investigation."
        )

    # Build transaction data for agents
    txn_data = {
        "transaction_id": transaction_id,
        "amount": txn.get("amount", 0),
        "hour_of_day": 0,
        "day_of_week": 0,
        "merchant_category": txn.get("merchant_category", "unknown"),
        "is_foreign_merchant": 0,
        "customer_account_age_days": 0,
        "customer_avg_amount_30d": 0,
        "amount_deviation": 0,
        "transaction_velocity_1h": 0,
        "device_changed": 0,
        "distance_from_home_km": 0,
        "fraud_probability": txn.get("fraud_probability", 0),
        "risk_level": txn.get("risk_level", "MEDIUM"),
    }

    # Create Langfuse handler for observability
    langfuse_handler = create_investigation_handler(
        transaction_id=transaction_id,
        fraud_probability=txn.get("fraud_probability", 0),
    )

    # Run the multi-agent investigation
    result = await run_investigation(
        transaction_data=txn_data,
        langfuse_handler=langfuse_handler,
    )

    # Build investigation summary
    summary = (
        f"RECOMMENDATION: {result['recommendation']} "
        f"(confidence: {result['confidence']:.0%})\n"
        f"REASONING: {result['reasoning']}\n\n"
        f"--- Transaction Analysis ---\n{result.get('transaction_analysis', 'N/A')}\n\n"
        f"--- Behavioral Analysis ---\n{result.get('behavioral_analysis', 'N/A')}\n\n"
        f"--- Risk Summary ---\n{result.get('risk_summary', 'N/A')}"
    )

    # Update audit trail with investigation findings
    audit.update_investigation(
        transaction_id=transaction_id,
        summary=summary,
        trace_id=transaction_id,
    )

    # If agents are confident, auto-update the review status
    if result["recommendation"] == "APPROVE" and result["confidence"] >= 0.7:
        audit.update_review(
            transaction_id=transaction_id,
            status="APPROVED",
            reviewer_id="agent_system",
            notes=f"Auto-approved by agents (confidence: {result['confidence']:.0%}). "
                  f"Reason: {result['reasoning']}",
        )
    elif result["recommendation"] == "BLOCK" and result["confidence"] >= 0.8:
        audit.update_review(
            transaction_id=transaction_id,
            status="REJECTED",
            reviewer_id="agent_system",
            notes=f"Auto-blocked by agents (confidence: {result['confidence']:.0%}). "
                  f"Reason: {result['reasoning']}",
        )
    # If ESCALATE_TO_HUMAN → status stays PENDING for human dashboard

    return {
        "transaction_id": transaction_id,
        "recommendation": result["recommendation"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
        "needs_human_review": result.get("needs_human_review", True),
        "transaction_analysis": result.get("transaction_analysis", ""),
        "behavioral_analysis": result.get("behavioral_analysis", ""),
        "risk_summary": result.get("risk_summary", ""),
        "error": result.get("error"),
    }


@app.post("/score-and-investigate", response_model=None)
async def score_and_investigate(txn: TransactionRequest):
    """
    Full pipeline in one call: Score → Policy → (If gray zone) → Investigate.
    Combines /score and /investigate for convenience.
    """
    # First, score the transaction
    score_response = await score_transaction(txn)

    response = score_response.model_dump()
    response["investigation"] = None

    # If policy says INVESTIGATE, run the agents
    if score_response.policy.decision.value == "INVESTIGATE":
        investigation = await investigate_transaction(score_response.transaction_id)
        response["investigation"] = investigation

    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — used by load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        model_loaded=scorer is not None,
        uptime_seconds=round(time.time() - start_time, 1),
        total_scored=request_count,
        decisions=decision_counts,
    )
