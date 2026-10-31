"""
Langfuse Observability
========================
Integrates Langfuse tracing with the LangGraph investigation pipeline.

What gets traced:
  - Every LLM call (prompt, response, tokens, latency)
  - Every agent node execution
  - The full investigation flow (start → analyst → behavioral → summary → end)
  - Custom metadata (transaction_id, fraud_probability, etc.)

Setup:
  1. Create free account at https://cloud.langfuse.com
  2. Create a project, get API keys
  3. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY in .env

Free tier: 50,000 observations/month — plenty for a portfolio project.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def get_langfuse_handler(
    session_id: Optional[str] = None,
    trace_name: str = "fraud-investigation",
    metadata: Optional[dict] = None,
):
    """
    Create a Langfuse callback handler for LangGraph/LangChain.

    Returns None if Langfuse is not configured (graceful degradation).
    The investigation pipeline works fine without it — just no tracing.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        return None

    try:
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            session_id=session_id,
            trace_name=trace_name,
            metadata=metadata or {},
        )
        return handler
    except ImportError:
        print("⚠️  langfuse not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        print(f"⚠️  Langfuse init failed: {e}. Continuing without tracing.")
        return None


def create_investigation_handler(transaction_id: str, fraud_probability: float):
    """
    Convenience wrapper — creates a Langfuse handler pre-configured
    for a fraud investigation trace.

    In the Langfuse UI, you'll see:
      Session: txn_gray_001
      Trace: fraud-investigation
      Metadata: {transaction_id, fraud_probability}
      Spans: transaction_analyst → behavioral_analyst → risk_summarizer
    """
    return get_langfuse_handler(
        session_id=transaction_id,
        trace_name="fraud-investigation",
        metadata={
            "transaction_id": transaction_id,
            "fraud_probability": fraud_probability,
            "pipeline": "agentic-fraud-detection",
        },
    )
