"""
Investigation Graph (LangGraph)
=================================
Multi-agent investigation pipeline for gray-zone fraud transactions.

Graph structure:
    START → transaction_analyst → behavioral_analyst → risk_summarizer → END

Supports multiple LLM providers:
  - Google Gemini Flash (free tier — default)
  - OpenAI GPT-4o-mini (paid, cheap)
  - Configurable via LLM_PROVIDER env var

Usage:
    from agents.graph import investigate_transaction
    result = await investigate_transaction(transaction_data)

    # Or run from CLI:
    python -m agents.graph
"""

import os
import json
import asyncio
from typing import Optional
from functools import partial

from langgraph.graph import StateGraph, START, END

from agents.state import InvestigationState
from agents.nodes.transaction_analyst import transaction_analyst
from agents.nodes.behavioral_analyst import behavioral_analyst
from agents.nodes.risk_summarizer import risk_summarizer


def get_llm(provider: str = None, temperature: float = 0.1, callbacks=None):
    """
    Factory: create LLM based on provider config.

    Checks in order:
      1. Explicit provider arg
      2. LLM_PROVIDER env var
      3. Auto-detect based on which API key is set
    """
    provider = provider or os.getenv("LLM_PROVIDER", "auto")

    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GOOGLE_API_KEY", "")

    kwargs = {"temperature": temperature}
    if callbacks:
        kwargs["callbacks"] = callbacks

    # Auto-detect: prefer Groq (free) → Gemini (free) → OpenAI (paid)
    if provider == "auto":
        if groq_key:
            provider = "groq"
        elif gemini_key:
            provider = "gemini"
        elif openai_key:
            provider = "openai"
        else:
            return None, "No LLM API key configured. Set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in .env"

    if provider == "groq":
        if not groq_key:
            return None, "GROQ_API_KEY not set in .env"
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )
        return llm, None

    if provider == "gemini":
        if not gemini_key:
            return None, "GOOGLE_API_KEY not set in .env"
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            google_api_key=gemini_key,
            **kwargs,
        )
        return llm, None

    if provider == "openai":
        if not openai_key:
            return None, "OPENAI_API_KEY not set in .env"
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            **kwargs,
        )
        return llm, None

    return None, f"Unknown LLM_PROVIDER: {provider}. Use 'groq', 'gemini', or 'openai'."


def build_graph(llm) -> StateGraph:
    """
    Build the investigation state graph.

    Each node is an async function that:
      1. Reads from InvestigationState
      2. Calls the LLM
      3. Returns a dict of state updates
    """
    graph = StateGraph(InvestigationState)

    graph.add_node("transaction_analyst", partial(transaction_analyst, llm=llm))
    graph.add_node("behavioral_analyst", partial(behavioral_analyst, llm=llm))
    graph.add_node("risk_summarizer", partial(risk_summarizer, llm=llm))

    graph.add_edge(START, "transaction_analyst")
    graph.add_edge("transaction_analyst", "behavioral_analyst")
    graph.add_edge("behavioral_analyst", "risk_summarizer")
    graph.add_edge("risk_summarizer", END)

    return graph.compile()


async def investigate_transaction(
    transaction_data: dict,
    provider: str = None,
    temperature: float = 0.1,
    langfuse_handler=None,
) -> dict:
    """
    Run a full investigation on a transaction.

    Args:
        transaction_data: Transaction dict (from ML scorer output)
        provider: LLM provider ('gemini', 'openai', or 'auto')
        temperature: LLM temperature (low for consistency)
        langfuse_handler: Optional Langfuse callback for tracing

    Returns:
        dict with recommendation, confidence, reasoning, and full analysis
    """
    callbacks = [langfuse_handler] if langfuse_handler else None
    llm, error = get_llm(provider=provider, temperature=temperature, callbacks=callbacks)

    if error:
        return {
            "recommendation": "ESCALATE_TO_HUMAN",
            "confidence": 0.0,
            "reasoning": error,
            "error": error,
        }

    graph = build_graph(llm)

    initial_state: InvestigationState = {
        "transaction_data": transaction_data,
        "transaction_analysis": "",
        "behavioral_analysis": "",
        "risk_summary": "",
        "recommendation": "ESCALATE_TO_HUMAN",
        "confidence": 0.0,
        "reasoning": "",
        "needs_human_review": True,
        "investigation_complete": False,
        "error": None,
    }

    config = {}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        return {
            "recommendation": "ESCALATE_TO_HUMAN",
            "confidence": 0.0,
            "reasoning": f"Investigation failed: {str(e)}. Escalating to human.",
            "error": str(e),
            "transaction_analysis": "",
            "behavioral_analysis": "",
            "risk_summary": "",
            "needs_human_review": True,
            "investigation_complete": False,
        }

    return {
        "recommendation": result.get("recommendation", "ESCALATE_TO_HUMAN"),
        "confidence": result.get("confidence", 0.0),
        "reasoning": result.get("reasoning", ""),
        "transaction_analysis": result.get("transaction_analysis", ""),
        "behavioral_analysis": result.get("behavioral_analysis", ""),
        "risk_summary": result.get("risk_summary", ""),
        "needs_human_review": result.get("needs_human_review", True),
        "investigation_complete": result.get("investigation_complete", False),
    }


def investigate_sync(transaction_data: dict, **kwargs) -> dict:
    """Synchronous wrapper for use in non-async contexts."""
    return asyncio.run(investigate_transaction(transaction_data, **kwargs))


# ── CLI for testing ──

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    test_txn = {
        "transaction_id": "txn_investigate_001",
        "amount": 780.00,
        "hour_of_day": 23,
        "day_of_week": 5,
        "merchant_category": "electronics",
        "is_foreign_merchant": 1,
        "customer_account_age_days": 45,
        "customer_avg_amount_30d": 120.00,
        "amount_deviation": 6.5,
        "transaction_velocity_1h": 4,
        "device_changed": 1,
        "distance_from_home_km": 5200.0,
        "fraud_probability": 0.65,
        "risk_level": "HIGH",
    }

    print("=" * 60)
    print("  AGENTIC INVESTIGATION — CLI Test")
    print("=" * 60)
    print(f"\nTransaction: {test_txn['transaction_id']}")
    print(f"Amount: ${test_txn['amount']:,.2f} ({test_txn['amount_deviation']}x average)")
    print(f"ML Score: {test_txn['fraud_probability']:.0%} ({test_txn['risk_level']})")

    provider = os.getenv("LLM_PROVIDER", "auto")
    print(f"LLM Provider: {provider}")
    print(f"\nRunning investigation agents...")
    print("-" * 60)

    result = investigate_sync(test_txn)

    print(f"\n{'='*60}")
    print(f"  INVESTIGATION RESULTS")
    print(f"{'='*60}")

    print(f"\n📋 Transaction Analysis:")
    print(f"   {result.get('transaction_analysis', 'N/A')}")

    print(f"\n🔍 Behavioral Analysis:")
    print(f"   {result.get('behavioral_analysis', 'N/A')}")

    print(f"\n📊 Risk Summary:")
    print(f"   {result.get('risk_summary', 'N/A')}")

    print(f"\n{'─'*60}")
    print(f"  RECOMMENDATION: {result['recommendation']}")
    print(f"  CONFIDENCE:     {result['confidence']:.0%}")
    print(f"  REASONING:      {result['reasoning']}")
    print(f"  NEEDS HUMAN:    {'Yes' if result.get('needs_human_review', True) else 'No'}")
    print(f"{'─'*60}")

    if result.get("error"):
        print(f"\n⚠️  Error: {result['error']}")
