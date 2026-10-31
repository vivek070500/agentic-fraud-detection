"""
Risk Summarizer Agent
=======================
Combines findings from Transaction Analyst and Behavioral Analyst
into a final structured recommendation.

This is the "judge" — it reads all evidence and decides:
  APPROVE  → confident this is legitimate (releases the hold)
  BLOCK    → confident this is fraud (permanent block)
  ESCALATE → still ambiguous, needs human judgment
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import InvestigationState

SYSTEM_PROMPT = """You are a Senior Risk Analyst making the final fraud determination.
You have received analyses from two specialist agents:
1. Transaction Analyst — analyzed the financial characteristics
2. Behavioral Analyst — analyzed user behavior patterns

Your job is to COMBINE their findings and make a final recommendation.

DECISION FRAMEWORK:
- APPROVE: If most signals are LOW risk and there's a reasonable explanation
  for any medium signals. Confidence should be > 0.7.
- BLOCK: If multiple HIGH risk signals align and the pattern matches known
  fraud tactics. Confidence should be > 0.8.
- ESCALATE_TO_HUMAN: If signals are mixed, contradictory, or you are not
  confident enough in either APPROVE or BLOCK. This is the safe choice
  when uncertain.

IMPORTANT: When in doubt, ESCALATE. A false block costs the customer
relationship. A missed fraud costs money. Let a human decide the hard cases.

Output your analysis in this EXACT JSON format (no markdown, no code fences):
{
  "recommendation": "APPROVE" or "BLOCK" or "ESCALATE_TO_HUMAN",
  "confidence": 0.0 to 1.0,
  "reasoning": "2-3 sentence explanation combining key evidence from both analyses",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "risk_score": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL"
}"""


def build_summary_prompt(state: InvestigationState) -> str:
    txn = state["transaction_data"]
    return f"""TRANSACTION CONTEXT:
- ID: {txn['transaction_id']}
- Amount: ${txn['amount']:,.2f} ({txn['amount_deviation']:.1f}x customer average)
- ML fraud score: {txn['fraud_probability']:.2%}
- Risk level from ML: {txn['risk_level']}

TRANSACTION ANALYST FINDINGS:
{state.get('transaction_analysis', 'No analysis available')}

BEHAVIORAL ANALYST FINDINGS:
{state.get('behavioral_analysis', 'No analysis available')}

Based on ALL the above evidence, provide your final recommendation."""


async def risk_summarizer(state: InvestigationState, llm) -> dict:
    """Combine all evidence and produce final recommendation."""
    prompt = build_summary_prompt(state)

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    content = response.content.strip()

    # Parse the JSON response
    import json
    try:
        # Handle potential markdown fences
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        recommendation = result.get("recommendation", "ESCALATE_TO_HUMAN")
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "Unable to parse agent response")

        # Validate recommendation
        if recommendation not in ("APPROVE", "BLOCK", "ESCALATE_TO_HUMAN"):
            recommendation = "ESCALATE_TO_HUMAN"
            reasoning = f"Invalid recommendation from agent. Original: {content[:200]}"

    except (json.JSONDecodeError, KeyError, ValueError):
        recommendation = "ESCALATE_TO_HUMAN"
        confidence = 0.3
        reasoning = f"Could not parse agent recommendation. Raw response will be reviewed by human. Summary: {content[:300]}"

    needs_human = recommendation == "ESCALATE_TO_HUMAN"

    return {
        "risk_summary": content,
        "recommendation": recommendation,
        "confidence": confidence,
        "reasoning": reasoning,
        "needs_human_review": needs_human,
        "investigation_complete": True,
    }
