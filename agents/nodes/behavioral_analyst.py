"""
Behavioral Analyst Agent
==========================
Analyzes behavioral signals: device changes, geographic anomalies,
velocity patterns, account age.

This agent answers: "Does this user's BEHAVIOR look fraudulent?"
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import InvestigationState

SYSTEM_PROMPT = """You are a Behavioral Analyst at a fraud detection unit.
Your job is to analyze user behavior patterns for signs of account compromise or fraud.

Focus on:
1. DEVICE ANALYSIS: Has the device changed? A new device is a risk signal,
   especially combined with other factors.
2. GEOGRAPHIC ANALYSIS: How far is this transaction from the user's home?
   >3000km is very suspicious. >500km is notable.
3. VELOCITY ANALYSIS: How many transactions in the last hour?
   Normal is 1-2. 5+ in one hour suggests automated fraud or card testing.
4. ACCOUNT AGE: New accounts (<30 days) are much higher risk.
   Established accounts (>365 days) have earned trust.

Output your analysis in this exact format:
DEVICE RISK: [LOW/MEDIUM/HIGH] - [one sentence explanation]
GEOGRAPHIC RISK: [LOW/MEDIUM/HIGH] - [one sentence explanation]
VELOCITY RISK: [LOW/MEDIUM/HIGH] - [one sentence explanation]
ACCOUNT RISK: [LOW/MEDIUM/HIGH] - [one sentence explanation]
OVERALL BEHAVIORAL RISK: [LOW/MEDIUM/HIGH]
KEY CONCERN: [one sentence summary of the biggest behavioral red flag, or "None identified"]

Be concise. Do not speculate beyond the data given."""


def build_behavioral_prompt(txn: dict) -> str:
    return f"""Analyze user behavior for this transaction:

Transaction ID: {txn['transaction_id']}
Device changed: {'Yes — using a different device than usual' if txn['device_changed'] else 'No — same device as usual'}
Distance from home: {txn['distance_from_home_km']:,.1f} km
Transactions in last hour: {txn['transaction_velocity_1h']}
Customer account age: {txn['customer_account_age_days']} days
Foreign merchant: {'Yes' if txn['is_foreign_merchant'] else 'No'}
ML fraud score: {txn['fraud_probability']:.2%}"""


async def behavioral_analyst(state: InvestigationState, llm) -> dict:
    """Analyze the user's behavioral patterns."""
    txn = state["transaction_data"]
    prompt = build_behavioral_prompt(txn)

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {"behavioral_analysis": response.content}
