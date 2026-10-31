"""
Transaction Analyst Agent
===========================
Analyzes the transaction itself: amount patterns, merchant risk,
timing anomalies.

This agent answers: "Is this transaction financially suspicious?"
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import InvestigationState

SYSTEM_PROMPT = """You are a Transaction Analyst at a fraud detection unit.
Your job is to analyze a single transaction and assess financial risk indicators.

Focus on:
1. AMOUNT ANALYSIS: Is the amount unusual for this customer? Compare to their 30-day average.
2. MERCHANT RISK: Is this merchant category commonly associated with fraud?
   High-risk: electronics, travel, online_retail. Low-risk: grocery, gas_station, subscription.
3. TIMING: Is the transaction at an unusual time? Night transactions (10 PM - 5 AM) are riskier.
4. VALUE PATTERN: Is this a round number? Fraudsters often use round amounts.

Output your analysis in this exact format:
AMOUNT RISK: [LOW/MEDIUM/HIGH] - [one sentence explanation]
MERCHANT RISK: [LOW/MEDIUM/HIGH] - [one sentence explanation]
TIMING RISK: [LOW/MEDIUM/HIGH] - [one sentence explanation]
OVERALL FINANCIAL RISK: [LOW/MEDIUM/HIGH]
KEY CONCERN: [one sentence summary of the biggest red flag, or "None identified"]

Be concise. Do not speculate beyond the data given."""


def build_transaction_prompt(txn: dict) -> str:
    return f"""Analyze this transaction:

Transaction ID: {txn['transaction_id']}
Amount: ${txn['amount']:,.2f}
Customer's 30-day average: ${txn['customer_avg_amount_30d']:,.2f}
Amount deviation: {txn['amount_deviation']:.1f}x their average
Merchant category: {txn['merchant_category']}
Foreign merchant: {'Yes' if txn['is_foreign_merchant'] else 'No'}
Hour of day: {txn['hour_of_day']}:00
Day of week: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][txn['day_of_week']]}
ML fraud score: {txn['fraud_probability']:.2%}"""


async def transaction_analyst(state: InvestigationState, llm) -> dict:
    """Analyze the transaction's financial characteristics."""
    txn = state["transaction_data"]
    prompt = build_transaction_prompt(txn)

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {"transaction_analysis": response.content}
