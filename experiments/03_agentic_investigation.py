"""
Experiment 3: Agentic Investigation for the Gray Zone
=======================================================
From Experiment 2, we know ML leaves a "gray zone" (score 0.2–0.8)
where it's not confident. These ambiguous cases need deeper analysis.

Instead of routing ALL gray zone cases to human reviewers (expensive,
slow, doesn't scale), we use an agentic AI investigation pipeline:

  1. Multiple specialized agents analyze the transaction from different angles
  2. A supervisor combines their findings into a structured recommendation
  3. Only cases that STILL need judgment go to human review

This is the experiment that validates the architecture.
Actual implementation will be in agents/ (Milestone 3).

Hypothesis: "Agentic investigation can confidently resolve 60-70% of
            gray zone cases, reducing human review volume by 2-3x."

Run:
    python -m experiments.03_agentic_investigation
    (Requires OPENAI_API_KEY in .env — implemented in Milestone 3)
"""

print("""
╔══════════════════════════════════════════════════════════════╗
║  EXPERIMENT 3: Agentic Investigation Design                  ║
║                                                              ║
║  This experiment defines the ARCHITECTURE we'll build        ║
║  in Milestone 3. Run it after Milestone 3 is complete.       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  From Experiments 1 & 2, we established:                     ║
║                                                              ║
║  1. Rules catch obvious fraud but miss combinations          ║
║     (Exp 1: ROC-AUC ~0.93)                                  ║
║                                                              ║
║  2. ML catches combinations but has a gray zone              ║
║     (Exp 2: ROC-AUC ~0.997, but gray zone = ~10% of txns)   ║
║                                                              ║
║  3. Gray zone has mixed signals — needs CONTEXTUAL analysis  ║
║     that neither rules nor ML scores can provide.            ║
║                                                              ║
║  ─────────────────────────────────────────────────────────── ║
║                                                              ║
║  PROPOSED ARCHITECTURE (Multi-Agent Investigation):          ║
║                                                              ║
║  Transaction (gray zone)                                     ║
║       │                                                      ║
║       ▼                                                      ║
║  ┌─────────────────────────────────────────┐                 ║
║  │  Supervisor / Orchestrator              │                 ║
║  │  Decides which agents to invoke         │                 ║
║  └─────┬──────────┬──────────┬─────────────┘                 ║
║        │          │          │                               ║
║        ▼          ▼          ▼                               ║
║   Transaction  Behavior   Risk                               ║
║   Analyst      Analyst    Summarizer                         ║
║   (amount,     (device,   (combines all                      ║
║    patterns,   location,   evidence into                     ║
║    merchant)   velocity)   recommendation)                   ║
║        │          │          │                               ║
║        └──────────┴──────────┘                               ║
║                   │                                          ║
║                   ▼                                          ║
║          ┌────────────────┐                                  ║
║          │ Decision:      │                                  ║
║          │ APPROVE /      │                                  ║
║          │ BLOCK /        │                                  ║
║          │ ESCALATE to    │                                  ║
║          │ human review   │                                  ║
║          └────────────────┘                                  ║
║                                                              ║
║  KEY DESIGN DECISIONS:                                       ║
║                                                              ║
║  • LangGraph for orchestration (state machine + HITL)        ║
║  • GPT-4o-mini for LLM calls (cheap, fast, good enough)     ║
║  • Langfuse for tracing every agent step                     ║
║  • DynamoDB for persisting investigation + audit trail       ║
║  • Streamlit for human review dashboard                      ║
║                                                              ║
║  EXPECTED OUTCOMES:                                          ║
║  • Agents resolve ~60-70% of gray zone cases confidently     ║
║  • Human reviewers only see the truly ambiguous ~30-40%      ║
║  • Full traceability: every agent thought is logged          ║
║  • Every human decision creates an audit trail               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

To run the actual agentic pipeline, complete Milestone 3 and then:
    python -m agents.graph --transaction-id txn_001234
""")
