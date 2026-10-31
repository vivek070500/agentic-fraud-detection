"""
Fraud Investigation Dashboard
================================
Human-in-the-loop review interface for fraud analysts.

Features:
  1. Live transaction feed with ML scores
  2. Pending review queue (gray-zone cases)
  3. Transaction detail view with agent investigation findings
  4. Approve/Reject/Escalate actions with audit trail
  5. System stats and monitoring

Run:
    # Start the API first:
    uvicorn api.main:app --port 8000
    # Then start the dashboard:
    streamlit run dashboard/app.py --server.port 8501
"""

import requests
import streamlit as st
from datetime import datetime

API_URL = "http://127.0.0.1:8000"


# ── Page Config ──
st.set_page_config(
    page_title="Fraud Detection — Review Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_call(method, endpoint, json_data=None):
    """Make API call with error handling."""
    try:
        if method == "GET":
            resp = requests.get(f"{API_URL}{endpoint}", timeout=10)
        else:
            resp = requests.post(f"{API_URL}{endpoint}", json=json_data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure it's running: `uvicorn api.main:app --port 8000`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Sidebar ──
st.sidebar.title("🔍 Fraud Detection")
st.sidebar.markdown("**Agentic AI + Human-in-the-Loop**")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🔎 Review Queue", "📝 Submit Transaction", "📈 System Stats"],
)

# Health check
health = api_call("GET", "/health")
if health:
    status_color = "🟢" if health["status"] == "healthy" else "🔴"
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"{status_color} **API Status:** {health['status']}")
    st.sidebar.markdown(f"📦 **Model loaded:** {health['model_loaded']}")
    st.sidebar.markdown(f"📊 **Scored:** {health['total_scored']} transactions")
    st.sidebar.markdown(f"⏱️ **Uptime:** {health['uptime_seconds']:.0f}s")


# ════════════════════════════════════════
# PAGE: Dashboard
# ════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Fraud Detection Dashboard")

    stats = api_call("GET", "/stats")
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", stats.get("total_transactions", 0))
        with col2:
            by_decision = stats.get("by_decision", {})
            st.metric("Auto-Approved", by_decision.get("AUTO_APPROVE", 0))
        with col3:
            st.metric("Auto-Blocked", by_decision.get("AUTO_BLOCK", 0))
        with col4:
            st.metric("Investigating", by_decision.get("INVESTIGATE", 0))

        st.markdown("---")

        # Review status breakdown
        by_status = stats.get("by_review_status", {})
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            pending = by_status.get("PENDING", 0)
            st.metric("⏳ Pending Review", pending,
                       delta=f"{pending} need attention" if pending > 0 else None,
                       delta_color="inverse")
        with col_b:
            st.metric("✅ Approved", by_status.get("APPROVED", 0))
        with col_c:
            st.metric("❌ Rejected", by_status.get("REJECTED", 0))

    # Recent transactions
    st.markdown("### Recent Transactions")
    recent = api_call("GET", "/recent?limit=20")
    if recent and recent.get("transactions"):
        for txn in recent["transactions"]:
            prob = txn.get("fraud_probability", 0)
            if prob > 0.8:
                color = "🔴"
            elif prob > 0.5:
                color = "🟡"
            elif prob > 0.2:
                color = "🟠"
            else:
                color = "🟢"

            decision = txn.get("policy_decision", "?")
            review = txn.get("review_status", "?")

            with st.expander(
                f"{color} **{txn['transaction_id']}** — ${txn.get('amount', 0):,.2f} | "
                f"Score: {prob:.0%} | {decision} | Review: {review}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Amount:** ${txn.get('amount', 0):,.2f}")
                    st.markdown(f"**Merchant:** {txn.get('merchant_category', '?')}")
                    st.markdown(f"**Fraud Probability:** {prob:.4f}")
                    st.markdown(f"**Risk Level:** {txn.get('risk_level', '?')}")
                with col2:
                    st.markdown(f"**Policy Decision:** {decision}")
                    st.markdown(f"**Review Status:** {review}")
                    if txn.get("reviewer_id"):
                        st.markdown(f"**Reviewed by:** {txn['reviewer_id']}")
                    if txn.get("reviewer_notes"):
                        st.markdown(f"**Notes:** {txn['reviewer_notes']}")

                if txn.get("investigation_summary"):
                    st.markdown("---")
                    st.markdown("**🤖 Agent Investigation:**")
                    st.code(txn["investigation_summary"], language=None)
    else:
        st.info("No transactions yet. Submit one using the 'Submit Transaction' page.")


# ════════════════════════════════════════
# PAGE: Review Queue
# ════════════════════════════════════════
elif page == "🔎 Review Queue":
    st.title("🔎 Human Review Queue")
    st.markdown("Transactions flagged for investigation, sorted by risk (highest first).")

    pending = api_call("GET", "/pending?limit=50")

    if pending and pending.get("transactions"):
        st.warning(f"⏳ **{pending['count']} transaction(s) awaiting review**")

        for txn in pending["transactions"]:
            prob = float(txn.get("fraud_probability", 0))
            st.markdown("---")
            st.subheader(f"🔍 {txn['transaction_id']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Amount", f"${float(txn.get('amount', 0)):,.2f}")
            with col2:
                st.metric("Fraud Score", f"{prob:.2%}")
            with col3:
                st.metric("Risk Level", txn.get("risk_level", "?"))

            st.markdown(f"**Merchant:** {txn.get('merchant_category', '?')}")
            st.markdown(f"**Policy Reason:** {txn.get('policy_decision', '?')}")

            # Show investigation results if available
            if txn.get("investigation_summary"):
                with st.expander("🤖 Agent Investigation Findings"):
                    st.code(txn["investigation_summary"], language=None)

            # Review actions
            st.markdown("**Your Decision:**")
            col_a, col_b, col_c = st.columns(3)

            txn_id = txn["transaction_id"]
            notes_key = f"notes_{txn_id}"
            notes = st.text_input("Reviewer notes:", key=notes_key, placeholder="Optional notes...")

            with col_a:
                if st.button(f"✅ Approve", key=f"approve_{txn_id}"):
                    result = api_call("POST", "/review", {
                        "transaction_id": txn_id,
                        "decision": "APPROVED",
                        "reviewer_notes": notes or "Approved by human reviewer",
                        "reviewer_id": "dashboard_analyst",
                    })
                    if result:
                        st.success(f"✅ {txn_id} APPROVED")
                        st.rerun()

            with col_b:
                if st.button(f"❌ Reject", key=f"reject_{txn_id}"):
                    result = api_call("POST", "/review", {
                        "transaction_id": txn_id,
                        "decision": "REJECTED",
                        "reviewer_notes": notes or "Rejected by human reviewer",
                        "reviewer_id": "dashboard_analyst",
                    })
                    if result:
                        st.error(f"❌ {txn_id} REJECTED")
                        st.rerun()

            with col_c:
                if st.button(f"⬆️ Escalate", key=f"escalate_{txn_id}"):
                    result = api_call("POST", "/review", {
                        "transaction_id": txn_id,
                        "decision": "ESCALATED",
                        "reviewer_notes": notes or "Escalated to senior analyst",
                        "reviewer_id": "dashboard_analyst",
                    })
                    if result:
                        st.warning(f"⬆️ {txn_id} ESCALATED")
                        st.rerun()
    else:
        st.success("🎉 No pending reviews! All caught up.")


# ════════════════════════════════════════
# PAGE: Submit Transaction
# ════════════════════════════════════════
elif page == "📝 Submit Transaction":
    st.title("📝 Submit a Transaction")
    st.markdown("Test the pipeline by submitting a transaction for scoring and investigation.")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input("Amount ($)", min_value=0.5, max_value=50000.0, value=250.0)
            hour = st.slider("Hour of day", 0, 23, 22)
            day = st.selectbox("Day of week", [
                (0, "Monday"), (1, "Tuesday"), (2, "Wednesday"),
                (3, "Thursday"), (4, "Friday"), (5, "Saturday"), (6, "Sunday")
            ], format_func=lambda x: x[1], index=4)
            merchant = st.selectbox("Merchant category", [
                "grocery", "restaurant", "gas_station", "online_retail",
                "electronics", "travel", "subscription", "atm"
            ], index=3)

        with col2:
            foreign = st.checkbox("Foreign merchant", value=True)
            account_age = st.number_input("Account age (days)", min_value=1, max_value=5000, value=5)
            avg_amount = st.number_input("Customer 30-day avg ($)", min_value=1.0, max_value=10000.0, value=80.0)
            velocity = st.slider("Transactions in last hour", 0, 20, 2)
            device_changed = st.checkbox("Device changed", value=False)
            distance = st.number_input("Distance from home (km)", min_value=0.0, max_value=20000.0, value=100.0)

        investigate = st.checkbox("🤖 Run agent investigation (if gray zone)", value=True)
        submitted = st.form_submit_button("🚀 Score Transaction", use_container_width=True)

    if submitted:
        txn_data = {
            "amount": amount,
            "hour_of_day": hour,
            "day_of_week": day[0],
            "merchant_category": merchant,
            "is_foreign_merchant": int(foreign),
            "customer_account_age_days": account_age,
            "customer_avg_amount_30d": avg_amount,
            "transaction_velocity_1h": velocity,
            "device_changed": int(device_changed),
            "distance_from_home_km": distance,
        }

        endpoint = "/score-and-investigate" if investigate else "/score"

        with st.spinner("Scoring transaction..." + (" + Running agent investigation..." if investigate else "")):
            result = api_call("POST", endpoint, txn_data)

        if result:
            st.markdown("---")

            # ML Score
            ml = result.get("ml_score", {})
            prob = ml.get("fraud_probability", 0)
            if prob > 0.8:
                st.error(f"🔴 **Fraud Probability: {prob:.2%}** — {ml.get('risk_level', '?')}")
            elif prob > 0.2:
                st.warning(f"🟡 **Fraud Probability: {prob:.2%}** — {ml.get('risk_level', '?')}")
            else:
                st.success(f"🟢 **Fraud Probability: {prob:.2%}** — {ml.get('risk_level', '?')}")

            # Policy
            policy = result.get("policy", {})
            st.info(f"📋 **Policy Decision:** {policy.get('decision', '?')}\n\n{policy.get('reason', '')}")

            # Top features
            features = ml.get("top_features", [])
            if features:
                st.markdown("**Top Risk Factors:**")
                for f in features:
                    bar_len = int(f["importance"] * 50)
                    st.markdown(f"- `{f['feature']}` = {f['value']:.2f} (importance: {'█' * bar_len} {f['importance']:.3f})")

            # Investigation results
            investigation = result.get("investigation")
            if investigation:
                st.markdown("---")
                st.markdown("### 🤖 Agent Investigation Results")

                rec = investigation.get("recommendation", "?")
                conf = investigation.get("confidence", 0)

                if rec == "APPROVE":
                    st.success(f"✅ Agents recommend **APPROVE** (confidence: {conf:.0%})")
                elif rec == "BLOCK":
                    st.error(f"🚫 Agents recommend **BLOCK** (confidence: {conf:.0%})")
                else:
                    st.warning(f"⬆️ Agents recommend **ESCALATE TO HUMAN** (confidence: {conf:.0%})")

                st.markdown(f"**Reasoning:** {investigation.get('reasoning', '?')}")

                if investigation.get("transaction_analysis"):
                    with st.expander("📋 Transaction Analysis"):
                        st.markdown(investigation["transaction_analysis"])

                if investigation.get("behavioral_analysis"):
                    with st.expander("🔍 Behavioral Analysis"):
                        st.markdown(investigation["behavioral_analysis"])

                if investigation.get("error"):
                    st.error(f"⚠️ Investigation error: {investigation['error']}")

            st.markdown("---")
            st.markdown(f"**Transaction ID:** `{result.get('transaction_id', '?')}`")
            st.markdown(f"**Review Status:** {result.get('review_status', '?')}")


# ════════════════════════════════════════
# PAGE: System Stats
# ════════════════════════════════════════
elif page == "📈 System Stats":
    st.title("📈 System Statistics")

    stats = api_call("GET", "/stats")
    if stats:
        st.markdown("### Overall")
        st.json(stats)

        st.markdown("### Session Metrics")
        session = stats.get("session", {})
        decisions = session.get("decisions_this_session", {})

        total = sum(decisions.values()) or 1
        st.markdown(f"- **Auto-Approved:** {decisions.get('AUTO_APPROVE', 0)} ({decisions.get('AUTO_APPROVE', 0)/total:.0%})")
        st.markdown(f"- **Auto-Blocked:** {decisions.get('AUTO_BLOCK', 0)} ({decisions.get('AUTO_BLOCK', 0)/total:.0%})")
        st.markdown(f"- **Investigated:** {decisions.get('INVESTIGATE', 0)} ({decisions.get('INVESTIGATE', 0)/total:.0%})")

        st.markdown("### Architecture")
        st.markdown("""
        ```
        Transaction → ML Scoring (XGBoost) → Policy Engine → Decision
                                                    │
                                              ┌─────┴─────┐
                                              │           │
                                         AUTO_APPROVE  INVESTIGATE
                                         AUTO_BLOCK        │
                                                           ▼
                                                  LangGraph Agents
                                                  (GPT-4o-mini)
                                                       │
                                                  ┌────┴────┐
                                                  │         │
                                              CONFIDENT  UNCERTAIN
                                              (auto)    (→ human)
                                                           │
                                                           ▼
                                                  Human Review Dashboard
                                                  (this page!)
        ```
        """)
