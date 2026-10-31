# 🔍 Agentic Fraud Detection System

**Built by [Vaibhav Tarachandani](https://github.com/vivek070500)**

> Imagine you're a bank. Thousands of credit card transactions happen every second. Some are real people buying groceries. Some are thieves stealing money. How do you tell the difference — instantly, accurately, and without blocking real customers?
>
> That's what this project does.

---

## 🧠 What Does This Project Do?

It catches fraudulent transactions using **three layers of intelligence**, each smarter than the last:

### Layer 1: Machine Learning Model (The Fast Filter)
Think of this as a super-fast calculator. It looks at a transaction and says:
- *"This $25 grocery purchase at 2 PM from a regular customer? 0.01% chance of fraud. Let it through."*
- *"This $3,000 electronics purchase at 3 AM from a new account in another country? 99.9% fraud. Block it."*

It scores **every single transaction in under 1 millisecond**.

### Layer 2: Business Rules (The Policy Guard)
Some transactions need extra rules on top of the ML score:
- Brand new account (less than 7 days old) + foreign merchant? → **Investigate**, even if ML says it's fine.
- Amount over $5,000 with any risk at all? → **Investigate**.

These are rules that a compliance team would set — they don't change often.

### Layer 3: AI Agents (The Investigators)
This is where it gets interesting. For the **ambiguous cases** (maybe 5-10% of transactions), the system sends the case to three AI agents:

| Agent | What It Does | Like... |
|-------|-------------|---------|
| **Transaction Analyst** | Looks at the money side — amount, merchant, timing | A financial analyst reading the receipt |
| **Behavioral Analyst** | Looks at the person's behavior — new device? far from home? too many purchases? | A detective checking if the behavior matches the person |
| **Risk Summarizer** | Reads both reports and makes a final call: Approve, Block, or Escalate to Human | A senior manager making the decision |

If the AI agents are confident → they auto-decide.
If they're not sure → it goes to a **human reviewer** on a dashboard.

### Layer 4: Human Review (The Final Say)
A real person sees the flagged transaction on a dashboard, reads the AI's investigation, and clicks Approve or Reject. Every click is recorded for audit.

---

## 🏗️ How It All Fits Together

```
Customer swipes card
        │
        ▼
   ML Model scores it (< 1ms)
        │
        ├── Score < 0.2  →  ✅ AUTO-APPROVE (done!)
        ├── Score > 0.8  →  🚫 AUTO-BLOCK (done!)
        │
        └── Score 0.2–0.8 (gray zone)
                │
                ▼
        AI Agents investigate (~5 seconds)
         ├── Transaction Analyst
         ├── Behavioral Analyst
         └── Risk Summarizer
                │
                ├── Confident  →  Auto-decide
                └── Uncertain  →  Send to human dashboard
                                        │
                                        ▼
                                Human clicks Approve/Reject
                                        │
                                        ▼
                                 Audit trail saved ✅
```

---

## 📊 How Well Does It Work?

We tested three approaches (you can run these experiments yourself):

| Approach | Catches Fraud (Recall) | Doesn't Block Real People (Precision) | Smart Enough? |
|----------|----------------------|---------------------------------------|---------------|
| **Rules only** | 91% | 84% | ❌ Misses sneaky fraud |
| **ML model** | 95% | 94% | ⚠️ Good, but uncertain on ~5% of cases |
| **ML + AI Agents + Human** | ~99% | ~97% | ✅ Best of all worlds |

The ML model alone is great. But for the cases it's not sure about, the AI agents add a second opinion, and humans add the final safety net.

---

## 🚀 How To Run This Project

### What You Need
- **Python 3.9+** (you probably already have this)
- **A Groq API key** (free, no credit card needed — [get one here](https://console.groq.com/keys))

### Step 1: Clone the project

```bash
git clone https://github.com/vivek070500/agentic-fraud-detection.git
cd agentic-fraud-detection
```

### Step 2: Install dependencies

```bash
# Create a virtual environment (or use an existing one)
python3 -m venv venv
source venv/bin/activate

# Install everything
pip install -r requirements.txt
```

### Step 3: Set up your API key

```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your Groq API key:
```
GROQ_API_KEY=gsk_your_key_here
```

### Step 4: Generate data and train the model

```bash
# Generate 100,000 synthetic transactions
python -m data.generate_dataset

# Train the fraud detection model
python -m ml.train_model
```

You'll see the model achieve **0.997 ROC-AUC** and **95% F1 score**.

### Step 5: Run the experiments (optional but cool)

See how each approach compares:
```bash
# Experiment 1: How well do simple rules work?
python -m experiments.01_baseline_rules

# Experiment 2: How much better is ML?
python -m experiments.02_ml_model_improvement
```

### Step 6: Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to see the interactive API documentation.

### Step 7: Start the Dashboard

Open a **second terminal** and run:
```bash
streamlit run dashboard/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) to see the human review dashboard.

### Step 8: Try it out!

Submit a suspicious transaction via the dashboard or via curl:
```bash
curl -X POST http://localhost:8000/score-and-investigate \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 780.00,
    "hour_of_day": 23,
    "day_of_week": 5,
    "merchant_category": "electronics",
    "is_foreign_merchant": 1,
    "customer_account_age_days": 5,
    "customer_avg_amount_30d": 120.00,
    "transaction_velocity_1h": 4,
    "device_changed": 1,
    "distance_from_home_km": 5200.0
  }'
```

You'll see the ML model score it, the policy engine route it, and the AI agents investigate it — all in ~7 seconds.

---

## 📁 Project Structure

```
agentic-fraud-detection/
│
├── data/                          ← Synthetic transaction dataset
│   └── generate_dataset.py        ← Creates 100K transactions with fraud patterns
│
├── ml/                            ← Machine Learning
│   ├── train_model.py             ← Trains XGBoost, evaluates, exports model
│   ├── predict.py                 ← FraudScorer class used by the API
│   └── models/                    ← Trained model artifacts (.pkl)
│
├── experiments/                   ← The story of how we got here
│   ├── 01_baseline_rules.py       ← Start with rules, see where they fail
│   ├── 02_ml_model_improvement.py ← ML beats rules, but has a gray zone
│   └── 03_agentic_investigation.py← Why we need AI agents for the gray zone
│
├── api/                           ← FastAPI backend
│   ├── main.py                    ← All API endpoints
│   ├── policy_engine.py           ← Business rules for routing decisions
│   ├── audit.py                   ← SQLite/DynamoDB audit trail
│   └── schemas.py                 ← Request/response data models
│
├── agents/                        ← LangGraph multi-agent system
│   ├── graph.py                   ← The investigation pipeline
│   ├── state.py                   ← Shared state between agents
│   └── nodes/
│       ├── transaction_analyst.py ← Analyzes the financial side
│       ├── behavioral_analyst.py  ← Analyzes user behavior
│       └── risk_summarizer.py     ← Combines evidence, makes decision
│
├── dashboard/                     ← Streamlit human review UI
│   └── app.py                     ← Review queue, submit transactions, stats
│
├── observability/                 ← Langfuse tracing integration
│   └── langfuse_config.py         ← Traces every agent step
│
├── deploy/                        ← Deployment configs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── start.sh
│   └── aws/setup_ec2.sh           ← One-command EC2 deployment
│
├── .env.example                   ← Template for API keys
├── requirements.txt               ← Python dependencies
└── README.md                      ← You are here!
```

---

## 🔧 Tech Stack

| What | Technology | Why |
|------|-----------|-----|
| ML Model | XGBoost | Fast, accurate, small model size (601 KB) |
| API | FastAPI | Modern Python API framework, auto-generates docs |
| Agent Orchestration | LangGraph | State machine for multi-agent workflows with built-in HITL |
| LLM | Groq (Llama 3) / GPT-4o-mini | Groq is free and fast; OpenAI as paid alternative |
| Human Dashboard | Streamlit | Quick to build, looks good, interactive |
| Audit Trail | SQLite (local) / DynamoDB (AWS) | Queryable, mutable, transactional storage |
| Observability | Langfuse | Visual traces of every agent step and LLM call |
| Deployment | Docker + EC2 (AWS Free Tier) | One-command deployment, runs on a free t2.micro |

---

## 💡 Design Decisions Explained

**Why not just use rules?**
Rules miss sneaky fraud. They check each feature independently but can't learn that "slightly high amount + night + new device" together means fraud. ML can.

**Why not just use ML?**
ML gives a probability, not a reason. For the 5-10% of cases where it's uncertain, we need something that can reason about WHY a transaction looks suspicious. That's where agents come in.

**Why not just use AI agents for everything?**
Cost and speed. LLM calls take ~2 seconds and cost money. ML scores in <1ms for free. We only use agents on the cases that actually need them.

**Why DynamoDB over S3 for the audit trail?**
You need to query by transaction ID, filter by status, and update records (PENDING → APPROVED). That's a database job. S3 is for storing files, not querying data.

**Why SQLite locally?**
Zero setup. No AWS account needed to develop and test. The code auto-switches to DynamoDB when AWS credentials are configured.

---

## ☁️ Deploying to AWS (Free Tier)

The whole system runs on a single **EC2 t2.micro** instance (free for 12 months):

1. Launch an EC2 instance (Amazon Linux 2023, t2.micro)
2. Open ports 22, 8000, 8501 in the security group
3. SSH in and run:
```bash
git clone https://github.com/vivek070500/agentic-fraud-detection.git
cd agentic-fraud-detection
nano .env  # Add your GROQ_API_KEY
cd deploy && bash ../deploy/aws/setup_ec2.sh
```

That's it. Your API will be at `http://<your-ip>:8000` and dashboard at `http://<your-ip>:8501`.

---

## 📈 What I'd Add Next

- **SHAP explanations** in the dashboard (e.g., "amount was 6.5x average — that's why it was flagged")
- **Slack/email notifications** when a transaction needs human review
- **Model retraining pipeline** that detects data drift and auto-retrains
- **A/B testing** — shadow mode for new models before production deployment
- **Graph neural networks** for detecting fraud rings (multiple accounts working together)

---

## 📜 License

MIT — use it however you like.
