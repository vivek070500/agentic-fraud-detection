"""
Centralized configuration — loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH", "ml/models/fraud_xgboost.pkl")
SCALER_PATH = PROJECT_ROOT / "ml/models/scaler.pkl"
DATA_DIR = PROJECT_ROOT / "data"

# --- ML Thresholds ---
FRAUD_THRESHOLD_AUTO_APPROVE = float(os.getenv("FRAUD_THRESHOLD_AUTO_APPROVE", "0.2"))
FRAUD_THRESHOLD_AUTO_BLOCK = float(os.getenv("FRAUD_THRESHOLD_AUTO_BLOCK", "0.8"))

# --- LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# --- AWS ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "fraud_audit_trail")

# --- Langfuse ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Dataset Generation ---
DATASET_SIZE = 100_000
FRAUD_RATE = 0.03
RANDOM_SEED = 42
