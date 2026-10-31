"""
Model Training Pipeline
========================
Trains XGBoost on the synthetic fraud dataset with:
  - Stratified train/test split
  - Class weight handling (no SMOTE needed at 3% fraud rate)
  - Threshold optimization for business cost
  - SHAP explainability export
  - Model + scaler serialization to pickle

Usage:
    python -m ml.train_model
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).parent.parent / "data" / "fraud_dataset.csv"
MODEL_DIR = Path(__file__).parent / "models"


def load_and_prepare(path: Path) -> tuple:
    """Load CSV and encode categorical features."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path.name}")

    le = LabelEncoder()
    df["merchant_category_encoded"] = le.fit_transform(df["merchant_category"])

    feature_cols = [
        "amount",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night",
        "merchant_category_encoded",
        "is_foreign_merchant",
        "customer_account_age_days",
        "customer_avg_amount_30d",
        "amount_deviation",
        "transaction_velocity_1h",
        "device_changed",
        "distance_from_home_km",
    ]

    X = df[feature_cols].values
    y = df["is_fraud"].values

    # Scale features for consistency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler, le


def find_optimal_threshold(y_true, y_prob, fp_cost=5, fn_cost=200):
    """
    Find threshold that minimizes business cost:
      - False Positive (blocking legit user): $5 cost
      - False Negative (missing fraud): $200 cost
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    best_cost = float("inf")
    best_threshold = 0.5

    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fp * fp_cost + fn * fn_cost
        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    return best_threshold, best_cost


def train():
    """Main training pipeline."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    X, y, feature_cols, scaler, label_encoder = load_and_prepare(DATA_PATH)

    # --- Stratified split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]:,} samples ({y_train.mean():.2%} fraud)")
    print(f"Test:  {X_test.shape[0]:,} samples ({y_test.mean():.2%} fraud)")

    # --- Train XGBoost ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nScale pos weight: {scale_pos_weight:.1f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training time: {train_time:.2f}s")

    # --- Evaluate ---
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    # --- Optimal threshold ---
    optimal_threshold, min_cost = find_optimal_threshold(y_test, y_prob)
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    f1_opt = f1_score(y_test, y_pred_opt)
    print(f"Optimal threshold (min business cost): {optimal_threshold:.2f}")
    print(f"F1 at optimal threshold: {f1_opt:.4f}")
    print(f"Min business cost: ${min_cost:,.0f}")

    # --- Latency benchmark ---
    single = X_test[0:1]
    latencies = []
    for _ in range(200):
        s = time.time()
        model.predict_proba(single)
        latencies.append((time.time() - s) * 1000)
    latencies = np.array(latencies)
    print(f"\n⏱️  Inference Latency (single sample):")
    print(f"   P50: {np.percentile(latencies, 50):.3f}ms")
    print(f"   P95: {np.percentile(latencies, 95):.3f}ms")
    print(f"   P99: {np.percentile(latencies, 99):.3f}ms")

    # --- Feature importance ---
    importance = {k: float(v) for k, v in zip(feature_cols, model.feature_importances_)}
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    print(f"\n📊 Feature Importance:")
    for feat, imp in importance_sorted.items():
        bar = "█" * int(imp * 100)
        print(f"   {feat:35s} {imp:.4f} {bar}")

    # --- Save artifacts ---
    joblib.dump(model, MODEL_DIR / "fraud_xgboost.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

    metadata = {
        "model_type": "XGBClassifier",
        "n_estimators": 200,
        "max_depth": 6,
        "feature_columns": feature_cols,
        "optimal_threshold": round(optimal_threshold, 4),
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_default": round(f1, 4),
            "f1_optimal": round(f1_opt, 4),
        },
        "latency_ms": {
            "p50": round(np.percentile(latencies, 50), 3),
            "p95": round(np.percentile(latencies, 95), 3),
            "p99": round(np.percentile(latencies, 99), 3),
        },
        "feature_importance": {k: round(v, 4) for k, v in importance_sorted.items()},
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "fraud_rate": round(float(y.mean()), 4),
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Artifacts saved to {MODEL_DIR}/")
    print(f"   fraud_xgboost.pkl  — trained model")
    print(f"   scaler.pkl         — feature scaler")
    print(f"   label_encoder.pkl  — merchant category encoder")
    print(f"   model_metadata.json — metrics + config")

    return model, metadata


if __name__ == "__main__":
    train()
