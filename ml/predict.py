"""
Prediction Service
===================
Loads trained model artifacts and scores incoming transactions.
Designed to be imported by the FastAPI app.

Usage:
    from ml.predict import FraudScorer
    scorer = FraudScorer()
    result = scorer.score(transaction_dict)
"""

import json
import joblib
import numpy as np
from pathlib import Path


class FraudScorer:
    """Loads model artifacts once, scores transactions with sub-5ms latency."""

    FEATURE_ORDER = [
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

    def __init__(self, model_dir: Path = None):
        if model_dir is None:
            model_dir = Path(__file__).parent / "models"

        self.model = joblib.load(model_dir / "fraud_xgboost.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.label_encoder = joblib.load(model_dir / "label_encoder.pkl")

        with open(model_dir / "model_metadata.json") as f:
            self.metadata = json.load(f)

        self.optimal_threshold = self.metadata["optimal_threshold"]
        self._merchant_classes = list(self.label_encoder.classes_)

    def _encode_merchant(self, category: str) -> int:
        """Encode merchant category, defaulting to 'online_retail' for unknowns."""
        if category in self._merchant_classes:
            return self.label_encoder.transform([category])[0]
        return self.label_encoder.transform(["online_retail"])[0]

    def _derive_features(self, txn: dict) -> dict:
        """Compute derived features from raw transaction data."""
        hour = txn.get("hour_of_day", 12)
        day = txn.get("day_of_week", 3)
        amount = txn.get("amount", 0)
        avg_30d = txn.get("customer_avg_amount_30d", amount)

        return {
            "is_weekend": int(day >= 5),
            "is_night": int(hour >= 22 or hour <= 5),
            "amount_deviation": round(amount / max(avg_30d, 0.01), 2),
        }

    def score(self, transaction: dict) -> dict:
        """
        Score a single transaction.

        Args:
            transaction: dict with keys like amount, hour_of_day, merchant_category, etc.

        Returns:
            dict with fraud_probability, is_fraud (at optimal threshold),
            risk_level, and top contributing features.
        """
        derived = self._derive_features(transaction)
        txn = {**transaction, **derived}

        merchant_encoded = self._encode_merchant(txn.get("merchant_category", "online_retail"))
        txn["merchant_category_encoded"] = merchant_encoded

        features = np.array([[txn.get(f, 0) for f in self.FEATURE_ORDER]])
        features_scaled = self.scaler.transform(features)

        fraud_prob = float(self.model.predict_proba(features_scaled)[0][1])
        is_fraud = fraud_prob >= self.optimal_threshold

        # Risk level for policy engine
        if fraud_prob < 0.2:
            risk_level = "LOW"
        elif fraud_prob < 0.5:
            risk_level = "MEDIUM"
        elif fraud_prob < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        # Top contributing features (via feature importance * feature deviation)
        feature_values = {self.FEATURE_ORDER[i]: float(features[0][i]) for i in range(len(self.FEATURE_ORDER))}
        importance = self.metadata.get("feature_importance", {})
        contributions = []
        for feat, val in feature_values.items():
            imp = importance.get(feat, 0)
            contributions.append({"feature": feat, "value": val, "importance": imp})
        contributions.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "fraud_probability": round(fraud_prob, 4),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "optimal_threshold": self.optimal_threshold,
            "top_features": contributions[:5],
        }
