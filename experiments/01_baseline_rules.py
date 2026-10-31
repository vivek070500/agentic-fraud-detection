"""
Experiment 1: Rule-Based Baseline
===================================
Start with what any team would build first — simple deterministic rules.
Evaluate where it works and WHERE IT BREAKS.

Hypothesis: "Can we catch fraud with basic business rules?"

Run:
    python -m experiments.01_baseline_rules
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from pathlib import Path
from experiments.utils import rule_based_scorer

DATA_PATH = Path(__file__).parent.parent / "data" / "fraud_dataset.csv"


def evaluate_at_threshold(y_true, y_scores, threshold, label=""):
    """Evaluate binary predictions at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Business cost: FP = $5 (blocked legit user), FN = $200 (missed fraud)
    business_cost = fp * 5 + fn * 200

    print(f"\n{'─'*60}")
    print(f"  {label} (threshold = {threshold})")
    print(f"{'─'*60}")
    print(f"  TP: {tp:,}  (caught fraud)")
    print(f"  FP: {fp:,}  (wrongly blocked legit users)")
    print(f"  FN: {fn:,}  (MISSED fraud ⚠️)")
    print(f"  TN: {tn:,}  (correctly approved)")
    print(f"")
    print(f"  Precision:     {precision:.2%}")
    print(f"  Recall:        {recall:.2%}")
    print(f"  F1:            {f1:.4f}")
    print(f"  Business cost: ${business_cost:,.0f}")

    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "fp": fp, "fn": fn, "business_cost": business_cost}


def main():
    print("=" * 60)
    print("  EXPERIMENT 1: Rule-Based Baseline")
    print("  Can simple rules catch fraud?")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset: {len(df):,} transactions, {df['is_fraud'].sum():,} fraud ({df['is_fraud'].mean():.2%})")

    # Score every transaction
    print("\nScoring all transactions with rules...")
    df["rule_score"] = df.apply(rule_based_scorer, axis=1)

    y_true = df["is_fraud"].values
    y_scores = df["rule_score"].values

    # ── Overall metrics ──
    auc = roc_auc_score(y_true, y_scores)
    print(f"\nROC-AUC: {auc:.4f}")

    # ── Try different thresholds ──
    results = []
    for t in [0.3, 0.4, 0.5, 0.6]:
        r = evaluate_at_threshold(y_true, y_scores, t, f"Rules @ {t}")
        results.append(r)

    # ── Find the best threshold ──
    best = min(results, key=lambda x: x["business_cost"])
    print(f"\n{'='*60}")
    print(f"  BEST THRESHOLD: {best['threshold']}")
    print(f"  Business cost: ${best['business_cost']:,.0f}")
    print(f"  F1: {best['f1']:.4f}")
    print(f"{'='*60}")

    # ── WHERE THE BASELINE BREAKS ──
    print(f"\n{'='*60}")
    print(f"  🔍 WHERE RULES FAIL — Gap Analysis")
    print(f"{'='*60}")

    best_threshold = best["threshold"]
    df["rule_pred"] = (df["rule_score"] >= best_threshold).astype(int)

    # False Negatives: fraud we MISSED
    fn_mask = (df["is_fraud"] == 1) & (df["rule_pred"] == 0)
    fn_df = df[fn_mask]

    print(f"\n⚠️  MISSED FRAUD (False Negatives): {len(fn_df):,} / {df['is_fraud'].sum():,}")
    print(f"   These are the cases rules cannot handle:")
    print(f"")
    print(f"   Avg amount deviation: {fn_df['amount_deviation'].mean():.2f}x")
    print(f"     (vs caught fraud:   {df[(df.is_fraud==1) & (df.rule_pred==1)]['amount_deviation'].mean():.2f}x)")
    print(f"   Night txns:           {fn_df['is_night'].mean():.1%}")
    print(f"   Device changed:       {fn_df['device_changed'].mean():.1%}")
    print(f"   Foreign merchant:     {fn_df['is_foreign_merchant'].mean():.1%}")
    print(f"   Avg velocity:         {fn_df['transaction_velocity_1h'].mean():.1f}")
    print(f"   Avg distance:         {fn_df['distance_from_home_km'].mean():.0f}km")

    print(f"\n   💡 INSIGHT: Missed fraud has LOWER individual signals.")
    print(f"      Rules check each feature independently — they miss")
    print(f"      COMBINATIONS of weak signals that together indicate fraud.")
    print(f"      Example: amount 2.5x normal + night + velocity 3 = each below")
    print(f"      threshold alone, but together suspicious.")
    print(f"")
    print(f"   👉 This is exactly where ML excels — learning non-linear")
    print(f"      feature interactions. Let's try that in Experiment 2.")

    # False Positives: legit users we BLOCKED
    fp_mask = (df["is_fraud"] == 0) & (df["rule_pred"] == 1)
    fp_df = df[fp_mask]

    print(f"\n⚠️  WRONGLY BLOCKED (False Positives): {fp_df.shape[0]:,} / {(df.is_fraud==0).sum():,}")
    print(f"   These are legit users having a bad experience:")
    print(f"")
    print(f"   Top merchant categories blocked:")
    if not fp_df.empty:
        for cat, count in fp_df["merchant_category"].value_counts().head(5).items():
            print(f"      {cat}: {count:,}")
    print(f"")
    print(f"   💡 INSIGHT: Rules have no nuance — a night purchase of electronics")
    print(f"      from a foreign merchant triggers multiple rules even if the")
    print(f"      customer always shops like this. Rules don't learn per-user patterns.")

    # ── The Gray Zone ──
    gray_mask = (df["rule_score"] >= 0.2) & (df["rule_score"] <= 0.6)
    gray_df = df[gray_mask]
    print(f"\n{'─'*60}")
    print(f"  📊 THE GRAY ZONE (score 0.2 – 0.6): {len(gray_df):,} transactions")
    print(f"{'─'*60}")
    print(f"   Actual fraud rate in gray zone: {gray_df['is_fraud'].mean():.2%}")
    print(f"   This is {gray_df['is_fraud'].mean() / df['is_fraud'].mean():.1f}x the overall rate")
    print(f"")
    print(f"   These ambiguous cases are where we need SMARTER analysis —")
    print(f"   either ML (Experiment 2) or agentic investigation (Experiment 3).")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1 SUMMARY")
    print(f"{'='*60}")
    print(f"")
    print(f"  ✅ What rules do well:")
    print(f"     - Fast (no model to load, <0.1ms per transaction)")
    print(f"     - Transparent (every decision is explainable)")
    print(f"     - Catches obvious fraud (high amount + foreign + night)")
    print(f"")
    print(f"  ❌ Where rules fail:")
    print(f"     - Miss fraud with COMBINATIONS of weak signals")
    print(f"     - High false positive rate (no user context)")
    print(f"     - Large gray zone requiring manual review")
    print(f"     - Static — don't adapt to new fraud patterns")
    print(f"")
    print(f"  📈 Next: Experiment 2 — Can ML learn the feature interactions")
    print(f"     that rules miss?")
    print(f"")
    print(f"  ROC-AUC: {auc:.4f}  (our target to beat)")

    return auc, best


if __name__ == "__main__":
    main()
