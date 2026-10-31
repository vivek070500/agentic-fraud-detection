"""
Synthetic Fraud Dataset Generator
==================================
Generates 100K transactions with 12 interpretable features and ~3% fraud rate.

Fraud patterns embedded (each fraud txn follows 1-3 of these):
  1. High amount relative to customer's history (amount_deviation > 5x)
  2. Burst velocity — many transactions in short window
  3. Night transactions (22:00 - 05:00)
  4. Device change + foreign merchant combo
  5. New account (< 30 days) with high-value purchase
  6. Geographic anomaly (distance > 3000km from home)

Usage:
    python -m data.generate_dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 100_000
FRAUD_RATE = 0.03
OUTPUT_DIR = Path(__file__).parent


def generate_customers(n_customers: int = 20_000, rng: np.random.Generator = None) -> dict:
    """Generate customer profiles that transactions will reference."""
    return {
        "avg_amount": rng.lognormal(mean=3.5, sigma=0.8, size=n_customers),  # median ~$33
        "account_age_days": rng.integers(1, 2000, size=n_customers),
        "home_lat": rng.uniform(8, 37, size=n_customers),    # India + US spread
        "home_lon": rng.uniform(-122, 80, size=n_customers),
    }


def generate_legitimate(n: int, customers: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate legitimate transactions — normal spending behavior."""
    cust_idx = rng.integers(0, len(customers["avg_amount"]), size=n)

    avg_amounts = customers["avg_amount"][cust_idx]
    account_ages = customers["account_age_days"][cust_idx]

    amounts = rng.lognormal(
        mean=np.log(avg_amounts),
        sigma=0.4
    )
    amounts = np.clip(amounts, 0.50, 10_000)

    hour_probs = np.array([
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01,   # 0-5: very low
        0.03, 0.05, 0.07, 0.08, 0.09, 0.09,        # 6-11: morning ramp
        0.08, 0.07, 0.06, 0.06, 0.05, 0.05,         # 12-17: afternoon
        0.04, 0.04, 0.03, 0.03, 0.02, 0.015         # 18-23: evening decline
    ])
    hour_probs = hour_probs / hour_probs.sum()
    hours = rng.choice(range(24), size=n, p=hour_probs)
    days = rng.integers(0, 7, size=n)

    return pd.DataFrame({
        "amount": np.round(amounts, 2),
        "hour_of_day": hours,
        "day_of_week": days,
        "is_weekend": (days >= 5).astype(int),
        "is_night": ((hours >= 22) | (hours <= 5)).astype(int),
        "merchant_category": rng.choice(
            ["grocery", "restaurant", "gas_station", "online_retail",
             "electronics", "travel", "subscription", "atm"],
            size=n,
            p=[0.25, 0.15, 0.10, 0.20, 0.08, 0.05, 0.12, 0.05]
        ),
        "is_foreign_merchant": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
        "customer_account_age_days": account_ages,
        "customer_avg_amount_30d": np.round(avg_amounts, 2),
        "amount_deviation": np.round(amounts / avg_amounts, 2),
        "transaction_velocity_1h": rng.poisson(lam=1.2, size=n).clip(1, 20),
        "device_changed": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
        "distance_from_home_km": np.round(
            rng.exponential(scale=15, size=n).clip(0, 500), 1
        ),
        "is_fraud": 0,
    })


def generate_fraud(n: int, customers: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate fraudulent transactions — each follows 1-3 fraud patterns.
    The patterns are probabilistic, not deterministic, so the model
    has to learn combinations, not simple rules.
    """
    cust_idx = rng.integers(0, len(customers["avg_amount"]), size=n)
    avg_amounts = customers["avg_amount"][cust_idx]
    account_ages = customers["account_age_days"][cust_idx]

    # Pattern selection: each fraud txn gets 1-3 patterns
    patterns = {
        "high_amount": rng.random(n) < 0.6,
        "velocity_burst": rng.random(n) < 0.4,
        "night_txn": rng.random(n) < 0.5,
        "device_foreign": rng.random(n) < 0.45,
        "new_account": rng.random(n) < 0.3,
        "geo_anomaly": rng.random(n) < 0.4,
    }

    # --- Amount ---
    amounts = np.where(
        patterns["high_amount"],
        avg_amounts * rng.uniform(5, 25, size=n),  # 5-25x normal
        avg_amounts * rng.uniform(0.8, 3, size=n)  # some fraud is normal-amount
    )
    amounts = np.clip(amounts, 1.0, 15_000)

    # --- Time ---
    hours_normal = rng.choice(range(24), size=n)
    hours_night = rng.choice([22, 23, 0, 1, 2, 3, 4], size=n)
    hours = np.where(patterns["night_txn"], hours_night, hours_normal)

    days = rng.integers(0, 7, size=n)

    # --- Velocity ---
    velocity_normal = rng.poisson(lam=1.5, size=n).clip(1, 5)
    velocity_burst = rng.poisson(lam=8, size=n).clip(5, 30)
    velocity = np.where(patterns["velocity_burst"], velocity_burst, velocity_normal)

    # --- Device & foreign ---
    device_changed = np.where(patterns["device_foreign"], 1, rng.choice([0, 1], size=n, p=[0.7, 0.3]))
    is_foreign = np.where(patterns["device_foreign"], 1, rng.choice([0, 1], size=n, p=[0.6, 0.4]))

    # --- Account age (new account pattern) ---
    account_age = np.where(
        patterns["new_account"],
        rng.integers(1, 30, size=n),
        account_ages
    )

    # --- Distance ---
    distance_normal = rng.exponential(scale=50, size=n).clip(0, 500)
    distance_far = rng.uniform(3000, 12000, size=n)
    distance = np.where(patterns["geo_anomaly"], distance_far, distance_normal)

    merchant_cats = rng.choice(
        ["grocery", "restaurant", "gas_station", "online_retail",
         "electronics", "travel", "subscription", "atm"],
        size=n,
        p=[0.08, 0.05, 0.05, 0.30, 0.25, 0.12, 0.05, 0.10]  # fraud skews electronics/online/travel
    )

    return pd.DataFrame({
        "amount": np.round(amounts, 2),
        "hour_of_day": hours,
        "day_of_week": days,
        "is_weekend": (days >= 5).astype(int),
        "is_night": ((hours >= 22) | (hours <= 5)).astype(int),
        "merchant_category": merchant_cats,
        "is_foreign_merchant": is_foreign.astype(int),
        "customer_account_age_days": account_age,
        "customer_avg_amount_30d": np.round(avg_amounts, 2),
        "amount_deviation": np.round(amounts / avg_amounts, 2),
        "transaction_velocity_1h": velocity,
        "device_changed": device_changed.astype(int),
        "distance_from_home_km": np.round(distance, 1),
        "is_fraud": 1,
    })


def generate_dataset() -> pd.DataFrame:
    """Main entry point — generates full dataset, shuffles, and saves."""
    rng = np.random.default_rng(SEED)

    n_fraud = int(N_SAMPLES * FRAUD_RATE)
    n_legit = N_SAMPLES - n_fraud

    print(f"Generating {N_SAMPLES:,} transactions ({FRAUD_RATE:.0%} fraud rate)...")
    print(f"  Legitimate: {n_legit:,}")
    print(f"  Fraudulent: {n_fraud:,}")

    customers = generate_customers(rng=rng)

    df_legit = generate_legitimate(n_legit, customers, rng)
    df_fraud = generate_fraud(n_fraud, customers, rng)

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Add transaction IDs
    df.insert(0, "transaction_id", [f"txn_{i:06d}" for i in range(len(df))])

    output_path = OUTPUT_DIR / "fraud_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"   File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Quick sanity stats
    print(f"\n📊 Feature Summary:")
    print(f"   Amount — Legit avg: ${df[df.is_fraud==0]['amount'].mean():.2f}, "
          f"Fraud avg: ${df[df.is_fraud==1]['amount'].mean():.2f}")
    print(f"   Night txns — Legit: {df[df.is_fraud==0]['is_night'].mean():.1%}, "
          f"Fraud: {df[df.is_fraud==1]['is_night'].mean():.1%}")
    print(f"   Device changed — Legit: {df[df.is_fraud==0]['device_changed'].mean():.1%}, "
          f"Fraud: {df[df.is_fraud==1]['device_changed'].mean():.1%}")
    print(f"   Velocity (avg) — Legit: {df[df.is_fraud==0]['transaction_velocity_1h'].mean():.1f}, "
          f"Fraud: {df[df.is_fraud==1]['transaction_velocity_1h'].mean():.1f}")
    print(f"   Distance (avg) — Legit: {df[df.is_fraud==0]['distance_from_home_km'].mean():.0f}km, "
          f"Fraud: {df[df.is_fraud==1]['distance_from_home_km'].mean():.0f}km")

    return df


if __name__ == "__main__":
    generate_dataset()
