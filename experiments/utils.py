"""Shared utilities across experiments."""


def rule_based_scorer(row) -> float:
    """
    V1 rule-based fraud scorer.
    Each rule adds to a risk score between 0 and 1.
    """
    risk = 0.0

    if row["amount_deviation"] > 5:
        risk += 0.35
    elif row["amount_deviation"] > 3:
        risk += 0.20

    if row["is_night"] == 1:
        risk += 0.15

    if row["device_changed"] == 1:
        risk += 0.15

    if row["is_foreign_merchant"] == 1:
        risk += 0.10

    if row["transaction_velocity_1h"] > 5:
        risk += 0.15
    elif row["transaction_velocity_1h"] > 3:
        risk += 0.08

    if row["distance_from_home_km"] > 3000:
        risk += 0.20
    elif row["distance_from_home_km"] > 500:
        risk += 0.10

    if row["customer_account_age_days"] < 30:
        risk += 0.10

    return min(risk, 1.0)
