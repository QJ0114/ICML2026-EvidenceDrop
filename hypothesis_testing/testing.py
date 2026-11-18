"""Finite-sample calibration and held-out evaluation of scalar risk scores."""

from itertools import groupby
from math import isfinite

from scipy.stats import binom


def ordered_results(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON list of scored examples.")
    for row in rows:
        if row["is_correct"] not in (0, 1) or not isfinite(row["risk_score"]):
            raise ValueError("Each example needs binary is_correct and finite risk_score.")
    return sorted(rows, key=lambda row: row["risk_score"])


def calibrate(rows, alpha, delta):
    """Choose the largest order statistic supporting the strict rule risk < threshold."""
    if not all(0 < value < 1 for value in (alpha, delta)):
        raise ValueError("alpha and delta must both be in (0, 1).")
    risks = [row["risk_score"] for row in ordered_results(rows) if not row["is_correct"]]
    n = len(risks)
    # For ascending null risks r_(j), P(R < r_(j)) exceeds alpha with
    # probability at most P(Binomial(n, alpha) <= j - 1).
    lo, hi = 0, n
    while lo < hi:
        j = (lo + hi + 1) // 2
        if binom.cdf(j - 1, n, alpha) <= delta:
            lo = j
        else:
            hi = j - 1
    return {
        "alpha": alpha,
        "delta": delta,
        "n_incorrect_calibration": n,
        "order_statistic": lo,
        "threshold": risks[lo - 1] if lo else None,
    }


def evaluate(rows, threshold):
    """Report selective prediction metrics, averaging ranks within tied scores."""
    if threshold is not None and not isfinite(threshold):
        raise ValueError("The threshold must be finite or null (reject all).")
    rows = ordered_results(rows)
    accepted = [row for row in rows if threshold is not None and row["risk_score"] < threshold]
    correct = sum(row["is_correct"] for row in rows)
    accepted_correct = sum(row["is_correct"] for row in accepted)
    accepted_incorrect = len(accepted) - accepted_correct
    incorrect = len(rows) - correct

    area = 0.0
    count = wrong = 0
    for _, tied_rows in groupby(rows, key=lambda row: row["risk_score"]):
        group = list(tied_rows)
        group_wrong = sum(not row["is_correct"] for row in group)
        for index in range(1, len(group) + 1):
            area += (wrong + index * group_wrong / len(group)) / (count + index)
        count += len(group)
        wrong += group_wrong

    return {
        "n_test": len(rows),
        "n_accepted": len(accepted),
        "accuracy": correct / len(rows),
        "coverage": len(accepted) / len(rows),
        "selective_accuracy": accepted_correct / len(accepted) if accepted else None,
        "false_acceptance_rate": accepted_incorrect / incorrect if incorrect else None,
        "false_rejection_rate": (correct - accepted_correct) / correct if correct else None,
        "aurc": area / len(rows),
    }
