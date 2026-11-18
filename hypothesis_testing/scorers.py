"""Map vLLM token log probabilities to a scalar risk, higher for less reliable traces."""

import numpy as np

from uncertainty_estimation.utils.metrics import (
    calculate_ema_numpy,
    calculate_risk_with_running_mean_drop,
    get_logprob_curve,
)


def shannon_evidence(step):
    log_probs = np.array([token.logprob for token in step.values()])
    probs = np.exp(log_probs - log_probs.max())
    probs /= probs.sum()
    return float(np.sum(probs * np.log(probs + 1e-10)))


def logtoku_evidence(step):
    return sum(token.logprob for token in step.values())


def worst_drops(curve, ema_span, drop_k):
    deltas = np.diff(calculate_ema_numpy(curve, span=ema_span))
    drops = deltas[deltas < 0]
    return -float(np.mean(np.sort(drops)[:drop_k])) if len(drops) else 0.0


def score(method, logprobs, *, ema_span, drop_k):
    if method.startswith("ln-s"):
        curve = get_logprob_curve(logprobs)
        if method == "ln-s-drop":
            return calculate_risk_with_running_mean_drop(curve, ema_span, drop_k)
    else:
        evidence = shannon_evidence if method.startswith("shannon") else logtoku_evidence
        curve = [evidence(step) for step in logprobs if step]
        if method.endswith("-drop"):
            return worst_drops(curve, ema_span, drop_k)
    return -float(np.mean(curve))
