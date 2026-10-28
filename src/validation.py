"""Shared input contracts for probability and bankroll calculations."""
import numpy as np


def vector(values, name, lower=None, upper=None, strict_lower=False):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    if lower is not None and np.any(arr <= lower if strict_lower else arr < lower):
        raise ValueError(f"{name} outside lower bound {lower}")
    if upper is not None and np.any(arr > upper):
        raise ValueError(f"{name} outside upper bound {upper}")
    return arr


def probability_odds(probs, odds):
    p = vector(probs, "probabilities", 0, 1)
    o = vector(odds, "decimal odds", 1, strict_lower=True)
    if p.shape != o.shape:
        raise ValueError("Probability and odds vectors must have equal length")
    return p, o


def simplex(probs):
    p = vector(probs, "probabilities", 0, 1)
    if not len(p) or not np.isclose(p.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("Probabilities must sum to one")
    return p
