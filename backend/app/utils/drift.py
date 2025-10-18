from __future__ import annotations
import math
from typing import List, Tuple

def _safe_probs(vec: List[float]) -> List[float]:
    s = sum(vec)
    if s <= 0:  # avoid division by zero
        n = len(vec)
        return [1.0 / n] * n
    return [max(1e-12, x / s) for x in vec]

def population_stability_index(expected: List[float], actual: List[float]) -> float:
    """PSI using binned percentages. expected/actual can be raw counts or percentages."""
    e = _safe_probs(expected)
    a = _safe_probs(actual)
    if len(e) != len(a):
        raise ValueError("expected and actual must be same length")
    psi = 0.0
    for ei, ai in zip(e, a):
        psi += (ai - ei) * math.log(ai / ei)
    return float(psi)

def classify_psi(psi: float) -> Tuple[str, str]:
    """Return (severity, message)."""
    if psi >= 0.25:
        return ("high", "Major drift detected")
    if psi >= 0.10:
        return ("medium", "Moderate drift detected")
    return ("low", "Minor/no drift")
