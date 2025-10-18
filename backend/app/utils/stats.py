from typing import List, Tuple

def ks_d_stat(sample_a: List[float], sample_b: List[float]) -> float:
    # Empirical CDF difference (two-sample)
    a = sorted(sample_a); b = sorted(sample_b)
    i=j=0; na=len(a); nb=len(b); d=0.0
    while i<na and j<nb:
        if a[i] <= b[j]: i+=1
        else: j+=1
        fa = i/na; fb = j/nb
        d = max(d, abs(fa-fb))
    # finish tails
    d = max(d, abs(1 - j/nb), abs(1 - i/na))
    return d

def ks_severity(d: float, warn: float=0.10, high: float=0.20) -> str:
    return "high" if d>=high else ("medium" if d>=warn else "low")
