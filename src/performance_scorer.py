#!/usr/bin/env python3
"""
performance_scorer.py  --  Standalone T+5 Performance Scoring & Labeling

Decoupled utility module for the Vela MQR ablation study harness.
No imports from any other project file -- fully self-contained.

Public API
----------
    score_and_label(markets, bands='symmetric') -> list[dict]

        Takes a list of market dicts that already have a 't5_metrics' key,
        computes actual_performance_score (0-100) via power-law log transform
        + population-level min-max scaling, then assigns actual_rating (L1-L5).

        bands='symmetric'  (default)
            Percentile-based labeling dynamically calibrated to the length of
            the supplied list.  Same bands as the Mirror Validation pipeline:
                Top 10%  (>=90th)  -> L5
                70th-90th          -> L4
                45th-70th          -> L3
                20th-45th          -> L2
                <20th              -> L1

        bands='absolute'
            Bypasses all percentile math.  Labels assigned from hard USD
            thresholds on peak_exit_usd (the single most interpretable
            venture signal):
                >= $10 B   -> L5
                $1B - $10B -> L4
                $100M - $1B -> L3
                $10M - $100M -> L2
                < $10M     -> L1

Each input dict is mutated in-place (keys added/updated):
    actual_performance_score  float | None
    actual_percentile         float | None   (symmetric mode only)
    actual_rating             str   | None   ("L1" ... "L5")
    t5_scaled                 dict  | None   (component breakdown, audit use)

The list is also returned so the call can be chained.

Metrics schema expected inside each market['t5_metrics']:
    peak_exit_usd       numeric or None/"Not Found"/etc.
    total_funding_usd   numeric or None/"Not Found"/etc.
    unicorn_count       numeric or None/"Not Found"/etc.
"""

from __future__ import annotations

import math
from typing import Literal

# ── Labeling constants ────────────────────────────────────────────────────────

# Symmetric percentile bands (used in bands='symmetric' mode)
_SYMMETRIC_BANDS: list[tuple[float, float, str]] = [
    (90.0, 101.0, "L5"),   # top 10%
    (70.0,  90.0, "L4"),
    (45.0,  70.0, "L3"),
    (20.0,  45.0, "L2"),
    ( 0.0,  20.0, "L1"),   # bottom 20%
]

# Absolute USD thresholds applied to peak_exit_usd (bands='absolute' mode)
# Each entry: (minimum_usd_inclusive, label)
_ABSOLUTE_THRESHOLDS: list[tuple[float, str]] = [
    (10_000_000_000, "L5"),   # >= $10B
    ( 1_000_000_000, "L4"),   # $1B - $10B
    (   100_000_000, "L3"),   # $100M - $1B
    (    10_000_000, "L2"),   # $10M - $100M
    (             0, "L1"),   # < $10M
]

RATING_LABELS: dict[str, str] = {
    "L5": "Ideal",
    "L4": "Attractive",
    "L3": "Viable",
    "L2": "Headwinds",
    "L1": "Hostile",
}

RATING_NUM: dict[str, int] = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_float(v: object, default: float = 0.0) -> float:
    """
    Coerce v to a non-negative float.
    Returns default for None, empty string, 'Not Found', 'null', 'n/a',
    'none', non-numeric strings, or negative values.
    """
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return max(0.0, float(v))
    s = str(v).strip().lower()
    if s in ("", "not found", "null", "n/a", "none"):
        return default
    try:
        result = float(s.replace(",", ""))
        return max(0.0, result)
    except ValueError:
        return default


def _minmax(values: list[float]) -> list[float]:
    """
    Min-max scale a list of floats to [0, 100].
    If all values are identical (zero variance), returns 50.0 for each.
    """
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [50.0] * len(values)
    return [(v - lo) / (hi - lo) * 100.0 for v in values]


def _assign_from_bands(percentile: float, bands: list[tuple[float, float, str]]) -> str:
    for lo, hi, label in bands:
        if lo <= percentile < hi:
            return label
    return "L1"


def _assign_absolute(peak_exit_usd: float) -> str:
    for threshold, label in _ABSOLUTE_THRESHOLDS:
        if peak_exit_usd >= threshold:
            return label
    return "L1"


def _compute_performance_scores(markets: list[dict]) -> None:
    """
    Mutates each market in-place, adding:
        actual_performance_score  (float, 0-100)
        t5_scaled                 (dict with component breakdown)

    Markets with no t5_metrics get actual_performance_score = None.
    Scoring formula (power-law, population-level):
        LogPeak    = log10(1 + peak_exit_usd)
        LogFunding = log10(1 + total_funding_usd)
        unicorn_raw = unicorn_count  (floored at 0)
        -- min-max scale each feature across the whole population --
        score = 0.50 * scaled_log_peak
              + 0.30 * scaled_log_funding
              + 0.20 * scaled_unicorn
    """
    scored = [m for m in markets if m.get("t5_metrics") is not None]
    for m in markets:
        if m.get("t5_metrics") is None:
            m.setdefault("actual_performance_score", None)
            m.setdefault("t5_scaled", None)

    if not scored:
        return

    log_peaks:    list[float] = []
    log_fundings: list[float] = []
    unicorn_raws: list[float] = []

    for m in scored:
        mx = m["t5_metrics"]
        log_peaks.append(math.log10(1.0 + _safe_float(mx.get("peak_exit_usd"))))
        log_fundings.append(math.log10(1.0 + _safe_float(mx.get("total_funding_usd"))))
        unicorn_raws.append(_safe_float(mx.get("unicorn_count"), default=0.0))

    scaled_peaks    = _minmax(log_peaks)
    scaled_fundings = _minmax(log_fundings)
    scaled_unicorns = _minmax(unicorn_raws)

    for m, sp, sf, su in zip(scored, scaled_peaks, scaled_fundings, scaled_unicorns):
        m["actual_performance_score"] = round(0.50 * sp + 0.30 * sf + 0.20 * su, 2)
        m["t5_scaled"] = {
            "scaled_log_peak":    round(sp, 2),
            "scaled_log_funding": round(sf, 2),
            "scaled_unicorn":     round(su, 2),
        }


# ── Public API ────────────────────────────────────────────────────────────────

def score_and_label(
    markets: list[dict],
    bands: Literal["symmetric", "absolute"] = "symmetric",
) -> list[dict]:
    """
    Compute actual_performance_score and assign actual_rating for each market.

    Parameters
    ----------
    markets : list[dict]
        Each dict must have a 't5_metrics' key with at least:
            peak_exit_usd, total_funding_usd, unicorn_count
        (all values may be None / 0 / "Not Found" -- _safe_float handles them).
        Markets without 't5_metrics' receive None for all output fields.

    bands : 'symmetric' | 'absolute'
        'symmetric' (default)
            Percentile-based.  Scale self-calibrates to the length of this
            specific list -- feeding 10 vs 60 vs 100 markets will produce
            different absolute thresholds at each band boundary, but the
            relative ordering within the cohort is preserved.

        'absolute'
            Hard USD thresholds on peak_exit_usd.  L1-L5 assignment is
            independent of cohort size, enabling cross-cohort comparison
            without re-calibration.

    Returns
    -------
    The same list (mutated in-place), returned for chaining.
    Added/updated keys per market:
        actual_performance_score  float | None
        actual_percentile         float | None   (symmetric only; None for absolute)
        actual_rating             str   | None
        t5_scaled                 dict  | None
    """
    if bands not in ("symmetric", "absolute"):
        raise ValueError(f"bands must be 'symmetric' or 'absolute', got {bands!r}")

    # Step 1: compute population-level performance scores for all modes
    _compute_performance_scores(markets)

    # Step 2: assign ratings
    if bands == "absolute":
        for m in markets:
            if m.get("t5_metrics") is None:
                m["actual_percentile"] = None
                m["actual_rating"]     = None
            else:
                peak = _safe_float(m["t5_metrics"].get("peak_exit_usd"))
                m["actual_percentile"] = None   # not applicable in absolute mode
                m["actual_rating"]     = _assign_absolute(peak)

    else:  # symmetric
        scored = [m for m in markets if m.get("actual_performance_score") is not None]
        n = len(scored)
        if n > 0:
            sorted_scored = sorted(scored, key=lambda m: m["actual_performance_score"])
            for rank, m in enumerate(sorted_scored):
                pct = (rank / (n - 1) * 100.0) if n > 1 else 50.0
                m["actual_percentile"] = round(pct, 2)
                m["actual_rating"]     = _assign_from_bands(pct, _SYMMETRIC_BANDS)
        for m in markets:
            if m.get("actual_performance_score") is None:
                m.setdefault("actual_percentile", None)
                m.setdefault("actual_rating",     None)

    return markets


def rating_delta(predicted: str, actual: str) -> int:
    """
    Return the signed band distance between predicted and actual rating.
    Positive = over-predicted, negative = under-predicted.
    Returns 0 if either value is missing/invalid.
    """
    p = RATING_NUM.get(predicted, 0)
    a = RATING_NUM.get(actual, 0)
    if p == 0 or a == 0:
        return 0
    return p - a


def accuracy_summary(markets: list[dict]) -> dict:
    """
    Compute exact-match and within-1-band accuracy for a scored list.

    Only markets with both predicted_rating and actual_rating populated
    are included in the denominator.

    Returns
    -------
    dict with keys: total, exact, off1, miss,
                    exact_pct, off1_pct, miss_pct
    """
    eligible = [
        m for m in markets
        if m.get("predicted_rating") and m.get("actual_rating")
    ]
    total = len(eligible)
    if total == 0:
        return {"total": 0, "exact": 0, "off1": 0, "miss": 0,
                "exact_pct": 0.0, "off1_pct": 0.0, "miss_pct": 0.0}

    exact = sum(1 for m in eligible if m["predicted_rating"] == m["actual_rating"])
    off1  = sum(1 for m in eligible if abs(rating_delta(m["predicted_rating"], m["actual_rating"])) <= 1)
    miss  = total - off1

    return {
        "total":     total,
        "exact":     exact,
        "off1":      off1,
        "miss":      miss,
        "exact_pct": round(exact / total * 100, 1),
        "off1_pct":  round(off1  / total * 100, 1),
        "miss_pct":  round(miss  / total * 100, 1),
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _test_markets = [
        {"id": "t1", "name": "Ride-Hailing (2010)",
         "t5_metrics": {"peak_exit_usd": 70_000_000_000, "total_funding_usd": 24_000_000_000, "unicorn_count": 3}},
        {"id": "t2", "name": "CSM SaaS (2013)",
         "t5_metrics": {"peak_exit_usd": 700_000_000, "total_funding_usd": 150_000_000, "unicorn_count": 0}},
        {"id": "t3", "name": "3D Printing Consumer (2013)",
         "t5_metrics": {"peak_exit_usd": 400_000_000, "total_funding_usd": 80_000_000, "unicorn_count": 0}},
        {"id": "t4", "name": "Connected Fitness (2013)",
         "t5_metrics": {"peak_exit_usd": 4_150_000_000, "total_funding_usd": 994_000_000, "unicorn_count": 1}},
        {"id": "t5", "name": "Programmatic DOOH (2013)",
         "t5_metrics": {"peak_exit_usd": 27_000_000, "total_funding_usd": 30_000_000, "unicorn_count": 0}},
        {"id": "t6", "name": "No T+5 data",
         "t5_metrics": None},
    ]

    import copy, json

    print("=== bands='symmetric' ===")
    sym = score_and_label(copy.deepcopy(_test_markets), bands="symmetric")
    for m in sym:
        print(f"  {m['id']}  score={m['actual_performance_score']}  "
              f"pct={m['actual_percentile']}  rating={m['actual_rating']}  ({m['name']})")

    print("\n=== bands='absolute' ===")
    abs_ = score_and_label(copy.deepcopy(_test_markets), bands="absolute")
    for m in abs_:
        print(f"  {m['id']}  score={m['actual_performance_score']}  "
              f"rating={m['actual_rating']}  ({m['name']})")
