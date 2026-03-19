#!/usr/bin/env python3
"""
pipeline_step4_rating.py  —  Step 4: Composite Scoring, Rating Assignment,
                              Probabilities & Nearest Neighbours

Reads  : reference_population_master.json  (root of project)
Writes : final_rated_population.json       (root of project)
         sample_report.md                  (root of project)

Processing steps
----------------
1. Apply causal weights  → composite_score  (0-100)
2. Empirical CDF         → percentile rank within each market_structure group
3. Logistic regression   → P(outcome ≥ L3)  + "X out of 20" language
4. Cosine nearest-neighbour search (7-D normalised score vectors)
5. L1-L5 rating assignment from percentile bands
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
INPUT_FILE  = BASE_DIR / "reference_population_master.json"
OUTPUT_FILE = BASE_DIR / "final_rated_population.json"
REPORT_FILE = BASE_DIR / "sample_report.md"

# ── 7 scored dimensions (order also used for NN vector) ───────────────────────
SCORE_DIMS = [
    "timing",
    "competition",
    "market_size",
    "customer_readiness",
    "regulatory",
    "infrastructure",
    "market_structure",
]

# ── Causal weight tables (primary 3 dimensions) ───────────────────────────────
# Keys must match the canonical market_structure classification enum.
#
# For regulated_infrastructure the three primary weights apply to *50%* of the
# composite score; the remaining four dimensions equally share the other 50%.
# (Section 5.5 Step 2 of the methodology)  All other structure types: primary
# weights sum to 100% and residual dimensions contribute 0%.

CAUSAL_WEIGHTS: dict[str, dict[str, float]] = {
    "winner_take_most": {
        "timing":       0.32,
        "competition":  0.52,
        "market_size":  0.16,
    },
    "platform_two_sided": {
        "timing":       0.55,
        "competition":  0.22,
        "market_size":  0.23,
    },
    "technology_enablement": {
        "timing":       0.60,
        "competition":  0.18,
        "market_size":  0.22,
    },
    "fragmented_niche": {
        "timing":       0.25,
        "competition":  0.15,
        "market_size":  0.60,
    },
    "regulated_infrastructure": {
        # These weights apply to 50% of the score (Section 5.5 Step 2)
        "timing":       0.22,
        "competition":  0.28,
        "market_size":  0.50,
    },
}

# Residual dimensions (used only for regulated_infrastructure)
RESIDUAL_DIMS = ["customer_readiness", "regulatory", "infrastructure", "market_structure"]

# ── Percentile bands → rating ──────────────────────────────────────────────────
# Bands are [lo, hi) where 100 is included in L5 via the sentinel 101.
PERCENTILE_BANDS: list[tuple[float, float, str]] = [
    (90.0, 101.0, "L5"),
    (70.0,  90.0, "L4"),
    (45.0,  70.0, "L3"),
    (25.0,  45.0, "L2"),
    ( 0.0,  25.0, "L1"),
]

RATING_DESCRIPTIONS: dict[str, str] = {
    "L5": "Ideal",
    "L4": "Attractive",
    "L3": "Viable — Investment Grade Threshold",
    "L2": "Headwinds",
    "L1": "Hostile",
}

# ── Score-tier reference (per individual dimension, for report footnotes) ──────
SCORE_TIERS = [
    (90, 100, "Ideal"),
    (75,  89, "Attractive"),
    (60,  74, "Viable"),
    (40,  59, "Headwinds"),
    ( 0,  39, "Hostile"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _score_tier_label(score: float) -> str:
    for lo, hi, label in SCORE_TIERS:
        if lo <= score <= hi:
            return label
    return "—"


def _get_structure(market: dict) -> str:
    """Return the verified market_structure classification from step3 (canonical)."""
    return market["step3"]["feature_matrix"]["market_structure"]["value"]


def _get_scores(market: dict) -> dict[str, float]:
    """Return the step3 scores dict, defaulting missing dims to 50."""
    raw = market.get("step3", {}).get("scores", {})
    return {d: float(raw.get(d, 50)) for d in SCORE_DIMS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Load
# ══════════════════════════════════════════════════════════════════════════════

def load_markets(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    all_markets = data.get("markets", [])
    valid = [m for m in all_markets if m.get("step3", {}).get("scores")]

    print(f"  Loaded {len(all_markets)} markets total; "
          f"{len(valid)} have step3 scores and will be rated.")

    if len(valid) < 5:
        print("  [WARN] Fewer than 5 scored markets — statistical outputs "
              "(percentiles, logistic regression) will be low-confidence.")

    return valid


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Composite Score
# ══════════════════════════════════════════════════════════════════════════════

def compute_composite(market: dict) -> float:
    """
    Compute composite_score (0-100) using structure-specific causal weights.

    For regulated_infrastructure (Section 5.5 Step 2):
        composite = 0.50 × primary_component
                  + 0.50 × mean(customer_readiness, regulatory,
                                infrastructure, market_structure)
        where primary_component = 0.22·T + 0.28·C + 0.50·S  (sums to 100 %)

    For all other structures:
        composite = w_timing·T + w_competition·C + w_market_size·S
        (weights sum to 100 %; residual dims contribute 0 %)
    """
    scores  = _get_scores(market)
    struct  = _get_structure(market)
    weights = CAUSAL_WEIGHTS.get(struct, CAUSAL_WEIGHTS["winner_take_most"])

    # Primary weighted component
    primary = (
        weights["timing"]       * scores["timing"]
        + weights["competition"]  * scores["competition"]
        + weights["market_size"]  * scores["market_size"]
    )

    if struct == "regulated_infrastructure":
        # Section 5.5 Step 2: primary dims → 50 %; residual dims → 50 %
        residual_vals = [scores[d] for d in RESIDUAL_DIMS]
        residual      = float(np.mean(residual_vals))
        composite     = 0.50 * primary + 0.50 * residual
    else:
        # Primary weights already sum to 1.0; no residual contribution
        composite = primary

    return round(float(composite), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Percentile Rank (empirical CDF within structure group)
# ══════════════════════════════════════════════════════════════════════════════

def compute_percentiles(markets: list[dict]) -> dict[str, float]:
    """
    Compute each market's percentile rank relative to *other markets of the
    same market_structure type* (within-group empirical CDF).

    Formula: percentile_i = rank_i / (n - 1) × 100
      where rank_i is the 0-based rank among the group (0 = lowest score).
    For singleton groups the percentile is set to 50.0.
    """
    # Group by structure
    groups: dict[str, list[dict]] = {}
    for m in markets:
        groups.setdefault(_get_structure(m), []).append(m)

    percentiles: dict[str, float] = {}

    for struct, group in groups.items():
        n = len(group)
        sorted_group = sorted(group, key=lambda m: m["_composite"])

        for rank, m in enumerate(sorted_group):
            pct = (rank / (n - 1) * 100.0) if n > 1 else 50.0
            percentiles[m["id"]] = round(pct, 2)

        print(f"    {struct}: {n} markets — "
              f"composite range [{sorted_group[0]['_composite']:.1f}, "
              f"{sorted_group[-1]['_composite']:.1f}]")

    return percentiles


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Logistic Regression → P(outcome ≥ L3)
# ══════════════════════════════════════════════════════════════════════════════

def fit_logistic_models(markets: list[dict]) -> dict[str, dict]:
    """
    Fit one logistic regression per market_structure type.

    Predictor : composite_score  (standardised internally)
    Target    : outcome_T5 = 1 if percentile_rank ≥ 45  (L3 band or above)
                           = 0 otherwise

    Design note — why this is not purely circular
    ─────────────────────────────────────────────
    The percentile threshold defines a *hard* step function boundary.
    Logistic regression fits a smooth sigmoid through the composite-score
    distribution, producing continuous probabilities that carry information
    about *distance from the boundary* — markets near the 45th-percentile
    boundary get probabilities near 0.50, not a binary flip.  This is
    especially useful when evaluating a *new* target company against the
    reference population.

    Returns a dict:  { structure_type → {"model": LR, "scaler": Scaler}
                                       | {"model": None, "p_const": float} }
    """
    groups: dict[str, list[dict]] = {}
    for m in markets:
        groups.setdefault(_get_structure(m), []).append(m)

    fitted: dict[str, dict] = {}

    for struct, group in groups.items():
        X = np.array([[m["_composite"]] for m in group], dtype=float)
        y = np.array([1 if m["_percentile"] >= 45.0 else 0 for m in group])

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        # Degenerate: only one class present — return constant probability
        if n_pos == 0 or n_neg == 0:
            p_const = 1.0 if n_pos > 0 else 0.0
            fitted[struct] = {"model": None, "p_const": p_const}
            print(f"    {struct}: degenerate (all {'positive' if n_pos else 'negative'}) "
                  f"→ p_const={p_const:.2f}")
            continue

        scaler  = StandardScaler()
        X_std   = scaler.fit_transform(X)

        lr = LogisticRegression(
            solver="lbfgs",
            C=1.0,          # L2 regularisation
            max_iter=1000,
            random_state=42,
        )
        lr.fit(X_std, y)

        b0 = float(lr.intercept_[0])
        b1 = float(lr.coef_[0][0])
        print(f"    {struct} ({len(group)} obs): "
              f"b0={b0:+.3f}  b1={b1:+.3f}  "
              f"[pos={n_pos}, neg={n_neg}]")

        fitted[struct] = {"model": lr, "scaler": scaler}

    return fitted


def predict_probability(market: dict, models: dict[str, dict]) -> float:
    """Return P(outcome ≥ L3) from the fitted logistic model for this market."""
    struct = _get_structure(market)
    entry  = models.get(struct, {})

    if entry.get("model") is None:
        return float(entry.get("p_const", 0.5))

    X     = np.array([[market["_composite"]]])
    X_std = entry["scaler"].transform(X)
    prob  = entry["model"].predict_proba(X_std)[0][1]
    return round(float(prob), 4)


def prob_to_x_of_20(prob: float) -> str:
    x = min(20, max(0, round(prob * 20)))
    return f"{x} out of 20"


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Nearest Neighbour Analysis (cosine similarity, 7-D)
# ══════════════════════════════════════════════════════════════════════════════

def compute_nearest_neighbours(
    markets: list[dict],
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """
    For each market find the top-k most similar historical markets by cosine
    similarity on normalised 7-dimensional step3 score vectors.

    Returns: { market_id → [{"id", "market_name", "ref_year",
                              "cosine_similarity", "rating"}, ...] }
    """
    ids   = [m["id"]                                              for m in markets]
    names = [m["base_profile"].get("market_name", m["id"])       for m in markets]
    years = [m.get("ref_year", "?")                              for m in markets]

    # Build N × 7 raw score matrix
    raw_matrix = np.array(
        [[_get_scores(m)[d] for d in SCORE_DIMS] for m in markets],
        dtype=float,
    )

    # L2-normalise each row (unit vectors → dot product == cosine similarity)
    norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)
    normed = raw_matrix / norms

    # Pairwise cosine distance (N × N); distance = 1 − similarity
    dist_matrix = cdist(normed, normed, metric="cosine")

    neighbours: dict[str, list[dict]] = {}
    for i, mid in enumerate(ids):
        dists          = dist_matrix[i].copy()
        dists[i]       = np.inf                          # exclude self
        top_idx        = np.argsort(dists)[:top_k]
        neighbours[mid] = [
            {
                "id":               ids[j],
                "market_name":      names[j],
                "ref_year":         years[j],
                "cosine_similarity": round(1.0 - float(dist_matrix[i, j]), 4),
                "rating":           None,               # filled after Step 5
            }
            for j in top_idx
        ]

    return neighbours


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Rating Assignment
# ══════════════════════════════════════════════════════════════════════════════

def assign_rating(percentile: float) -> str:
    for lo, hi, label in PERCENTILE_BANDS:
        if lo <= percentile < hi:
            return label
    return "L1"     # fallback for edge case percentile == 100


# ══════════════════════════════════════════════════════════════════════════════
# Markdown Report
# ══════════════════════════════════════════════════════════════════════════════

def _weight_annotation(struct: str) -> dict[str, str]:
    """Return a human-readable weight annotation for each dimension."""
    weights = CAUSAL_WEIGHTS.get(struct, {})
    ann: dict[str, str] = {}

    if struct == "regulated_infrastructure":
        # Primary dims: weight × 50 % share of composite
        for d in ["timing", "competition", "market_size"]:
            eff = weights[d] * 0.50 * 100
            ann[d] = f"{eff:.1f}% of composite"
        # Residual dims: 50% / 4 each
        for d in RESIDUAL_DIMS:
            ann[d] = "12.5% of composite"
    else:
        for d in ["timing", "competition", "market_size"]:
            ann[d] = f"{weights.get(d, 0) * 100:.0f}%"
        for d in RESIDUAL_DIMS:
            ann[d] = "0% (residual)"

    return ann


def build_report(markets: list[dict], n: int = 3) -> str:
    lines: list[str] = [
        "# Vela Market Quality Rating (MQR)",
        "",
        f"*Generated: {_utc_now()}*  ",
        f"*Reference population: {len(markets)} rated markets | "
        f"Report shows first {min(n, len(markets))}*",
        "",
        "---",
        "",
    ]

    for m in markets[:n]:
        r4      = m["step4"]
        scores  = _get_scores(m)
        struct  = _get_structure(m)
        name    = m["base_profile"].get("market_name", m["id"])
        ref_yr  = m.get("ref_year", "?")
        t5_yr   = (m.get("step2", {})
                    .get("outcome_verification", {})
                    .get("t5_year", "?"))
        rating  = r4["rating"]
        struct_label = struct.replace("_", " ").title()

        # ── Header ────────────────────────────────────────────────────────────
        lines += [
            f"## {name}",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Market ID** | `{m['id']}` |",
            f"| **Market Structure** | {struct_label} |",
            f"| **Reference Year** | {ref_yr} |",
            f"| **T+5 Outcome Year** | {t5_yr} |",
            "",
        ]

        # ── Rating badge ──────────────────────────────────────────────────────
        lines += [
            f"### {rating} — {RATING_DESCRIPTIONS[rating]}",
            "",
        ]

        # ── Summary metrics ───────────────────────────────────────────────────
        lines += [
            "#### Summary Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Composite Score | **{r4['composite_score']:.1f}** / 100 "
            f"({_score_tier_label(r4['composite_score'])}) |",
            f"| Percentile (within {struct_label} group) "
            f"| **{r4['percentile_rank']:.1f}th** |",
            f"| P(outcome ≥ L3) | **{r4['p_outcome_ge_l3']:.1%}** "
            f"({r4['prob_x_of_20']}) |",
            "",
        ]

        # ── Dimension scores ──────────────────────────────────────────────────
        ann = _weight_annotation(struct)
        lines += [
            "#### Dimension Scores",
            "",
            "| Dimension | Raw Score | Tier | Weight in Composite |",
            "|-----------|-----------|------|---------------------|",
        ]
        for d in SCORE_DIMS:
            s     = scores[d]
            tier  = _score_tier_label(s)
            label = d.replace("_", " ").title()
            lines.append(f"| {label} | {s:.0f} | {tier} | {ann.get(d, '—')} |")

        if struct == "regulated_infrastructure":
            lines += [
                "",
                "> **Note (Sec 5.5 Step 2):** Composite = 50 % × primary component "
                "(T×22 % + C×28 % + S×50 %) + 50 % × mean residual 4 dimensions.",
            ]

        # ── Nearest neighbours ────────────────────────────────────────────────
        lines += [
            "",
            "#### Top 3 Nearest Neighbours  *(cosine similarity, 7-D score vector)*",
            "",
            "| Rank | Comparable Market | Ref Year | MQR | Cosine Similarity |",
            "|------|-------------------|----------|-----|-------------------|",
        ]
        for rank, nb in enumerate(r4["nearest_neighbours"], 1):
            sim_pct = f"{nb['cosine_similarity']:.3f}"
            lines.append(
                f"| {rank} | {nb['market_name']} | {nb['ref_year']} "
                f"| {nb.get('rating', '?')} | {sim_pct} |"
            )

        lines += ["", "---", ""]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Pipeline Step 4 — Rating Engine")
    print("=" * 60)
    print(f"  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"  Report : {REPORT_FILE}")
    print()

    # ── Step 0: Load ──────────────────────────────────────────────────────────
    markets = load_markets(INPUT_FILE)
    if not markets:
        print("[ERROR] No valid scored markets found. Exiting.")
        return

    # ── Step 1: Composite scores ───────────────────────────────────────────────
    print("\n[1/5] Computing composite scores...")
    for m in markets:
        m["_composite"] = compute_composite(m)

    composites = [m["_composite"] for m in markets]
    print(f"  Population composite: "
          f"min={min(composites):.1f}  "
          f"mean={np.mean(composites):.1f}  "
          f"max={max(composites):.1f}")

    # ── Step 2: Percentile ranks (within structure group) ─────────────────────
    print("\n[2/5] Computing intra-group percentile ranks...")
    percentiles = compute_percentiles(markets)
    for m in markets:
        m["_percentile"] = percentiles[m["id"]]

    # ── Step 3: Logistic regression ────────────────────────────────────────────
    print("\n[3/5] Fitting per-structure logistic regression models...")
    lr_models = fit_logistic_models(markets)

    # ── Step 4: Nearest neighbours ─────────────────────────────────────────────
    print("\n[4/5] Computing nearest neighbours (cosine, 7-D)...")
    neighbours = compute_nearest_neighbours(markets, top_k=3)

    # ── Step 5: Rating assignment & assembly ───────────────────────────────────
    print("\n[5/5] Assigning L1-L5 ratings...")

    # First pass: assign ratings so neighbour entries can reference them
    id_to_rating: dict[str, str] = {
        m["id"]: assign_rating(m["_percentile"]) for m in markets
    }

    # Backfill neighbour ratings
    for m in markets:
        for nb in neighbours[m["id"]]:
            nb["rating"] = id_to_rating.get(nb["id"], "?")

    rating_counts: dict[str, int] = {}

    for m in markets:
        rating       = id_to_rating[m["id"]]
        prob         = predict_probability(m, lr_models)
        nbs          = neighbours[m["id"]]

        m["step4"] = {
            "composite_score":        m["_composite"],
            "percentile_rank":        m["_percentile"],
            "rating":                 rating,
            "rating_description":     RATING_DESCRIPTIONS[rating],
            "p_outcome_ge_l3":        prob,
            "prob_x_of_20":           prob_to_x_of_20(prob),
            "nearest_neighbours":     nbs,
        }

        rating_counts[rating] = rating_counts.get(rating, 0) + 1

        # Remove temp keys
        del m["_composite"], m["_percentile"]

    # ── Print distribution ─────────────────────────────────────────────────────
    print("\n  Rating distribution (within-structure percentile bands):")
    total = len(markets)
    for label in ["L5", "L4", "L3", "L2", "L1"]:
        count = rating_counts.get(label, 0)
        bar   = "█" * count
        print(f"    {label}  {RATING_DESCRIPTIONS[label]:<38}  "
              f"{count:>3}  {bar}")
    print(f"    {'Total':>42}  {total:>3}")

    # ── Save final_rated_population.json ──────────────────────────────────────
    print(f"\n  Writing {OUTPUT_FILE.name}...")
    output_payload = {
        "schema_version":      "3.0",
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "total_markets":       total,
        "rating_distribution": rating_counts,
        "methodology": {
            "composite_score": (
                "Causal-weighted sum of timing, competition, market_size "
                "(structure-specific weights). For regulated_infrastructure: "
                "primary 3 dims weighted at 50% of composite + residual 4 dims "
                "equally weighted at 50% (Section 5.5 Step 2)."
            ),
            "percentile_basis": "Within same market_structure group (empirical CDF)",
            "logistic_target":  "outcome_T5 = 1 if percentile_rank >= 45 (L3+), else 0",
            "nn_metric":        "Cosine similarity on L2-normalised 7-D step3 score vector",
            "rating_bands": {
                "L5": "Top 10%  (percentile >= 90)",
                "L4": "70th – 90th percentile",
                "L3": "45th – 70th percentile  (Investment Grade Threshold)",
                "L2": "25th – 45th percentile",
                "L1": "Bottom 25%  (percentile < 25)",
            },
        },
        "markets": markets,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2, ensure_ascii=False)
    print(f"  Saved {total} markets → {OUTPUT_FILE}")

    # ── Generate sample Markdown report ───────────────────────────────────────
    print(f"\n  Writing sample report for first 3 markets → {REPORT_FILE.name}...")
    report_md = build_report(markets, n=3)
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write(report_md)
    print(f"  Report saved → {REPORT_FILE}")

    print("\n  Step 4 complete.\n")


if __name__ == "__main__":
    main()
