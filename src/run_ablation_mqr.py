#!/usr/bin/env python3
"""
run_ablation_mqr.py  --  2x4 Ablation Study: Reference Population Size vs.
                         Validation Calibration Method

Experiment matrix
-----------------
Dimension 1 — Prediction Reference (rows):
    REF-120   rated on a 120-market within-structure cohort
    REF-350   rated on a 350-market within-structure cohort

Dimension 2 — Performance Calibration (columns):
    Absolute       Direct milestone thresholds on peak_exit_usd
    Symmetric-10   Percentile bands calibrated to a 10-market validation subset
    Symmetric-60   Percentile bands calibrated to a 60-market validation subset
    Symmetric-100  Percentile bands calibrated to the full 100-market set

Inputs (must exist before running):
    data/validation_population.json   100 validation markets with t5_metrics
    data/rated_120.json               Step-4 output for 120-market predictor
    data/rated_350.json               Step-4 output for 350-market predictor

Outputs:
    Rich terminal progress display
    experiments/case_study_2x4/ablation_results.md   publication-ready table
"""

from __future__ import annotations

import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Path bootstrap so sibling modules are importable ─────────────────────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from performance_scorer import score_and_label, accuracy_summary, RATING_LABELS

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
VAL_FILE        = BASE_DIR / "data" / "validation_population.json"
RATED_120_FILE  = BASE_DIR / "data" / "rated_120.json"
RATED_350_FILE  = BASE_DIR / "data" / "rated_350.json"
OUT_DIR         = BASE_DIR / "experiments" / "case_study_2x4"
OUT_MD          = OUT_DIR / "ablation_results.md"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Percentile bands used for predicted-rating assignment (Step 4 standard) ──
_PRED_BANDS: list[tuple[float, float, str]] = [
    (90.0, 101.0, "L5"),
    (70.0,  90.0, "L4"),
    (45.0,  70.0, "L3"),
    (25.0,  45.0, "L2"),
    ( 0.0,  25.0, "L1"),
]

console = Console() if _RICH else None


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> dict | list:
    if not path.exists():
        raise FileNotFoundError(
            f"\n  Missing required file: {path}"
            f"\n  Run the appropriate pipeline step to generate it first."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_validation_markets(path: Path) -> list[dict]:
    raw = load_json(path)
    if isinstance(raw, list):
        markets = raw
    else:
        markets = raw.get("markets", raw.get("validation_markets", []))
    # Filter to markets that have both t5_metrics and composite_score
    valid = [
        m for m in markets
        if m.get("t5_metrics") is not None
        and m.get("composite_score") is not None
    ]
    if len(valid) < len(markets):
        _log(f"[warn] {len(markets) - len(valid)} validation markets skipped "
             f"(missing t5_metrics or composite_score)")
    return valid


def load_ref_markets(path: Path) -> list[dict]:
    raw = load_json(path)
    return raw.get("markets", [])


# ══════════════════════════════════════════════════════════════════════════════
# Reproducible subset generation
# ══════════════════════════════════════════════════════════════════════════════

def make_subsets(markets: list[dict]) -> dict[str, list[dict]]:
    """
    From the full validation set, derive nested subsets with seed=42.

    Strategy:
        1. Shuffle a copy of the full list with seed=42  -> base order
        2. Take first 60  -> subset_60
        3. From within subset_60, take first 10  -> subset_10
        (subset_10 is always a strict subset of subset_60)

    Each subset is a deep copy so performance_scorer mutations don't bleed
    across calibration runs.
    """
    rng = random.Random(RANDOM_SEED)
    shuffled = list(markets)
    rng.shuffle(shuffled)

    subset_60  = shuffled[:60]
    subset_10  = subset_60[:10]

    return {
        "n100": markets,      # full set (not shuffled — preserves insertion order)
        "n60":  subset_60,
        "n10":  subset_10,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Predicted-rating computation
# ══════════════════════════════════════════════════════════════════════════════

def _get_ref_structure(ref_market: dict) -> str:
    """
    Extract market_structure from a rated reference market.
    Handles both new schema (dimensions.market_structure.classification)
    and legacy schema (step3.feature_matrix.market_structure.value).
    """
    new = (ref_market
           .get("dimensions", {})
           .get("market_structure", {})
           .get("classification"))
    if new:
        return new
    legacy = (ref_market
              .get("step3", {})
              .get("feature_matrix", {})
              .get("market_structure", {})
              .get("value"))
    return legacy or "winner_take_most"


def _assign_from_pred_bands(percentile: float) -> str:
    for lo, hi, label in _PRED_BANDS:
        if lo <= percentile < hi:
            return label
    return "L1"


def predict_rating_vs_ref(
    val_market: dict,
    ref_markets: list[dict],
) -> tuple[str, float]:
    """
    Compute the predicted L1-L5 rating for a validation market by comparing
    its composite_score against the same-structure cohort in the reference
    population (empirical CDF percentile -> band assignment).

    Returns (rating_label, percentile_rank).
    Falls back to ("L3", 50.0) if no same-structure cohort found.
    """
    structure = val_market.get("market_structure", "winner_take_most")
    composite = float(val_market.get("composite_score", 50.0))

    cohort = [
        float(m["step4"]["composite_score"])
        for m in ref_markets
        if _get_ref_structure(m) == structure
        and m.get("step4", {}).get("composite_score") is not None
    ]

    if not cohort:
        return "L3", 50.0  # no cohort match — neutral fallback

    n     = len(cohort)
    below = sum(1 for c in cohort if c < composite)
    pct   = below / n * 100.0
    return _assign_from_pred_bands(pct), round(pct, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Single cell execution
# ══════════════════════════════════════════════════════════════════════════════

def run_cell(
    val_subset: list[dict],
    ref_markets: list[dict],
    bands: str,
) -> dict:
    """
    Execute one cell of the 2x4 matrix.

    Steps:
        1. Deep-copy subset so mutations don't bleed across cells.
        2. Score T+5 performance using performance_scorer.score_and_label().
        3. Re-derive predicted_rating for each market vs the reference cohort.
        4. Compute accuracy.

    Returns accuracy dict from performance_scorer.accuracy_summary().
    """
    subset = copy.deepcopy(val_subset)

    # Step 1: T+5 performance scoring (mutates actual_rating in-place)
    score_and_label(subset, bands=bands)

    # Step 2: derive predicted_rating vs reference for each market
    for m in subset:
        pred, pct = predict_rating_vs_ref(m, ref_markets)
        m["predicted_rating"]   = pred
        m["pred_percentile_vs_ref"] = pct

    # Step 3: accuracy
    return accuracy_summary(subset)


# ══════════════════════════════════════════════════════════════════════════════
# Output utilities
# ══════════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    if console:
        console.print(msg)
    else:
        print(msg)


def _cell_str(acc: dict) -> str:
    return f"Exact {acc['exact_pct']:.1f}%  Off-1 {acc['off1_pct']:.1f}%"


def print_rich_matrix(results: dict[str, dict[str, dict]]) -> None:
    """Print a Rich table of the 2x4 results to terminal."""
    if not _RICH:
        # Plain-text fallback
        col_labels = ["Absolute", "Sym-10", "Sym-60", "Sym-100"]
        row_labels = ["REF-120", "REF-350"]
        print("\n2x4 Ablation Results")
        print("-" * 80)
        header = f"{'':10}" + "".join(f"{c:^24}" for c in col_labels)
        print(header)
        for row in row_labels:
            line = f"{row:10}"
            for col in col_labels:
                acc = results[row][col]
                line += f"  E={acc['exact_pct']:.1f}% O1={acc['off1_pct']:.1f}%    "
            print(line)
        return

    table = Table(
        title="[bold]Vela MQR — 2x4 Ablation Study[/bold]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Predictor Ref", style="bold", min_width=12)
    for col in ["Absolute\n(Direct Label)", "Symmetric-10\n(n=10)", "Symmetric-60\n(n=60)", "Symmetric-100\n(n=100)"]:
        table.add_column(col, justify="center", min_width=20)

    row_keys = ["REF-120", "REF-350"]
    col_keys = ["Absolute", "Sym-10", "Sym-60", "Sym-100"]

    for row in row_keys:
        cells = []
        for col in col_keys:
            acc = results[row][col]
            e   = acc["exact_pct"]
            o1  = acc["off1_pct"]
            # Colour-code exact match: green >= 50, yellow >= 35, red < 35
            e_colour = "green" if e >= 50 else ("yellow" if e >= 35 else "red")
            cells.append(
                f"[{e_colour}]Exact {e:.1f}%[/]\n"
                f"Off-1  {o1:.1f}%\n"
                f"(n={acc['total']})"
            )
        table.add_row(row, *cells)

    console.print()
    console.print(table)
    console.print()


def build_markdown(
    results: dict[str, dict[str, dict]],
    subsets: dict[str, list],
    ts: str,
) -> str:
    row_keys = ["REF-120", "REF-350"]
    col_keys = ["Absolute", "Sym-10", "Sym-60", "Sym-100"]

    col_headers = {
        "Absolute":  "**Absolute**<br>Direct Labeling",
        "Sym-10":    "**Symmetric-10**<br>n=10 calibration",
        "Sym-60":    "**Symmetric-60**<br>n=60 calibration",
        "Sym-100":   "**Symmetric-100**<br>n=100 calibration",
    }

    lines = [
        "# Vela MQR — 2x4 Ablation Study Results",
        "",
        f"*Generated: {ts}*  ",
        f"*Random seed: {RANDOM_SEED} | "
        f"Validation set: n=100 total, "
        f"n60 subset, n10 nested subset*",
        "",
        "## Experiment Design",
        "",
        "| Dimension | Levels |",
        "|-----------|--------|",
        "| **Prediction Reference** | REF-120 (120-market cohort), REF-350 (350-market cohort) |",
        "| **Performance Calibration** | Absolute thresholds, Symmetric-10, Symmetric-60, Symmetric-100 |",
        "",
        "---",
        "",
        "## Results: Exact Match %",
        "",
        "| Predictor \\ Calibration | " + " | ".join(col_headers[c] for c in col_keys) + " |",
        "|" + "---|" * (len(col_keys) + 1),
    ]

    for row in row_keys:
        cells = []
        for col in col_keys:
            acc = results[row][col]
            cells.append(f"**{acc['exact_pct']:.1f}%** (n={acc['total']})")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Results: Within ±1 Band %",
        "",
        "| Predictor \\ Calibration | " + " | ".join(col_headers[c] for c in col_keys) + " |",
        "|" + "---|" * (len(col_keys) + 1),
    ]

    for row in row_keys:
        cells = []
        for col in col_keys:
            acc = results[row][col]
            cells.append(f"**{acc['off1_pct']:.1f}%** (n={acc['total']})")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "---",
        "",
        "## Full Cell Detail",
        "",
        "| Predictor | Calibration | n | Exact | Off-1 | Miss | Exact % | Off-1 % | Miss % |",
        "|-----------|-------------|---|-------|-------|------|---------|---------|--------|",
    ]

    for row in row_keys:
        for col in col_keys:
            acc = results[row][col]
            lines.append(
                f"| {row} | {col} | {acc['total']} "
                f"| {acc['exact']} | {acc['off1']} | {acc['miss']} "
                f"| {acc['exact_pct']:.1f}% | {acc['off1_pct']:.1f}% | {acc['miss_pct']:.1f}% |"
            )

    lines += [
        "",
        "---",
        "",
        "## Methodology Notes",
        "",
        "- **Predicted rating**: composite_score of each validation market compared "
        "against same-structure cohort in the predictor reference population "
        "(empirical CDF percentile → L1-L5 via standard bands: "
        "L5≥90th, L4 70-90th, L3 45-70th, L2 25-45th, L1<25th).",
        "- **Absolute calibration**: L1-L5 from peak_exit_usd hard thresholds "
        "(L5≥$10B, L4≥$1B, L3≥$100M, L2≥$10M, L1<$10M).",
        "- **Symmetric calibration**: power-law log10 transform on peak_exit_usd "
        "and total_funding_usd + population-level min-max scaling → "
        "actual_performance_score → within-cohort percentile rank → L1-L5 "
        "(bands: L5≥90th, L4 70-90th, L3 45-70th, L2 20-45th, L1<20th).",
        "- **Nested subsets** drawn with random.seed(42): n60 is the first 60 "
        "of a seeded shuffle of the full 100; n10 is the first 10 of n60.",
        "- Exact Match: predicted_rating == actual_rating.",
        "- Off-1: |band_distance| <= 1.",
        "",
        f"*Script: `src/run_ablation_mqr.py` | "
        f"Generated by Claude Sonnet 4.6*",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if _RICH:
        console.print(Panel(
            "[bold cyan]Vela MQR — 2x4 Ablation Study[/bold cyan]\n"
            "Prediction Reference Size  x  Validation Calibration Method",
            expand=False,
        ))

    # ── Load data ──────────────────────────────────────────────────────────────
    _log("\n[1/4] Loading data files...")

    val_markets = load_validation_markets(VAL_FILE)
    _log(f"  Validation markets loaded: {len(val_markets)}")

    ref_120 = load_ref_markets(RATED_120_FILE)
    ref_350 = load_ref_markets(RATED_350_FILE)
    _log(f"  REF-120: {len(ref_120)} markets  |  REF-350: {len(ref_350)} markets")

    if len(val_markets) < 10:
        _log("[ERROR] Need at least 10 validation markets. Exiting.")
        sys.exit(1)

    # ── Build subsets ──────────────────────────────────────────────────────────
    _log("\n[2/4] Building reproducible subsets (seed=42)...")
    subsets = make_subsets(val_markets)
    _log(f"  n100={len(subsets['n100'])}  n60={len(subsets['n60'])}  n10={len(subsets['n10'])}")

    # ── Define experiment matrix ───────────────────────────────────────────────
    # Each calibration mode: (column_label, bands_arg, subset_key)
    calibrations: list[tuple[str, str, str]] = [
        ("Absolute",  "absolute",  "n100"),   # subset doesn't matter for absolute
        ("Sym-10",    "symmetric", "n10"),
        ("Sym-60",    "symmetric", "n60"),
        ("Sym-100",   "symmetric", "n100"),
    ]

    predictors: list[tuple[str, list[dict]]] = [
        ("REF-120", ref_120),
        ("REF-350", ref_350),
    ]

    # ── Execute 8 cells ────────────────────────────────────────────────────────
    _log("\n[3/4] Running 8 experiment cells...")

    results: dict[str, dict[str, dict]] = {}
    total_cells = len(predictors) * len(calibrations)
    cell_num = 0

    for pred_label, ref_markets in predictors:
        results[pred_label] = {}
        for col_label, bands, subset_key in calibrations:
            cell_num += 1
            val_subset = subsets[subset_key]
            _log(f"  [{cell_num}/{total_cells}] {pred_label} x {col_label}  "
                 f"(n={len(val_subset)}, bands={bands!r})")

            acc = run_cell(val_subset, ref_markets, bands)
            results[pred_label][col_label] = acc

            _log(f"         -> Exact {acc['exact_pct']:.1f}%  "
                 f"Off-1 {acc['off1_pct']:.1f}%  "
                 f"Miss {acc['miss_pct']:.1f}%")

    # ── Display results table ──────────────────────────────────────────────────
    _log("\n[4/4] Results")
    print_rich_matrix(results)

    # ── Write Markdown artifact ────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = build_markdown(results, subsets, ts)

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(md)
    _log(f"  Markdown report saved -> {OUT_MD}")


if __name__ == "__main__":
    main()
