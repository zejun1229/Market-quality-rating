"""
Vela MQR — Mirror Validation Pipeline
======================================
Generates a 60-market validation set (6 batches of 10) to compare T=0
predictions against actual T+5 business outcomes using a Symmetrical
Percentile-Based Labeling method.

Architecture
------------
Phase 1  (T=0 Prediction)
  Role 0  — Gemini web-search generates 10 distinct seeds per batch.
             Entry years strictly between 2009 and 2020 (i.e. 2010-2019).
             Seeds must NOT appear in reference_population_master.json.
             20-second timeout; timed-out batches are retried.

  Role 1  — Claude generates market_name, canonical market_structure type,
             7 integer dimension scores (0-100) and rationale, all anchored
             strictly to T=0 entry-year information only.

  Step 4  — T=0 predicted_rating assigned by comparing the new market's
             composite score against the reference population's same-structure
             cohort (final_rated_population.json).

Phase 2  (T+5 Ground Truth)
  Role 2  — Gemini web-search retrieves 4 numerical T+5 metrics per market:
               peak_exit_value           (USD float)
               top_3_aggregate_valuation (USD float)
               unicorn_count             (integer)
               capital_efficiency_ratio  (VFR float)
             20-second timeout; timed-out markets are blacklisted and skipped.

Phase 3  (Symmetrical Scoring & Labeling)
  Normalize metrics -> actual_performance_score (0-100):
     40% peak_exit_value
     30% top_3_aggregate_valuation
     20% unicorn_count
     10% capital_efficiency_ratio
  After each batch: re-rank all markets by actual_performance_score and
  assign actual_rating using symmetrical percentile bands:
     Top 10% = L5 | 70-90th = L4 | 45-70th = L3 | 20-45th = L2 | <20th = L1
  Results saved progressively to validation_population.json.

Phase 4  (Accuracy)
  Compute delta: predicted_rating vs actual_rating.
  Output Exact Match % and Off-by-1 % in terminal and final report.
"""

import asyncio
import difflib
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Third-party imports ───────────────────────────────────────────────────────
try:
    import anthropic
except ImportError:
    sys.exit("ERROR: anthropic not found.  pip install anthropic")

try:
    from google import genai  # noqa: F401
except ImportError:
    sys.exit("ERROR: google-genai not found.  pip install google-genai")

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    sys.exit("ERROR: rich not found.  pip install rich")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy not found.  pip install numpy")

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline_step1 import DIMENSIONS
from pipeline_step2 import get_gemini_client, query_gemini_grounded

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_COUNT         = 60
BATCH_SIZE           = 10
NUM_BATCHES          = TARGET_COUNT // BATCH_SIZE          # 6
REF_YEARS            = list(range(2010, 2020))             # 2010-2019 inclusive
CLAUDE_MODEL         = "claude-sonnet-4-6"
GEMINI_TIMEOUT_SECS  = 20.0   # hard timeout for Phase 2 T+5 searches
SEED_TIMEOUT_SECS    = 60.0   # timeout for Role 0 seed generation (batch call)
DEDUP_THRESHOLD      = 0.75
MAX_WORKERS          = BATCH_SIZE

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(_ROOT)
MASTER_JSON = ROOT / "reference_population_master.json"
RATED_JSON  = ROOT / "final_rated_population.json"
VAL_JSON    = ROOT / "validation_population.json"
LOG_PATH    = ROOT / "lab_notes" / "Validation_Run_Log.md"

# ── Step 4 rating machinery ───────────────────────────────────────────────────
SCORE_DIMS = [
    "timing", "competition", "market_size", "customer_readiness",
    "regulatory", "infrastructure", "market_structure",
]

CAUSAL_WEIGHTS: dict = {
    "winner_take_most":         {"timing": 0.32, "competition": 0.52, "market_size": 0.16},
    "platform_two_sided":       {"timing": 0.55, "competition": 0.22, "market_size": 0.23},
    "technology_enablement":    {"timing": 0.60, "competition": 0.18, "market_size": 0.22},
    "fragmented_niche":         {"timing": 0.25, "competition": 0.15, "market_size": 0.60},
    "regulated_infrastructure": {"timing": 0.22, "competition": 0.28, "market_size": 0.50},
}
RESIDUAL_DIMS = ["customer_readiness", "regulatory", "infrastructure", "market_structure"]
VALID_STRUCTURES = list(CAUSAL_WEIGHTS.keys())

# Predicted rating bands (matches reference population Step 4)
PRED_BANDS: list = [
    (90.0, 101.0, "L5"),
    (70.0,  90.0, "L4"),
    (45.0,  70.0, "L3"),
    (25.0,  45.0, "L2"),
    ( 0.0,  25.0, "L1"),
]

# Actual rating bands (symmetrical, user-specified)
ACTUAL_BANDS: list = [
    (90.0, 101.0, "L5"),   # top 10%
    (70.0,  90.0, "L4"),   # 70th-90th
    (45.0,  70.0, "L3"),   # 45th-70th
    (20.0,  45.0, "L2"),   # 20th-45th
    ( 0.0,  20.0, "L1"),   # bottom 20%
]

RATING_LABELS = {
    "L5": "Ideal", "L4": "Attractive", "L3": "Viable",
    "L2": "Headwinds", "L1": "Hostile",
}
RATING_NUM = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}

# ── Rich console ──────────────────────────────────────────────────────────────
console = Console(highlight=False)

# ── Live display state (all mutations guarded by _status_lock) ────────────────
_status_lock   = threading.Lock()
_worker_status: dict = {}    # market_id -> {"stage": str, "name": str}
_recent_done:   list = []    # last 6 completions
_dedup_skipped  = 0
_blacklist_hits = 0
_t5_failures    = 0


def _set_stage(market_id: str, stage: str, name: str = "") -> None:
    with _status_lock:
        _worker_status[market_id] = {"stage": stage, "name": name}


def _clear_stage(market_id: str, completion: dict | None = None) -> None:
    global _recent_done
    with _status_lock:
        _worker_status.pop(market_id, None)
        if completion:
            _recent_done = ([completion] + _recent_done)[:6]


# ── Display ───────────────────────────────────────────────────────────────────

def _compute_accuracy(val_markets: list) -> dict:
    completed = [
        m for m in val_markets
        if m.get("predicted_rating") and m.get("actual_rating")
    ]
    if not completed:
        return {"exact": 0, "off1": 0, "total": 0, "exact_pct": 0, "off1_pct": 0}
    exact = sum(1 for m in completed if m["predicted_rating"] == m["actual_rating"])
    off1  = sum(
        1 for m in completed
        if abs(RATING_NUM[m["predicted_rating"]] - RATING_NUM[m["actual_rating"]]) <= 1
    )
    total = len(completed)
    return {
        "exact": exact, "off1": off1, "total": total,
        "exact_pct": round(exact / total * 100) if total else 0,
        "off1_pct":  round(off1  / total * 100) if total else 0,
    }


def make_display(total: int, batch_num: int, blacklist: set, val_markets: list) -> Panel:
    t = Table(box=box.SIMPLE, expand=True, show_header=False, padding=(0, 1))
    t.add_column("Label", style="bold cyan", no_wrap=True, width=18)
    t.add_column("Value", style="white")

    pct    = total / TARGET_COUNT * 100 if TARGET_COUNT else 0
    filled = int(pct / 5)
    t.add_row(
        "Progress",
        f"[green]{'#' * filled}[/][dim]{'.' * (20 - filled)}[/]  "
        f"[bold]{total}[/] / {TARGET_COUNT}  ([cyan]{pct:.0f}%[/])  Batch {batch_num}",
    )
    t.add_row(
        "Flags",
        f"Blacklisted: [red]{len(blacklist)}[/]   "
        f"Deduped: [yellow]{_dedup_skipped}[/]   "
        f"T5-fail: [yellow]{_t5_failures}[/]",
    )

    # Accuracy
    acc = _compute_accuracy(val_markets)
    if acc["total"] > 0:
        t.add_row(
            "Accuracy",
            f"Exact: [green]{acc['exact']}/{acc['total']} ({acc['exact_pct']}%)[/]   "
            f"Within-1: [cyan]{acc['off1']}/{acc['total']} ({acc['off1_pct']}%)[/]",
        )

    # Active workers
    with _status_lock:
        active = dict(_worker_status)
        recent = list(_recent_done)

    if active:
        t.add_row("", "")
        t.add_row("[bold]Active[/]", "")
        stage_colour = {
            "Role 0": "magenta", "Role 1": "blue",
            "Phase 2": "yellow", "T=0": "cyan",
        }
        for mid, info in sorted(active.items()):
            stage  = info.get("stage", "?")
            name   = info.get("name", "")[:46]
            key    = stage.split(":")[0].strip().split(" ")[0:2]
            colour = stage_colour.get(" ".join(key), "white")
            t.add_row(
                f"  [dim]{mid}[/]",
                f"[{colour}][{stage}][/{colour}]  {name}",
            )

    if recent:
        t.add_row("", "")
        t.add_row("[bold]Completed[/]", "")
        for r in recent:
            pred   = r.get("predicted", "?")
            actual = r.get("actual") or "--"
            match  = ""
            if r.get("actual"):
                if pred == actual:
                    match = " [green]EXACT[/]"
                elif abs(RATING_NUM.get(pred, 0) - RATING_NUM.get(actual, 0)) <= 1:
                    match = " [cyan]OFF-1[/]"
                else:
                    match = " [red]MISS[/]"
            t.add_row(
                f"  [dim]{r.get('id','?')}[/]",
                f"[green]+[/] {r.get('name','')[:44]}  "
                f"pred=[bold]{pred}[/] act={actual}{match}",
            )

    return Panel(
        t,
        title="[bold cyan]VELA MQR -- Mirror Validation Pipeline  "
              "(60 markets | 6 batches | Symmetrical Percentile Labels)[/]",
        border_style="cyan",
    )


# ── Client helpers ────────────────────────────────────────────────────────────

def get_claude_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your_"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=key)


# ── Dedup helpers ─────────────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _is_duplicate(name: str, registry: set, threshold: float = DEDUP_THRESHOLD) -> bool:
    norm = _normalise(name)
    for existing in registry:
        if difflib.SequenceMatcher(None, norm, existing).ratio() >= threshold:
            return True
    return False


def _registry_block(registry: set, limit: int = 80) -> str:
    items = sorted(registry)[:limit]
    return "\n".join(f"  - {n}" for n in items) if items else "  (none yet)"


# ── Gemini with asyncio timeout ───────────────────────────────────────────────

def _gemini_with_timeout(
    gemini_client,
    prompt: str,
    timeout_secs: float = GEMINI_TIMEOUT_SECS,
) -> tuple[str, list]:
    """
    Run query_gemini_grounded wrapped in asyncio.wait_for.
    Raises asyncio.TimeoutError on timeout.
    """
    async def _inner():
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, query_gemini_grounded, gemini_client, prompt),
            timeout=timeout_secs,
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_inner())
    finally:
        loop.close()


# ── Rating engine helpers ─────────────────────────────────────────────────────

def assign_rating(percentile: float, bands: list = PRED_BANDS) -> str:
    for lo, hi, label in bands:
        if lo <= percentile < hi:
            return label
    return "L1"


def compute_composite(scores: dict, structure: str) -> float:
    weights = CAUSAL_WEIGHTS.get(structure, CAUSAL_WEIGHTS["technology_enablement"])
    primary = (
        weights["timing"]       * scores.get("timing", 50)
        + weights["competition"]  * scores.get("competition", 50)
        + weights["market_size"]  * scores.get("market_size", 50)
    )
    if structure == "regulated_infrastructure":
        residual_vals = [float(scores.get(d, 50)) for d in RESIDUAL_DIMS]
        residual = sum(residual_vals) / len(residual_vals)
        return round(0.50 * primary + 0.50 * residual, 4)
    return round(primary, 4)


def load_reference_cohort_map() -> dict:
    """
    Load reference population composite scores grouped by structure type.
    Returns {structure_str: [composite_score, ...]}
    """
    if not RATED_JSON.exists():
        console.log(f"  [WARN] {RATED_JSON.name} not found — using empty reference cohort")
        return {}
    with open(RATED_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    cohort_map: dict = {}
    for m in data.get("markets", []):
        try:
            struct = m["step3"]["feature_matrix"]["market_structure"]["value"]
            comp   = float(m["step4"]["composite_score"])
            cohort_map.setdefault(struct, []).append(comp)
        except (KeyError, TypeError):
            pass
    total = sum(len(v) for v in cohort_map.values())
    console.log(
        f"  Reference cohort loaded: {total} markets | "
        + " | ".join(f"{s}={len(v)}" for s, v in cohort_map.items())
    )
    return cohort_map


def predict_rating_vs_reference(
    composite: float,
    structure: str,
    ref_cohort_map: dict,
) -> tuple[str, float]:
    """
    Predict L1-L5 by comparing composite against reference population cohort.
    percentile = fraction of same-structure reference markets below new composite.
    Falls back to all-structure pool if no same-structure markets exist.
    """
    cohort = ref_cohort_map.get(structure, [])
    if not cohort:
        cohort = [c for vals in ref_cohort_map.values() for c in vals]
    if not cohort:
        return "L3", 50.0
    n      = len(cohort)
    below  = sum(1 for c in cohort if c < composite)
    pct    = below / n * 100.0
    return assign_rating(pct, PRED_BANDS), round(pct, 2)


# ── T+5 metrics normalization ─────────────────────────────────────────────────

def normalize_t5_metrics(metrics: dict) -> float:
    """
    Normalize 4 T+5 metrics to actual_performance_score (0-100).
    Weights: peak_exit_value=40%, top_3_aggregate=30%, unicorn_count=20%, cap_eff=10%

    Normalization anchors:
      peak_exit_value           log10 scale: $1M=0, $10B=100
      top_3_aggregate_valuation log10 scale: $1M=0, $30B=100
      unicorn_count             linear: 0=0, 5+=100
      capital_efficiency_ratio  linear: 0=0, 20x+=100
    """
    def _log_norm(v: float | None, log_scale: float) -> float:
        if not v or float(v) <= 0:
            return 0.0
        return min(100.0, math.log10(max(float(v), 1e6) / 1e6) / log_scale * 100.0)

    pev = _log_norm(metrics.get("peak_exit_value"),           log_scale=4.0)
    t3a = _log_norm(metrics.get("top_3_aggregate_valuation"), log_scale=4.5)
    uc  = min(100.0, max(0.0, float(metrics.get("unicorn_count") or 0) / 5.0 * 100.0))
    cer = min(100.0, max(0.0, float(metrics.get("capital_efficiency_ratio") or 0) / 20.0 * 100.0))

    return round(0.40 * pev + 0.30 * t3a + 0.20 * uc + 0.10 * cer, 2)


def reassign_actual_ratings(val_markets: list) -> None:
    """
    Re-rank all markets with an actual_performance_score by that score and
    assign actual_rating using the symmetrical percentile bands.
    Called after every batch completes.
    """
    scored = [m for m in val_markets if m.get("actual_performance_score") is not None]
    n = len(scored)
    if n == 0:
        return
    sorted_scored = sorted(scored, key=lambda m: m["actual_performance_score"])
    for rank, m in enumerate(sorted_scored):
        pct = (rank / (n - 1) * 100.0) if n > 1 else 50.0
        m["actual_percentile"] = round(pct, 2)
        m["actual_rating"]     = assign_rating(pct, ACTUAL_BANDS)
    for m in val_markets:
        if m.get("actual_performance_score") is None:
            m.setdefault("actual_rating",    None)
            m.setdefault("actual_percentile", None)


# ── Persistence ───────────────────────────────────────────────────────────────

def load_validation() -> dict:
    if VAL_JSON.exists():
        with open(VAL_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "schema_version": "1.0",
        "metadata": {
            "target":            TARGET_COUNT,
            "started":           datetime.now().isoformat(),
            "last_updated":      "",
            "batches_completed": 0,
            "blacklist":         [],
        },
        "markets": [],
    }


def save_validation(data: dict, val_markets: list, blacklist: set, batch_num: int) -> None:
    data["markets"] = val_markets
    data["metadata"]["last_updated"]      = datetime.now().isoformat()
    data["metadata"]["blacklist"]         = sorted(blacklist)
    data["metadata"]["batches_completed"] = batch_num
    with open(VAL_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ── Role 0: Gemini seed generation ───────────────────────────────────────────

def role0_generate_seeds(
    gemini_client,
    registry: set,
    blacklist: set,
    count: int,
) -> list:
    """
    Role 0: Gemini + Google Search generates 'count' market seeds.
    Entry years strictly between 2009 and 2020 (i.e. 2010-2019).
    Returns list of {domain, ref_year, knowledge_brief} dicts.
    """
    console.log(f"  [Role 0] Searching for {count} seeds (ref years 2010-2019) ...")

    prompt = (
        f"You are a venture capital research assistant. Search the web and identify "
        f"exactly {count} real venture capital markets for a validation study.\n\n"
        "REQUIREMENTS:\n"
        "- The market's major founding event or funding inflection must have occurred "
        "between 2010 and 2019 INCLUSIVE\n"
        "- Must be verifiably mainstream markets with abundant data: Crunchbase records, "
        "SEC filings, analyst reports, IPO filings, press coverage\n"
        "- Must span diverse sectors: fintech, healthtech, SaaS, consumer internet, "
        "logistics, climate, edtech, marketplace, deeptech, hardware, etc.\n"
        "- Must be distinct: no two markets from the same sub-category\n"
        "- By T+5 (ref_year + 5), the outcome must be clearly determinable from "
        "public records (valuations, exits, unicorn counts)\n\n"
        "DO NOT suggest any market conceptually identical to:\n"
        f"{_registry_block(registry)}\n\n"
        "DO NOT suggest any market from this blacklist:\n"
        + ("\n".join(f"  - {b}" for b in sorted(blacklist)[:30]) if blacklist else "  (none)")
        + "\n\n"
        "For each market return:\n"
        "1. domain: precise 10-20 word description of the product/service category\n"
        "2. ref_year: integer year of the key founding or funding inflection (2010-2019)\n"
        "3. knowledge_brief: 3-5 sentences with SPECIFIC verifiable facts — "
        "company names, funding round sizes, user/revenue metrics, "
        "key enabling technology, and market state at that reference year\n\n"
        "Return ONLY a raw valid JSON array (no markdown, no preamble):\n"
        "[\n"
        '  {"domain": "...", "ref_year": <int>, "knowledge_brief": "..."},\n'
        "  ...\n"
        "]\n"
        f"Return exactly {count} entries."
    )

    try:
        text, _ = _gemini_with_timeout(gemini_client, prompt, timeout_secs=SEED_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        console.log("  [Role 0] Seed generation TIMED OUT")
        return []
    except Exception as exc:
        console.log(f"  [Role 0] ERROR: {exc}")
        return []

    clean = text.strip()
    if "```" in clean:
        clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
    bracket = clean.find("[")
    if bracket > 0:
        clean = clean[bracket:]

    try:
        suggestions = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as exc:
        console.log(f"  [Role 0] Parse error: {exc}  raw[:200]={clean[:200]!r}")
        return []

    valid: list = []
    seen  = set(registry)
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        domain  = s.get("domain", "")
        ref_yr  = s.get("ref_year")
        brief   = s.get("knowledge_brief", "")
        try:
            ref_yr = int(ref_yr)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(domain, str) and len(domain) > 5
            and ref_yr in REF_YEARS
            and isinstance(brief, str) and len(brief) > 20
            and not _is_duplicate(domain, seen)
        ):
            valid.append({"domain": domain, "ref_year": ref_yr, "knowledge_brief": brief})
            seen.add(_normalise(domain))

    console.log(f"  [Role 0] {len(valid)} valid seeds / {len(suggestions)} suggested")
    return valid[:count]


# ── Role 1: Claude market profile + T=0 scores ───────────────────────────────

def role1_generate_market(
    claude_client: anthropic.Anthropic,
    seed: dict,
    registry: set,
) -> tuple[dict | None, str | None]:
    """
    Role 1: Single Claude call generating:
      - market_name  (4-8 words, no year)
      - market_structure  (canonical enum)
      - dimension_scores  (7 integers 0-100, based ONLY on T=0 info)
      - rationale  (1-sentence per dimension)

    Returns (result_dict, error_str_or_None).
    """
    structure_opts = "|".join(VALID_STRUCTURES)
    dim_lines = []
    for d in DIMENSIONS:
        dim_lines.append(
            f"  {d['name']}: {d['description'][:100]}"
        )
    dim_guide = "\n".join(dim_lines)

    avoidance = _registry_block(registry)

    prompt = (
        "You are a senior venture capital market analyst performing a strict T=0 assessment.\n\n"
        f"Reference year: {seed['ref_year']}  "
        f"(T=0 — you MUST NOT use any knowledge of events after {seed['ref_year']})\n\n"
        "=== MARKET SEED ===\n"
        f"Domain: {seed['domain']}\n"
        f"Knowledge brief: {seed['knowledge_brief']}\n\n"
        "=== DEDUPLICATION CONSTRAINT ===\n"
        "DO NOT generate a market identical or near-identical to:\n"
        f"{avoidance}\n\n"
        "=== SCORING SCALE ===\n"
        "  0-24  = Hostile  |  25-44 = Headwinds  |  45-69 = Viable\n"
        "  70-89 = Attractive  |  90-100 = Ideal\n\n"
        "=== DIMENSION REFERENCE ===\n"
        f"{dim_guide}\n\n"
        "=== TASK ===\n"
        "Generate a T=0 assessment. Return ONLY a raw valid JSON object:\n"
        "{\n"
        '  "market_name": "<concise 4-8 word name, no year suffix>",\n'
        f'  "market_structure": "<{structure_opts}>",\n'
        '  "scores": {\n'
        '    "timing": <int 0-100>,\n'
        '    "competition": <int 0-100>,\n'
        '    "market_size": <int 0-100>,\n'
        '    "customer_readiness": <int 0-100>,\n'
        '    "regulatory": <int 0-100>,\n'
        '    "infrastructure": <int 0-100>,\n'
        '    "market_structure": <int 0-100>\n'
        "  },\n"
        '  "rationale": {\n'
        '    "timing": "<1-sentence T=0 evidence>",\n'
        '    "competition": "<1-sentence T=0 evidence>",\n'
        '    "market_size": "<1-sentence T=0 evidence>",\n'
        '    "customer_readiness": "<1-sentence T=0 evidence>",\n'
        '    "regulatory": "<1-sentence T=0 evidence>",\n'
        '    "infrastructure": "<1-sentence T=0 evidence>",\n'
        '    "market_structure": "<1-sentence T=0 evidence>"\n'
        "  }\n"
        "}"
    )

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    except Exception as exc:
        return None, f"Claude API error: {exc}"

    # Strip code fences
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}  raw[:120]={raw[:120]!r}"

    # Extract and validate
    market_name = str(parsed.get("market_name", seed["domain"][:60])).strip()
    structure   = str(parsed.get("market_structure", "technology_enablement")).strip()
    if structure not in VALID_STRUCTURES:
        structure = "technology_enablement"

    raw_scores = parsed.get("scores", {})
    scores: dict = {}
    for d in SCORE_DIMS:
        v = raw_scores.get(d, 50)
        try:
            scores[d] = max(0, min(100, int(v)))
        except (TypeError, ValueError):
            scores[d] = 50

    return {
        "market_name":     market_name,
        "market_structure": structure,
        "scores":          scores,
        "rationale":       parsed.get("rationale", {}),
    }, None


# ── Phase 2: Gemini T+5 ground truth search ───────────────────────────────────

def phase2_t5_ground_truth(
    gemini_client,
    market_name: str,
    ref_year: int,
) -> tuple[dict, list]:
    """
    Phase 2: Gemini web-search for T+5 (ref_year+5) business outcomes.
    Returns (metrics_dict, source_urls).
    Raises asyncio.TimeoutError if the search exceeds GEMINI_TIMEOUT_SECS.
    """
    t5_year = ref_year + 5

    prompt = (
        "You are a venture capital outcome researcher. Search the web for actual "
        "business outcomes for the following market category at the T+5 measurement date.\n\n"
        f"Market Category: {market_name}\n"
        f"Entry Year (T=0): {ref_year}\n"
        f"Measurement Date (T+5): {t5_year}\n\n"
        "Search for: IPO filings, M&A transactions, Crunchbase funding records, "
        f"analyst reports, and press coverage dated on or before {t5_year}.\n\n"
        "Return ONLY a raw valid JSON object (no markdown, no preamble):\n"
        "{\n"
        f'  "peak_exit_value": <highest single USD valuation achieved by any company in '
        f'this market by {t5_year} — IPO market cap, M&A deal price, or late-stage '
        f'round at $1B+ valuation — as a float, e.g. 2500000000.0 for $2.5B, or null>,\n'
        f'  "top_3_aggregate_valuation": <combined USD valuation of the top 3 category '
        f'companies at {t5_year} as a float, or null if data unavailable>,\n'
        f'  "unicorn_count": <integer count of companies specifically in this market '
        f'category reaching $1B+ valuation by {t5_year}>,\n'
        f'  "capital_efficiency_ratio": <average valuation-to-total-funding ratio for '
        f'category leaders — e.g. 8.5 means $850M valuation on $100M raised — as a float, '
        f'or null if data unavailable>,\n'
        '  "search_notes": "<2-3 sentences: key evidence found, company names, '
        'specific dollar figures>"\n'
        "}\n\n"
        "Guidelines:\n"
        "  - peak_exit_value: the SINGLE highest individual exit or valuation event\n"
        "  - top_3_aggregate_valuation: SUM of valuations of top 3 players (not total market size)\n"
        "  - unicorn_count: use 0 if confirmed no unicorns; use null only if genuinely unknown\n"
        "  - capital_efficiency_ratio: estimate from revenue multiples if direct VFR unavailable\n"
        "Return the JSON object only."
    )

    # Raises asyncio.TimeoutError if exceeded
    text, urls = _gemini_with_timeout(gemini_client, prompt, timeout_secs=GEMINI_TIMEOUT_SECS)

    clean = text.strip()
    if "```" in clean:
        clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
    brace = clean.find("{")
    if brace > 0:
        clean = clean[brace:]

    try:
        metrics = json.loads(clean)
    except json.JSONDecodeError:
        metrics = {
            "peak_exit_value":           None,
            "top_3_aggregate_valuation": None,
            "unicorn_count":             0,
            "capital_efficiency_ratio":  None,
            "search_notes":              f"Parse error. Raw: {text[:200]}",
        }

    return metrics, urls


# ── Per-market worker ─────────────────────────────────────────────────────────

def _process_market_worker(
    seed: dict,
    market_id: str,
    registry: set,
    blacklist: set,
    ref_cohort_map: dict,
    claude_client: anthropic.Anthropic,
    gemini_client,
) -> tuple[dict | None, str, bool]:
    """
    Full pipeline for one market:
      Role 1 -> T=0 composite + predicted_rating -> Phase 2 T+5 -> normalize

    Returns (val_market_dict, market_name, timed_out_bool).
    val_market_dict is None on failure, dedup skip, or timeout.
    timed_out_bool=True only on Phase 2 timeout (caller adds to blacklist).
    """
    global _dedup_skipped, _t5_failures

    # ── Role 1 ────────────────────────────────────────────────────────────────
    _set_stage(market_id, "Role 1: Scoring...", seed["domain"][:46])
    result, err = role1_generate_market(claude_client, seed, registry)
    if result is None:
        _clear_stage(market_id)
        console.log(f"  [{market_id}] Role 1 FAILED: {err}")
        return None, "", False

    market_name = result["market_name"]
    structure   = result["market_structure"]
    scores      = result["scores"]

    # Python-level dedup check
    if _is_duplicate(market_name, registry):
        _clear_stage(market_id)
        with _status_lock:
            _dedup_skipped += 1
        console.log(f"  [{market_id}] DEDUP SKIP: {market_name!r}")
        return None, market_name, False

    # ── T=0 predicted rating ──────────────────────────────────────────────────
    composite = compute_composite(scores, structure)
    predicted_rating, pred_pct = predict_rating_vs_reference(
        composite, structure, ref_cohort_map
    )
    _set_stage(market_id, "Phase 2: T+5 Search", market_name[:46])
    console.log(
        f"  [{market_id}] T=0: {market_name!r}  struct={structure}  "
        f"composite={composite:.1f}  pred={predicted_rating} (pct={pred_pct:.1f})"
    )

    # ── Phase 2: T+5 ground truth ─────────────────────────────────────────────
    t5_metrics: dict | None = None
    t5_urls: list = []

    try:
        t5_metrics, t5_urls = phase2_t5_ground_truth(
            gemini_client, market_name, seed["ref_year"]
        )
    except asyncio.TimeoutError:
        _clear_stage(market_id)
        console.log(
            f"  [{market_id}] Phase 2 TIMEOUT ({GEMINI_TIMEOUT_SECS}s): "
            f"{market_name!r} -> blacklisted"
        )
        return None, market_name, True   # caller adds to blacklist
    except Exception as exc:
        console.log(f"  [{market_id}] Phase 2 ERROR (non-fatal): {exc}")
        with _status_lock:
            _t5_failures += 1
        t5_metrics = {
            "peak_exit_value": None, "top_3_aggregate_valuation": None,
            "unicorn_count": 0, "capital_efficiency_ratio": None,
            "search_notes": f"Error: {exc}",
        }

    # ── Normalize T+5 → actual_performance_score ──────────────────────────────
    actual_performance_score = normalize_t5_metrics(t5_metrics or {})
    console.log(
        f"  [{market_id}] Phase 2 done: "
        f"pev={t5_metrics.get('peak_exit_value')!r}  "
        f"uc={t5_metrics.get('unicorn_count')}  "
        f"perf_score={actual_performance_score:.1f}"
    )

    # ── Assemble validation record ────────────────────────────────────────────
    val_market = {
        "id":                      market_id,
        "name":                    market_name,
        "domain":                  seed["domain"],
        "ref_year":                seed["ref_year"],
        "t5_year":                 seed["ref_year"] + 5,
        "market_structure":        structure,
        "dimension_scores":        scores,
        "score_rationale":         result.get("rationale", {}),
        "composite_score":         composite,
        "pred_percentile_vs_ref":  pred_pct,
        "predicted_rating":        predicted_rating,
        "t5_metrics":              t5_metrics,
        "t5_source_count":         len(t5_urls),
        "actual_performance_score": actual_performance_score,
        "actual_rating":           None,    # assigned by reassign_actual_ratings
        "actual_percentile":       None,
        "processed_at":            datetime.now().isoformat(),
    }

    _clear_stage(
        market_id,
        completion={
            "id":        market_id,
            "name":      market_name,
            "predicted": predicted_rating,
            "actual":    None,
        },
    )
    return val_market, market_name, False


# ── Batch log ─────────────────────────────────────────────────────────────────

def append_batch_log(
    batch_num: int,
    batch_markets: list,
    total_processed: int,
    run_ts: str,
    acc: dict,
    blacklist: set,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        "---",
        "",
        f"## Batch {batch_num}  --  {run_ts}",
        "",
        f"**Markets this batch:** {len(batch_markets)}  ",
        f"**Cumulative total:** {total_processed} / {TARGET_COUNT}  ",
        f"**Blacklisted:** {len(blacklist)}  ",
        f"**Running accuracy:**  "
        f"Exact={acc['exact']}/{acc['total']} ({acc['exact_pct']}%)  "
        f"Within-1={acc['off1']}/{acc['total']} ({acc['off1_pct']}%)",
        "",
        "| # | Market | Struct | Comp | Pred | Perf | Actual |",
        "|---|--------|--------|------|------|------|--------|",
    ]
    for m in batch_markets:
        lines.append(
            f"| {m['id']} | {m['name'][:48]} | {m['market_structure'][:12]} "
            f"| {m['composite_score']:.1f} | {m['predicted_rating']} "
            f"| {m['actual_performance_score']:.1f} | {m.get('actual_rating') or '--'} |"
        )
    lines.append("")

    header_needed = not LOG_PATH.exists()
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        if header_needed:
            fh.write(
                "# Vela MQR -- Mirror Validation Pipeline Run Log\n\n"
                f"Target: {TARGET_COUNT} | Batch size: {BATCH_SIZE} | "
                f"Ref years: {REF_YEARS[0]}-{REF_YEARS[-1]} | "
                f"Timeout: {GEMINI_TIMEOUT_SECS}s\n"
            )
        fh.write("\n".join(lines) + "\n")
    console.log(f"  + Appended batch {batch_num} to Validation_Run_Log.md")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load reference cohort map for predicted rating ─────────────────────────
    ref_cohort_map = load_reference_cohort_map()

    # ── Load or init validation population ────────────────────────────────────
    val_data    = load_validation()
    val_markets = val_data["markets"]
    total_processed = len(val_markets)
    batch_num       = val_data["metadata"].get("batches_completed", 0)
    blacklist: set  = set(val_data["metadata"].get("blacklist", []))

    # ── Build dedup registry: reference population + existing validation ───────
    registry: set = set()
    if MASTER_JSON.exists():
        with open(MASTER_JSON, encoding="utf-8") as fh:
            ref_data = json.load(fh)
        for m in ref_data.get("markets", []):
            mn = m.get("base_profile", {}).get("market_name", "") or m.get("domain", "")
            if mn:
                registry.add(_normalise(mn))
    for m in val_markets:
        if m.get("name"):
            registry.add(_normalise(m["name"]))
    console.log(f"  Dedup registry: {len(registry)} entries (ref + existing val)")

    claude_client = get_claude_client()
    gemini_client = get_gemini_client()

    with Live(
        make_display(total_processed, batch_num, blacklist, val_markets),
        console=console,
        refresh_per_second=4,
        redirect_stdout=False,
    ) as live:

        def _refresh() -> None:
            live.update(make_display(total_processed, batch_num, blacklist, val_markets))

        console.log("=" * 68)
        console.log("  VELA MQR -- MIRROR VALIDATION PIPELINE")
        console.log(f"  Target: {TARGET_COUNT} markets  |  Batches: {NUM_BATCHES}  "
                    f"|  Batch size: {BATCH_SIZE}")
        console.log(f"  Ref years: {REF_YEARS[0]}-{REF_YEARS[-1]}  "
                    f"|  T+5 timeout: {GEMINI_TIMEOUT_SECS}s  "
                    f"|  Workers: {MAX_WORKERS}")
        console.log(f"  Output: {VAL_JSON}")
        console.log("=" * 68)

        while total_processed < TARGET_COUNT:
            batch_num += 1
            run_ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            remaining = TARGET_COUNT - total_processed
            batch_sz  = min(BATCH_SIZE, remaining)

            console.log(f"\n{'─' * 68}")
            console.log(
                f"  BATCH {batch_num}  |  "
                f"{total_processed}/{TARGET_COUNT}  |  {run_ts}"
            )
            console.log(f"{'─' * 68}")
            _refresh()

            # ── Role 0: Gemini seed generation ────────────────────────────────
            seeds: list = []
            for attempt in range(1, 3):   # up to 2 attempts
                _set_stage("role0", "Role 0: Seeds", "Gemini web search")
                _refresh()
                seeds = role0_generate_seeds(gemini_client, registry, blacklist, batch_sz)
                _clear_stage("role0")
                _refresh()
                if seeds:
                    break
                console.log(f"  [Role 0] Attempt {attempt} produced no seeds. Retrying ...")

            if not seeds:
                console.log("  WARNING: Role 0 failed to produce seeds. Skipping batch ...")
                batch_num -= 1
                continue

            batch_markets:    list = []
            start_num = total_processed + 1

            console.log(f"\n  Launching {len(seeds)} workers (MAX_WORKERS={MAX_WORKERS}) ...")
            _refresh()

            # ── Parallel Roles 1 -> Phase 2 ───────────────────────────────────
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_seed = {
                    executor.submit(
                        _process_market_worker,
                        seed,
                        f"val_{start_num + i:03d}",
                        set(registry),     # thread-safe snapshot
                        set(blacklist),
                        ref_cohort_map,
                        claude_client,
                        gemini_client,
                    ): (i, seed)
                    for i, seed in enumerate(seeds)
                }

                for future in as_completed(future_to_seed):
                    val_market, mkt_name, timed_out = future.result()
                    _refresh()

                    if timed_out and mkt_name:
                        blacklist.add(_normalise(mkt_name))
                        with _status_lock:
                            global _blacklist_hits
                            _blacklist_hits += 1
                        continue

                    if val_market is None:
                        continue

                    # Update registry (main thread only)
                    registry.add(_normalise(mkt_name))

                    # Append and save incrementally
                    batch_markets.append(val_market)
                    val_markets.append(val_market)
                    total_processed += 1
                    save_validation(val_data, val_markets, blacklist, batch_num)
                    _refresh()

            # ── Batch complete: re-label actual_rating ─────────────────────────
            if not batch_markets:
                console.log("  WARNING: No markets completed this batch. Continuing ...")
                continue

            reassign_actual_ratings(val_markets)

            # Back-fill _recent_done with now-available actual ratings
            with _status_lock:
                for r in _recent_done:
                    mid = r.get("id", "")
                    for m in val_markets:
                        if m["id"] == mid and m.get("actual_rating"):
                            r["actual"] = m["actual_rating"]

            save_validation(val_data, val_markets, blacklist, batch_num)
            _refresh()

            acc = _compute_accuracy(val_markets)
            console.log(
                f"\n  Batch {batch_num} complete: {len(batch_markets)} saved  "
                f"| Total: {total_processed}/{TARGET_COUNT}  "
                f"| Blacklist: {len(blacklist)}  "
                f"| Exact: {acc['exact_pct']}%  Within-1: {acc['off1_pct']}%"
            )

            append_batch_log(
                batch_num, batch_markets, total_processed,
                run_ts, acc, blacklist,
            )
            _refresh()

    # ── Pipeline complete ──────────────────────────────────────────────────────
    acc = _compute_accuracy(val_markets)

    console.print(f"\n{'=' * 68}", style="bold green")
    console.print("  MIRROR VALIDATION PIPELINE COMPLETE", style="bold green")
    console.print(f"{'=' * 68}", style="bold green")
    console.print(f"  Markets processed    : {total_processed}")
    console.print(f"  Batches run          : {batch_num}")
    console.print(f"  Blacklisted          : {len(blacklist)}")
    console.print(f"  Deduped (skipped)    : {_dedup_skipped}")
    console.print(f"  T+5 search failures  : {_t5_failures}")
    console.print("")
    console.print(f"  --- Accuracy Report ---")
    console.print(f"  Total evaluated      : {acc['total']}")
    console.print(f"  Exact match          : {acc['exact']}/{acc['total']} ({acc['exact_pct']}%)")
    console.print(f"  Within +/-1 band     : {acc['off1']}/{acc['total']} ({acc['off1_pct']}%)")
    console.print(f"  Output               : {VAL_JSON}")
    console.print(f"  Run log              : {LOG_PATH}")

    # Append final entry to LOGBOOK_MASTER.md
    logbook = ROOT / "lab_notes" / "LOGBOOK_MASTER.md"
    if logbook.exists():
        with open(logbook, "a", encoding="utf-8") as fh:
            fh.write(
                f"* Mirror Validation Pipeline Complete: {total_processed}/60 markets | "
                f"Exact={acc['exact_pct']}% | Within-1={acc['off1_pct']}% | "
                f"blacklisted={len(blacklist)}\n"
            )


if __name__ == "__main__":
    main()
