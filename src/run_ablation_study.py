"""
Vela MQR — Ablation Study Orchestrator

Runs each market in the Step 2 JSON through three parallel paths to test
different conflict-resolution strategies between Role 1 (Claude) and
Role 2 (Gemini):

  Path 1  Baseline     Role 1 → Role 2 → Role 3
                       No tie-breaker; Gemini-preferred feature matrix.

  Path 2  Parametric   Role 1 → Role 2 → Step 2b Parametric (GPT-4o) → Role 3
                       OpenAI resolves disagreements using parametric knowledge.

  Path 3  Search       Role 1 → Role 2 → Step 2b Search (GPT-4o + web) → Role 3
                       OpenAI resolves disagreements with live web search.

Outputs
-------
- Comparative Markdown report  →  lab_notes/Ablation_Run_<timestamp>.md
- LOGBOOK_MASTER.md entry appended automatically
- Auto git-commit and git-push on completion
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Bootstrap: add src/ to sys.path for sibling imports
# ---------------------------------------------------------------------------
_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline_step2b_parametric import resolve_market_parametric
from pipeline_step2b_search     import resolve_market_search

try:
    import anthropic
except ImportError:
    sys.exit("ERROR: anthropic package not found.  pip install anthropic")
try:
    from openai import OpenAI
except ImportError:
    sys.exit("ERROR: openai package not found.  pip install openai")

# ---------------------------------------------------------------------------
# Constants (mirror pipeline_step3.py — inlined to avoid import side-effects)
# ---------------------------------------------------------------------------

SCORER_MODEL = "claude-sonnet-4-6"

DIMENSIONS = [
    "timing",
    "competition",
    "market_size",
    "customer_readiness",
    "regulatory",
    "infrastructure",
    "market_structure",
]

_SCORE_RUBRIC = (
    "90-100: Ideal conditions for outsized venture returns.\n"
    "75-89:  Highly attractive, minor friction.\n"
    "60-74:  Viable but requires execution excellence.\n"
    "40-59:  Sub-optimal conditions, significant structural headwinds.\n"
    "0-39:   Hostile conditions for new venture entrants."
)

SCORER_SYSTEM_PROMPT = (
    "You are an objective quantitative venture analyst scoring anonymous market data. "
    "You are evaluating a market's readiness for venture-scale entry based solely on "
    "structured categorical variables. "
    "Score each of the 7 dimensions from 0 to 100 based purely on the structured "
    "features provided. Use this numeric scale as your calibration guide:\n\n"
    + _SCORE_RUBRIC
    + "\n\n"
    "STRICT OUTPUT RULES — you must follow these exactly:\n"
    "1. Output ONLY a JSON object containing the 7 dimension scores as integers.\n"
    "2. Do NOT output L-tier labels, tier names, or any classification strings.\n"
    "3. Do NOT compute or output a composite score or overall rating.\n"
    "4. Do NOT output an investment recommendation or decision.\n"
    "5. Do NOT add any keys beyond the 7 required dimension keys.\n"
    "Tier classification is performed separately via percentile lookup — "
    "your sole responsibility is the 7 integer scores."
)

# ---------------------------------------------------------------------------
# Role 3 scorer (self-contained)
# ---------------------------------------------------------------------------

def _build_scoring_prompt(matrix: dict) -> str:
    """Build the anonymized scoring prompt from a feature matrix."""
    lines = [
        "=== ANONYMIZED MARKET FEATURE MATRIX ===",
        "",
        f"  {'DIMENSION':<24}  {'VERIFIED CLASSIFICATION':<36}  INTER-SOURCE AGREEMENT",
        f"  {'-'*24}  {'-'*36}  {'-'*22}",
    ]
    for dim in DIMENSIONS:
        entry = matrix.get(dim, {})
        lines.append(
            f"  {dim:<24}  {entry.get('value','?'):<36}  {entry.get('agreement','?')}"
        )
    lines += [
        "",
        "=== SCORING TASK ===",
        "Map each dimension's categorical value to a 0-100 integer using the numeric",
        "scale in your system prompt. Score each dimension independently.",
        "Base your scores ONLY on the categorical values above — do not infer market identity.",
        "",
        "REQUIRED OUTPUT: a JSON object with EXACTLY these 7 integer keys.",
        "Do NOT add any other keys. Do NOT include tier labels, composite scores,",
        "investment ratings, or any text outside the JSON object.",
        "{",
        '  "timing": 0,',
        '  "competition": 0,',
        '  "market_size": 0,',
        '  "customer_readiness": 0,',
        '  "regulatory": 0,',
        '  "infrastructure": 0,',
        '  "market_structure": 0',
        "}",
        "",
        "Replace each 0 with your integer score. Return nothing but this JSON object.",
    ]
    return "\n".join(lines)


def _claude_retry(fn, max_retries: int = 4, base_delay: float = 20.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            exc_str = str(exc).lower()
            if ("529" in exc_str or "overloaded" in exc_str) and attempt < max_retries:
                delay = base_delay * (attempt + 1)
                print(
                    f"\n    [Claude 529; retry {attempt+1}/{max_retries} in {delay:.0f}s]",
                    end=" ", flush=True,
                )
                time.sleep(delay)
            else:
                raise


def score_from_matrix(
    claude_client,
    matrix: dict,
    label: str,
) -> tuple:
    """
    Score an anonymized feature matrix with Claude Role 3.
    Returns (scores_dict, prompt_string).
    scores_dict maps dim → int (or None on error).
    """
    prompt = _build_scoring_prompt(matrix)

    try:
        response = _claude_retry(
            lambda: claude_client.messages.create(
                model=SCORER_MODEL,
                max_tokens=400,
                system=SCORER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        )
    except Exception as exc:
        print(f"  [{label}] Claude API error: {exc}")
        return {dim: None for dim in DIMENSIONS}, prompt

    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [{label}] JSON parse error — raw: {raw[:80]!r}")
        return {dim: None for dim in DIMENSIONS}, prompt

    scores = {}
    for dim in DIMENSIONS:
        val = parsed.get(dim)
        if val is None:
            scores[dim] = None
        else:
            try:
                scores[dim] = max(0, min(100, int(val)))
            except (TypeError, ValueError):
                scores[dim] = None

    return scores, prompt


# ---------------------------------------------------------------------------
# Baseline feature-matrix builder (mirrors pipeline_step3.build_feature_matrix)
# ---------------------------------------------------------------------------

def build_baseline_matrix(market: dict) -> dict:
    """Gemini-preferred, Claude fallback — identical to pipeline_step3 logic."""
    step1_dims   = market.get("dimensions", {})
    step2_verifs = market.get("step2", {}).get("dimension_verifications", {})
    matrix = {}
    for dim in DIMENSIONS:
        s1 = step1_dims.get(dim, {})
        s2 = step2_verifs.get(dim, {})
        gemini = s2.get("gemini_classification", "")
        claude = s1.get("classification", "unknown")
        matrix[dim] = {
            "value":     gemini if gemini and gemini not in ("unknown", "") else claude,
            "agreement": s2.get("agreement", "unverified"),
        }
    return matrix


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def _url_cell(url: str) -> str:
    if not url:
        return "_no URL_"
    # Shorten long redirect URLs for readability
    display = url if len(url) <= 80 else url[:77] + "..."
    return f"[link]({url})" if url.startswith("http") else display


def generate_report(
    all_results: list,
    run_ts: str,
    prompt_audit: str,
) -> str:
    lines = [
        "# Vela MQR — Ablation Study Report",
        "",
        f"**Run:** `{run_ts}`  ",
        f"**Markets:** {len(all_results)}  ",
        "**Paths compared:**",
        "- **Baseline** — Role 1 (Claude) → Role 2 (Gemini) → Role 3 (Claude scorer)",
        "- **Parametric Judge** — + GPT-4o tie-breaker (parametric knowledge, no search)",
        "- **Search Judge** — + GPT-4o tie-breaker (live web search via Responses API)",
        "",
        "---",
        "",
    ]

    # Collect all URLs for the bottom audit section
    url_audit_rows = []

    for r in all_results:
        market     = r["market"]
        name       = market.get("base_profile", {}).get("market_name", market.get("domain", "?"))
        ref_year   = market.get("ref_year", "?")
        b_scores   = r["baseline_scores"]
        p_scores   = r["parametric_scores"]
        s_scores   = r["search_scores"]
        b_matrix   = r["baseline_matrix"]
        p_matrix   = r["parametric_matrix"]
        s_matrix   = r["search_matrix"]
        step1_dims = market.get("dimensions", {})
        step2_v    = market.get("step2", {}).get("dimension_verifications", {})

        lines += [f"## {name} ({ref_year})", ""]

        # ── Path 1: Baseline ──────────────────────────────────────────────
        lines += [
            "### Path 1: Baseline (Gemini-Preferred, no tie-breaker)",
            "",
            "| Dimension | Classification | Agreement | Role 2 Evidence | Score |",
            "|-----------|---------------|:---------:|-----------------|:-----:|",
        ]
        for dim in DIMENSIONS:
            s2  = step2_v.get(dim, {})
            agr = s2.get("agreement", "unverified")
            val = b_matrix.get(dim, {}).get("value", "?")
            # Pull best available evidence summary from Gemini step
            evidence = (
                s2.get("key_fact", "")
                or s2.get("gemini_evidence", "")
                or s2.get("evidence", "")
                or "—"
            )
            # Truncate very long evidence for table readability
            if len(evidence) > 120:
                evidence = evidence[:117] + "..."
            score = b_scores.get(dim)
            score_s = str(score) if score is not None else "ERR"
            lines.append(f"| `{dim}` | `{val}` | {agr} | {evidence} | {score_s} |")

        # ── Path 2: Parametric Judge ──────────────────────────────────────
        lines += [
            "",
            "### Path 2: Parametric Judge (GPT-4o — parametric knowledge, no search)",
            "",
            "| Dimension | Classification | Source | Rationale | Score |",
            "|-----------|---------------|--------|-----------|:-----:|",
        ]
        for dim in DIMENSIONS:
            entry   = p_matrix.get(dim, {})
            val     = entry.get("value", "?")
            source  = entry.get("source", "agreed")
            rat     = entry.get("rationale", "Claude and Gemini agreed — no tie-break required.")
            if len(rat) > 140:
                rat = rat[:137] + "..."
            score   = p_scores.get(dim)
            score_s = str(score) if score is not None else "ERR"
            src_tag = f"_{source}_" if source != "agreed" else "agreed"
            lines.append(f"| `{dim}` | `{val}` | {src_tag} | {rat} | {score_s} |")

        # ── Path 3: Search Judge ──────────────────────────────────────────
        lines += [
            "",
            "### Path 3: Search Judge (GPT-4o + live web search)",
            "",
            "| Dimension | Classification | Source | Rationale | Search Source | Score |",
            "|-----------|---------------|--------|-----------|---------------|:-----:|",
        ]
        for dim in DIMENSIONS:
            entry    = s_matrix.get(dim, {})
            val      = entry.get("value", "?")
            source   = entry.get("source", "agreed")
            rat      = entry.get("rationale", "Claude and Gemini agreed — no tie-break required.")
            if len(rat) > 120:
                rat = rat[:117] + "..."
            src_url  = _url_cell(entry.get("search_source", ""))
            score    = s_scores.get(dim)
            score_s  = str(score) if score is not None else "ERR"
            src_tag  = f"_{source}_" if source != "agreed" else "agreed"
            lines.append(f"| `{dim}` | `{val}` | {src_tag} | {rat} | {src_url} | {score_s} |")

        # ── 3-Path Score Comparison ───────────────────────────────────────
        lines += [
            "",
            "### 3-Path Score Comparison",
            "",
            "| Dimension | Baseline | Parametric | Search | Δ Param | Δ Search |",
            "|-----------|:--------:|:----------:|:------:|:-------:|:--------:|",
        ]
        for dim in DIMENSIONS:
            b = b_scores.get(dim)
            p = p_scores.get(dim)
            s = s_scores.get(dim)
            b_s = str(b) if b is not None else "ERR"
            p_s = str(p) if p is not None else "ERR"
            s_s = str(s) if s is not None else "ERR"
            d_p = f"{p - b:+d}" if (p is not None and b is not None) else "—"
            d_s = f"{s - b:+d}" if (s is not None and b is not None) else "—"
            lines.append(f"| `{dim}` | {b_s} | {p_s} | {s_s} | {d_p} | {d_s} |")

        b_vals = [v for v in b_scores.values() if v is not None]
        p_vals = [v for v in p_scores.values() if v is not None]
        s_vals = [v for v in s_scores.values() if v is not None]
        b_m = round(sum(b_vals) / len(b_vals)) if b_vals else None
        p_m = round(sum(p_vals) / len(p_vals)) if p_vals else None
        s_m = round(sum(s_vals) / len(s_vals)) if s_vals else None
        d_pm = f"{p_m - b_m:+d}" if (p_m is not None and b_m is not None) else "—"
        d_sm = f"{s_m - b_m:+d}" if (s_m is not None and b_m is not None) else "—"
        lines.append(
            f"| **Mean** | **{b_m}** | **{p_m}** | **{s_m}** | **{d_pm}** | **{d_sm}** |"
        )

        lines += ["", "---", ""]

        # Collect URLs for bottom audit section
        for dim in DIMENSIONS:
            dv  = step2_v.get(dim, {})
            url = (
                dv.get("verification_url", "")
                or (dv.get("grounding_urls") or [""])[0]
            )
            url_audit_rows.append((name, ref_year, dim, url))

    # ── URL Audit Section (bottom, all markets) ───────────────────────────
    lines += [
        "## URL Audit — Role 2 Verification Sources (All Markets)",
        "",
        "_All Gemini grounding URLs retrieved during Role 2 verification, "
        "listed here for provenance and reproducibility._",
        "",
        "| Market | Year | Dimension | Verification URL |",
        "|--------|------|-----------|-----------------|",
    ]
    for mname, myear, dim, url in url_audit_rows:
        lines.append(f"| {mname} | {myear} | `{dim}` | {_url_cell(url)} |")

    lines += ["", "---", ""]

    # ── Role 3 Prompt Audit ───────────────────────────────────────────────
    lines += [
        "## Role 3 Prompt Audit",
        "",
        "_Exact prompt sent to Claude (Role 3 Scorer) for **Market 1 — Baseline path**._  ",
        "_Confirms that market name, base profile, rationale, and all identifying "
        "information have been stripped before scoring._",
        "",
        "```",
        prompt_audit,
        "```",
        "",
        "---",
        "",
        f"_Auto-generated by `run_ablation_study.py` · run `{run_ts}`_",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Git auto-push
# ---------------------------------------------------------------------------

def auto_git_push(commit_message: str) -> None:
    print("\n  [Git] Staging all changes ...")
    for args in [
        ["git", "add", "."],
        ["git", "commit", "-m", commit_message],
        ["git", "push", "origin", "main"],
    ]:
        result = subprocess.run(args, capture_output=True, text=True, cwd=_ROOT)
        combined = (result.stdout + result.stderr).strip()
        if result.returncode != 0 and "nothing to commit" not in combined:
            print(f"  [Git] Error ({' '.join(args)}): {combined}")
        elif combined:
            print(f"  [Git] {combined}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  VELA MQR — ABLATION STUDY")
    print(f"  Run: {run_ts}")
    print("=" * 65)

    # --- Load Step 2 data ---
    candidates = [
        os.path.join(_ROOT, "reference_population_scored.json"),
        os.path.join(_ROOT, "reference_population_v3.json"),
        os.path.join(_ROOT, "reference_population.json"),
    ]
    json_path = next((p for p in candidates if os.path.exists(p)), None)
    if not json_path:
        sys.exit("ERROR: No reference_population JSON found.")

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    markets = data["markets"]
    print(f"\n  Input : {os.path.basename(json_path)}  ({len(markets)} markets)")

    # --- Clients ---
    claude_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not claude_key or claude_key.startswith("your_"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set.")
    claude_client = anthropic.Anthropic(api_key=claude_key)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key.startswith("your_"):
        sys.exit("ERROR: OPENAI_API_KEY not set.")
    openai_client = OpenAI(api_key=openai_key)

    # --- Run all 3 paths for every market ---
    all_results  = []
    prompt_audit = ""   # captured from Market 1, Baseline path

    for i, market in enumerate(markets):
        name = market.get("base_profile", {}).get("market_name", market.get("domain", "?"))
        ref  = market.get("ref_year", "?")
        print(f"\n{'─'*55}")
        print(f"  Market {i+1}/{len(markets)}: {name}  ({ref})")
        print(f"{'─'*55}")

        # Path 1 — Baseline
        print("\n  [Path 1 — Baseline]")
        b_matrix  = build_baseline_matrix(market)
        b_scores, b_prompt = score_from_matrix(claude_client, b_matrix, "Baseline")
        if i == 0:
            prompt_audit = b_prompt
        print(f"  Scores: {list(b_scores.values())}")
        time.sleep(0.5)

        # Path 2 — Parametric judge
        print("\n  [Path 2 — Parametric Judge (GPT-4o, no search)]")
        p_matrix = resolve_market_parametric(market, openai_client)
        time.sleep(0.5)
        p_scores, _ = score_from_matrix(claude_client, p_matrix, "Parametric")
        print(f"  Scores: {list(p_scores.values())}")
        time.sleep(0.5)

        # Path 3 — Search judge
        print("\n  [Path 3 — Search Judge (GPT-4o + web)]")
        s_matrix = resolve_market_search(market, openai_client)
        time.sleep(0.5)
        s_scores, _ = score_from_matrix(claude_client, s_matrix, "Search")
        print(f"  Scores: {list(s_scores.values())}")
        time.sleep(0.5)

        all_results.append({
            "market":           market,
            "baseline_scores":  b_scores,
            "baseline_matrix":  b_matrix,
            "parametric_scores": p_scores,
            "parametric_matrix": p_matrix,
            "search_scores":    s_scores,
            "search_matrix":    s_matrix,
        })

    # --- Generate and save report ---
    print("\n  Generating Markdown report ...")
    report_md = generate_report(all_results, run_ts, prompt_audit)

    lab_notes_dir = os.path.join(_ROOT, "lab_notes")
    os.makedirs(lab_notes_dir, exist_ok=True)
    report_filename = f"Ablation_Run_{run_ts}.md"
    report_path = os.path.join(lab_notes_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)
    print(f"  + Saved → lab_notes/{report_filename}")

    # --- Update LOGBOOK_MASTER ---
    logbook_path = os.path.join(lab_notes_dir, "LOGBOOK_MASTER.md")
    if os.path.exists(logbook_path):
        with open(logbook_path, "a", encoding="utf-8") as fh:
            fh.write(f"* [Ablation Run {run_ts}](./Ablation_Run_{run_ts}.md)\n")
        print("  + Updated LOGBOOK_MASTER.md")

    # --- Auto git-commit and push ---
    auto_git_push(f"Auto-commit: Ablation run {run_ts} completed")

    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  Report : lab_notes/{report_filename}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
