"""
Vela MQR — Ablation Study Orchestrator  [gpt-5.4 edition]

Runs ONE test market through three conflict-resolution paths and produces
a single Markdown report showing judge logic, final categorical states,
and Role 3 scores for each path.

  Path 1  Baseline    Role 1 → Role 2 → Role 3
                      No judge.  MEDIUM / LOW conflicts are kept FLAGGED
                      so Role 3 sees both candidate values.

  Path 2  Parametric  Role 1 → Role 2 → GPT-5.4 (parametric) → Role 3
                      Judge uses internal knowledge only; may resolve or flag.

  Path 3  Search      Role 1 → Role 2 → GPT-5.4 + web search → Role 3
                      Judge uses live search; may resolve with citations or flag.

Output: lab_notes/Ablation_Run_GPT5.4.md
No automated git operations.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Bootstrap sys.path
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
# Constants (Role 3 scorer — inlined to avoid circular imports)
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
    "6. For dimensions marked [CONFLICT — UNRESOLVED], score conservatively using "
    "a midpoint interpretation between the two candidate values shown.\n"
    "Tier classification is performed separately via percentile lookup — "
    "your sole responsibility is the 7 integer scores."
)

# ---------------------------------------------------------------------------
# Role 3 scorer
# ---------------------------------------------------------------------------

def _build_scoring_prompt(matrix: dict) -> str:
    """
    Build the anonymised scoring prompt from a feature matrix.
    Handles FLAGGED dimensions by showing both candidate values to the scorer.
    """
    lines = [
        "=== ANONYMISED MARKET FEATURE MATRIX ===",
        "",
        f"  {'DIMENSION':<24}  {'VERIFIED CLASSIFICATION':<44}  INTER-SOURCE AGREEMENT",
        f"  {'-'*24}  {'-'*44}  {'-'*22}",
    ]
    for dim in DIMENSIONS:
        entry  = matrix.get(dim, {})
        value  = entry.get("value", "?")
        agr    = entry.get("agreement", "?")
        if value == "FLAGGED":
            a = entry.get("conflict_a", "?")
            b = entry.get("conflict_b", "?")
            display = f"{a} / {b}  [CONFLICT — UNRESOLVED]"
        else:
            display = value
        lines.append(f"  {dim:<24}  {display:<44}  {agr}")

    lines += [
        "",
        "=== SCORING TASK ===",
        "Map each dimension's categorical value to a 0-100 integer using the numeric",
        "scale in your system prompt. Score each dimension independently.",
        "Base your scores ONLY on the categorical values above — do not infer market identity.",
        "For any dimension marked [CONFLICT — UNRESOLVED], apply a conservative score",
        "that reflects the uncertainty between the two candidate values shown.",
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
    Score an anonymised feature matrix with Claude Role 3.
    Returns (scores_dict, prompt_string).
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
# Baseline feature-matrix builder — KEEPS CONFLICTS FLAGGED
# ---------------------------------------------------------------------------

def build_baseline_matrix_flagged(market: dict) -> dict:
    """
    Path 1 (Baseline): no judge.
    - HIGH agreement → Gemini-preferred value, unflagged.
    - MEDIUM / LOW agreement → FLAGGED; both candidate values preserved.
    """
    step1_dims   = market.get("dimensions", {})
    step2_verifs = market.get("step2", {}).get("dimension_verifications", {})
    matrix = {}
    for dim in DIMENSIONS:
        s1 = step1_dims.get(dim, {})
        s2 = step2_verifs.get(dim, {})
        gemini = s2.get("gemini_classification", "")
        claude = s1.get("classification", "unknown")
        agr    = s2.get("agreement", "unverified")

        if agr in ("HIGH", "agreed", "unverified", ""):
            # Agreed — pass Gemini value (Claude fallback)
            value = gemini if gemini and gemini not in ("unknown", "") else claude
            matrix[dim] = {
                "value":      value,
                "agreement":  agr,
                "source":     "agreed",
                "flagged":    False,
                "conflict_a": "",
                "conflict_b": "",
            }
        else:
            # Disagreement — keep FLAGGED, preserve both options
            matrix[dim] = {
                "value":      "FLAGGED",
                "agreement":  agr,
                "source":     "baseline_no_judge",
                "flagged":    True,
                "conflict_a": claude,
                "conflict_b": gemini if gemini and gemini not in ("unknown", "") else "unknown",
            }
    return matrix


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def _url_cell(url: str) -> str:
    if not url:
        return "_no URL_"
    display = url if len(url) <= 80 else url[:77] + "..."
    return f"[link]({url})" if url.startswith("http") else display


def _flag_badge(flagged: bool) -> str:
    return "⚑ FLAGGED" if flagged else "✓ resolved"


def generate_report(
    market: dict,
    b_matrix: dict, b_scores: dict,
    p_matrix: dict, p_scores: dict,
    s_matrix: dict, s_scores: dict,
    prompt_audit: str,
    run_ts: str,
) -> str:
    name     = market.get("base_profile", {}).get("market_name", market.get("domain", "?"))
    ref_year = market.get("ref_year", "?")
    step1    = market.get("dimensions", {})
    step2_v  = market.get("step2", {}).get("dimension_verifications", {})

    lines = [
        "# Vela MQR — Ablation Study Report  (GPT-5.4 Judge)",
        "",
        f"**Test market:** {name} ({ref_year})  ",
        f"**Run:** `{run_ts}`  ",
        "**Paths:**",
        "- **Path 1 — Baseline:** no judge; conflicts kept FLAGGED",
        "- **Path 2 — Parametric:** GPT-5.4, internal knowledge only; may resolve or flag",
        "- **Path 3 — Search:** GPT-5.4 + live web search; may resolve with citations or flag",
        "",
        "---",
        "",
    ]

    # ── Section 1: Conflict inventory (from Role 2) ───────────────────────
    lines += [
        f"## {name} ({ref_year})",
        "",
        "### Role 2 Verification — Agreement & Source URLs",
        "",
        "| Dimension | Claude | Gemini | Agreement | Verification URL |",
        "|-----------|--------|--------|:---------:|-----------------|",
    ]
    for dim in DIMENSIONS:
        s1  = step1.get(dim, {})
        dv  = step2_v.get(dim, {})
        cc  = s1.get("classification", "?")
        gc  = dv.get("gemini_classification", "?")
        agr = dv.get("agreement", "?")
        url = dv.get("verification_url", "") or (dv.get("grounding_urls") or [""])[0]
        lines.append(f"| `{dim}` | `{cc}` | `{gc}` | {agr} | {_url_cell(url)} |")

    # ── Section 2: Full analysis per path ─────────────────────────────────
    lines += ["", "---", ""]

    for path_num, (path_name, path_desc, matrix, scores) in enumerate([
        ("Path 1: Baseline",   "No judge — conflicts kept FLAGGED",             b_matrix, b_scores),
        ("Path 2: Parametric", f"GPT-5.4 parametric judge (no web search)",      p_matrix, p_scores),
        ("Path 3: Search",     f"GPT-5.4 search judge (live web evidence)",       s_matrix, s_scores),
    ], start=1):
        flagged_count  = sum(1 for e in matrix.values() if e.get("flagged"))
        resolved_count = sum(1 for e in matrix.values() if not e.get("flagged"))

        lines += [
            f"### {path_name}",
            f"_{path_desc}_  ",
            f"**Resolved:** {resolved_count}  **Flagged:** {flagged_count}",
            "",
        ]

        # Full dimension table
        if path_num == 3:  # Search path — include search source column
            lines += [
                "| Dimension | Final Classification | Status | Judge Rationale | Search Source | Score |",
                "|-----------|-------------------|:------:|-----------------|---------------|:-----:|",
            ]
            for dim in DIMENSIONS:
                entry   = matrix.get(dim, {})
                val     = entry.get("value", "?")
                flagged = entry.get("flagged", False)
                rat     = entry.get("rationale", "—")
                if val == "FLAGGED":
                    disp = f"`{entry.get('conflict_a','?')}` / `{entry.get('conflict_b','?')}`"
                else:
                    disp = f"`{val}`"
                badge   = _flag_badge(flagged)
                rat_s   = (rat[:120] + "…") if len(rat) > 120 else rat
                src     = _url_cell(entry.get("search_source", ""))
                sc      = scores.get(dim)
                sc_s    = str(sc) if sc is not None else "ERR"
                lines.append(
                    f"| `{dim}` | {disp} | {badge} | {rat_s} | {src} | {sc_s} |"
                )
        else:
            lines += [
                "| Dimension | Final Classification | Status | Judge Rationale | Score |",
                "|-----------|-------------------|:------:|-----------------|:-----:|",
            ]
            for dim in DIMENSIONS:
                entry   = matrix.get(dim, {})
                val     = entry.get("value", "?")
                flagged = entry.get("flagged", False)
                rat     = entry.get("rationale", "—")
                if val == "FLAGGED":
                    disp = f"`{entry.get('conflict_a','?')}` / `{entry.get('conflict_b','?')}`"
                else:
                    disp = f"`{val}`"
                badge   = _flag_badge(flagged)
                rat_s   = (rat[:120] + "…") if len(rat) > 120 else rat
                sc      = scores.get(dim)
                sc_s    = str(sc) if sc is not None else "ERR"
                lines.append(
                    f"| `{dim}` | {disp} | {badge} | {rat_s} | {sc_s} |"
                )

        # Mean score
        valid = [v for v in scores.values() if v is not None]
        mean  = round(sum(valid) / len(valid)) if valid else None
        lines += ["", f"**Mean score:** {mean}", ""]

    # ── Section 3: 3-Path Score Comparison ───────────────────────────────
    lines += [
        "---",
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
        # Flag status markers
        b_flag = " ⚑" if b_matrix.get(dim, {}).get("flagged") else ""
        p_flag = " ⚑" if p_matrix.get(dim, {}).get("flagged") else ""
        s_flag = " ⚑" if s_matrix.get(dim, {}).get("flagged") else ""
        lines.append(
            f"| `{dim}` | {b_s}{b_flag} | {p_s}{p_flag} | {s_s}{s_flag} "
            f"| {d_p} | {d_s} |"
        )

    b_vals = [v for v in b_scores.values() if v is not None]
    p_vals = [v for v in p_scores.values() if v is not None]
    s_vals = [v for v in s_scores.values() if v is not None]
    b_m = round(sum(b_vals) / len(b_vals)) if b_vals else None
    p_m = round(sum(p_vals) / len(p_vals)) if p_vals else None
    s_m = round(sum(s_vals) / len(s_vals)) if s_vals else None
    d_pm = f"{p_m - b_m:+d}" if (p_m and b_m) else "—"
    d_sm = f"{s_m - b_m:+d}" if (s_m and b_m) else "—"
    lines.append(f"| **Mean** | **{b_m}** | **{p_m}** | **{s_m}** | **{d_pm}** | **{d_sm}** |")
    lines += [
        "",
        "_⚑ = dimension was FLAGGED (unresolved conflict); score is conservative midpoint._",
        "",
        "---",
        "",
    ]

    # ── Section 4: GPT-5.4 Judge Logic Deep-Dive ────────────────────────
    lines += [
        "### GPT-5.4 Judge Logic — Resolved Dimensions",
        "",
        "_Full rationale for every dimension the judge actively resolved or flagged._",
        "",
    ]

    # Collect all judged dimensions from both paths
    for path_label, matrix, path_name in [
        ("Parametric", p_matrix, "GPT-5.4 Parametric"),
        ("Search",     s_matrix, "GPT-5.4 Search"),
    ]:
        judged = [
            (dim, matrix[dim])
            for dim in DIMENSIONS
            if matrix.get(dim, {}).get("source", "agreed") != "agreed"
        ]
        if not judged:
            continue
        lines += [f"#### {path_name}", ""]
        for dim, entry in judged:
            val     = entry.get("value", "?")
            flagged = entry.get("flagged", False)
            chosen  = entry.get("chosen_classifier", "?")
            rat     = entry.get("rationale", "—")
            agr     = entry.get("agreement", "?")
            a       = entry.get("conflict_a", "?")
            b_val   = entry.get("conflict_b", "?")
            status  = _flag_badge(flagged)

            if not flagged:
                lines.append(
                    f"**`{dim}`** — {status}  \n"
                    f"Original conflict: `{a}` (A) vs `{b_val}` (B) — agreement={agr}  \n"
                    f"Judge chose: **`{val}`** (Classifier {chosen})  \n"
                    f"Rationale: _{rat}_"
                )
            else:
                lines.append(
                    f"**`{dim}`** — {status}  \n"
                    f"Original conflict: `{a}` (A) vs `{b_val}` (B) — agreement={agr}  \n"
                    f"Judge: **could not resolve**  \n"
                    f"Rationale: _{rat}_"
                )

            # Search sources if available
            sources = entry.get("search_sources", [])
            if sources:
                lines.append(f"Sources: {', '.join(_url_cell(u) for u in sources)}")

            lines.append("")

    lines += ["---", ""]

    # ── Section 5: Role 3 Prompt Audit ───────────────────────────────────
    lines += [
        "### Role 3 Prompt Audit — Baseline Path",
        "",
        "_Exact prompt sent to Claude (Role 3 Scorer). Proves that market name, "
        "base profile, and all identifying information are stripped before scoring. "
        "FLAGGED dimensions show both candidate values — the scorer cannot infer "
        "market identity from these._",
        "",
        "```",
        prompt_audit,
        "```",
        "",
        "---",
        "",
        f"_Generated by `run_ablation_study.py` · {run_ts} · model: GPT-5.4_",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 65)
    print("  VELA MQR — ABLATION STUDY  (GPT-5.4 Judge)")
    print(f"  Run: {run_ts}")
    print("=" * 65)

    # --- Load data (first market only) ---
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

    market = data["markets"][0]
    name   = market.get("base_profile", {}).get("market_name", market.get("domain", "?"))
    print(f"\n  Test market: {name} ({market.get('ref_year','?')})")
    print(f"  Input file : {os.path.basename(json_path)}")

    # --- Clients ---
    claude_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not claude_key or claude_key.startswith("your_"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set.")
    claude_client = anthropic.Anthropic(api_key=claude_key)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key.startswith("your_"):
        sys.exit("ERROR: OPENAI_API_KEY not set.")
    openai_client = OpenAI(api_key=openai_key)

    # --- Path 1: Baseline (conflicts flagged, no judge) ---
    print("\n  [Path 1 — Baseline: conflicts FLAGGED, no judge]")
    b_matrix = build_baseline_matrix_flagged(market)
    flagged_b = sum(1 for e in b_matrix.values() if e.get("flagged"))
    print(f"  {flagged_b} dimension(s) flagged as conflicts")
    b_scores, b_prompt = score_from_matrix(claude_client, b_matrix, "Baseline")
    print(f"  Scores: {list(b_scores.values())}")
    time.sleep(0.5)

    # --- Path 2: Parametric judge ---
    print("\n  [Path 2 — Parametric Judge (GPT-5.4, no search)]")
    p_matrix = resolve_market_parametric(market, openai_client)
    flagged_p = sum(1 for e in p_matrix.values() if e.get("flagged"))
    print(f"  {flagged_p} dimension(s) remain flagged after judge")
    time.sleep(0.5)
    p_scores, _ = score_from_matrix(claude_client, p_matrix, "Parametric")
    print(f"  Scores: {list(p_scores.values())}")
    time.sleep(0.5)

    # --- Path 3: Search judge ---
    print("\n  [Path 3 — Search Judge (GPT-5.4 + web)]")
    s_matrix = resolve_market_search(market, openai_client)
    flagged_s = sum(1 for e in s_matrix.values() if e.get("flagged"))
    print(f"  {flagged_s} dimension(s) remain flagged after judge")
    time.sleep(0.5)
    s_scores, _ = score_from_matrix(claude_client, s_matrix, "Search")
    print(f"  Scores: {list(s_scores.values())}")

    # --- Generate report ---
    print("\n  Generating Markdown report ...")
    report_md = generate_report(
        market    = market,
        b_matrix  = b_matrix, b_scores = b_scores,
        p_matrix  = p_matrix, p_scores = p_scores,
        s_matrix  = s_matrix, s_scores = s_scores,
        prompt_audit = b_prompt,
        run_ts    = run_ts,
    )

    lab_notes_dir = os.path.join(_ROOT, "lab_notes")
    os.makedirs(lab_notes_dir, exist_ok=True)
    report_path = os.path.join(lab_notes_dir, "Ablation_Run_GPT5.4.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)

    print(f"  + Saved → lab_notes/Ablation_Run_GPT5.4.md")
    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  Baseline  : {list(b_scores.values())}  mean={round(sum(v for v in b_scores.values() if v is not None)/7)}")
    print(f"  Parametric: {list(p_scores.values())}  mean={round(sum(v for v in p_scores.values() if v is not None)/7)}")
    print(f"  Search    : {list(s_scores.values())}  mean={round(sum(v for v in s_scores.values() if v is not None)/7)}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
