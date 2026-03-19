"""
Vela MQR — Scale Pipeline (Baseline, 120 Markets)

Autonomous orchestration that loops in batches of 15 until 120 markets
have been generated (Role 1), verified (Role 2), and scored (Role 3)
using the Baseline method.

Key features
------------
- Market suggestion: Claude proposes 15 diverse domain + ref_year seeds
  per batch; existing domains are passed to avoid duplicates.
- Acceptance gate: T+5 outcome must be supported by >= 3 grounded sources.
- Auto-correction: after each batch, the mean inter-source agreement is
  computed.  If it falls below LOW_BATCH_THRESHOLD, a meta-prompt asks
  Claude to rewrite the Step 1 extraction addendum for the next batch.
- Incremental save: reference_population_master.json is written after
  every individual market so no data is lost on crash.
- Batch log: lab_notes/Scaling_Run_Log.md is appended after each batch
  with verification URLs, Role 3 scores, and a Role 3 prompt audit.
- Auto git push after every batch.
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
# Bootstrap sys.path so sibling modules are importable
# ---------------------------------------------------------------------------
_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    sys.exit("ERROR: anthropic not found.  pip install anthropic")

try:
    from google import genai  # noqa: F401 — validated inside pipeline_step2 import
except ImportError:
    sys.exit("ERROR: google-genai not found.  pip install google-genai")

# ---------------------------------------------------------------------------
# Selective imports from existing pipeline modules
# (keeps all prompt text, enum definitions, and API logic in one place)
# ---------------------------------------------------------------------------
from pipeline_step1 import (
    DIMENSIONS,                         # list of {name, description, options, scoring_guide}
    _call_with_retry as _claude_retry,  # 529/overload retry wrapper
)
from pipeline_step2 import (
    get_gemini_client,
    process_market_verification,        # runs T+5 + 7-dim agreement for one market
    ORDINAL_SCALES,
)
from pipeline_step3 import (
    build_feature_matrix,   # anonymises market → {dim: {value, agreement}}
    build_scoring_prompt,   # produces the text prompt for Role 3
    SYSTEM_PROMPT as SCORER_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_COUNT  = 120
BATCH_SIZE    = 15
REF_YEARS     = list(range(2005, 2020))   # 2005–2019 inclusive
CLAUDE_MODEL  = "claude-sonnet-4-6"

MASTER_JSON   = os.path.join(_ROOT, "reference_population_master.json")
LOG_PATH      = os.path.join(_ROOT, "lab_notes", "Scaling_Run_Log.md")

# Batch agreement mean below this (scale 0–1) triggers auto-correction.
# step2 formula: score = (HIGH*1.0 + MEDIUM*0.5 + LOW*0.0) / total_dims
# 0.40 = overall "LOW" boundary used by step2 itself.
LOW_BATCH_THRESHOLD = 0.40

_DIM_NAMES = [d["name"] for d in DIMENSIONS]

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def get_claude_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your_"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# Persistent state — master JSON
# ---------------------------------------------------------------------------

def load_master() -> dict:
    if os.path.exists(MASTER_JSON):
        with open(MASTER_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "schema_version": "2.0",
        "run_metadata": {
            "target":              TARGET_COUNT,
            "started":             datetime.now().isoformat(),
            "last_updated":        "",
            "batches_completed":   0,
            "prompt_corrections":  [],
        },
        "markets": [],
    }


def save_master(data: dict) -> None:
    data["run_metadata"]["last_updated"] = datetime.now().isoformat()
    with open(MASTER_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Step 1-A — Market suggestion (one Claude call for a full batch)
# ---------------------------------------------------------------------------

def suggest_markets_batch(
    client: anthropic.Anthropic,
    existing_domains: list,
    count: int,
    ref_years: list,
) -> list:
    """
    Ask Claude to propose `count` new venture market seeds.
    Returns list of {domain, ref_year, knowledge_brief}.
    """
    existing_block = (
        "\n".join(f"  - {d}" for d in existing_domains)
        if existing_domains else "  (none yet)"
    )

    prompt = (
        "You are building a 120-market historical venture reference database "
        "covering 2005–2019. Each entry must be a distinct, real, "
        "venture-backed sector with verifiable historical data.\n\n"
        "=== ALREADY IN DATABASE — DO NOT REPEAT ===\n"
        f"{existing_block}\n\n"
        f"=== TASK ===\n"
        f"Propose exactly {count} NEW market opportunities. Requirements:\n"
        "- Diverse sectors: fintech, healthtech, enterprise SaaS, consumer internet, "
        "deep tech, marketplace, climate, edtech, logistics, etc.\n"
        f"- Reference years drawn from: {ref_years}\n"
        "- Spread years across the full 2005–2019 range (avoid clustering).\n"
        "- Include a mix of US and international (EU, UK, China, India, etc.) markets.\n"
        "- Avoid speculative or hypothetical markets — must be real, verifiable sectors.\n\n"
        "For each market provide:\n"
        "1. domain: a precise 10-20 word description of the product/service category.\n"
        "2. ref_year: integer founding/inflection year from the list above.\n"
        "3. knowledge_brief: 3-5 sentences with specific verifiable facts — company names, "
        "funding amounts, user counts, key regulatory events, enabling technology state "
        "at that reference year.  Must be historically accurate.\n\n"
        "Return ONLY a valid JSON array.  No markdown, no preamble:\n"
        "[\n"
        '  {"domain": "...", "ref_year": <int>, "knowledge_brief": "..."},\n'
        "  ...\n"
        "]\n\n"
        f"Return exactly {count} entries."
    )

    raw = ""
    try:
        response = _claude_retry(
            lambda: client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=3500,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        raw = response.content[0].text.strip()
    except Exception as exc:
        print(f"  [suggest_markets ERROR] {exc}")
        return []

    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    brace = raw.find("[")
    if brace > 0:
        raw = raw[brace:]

    try:
        suggestions = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  [suggest_markets PARSE ERROR] {exc}  raw[:300]={raw[:300]!r}")
        return []

    valid = []
    seen_domains = set(d.lower() for d in existing_domains)
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        domain  = s.get("domain", "")
        ref_yr  = s.get("ref_year")
        brief   = s.get("knowledge_brief", "")
        # Coerce ref_year to int if returned as string
        try:
            ref_yr = int(ref_yr)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(domain, str) and len(domain) > 5
            and ref_yr in ref_years
            and isinstance(brief, str) and len(brief) > 20
            and domain.lower() not in seen_domains
        ):
            valid.append({"domain": domain, "ref_year": ref_yr, "knowledge_brief": brief})
            seen_domains.add(domain.lower())

    print(f"  Suggested {len(valid)} valid / {len(suggestions)} returned")
    return valid[:count]


# ---------------------------------------------------------------------------
# Step 1-B — Generate base profile for one seed
# ---------------------------------------------------------------------------

def generate_market_profile(
    client: anthropic.Anthropic,
    seed: dict,
) -> dict:
    """
    Generate a market base profile.  Mirrors pipeline_step1.generate_base_profile
    but works with a Claude-generated knowledge_brief rather than a hand-crafted one.
    """
    prompt = (
        "You are a senior venture capital market analyst producing a historical "
        "market profile for a quantitative classification system. "
        "Ground all claims in the knowledge brief — do not invent details.\n\n"
        f"Domain: {seed['domain']}\n"
        f"Reference year: {seed['ref_year']}\n\n"
        "=== KNOWLEDGE BRIEF ===\n"
        f"{seed['knowledge_brief'].strip()}\n\n"
        "=== OUTPUT FORMAT ===\n"
        "Use EXACTLY these labelled fields, each starting on its own line:\n\n"
        "Market: [concise 4-8 word market name, no year]\n\n"
        "Context: [3-4 sentences: state of market at reference year, key inflection "
        "point, investor sentiment, one landmark event.]\n\n"
        "Buyers: [2-3 sentences: primary beachhead buyer segment with specifics, "
        "their pain point, purchasing behaviour at this year.]\n\n"
        "Players: [2-3 sentences: specific companies with funding/revenue/customer "
        "counts from the brief.  Which major players had NOT yet entered.]\n\n"
        "Key Metrics: [3-5 bullet points from the knowledge brief:\n"
        "- [metric]: [value] ([source/year])]\n\n"
        "Exclusions: [2 sentences: what was NOT yet true at this reference year — "
        "competitors not launched, regulations not enacted, tech not shipped.]\n\n"
        f"Reference year: {seed['ref_year']}\n\n"
        "Rules: do not project post-reference-year events; use specific numbers and dates."
    )

    try:
        response = _claude_retry(
            lambda: client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        raw_text = response.content[0].text.strip()
    except Exception as exc:
        return {
            "raw_text": "",
            "ref_year": seed["ref_year"],
            "domain":   seed["domain"],
            "market_name": seed["domain"][:60],
            "error": str(exc),
        }

    profile: dict = {"raw_text": raw_text, "ref_year": seed["ref_year"], "domain": seed["domain"]}

    field_map = {
        "Market:":         "market_name",
        "Context:":        "context",
        "Buyers:":         "buyers",
        "Players:":        "players",
        "Key Metrics:":    "key_metrics",
        "Exclusions:":     "exclusions",
        "Reference year:": "reference_year_label",
    }
    current_field = None
    current_lines: list = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        matched  = False
        for label, key in field_map.items():
            if stripped.startswith(label):
                if current_field:
                    profile[current_field] = "\n".join(current_lines).strip()
                current_field = key
                current_lines = [stripped[len(label):].strip()]
                matched = True
                break
        if not matched and current_field and stripped:
            current_lines.append(stripped)
    if current_field:
        profile[current_field] = "\n".join(current_lines).strip()

    if not profile.get("market_name"):
        profile["market_name"] = seed["domain"][:60]

    return profile


# ---------------------------------------------------------------------------
# Step 1-C — Extract one dimension with optional correction addendum
# ---------------------------------------------------------------------------

def extract_dimension_scaled(
    client: anthropic.Anthropic,
    profile: dict,
    dimension: dict,
    correction_addendum: str = "",
) -> dict:
    """
    Mirrors pipeline_step1.extract_dimension but accepts an optional
    correction_addendum injected after the scoring guide.
    """
    addendum_block = ""
    if correction_addendum:
        addendum_block = (
            "\n\n=== CALIBRATION NOTE (auto-corrected from prior batch) ===\n"
            + correction_addendum.strip()
            + "\n"
        )

    prompt = (
        "You are a senior quantitative venture capital researcher performing structured "
        "market classification. Your classifications will be cross-validated against "
        "independently retrieved evidence — precision and intellectual honesty are essential.\n\n"
        "=== MARKET PROFILE ===\n"
        f"{profile['raw_text']}\n\n"
        "=== DIMENSION TO CLASSIFY ===\n"
        f"Dimension:   {dimension['name']}\n"
        f"Description: {dimension['description']}\n\n"
        f"Scoring guide:\n{dimension['scoring_guide']}"
        + addendum_block
        + "\n=== CLASSIFICATION TASK ===\n"
        f"Choose ONE value from this exact list: {dimension['options']}\n\n"
        "Return ONLY a valid JSON object — no markdown, no preamble, no trailing text.\n"
        "{\n"
        f'  "dimension": "{dimension["name"]}",\n'
        '  "classification": "<one value from the options list>",\n'
        '  "confidence": "<high|medium|low>",\n'
        '  "rationale": "<3 sentences: evidence from profile, scale position, key anchor fact>",\n'
        '  "contradicting_evidence": "<1-2 sentences: adjacent classification argument and why rejected>"\n'
        "}"
    )

    response = _claude_retry(
        lambda: client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    if parsed.get("classification") not in dimension["options"]:
        parsed["classification"] = dimension["options"][0]
        parsed["validation_warning"] = "Classification not in options; defaulted to first."
    return parsed


# ---------------------------------------------------------------------------
# Step 1 — Full market: profile + 7 dimension extractions
# ---------------------------------------------------------------------------

def run_step1_market(
    client: anthropic.Anthropic,
    seed: dict,
    market_id: str,
    correction_addendum: str = "",
) -> dict:
    """Run full Step 1 for one market seed. Returns market dict or None on failure."""
    profile = generate_market_profile(client, seed)
    if not profile.get("raw_text"):
        print(f"    PROFILE FAILED: {profile.get('error','unknown error')}")
        return None
    print(f"    Profile OK → {profile.get('market_name','?')}")

    dimensions_result = {}
    for dim in DIMENSIONS:
        print(f"      [{dim['name']}]...", end=" ", flush=True)
        time.sleep(0.4)
        try:
            result = extract_dimension_scaled(client, profile, dim, correction_addendum)
            dimensions_result[dim["name"]] = result
            print(f"{result['classification']} [{result.get('confidence','?')}]")
        except Exception as exc:
            print(f"ERR: {exc}")
            dimensions_result[dim["name"]] = {
                "dimension":              dim["name"],
                "classification":         "unknown",
                "confidence":             "low",
                "rationale":              "",
                "contradicting_evidence": "",
                "error":                  str(exc),
            }

    return {
        "id":             market_id,
        "domain":         seed["domain"],
        "ref_year":       seed["ref_year"],
        "base_profile":   profile,
        "dimensions":     dimensions_result,
        "step1_complete": True,
    }


# ---------------------------------------------------------------------------
# Step 2 — Gemini verification (thin wrapper around existing function)
# ---------------------------------------------------------------------------

def run_step2_market(gemini_client, market: dict) -> dict:
    """Verify one market via Gemini.  Returns the step2 result block."""
    return process_market_verification(gemini_client, market)


# ---------------------------------------------------------------------------
# Step 3 — Claude blind scorer (mirrors run_ablation_study.score_from_matrix)
# ---------------------------------------------------------------------------

def run_step3_market(
    claude_client: anthropic.Anthropic,
    market: dict,
) -> tuple:
    """
    Score the anonymised feature matrix.
    Returns (step3_result_dict, prompt_string).
    """
    matrix = build_feature_matrix(market)
    prompt = build_scoring_prompt(matrix)

    try:
        response = _claude_retry(
            lambda: claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=400,
                system=SCORER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        )
    except Exception as exc:
        return {
            "scores":            {dim: None for dim in _DIM_NAMES},
            "feature_matrix":    matrix,
            "validation_errors": [str(exc)],
        }, prompt

    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "scores":            {dim: None for dim in _DIM_NAMES},
            "feature_matrix":    matrix,
            "validation_errors": [f"JSON parse error: {exc}"],
            "raw_response":      raw,
        }, prompt

    scores = {}
    errors = []
    for dim in _DIM_NAMES:
        val = parsed.get(dim)
        if val is None:
            errors.append(f"missing key '{dim}'")
            scores[dim] = None
        else:
            try:
                scores[dim] = max(0, min(100, int(val)))
            except (TypeError, ValueError):
                errors.append(f"'{dim}' invalid: {val!r}")
                scores[dim] = None

    return {
        "scores":            scores,
        "feature_matrix":    matrix,
        "validation_errors": errors,
    }, prompt


# ---------------------------------------------------------------------------
# Agreement analysis — batch-level stats
# ---------------------------------------------------------------------------

def compute_batch_agreement(batch_markets: list) -> dict:
    """
    Compute aggregate agreement statistics across all markets in a batch.
    Uses the same scoring formula as pipeline_step2 (HIGH=1.0, MEDIUM=0.5, LOW=0.0).
    Returns {mean_score, overall, HIGH, MEDIUM, LOW, total, per_dim}.
    """
    counts  = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    per_dim = {name: {"HIGH": 0, "MEDIUM": 0, "LOW": 0} for name in _DIM_NAMES}

    for m in batch_markets:
        dim_v = m.get("step2", {}).get("dimension_verifications", {})
        for dim_name, dv in dim_v.items():
            agr = dv.get("agreement", "LOW")
            if agr in counts:
                counts[agr] += 1
            if dim_name in per_dim and agr in per_dim[dim_name]:
                per_dim[dim_name][agr] += 1

    total = sum(counts.values())
    score = (counts["HIGH"] * 1.0 + counts["MEDIUM"] * 0.5) / total if total else 0.0

    if score >= 0.70:
        overall = "HIGH"
    elif score >= 0.40:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    return {
        "mean_score": round(score, 3),
        "overall":    overall,
        "HIGH":       counts["HIGH"],
        "MEDIUM":     counts["MEDIUM"],
        "LOW":        counts["LOW"],
        "total":      total,
        "per_dim":    per_dim,
    }


# ---------------------------------------------------------------------------
# Auto-correction — meta-prompt to rewrite Step 1 extraction addendum
# ---------------------------------------------------------------------------

def run_meta_correction(
    claude_client: anthropic.Anthropic,
    batch_markets: list,
    current_addendum: str,
    agreement_stats: dict,
) -> str:
    """
    Feed conflicting Claude-vs-Gemini cases back to Claude.
    Returns a new extraction_addendum string (or current one on failure).
    """
    conflict_lines = []
    for m in batch_markets:
        name  = m.get("base_profile", {}).get("market_name", m.get("domain", "?"))
        ref   = m.get("ref_year", "?")
        dim_v = m.get("step2", {}).get("dimension_verifications", {})
        for dim_name, dv in dim_v.items():
            agr = dv.get("agreement", "HIGH")
            if agr in ("LOW", "MEDIUM"):
                conflict_lines.append(
                    f"  [{dim_name}] market={name!r} year={ref} | "
                    f"agreement={agr} | "
                    f"claude={dv.get('claude_classification','?')!r} | "
                    f"gemini={dv.get('gemini_classification','?')!r} | "
                    f"evidence={dv.get('key_fact','')[:80]!r}"
                )

    if not conflict_lines:
        return current_addendum

    conflicts_text = "\n".join(conflict_lines[:40])   # cap at 40 cases

    prompt = (
        "You are calibrating a venture market classification pipeline.\n\n"
        "Step 1 (Claude) classifies 7 dimensions from a market profile.\n"
        "Step 2 (Gemini + Google Search) independently verifies with live web evidence.\n"
        "When they disagree, Gemini's web-grounded answer is the ground truth.\n\n"
        "=== CURRENT STEP 1 EXTRACTION ADDENDUM ===\n"
        f"{current_addendum if current_addendum else '(none)'}\n\n"
        "=== BATCH AGREEMENT STATS ===\n"
        f"Mean score: {agreement_stats['mean_score']:.3f}  "
        f"(overall: {agreement_stats['overall']})\n"
        f"HIGH: {agreement_stats['HIGH']}  MEDIUM: {agreement_stats['MEDIUM']}  "
        f"LOW: {agreement_stats['LOW']}\n\n"
        "=== CONFLICTING CASES (Claude vs Gemini ground truth) ===\n"
        f"{conflicts_text}\n\n"
        "=== YOUR TASK ===\n"
        "Analyse the pattern of disagreements. Write a SHORT calibration note "
        "(3-6 bullet points, plain text) to be injected into the Step 1 prompt. "
        "The note should tell Claude where its systematic biases lie and how to "
        "correct them to better match web-grounded evidence.\n\n"
        "Focus on PATTERNS across multiple markets and dimensions — not individual cases.\n"
        "Return ONLY the bullet-point calibration note. No preamble, no headers."
    )

    try:
        response = _claude_retry(
            lambda: claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        new_addendum = response.content[0].text.strip()
        print(f"\n  [Auto-correction] New addendum generated ({len(new_addendum)} chars)")
        return new_addendum
    except Exception as exc:
        print(f"  [Auto-correction ERROR] {exc}")
        return current_addendum


# ---------------------------------------------------------------------------
# Markdown batch log
# ---------------------------------------------------------------------------

def _url_cell(url: str) -> str:
    if not url:
        return "_no URL_"
    display = url if len(url) <= 80 else url[:77] + "..."
    return f"[link]({url})" if url.startswith("http") else display


def append_batch_log(
    batch_num: int,
    batch_markets: list,
    agreement_stats: dict,
    correction_triggered: bool,
    new_addendum: str,
    role3_audit_prompt: str,
    run_ts: str,
    total_accepted: int,
) -> None:
    """Append a complete batch summary to lab_notes/Scaling_Run_Log.md."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    lines = [
        "",
        "---",
        "",
        f"## Batch {batch_num}  —  {run_ts}",
        "",
        f"**Markets accepted this batch:** {len(batch_markets)}  ",
        f"**Cumulative total:** {total_accepted} / {TARGET_COUNT}  ",
        f"**Batch agreement:** {agreement_stats['overall']} "
        f"(score={agreement_stats['mean_score']:.3f})  ",
        f"**HIGH:** {agreement_stats['HIGH']}  "
        f"**MEDIUM:** {agreement_stats['MEDIUM']}  "
        f"**LOW:** {agreement_stats['LOW']}  ",
        f"**Auto-correction triggered:** {'**YES**' if correction_triggered else 'no'}",
        "",
    ]

    if correction_triggered and new_addendum:
        lines += [
            "### Auto-Correction — Revised Extraction Addendum",
            "",
            "```",
            new_addendum,
            "```",
            "",
        ]

    # Per-market detail sections
    for m in batch_markets:
        name    = m.get("base_profile", {}).get("market_name", m.get("domain", "?"))
        ref     = m.get("ref_year", "?")
        step2   = m.get("step2", {})
        step3   = m.get("step3", {})
        dim_v   = step2.get("dimension_verifications", {})
        scores  = step3.get("scores", {})
        fm      = step3.get("feature_matrix", {})
        outcome = step2.get("outcome_verification", {})
        agr_sum = step2.get("agreement_summary", {})

        lines += [
            f"### {name} ({ref})",
            "",
            f"**T+5 outcome:** {outcome.get('verification_status','?')} "
            f"({outcome.get('source_count',0)} sources)  ",
            f"**Agreement:** {agr_sum.get('overall','?')} "
            f"(score={agr_sum.get('score',0):.2f}  "
            f"HIGH={agr_sum.get('HIGH',0)} MEDIUM={agr_sum.get('MEDIUM',0)} "
            f"LOW={agr_sum.get('LOW',0)})",
            "",
        ]

        # Role 2 verification table — includes clickable URLs
        lines += [
            "#### Role 2 Evidence & Verification URLs",
            "",
            "| Dimension | Claude | Gemini | Agreement | Key Fact | Verification URL |",
            "|-----------|--------|--------|:---------:|----------|-----------------|",
        ]
        for dim in _DIM_NAMES:
            dv  = dim_v.get(dim, {})
            cc  = dv.get("claude_classification", "?")
            gc  = dv.get("gemini_classification", "?")
            agr = dv.get("agreement", "?")
            kf  = dv.get("key_fact", "") or dv.get("evidence", "")
            kf  = (kf[:80] + "...") if len(kf) > 80 else kf
            url = dv.get("verification_url", "") or (dv.get("grounding_urls") or [""])[0]
            lines.append(
                f"| `{dim}` | `{cc}` | `{gc}` | {agr} | {kf} | {_url_cell(url)} |"
            )

        # Role 3 scores table
        lines += [
            "",
            "#### Role 3 Scores",
            "",
            "| Dimension | Verified Classification | Score |",
            "|-----------|------------------------|:-----:|",
        ]
        for dim in _DIM_NAMES:
            val  = fm.get(dim, {}).get("value", "?")
            sc   = scores.get(dim)
            sc_s = str(sc) if sc is not None else "ERR"
            lines.append(f"| `{dim}` | `{val}` | {sc_s} |")

        valid_sc = [v for v in scores.values() if v is not None]
        if valid_sc:
            mean_sc = round(sum(valid_sc) / len(valid_sc))
            lines.append(f"| **Mean** | — | **{mean_sc}** |")

        lines.append("")

    # Role 3 Prompt Audit
    if role3_audit_prompt and batch_markets:
        first = batch_markets[0].get("base_profile", {}).get("market_name", "Market 1")
        lines += [
            f"### Role 3 Prompt Audit — {first}",
            "",
            "_Exact prompt sent to Claude scorer. Proves market name and all identifying "
            "information are stripped before scoring._",
            "",
            "```",
            role3_audit_prompt,
            "```",
            "",
        ]

    # Per-dimension agreement breakdown
    lines += [
        "### Batch Agreement by Dimension",
        "",
        "| Dimension | HIGH | MEDIUM | LOW |",
        "|-----------|:----:|:------:|:---:|",
    ]
    for dim_name, cnts in agreement_stats.get("per_dim", {}).items():
        lines.append(
            f"| `{dim_name}` | {cnts['HIGH']} | {cnts['MEDIUM']} | {cnts['LOW']} |"
        )

    lines.append("")

    # Write (create with header if first batch, else append)
    header_needed = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        if header_needed:
            fh.write(
                "# Vela MQR — Scale Pipeline Run Log (120 Markets)\n\n"
                f"Target: {TARGET_COUNT} markets  |  Batch size: {BATCH_SIZE}  |  "
                "Ref years: 2005–2019  |  Method: Baseline\n"
            )
        fh.write("\n".join(lines))
        fh.write("\n")

    print(f"  + Appended batch {batch_num} to Scaling_Run_Log.md")


# ---------------------------------------------------------------------------
# Git auto-push
# ---------------------------------------------------------------------------

def auto_git_push(commit_message: str) -> None:
    print(f"\n  [Git] {commit_message!r}")
    for args in [
        ["git", "add", "."],
        ["git", "commit", "-m", commit_message],
        ["git", "push", "origin", "main"],
    ]:
        result  = subprocess.run(args, capture_output=True, text=True, cwd=_ROOT)
        combined = (result.stdout + result.stderr).strip()
        if result.returncode != 0 and "nothing to commit" not in combined:
            print(f"  [Git] Error ({' '.join(args)}): {combined[:200]}")
        elif combined:
            print(f"  [Git] {combined[:200]}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  VELA MQR — SCALE PIPELINE  (Baseline, 120 Markets)")
    print("=" * 70)

    # Load or initialise master dataset
    data     = load_master()
    markets  = data["markets"]
    metadata = data["run_metadata"]

    total_accepted    = len(markets)
    batch_num         = metadata.get("batches_completed", 0)
    correction_addendum = ""

    # Restore last correction addendum if pipeline is being resumed
    for rec in metadata.get("prompt_corrections", []):
        correction_addendum = rec.get("addendum", correction_addendum)

    print(f"\n  Resuming from : {total_accepted} accepted markets")
    print(f"  Batches done  : {batch_num}")
    print(f"  Target        : {TARGET_COUNT}")
    if correction_addendum:
        print(f"  Active addendum: {correction_addendum[:80]}...")

    claude_client = get_claude_client()
    gemini_client = get_gemini_client()

    while total_accepted < TARGET_COUNT:
        batch_num += 1
        run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'─' * 70}")
        print(
            f"  BATCH {batch_num}  |  "
            f"Accepted: {total_accepted}/{TARGET_COUNT}  |  {run_ts}"
        )
        print(f"{'─' * 70}")

        # ── Step 0: Suggest batch seeds ────────────────────────────────
        existing_domains = [m.get("domain", "") for m in markets]
        print(f"\n  Suggesting {BATCH_SIZE} new markets ...")
        suggestions = suggest_markets_batch(
            claude_client, existing_domains, BATCH_SIZE, REF_YEARS
        )
        if not suggestions:
            print("  WARNING: no valid suggestions.  Waiting 30s then retrying ...")
            time.sleep(30)
            continue

        batch_markets       = []
        role3_audit_prompt  = ""

        for i, seed in enumerate(suggestions):
            if total_accepted >= TARGET_COUNT:
                break

            mkt_num = total_accepted + 1
            print(
                f"\n  [{i+1}/{len(suggestions)}] Market #{mkt_num}: "
                f"{seed['domain'][:55]}  ({seed['ref_year']})"
            )

            # ── Step 1 ────────────────────────────────────────────────
            print("  [Step 1: Generate & Extract]")
            market = run_step1_market(
                claude_client, seed,
                f"market_{mkt_num:03d}",
                correction_addendum,
            )
            if market is None:
                print("  SKIPPED (Step 1 failed)")
                continue

            # ── Step 2 ────────────────────────────────────────────────
            print("  [Step 2: Gemini Verify]")
            try:
                step2_result = run_step2_market(gemini_client, market)
                market["step2"] = step2_result
            except Exception as exc:
                print(f"  SKIPPED (Step 2 error: {exc})")
                continue

            # Acceptance gate: T+5 must have >= 3 grounded sources
            outcome = step2_result.get("outcome_verification", {})
            if outcome.get("verification_status") != "VERIFIED":
                print(
                    f"  REJECTED: {outcome.get('rejection_reason', '<3 grounded sources')}"
                )
                continue

            # ── Step 3 ────────────────────────────────────────────────
            print("  [Step 3: Score]")
            step3_result, s3_prompt = run_step3_market(claude_client, market)
            market["step3"] = step3_result
            valid_sc = [v for v in step3_result["scores"].values() if v is not None]
            mean_sc  = round(sum(valid_sc) / len(valid_sc)) if valid_sc else None
            print(f"  Scores: {list(step3_result['scores'].values())}  mean={mean_sc}")

            if not role3_audit_prompt:
                role3_audit_prompt = s3_prompt

            # ── Accept & save ──────────────────────────────────────────
            batch_markets.append(market)
            markets.append(market)
            total_accepted += 1

            # Incremental save after every accepted market
            data["markets"] = markets
            save_master(data)

            time.sleep(0.5)

        # ── Batch agreement analysis ───────────────────────────────────
        if not batch_markets:
            print("  WARNING: no markets accepted this batch.  Continuing ...")
            continue

        agreement_stats = compute_batch_agreement(batch_markets)
        print(
            f"\n  Batch agreement: {agreement_stats['overall']}  "
            f"(score={agreement_stats['mean_score']:.3f}  "
            f"HIGH={agreement_stats['HIGH']} MEDIUM={agreement_stats['MEDIUM']} "
            f"LOW={agreement_stats['LOW']})"
        )

        # ── Auto-correction ────────────────────────────────────────────
        correction_triggered = False
        if agreement_stats["mean_score"] < LOW_BATCH_THRESHOLD:
            print(
                "  [Auto-correction TRIGGERED] Mean agreement below threshold "
                f"({agreement_stats['mean_score']:.3f} < {LOW_BATCH_THRESHOLD}) ..."
            )
            new_addendum = run_meta_correction(
                claude_client, batch_markets, correction_addendum, agreement_stats
            )
            if new_addendum != correction_addendum:
                correction_addendum  = new_addendum
                correction_triggered = True
                metadata.setdefault("prompt_corrections", []).append({
                    "batch":         batch_num,
                    "timestamp":     datetime.now().isoformat(),
                    "trigger_score": agreement_stats["mean_score"],
                    "addendum":      correction_addendum,
                })

        # ── Update metadata & save ─────────────────────────────────────
        metadata["batches_completed"] = batch_num
        data["run_metadata"] = metadata
        data["markets"]      = markets
        save_master(data)

        # ── Append batch log ───────────────────────────────────────────
        append_batch_log(
            batch_num           = batch_num,
            batch_markets       = batch_markets,
            agreement_stats     = agreement_stats,
            correction_triggered= correction_triggered,
            new_addendum        = correction_addendum if correction_triggered else "",
            role3_audit_prompt  = role3_audit_prompt,
            run_ts              = run_ts,
            total_accepted      = total_accepted,
        )

        # ── Git push ───────────────────────────────────────────────────
        auto_git_push(
            f"Auto-commit: Batch {batch_num} — "
            f"{len(batch_markets)} markets accepted "
            f"({total_accepted}/{TARGET_COUNT} total)"
        )

        print(
            f"\n  Batch {batch_num} done: {len(batch_markets)} accepted  "
            f"| Total: {total_accepted}/{TARGET_COUNT}"
        )

    # ── Pipeline complete ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SCALE PIPELINE COMPLETE")
    print(f"  Markets accepted : {total_accepted}")
    print(f"  Batches run      : {batch_num}")
    print(f"  Master JSON      : {MASTER_JSON}")
    print(f"  Run log          : {LOG_PATH}")
    print(f"{'=' * 70}\n")

    # Update LOGBOOK_MASTER
    logbook = os.path.join(_ROOT, "lab_notes", "LOGBOOK_MASTER.md")
    if os.path.exists(logbook):
        with open(logbook, "a", encoding="utf-8") as fh:
            fh.write(
                f"* [Scale Pipeline — {total_accepted} markets "
                f"({batch_num} batches)](./Scaling_Run_Log.md)\n"
            )

    auto_git_push(f"Auto-commit: Scale pipeline complete — {total_accepted} markets")


if __name__ == "__main__":
    main()
