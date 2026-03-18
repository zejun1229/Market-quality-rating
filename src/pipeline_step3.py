"""
Vela Market Quality Rating System — Step 3
Scorer & Rater

Reads the Step 2 enriched JSON, anonymizes each market to its verified
categorical feature matrix, calls Claude to score all 7 dimensions on a
0–100 venture-readiness scale, and saves the fully enriched dataset as
reference_population_scored.json.

Anonymization contract
----------------------
Stripped (never sent to LLM scorer):
  - market_name, domain, base profile text
  - ref_year, T+5 outcome summaries
  - any Role-1 rationale or contradicting_evidence
  - any Role-2 evidence strings or grounding URLs

Passed to LLM scorer (only):
  - 7 verified categorical dimension values
  - per-dimension inter-source agreement level (HIGH / MEDIUM / LOW)
"""

import json
import os
import re
import sys
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"

DIMENSIONS = [
    "timing",
    "competition",
    "market_size",
    "customer_readiness",
    "regulatory",
    "infrastructure",
    "market_structure",
]

SCORE_RUBRIC = (
    "90-100 (L5 - Exceptional): Ideal conditions for outsized venture returns.\n"
    "75-89  (L4 - Strong):      Highly attractive, minor friction.\n"
    "60-74  (L3 - Favorable):   The baseline investment-grade threshold. "
    "Viable but requires execution excellence.\n"
    "40-59  (L2 - Neutral):     Sub-optimal conditions, significant structural headwinds.\n"
    "0-39   (L1 - Speculative): Hostile conditions for new venture entrants."
)

SYSTEM_PROMPT = (
    "You are an objective quantitative venture analyst scoring anonymous market data. "
    "You are evaluating a market's readiness for venture-scale entry based solely on "
    "structured categorical variables. Score each of the 7 dimensions from 0 to 100 "
    "based purely on the structured features provided. Use this standard:\n\n"
    + SCORE_RUBRIC
)

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _call_with_retry(fn, max_retries: int = 4, base_delay: float = 20.0):
    """Call fn() with retry on Claude overload (529) errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            exc_str = str(exc).lower()
            is_overloaded = "529" in exc_str or "overloaded" in exc_str
            if is_overloaded and attempt < max_retries:
                delay = base_delay * (attempt + 1)
                print(
                    f"\n    [529 overloaded; retry {attempt + 1}/{max_retries} "
                    f"in {delay:.0f}s]",
                    end=" ", flush=True,
                )
                time.sleep(delay)
            else:
                raise


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your_"):
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY is not set or is a placeholder. "
            "Please update your .env file."
        )
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------

def build_feature_matrix(market: dict) -> dict:
    """
    Build the anonymized, cross-validated feature matrix.

    Per-dimension selection priority:
      1. Gemini's verified classification (step2) when present and not 'unknown'
      2. Claude's original classification (step1 dimensions block) as fallback

    Returns:
      {dim_name: {"value": str, "agreement": str}}
    """
    step1_dims = market.get("dimensions", {})
    step2_verifs = market.get("step2", {}).get("dimension_verifications", {})

    matrix = {}
    for dim in DIMENSIONS:
        s1 = step1_dims.get(dim, {})
        s2 = step2_verifs.get(dim, {})

        gemini_class = s2.get("gemini_classification", "")
        claude_class = s1.get("classification", "unknown")
        agreement = s2.get("agreement", "unverified")

        # Prefer Gemini-verified value; fall back to Claude
        value = (
            gemini_class
            if gemini_class and gemini_class not in ("unknown", "")
            else claude_class
        )

        matrix[dim] = {"value": value, "agreement": agreement}

    return matrix


# ---------------------------------------------------------------------------
# Scoring prompt
# ---------------------------------------------------------------------------

def build_scoring_prompt(matrix: dict) -> str:
    """Construct the user-turn prompt from the anonymized feature matrix."""
    lines = [
        "=== ANONYMIZED MARKET FEATURE MATRIX ===",
        "",
        f"  {'DIMENSION':<24}  {'VERIFIED CLASSIFICATION':<36}  INTER-SOURCE AGREEMENT",
        f"  {'-'*24}  {'-'*36}  {'-'*22}",
    ]
    for dim, entry in matrix.items():
        lines.append(
            f"  {dim:<24}  {entry['value']:<36}  {entry['agreement']}"
        )
    lines += [
        "",
        "=== SCORING TASK ===",
        "Score each of the 7 dimensions from 0 to 100 using the L1–L5 rubric in your system prompt.",
        "Base your scores ONLY on the categorical values above — do not infer market identity.",
        "",
        "Return ONLY a valid JSON object — no markdown, no commentary, no trailing text.",
        "All values must be integers between 0 and 100.",
        "{",
        '  "timing": <integer 0-100>,',
        '  "competition": <integer 0-100>,',
        '  "market_size": <integer 0-100>,',
        '  "customer_readiness": <integer 0-100>,',
        '  "regulatory": <integer 0-100>,',
        '  "infrastructure": <integer 0-100>,',
        '  "market_structure": <integer 0-100>',
        "}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Score one market
# ---------------------------------------------------------------------------

def score_market(client: anthropic.Anthropic, market: dict, index: int) -> dict:
    """
    Anonymize, score via Claude, validate, and return the step3 result block.
    """
    matrix = build_feature_matrix(market)
    prompt = build_scoring_prompt(matrix)

    print(f"\n[Market {index + 1}]  Scoring anonymized matrix ...", end=" ", flush=True)

    try:
        response = _call_with_retry(
            lambda: client.messages.create(
                model=MODEL,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        )
    except Exception as exc:
        print(f"API ERROR: {exc}")
        return {
            "scores": {},
            "feature_matrix": matrix,
            "composite_score": None,
            "composite_level": "ERROR",
            "validation_errors": [str(exc)],
        }

    raw = response.content[0].text.strip()
    # Strip markdown fences if model adds them
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Parse JSON
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"PARSE ERROR: {exc}  raw={raw[:120]!r}")
        return {
            "scores": {},
            "feature_matrix": matrix,
            "composite_score": None,
            "composite_level": "ERROR",
            "validation_errors": [f"JSON parse error: {exc}"],
            "raw_response": raw,
        }

    # Validate: exactly 7 keys, integers in [0, 100]
    validation_errors = []
    scores = {}
    for dim in DIMENSIONS:
        if dim not in parsed:
            validation_errors.append(f"missing key '{dim}'")
            scores[dim] = None
            continue
        val = parsed[dim]
        if not isinstance(val, int):
            try:
                val = int(val)
                validation_errors.append(f"'{dim}' coerced from {type(parsed[dim]).__name__} to int")
            except (TypeError, ValueError):
                validation_errors.append(f"'{dim}' cannot be converted to int: {val!r}")
                scores[dim] = None
                continue
        if not (0 <= val <= 100):
            validation_errors.append(f"'{dim}' clamped from {val} to valid range [0, 100]")
            val = max(0, min(100, val))
        scores[dim] = val

    valid_scores = [v for v in scores.values() if v is not None]
    composite = round(sum(valid_scores) / len(valid_scores)) if valid_scores else None
    level = _score_to_level(composite)

    if validation_errors:
        print(f"WARNINGS: {'; '.join(validation_errors)}")
    print(f"composite={composite}/100  [{level}]")

    return {
        "scores": scores,
        "feature_matrix": matrix,
        "composite_score": composite,
        "composite_level": level,
        "validation_errors": validation_errors,
    }


# ---------------------------------------------------------------------------
# Level label
# ---------------------------------------------------------------------------

def _score_to_level(score) -> str:
    if score is None:
        return "N/A"
    if score >= 90:
        return "L5-Exceptional"
    if score >= 75:
        return "L4-Strong"
    if score >= 60:
        return "L3-Favorable"
    if score >= 40:
        return "L2-Neutral"
    return "L1-Speculative"


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(markets: list) -> None:
    divider = "=" * 76
    thin    = "-" * 76

    print(f"\n\n{divider}")
    print("  VELA MARKET QUALITY RATING — STEP 3  SCORER & RATER REPORT")
    print(divider)

    for market in markets:
        s3      = market.get("step3", {})
        scores  = s3.get("scores", {})
        fm      = s3.get("feature_matrix", {})
        profile = market.get("base_profile", {})
        name    = profile.get("market_name", market.get("domain", "Unknown"))
        ref     = market.get("ref_year", "?")
        comp    = s3.get("composite_score")
        level   = s3.get("composite_level", "N/A")

        print(f"\n{thin}")
        print(f"  Market : {name}")
        print(f"  Year   : {ref}   |   Composite : {comp}/100  [{level}]")
        print(f"\n  {'Dimension':<24}  {'Score':>5}  {'Level':<18}  {'Verified Value':<30}  Agreement")
        print(f"  {'-'*24}  {'-'*5}  {'-'*18}  {'-'*30}  {'-'*10}")

        for dim in DIMENSIONS:
            sc  = scores.get(dim)
            lv  = _score_to_level(sc)
            val = fm.get(dim, {}).get("value", "?")
            agr = fm.get(dim, {}).get("agreement", "?")
            sc_str = str(sc) if sc is not None else "ERR"
            print(f"  {dim:<24}  {sc_str:>5}  {lv:<18}  {val:<30}  {agr}")

        if s3.get("validation_errors"):
            print(f"\n  Validation notes: {s3['validation_errors']}")

    # League table
    def _composite(m):
        return m.get("step3", {}).get("composite_score") or 0

    print(f"\n{divider}")
    print("  COMPOSITE SCORE LEAGUE TABLE  (highest → lowest)")
    print(f"  {'Market':<42}  {'Year':>4}  {'Score':>5}  Level")
    print(f"  {'-'*42}  {'-'*4}  {'-'*5}  {'-'*18}")

    for m in sorted(markets, key=_composite, reverse=True):
        s3   = m.get("step3", {})
        comp = s3.get("composite_score")
        lv   = s3.get("composite_level", "N/A")
        name = m.get("base_profile", {}).get(
            "market_name", m.get("domain", "?")
        )[:42]
        ref  = m.get("ref_year", "?")
        print(f"  {name:<42}  {ref:>4}  {str(comp):>5}  {lv}")

    print(f"\n{divider}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  VELA MARKET QUALITY RATING — STEP 3")
    print("  Scorer & Rater  (Claude, anonymized input)")
    print("=" * 65)

    # Locate input: prefer v3, fall back to root, then src/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, "reference_population_v3.json"),
        os.path.join(project_root, "reference_population.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_population.json"),
    ]
    json_path = next((p for p in candidates if os.path.exists(p)), None)
    if not json_path:
        sys.exit(
            "ERROR: No reference_population JSON found. "
            "Run pipeline_step1.py and pipeline_step2.py first."
        )

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    markets = data["markets"]
    print(f"\n  Input file     : {os.path.basename(json_path)}")
    print(f"  Schema version : {data.get('schema_version', 'unknown')}")
    print(f"  Markets loaded : {len(markets)}")
    print(
        "\n  Anonymization  : stripping market names, profiles, rationale, "
        "and grounding evidence.\n"
        "  Scorer input   : verified categorical feature matrix only.\n"
    )

    client = get_client()

    for i, market in enumerate(markets):
        market["step3"] = score_market(client, market, i)
        time.sleep(0.5)

    # Save
    output_path = os.path.join(project_root, "reference_population_scored.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"\n  + Saved → reference_population_scored.json")

    print_final_report(markets)


if __name__ == "__main__":
    main()
