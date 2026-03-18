"""
Vela Market Quality Rating System — Step 3
Scorer & Rater

Reads the Step 2 enriched JSON, anonymizes each market to its verified
categorical feature matrix, calls Claude to score all 7 dimensions on a
0–100 integer scale, and saves the fully enriched dataset as
reference_population_scored.json.

Scoring contract
----------------
The LLM's ONLY job is to map each categorical dimension value to a
0–100 integer score. It must NOT output L-tier labels, composite scores,
investment decisions, or any field beyond the 7 dimension integers.
Tier classification is computed in a later step via percentile lookup
across the full 120-market database.

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
    "90-100: Ideal conditions for outsized venture returns.\n"
    "75-89:  Highly attractive, minor friction.\n"
    "60-74:  Viable but requires execution excellence.\n"
    "40-59:  Sub-optimal conditions, significant structural headwinds.\n"
    "0-39:   Hostile conditions for new venture entrants."
)

SYSTEM_PROMPT = (
    "You are an objective quantitative venture analyst scoring anonymous market data. "
    "You are evaluating a market's readiness for venture-scale entry based solely on "
    "structured categorical variables. "
    "Score each of the 7 dimensions from 0 to 100 based purely on the structured "
    "features provided. Use this numeric scale as your calibration guide:\n\n"
    + SCORE_RUBRIC
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

    if validation_errors:
        print(f"WARNINGS: {'; '.join(validation_errors)}")

    valid_scores = [v for v in scores.values() if v is not None]
    score_summary = (
        f"scores={list(scores.values())}"
        if valid_scores
        else "no valid scores"
    )
    print(score_summary)

    return {
        "scores": scores,
        "feature_matrix": matrix,
        "validation_errors": validation_errors,
    }


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(markets: list) -> None:
    divider = "=" * 76
    thin    = "-" * 76

    print(f"\n\n{divider}")
    print("  VELA MARKET QUALITY RATING — STEP 3  DIMENSION SCORES")
    print("  (No composite or tier — tier computed via percentile in a later step)")
    print(divider)

    for market in markets:
        s3      = market.get("step3", {})
        scores  = s3.get("scores", {})
        fm      = s3.get("feature_matrix", {})
        profile = market.get("base_profile", {})
        name    = profile.get("market_name", market.get("domain", "Unknown"))
        ref     = market.get("ref_year", "?")

        print(f"\n{thin}")
        print(f"  Market : {name}  ({ref})")
        print(f"\n  {'Dimension':<24}  {'Score':>5}  {'Verified Value':<30}  Agreement")
        print(f"  {'-'*24}  {'-'*5}  {'-'*30}  {'-'*10}")

        for dim in DIMENSIONS:
            sc     = scores.get(dim)
            val    = fm.get(dim, {}).get("value", "?")
            agr    = fm.get(dim, {}).get("agreement", "?")
            sc_str = str(sc) if sc is not None else "ERR"
            print(f"  {dim:<24}  {sc_str:>5}  {val:<30}  {agr}")

        if s3.get("validation_errors"):
            print(f"\n  Validation notes: {s3['validation_errors']}")

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
