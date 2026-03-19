"""
Vela Market Quality Rating System — Step 2  (consolidated single-call edition)
Grounded Verification & Agreement Scoring

ONE Gemini API call per market verifies all 7 categorical dimensions and the
T+5 outcome simultaneously, returning a single comprehensive JSON object.
This replaces the previous 8-call-per-market approach and eliminates inter-call
sleep delays while preserving full backward-compatible output schema.
"""

import json
import os
import re
import sys
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
except ImportError:
    sys.exit(
        "ERROR: google-genai package not found.\n"
        "Install it with:  pip install google-genai"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = "models/gemini-2.5-flash"

# Ordinal scales for agreement scoring (order matters for distance computation).
# market_structure is categorical — listed so unknown values still score LOW.
ORDINAL_SCALES: dict[str, list[str]] = {
    "timing": [
        "innovators", "early_adopters", "early_majority", "late_majority", "laggards",
    ],
    "competition": [
        "monopoly", "oligopoly", "monopolistic_competition", "perfect_competition",
    ],
    "market_size": [
        "micro", "small", "medium", "large", "massive",
    ],
    "customer_readiness": [
        "innovation_trigger", "peak_of_inflated_expectations", "trough_of_disillusionment",
        "slope_of_enlightenment", "plateau_of_productivity",
    ],
    "regulatory": [
        "unregulated", "light_touch", "moderate_compliance", "heavily_regulated", "prohibitive",
    ],
    "infrastructure": [
        "nascent", "emerging", "developing", "mature",
    ],
    "market_structure": [
        "winner_take_most", "platform_two_sided", "technology_enablement",
        "fragmented_niche", "regulated_infrastructure",
    ],
}

# Canonical dimension order
_DIM_ORDER = [
    "timing", "competition", "market_size", "customer_readiness",
    "regulatory", "infrastructure", "market_structure",
]

# ---------------------------------------------------------------------------
# Gemini client helpers
# ---------------------------------------------------------------------------

def get_gemini_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key.startswith("your_"):
        sys.exit(
            "ERROR: GEMINI_API_KEY is not set or is a placeholder. "
            "Please update your .env file."
        )
    return genai.Client(api_key=key)


def _parse_retry_delay(exc: Exception) -> float:
    """Extract retryDelay seconds from a Gemini 429 error string, default 60s."""
    text = str(exc)
    for pattern in [r"retryDelay.*?'(\d+)s", r"retry in (\d+(?:\.\d+)?)s"]:
        m = re.search(pattern, text)
        if m:
            return float(m.group(1)) + 2
    return 65.0


def query_gemini_grounded(
    client: genai.Client, prompt: str, retries: int = 4
) -> tuple[str, list[str]]:
    """
    Query Gemini with Google Search grounding.
    Returns (response_text, list_of_source_urls).
    Handles 429 rate-limit errors with the retry delay specified in the error.
    """
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[Tool(google_search=GoogleSearch())],
                    temperature=0.1,
                ),
            )

            text = ""
            if hasattr(response, "text") and response.text:
                text = response.text
            elif response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text

            urls: list[str] = []
            if response.candidates:
                candidate = response.candidates[0]
                gm = getattr(candidate, "grounding_metadata", None)
                if gm:
                    chunks = getattr(gm, "grounding_chunks", None) or []
                    for chunk in chunks:
                        web = getattr(chunk, "web", None)
                        if web:
                            uri = getattr(web, "uri", None)
                            if uri and uri not in urls:
                                urls.append(uri)
                    if not urls:
                        attrs = getattr(gm, "grounding_attributions", None) or []
                        for attr in attrs:
                            web = getattr(attr, "web", None)
                            if web:
                                uri = getattr(web, "uri", None)
                                if uri and uri not in urls:
                                    urls.append(uri)

            return text, urls

        except Exception as exc:
            exc_str = str(exc)
            is_quota = "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str

            if attempt < retries and not is_quota:
                wait = min(4 ** attempt, 30)
                print(f"    [retry {attempt + 1}/{retries} in {wait}s — {exc_str[:80]}]", end=" ")
                time.sleep(wait)
            elif is_quota and attempt == 0:
                print(f"\n    [grounding quota exhausted; falling back to ungrounded query]",
                      flush=True)
                try:
                    fallback_resp = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=GenerateContentConfig(temperature=0.1),
                    )
                    fb_text = ""
                    if hasattr(fallback_resp, "text") and fallback_resp.text:
                        fb_text = fallback_resp.text
                    elif fallback_resp.candidates:
                        for part in fallback_resp.candidates[0].content.parts:
                            if hasattr(part, "text") and part.text:
                                fb_text += part.text
                    return fb_text, []
                except Exception as fb_exc:
                    fb_str = str(fb_exc)
                    if "429" in fb_str or "RESOURCE_EXHAUSTED" in fb_str:
                        wait = _parse_retry_delay(fb_exc)
                        print(f"    [ungrounded also rate-limited; waiting {wait:.0f}s]", flush=True)
                        time.sleep(wait)
                    return f"ERROR: {fb_exc}", []
            else:
                print(f"\n    [FAILED after {retries} retries: {exc_str[:120]}]")
                return f"ERROR: {exc}", []

    return "", []


# ---------------------------------------------------------------------------
# Agreement scoring
# ---------------------------------------------------------------------------

def score_agreement(dim_name: str, claude_class: str, gemini_class: str) -> str:
    """
    Compute ordinal-distance-based agreement level.
    HIGH   = exact match (distance 0)
    MEDIUM = adjacent on scale (distance 1)
    LOW    = 2+ steps apart, or unknown classification
    """
    scale = ORDINAL_SCALES.get(dim_name, [])
    c = claude_class.lower().strip()
    g = gemini_class.lower().strip()

    if not scale or c not in scale or g not in scale:
        return "HIGH" if c == g else "LOW"

    distance = abs(scale.index(c) - scale.index(g))
    if distance == 0:
        return "HIGH"
    elif distance == 1:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Single-call market verification
# ---------------------------------------------------------------------------

def _build_consolidated_prompt(market_name: str, ref_year: int, t5_year: int) -> str:
    """Build the single comprehensive prompt for all verification tasks."""
    dim_options = "\n".join(
        f"  {dim}: {ORDINAL_SCALES[dim]}"
        for dim in _DIM_ORDER
    )
    # Build the JSON template explicitly to avoid f-string brace escaping issues
    dim_template_lines = []
    for dim in _DIM_ORDER:
        opts = ORDINAL_SCALES[dim]
        dim_template_lines.append(
            f'    "{dim}": {{"classification": "<one of {opts}>", '
            f'"evidence": "<2 sentences of factual evidence>", '
            f'"key_fact": "<1 specific statistic or date>"}}'
        )
    dim_template = ",\n".join(dim_template_lines)

    return (
        f"Research the historical venture capital market: {market_name} "
        f"(reference year: {ref_year}).\n\n"
        "Use Google Search to complete two research tasks in a single response:\n\n"
        f"TASK 1 — T+5 OUTCOME: Describe what happened to this market by {t5_year}. "
        "Did it grow, plateau, or decline? Include key funding rounds, revenue milestones, "
        "and overall outcome for venture investors who entered in "
        f"{ref_year}.\n\n"
        f"TASK 2 — DIMENSION CLASSIFICATION: Using evidence as of {ref_year}, "
        "classify this market on all 7 dimensions below. "
        "Choose EXACTLY ONE option per dimension from the lists provided:\n"
        f"{dim_options}\n\n"
        "Return ONLY a raw, valid JSON object. Do not include markdown formatting like "
        "```json, and do not include any conversational preamble or postamble.\n\n"
        "{\n"
        f'  "outcome_summary": "<2-3 sentences on what happened by {t5_year}>",\n'
        '  "dimensions": {\n'
        + dim_template + "\n"
        "  }\n"
        "}"
    )


def _parse_consolidated_response(
    text: str,
    urls: list[str],
    market: dict,
    t5_year: int,
) -> dict:
    """
    Parse the single Gemini response into the standard step2 output schema.
    Falls back gracefully if JSON is malformed.
    """
    clean = text.strip()
    if "```" in clean:
        clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
    brace = clean.find("{")
    if brace > 0:
        clean = clean[brace:]

    parsed_json = None
    try:
        parsed_json = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Outcome verification ──────────────────────────────────────────
    outcome_text = ""
    if parsed_json:
        outcome_text = parsed_json.get("outcome_summary", "")
    if not outcome_text:
        outcome_text = text[:2500]

    outcome = {
        "t5_year":             t5_year,
        "outcome_summary":     outcome_text,
        "source_urls":         urls,
        "source_count":        len(urls),
        "verification_status": "VERIFIED" if urls else "UNVERIFIED",
        "rejection_reason":    None if urls else "No grounded sources returned.",
    }

    # ── Dimension verifications ───────────────────────────────────────
    dims_data = {}
    if parsed_json:
        dims_data = parsed_json.get("dimensions", {})

    dimension_verifications: dict = {}
    for dim_name in _DIM_ORDER:
        claude_class = (
            market.get("dimensions", {})
                  .get(dim_name, {})
                  .get("classification", "unknown")
        )
        options = ORDINAL_SCALES.get(dim_name, [])
        dim_blob = dims_data.get(dim_name, {})

        # Parse and normalise Gemini's classification
        raw_gc = str(dim_blob.get("classification", "unknown")).lower().strip()
        gemini_class = "unknown"
        if raw_gc in options:
            gemini_class = raw_gc
        else:
            # Fuzzy match (e.g. "early adopters" → "early_adopters")
            for opt in options:
                if opt in raw_gc or raw_gc in opt or opt.replace("_", " ") == raw_gc:
                    gemini_class = opt
                    break

        if gemini_class == "unknown" and not parsed_json:
            # Full parse failure: scan raw text for option strings
            lower_text = text.lower()
            for opt in options:
                if opt.replace("_", " ") in lower_text or opt in lower_text:
                    gemini_class = opt
                    break

        agreement = score_agreement(dim_name, claude_class, gemini_class)

        dimension_verifications[dim_name] = {
            "dimension":             dim_name,
            "claude_classification": claude_class,
            "gemini_classification": gemini_class,
            "agreement":             agreement,
            "evidence":              dim_blob.get("evidence", ""),
            "key_fact":              dim_blob.get("key_fact", ""),
            "verification_url":      urls[0] if urls else "",
            "grounding_urls":        urls[:3],
        }

    # ── Agreement summary ─────────────────────────────────────────────
    agreements = [v["agreement"] for v in dimension_verifications.values()]
    high   = agreements.count("HIGH")
    medium = agreements.count("MEDIUM")
    low    = agreements.count("LOW")
    total  = len(agreements)
    score  = (high * 1.0 + medium * 0.5) / total if total else 0.0

    overall = "HIGH" if score >= 0.70 else ("MEDIUM" if score >= 0.40 else "LOW")

    return {
        "outcome_verification":    outcome,
        "dimension_verifications": dimension_verifications,
        "agreement_summary": {
            "overall":          overall,
            "score":            round(score, 3),
            "HIGH":             high,
            "MEDIUM":           medium,
            "LOW":              low,
            "total_dimensions": total,
        },
    }


# ---------------------------------------------------------------------------
# Market processing — public API (single call per market)
# ---------------------------------------------------------------------------

def process_market_verification(client: genai.Client, market: dict) -> dict:
    """
    Verify one market via a SINGLE Gemini API call.

    Asks Gemini to simultaneously:
      - Research the T+5 outcome with grounded web sources
      - Classify all 7 dimensions based on evidence at the reference year

    Returns the standard step2 result dict compatible with all downstream code.
    """
    market_name = market["base_profile"].get("market_name", market["domain"])
    ref_year    = market["ref_year"]
    t5_year     = ref_year + 5

    print(f"\n[Verifying]  {market_name}  ({ref_year})")
    print("  -> Single Gemini call (T+5 + 7 dims) ...", end=" ", flush=True)

    prompt = _build_consolidated_prompt(market_name, ref_year, t5_year)
    text, urls = query_gemini_grounded(client, prompt)

    result = _parse_consolidated_response(text, urls, market, t5_year)

    # Print summary line
    outcome = result["outcome_verification"]
    agr     = result["agreement_summary"]
    icon    = "+" if outcome["verification_status"] == "VERIFIED" else "~"
    print(
        f"{icon} {outcome['verification_status']} ({outcome['source_count']} sources)  "
        f"Agreement: {agr['overall']} "
        f"(H={agr['HIGH']} M={agr['MEDIUM']} L={agr['LOW']})"
    )

    # Print per-dimension agreement for observability
    for dim_name, dv in result["dimension_verifications"].items():
        marker = {"HIGH": "[*]", "MEDIUM": "[~]", "LOW": "[ ]"}.get(dv["agreement"], "?")
        print(
            f"  -> Agreement [{dim_name}] ... "
            f"{marker} {dv['agreement']:<6}  "
            f"Claude:{dv['claude_classification']:<20}  "
            f"Gemini:{dv['gemini_classification']}"
        )

    return result


# ---------------------------------------------------------------------------
# Final report (used by standalone main)
# ---------------------------------------------------------------------------

def print_final_report(markets: list[dict]) -> None:
    verified = [
        m for m in markets
        if m.get("step2", {})
             .get("outcome_verification", {})
             .get("verification_status") == "VERIFIED"
    ]

    divider = "=" * 72
    thin    = "-" * 72

    print(f"\n\n{divider}")
    print("  VELA MARKET QUALITY RATING — FINAL VERIFICATION REPORT")
    print(divider)
    print(
        f"\n  Batch size : {len(markets)} markets"
        f"  |  Grounded: {len(verified)}"
        f"  |  Ungrounded : {len(markets) - len(verified)}"
    )

    for market in markets:
        s2      = market.get("step2", {})
        outcome = s2.get("outcome_verification", {})
        agr     = s2.get("agreement_summary", {})
        dim_v   = s2.get("dimension_verifications", {})
        profile = market.get("base_profile", {})

        name     = profile.get("market_name", market.get("domain", "Unknown"))
        status   = outcome.get("verification_status", "UNKNOWN")
        ref      = market.get("ref_year", "?")
        t5       = outcome.get("t5_year", "?")
        n_src    = outcome.get("source_count", 0)
        urls     = outcome.get("source_urls", [])
        badge    = "+ PASS" if status == "VERIFIED" else "~ UNGROUNDED"

        print(f"\n{thin}")
        print(f"  {badge}   {name}")
        print(f"           Reference year: {ref}   |   T+5 year: {t5}")
        print(f"\n  Outcome Verification")
        print(f"    Sources found : {n_src}")
        for url in urls[:5]:
            print(f"    * {url[:80]}")

        print(f"\n  Inter-Source Agreement")
        oa  = agr.get("overall", "N/A")
        sc  = agr.get("score", 0)
        h   = agr.get("HIGH", 0)
        med = agr.get("MEDIUM", 0)
        lo  = agr.get("LOW", 0)
        print(f"    Overall: {oa}  (score={sc:.2f})   HIGH={h}  MEDIUM={med}  LOW={lo}")

        print(f"\n  Dimension Matrix")
        print(f"    {'Dimension':<24}  {'Claude':<28}  {'Gemini':<28}  Agree")
        print(f"    {'-'*24}  {'-'*28}  {'-'*28}  -----")
        for dim_name, dv in dim_v.items():
            marker = {"HIGH": "[*]", "MEDIUM": "[~]", "LOW": "[ ]"}.get(
                dv.get("agreement", ""), "?"
            )
            cc = dv.get("claude_classification", "?")
            gc = dv.get("gemini_classification", "?")
            ag = dv.get("agreement", "?")
            print(f"  {marker} {dim_name:<24}  {cc:<28}  {gc:<28}  {ag}")

    print(f"\n{divider}")
    print(f"  PIPELINE COMPLETE:  {len(markets)} markets processed.")
    print(f"{divider}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  VELA MARKET QUALITY RATING — STEP 2")
    print("  Grounded Verification & Agreement Scoring  (Gemini, single-call)")
    print("=" * 65)

    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "reference_population.json"
    )
    if not os.path.exists(json_path):
        sys.exit(
            "ERROR: reference_population.json not found. "
            "Run pipeline_step1.py first."
        )

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    markets: list[dict] = data["markets"]
    print(f"\n  Loaded {len(markets)} markets from reference_population.json")

    client = get_gemini_client()

    for market in markets:
        step2_result = process_market_verification(client, market)
        market["step2"] = step2_result

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"\n  + Updated reference_population.json with verification results")

    print_final_report(markets)


if __name__ == "__main__":
    main()
