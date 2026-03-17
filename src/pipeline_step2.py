"""
Vela Market Quality Rating System — Step 2
Grounded Verification & Agreement Scoring

Reads reference_population.json, queries Gemini (with Google Search grounding)
for T+5 outcomes and dimension evidence, computes inter-source agreement,
appends results to the JSON, and prints a final verification report.
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

GEMINI_MODEL = "models/gemini-3.1-pro-preview"  # best available; grounding confirmed

# Ordinal scales for agreement scoring (order matters)
ORDINAL_SCALES: dict[str, list[str]] = {
    "timing":             ["pre_chasm", "early_chasm", "early_majority", "late_majority", "peak"],
    "competition":        ["nascent", "fragmented", "consolidating", "consolidated", "commoditized"],
    "market_size":        ["micro", "small", "medium", "large", "mega"],
    "customer_readiness": ["unaware", "aware", "interested", "ready", "adopting"],
    "regulatory":         ["unregulated", "light_touch", "moderate", "heavy", "restricted"],
    "infrastructure":     ["non_existent", "emerging", "developing", "mature", "commoditized"],
    "market_structure":   ["undefined", "emerging", "forming", "defined", "mature"],
}

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
    # Pattern: 'retryDelay': '37s' or retry in 37.9s
    for pattern in [r"retryDelay.*?'(\d+)s", r"retry in (\d+(?:\.\d+)?)s"]:
        m = re.search(pattern, text)
        if m:
            return float(m.group(1)) + 2  # small buffer
    return 65.0  # conservative default


def query_gemini_grounded(
    client: genai.Client, prompt: str, retries: int = 4
) -> tuple[str, list[str]]:
    """
    Query Gemini with Google Search grounding.
    Returns (response_text, list_of_source_urls).
    Handles 429 rate-limit errors by waiting the retry delay specified in the error.
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

            # Extract text
            text = ""
            if hasattr(response, "text") and response.text:
                text = response.text
            elif response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text

            # Extract grounding URLs
            urls: list[str] = []
            if response.candidates:
                candidate = response.candidates[0]
                gm = getattr(candidate, "grounding_metadata", None)
                if gm:
                    # Newer SDK: grounding_chunks
                    chunks = getattr(gm, "grounding_chunks", None) or []
                    for chunk in chunks:
                        web = getattr(chunk, "web", None)
                        if web:
                            uri = getattr(web, "uri", None)
                            if uri and uri not in urls:
                                urls.append(uri)
                    # Older SDK fallback: grounding_attributions
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
                # Transient error — retry with backoff
                wait = min(4 ** attempt, 30)
                print(f"    [retry {attempt + 1}/{retries} in {wait}s — {exc_str[:80]}]", end=" ")
                time.sleep(wait)
            elif is_quota and attempt == 0:
                # Grounding quota exhausted — immediately fall back to ungrounded query
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
                    # No real URLs available — return empty list so outcome gets REJECTED
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
    HIGH  = exact match (distance 0)
    MEDIUM = adjacent on scale (distance 1)
    LOW   = 2+ steps apart, or unknown classification
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
# Task A — T+5 outcome verification
# ---------------------------------------------------------------------------

def verify_outcome(client: genai.Client, market: dict) -> dict:
    """
    Query Gemini for the T+5 year outcome with grounded web sources.
    Requires >= 3 citable URLs; otherwise flags market as REJECTED.
    """
    market_name = market["base_profile"].get("market_name", market["domain"])
    ref_year = market["ref_year"]
    t5_year = ref_year + 5

    prompt = (
        f"Research the historical venture capital market: **{market_name}** (reference year: {ref_year}).\n\n"
        f"Using web sources, describe what happened to this market by {t5_year} (5 years later):\n"
        "1. Did the market grow significantly, plateau, or decline by this year?\n"
        "2. What were key revenue, funding, or user-adoption milestones reached by this year?\n"
        "3. Which companies emerged as market leaders?\n"
        "4. What was the overall outcome for venture investors who entered in "
        f"{ref_year}?\n\n"
        "Cite specific facts with numbers and dates. Draw on industry reports, "
        "news articles, company filings, and analyst data."
    )

    text, urls = query_gemini_grounded(client, prompt)
    status = "VERIFIED" if len(urls) >= 3 else "REJECTED"

    return {
        "t5_year": t5_year,
        "outcome_summary": text[:2500] if text else "",
        "source_urls": urls,
        "source_count": len(urls),
        "verification_status": status,
        "rejection_reason": (
            None
            if status == "VERIFIED"
            else f"Only {len(urls)} grounded source(s) returned — minimum 3 required."
        ),
    }


# ---------------------------------------------------------------------------
# Task B — Dimension agreement verification
# ---------------------------------------------------------------------------

def _parse_gemini_classification(
    text: str, options: list[str], fallback_text: str
) -> tuple[str, str, str]:
    """
    Try to parse a JSON classification from Gemini's response.
    Returns (classification, evidence, key_fact).
    """
    clean = text.strip()
    if "```" in clean:
        clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()

    classification = "unknown"
    evidence = ""
    key_fact = ""

    try:
        parsed = json.loads(clean)
        classification = str(parsed.get("classification", "unknown")).lower().strip()
        evidence = parsed.get("evidence", "")
        key_fact = parsed.get("key_fact", "")
    except (json.JSONDecodeError, ValueError):
        # Fall back: scan text for known option strings
        lower_text = fallback_text.lower()
        for opt in options:
            if opt.replace("_", " ") in lower_text or opt in lower_text:
                classification = opt
                break
        evidence = fallback_text[:400]

    # Normalize: if returned value not in options, fuzzy-match
    if classification not in options:
        for opt in options:
            if opt in classification or classification in opt:
                classification = opt
                break
        else:
            classification = "unknown"

    return classification, evidence, key_fact


def verify_dimension(
    client: genai.Client,
    market: dict,
    dim_name: str,
    claude_classification: str,
    options: list[str],
) -> dict:
    """
    Query Gemini for grounded evidence on one dimension, classify it,
    and compute agreement against Claude's classification.
    """
    market_name = market["base_profile"].get("market_name", market["domain"])
    ref_year = market["ref_year"]

    prompt = (
        f"Research this historical venture market using web sources:\n"
        f"Market: {market_name}\n"
        f"Reference year: {ref_year}\n\n"
        f"Classify the market along this analytical dimension: **{dim_name}**\n"
        f"Available options: {options}\n\n"
        "Based on historical evidence available for this market at the reference year, "
        "which option best describes it?\n\n"
        "Respond with ONLY a JSON object — no markdown, no extra text:\n"
        "{\n"
        '  "classification": "<one value from the options list>",\n'
        '  "evidence": "<2-3 sentences of factual evidence supporting your classification>",\n'
        '  "key_fact": "<one specific statistic, date, or data point>"\n'
        "}"
    )

    text, urls = query_gemini_grounded(client, prompt)
    gemini_class, evidence, key_fact = _parse_gemini_classification(text, options, text)
    agreement = score_agreement(dim_name, claude_classification, gemini_class)

    return {
        "dimension": dim_name,
        "claude_classification": claude_classification,
        "gemini_classification": gemini_class,
        "agreement": agreement,
        "evidence": evidence,
        "key_fact": key_fact,
        "grounding_urls": urls[:3],
    }


# ---------------------------------------------------------------------------
# Market processing
# ---------------------------------------------------------------------------

def process_market_verification(client: genai.Client, market: dict) -> dict:
    """Run all Step 2 verification tasks for one market."""
    market_name = market["base_profile"].get("market_name", market["domain"])
    print(f"\n[Verifying]  {market_name}  ({market['ref_year']})")

    # --- Task A: T+5 outcome ---
    print("  -> T+5 outcome verification ...", end=" ", flush=True)
    outcome = verify_outcome(client, market)
    icon = "+" if outcome["verification_status"] == "VERIFIED" else "X"
    print(f"{icon} {outcome['verification_status']}  ({outcome['source_count']} sources)")
    time.sleep(3)  # inter-request buffer

    # --- Task B: Dimension agreement ---
    dimension_verifications: dict = {}
    for dim_name, dim_data in market["dimensions"].items():
        claude_class = dim_data.get("classification", "unknown")
        if claude_class == "unknown":
            continue
        options = ORDINAL_SCALES.get(dim_name, [])
        print(f"  -> Agreement [{dim_name}] ...", end=" ", flush=True)
        time.sleep(3)  # inter-request buffer
        dv = verify_dimension(client, market, dim_name, claude_class, options)
        dimension_verifications[dim_name] = dv
        marker = {"HIGH": "[*]", "MEDIUM": "[~]", "LOW": "[ ]"}.get(dv["agreement"], "?")
        print(
            f"{marker} {dv['agreement']:<6}  "
            f"Claude:{dv['claude_classification']:<20}  "
            f"Gemini:{dv['gemini_classification']}"
        )

    # --- Aggregate agreement score ---
    agreements = [v["agreement"] for v in dimension_verifications.values()]
    high = agreements.count("HIGH")
    medium = agreements.count("MEDIUM")
    low = agreements.count("LOW")
    total = len(agreements)
    score = (high * 1.0 + medium * 0.5 + low * 0.0) / total if total else 0.0

    if score >= 0.70:
        overall = "HIGH"
    elif score >= 0.40:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    return {
        "outcome_verification": outcome,
        "dimension_verifications": dimension_verifications,
        "agreement_summary": {
            "overall": overall,
            "score": round(score, 3),
            "HIGH": high,
            "MEDIUM": medium,
            "LOW": low,
            "total_dimensions": total,
        },
    }


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(markets: list[dict]) -> None:
    verified = [
        m for m in markets
        if m.get("step2", {}).get("outcome_verification", {}).get("verification_status") == "VERIFIED"
    ]
    rejected = [m for m in markets if m not in verified]

    divider = "=" * 72
    thin = "-" * 72

    print(f"\n\n{divider}")
    print("  VELA MARKET QUALITY RATING — FINAL VERIFICATION REPORT")
    print(divider)
    print(
        f"\n  Batch size : {len(markets)} markets"
        f"  |  Passed : {len(verified)}"
        f"  |  Rejected : {len(rejected)}"
        f"  |  Source threshold : ≥ 3 citable URLs"
    )

    for market in markets:
        s2 = market.get("step2", {})
        outcome = s2.get("outcome_verification", {})
        agreement = s2.get("agreement_summary", {})
        dim_v = s2.get("dimension_verifications", {})
        profile = market.get("base_profile", {})

        name = profile.get("market_name", market.get("domain", "Unknown"))
        status = outcome.get("verification_status", "UNKNOWN")
        ref = market.get("ref_year", "?")
        t5 = outcome.get("t5_year", "?")
        n_sources = outcome.get("source_count", 0)
        urls = outcome.get("source_urls", [])

        status_badge = "+ PASS" if status == "VERIFIED" else "X FAIL"

        print(f"\n{thin}")
        print(f"  {status_badge}   {name}")
        print(f"           Reference year: {ref}   |   T+5 year: {t5}")
        print(f"\n  Outcome Verification")
        print(f"    Sources found : {n_sources}")
        for url in urls[:5]:
            print(f"    * {url[:80]}")
        if outcome.get("rejection_reason"):
            print(f"    REJECTION: {outcome['rejection_reason']}")

        print(f"\n  Inter-Source Agreement")
        oa = agreement.get("overall", "N/A")
        sc = agreement.get("score", 0)
        h, med, lo = agreement.get("HIGH", 0), agreement.get("MEDIUM", 0), agreement.get("LOW", 0)
        print(f"    Overall: {oa}  (score={sc:.2f})   HIGH={h}  MEDIUM={med}  LOW={lo}")

        print(f"\n  Dimension Matrix")
        print(f"    {'Dimension':<24}  {'Claude':<22}  {'Gemini':<22}  {'Agreement'}")
        print(f"    {'-'*24}  {'-'*22}  {'-'*22}  {'-'*9}")
        for dim_name, dv in dim_v.items():
            marker = {"HIGH": "[*]", "MEDIUM": "[~]", "LOW": "[ ]"}.get(dv.get("agreement", ""), "?")
            cc = dv.get("claude_classification", "?")
            gc = dv.get("gemini_classification", "?")
            ag = dv.get("agreement", "?")
            print(f"  {marker} {dim_name:<24}  {cc:<22}  {gc:<22}  {ag}")

    print(f"\n{divider}")
    print(
        f"  PIPELINE COMPLETE:  {len(verified)} / {len(markets)} markets "
        f"passed the 3-source verification threshold."
    )
    print(f"{divider}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  VELA MARKET QUALITY RATING — STEP 2")
    print("  Grounded Verification & Agreement Scoring  (Gemini)")
    print("=" * 65)

    # Load Step 1 output
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

    # Persist enriched data
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"\n  + Updated reference_population.json with verification results")

    print_final_report(markets)


if __name__ == "__main__":
    main()
