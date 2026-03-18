"""
Vela MQR — Step 2b (Search-Grounded Judge)

Identical purpose to the Parametric Judge, but the GPT-4o call is equipped
with a live web-search tool (OpenAI Responses API, web_search_preview).

This allows the judge to retrieve current web evidence — news, analyst
reports, Wikipedia, company filings — to break the Claude/Gemini tie
with grounded, citable sources rather than parametric recall alone.

API note
--------
Uses the OpenAI Responses API (`client.responses.create`) which requires
openai >= 1.66.0.  If the Responses API is unavailable (AttributeError),
the script falls back to Chat Completions without search and logs a warning.
"""

import json
import os
import re
import sys
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    sys.exit(
        "ERROR: openai package not found.\n"
        "Install with:  pip install openai"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEARCH_MODEL = "gpt-4o"

DIMENSIONS = [
    "timing",
    "competition",
    "market_size",
    "customer_readiness",
    "regulatory",
    "infrastructure",
    "market_structure",
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("your_"):
        sys.exit(
            "ERROR: OPENAI_API_KEY is not set or is a placeholder. "
            "Please update your .env file."
        )
    return OpenAI(api_key=key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_disagreement(agreement: str) -> bool:
    return agreement not in ("HIGH", "agreed", "unverified", "")


def _call_with_retry(fn, max_retries: int = 3, base_delay: float = 10.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = "429" in exc_str or "rate_limit" in exc_str
            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (attempt + 1)
                print(
                    f"\n    [OpenAI 429; retry {attempt + 1}/{max_retries} in {delay:.0f}s]",
                    end=" ", flush=True,
                )
                time.sleep(delay)
            else:
                raise


def _extract_url_from_response(response) -> str:
    """
    Try to extract the first cited URL from a Responses API output object.
    Looks for url_citation annotations on message content blocks.
    """
    try:
        for item in getattr(response, "output", []):
            if getattr(item, "type", "") == "message":
                for block in getattr(item, "content", []):
                    for ann in getattr(block, "annotations", []):
                        if getattr(ann, "type", "") == "url_citation":
                            url = getattr(ann, "url", "")
                            if url:
                                return url
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Single-dimension resolver
# ---------------------------------------------------------------------------

def _resolve_dimension(
    client: OpenAI,
    dim: str,
    claude_value: str,
    gemini_value: str,
    ref_year: int,
    market_domain: str,
) -> dict:
    """
    Call GPT-4o with live web search to choose between claude_value and
    gemini_value for the given dimension.  Returns a resolution dict that
    includes the best search source URL found.
    """
    prompt = (
        f"Research the following historical venture market to resolve a classification dispute.\n\n"
        f"Market : {market_domain}\n"
        f"Reference year : {ref_year}\n"
        f"Dimension : {dim}\n\n"
        f"Two AI classifiers disagreed:\n"
        f"  Classifier A: \"{claude_value}\"\n"
        f"  Classifier B: \"{gemini_value}\"\n\n"
        f"Search the web for historical evidence about this market and this specific "
        f"dimension at {ref_year}. Determine which classification is more historically accurate.\n"
        f"You MUST choose exactly one of the two provided values — do not invent a new value.\n\n"
        "Respond with ONLY a valid JSON object (no markdown, no preamble):\n"
        "{\n"
        f'  "resolved_classification": "<exactly: {claude_value} OR {gemini_value}>",\n'
        '  "chosen": "A" or "B",\n'
        '  "rationale": "<1-2 sentences citing specific evidence from your search>",\n'
        '  "search_source": "<the single most relevant URL you found>"\n'
        "}"
    )

    search_source = ""
    raw = ""

    # --- Primary: Responses API with web_search_preview ---
    try:
        response = _call_with_retry(
            lambda: client.responses.create(
                model=SEARCH_MODEL,
                tools=[{"type": "web_search_preview"}],
                input=prompt,
            )
        )
        raw = getattr(response, "output_text", "") or ""
        search_source = _extract_url_from_response(response)

    except AttributeError:
        # Responses API not available in this openai version — fall back
        print("\n    [Responses API unavailable; falling back to chat completions]", end=" ", flush=True)
        fb = _call_with_retry(
            lambda: client.chat.completions.create(
                model=SEARCH_MODEL,
                max_tokens=400,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        raw = fb.choices[0].message.content.strip()

    # Strip markdown fences
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # In case model added prose before the JSON, find first {
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]

    try:
        parsed = json.loads(raw)
        resolved = parsed.get("resolved_classification", "").strip()
        if resolved not in (claude_value, gemini_value):
            resolved = claude_value
        # Use API-extracted URL first; fall back to what GPT reported in JSON
        if not search_source:
            search_source = parsed.get("search_source", "")
        return {
            "value":             resolved,
            "source":            "openai_search",
            "chosen_classifier": parsed.get("chosen", "?"),
            "rationale":         parsed.get("rationale", ""),
            "search_source":     search_source,
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "value":             claude_value,
            "source":            "fallback_claude",
            "chosen_classifier": "A",
            "rationale":         f"JSON parse error — fell back to Classifier A. Raw: {raw[:120]}",
            "search_source":     search_source,
        }


# ---------------------------------------------------------------------------
# Market-level resolver (main export)
# ---------------------------------------------------------------------------

def resolve_market_search(market: dict, client: OpenAI) -> dict:
    """
    Build a fully resolved feature matrix for one market using web search.

    - Agreed dimensions (HIGH): pass through Gemini-verified value.
    - Disagreed dimensions (MEDIUM / LOW): call GPT-4o + web search to resolve.

    Returns:
      {dim: {"value": str, "agreement": str, "source": str,
             "rationale": str, "search_source": str}}
    """
    step1_dims   = market.get("dimensions", {})
    step2_verifs = market.get("step2", {}).get("dimension_verifications", {})
    ref_year     = market.get("ref_year", 0)
    domain       = market.get("domain", "unknown market")

    resolved = {}
    for dim in DIMENSIONS:
        s1 = step1_dims.get(dim, {})
        s2 = step2_verifs.get(dim, {})

        claude_value = s1.get("classification", "unknown")
        gemini_value = s2.get("gemini_classification", "") or "unknown"
        agreement    = s2.get("agreement", "unverified")

        if not _is_disagreement(agreement) or gemini_value == "unknown":
            value = gemini_value if gemini_value not in ("unknown", "") else claude_value
            resolved[dim] = {
                "value":         value,
                "agreement":     agreement,
                "source":        "agreed",
                "rationale":     "Claude and Gemini agreed — no tie-break required.",
                "search_source": "",
            }
        else:
            print(
                f"    [{dim}] {agreement} — "
                f"claude={claude_value!r} vs gemini={gemini_value!r} → GPT-4o+search ...",
                end=" ", flush=True,
            )
            result = _resolve_dimension(
                client, dim, claude_value, gemini_value, ref_year, domain
            )
            result["agreement"] = agreement
            resolved[dim] = result
            print(f"resolved={result['value']!r}  (chose {result['chosen_classifier']})")
            time.sleep(1.5)  # slightly longer buffer for search calls

    return resolved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  VELA MQR — STEP 2b  SEARCH-GROUNDED JUDGE  (GPT-4o + Web)")
    print("=" * 65)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, "reference_population_scored.json"),
        os.path.join(project_root, "reference_population_v3.json"),
        os.path.join(project_root, "reference_population.json"),
    ]
    json_path = next((p for p in candidates if os.path.exists(p)), None)
    if not json_path:
        sys.exit("ERROR: No reference_population JSON found.")

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    client  = get_openai_client()
    markets = data["markets"]
    print(f"\n  Loaded {len(markets)} markets from {os.path.basename(json_path)}\n")

    for i, market in enumerate(markets):
        name = market.get("base_profile", {}).get("market_name", market.get("domain", "?"))
        print(f"\n[Market {i + 1}]  {name}  ({market.get('ref_year', '?')})")
        market["step2b_search"] = {
            "resolved_feature_matrix": resolve_market_search(market, client)
        }

    output_path = os.path.join(project_root, "reference_population_search.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"\n  + Saved → reference_population_search.json")


if __name__ == "__main__":
    main()
