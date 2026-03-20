"""
Vela MQR — Step 2b (Search-Grounded Judge)  [gpt-5.4 edition]

Identical purpose to the Parametric Judge, but the GPT-5.4 call is equipped
with a live web-search tool (OpenAI Responses API, web_search_preview).

This allows the judge to retrieve current web evidence — news, analyst
reports, Wikipedia, company filings — and make a grounded ruling.

Key change from previous version
----------------------------------
The judge now has two options rather than being forced to choose:

  (A) RESOLVE — choose one value backed by specific retrieved evidence.
  (B) FLAG    — mark as AMBIGUOUS when retrieved sources contradict each
                other, support both interpretations equally, or when no
                authoritative source can be found for the reference year.

FLAGGED dimensions are passed to Role 3 with both options visible so the
scorer can apply a conservative interpretation.

API note
--------
Uses the OpenAI Responses API (`client.responses.create`) with the
`web_search_preview` tool.  Falls back to Chat Completions (no search)
if the Responses API is unavailable and logs a warning.
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

SEARCH_MODEL = "gpt-5.4"

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
            is_server_err = "500" in exc_str or "503" in exc_str or "502" in exc_str
            if (is_rate_limit or is_server_err) and attempt < max_retries:
                delay = base_delay * (attempt + 1)
                print(
                    f"\n    [OpenAI retry {attempt + 1}/{max_retries} in {delay:.0f}s]",
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


def _extract_all_urls_from_response(response) -> list:
    """Extract all cited URLs from a Responses API output object."""
    urls = []
    try:
        for item in getattr(response, "output", []):
            if getattr(item, "type", "") == "message":
                for block in getattr(item, "content", []):
                    for ann in getattr(block, "annotations", []):
                        if getattr(ann, "type", "") == "url_citation":
                            url = getattr(ann, "url", "")
                            if url and url not in urls:
                                urls.append(url)
    except Exception:
        pass
    return urls


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
    Call GPT-5.4 with live web search to judge between claude_value and
    gemini_value for the given dimension.

    The judge may either resolve to one value or flag as AMBIGUOUS.
    Returns a resolution dict with 'flagged: bool' and 'search_sources: list'.
    """
    prompt = (
        f"Research the following historical venture market to adjudicate "
        f"a classification dispute.\n\n"
        f"Market        : {market_domain}\n"
        f"Reference year: {ref_year}\n"
        f"Dimension     : {dim}\n\n"
        f"Two AI classifiers disagreed:\n"
        f"  Classifier A: \"{claude_value}\"\n"
        f"  Classifier B: \"{gemini_value}\"\n\n"
        f"Search the web for historical evidence about this market and this specific "
        f"dimension as it stood in {ref_year}.\n\n"
        "You have two options:\n"
        "  (A) RESOLVE — if your search finds clear evidence favouring one "
        "classification, choose it and cite the specific source.\n"
        "  (B) FLAG    — if retrieved sources contradict each other, support both "
        "interpretations equally, or if no authoritative source can be found for "
        "the reference year, mark the dimension as AMBIGUOUS.\n\n"
        "Respond with ONLY a valid JSON object (no markdown, no preamble):\n"
        "{\n"
        f'  "resolved_classification": "<exactly: {claude_value} OR {gemini_value} OR FLAGGED>",\n'
        '  "chosen": "<A or B or AMBIGUOUS>",\n'
        '  "flagged": <true or false>,\n'
        '  "rationale": "<2-3 sentences: cite specific evidence from your search, '
        'or explain why sources are contradictory/unavailable>",\n'
        '  "search_source": "<the single most relevant URL you found, or empty string>"\n'
        "}"
    )

    search_sources: list = []
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
        raw           = getattr(response, "output_text", "") or ""
        search_sources = _extract_all_urls_from_response(response)

    except AttributeError:
        # Responses API not available — fall back to chat completions (no live search)
        print(
            "\n    [Responses API unavailable; falling back to chat completions — no live search]",
            end=" ", flush=True,
        )
        fb = _call_with_retry(
            lambda: client.chat.completions.create(
                model=SEARCH_MODEL,
                max_completion_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        raw = fb.choices[0].message.content.strip()

    # Strip markdown fences
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]

    try:
        parsed   = json.loads(raw)
        resolved = parsed.get("resolved_classification", "").strip()
        flagged  = bool(parsed.get("flagged", False))
        chosen   = parsed.get("chosen", "?")

        # Normalise: not a valid value and not FLAGGED → force FLAGGED
        if resolved not in (claude_value, gemini_value, "FLAGGED"):
            resolved = "FLAGGED"
            flagged  = True
            chosen   = "AMBIGUOUS"

        if resolved == "FLAGGED":
            flagged = True
            chosen  = "AMBIGUOUS"

        # Merge API-extracted URLs with whatever GPT reported in JSON
        json_url = parsed.get("search_source", "")
        if json_url and json_url not in search_sources:
            search_sources.append(json_url)

        primary_url = search_sources[0] if search_sources else ""

        return {
            "value":             resolved,
            "source":            "openai_search_flagged" if flagged else "openai_search",
            "chosen_classifier": chosen,
            "flagged":           flagged,
            "rationale":         parsed.get("rationale", ""),
            "search_source":     primary_url,
            "search_sources":    search_sources[:3],
            "conflict_a":        claude_value,
            "conflict_b":        gemini_value,
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "value":             "FLAGGED",
            "source":            "parse_error_flagged",
            "chosen_classifier": "AMBIGUOUS",
            "flagged":           True,
            "rationale":         f"JSON parse error — flagged as ambiguous. Raw: {raw[:120]}",
            "search_source":     search_sources[0] if search_sources else "",
            "search_sources":    search_sources[:3],
            "conflict_a":        claude_value,
            "conflict_b":        gemini_value,
        }


# ---------------------------------------------------------------------------
# Market-level resolver (main export)
# ---------------------------------------------------------------------------

def resolve_market_search(market: dict, client: OpenAI) -> dict:
    """
    Build a fully judged feature matrix for one market using web search.

    - Agreed dimensions (HIGH): pass through Gemini-verified value.
    - Disagreed dimensions (MEDIUM / LOW): call GPT-5.4 + web search to judge.

    Returns:
      {dim: {"value": str, "agreement": str, "source": str, "rationale": str,
             "flagged": bool, "search_source": str, "search_sources": list,
             "conflict_a": str, "conflict_b": str}}
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
                "value":          value,
                "agreement":      agreement,
                "source":         "agreed",
                "rationale":      "Claude and Gemini agreed — no judge required.",
                "flagged":        False,
                "search_source":  "",
                "search_sources": [],
                "conflict_a":     "",
                "conflict_b":     "",
            }
        else:
            print(
                f"    [{dim}] {agreement} — "
                f"claude={claude_value!r} vs gemini={gemini_value!r} "
                f"→ {SEARCH_MODEL}+search ...",
                end=" ", flush=True,
            )
            result = _resolve_dimension(
                client, dim, claude_value, gemini_value, ref_year, domain
            )
            result["agreement"] = agreement
            resolved[dim] = result
            status = "FLAGGED" if result["flagged"] else f"resolved={result['value']!r}"
            print(f"{status}  (chose {result['chosen_classifier']})")
            time.sleep(1.5)

    return resolved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print(f"  VELA MQR — STEP 2b  SEARCH JUDGE  ({SEARCH_MODEL} + Web)")
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
