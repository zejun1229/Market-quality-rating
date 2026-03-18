"""
Vela MQR — Step 2b (Parametric Judge)

For each dimension where Claude (Role 1) and Gemini (Role 2) disagreed,
calls GPT-4o using ONLY its internal parametric knowledge to act as a
neutral third-party tie-breaker.

Circularity avoidance
---------------------
- Claude generated the original classifications (Role 1).
- Gemini verified them with Google Search grounding (Role 2).
- GPT-4o resolves remaining disagreements from a third, independent
  knowledge base without any web retrieval (parametric only).

Dimensions where Claude and Gemini already agreed (HIGH) are passed
through unchanged. Only MEDIUM and LOW agreement dimensions are sent
to GPT-4o for resolution.
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

PARAMETRIC_MODEL = "gpt-4o"

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
    """Return True for MEDIUM or LOW agreement — i.e. any non-exact match."""
    return agreement not in ("HIGH", "agreed", "unverified", "")


def _call_with_retry(fn, max_retries: int = 3, base_delay: float = 10.0):
    """Retry on OpenAI rate-limit (429) errors."""
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


# ---------------------------------------------------------------------------
# Single-dimension resolver
# ---------------------------------------------------------------------------

def _resolve_dimension(
    client: OpenAI,
    dim: str,
    claude_value: str,
    gemini_value: str,
    ref_year: int,
) -> dict:
    """
    Call GPT-4o (parametric only, no web search) to choose between
    claude_value and gemini_value for the given dimension.
    Returns a resolution dict.
    """
    prompt = (
        "You are a neutral expert analyst acting as a tie-breaker for a historical "
        "market classification task. Use ONLY your internal parametric knowledge — "
        "do not perform or simulate any web search.\n\n"
        f"Reference year : {ref_year}\n"
        f"Dimension      : {dim}\n"
        f"Classifier A   : \"{claude_value}\"\n"
        f"Classifier B   : \"{gemini_value}\"\n\n"
        "Which classification is more historically accurate for this dimension at "
        "the given reference year?\n"
        "You MUST choose exactly one of the two values above — do not invent a new value.\n\n"
        "Respond with ONLY a valid JSON object (no markdown, no preamble):\n"
        "{\n"
        f'  "resolved_classification": "<exactly: {claude_value} OR {gemini_value}>",\n'
        '  "chosen": "A" or "B",\n'
        '  "rationale": "<1-2 sentences citing specific historical evidence>"\n'
        "}"
    )

    response = _call_with_retry(
        lambda: client.chat.completions.create(
            model=PARAMETRIC_MODEL,
            max_tokens=300,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        parsed = json.loads(raw)
        resolved = parsed.get("resolved_classification", "").strip()
        # Enforce: must be one of the two provided values
        if resolved not in (claude_value, gemini_value):
            resolved = claude_value  # safe fallback to Claude
        return {
            "value":              resolved,
            "source":             "openai_parametric",
            "chosen_classifier":  parsed.get("chosen", "?"),
            "rationale":          parsed.get("rationale", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "value":             claude_value,
            "source":            "fallback_claude",
            "chosen_classifier": "A",
            "rationale":         f"JSON parse error — fell back to Classifier A. Raw: {raw[:120]}",
        }


# ---------------------------------------------------------------------------
# Market-level resolver (main export)
# ---------------------------------------------------------------------------

def resolve_market_parametric(market: dict, client: OpenAI) -> dict:
    """
    Build a fully resolved feature matrix for one market.

    - Agreed dimensions (HIGH): pass through Gemini-verified value (Claude fallback).
    - Disagreed dimensions (MEDIUM / LOW): call GPT-4o to resolve.

    Returns:
      {dim: {"value": str, "agreement": str, "source": str, "rationale": str}}
    """
    step1_dims   = market.get("dimensions", {})
    step2_verifs = market.get("step2", {}).get("dimension_verifications", {})
    ref_year     = market.get("ref_year", 0)

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
                "value":     value,
                "agreement": agreement,
                "source":    "agreed",
                "rationale": "Claude and Gemini agreed — no tie-break required.",
            }
        else:
            print(
                f"    [{dim}] {agreement} — "
                f"claude={claude_value!r} vs gemini={gemini_value!r} → GPT-4o ...",
                end=" ", flush=True,
            )
            result = _resolve_dimension(client, dim, claude_value, gemini_value, ref_year)
            result["agreement"] = agreement
            resolved[dim] = result
            print(f"resolved={result['value']!r}  (chose {result['chosen_classifier']})")
            time.sleep(0.5)

    return resolved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  VELA MQR — STEP 2b  PARAMETRIC JUDGE  (GPT-4o, no search)")
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
        market["step2b_parametric"] = {
            "resolved_feature_matrix": resolve_market_parametric(market, client)
        }

    output_path = os.path.join(project_root, "reference_population_parametric.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"\n  + Saved → reference_population_parametric.json")


if __name__ == "__main__":
    main()
