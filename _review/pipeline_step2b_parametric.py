"""
Vela MQR — Step 2b (Parametric Judge)  [gpt-5.4 edition]

For each dimension where Claude (Role 1) and Gemini (Role 2) disagreed,
calls GPT-5.4 using ONLY its internal parametric knowledge to act as a
neutral third-party judge.

Key change from previous version
---------------------------------
The judge now has two options rather than being forced to choose:

  (A) RESOLVE — choose one of the two provided values with a clear rationale.
  (B) FLAG    — mark as AMBIGUOUS when the historical record genuinely
                supports both interpretations equally, or when the judge
                cannot determine the correct answer with confidence.

FLAGGED dimensions are passed to Role 3 with both options visible so the
scorer can apply a conservative interpretation.

Circularity avoidance
---------------------
- Claude generated the original classifications (Role 1).
- Gemini verified them with Google Search grounding (Role 2).
- GPT-5.4 resolves remaining disagreements from a third, independent
  knowledge base without any web retrieval (parametric only).
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

PARAMETRIC_MODEL = "gpt-5.4"

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
    """Retry on OpenAI rate-limit (429) or server errors."""
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
    Call GPT-5.4 (parametric only, no web search) to judge between
    claude_value and gemini_value for the given dimension.

    The judge may either resolve to one value or flag as AMBIGUOUS.
    Returns a resolution dict with 'flagged: bool'.
    """
    prompt = (
        "You are a neutral expert analyst acting as a third-party judge for a historical "
        "market classification dispute. Use ONLY your internal parametric knowledge — "
        "do not perform or simulate any web search.\n\n"
        f"Reference year : {ref_year}\n"
        f"Dimension      : {dim}\n"
        f"Classifier A   : \"{claude_value}\"\n"
        f"Classifier B   : \"{gemini_value}\"\n\n"
        "Your task: evaluate which classification is more historically accurate.\n\n"
        "You have two options:\n"
        "  (A) RESOLVE — if one classification is clearly more accurate, choose it "
        "and explain why with specific historical evidence.\n"
        "  (B) FLAG    — if the historical record genuinely supports both interpretations "
        "equally, or if you cannot determine the correct answer with confidence, "
        "mark the dimension as AMBIGUOUS. Do NOT flag just to avoid deciding; "
        "only flag when there is real evidential ambiguity.\n\n"
        "Respond with ONLY a valid JSON object (no markdown, no preamble):\n"
        "{\n"
        f'  "resolved_classification": "<exactly: {claude_value} OR {gemini_value} OR FLAGGED>",\n'
        '  "chosen": "<A or B or AMBIGUOUS>",\n'
        '  "flagged": <true or false>,\n'
        '  "rationale": "<2-3 sentences: cite specific historical facts that support '
        'your resolution or explain why both interpretations are defensible>"\n'
        "}"
    )

    response = _call_with_retry(
        lambda: client.chat.completions.create(
            model=PARAMETRIC_MODEL,
            max_completion_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]

    try:
        parsed = json.loads(raw)
        resolved = parsed.get("resolved_classification", "").strip()
        flagged  = bool(parsed.get("flagged", False))
        chosen   = parsed.get("chosen", "?")

        # Normalise: if not one of the two valid values and not FLAGGED, default to flagged
        if resolved not in (claude_value, gemini_value, "FLAGGED"):
            resolved = "FLAGGED"
            flagged  = True
            chosen   = "AMBIGUOUS"

        if resolved == "FLAGGED":
            flagged = True
            chosen  = "AMBIGUOUS"

        return {
            "value":              resolved,
            "source":             "openai_parametric_flagged" if flagged else "openai_parametric",
            "chosen_classifier":  chosen,
            "flagged":            flagged,
            "rationale":          parsed.get("rationale", ""),
            # Preserve both options so Role 3 can see the conflict
            "conflict_a":         claude_value,
            "conflict_b":         gemini_value,
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "value":             "FLAGGED",
            "source":            "parse_error_flagged",
            "chosen_classifier": "AMBIGUOUS",
            "flagged":           True,
            "rationale":         f"JSON parse error — flagged as ambiguous. Raw: {raw[:120]}",
            "conflict_a":        claude_value,
            "conflict_b":        gemini_value,
        }


# ---------------------------------------------------------------------------
# Market-level resolver (main export)
# ---------------------------------------------------------------------------

def resolve_market_parametric(market: dict, client: OpenAI) -> dict:
    """
    Build a fully judged feature matrix for one market.

    - Agreed dimensions (HIGH): pass through Gemini-verified value (Claude fallback).
    - Disagreed dimensions (MEDIUM / LOW): call GPT-5.4 to resolve or flag.

    Returns:
      {dim: {"value": str, "agreement": str, "source": str,
             "rationale": str, "flagged": bool,
             "conflict_a": str, "conflict_b": str}}
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
                "value":             value,
                "agreement":         agreement,
                "source":            "agreed",
                "rationale":         "Claude and Gemini agreed — no judge required.",
                "flagged":           False,
                "conflict_a":        "",
                "conflict_b":        "",
            }
        else:
            print(
                f"    [{dim}] {agreement} — "
                f"claude={claude_value!r} vs gemini={gemini_value!r} "
                f"→ {PARAMETRIC_MODEL} parametric ...",
                end=" ", flush=True,
            )
            result = _resolve_dimension(client, dim, claude_value, gemini_value, ref_year)
            result["agreement"] = agreement
            resolved[dim] = result
            status = "FLAGGED" if result["flagged"] else f"resolved={result['value']!r}"
            print(f"{status}  (chose {result['chosen_classifier']})")
            time.sleep(0.5)

    return resolved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print(f"  VELA MQR — STEP 2b  PARAMETRIC JUDGE  ({PARAMETRIC_MODEL}, no search)")
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
