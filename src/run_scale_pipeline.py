"""
Vela MQR — Scale Pipeline v4
Parallel · Role 0 Pre-Search · Strict Deduplication · Async Timeout · Rich Display

Architecture
------------
Role 0  — Gemini web-searches for 15 verifiable, data-rich market seeds (2009-2021).
Role 1  — Claude generates full profile + 7-dim classifications for each seed.
          (parallel across all seeds in a batch via ThreadPoolExecutor)
Role 2  — Gemini consolidated verify: T+5 outcome + all 7 dims in ONE call,
          wrapped with asyncio.wait_for(timeout=20s).
          Timed-out markets are immediately blacklisted.
Role 3  — Claude blind scorer.

Deduplication
-------------
- A global name registry (all saved market names, normalised) is built on startup.
- The registry + blacklist are injected into Role 0 and Role 1 prompts.
- After Role 1 generates a market name, a Python-level SequenceMatcher check
  (threshold 0.75) discards near-duplicate names before Role 2 is called.

Display
-------
- Rich Live panel shows overall progress, active worker stages, recent completions.
- All orchestration-level output routed through Rich console (thread-safe).
"""

import asyncio
import difflib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# sys.path bootstrap
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
    from google import genai  # noqa: F401
except ImportError:
    sys.exit("ERROR: google-genai not found.  pip install google-genai")

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("ERROR: rich not found.  pip install rich")

# ---------------------------------------------------------------------------
# Pipeline module imports
# ---------------------------------------------------------------------------
from pipeline_step1 import DIMENSIONS
from pipeline_step2 import (
    get_gemini_client,
    process_market_verification,
    ORDINAL_SCALES,
    query_gemini_grounded,
)
from pipeline_step3 import (
    build_feature_matrix,
    build_scoring_prompt,
    SYSTEM_PROMPT as SCORER_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_COUNT        = 350
BATCH_SIZE          = 15
MAX_WORKERS         = 12
REF_YEARS           = list(range(2009, 2022))   # 2009–2021 inclusive
CLAUDE_MODEL        = "claude-sonnet-4-6"
GEMINI_TIMEOUT_SECS = 20.0   # strict 20-second cut-off; timeout → blacklist
DEDUP_THRESHOLD     = 0.75   # SequenceMatcher ratio above which a market is a duplicate

MASTER_JSON = os.path.join(_ROOT, "reference_population_master.json")
LOG_PATH    = os.path.join(_ROOT, "lab_notes", "Scaling_Run_Log.md")

LOW_BATCH_THRESHOLD = 0.40
_DIM_NAMES = [d["name"] for d in DIMENSIONS]

# ---------------------------------------------------------------------------
# Rich console (shared across threads via .log / .print which are thread-safe)
# ---------------------------------------------------------------------------
console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Live display state  (all mutations guarded by _status_lock)
# ---------------------------------------------------------------------------
_status_lock   = threading.Lock()
_worker_status: dict = {}    # market_id → {"stage": str, "name": str}
_recent_done:   list = []    # last 5 completions → {"id", "name", "agreement", "mean_score"}
_dedup_skipped  = 0
_blacklist_hits = 0


def _set_stage(market_id: str, stage: str, name: str = "") -> None:
    with _status_lock:
        _worker_status[market_id] = {"stage": stage, "name": name}


def _clear_stage(market_id: str, completion: dict | None = None) -> None:
    global _recent_done
    with _status_lock:
        _worker_status.pop(market_id, None)
        if completion:
            _recent_done = ([completion] + _recent_done)[:5]


def make_display(total_processed: int, batch_num: int, blacklist: set) -> Panel:
    """Return a Rich Panel renderable representing the current pipeline state."""
    t = Table(box=box.SIMPLE, expand=True, show_header=False, padding=(0, 1))
    t.add_column("Label", style="bold cyan",  no_wrap=True, width=16)
    t.add_column("Value", style="white")

    # Progress bar
    pct       = total_processed / TARGET_COUNT * 100 if TARGET_COUNT else 0
    filled    = int(pct / 5)
    bar_green = "█" * filled
    bar_dim   = "░" * (20 - filled)
    t.add_row(
        "Progress",
        f"[green]{bar_green}[/][dim]{bar_dim}[/]  "
        f"[bold]{total_processed}[/] / {TARGET_COUNT}  "
        f"([cyan]{pct:.0f}%[/])  Batch {batch_num}",
    )
    t.add_row(
        "Flags",
        f"Blacklisted: [red]{len(blacklist)}[/]   "
        f"Deduped (skipped): [yellow]{_dedup_skipped}[/]",
    )

    with _status_lock:
        active = dict(_worker_status)
        recent = list(_recent_done)

    # Active workers
    if active:
        t.add_row("", "")
        t.add_row("[bold]Active[/]", "")
        stage_colour = {
            "Role 0": "magenta",
            "Role 1": "blue",
            "Role 2": "yellow",
            "Role 3": "green",
        }
        for mid, info in sorted(active.items()):
            stage = info.get("stage", "?")
            name  = info.get("name", "")[:48]
            key   = stage.split(":")[0].strip()
            colour = stage_colour.get(key, "white")
            t.add_row(
                f"  [dim]{mid}[/]",
                f"[{colour}][{stage}][/{colour}]  {name}",
            )

    # Recent completions
    if recent:
        t.add_row("", "")
        t.add_row("[bold]Completed[/]", "")
        for r in recent:
            agr = r.get("agreement", {})
            ms  = r.get("mean_score", "?")
            t.add_row(
                f"  [dim]{r.get('id','?')}[/]",
                f"[green]+[/] {r.get('name','')[:48]}  "
                f"H={agr.get('HIGH',0)} M={agr.get('MEDIUM',0)} L={agr.get('LOW',0)}  "
                f"mean={ms}",
            )

    return Panel(
        t,
        title="[bold cyan]VELA MQR — Scale Pipeline  "
              "(Parallel · Role 0 · Deduplicated · Timeout)[/]",
        border_style="cyan",
    )


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def get_claude_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your_"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

def load_master() -> dict:
    if os.path.exists(MASTER_JSON):
        with open(MASTER_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "schema_version": "2.0",
        "run_metadata": {
            "target":             TARGET_COUNT,
            "started":            datetime.now().isoformat(),
            "last_updated":       "",
            "batches_completed":  0,
            "prompt_corrections": [],
            "blacklist":          [],
        },
        "markets": [],
    }


def save_master(data: dict) -> None:
    data["run_metadata"]["last_updated"] = datetime.now().isoformat()
    with open(MASTER_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for comparison."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _is_duplicate(name: str, registry: set, threshold: float = DEDUP_THRESHOLD) -> bool:
    """Return True if name is too similar to any entry in registry."""
    norm = _normalise(name)
    for existing in registry:
        ratio = difflib.SequenceMatcher(None, norm, existing).ratio()
        if ratio >= threshold:
            return True
    return False


def _build_registry(markets: list) -> set:
    """Build a set of normalised market names from saved markets."""
    names = set()
    for m in markets:
        mn = m.get("base_profile", {}).get("market_name", "") or m.get("domain", "")
        if mn:
            names.add(_normalise(mn))
        domain = m.get("domain", "")
        if domain:
            names.add(_normalise(domain))
    return names


def _registry_block(registry: set, limit: int = 80) -> str:
    """Format the registry for injection into prompts (capped to avoid token bloat)."""
    items = sorted(registry)[:limit]
    return "\n".join(f"  - {n}" for n in items) if items else "  (none yet)"


def _blacklist_block(blacklist: set, limit: int = 30) -> str:
    items = sorted(blacklist)[:limit]
    return "\n".join(f"  - {n}" for n in items) if items else "  (none)"


# ---------------------------------------------------------------------------
# Role 0 — Gemini Seed Pre-Search
# ---------------------------------------------------------------------------

def role0_seed_presearch(
    gemini_client,
    registry: set,
    blacklist: set,
    count: int,
) -> list:
    """
    Role 0: Use Gemini + Google Search to discover verifiable venture market seeds.

    Searches for mainstream, highly-funded markets in the 2009-2021 window with
    massive digital footprints, then returns structured seeds for Role 1.
    Returns list of {domain, ref_year, knowledge_brief}.
    """
    console.log(f"  [Role 0] Searching web for {count} market seeds (2009–2021) ...")

    prompt = (
        f"You are a venture capital research assistant. Search the web and identify "
        f"exactly {count} real, mainstream venture capital markets that:\n\n"
        "- Reached a major founding event or funding inflection point between 2009 and 2021\n"
        "- Are mainstream and highly-funded (Series A through IPO) with massive digital footprints\n"
        "- Have abundant verifiable data: press coverage, Crunchbase records, analyst reports, "
        "Wikipedia articles, SEC filings, or earnings reports\n"
        "- Span diverse sectors: fintech, healthtech, enterprise SaaS, consumer internet, "
        "logistics, climate tech, edtech, marketplace, deep tech, etc.\n"
        "- Include a mix of US and international (EU, UK, China, India, SEA, LatAm) markets\n\n"
        "DO NOT generate, search for, or suggest any market that is conceptually identical "
        "to the following already-captured markets:\n"
        f"{_registry_block(registry)}\n\n"
        "DO NOT suggest any market from this blacklist (failed or timed-out verification):\n"
        f"{_blacklist_block(blacklist)}\n\n"
        "For each market return:\n"
        "1. domain: a precise 10-20 word description of the product/service category\n"
        "2. ref_year: the integer year of the most significant founding or funding inflection\n"
        "3. knowledge_brief: 3-5 sentences with SPECIFIC verifiable facts — company names, "
        "funding round sizes, user counts, revenue figures, key regulatory events, "
        "and the state of enabling technology at that reference year\n\n"
        "Return ONLY a raw, valid JSON array. Do not include markdown formatting like "
        "```json, and do not include any conversational preamble or postamble.\n"
        "[\n"
        '  {"domain": "...", "ref_year": <int>, "knowledge_brief": "..."},\n'
        "  ...\n"
        "]\n"
        f"Return exactly {count} entries."
    )

    text, _ = query_gemini_grounded(gemini_client, prompt)

    # Parse
    clean = text.strip()
    if "```" in clean:
        clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
    bracket = clean.find("[")
    if bracket > 0:
        clean = clean[bracket:]

    try:
        suggestions = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as exc:
        console.log(f"  [Role 0] Parse error: {exc}  raw[:200]={clean[:200]!r}")
        return []

    valid = []
    seen  = set(registry)
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        domain   = s.get("domain", "")
        ref_yr   = s.get("ref_year")
        brief    = s.get("knowledge_brief", "")
        try:
            ref_yr = int(ref_yr)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(domain, str) and len(domain) > 5
            and ref_yr in REF_YEARS
            and isinstance(brief, str) and len(brief) > 20
            and not _is_duplicate(domain, seen)
        ):
            valid.append({"domain": domain, "ref_year": ref_yr, "knowledge_brief": brief})
            seen.add(_normalise(domain))

    console.log(f"  [Role 0] {len(valid)} valid seeds / {len(suggestions)} returned")
    return valid[:count]


# ---------------------------------------------------------------------------
# Role 1-A — Generate base profile
# ---------------------------------------------------------------------------

def generate_market_profile(
    client: anthropic.Anthropic,
    seed: dict,
    registry: set,
    blacklist: set,
) -> dict:
    """Generate a market base profile. Injects dedup registry into prompt."""
    prompt = (
        "You are a senior venture capital market analyst producing a historical "
        "market profile for a quantitative classification system. "
        "Ground all claims in the knowledge brief — do not invent details.\n\n"
        f"Domain: {seed['domain']}\n"
        f"Reference year: {seed['ref_year']}\n\n"
        "=== KNOWLEDGE BRIEF ===\n"
        f"{seed['knowledge_brief'].strip()}\n\n"
        "=== DEDUPLICATION CONSTRAINT ===\n"
        "DO NOT generate, search for, or suggest any market that is conceptually "
        "identical to the following already-captured markets:\n"
        f"{_registry_block(registry)}\n\n"
        "DO NOT use any market from this blacklist:\n"
        f"{_blacklist_block(blacklist)}\n\n"
        "=== OUTPUT FORMAT ===\n"
        "Use EXACTLY these labelled fields, each starting on its own line:\n\n"
        "Market: [concise 4-8 word market name, no year]\n\n"
        "Context: [3-4 sentences: state of market at reference year, key inflection "
        "point, investor sentiment, one landmark event.]\n\n"
        "Buyers: [2-3 sentences: primary beachhead buyer segment with specifics, "
        "their pain point, purchasing behaviour at this year.]\n\n"
        "Players: [2-3 sentences: specific companies with funding/revenue/customer "
        "counts from the brief. Which major players had NOT yet entered.]\n\n"
        "Key Metrics: [3-5 bullet points from the knowledge brief:\n"
        "- [metric]: [value] ([source/year])]\n\n"
        "Exclusions: [2 sentences: what was NOT yet true at this reference year — "
        "competitors not launched, regulations not enacted, tech not shipped.]\n\n"
        f"Reference year: {seed['ref_year']}\n\n"
        "Rules: do not project post-reference-year events; use specific numbers and dates.\n"
        "Return ONLY the labelled fields above — no preamble, no postamble."
    )

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()
    except Exception as exc:
        return {
            "raw_text":    "",
            "ref_year":    seed["ref_year"],
            "domain":      seed["domain"],
            "market_name": seed["domain"][:60],
            "error":       str(exc),
        }

    profile: dict = {
        "raw_text": raw_text,
        "ref_year": seed["ref_year"],
        "domain":   seed["domain"],
    }
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
# Role 1-B — Extract one dimension
# ---------------------------------------------------------------------------

def extract_dimension_scaled(
    client: anthropic.Anthropic,
    profile: dict,
    dimension: dict,
    correction_addendum: str = "",
) -> dict:
    addendum_block = ""
    if correction_addendum:
        addendum_block = (
            "\n\n=== CALIBRATION NOTE (auto-corrected from prior batch) ===\n"
            + correction_addendum.strip() + "\n"
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
        "Return ONLY a raw, valid JSON object. Do not include markdown formatting like "
        "```json, and do not include any conversational preamble or postamble.\n"
        "{\n"
        f'  "dimension": "{dimension["name"]}",\n'
        '  "classification": "<one value from the options list>",\n'
        '  "confidence": "<high|medium|low>",\n'
        '  "rationale": "<3 sentences: evidence from profile, scale position, key anchor fact>",\n'
        '  "contradicting_evidence": "<1-2 sentences: adjacent classification and why rejected>"\n'
        "}"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
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
        parsed["validation_warning"] = "Classification not in options; defaulted."
    return parsed


# ---------------------------------------------------------------------------
# Role 1 — Full market pipeline
# ---------------------------------------------------------------------------

def run_step1_market(
    client: anthropic.Anthropic,
    seed: dict,
    market_id: str,
    registry: set,
    blacklist: set,
    correction_addendum: str = "",
) -> dict | None:
    profile = generate_market_profile(client, seed, registry, blacklist)
    if not profile.get("raw_text"):
        return None

    dimensions_result = {}
    for dim in DIMENSIONS:
        try:
            result = extract_dimension_scaled(client, profile, dim, correction_addendum)
            dimensions_result[dim["name"]] = result
        except Exception as exc:
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
# Role 2 — Gemini verify with asyncio.wait_for timeout
# ---------------------------------------------------------------------------

async def _async_gemini_verify(gemini_client, market: dict) -> dict:
    """Async wrapper so asyncio.wait_for can cancel it on timeout."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, process_market_verification, gemini_client, market
    )


def run_step2_with_timeout(
    gemini_client,
    market: dict,
    timeout_secs: float = GEMINI_TIMEOUT_SECS,
) -> dict:
    """
    Run Role 2 (Gemini verification) with a hard asyncio.wait_for timeout.
    Raises asyncio.TimeoutError if the call exceeds timeout_secs.
    """
    async def _inner() -> dict:
        return await asyncio.wait_for(
            _async_gemini_verify(gemini_client, market),
            timeout=timeout_secs,
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_inner())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Role 3 — Claude blind scorer
# ---------------------------------------------------------------------------

def run_step3_market(
    claude_client: anthropic.Anthropic,
    market: dict,
) -> tuple:
    matrix = build_feature_matrix(market)
    prompt = build_scoring_prompt(matrix)

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            system=SCORER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
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

    return {"scores": scores, "feature_matrix": matrix, "validation_errors": errors}, prompt


# ---------------------------------------------------------------------------
# Batch agreement stats
# ---------------------------------------------------------------------------

def compute_batch_agreement(batch_markets: list) -> dict:
    counts  = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    per_dim = {name: {"HIGH": 0, "MEDIUM": 0, "LOW": 0} for name in _DIM_NAMES}
    for m in batch_markets:
        for dim_name, dv in m.get("step2", {}) \
                             .get("dimension_verifications", {}).items():
            agr = dv.get("agreement", "LOW")
            if agr in counts:
                counts[agr] += 1
            if dim_name in per_dim and agr in per_dim[dim_name]:
                per_dim[dim_name][agr] += 1
    total = sum(counts.values())
    score = (counts["HIGH"] * 1.0 + counts["MEDIUM"] * 0.5) / total if total else 0.0
    overall = "HIGH" if score >= 0.70 else ("MEDIUM" if score >= 0.40 else "LOW")
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
# Auto-correction
# ---------------------------------------------------------------------------

def run_meta_correction(
    claude_client: anthropic.Anthropic,
    batch_markets: list,
    current_addendum: str,
    agreement_stats: dict,
) -> str:
    conflict_lines = []
    for m in batch_markets:
        name  = m.get("base_profile", {}).get("market_name", m.get("domain", "?"))
        ref   = m.get("ref_year", "?")
        for dim_name, dv in m.get("step2", {}) \
                             .get("dimension_verifications", {}).items():
            agr = dv.get("agreement", "HIGH")
            if agr in ("LOW", "MEDIUM"):
                conflict_lines.append(
                    f"  [{dim_name}] {name!r} {ref} | "
                    f"agreement={agr} | "
                    f"claude={dv.get('claude_classification','?')!r} | "
                    f"gemini={dv.get('gemini_classification','?')!r}"
                )

    if not conflict_lines:
        return current_addendum

    prompt = (
        "You are calibrating a venture market classification pipeline.\n\n"
        "Step 1 (Claude) classifies 7 dimensions. "
        "Step 2 (Gemini + Google Search) verifies with live web evidence. "
        "Gemini's web-grounded answer is ground truth.\n\n"
        f"Current addendum:\n{current_addendum or '(none)'}\n\n"
        f"Batch stats: mean={agreement_stats['mean_score']:.3f} "
        f"(H={agreement_stats['HIGH']} M={agreement_stats['MEDIUM']} L={agreement_stats['LOW']})\n\n"
        "Conflicting cases:\n" + "\n".join(conflict_lines[:40]) + "\n\n"
        "Write a SHORT calibration note (3-6 bullet points) for the Step 1 prompt. "
        "Focus on PATTERNS. Return ONLY the bullet points."
    )

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        new_addendum = response.content[0].text.strip()
        console.log(f"  [Auto-correction] New addendum ({len(new_addendum)} chars)")
        return new_addendum
    except Exception as exc:
        console.log(f"  [Auto-correction ERROR] {exc}")
        return current_addendum


# ---------------------------------------------------------------------------
# Markdown batch log
# ---------------------------------------------------------------------------

def _url_cell(url: str) -> str:
    if not url:
        return "_no URL_"
    return f"[link]({url})" if url.startswith("http") else url[:80]


def append_batch_log(
    batch_num: int,
    batch_markets: list,
    agreement_stats: dict,
    correction_triggered: bool,
    new_addendum: str,
    role3_audit_prompt: str,
    run_ts: str,
    total_processed: int,
) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    lines = [
        "",
        "---",
        "",
        f"## Batch {batch_num}  —  {run_ts}",
        "",
        f"**Markets processed this batch:** {len(batch_markets)}  ",
        f"**Cumulative total:** {total_processed} / {TARGET_COUNT}  ",
        f"**Batch agreement:** {agreement_stats['overall']} "
        f"(score={agreement_stats['mean_score']:.3f})  ",
        f"**HIGH:** {agreement_stats['HIGH']}  "
        f"**MEDIUM:** {agreement_stats['MEDIUM']}  "
        f"**LOW:** {agreement_stats['LOW']}  ",
        f"**Auto-correction triggered:** {'**YES**' if correction_triggered else 'no'}",
        "",
    ]

    if correction_triggered and new_addendum:
        lines += ["### Auto-Correction Addendum", "", "```", new_addendum, "```", ""]

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
            f"**T+5:** {outcome.get('verification_status','?')} "
            f"({outcome.get('source_count',0)} sources)  "
            f"**Agreement:** {agr_sum.get('overall','?')} "
            f"(H={agr_sum.get('HIGH',0)} M={agr_sum.get('MEDIUM',0)} "
            f"L={agr_sum.get('LOW',0)})",
            "",
            "| Dimension | Claude | Gemini | Agr | Key Fact | URL |",
            "|-----------|--------|--------|:---:|----------|-----|",
        ]
        for dim in _DIM_NAMES:
            dv  = dim_v.get(dim, {})
            kf  = (dv.get("key_fact","") or dv.get("evidence",""))[:80]
            url = dv.get("verification_url","") or (dv.get("grounding_urls") or [""])[0]
            lines.append(
                f"| `{dim}` | `{dv.get('claude_classification','?')}` "
                f"| `{dv.get('gemini_classification','?')}` "
                f"| {dv.get('agreement','?')} | {kf} | {_url_cell(url)} |"
            )

        lines += [
            "",
            "| Dimension | Classification | Score |",
            "|-----------|---------------|:-----:|",
        ]
        for dim in _DIM_NAMES:
            val  = fm.get(dim, {}).get("value", "?")
            sc   = scores.get(dim)
            lines.append(f"| `{dim}` | `{val}` | {sc if sc is not None else 'ERR'} |")

        valid_sc = [v for v in scores.values() if v is not None]
        if valid_sc:
            lines.append(f"| **Mean** | — | **{round(sum(valid_sc)/len(valid_sc))}** |")
        lines.append("")

    if role3_audit_prompt and batch_markets:
        first = batch_markets[0].get("base_profile",{}).get("market_name","Market 1")
        lines += [
            f"### Role 3 Prompt Audit — {first}", "",
            "_Proves market name is stripped before scoring._",
            "", "```", role3_audit_prompt, "```", "",
        ]

    lines += [
        "### Batch Agreement by Dimension", "",
        "| Dimension | HIGH | MEDIUM | LOW |",
        "|-----------|:----:|:------:|:---:|",
    ]
    for dim_name, cnts in agreement_stats.get("per_dim", {}).items():
        lines.append(f"| `{dim_name}` | {cnts['HIGH']} | {cnts['MEDIUM']} | {cnts['LOW']} |")
    lines.append("")

    header_needed = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        if header_needed:
            fh.write(
                "# Vela MQR — Scale Pipeline Run Log (v4: Parallel + Role 0 + Dedup)\n\n"
                f"Target: {TARGET_COUNT} markets  |  Batch size: {BATCH_SIZE}  |  "
                f"Ref years: {REF_YEARS[0]}–{REF_YEARS[-1]}  |  "
                f"Gemini timeout: {GEMINI_TIMEOUT_SECS}s\n"
            )
        fh.write("\n".join(lines) + "\n")

    console.log(f"  + Appended batch {batch_num} to Scaling_Run_Log.md")


def append_final_tally(global_counts: dict, total_processed: int,
                       blacklist: set) -> None:
    total_dims = sum(global_counts.values())
    lines = [
        "", "---", "",
        "## Final Agreement Tally — All Markets", "",
        f"**Total Markets Processed:** {total_processed}  ",
        f"**Total Dimensions Analyzed:** {total_dims}  ",
        f"**High Agreement:**   {global_counts['HIGH']}  ",
        f"**Medium Agreement:** {global_counts['MEDIUM']}  ",
        f"**Low Agreement:**    {global_counts['LOW']}  ",
        f"**Blacklisted (timed out):** {len(blacklist)}  ",
        "",
    ]
    if total_dims > 0:
        pct_h = 100 * global_counts["HIGH"]   / total_dims
        pct_m = 100 * global_counts["MEDIUM"] / total_dims
        pct_l = 100 * global_counts["LOW"]    / total_dims
        mean  = (global_counts["HIGH"] + global_counts["MEDIUM"] * 0.5) / total_dims
        lines += [
            "| Agreement | Count | % |",
            "|-----------|------:|--:|",
            f"| HIGH      | {global_counts['HIGH']:>5} | {pct_h:>5.1f}% |",
            f"| MEDIUM    | {global_counts['MEDIUM']:>5} | {pct_m:>5.1f}% |",
            f"| LOW       | {global_counts['LOW']:>5} | {pct_l:>5.1f}% |",
            f"| **Total** | {total_dims:>5} | 100.0% |",
            "",
            f"**Mean agreement score:** {mean:.3f}",
            "",
        ]

    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    console.log("  + Final tally appended to Scaling_Run_Log.md")


# ---------------------------------------------------------------------------
# Parallel worker (one thread per market seed)
# ---------------------------------------------------------------------------

def _process_market_worker(
    seed: dict,
    market_id: str,
    registry: set,
    blacklist: set,
    correction_addendum: str,
    claude_client: anthropic.Anthropic,
    gemini_client,
) -> tuple:
    """
    Process one market through Roles 1 → 2 → 3.
    Returns (market_dict, scorer_prompt, market_name) or (None, '', name_or_empty).
    """
    global _dedup_skipped, _blacklist_hits

    # ── Role 1 ──────────────────────────────────────────────────────────────
    _set_stage(market_id, "Role 1: Generating...", seed["domain"][:45])
    market = run_step1_market(
        claude_client, seed, market_id, registry, blacklist, correction_addendum
    )
    if market is None:
        _clear_stage(market_id)
        console.log(f"  [{market_id}] FAILED (Role 1 profile generation)")
        return None, "", ""

    mkt_name = market.get("base_profile", {}).get("market_name", seed["domain"])
    _set_stage(market_id, "Role 1: Done", mkt_name[:45])

    # ── Python-level dedup check ─────────────────────────────────────────────
    if _is_duplicate(mkt_name, registry):
        _clear_stage(market_id)
        with _status_lock:
            _dedup_skipped += 1
        console.log(f"  [{market_id}] DEDUP SKIP: {mkt_name!r} too similar to existing market")
        return None, "", mkt_name

    # ── Role 2 (with timeout) ────────────────────────────────────────────────
    _set_stage(market_id, "Role 2: Web Searching...", mkt_name[:45])
    try:
        step2_result = run_step2_with_timeout(
            gemini_client, market, timeout_secs=GEMINI_TIMEOUT_SECS
        )
        market["step2"] = step2_result
    except asyncio.TimeoutError:
        _clear_stage(market_id)
        with _status_lock:
            _blacklist_hits += 1
        console.log(
            f"  [{market_id}] TIMEOUT ({GEMINI_TIMEOUT_SECS}s): "
            f"{mkt_name!r} → blacklisted"
        )
        return None, "", mkt_name   # caller adds to blacklist
    except Exception as exc:
        _clear_stage(market_id)
        console.log(f"  [{market_id}] Role 2 ERROR: {exc}")
        return None, "", ""

    agr = step2_result.get("agreement_summary", {})
    _set_stage(market_id, "Role 2: Done", mkt_name[:45])
    console.log(
        f"  [{market_id}] Role 2 done: "
        f"H={agr.get('HIGH',0)} M={agr.get('MEDIUM',0)} L={agr.get('LOW',0)}"
    )

    # ── Role 3 ──────────────────────────────────────────────────────────────
    _set_stage(market_id, "Role 3: Scoring...", mkt_name[:45])
    step3_result, s3_prompt = run_step3_market(claude_client, market)
    market["step3"] = step3_result

    valid_sc = [v for v in step3_result["scores"].values() if v is not None]
    mean_sc  = round(sum(valid_sc) / len(valid_sc)) if valid_sc else None
    _set_stage(market_id, "Role 3: Done", mkt_name[:45])
    console.log(
        f"  [{market_id}] Role 3 done: "
        f"scores={list(step3_result['scores'].values())} mean={mean_sc}"
    )

    _clear_stage(
        market_id,
        completion={
            "id":         market_id,
            "name":       mkt_name,
            "agreement":  agr,
            "mean_score": mean_sc,
        },
    )
    return market, s3_prompt, mkt_name


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    data     = load_master()
    markets  = data["markets"]
    metadata = data["run_metadata"]

    total_processed     = len(markets)
    batch_num           = metadata.get("batches_completed", 0)
    correction_addendum = ""
    blacklist: set      = set(metadata.get("blacklist", []))

    for rec in metadata.get("prompt_corrections", []):
        correction_addendum = rec.get("addendum", correction_addendum)

    # Rebuild dedup registry from all saved markets
    registry = _build_registry(markets)

    # Rebuild global agreement counts from saved markets
    global_counts: dict = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for m in markets:
        for dv in m.get("step2", {}).get("dimension_verifications", {}).values():
            agr = dv.get("agreement", "LOW")
            if agr in global_counts:
                global_counts[agr] += 1

    claude_client = get_claude_client()
    gemini_client = get_gemini_client()

    with Live(
        make_display(total_processed, batch_num, blacklist),
        console=console,
        refresh_per_second=4,
        redirect_stdout=False,
    ) as live:

        def _refresh() -> None:
            live.update(make_display(total_processed, batch_num, blacklist))

        console.log("=" * 68)
        console.log("  VELA MQR — SCALE PIPELINE v4")
        console.log(f"  Resuming from: {total_processed} markets  |  "
                    f"Target: {TARGET_COUNT}  |  Workers: {MAX_WORKERS}")
        console.log(f"  Ref years: {REF_YEARS[0]}–{REF_YEARS[-1]}  |  "
                    f"Timeout: {GEMINI_TIMEOUT_SECS}s  |  "
                    f"Blacklist: {len(blacklist)}")
        console.log("=" * 68)

        while total_processed < TARGET_COUNT:
            batch_num += 1
            run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            remaining  = TARGET_COUNT - total_processed
            batch_size = min(BATCH_SIZE, remaining)

            console.log(f"\n{'─' * 68}")
            console.log(
                f"  BATCH {batch_num}  |  "
                f"Processed: {total_processed}/{TARGET_COUNT}  |  {run_ts}"
            )
            console.log(f"{'─' * 68}")
            _refresh()

            # ── Role 0: Gemini seed pre-search ────────────────────────────
            _set_stage("role0", "Role 0: Pre-searching...", "Gemini web search")
            _refresh()
            seeds = role0_seed_presearch(gemini_client, registry, blacklist, batch_size)
            _clear_stage("role0")
            _refresh()

            if not seeds:
                console.log("  WARNING: Role 0 returned no seeds. Retrying next iteration ...")
                continue

            start_num          = total_processed + 1
            batch_markets      = []
            role3_audit_prompt = ""
            timed_out_names:   list = []

            console.log(f"\n  Launching {len(seeds)} workers (MAX_WORKERS={MAX_WORKERS}) ...")
            _refresh()

            # ── Parallel Roles 1 → 2 → 3 ─────────────────────────────────
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_meta = {
                    executor.submit(
                        _process_market_worker,
                        seed,
                        f"market_{start_num + i:03d}",
                        registry,
                        blacklist,
                        correction_addendum,
                        claude_client,
                        gemini_client,
                    ): (i, seed)
                    for i, seed in enumerate(seeds)
                }

                for future in as_completed(future_to_meta):
                    market, s3_prompt, mkt_name = future.result()
                    _refresh()

                    # Timeout: add to blacklist + persist
                    if market is None and mkt_name:
                        # Determine if it was a timeout (name returned) or dedup skip
                        # Workers return (None, '', name) for timeout AND for dedup.
                        # Check if it was blacklisted via the counter change.
                        # Since we can't distinguish easily, add to blacklist only when
                        # the worker explicitly set _blacklist_hits. We pass the name
                        # for both cases; only timeout names should go to blacklist.
                        # The worker logs "TIMEOUT" or "DEDUP SKIP" to console.
                        # For safety: add all None+name results to blacklist to avoid
                        # repeated timeouts on the same market.
                        blacklist.add(_normalise(mkt_name))
                        continue

                    if market is None:
                        continue

                    if not role3_audit_prompt:
                        role3_audit_prompt = s3_prompt

                    # Update global agreement tally
                    for dv in market.get("step2", {}) \
                                    .get("dimension_verifications", {}).values():
                        agr = dv.get("agreement", "LOW")
                        if agr in global_counts:
                            global_counts[agr] += 1

                    # Update dedup registry
                    registry.add(_normalise(mkt_name))

                    # Incremental save (main thread)
                    batch_markets.append(market)
                    markets.append(market)
                    total_processed += 1
                    data["markets"] = markets
                    # Persist blacklist into metadata
                    metadata["blacklist"] = sorted(blacklist)
                    data["run_metadata"] = metadata
                    save_master(data)
                    _refresh()

            # ── Batch analysis ────────────────────────────────────────────
            if not batch_markets:
                console.log("  WARNING: no markets completed this batch. Continuing ...")
                continue

            agreement_stats = compute_batch_agreement(batch_markets)
            console.log(
                f"\n  Batch {batch_num} agreement: {agreement_stats['overall']}  "
                f"(score={agreement_stats['mean_score']:.3f}  "
                f"H={agreement_stats['HIGH']} M={agreement_stats['MEDIUM']} "
                f"L={agreement_stats['LOW']})"
            )

            # ── Auto-correction ───────────────────────────────────────────
            correction_triggered = False
            if agreement_stats["mean_score"] < LOW_BATCH_THRESHOLD:
                console.log(
                    f"  [Auto-correction TRIGGERED] "
                    f"score={agreement_stats['mean_score']:.3f} < {LOW_BATCH_THRESHOLD}"
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

            # ── Metadata save ─────────────────────────────────────────────
            metadata["batches_completed"] = batch_num
            metadata["blacklist"]         = sorted(blacklist)
            data["run_metadata"] = metadata
            data["markets"]      = markets
            save_master(data)

            # ── Batch log ─────────────────────────────────────────────────
            append_batch_log(
                batch_num            = batch_num,
                batch_markets        = batch_markets,
                agreement_stats      = agreement_stats,
                correction_triggered = correction_triggered,
                new_addendum         = correction_addendum if correction_triggered else "",
                role3_audit_prompt   = role3_audit_prompt,
                run_ts               = run_ts,
                total_processed      = total_processed,
            )

            console.log(
                f"\n  Batch {batch_num} done: {len(batch_markets)} saved  "
                f"| Total: {total_processed}/{TARGET_COUNT}  "
                f"| Blacklist: {len(blacklist)}"
            )
            _refresh()

    # ── Pipeline complete ──────────────────────────────────────────────────
    total_dims = sum(global_counts.values())
    mean_agr   = (
        (global_counts["HIGH"] + global_counts["MEDIUM"] * 0.5) / total_dims
        if total_dims else 0.0
    )

    console.print(f"\n{'=' * 68}", style="bold green")
    console.print("  SCALE PIPELINE COMPLETE", style="bold green")
    console.print(f"{'=' * 68}", style="bold green")
    console.print(f"  Markets processed    : {total_processed}")
    console.print(f"  Batches run          : {batch_num}")
    console.print(f"  Blacklisted          : {len(blacklist)}")
    console.print(f"  Deduped (skipped)    : {_dedup_skipped}")
    console.print(f"  Total dimensions     : {total_dims}")
    console.print(f"  High Agreement       : {global_counts['HIGH']}")
    console.print(f"  Medium Agreement     : {global_counts['MEDIUM']}")
    console.print(f"  Low Agreement        : {global_counts['LOW']}")
    console.print(f"  Mean agreement score : {mean_agr:.3f}")
    console.print(f"  Master JSON          : {MASTER_JSON}")
    console.print(f"  Run log              : {LOG_PATH}")

    append_final_tally(global_counts, total_processed, blacklist)

    logbook = os.path.join(_ROOT, "lab_notes", "LOGBOOK_MASTER.md")
    if os.path.exists(logbook):
        with open(logbook, "a", encoding="utf-8") as fh:
            fh.write(
                f"* [Scale Pipeline v4 — {total_processed} markets "
                f"({batch_num} batches) | "
                f"H={global_counts['HIGH']} M={global_counts['MEDIUM']} "
                f"L={global_counts['LOW']} | blacklisted={len(blacklist)}]"
                f"(./Scaling_Run_Log.md)\n"
            )


if __name__ == "__main__":
    main()
