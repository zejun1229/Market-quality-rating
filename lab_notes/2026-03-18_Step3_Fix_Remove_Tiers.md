# Step 3 Fix — Remove Premature L-Tier / Composite Logic
### Date: 2026-03-18 | Type: Bug Fix (logic error)

---

## Problem

The original `pipeline_step3.py` was asking the LLM to anchor its scoring
against L1–L5 tier labels (e.g. "L5 - Exceptional", "L2 - Neutral") and was
computing a composite score internally. Both were logically premature:

- **Tier labels in the prompt** biased the LLM toward producing scores that
  cluster around tier boundaries (0, 39, 40, 59, 60, ...) rather than making
  granular, independent per-dimension assessments.
- **Composite score** is mathematically meaningless on a 3-market sample.
  The correct method is percentile lookup across the full 120-market database
  (a later pipeline step).
- The LLM had no guardrail preventing it from leaking tier or composite
  fields into its JSON response.

---

## Changes Made to `src/pipeline_step3.py`

| Location | Before | After |
|----------|--------|-------|
| `SCORE_RUBRIC` | Included `L5 - Exceptional`, `L4 - Strong`, etc. labels | Plain numeric ranges only (`90-100:`, `75-89:`, …) |
| `SYSTEM_PROMPT` | No explicit prohibition on tier output | Added 5 STRICT OUTPUT RULES: no L-tiers, no composite, no investment decision, no extra JSON keys |
| `build_scoring_prompt` | Told LLM to use "L1–L5 rubric" | Tells LLM to score each dimension independently using the numeric scale only |
| `score_market` return dict | Included `composite_score`, `composite_level` | Removed both fields entirely |
| `_score_to_level()` | Helper function computing L-tier from integer | **Deleted** |
| `print_final_report` | Showed Level column, composite, league table | Shows only: Market, Year, 7 dimension scores, verified values, agreement |

---

## New System Prompt (key addition)

```
STRICT OUTPUT RULES — you must follow these exactly:
1. Output ONLY a JSON object containing the 7 dimension scores as integers.
2. Do NOT output L-tier labels, tier names, or any classification strings.
3. Do NOT compute or output a composite score or overall rating.
4. Do NOT output an investment recommendation or decision.
5. Do NOT add any keys beyond the 7 required dimension keys.
Tier classification is performed separately via percentile lookup —
your sole responsibility is the 7 integer scores.
```

---

## Re-run Results (reference_population_v3.json, schema v1.1)

| Market | timing | competition | market_size | customer_readiness | regulatory | infrastructure | market_structure |
|--------|--------|-------------|-------------|-------------------|------------|----------------|-----------------|
| Ride-hailing 2010 | 62 | 82 | 22 | 72 | 28 | 55 | 65 |
| Cloud Storage 2009 | 52 | 72 | 28 | 78 | 85 | 62 | 70 |
| Marketing Automation 2012 | 78 | 42 | 22 | 75 | 82 | 55 | 70 |

Scores are raw dimension integers only. No tier, no composite.
Tier classification deferred to Step 4 (percentile lookup).

---

## Output Schema (`step3` block in JSON)

```json
{
  "scores": {
    "timing": 62,
    "competition": 82,
    "market_size": 22,
    "customer_readiness": 72,
    "regulatory": 28,
    "infrastructure": 55,
    "market_structure": 65
  },
  "feature_matrix": {
    "timing": {"value": "pre_chasm", "agreement": "HIGH"},
    "...": "..."
  },
  "validation_errors": []
}
```
