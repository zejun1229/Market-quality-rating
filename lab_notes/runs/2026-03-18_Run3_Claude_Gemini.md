# Vela MQR — Run 3: Claude Gen + Gemini Pro Verify
### Date: 2026-03-18 | Output: `reference_population_v3.json`

---

## Overview

End-to-end clean pipeline run using the same 3 market seeds as Run 1, with the v2-calibrated prompts and a new retry-on-overload guard added to Step 1. Gemini verification was run with `models/gemini-3.1-pro-preview` and Google Search grounding.

```
pipeline_step1.py  →  reference_population_v3.json  →  pipeline_step2.py  →  final report
   (Claude Sonnet 4.6)        (enriched JSON)           (Gemini 3.1 Pro + Search)
```

**Models used**
- Step 1: `claude-sonnet-4-6`
- Step 2: `models/gemini-3.1-pro-preview` (Google Search grounding enabled)

---

## Final Verification Report

| # | Market | Ref Year | T+5 | Sources | Outcome | Agreement Score |
|---|--------|----------|-----|---------|---------|-----------------|
| 1 | US App-Based On-Demand Black Car Transportation | 2010 | 2015 | 16 | **VERIFIED** | HIGH (0.79) |
| 2 | Consumer and Enterprise Cloud File Storage and Sync | 2009 | 2014 | 21 | **VERIFIED** | HIGH (0.79) |
| 3 | B2B SaaS Marketing Automation Platforms, Mid-Market | 2012 | 2017 | 20 | **VERIFIED** | HIGH (0.93) |

**3 / 3 markets passed the ≥3-source verification threshold.**
**All 3 markets achieved HIGH inter-source agreement — best result across all runs.**

---

## Market 1 — US App-Based On-Demand Black Car Transportation

**Reference year:** 2010 | **T+5 year:** 2015 | **Sources:** 16 | **Agreement: HIGH (0.79)**

### 7-Dimension Feature Extraction & Agreement

| Dimension | Claude | Gemini | Agreement |
|-----------|--------|--------|-----------|
| timing | `pre_chasm` | `pre_chasm` | **HIGH** |
| competition | `nascent` | `nascent` | **HIGH** |
| market_size | `micro` | `micro` | **HIGH** |
| customer_readiness | `adopting` | `adopting` | **HIGH** |
| regulatory | `light_touch` | `heavy` | LOW |
| infrastructure | `developing` | `emerging` | MEDIUM |
| market_structure | `emerging` | `emerging` | **HIGH** |

**Score breakdown:** HIGH=5, MEDIUM=1, LOW=1 → score=0.79 → **HIGH**

### Key Findings

- **Market size calibration success:** In Run 1 Claude scored `large` (TAM-based framing: ~$11B US taxi/limo market). In Run 3 Claude correctly scores `micro` (actual on-category revenue: Uber's SF-only 2010 revenue was negligible). Gemini independently returns `micro` → **HIGH agreement**. The v2 scoring guide ("current observable market spend, NOT speculative future TAM") is working as intended.
- **Regulatory disagreement (genuine):** Claude=`light_touch` (Uber operated under existing livery rules; no new framework enacted). Gemini=`heavy` (likely weighting the SFMTA cease-and-desist Oct 2010 and the hostile-incumbents environment). Both readings are empirically defensible. This is a genuine ambiguity in the `light_touch` vs `moderate`/`heavy` boundary at reference year.
- **Infrastructure one-step gap:** `developing` (Claude) vs `emerging` (Gemini). Both models recognise the enabler stack was immature; the gap reflects whether GPS + App Store + payment APIs at mid-2010 maturity is called "limited to specialist access" or "functional but still maturing." Acceptable adjacent disagreement.

---

## Market 2 — Consumer and Enterprise Cloud File Storage and Sync

**Reference year:** 2009 | **T+5 year:** 2014 | **Sources:** 21 | **Agreement: HIGH (0.79)**

### 7-Dimension Feature Extraction & Agreement

| Dimension | Claude | Gemini | Agreement |
|-----------|--------|--------|-----------|
| timing | `pre_chasm` | `early_chasm` | MEDIUM |
| competition | `nascent` | `fragmented` | MEDIUM |
| market_size | `micro` | `small` | MEDIUM |
| customer_readiness | `adopting` | `adopting` | **HIGH** |
| regulatory | `light_touch` | `light_touch` | **HIGH** |
| infrastructure | `developing` | `developing` | **HIGH** |
| market_structure | `forming` | `forming` | **HIGH** |

**Score breakdown:** HIGH=4, MEDIUM=3, LOW=0 → score=0.79 → **HIGH**

### Key Findings

- **Zero outright disagreements:** Every dimension is within adjacent range (distance ≤ 1). This is the cleanest scoring profile in the batch.
- **Timing split (pre_chasm vs early_chasm):** Claude weighted the fact that Dropbox had 1M users but consumer mainstream had not yet crossed; Gemini weighted Dropbox's active public beta and early chasm-crossing trajectory. One-step difference; both defensible.
- **Competition (nascent vs fragmented):** Dropbox, Box, and SugarSync could be read as 3 players (`nascent`) or as a fragmented field with Microsoft/Apple indirect competitors. One-step gap.
- **Market size (micro vs small):** Dropbox was effectively pre-revenue (freemium) in 2009; total cloud sync category revenue was estimated at $20–50M. Claude=`micro` (<$100M); Gemini=`small`. Both are plausible given the uncertainty in 2009 estimates.

---

## Market 3 — B2B SaaS Marketing Automation Platforms, Mid-Market

**Reference year:** 2012 | **T+5 year:** 2017 | **Sources:** 20 | **Agreement: HIGH (0.93)**

### 7-Dimension Feature Extraction & Agreement

| Dimension | Claude | Gemini | Agreement |
|-----------|--------|--------|-----------|
| timing | `early_majority` | `early_majority` | **HIGH** |
| competition | `consolidating` | `consolidating` | **HIGH** |
| market_size | `small` | `small` | **HIGH** |
| customer_readiness | `adopting` | `adopting` | **HIGH** |
| regulatory | `light_touch` | `light_touch` | **HIGH** |
| infrastructure | `mature` | `developing` | MEDIUM |
| market_structure | `defined` | `defined` | **HIGH** |

**Score breakdown:** HIGH=6, MEDIUM=1, LOW=0 → score=0.93 → **HIGH**

### Key Findings

- **Dramatic improvement over Run 1:** Run 1 scored LOW (0.21) due to Gemini API rate limits returning `unknown` for 4 of 7 dimensions. Run 3 scored HIGH (0.93) with no rate-limit failures — confirming that the Run 1 score was an API artefact, not a genuine disagreement signal.
- **Infrastructure one-step gap:** Claude=`mature` (Salesforce $3B+, REST APIs, AWS fully proven). Gemini=`developing`. This is a borderline case: the specific email deliverability/CRM integration stack was mature, but broader infrastructure from Gemini's perspective may include aspects still evolving.
- **Strongest anchor market:** 6/7 HIGH agreement with 20 grounded sources. This market is well-suited for the reference population.

---

## Cross-Run Comparison (Run 1 vs Run 3)

| Market | Run 1 Score | Run 3 Score | Delta | Notes |
|--------|------------|------------|-------|-------|
| Ride-hailing 2010 | MEDIUM (0.43) | **HIGH (0.79)** | +0.36 | market_size fix (large→micro) resolves 1 LOW; framing now consistent |
| Cloud storage 2009 | HIGH (0.93) | **HIGH (0.79)** | -0.14 | Run 1 used gemini-2.5-flash; Run 3 used gemini-3.1-pro — model now classifies timing/competition/size as one step higher than Claude |
| Marketing automation 2012 | LOW (0.21)* | **HIGH (0.93)** | +0.72 | Run 1 rate-limit artefact; Run 3 is the correct score |

> *Run 1 Market 3 was artificially depressed; Run 3 is the authoritative score.

---

## Pipeline Execution Notes

- **No 529 overloaded errors** encountered in Step 1 (Claude). The `_call_with_retry` guard was added preemptively and was not triggered.
- **No Gemini rate-limit failures** in Step 2. All 24 verification calls (3×T+5 + 3×7-dim) completed successfully.
- **Gemini model:** `models/gemini-3.1-pro-preview` — API call succeeded; grounding returned valid URLs for all markets.
- **Source counts:** Market 1 = 16, Market 2 = 21, Market 3 = 20. All well above the ≥3 threshold.

---

## Observations & Next Steps

### What this run confirms
1. **v2 calibration is working:** The `market_size` scoring guide change (current revenue ≠ TAM) produces consistent Claude/Gemini alignment on Market 1.
2. **Market 3 is a strong reference candidate:** 6/7 HIGH with 20 grounded sources; prior low score was a run artefact.
3. **Regulatory dimension has the most persistent disagreement:** The `light_touch` / `moderate` / `heavy` boundary at pre-regulatory-framework years is the hardest to calibrate. Recommend adding a sub-dimension: `enforcement_action_taken: bool`.

### Recommended next steps
1. Scale to 10–20 markets for statistical validity — Run 3 batch validates the pipeline is stable end-to-end.
2. Add regulatory `enforcement_action_taken` sub-field to resolve the ride-hailing boundary ambiguity.
3. Consider splitting `market_size` into `current_revenue_tier` + `tam_tier` for markets where product is nascent but disrupting a large incumbent industry.
4. Run a `market_size` sensitivity test: re-score ride-hailing with both framings and document which produces better downstream model performance.
