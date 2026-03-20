# Step 3: Scorer & Rater — First Run
### Date: 2026-03-18 | Input: `reference_population_v3.json` | Output: `reference_population_scored.json`

---

## Overview

First run of `pipeline_step3.py`. Claude acts as a blind, objective scorer: it receives **only** the anonymized categorical feature matrix (no market names, no profiles, no rationale) and returns 7 integer scores (0–100) mapped to the L1–L5 venture-readiness rubric.

```
reference_population_v3.json  →  pipeline_step3.py  →  reference_population_scored.json
   (Step 2 enriched)               (Claude, blind scorer)    (fully enriched, all 3 steps)
```

**Model:** `claude-sonnet-4-6`
**Input schema version:** `1.1` (Run 3 data, pre-v3 enum upgrade — old classification strings)
**No 529 errors. No validation warnings. All 3 markets scored successfully.**

---

## Scoring Rubric (L1–L5)

| Level | Score Range | Meaning |
|-------|-------------|---------|
| L5 – Exceptional | 90–100 | Ideal conditions for outsized venture returns |
| L4 – Strong | 75–89 | Highly attractive, minor friction |
| L3 – Favorable | 60–74 | Baseline investment-grade threshold; viable with execution excellence |
| L2 – Neutral | 40–59 | Sub-optimal; significant structural headwinds |
| L1 – Speculative | 0–39 | Hostile conditions for new venture entrants |

---

## Composite Score League Table

| Rank | Market | Ref Year | Composite | Level |
|------|--------|----------|-----------|-------|
| 1 | Consumer & Enterprise Cloud File Storage Sync | 2009 | **62/100** | L3-Favorable |
| 1 | B2B SaaS Marketing Automation Platforms | 2012 | **62/100** | L3-Favorable |
| 3 | US On-Demand Black Car Transportation | 2010 | **57/100** | L2-Neutral |

---

## Market 1 — US On-Demand Black Car Transportation (2010)

**Composite: 57/100 [L2-Neutral]**

| Dimension | Score | Level | Verified Value | Agreement |
|-----------|-------|-------|----------------|-----------|
| timing | 72 | L3-Favorable | `pre_chasm` | HIGH |
| competition | 82 | L4-Strong | `nascent` | HIGH |
| market_size | 20 | L1-Speculative | `micro` | HIGH |
| customer_readiness | 78 | L4-Strong | `adopting` | HIGH |
| regulatory | 25 | L1-Speculative | `heavy` | LOW |
| infrastructure | 55 | L2-Neutral | `emerging` | MEDIUM |
| market_structure | 65 | L3-Favorable | `emerging` | HIGH |

**Analysis:** The nascent competition and strong customer_readiness are the brightest signals. Two dimensions drag the composite below investment grade: `market_size=micro` (actual 2010 on-category revenue was near-zero for a SF-only product) and `regulatory=heavy` (Gemini weighted the SFMTA cease-and-desist heavily). The micro market_size score of 20 is the single largest drag — retrospectively, this is the classic "market didn't exist yet" signal that VCs who got it right identified as a feature rather than a bug.

---

## Market 2 — Consumer & Enterprise Cloud File Storage Sync (2009)

**Composite: 62/100 [L3-Favorable]**

| Dimension | Score | Level | Verified Value | Agreement |
|-----------|-------|-------|----------------|-----------|
| timing | 52 | L2-Neutral | `early_chasm` | MEDIUM |
| competition | 72 | L3-Favorable | `fragmented` | MEDIUM |
| market_size | 25 | L1-Speculative | `small` | MEDIUM |
| customer_readiness | 72 | L3-Favorable | `adopting` | HIGH |
| regulatory | 82 | L4-Strong | `light_touch` | HIGH |
| infrastructure | 62 | L3-Favorable | `developing` | HIGH |
| market_structure | 70 | L3-Favorable | `forming` | HIGH |

**Analysis:** Regulatory and infrastructure are the standout positives. Market_size and timing both score low, reflecting that in 2009 the cloud storage category was still tiny and chasm-crossing was still underway. The four L3-Favorable dimensions form a coherent "early but viable" picture. The 62 composite correctly places this at the investment-grade floor — which aligns with the actual investment thesis: Sequoia's $7.2M Series A in Oct 2008 was a bet on a market that didn't fully exist yet.

---

## Market 3 — B2B SaaS Marketing Automation (2012)

**Composite: 62/100 [L3-Favorable]**

| Dimension | Score | Level | Verified Value | Agreement |
|-----------|-------|-------|----------------|-----------|
| timing | 78 | L4-Strong | `early_majority` | HIGH |
| competition | 45 | L2-Neutral | `consolidating` | HIGH |
| market_size | 25 | L1-Speculative | `small` | HIGH |
| customer_readiness | 76 | L4-Strong | `adopting` | HIGH |
| regulatory | 82 | L4-Strong | `light_touch` | HIGH |
| infrastructure | 55 | L2-Neutral | `developing` | MEDIUM |
| market_structure | 70 | L3-Favorable | `defined` | HIGH |

**Analysis:** The strongest timing and customer_readiness of the batch — this market had clearly crossed the chasm by 2012. Competition score of 45 correctly captures the M&A consolidation wave (Oracle/Eloqua, Salesforce/ExactTarget) that made new independent entry structurally harder. Market_size (25) is the primary drag — the total marketing automation category was ~$325M in 2012, scoring `small`. The combination of strong demand-side signals (timing=78, readiness=76) against supply-side headwinds (competition=45) is the characteristic late-entry tension.

---

## Anonymization Verification

The scorer received **no** market names, descriptions, base profiles, or Role-1/2 rationale. The prompt presented only:

```
DIMENSION                  VERIFIED CLASSIFICATION                   INTER-SOURCE AGREEMENT
------------------------   ------------------------------------      ----------------------
timing                     pre_chasm                                 HIGH
competition                nascent                                   HIGH
market_size                micro                                     HIGH
customer_readiness         adopting                                  HIGH
regulatory                 heavy                                     LOW
infrastructure             emerging                                  MEDIUM
market_structure           emerging                                  HIGH
```

---

## Observations & Next Steps

1. **Market_size is the dominant score depressant** across all 3 markets — `micro` and `small` both map to L1-Speculative. This is correct: all three were genuinely small at reference year. But it penalises pre-product-market-fit entries that were actually great investments. Consider adding a `market_size_trajectory` dimension (current × growth rate) in a future schema version.

2. **Regulatory agreement quality affects score confidence** — Market 1's regulatory score of 25 (`heavy`, LOW agreement) is the most uncertain score in the batch. Scores from LOW-agreement dimensions should be flagged as lower-confidence.

3. **Next run should use v2.0 schema data** — This run used `reference_population_v3.json` (schema v1.1, old enum values). Once a Run 4 is completed with the v3 canonical enums, Step 3 should be re-run to produce scores from the upgraded classification vocabulary (Rogers/Gartner/etc. values may shift some scores meaningfully).
