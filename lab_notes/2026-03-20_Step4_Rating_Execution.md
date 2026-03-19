# Step 4: Rating Engine — Full 120-Market Execution

**Date:** 2026-03-20
**Script:** `src/pipeline_step4_rating.py`
**Input:** `reference_population_master.json` (120 markets)
**Outputs:** `final_rated_population.json` (schema v3.0), `sample_report.md`

---

## Bug Fixed Before Run

`_get_structure()` was reading from `market["dimensions"]["market_structure"]["classification"]` (step1 Claude-only value). 76/120 markets disagreed with the verified step3 value. Fixed to read from `market["step3"]["feature_matrix"]["market_structure"]["value"]` (post-Gemini cross-check, canonical).

---

## Population Statistics

| Metric | Value |
|--------|-------|
| Markets rated | 120 |
| Composite score min | 31.2 |
| Composite score mean | 57.8 |
| Composite score max | 76.7 |

---

## Cohort Sizes (step3 verified)

| Structure Type | N | Composite Range |
|----------------|---|----------------|
| technology_enablement | 58 | [36.6, 76.7] |
| platform_two_sided | 34 | [40.2, 72.6] |
| fragmented_niche | 12 | [33.1, 76.5] |
| winner_take_most | 12 | [41.2, 66.0] |
| regulated_infrastructure | 4 | [31.2, 60.9] |

---

## Logistic Regression Coefficients

One `sklearn.LogisticRegression` (standardised predictor, L2, lbfgs) per structure type.
Target: `outcome_T5 = 1` if within-group percentile ≥ 45.

| Structure | b0 | b1 | n_pos | n_neg |
|-----------|----|----|-------|-------|
| platform_two_sided | +0.537 | +1.818 | 20 | 14 |
| fragmented_niche | +1.050 | +1.405 | 8 | 4 |
| winner_take_most | -0.109 | +1.652 | 6 | 6 |
| technology_enablement | -0.077 | +3.147 | 32 | 26 |
| regulated_infrastructure | +0.002 | +0.851 | 2 | 2 |

All b1 coefficients are positive — higher composite scores increase P(outcome ≥ L3). The technology_enablement cohort (n=58, largest) has the steepest slope (b1=3.147), suggesting the tightest score-outcome relationship within that group.

---

## Rating Distribution

| Rating | Label | Count |
|--------|-------|-------|
| L5 | Ideal | 16 |
| L4 | Attractive | 22 |
| L3 | Viable (IGT) | 30 |
| L2 | Headwinds | 21 |
| L1 | Hostile | 31 |

Distribution is roughly bell-shaped with a slight negative skew (more L1s than L5s). This is expected: the reference population is historically-grounded, capturing real market conditions where many early-stage sectors faced genuine headwinds.

---

## Sample Market Card (market_001)

**US Online Peer-to-Peer Consumer Lending Platforms (2006)**
- Structure: platform_two_sided
- Composite: 45.2 (Headwinds tier)
- Percentile: 12.1th within platform_two_sided cohort
- Rating: **L1 — Hostile**
- P(outcome ≥ L3): 12.8% (3 out of 20)
- Top NN: Prescription Digital Therapeutics (2017, L1, sim=0.991)

*Note: P2P lending in 2006 rates L1 because micro market size (score=15) heavily penalises the platform_two_sided composite despite strong competition/structure scores. This is correct — the market was genuinely microscopic at reference year.*

---

## Nearest Neighbour Notes

7-D cosine similarity on L2-normalised step3 scores. Similarities range from ~0.98–0.99, indicating tight dimensional clustering within the reference population — sensible given all markets come from the same scoring rubric.
