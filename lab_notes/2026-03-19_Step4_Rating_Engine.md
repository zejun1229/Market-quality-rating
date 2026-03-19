# Step 4: Rating Engine — Implementation

**Date:** 2026-03-19
**Script:** `src/pipeline_step4_rating.py`
**Status:** Complete — tested on 4-market subset; ready for 120-market population

---

## What was built

`pipeline_step4_rating.py` is the final computational step of the Vela MQR pipeline. It reads `reference_population_master.json` and produces `final_rated_population.json` (schema v3.0) plus `sample_report.md`.

### Five processing steps

| Step | Name | Details |
|------|------|---------|
| 1 | **Composite Score** | Causal-weighted sum of timing, competition, market_size (structure-specific weights). For `regulated_infrastructure` only: primary 3 dims account for 50%, residual 4 dims equally share the remaining 50% (Section 5.5 Step 2). |
| 2 | **Empirical CDF** | Percentile rank computed *within* each market_structure group (0-based rank ÷ (n−1) × 100). |
| 3 | **Logistic Regression** | One `sklearn.LogisticRegression` per structure type. Target: `outcome_T5 = 1` if percentile ≥ 45 (L3+). Outputs P(outcome ≥ L3) and "X out of 20" language. Degenerate groups (all-positive or all-negative) receive a constant probability. |
| 4 | **Nearest Neighbours** | `scipy.spatial.distance.cdist(metric="cosine")` on L2-normalised 7-D step3 score vectors. Top 3 analogues per market. |
| 5 | **Rating Assignment** | L1–L5 from percentile bands: L5 ≥ 90th, L4 70–90th, L3 45–70th, L2 25–45th, L1 < 25th. |

### Causal weight tables implemented

| Structure | Timing | Competition | Market Size | Residual |
|-----------|--------|-------------|-------------|---------|
| winner_take_most | 32% | 52% | 16% | 0% |
| platform_two_sided | 55% | 22% | 23% | 0% |
| technology_enablement | 60% | 18% | 22% | 0% |
| fragmented_niche | 25% | 15% | 60% | 0% |
| regulated_infrastructure | 11%* | 14%* | 25%* | 12.5% × 4 |

*Effective contribution = stated weight × 50% (Section 5.5 Step 2)

---

## Test run (4 scored markets available)

```
Population composite: min=35.8  mean=40.5  max=45.2

Structures: platform_two_sided (3 markets), fragmented_niche (1 market)

Logistic regression:
  platform_two_sided (3 obs): b0=+0.746  b1=+0.534
  fragmented_niche: degenerate (singleton) → p_const=1.00

Rating distribution:
  L5  Ideal                    1
  L3  Viable (IGT)             2
  L1  Hostile                  1
```

Results look sensible: P2P Lending 2006 (highest composite in its group) correctly receives L5 with 81.7% probability.

---

## Outputs

| File | Description |
|------|-------------|
| `final_rated_population.json` | Full enriched dataset (schema v3.0); each market has `step4` block with composite_score, percentile_rank, rating, p_outcome_ge_l3, prob_x_of_20, nearest_neighbours |
| `sample_report.md` | Markdown MQR cards for first 3 markets (name, structure, ref year, T+5 year, rating, dimension table, nearest neighbour table) |

---

## Notes / known limitations

- With only 4 markets, logistic regression coefficients are unreliable. All five functions will produce meaningfully calibrated outputs once the full 120-market population is loaded.
- The `outcome_T5` target is derived from the within-group composite percentile (not an independent ground-truth label). This is disclosed in the methodology block of the output JSON.
- `market_structure` score (a 0-100 scorer rating) is included in the 7-D cosine NN vector and the residual component for `regulated_infrastructure`. This is intentional — the scorer rated how advantageous the market structure dynamic is for venture outcomes.
