# Mirror Validation Pipeline: Architecture & Design

*Date: 2026-03-20*
*Script: `src/pipeline_validation.py`*
*Status: READY TO RUN*

---

## Objective

Generate a 60-market validation set (6 batches × 10 markets) to rigorously compare
T=0 predictions from the Vela MQR Step 4 engine against actual T+5 business outcomes,
using a **Symmetrical Percentile-Based Labeling** method so both sides of the comparison
are on the same within-cohort scale.

---

## Architecture

### Phase 1 — T=0 Prediction

| Stage | Role | Tool | Notes |
|-------|------|------|-------|
| Seed generation | Role 0 | Gemini + Google Search | 10 seeds/batch, ref years 2010-2019 |
| Profile + scoring | Role 1 | Claude | 7 dim scores 0-100, market_structure |
| Predicted rating | Step 4 | Python (in-process) | Compare composite vs reference cohort |

**Seed constraint:** Entry years strictly between 2009 and 2020 (2010-2019 inclusive),
guaranteeing at least 5 years of historical outcome data at every T+5 measurement.

**Predicted rating logic:**
1. Claude generates 7 integer dimension scores based *only* on T=0 information
2. Composite score computed using same causal weights as Step 4 (`pipeline_step4_rating.py`)
3. New market's composite is compared against `final_rated_population.json`'s same-structure
   cohort (empirical CDF): `percentile = (# reference markets below) / n × 100`
4. L1-L5 assigned from standard percentile bands (L5≥90, L4 70-90, L3 45-70, L2 25-45, L1<25)

### Phase 2 — T+5 Ground Truth

Gemini + Google Search retrieves 4 numerical metrics per market with **20-second hard timeout**.
Timed-out markets are blacklisted and skipped (same pattern as scale pipeline Role 2).

| Metric | Description | Normalization anchor |
|--------|-------------|----------------------|
| `peak_exit_value` | Highest single exit/valuation by T+5 (USD) | log10 scale: $1M=0, $10B=100 |
| `top_3_aggregate_valuation` | Combined top-3 player valuations (USD) | log10 scale: $1M=0, $30B=100 |
| `unicorn_count` | Companies reaching $1B+ by T+5 | linear: 0=0, 5+=100 |
| `capital_efficiency_ratio` | Average VFR of category leaders | linear: 0=0, 20x=100 |

### Phase 3 — Symmetrical Scoring & Labeling

**Actual performance score** (0-100) weighted combination:

```
actual_performance_score = 0.40 × norm(peak_exit_value)
                         + 0.30 × norm(top_3_aggregate_valuation)
                         + 0.20 × norm(unicorn_count)
                         + 0.10 × norm(capital_efficiency_ratio)
```

**Dynamic actual_rating** re-assigned after every batch:
- All 60 markets ranked by `actual_performance_score`
- Percentile bands (symmetrical, user-specified):

| Band | Label | Description |
|------|-------|-------------|
| Top 10% (≥90th) | L5 | Ideal |
| 70th-90th | L4 | Attractive |
| 45th-70th | L3 | Viable |
| 20th-45th | L2 | Headwinds |
| Bottom 20% (<20th) | L1 | Hostile |

### Phase 4 — Accuracy

- Delta: `predicted_rating` (Step 4 at T=0) vs `actual_rating` (T+5 percentile)
- Exact Match % and Off-by-1 % displayed in real-time Rich live panel
- Final summary printed to console and appended to LOGBOOK_MASTER.md

---

## Data Flow

```
reference_population_master.json  ->  dedup registry (seed exclusions)
final_rated_population.json       ->  reference cohort map (predicted rating)
                                        |
                                   [6 batches × 10 markets]
                                        |
                                   validation_population.json  (progressive)
                                        |
                                   Validation_Run_Log.md  (per-batch log)
```

---

## Fault Tolerance

- **Role 0 timeout (60s):** Retry up to 2× per batch before skipping
- **Phase 2 timeout (20s):** Blacklist market immediately; do not save
- **Deduplication:** Python-level SequenceMatcher (threshold=0.75) against both
  reference population and accumulated validation set
- **Incremental save:** `validation_population.json` updated after each market completes
- **Resume support:** Script reads existing `validation_population.json` on startup and
  resumes from where it left off

---

## Key Design Differences from Scale Pipeline

| Aspect | Scale Pipeline | Mirror Validation |
|--------|----------------|-------------------|
| Target | 120 reference markets | 60 validation markets |
| Role 2 | Gemini dimensional verification | Phase 2 T+5 ground-truth search |
| Rating basis | Within-population percentile | vs. reference cohort (T=0) |
| Actual label | T+5 outcome (historical) | Symmetrical percentile on perf score |
| Timeout | 90s (Role 2) | 20s (Phase 2), 60s (seeds) |
| Batch size | 15 | 10 |

---

*Lab note written by Claude Sonnet 4.6*
