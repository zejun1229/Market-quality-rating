# 120-Market EDA Report — Vela MQR Reference Population

*Generated: 2026-03-20 06:59*  
*Source: `reference_population_master.json` + `final_rated_population.json` + `Scaling_Run_Log.md`*

---

## 1. Execution Analytics

| Metric | Value |
|--------|-------|
| Pipeline start (Batch 1) | 2026-03-19 07:19:00 |
| Pipeline end (final batch start) | 2026-03-20 03:01:05 |
| **Estimated total wall-clock time** | **22h 9m 50s** |
| Total batches | 9 |
| Target markets | 120 |
| Markets accepted | 120 |
| Blacklisted / timed-out | 5 |
| Deduped / skipped | 5 |
| Total candidates attempted | 130 |

> **Note:** Timestamps parsed from `Scaling_Run_Log.md` batch headers. The pipeline ran across multiple sessions; wall-clock time reflects the span from Batch 1 start to the final batch header plus one average-batch estimate.

### Batch-by-Batch Timeline

| Batch | Timestamp | Markets Accepted | Cumulative |
|-------|-----------|-----------------|------------|
| 1 | 2026-03-19 07:19 | 15 | 30 / 120 |

---

## 2. Inter-Source Agreement Levels

Agreement is computed per-dimension (7 dimensions × 120 markets = 840 dimension-pair evaluations).

### 2a. Overall Agreement per Market

| Agreement Level | Markets | Percentage |
|-----------------|---------|------------|
| HIGH | 47 | 39.2% |
| MEDIUM | 70 | 58.3% |
| LOW | 3 | 2.5% |
| **Total** | **120** | **100%** |

### 2b. Dimension-Level Agreement (all 840 evaluations)

| Agreement Level | Count | Percentage |
|-----------------|-------|------------|
| HIGH | 381 | 45.4% |
| MEDIUM | 298 | 35.5% |
| LOW | 161 | 19.2% |
| **Total** | **840** | **100%** |

### 2c. Agreement by Dimension

| Dimension | HIGH | MEDIUM | LOW | HIGH% |
|-----------|------|--------|-----|-------|
| Timing | 82 | 35 | 3 | 68% |
| Competition | 74 | 40 | 6 | 62% |
| Market Size | 34 | 48 | 38 | 28% |
| Customer Readiness | 52 | 31 | 37 | 43% |
| Regulatory | 56 | 55 | 9 | 47% |
| Infrastructure | 40 | 53 | 27 | 33% |
| Market Structure | 43 | 36 | 41 | 36% |

### 2d. Grounding Source Count (T+5 Outcome Verification)

| Metric | Value |
|--------|-------|
| Mean sources per market | 13.6 |
| Median sources per market | 13 |
| Min sources | 4 |
| Max sources | 26 |
| Markets with >= 10 sources | 93 |
| Markets with < 5 sources | 1 |

---

## 3. Score Distributions — The 7 Dimensions

Scores are 0–100 integers assigned by Role 3 (Scorer) based on verified classifications.

### 3a. Summary Statistics

| Dimension | Mean | Median | Std Dev | Min | P25 | P75 | Max |
|-----------|------|--------|---------|-----|-----|-----|-----|
| Timing | 65.8 | 72 | 11.2 | 22 | 62 | 72 | 78 |
| Competition | 56.7 | 62 | 10.2 | 35 | 54 | 62 | 85 |
| Market Size | 43.7 | 52 | 21.9 | 15 | 22 | 65 | 85 |
| Customer Readiness | 53.6 | 52 | 15.3 | 25 | 41 | 68 | 78 |
| Regulatory | 65.8 | 71 | 18.5 | 22 | 63 | 79 | 85 |
| Infrastructure | 47.7 | 52 | 10.6 | 20 | 45 | 52 | 65 |
| Market Structure | 64.1 | 65 | 11.2 | 30 | 55 | 72 | 82 |

**Population mean score (across all dimensions):** 56.8  
**Population std (market-level means):** 8.5

### 3b. Dimension Score Profiles — Notable Findings

- **Highest mean score:** Regulatory (65.8) — markets in the reference population tend to have favourable regulatory conditions.
- **Lowest mean score:** Market Size (43.7) — the most challenging structural dimension on average.
- **Highest variance:** Market Size (σ = 21.9) — widest spread across markets.
- **Lowest variance:** Competition (σ = 10.2) — most consistent across markets.

### 3c. Inter-Dimension Correlation Matrix

Values close to +1 indicate dimensions that tend to move together; negative values indicate inverse relationships.

| | Timing | Competit | Market S | Customer | Regulato | Infrastr | Market S |
|---|---|---|---|---|---|---|---|
| **Timing** | 1.00 | -0.10 | 0.44 | 0.53 | 0.21 | 0.63 | 0.25 |
| **Competit** | -0.10 | 1.00 | -0.06 | -0.08 | 0.14 | -0.23 | 0.09 |
| **Market S** | 0.44 | -0.06 | 1.00 | 0.54 | 0.08 | 0.51 | 0.15 |
| **Customer** | 0.53 | -0.08 | 0.54 | 1.00 | 0.18 | 0.46 | 0.27 |
| **Regulato** | 0.21 | 0.14 | 0.08 | 0.18 | 1.00 | 0.20 | 0.31 |
| **Infrastr** | 0.63 | -0.23 | 0.51 | 0.46 | 0.20 | 1.00 | 0.18 |
| **Market S** | 0.25 | 0.09 | 0.15 | 0.27 | 0.31 | 0.18 | 1.00 |

---

## 4. Categorical & Temporal Breakdowns

### 4a. Rating Distribution (L1–L5)

Ratings are assigned by Step 4 from within-cohort composite percentile bands.

| Rating | Label | Count | % of Population | Bar |
|--------|-------|-------|-----------------|-----|
| **L5** | Ideal | 16 | 13.3% | `################` |
| **L4** | Attractive | 22 | 18.3% | `######################` |
| **L3** | Viable (IGT) | 30 | 25.0% | `##############################` |
| **L2** | Headwinds | 21 | 17.5% | `#####################` |
| **L1** | Hostile | 31 | 25.8% | `###############################` |

### 4b. Market Structure Distribution

| Structure Type | Count | % | Avg Composite | Rating Breakdown (L1→L5) |
|----------------|-------|---|--------------|--------------------------|
| Technology Enablement | 58 | 48.3% | 60.6 | 15 / 11 / 14 / 12 / 6 |
| Platform / Two-Sided | 34 | 28.3% | 58.3 | 8 / 6 / 9 / 6 / 5 |
| Fragmented Niche | 12 | 10.0% | 48.7 | 3 / 1 / 4 / 2 / 2 |
| Winner-Take-Most | 12 | 10.0% | 55.6 | 4 / 2 / 2 / 2 / 2 |
| Regulated Infrastructure | 4 | 3.3% | 45.8 | 1 / 1 / 1 / 0 / 1 |

### 4c. Reference Year Distribution

| Year | Count | Bar |
|------|-------|-----|
| 2005 | 1 | `#` |
| 2006 | 2 | `##` |
| 2008 | 2 | `##` |
| 2009 | 1 | `#` |
| 2010 | 6 | `######` |
| 2011 | 2 | `##` |
| 2012 | 7 | `#######` |
| 2013 | 6 | `######` |
| 2014 | 11 | `###########` |
| 2015 | 17 | `#################` |
| 2016 | 17 | `#################` |
| 2017 | 19 | `###################` |
| 2018 | 13 | `#############` |
| 2019 | 9 | `#########` |
| 2020 | 7 | `#######` |

**Year range covered:**
- Earliest: 2005
- Latest: 2020
- Mode: 2017 (19 markets)

### 4d. Structure × Rating Heatmap

Row = Market Structure | Column = L1–L5 rating count

| Structure | L1 | L2 | L3 | L4 | L5 | Total |
|-----------|----|----|----|----|-----|-------|
| Technology Enablement | 15 | 11 | 14 | 12 | 6 | 58 |
| Platform / Two-Sided | 8 | 6 | 9 | 6 | 5 | 34 |
| Fragmented Niche | 3 | 1 | 4 | 2 | 2 | 12 |
| Winner-Take-Most | 4 | 2 | 2 | 2 | 2 | 12 |
| Regulated Infrastructure | 1 | 1 | 1 | 0 | 1 | 4 |

### 4e. Composite Score Distribution by Structure

| Structure | N | Min | Mean | Median | Max | Std |
|-----------|---|-----|------|--------|-----|-----|
| Technology Enablement | 58 | 36.6 | 60.6 | 62.5 | 76.7 | 9.1 |
| Platform / Two-Sided | 34 | 40.2 | 58.3 | 57.9 | 72.6 | 9.8 |
| Fragmented Niche | 12 | 33.1 | 48.7 | 44.0 | 76.5 | 15.0 |
| Winner-Take-Most | 12 | 41.2 | 55.6 | 56.7 | 66.0 | 8.9 |
| Regulated Infrastructure | 4 | 31.2 | 45.8 | 45.6 | 60.9 | 12.3 |

---

## 5. Key Takeaways

1. **Dual-model agreement is solid but imperfect.** 45% of dimension evaluations reached HIGH agreement between Claude and Gemini; 19% were LOW. The `market_size` and `customer_readiness` dimensions showed the most disagreement — consistent with these being the hardest to pin historically.

2. **Technology Enablement dominates the population** (58 of 120 markets, 48%). This reflects the 2009–2021 window skewing toward software/SaaS infrastructure markets.

3. **Market Size is the lowest-scoring dimension** (mean = 43.7) with the widest spread (σ = 21.9). Early-stage venture markets in the reference population were predominantly micro-to-small at their reference year — the methodology correctly captures nascent market conditions.

4. **Rating distribution is slightly bottom-heavy** (L1=31, L5=16), which is expected: the reference population captures real markets including those that failed to sustain category independence.

5. **Reference year coverage is broadest in 2015–2020**, with 2017 as the mode (19 markets). Pre-2010 markets are under-represented (harder to find verifiable T+5 grounding sources).

6. **Grounding quality is high**: markets average 13.6 citable sources per T+5 verification, with 93 of 120 markets achieving 10 or more grounded sources.

---

*Report generated by `src/eda_120_markets.py` on 2026-03-20.*