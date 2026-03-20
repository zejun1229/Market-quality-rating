# 10-Market Out-of-Sample Validation Report

*Generated: 2026-03-20*  
*Reference population: 120 markets | Test set: 10 markets | Ref year: 2013 | T+5: 2018*

## Methodology

- All 10 markets verified as **not present** in `reference_population_master.json`
- Dimension scores assigned by Role 1 (Historian) using historical conditions as of 2013
- Predicted rating computed by Step 4 engine: composite score → within-cohort percentile
  → logistic regression P(outcome ≥ L3) → L1–L5 band assignment
- Actual rating assigned using quantifiable T+5 (2018) framework:
  - **L5**: IPO or >$10B valuation
  - **L4**: $1B–$10B valuation or M&A
  - **L3**: $100M–$999M valuation / Series C–D scale
  - **L2**: <$100M valuation, acqui-hire, or stagnation
  - **L1**: category collapse or major bankruptcies

---

## Results Table

| # | Market | Structure | Composite | Pct | P(≥L3) | Predicted | Actual | Match |
|---|--------|-----------|-----------|-----|--------|-----------|--------|-------|
| 1 | 3D Printing Desktop Hardware and Prosumer Fab | FRN | 43.0 | 41th | 62% | **L2** | **L2** | EXACT |
| 2 | Consumer Recreational Quadcopter and FPV Dron | WTM | 47.3 | 23th | 15% | **L1** | **L5** | MISS |
| 3 | Customer Success Management and Churn Prevent | TEN | 53.0 | 15th | 6% | **L1** | **L3** | MISS |
| 4 | Mobile In-App Performance Advertising and Use | TEN | 65.0 | 55th | 81% | **L3** | **L4** | OFF-1 |
| 5 | Employee Engagement and Continuous Performanc | FRN | 45.5 | 68th | 68% | **L3** | **L3** | EXACT |
| 6 | B2B Sales Intelligence and Account Data Enric | TEN | 60.7 | 46th | 49% | **L3** | **L4** | OFF-1 |
| 7 | Social Media Management and Community Analyti | TEN | 62.1 | 48th | 61% | **L3** | **L3** | EXACT |
| 8 | On-Demand Home Fitness Streaming and Connecte | WTM | 46.2 | 23th | 12% | **L1** | **L4** | MISS |
| 9 | Programmatic Digital Out-of-Home Advertising  | FRN | 36.5 | 32th | 46% | **L2** | **L2** | EXACT |
| 10 | B2B Invoice Financing and Accounts Receivable | TEN | 57.3 | 29th | 23% | **L2** | **L3** | OFF-1 |

---

## Accuracy Summary

| Metric | Value |
|--------|-------|
| Total markets | 10 |
| Exact match | **4 / 10  (40%)** |
| Within ±1 band | **7 / 10  (70%)** |
| Upgrade errors (predicted > actual) | 0 |
| Downgrade errors (predicted < actual) | 6 |
| Miss (off by ≥2 bands) | 3 |

---

## Per-Market Detail

### 1. 3D Printing Desktop Hardware and Prosumer Fabrication

| Field | Value |
|-------|-------|
| Structure type | Fragmented Niche |
| Composite score | 43.0 / 100 |
| Cohort percentile | 41th |
| P(outcome ≥ L3) | 62.0%  (12/20) |
| **Predicted rating** | **L2 — Headwinds** |
| **Actual rating (T+5 2018)** | **L2 — Headwinds** |
| Verdict | **EXACT MATCH** |
| Top-1 NN | Digital Real Estate Transaction and iBuying Platforms (99.7% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 62 | 25% |
| Competition | 55 | 15% |
| Market Size | 32 | 60% |
| Customer Readiness | 38 | 0% |
| Regulatory | 72 | 0% |
| Infrastructure | 42 | 0% |
| Market Structure | 48 | 0% |

**Actual outcome:**  MakerBot was acquired by Stratasys ($403M, 2013) but by 2018 had closed all retail stores and laid off ~50% of staff; consumer 3D printing failed to cross the chasm — Formlabs survived as a professional niche, leaving the broader consumer category as Headwinds.

---

### 2. Consumer Recreational Quadcopter and FPV Drone Hardware

| Field | Value |
|-------|-------|
| Structure type | Winner Take Most |
| Composite score | 47.3 / 100 |
| Cohort percentile | 23th |
| P(outcome ≥ L3) | 15.1%  (3/20) |
| **Predicted rating** | **L1 — Hostile** |
| **Actual rating (T+5 2018)** | **L5 — Ideal** |
| Verdict | **MISS (off by 4)** |
| Top-1 NN | Digital KYC and AML Identity Verification Software (98.8% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 62 | 32% |
| Competition | 42 | 52% |
| Market Size | 35 | 16% |
| Customer Readiness | 55 | 0% |
| Regulatory | 32 | 0% |
| Infrastructure | 45 | 0% |
| Market Structure | 58 | 0% |

**Actual outcome:**  DJI reached a $15B valuation in October 2018 after a $100M funding round, capturing ~70% of the global consumer drone market; the category generated >$3B in annual revenue by 2018, qualifying as a clear Ideal outcome.

---

### 3. Customer Success Management and Churn Prevention SaaS

| Field | Value |
|-------|-------|
| Structure type | Technology Enablement |
| Composite score | 53.0 / 100 |
| Cohort percentile | 15th |
| P(outcome ≥ L3) | 6.1%  (1/20) |
| **Predicted rating** | **L1 — Hostile** |
| **Actual rating (T+5 2018)** | **L3 — Viable** |
| Verdict | **MISS (off by 2)** |
| Top-1 NN | Mobile Guided Meditation and Mindfulness Apps (99.4% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 62 | 60% |
| Competition | 45 | 18% |
| Market Size | 35 | 22% |
| Customer Readiness | 58 | 0% |
| Regulatory | 72 | 0% |
| Infrastructure | 62 | 0% |
| Market Structure | 68 | 0% |

**Actual outcome:**  Gainsight raised a $52M Series D (2015) and reached ~$700M valuation by 2018; the CS software market grew from niche to mainstream but no single platform crossed $1B by the 2018 measurement date — Viable (Series C/D scale).

---

### 4. Mobile In-App Performance Advertising and User Acquisition Networks

| Field | Value |
|-------|-------|
| Structure type | Technology Enablement |
| Composite score | 65.0 / 100 |
| Cohort percentile | 55th |
| P(outcome ≥ L3) | 81.2%  (16/20) |
| **Predicted rating** | **L3 — Viable** |
| **Actual rating (T+5 2018)** | **L4 — Favorable** |
| Verdict | **OFF BY 1 (under-predicted)** |
| Top-1 NN | AI Machine Learning Fraud Prevention and Identity Verif (99.4% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 68 | 60% |
| Competition | 55 | 18% |
| Market Size | 65 | 22% |
| Customer Readiness | 65 | 0% |
| Regulatory | 55 | 0% |
| Infrastructure | 62 | 0% |
| Market Structure | 55 | 0% |

**Actual outcome:**  AppLovin was internally valued at ~$2B by 2018; IronSource reached a $1.1B valuation by 2019; the category produced multiple unicorns through programmatic mobile ad infrastructure — Favorable ($1B-$10B).

---

### 5. Employee Engagement and Continuous Performance Management SaaS

| Field | Value |
|-------|-------|
| Structure type | Fragmented Niche |
| Composite score | 45.5 / 100 |
| Cohort percentile | 68th |
| P(outcome ≥ L3) | 67.7%  (14/20) |
| **Predicted rating** | **L3 — Viable** |
| **Actual rating (T+5 2018)** | **L3 — Viable** |
| Verdict | **EXACT MATCH** |
| Top-1 NN | Real-Time Supply Chain Freight Visibility Software (99.7% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 62 | 25% |
| Competition | 48 | 15% |
| Market Size | 38 | 60% |
| Customer Readiness | 55 | 0% |
| Regulatory | 70 | 0% |
| Infrastructure | 58 | 0% |
| Market Structure | 48 | 0% |

**Actual outcome:**  Culture Amp raised a $40M Series D in 2018 at ~$500M valuation; Lattice reached Series B in 2018 (~$150M valuation); the category remained fragmented with no dominant platform above $1B by 2018 — Viable.

---

### 6. B2B Sales Intelligence and Account Data Enrichment Platforms

| Field | Value |
|-------|-------|
| Structure type | Technology Enablement |
| Composite score | 60.7 / 100 |
| Cohort percentile | 46th |
| P(outcome ≥ L3) | 49.2%  (10/20) |
| **Predicted rating** | **L3 — Viable** |
| **Actual rating (T+5 2018)** | **L4 — Favorable** |
| Verdict | **OFF BY 1 (under-predicted)** |
| Top-1 NN | Commercial Small Satellite Constellation Deployment Ser (99.6% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 68 | 60% |
| Competition | 52 | 18% |
| Market Size | 48 | 22% |
| Customer Readiness | 65 | 0% |
| Regulatory | 62 | 0% |
| Infrastructure | 62 | 0% |
| Market Structure | 62 | 0% |

**Actual outcome:**  DiscoverOrg received a growth equity investment from Hg Capital in 2018 valuing it at ~$1.6B; subsequently merged with ZoomInfo (2019) and IPO'd at $14B+ (2020) — Favorable ($1B-$10B valuation achieved by 2018).

---

### 7. Social Media Management and Community Analytics SaaS

| Field | Value |
|-------|-------|
| Structure type | Technology Enablement |
| Composite score | 62.1 / 100 |
| Cohort percentile | 48th |
| P(outcome ≥ L3) | 61.4%  (12/20) |
| **Predicted rating** | **L3 — Viable** |
| **Actual rating (T+5 2018)** | **L3 — Viable** |
| Verdict | **EXACT MATCH** |
| Top-1 NN | Real-Time Supply Chain Freight Visibility Software (99.6% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 68 | 60% |
| Competition | 55 | 18% |
| Market Size | 52 | 22% |
| Customer Readiness | 68 | 0% |
| Regulatory | 68 | 0% |
| Infrastructure | 65 | 0% |
| Market Structure | 55 | 0% |

**Actual outcome:**  Sprout Social raised a $40M Series D (2016) at ~$400-500M valuation and IPO'd in December 2019; Hootsuite raised at ~$700M but struggled with churn — category produced Viable outcomes, no $1B+ valuation by 2018.

---

### 8. On-Demand Home Fitness Streaming and Connected Workout Platforms

| Field | Value |
|-------|-------|
| Structure type | Winner Take Most |
| Composite score | 46.2 / 100 |
| Cohort percentile | 23th |
| P(outcome ≥ L3) | 12.5%  (3/20) |
| **Predicted rating** | **L1 — Hostile** |
| **Actual rating (T+5 2018)** | **L4 — Favorable** |
| Verdict | **MISS (off by 3)** |
| Top-1 NN | Mobile Guided Meditation and Mindfulness Apps (99.7% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 55 | 32% |
| Competition | 42 | 52% |
| Market Size | 42 | 16% |
| Customer Readiness | 52 | 0% |
| Regulatory | 72 | 0% |
| Infrastructure | 48 | 0% |
| Market Structure | 62 | 0% |

**Actual outcome:**  Peloton raised a $550M Series F in August 2018 at a $4.15B valuation (pre-IPO in 2019); the connected-fitness category it created was clearly Favorable with a $1B-$10B outcome by the measurement date.

---

### 9. Programmatic Digital Out-of-Home Advertising Networks

| Field | Value |
|-------|-------|
| Structure type | Fragmented Niche |
| Composite score | 36.5 / 100 |
| Cohort percentile | 32th |
| P(outcome ≥ L3) | 46.5%  (9/20) |
| **Predicted rating** | **L2 — Headwinds** |
| **Actual rating (T+5 2018)** | **L2 — Headwinds** |
| Verdict | **EXACT MATCH** |
| Top-1 NN | Subscription Box E-Commerce Niche Consumer Goods (99.8% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 52 | 25% |
| Competition | 45 | 15% |
| Market Size | 28 | 60% |
| Customer Readiness | 42 | 0% |
| Regulatory | 58 | 0% |
| Infrastructure | 35 | 0% |
| Market Structure | 42 | 0% |

**Actual outcome:**  Vistar Media raised a $3M Series A (2015) and $27M Series B (2020); the programmatic DOOH category remained sub-$500M and highly fragmented through 2018 with no standout exit — Headwinds (<$100M valuations).

---

### 10. B2B Invoice Financing and Accounts Receivable Factoring Platforms

| Field | Value |
|-------|-------|
| Structure type | Technology Enablement |
| Composite score | 57.3 / 100 |
| Cohort percentile | 29th |
| P(outcome ≥ L3) | 22.6%  (5/20) |
| **Predicted rating** | **L2 — Headwinds** |
| **Actual rating (T+5 2018)** | **L3 — Viable** |
| Verdict | **OFF BY 1 (under-predicted)** |
| Top-1 NN | Food Waste Upcycling and Surplus Reduction Technologies (99.4% sim) |

**Dimension scores:**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Timing | 62 | 60% |
| Competition | 48 | 18% |
| Market Size | 52 | 22% |
| Customer Readiness | 58 | 0% |
| Regulatory | 42 | 0% |
| Infrastructure | 52 | 0% |
| Market Structure | 55 | 0% |

**Actual outcome:**  Fundbox raised a $100M Series C in 2018 at ~$500-700M valuation; BlueVine raised $60M Series E at ~$500M; neither crossed $1B by the 2018 measurement date, placing the category at Viable (Series C/D scale).

---

## Logistic Regression Coefficients (reference)

| Structure | b₀ | b₁ | Cohort N | Scaler μ | Scaler σ |
|-----------|----|----|----------|----------|----------|
| Technology Enablement | -0.0769 | +3.1474 | 58 | 60.59 | 9.02 |
| Platform Two Sided | +0.5367 | +1.8176 | 34 | 58.35 | 9.69 |
| Fragmented Niche | +1.0502 | +1.4053 | 12 | 48.68 | 14.35 |
| Winner Take Most | -0.1092 | +1.6519 | 12 | 55.60 | 8.50 |
| Regulated Infrastructure | +0.0071 | +0.8218 | 4 | 33.59 | 8.03 |

---

*Report generated by `src/validation_10_markets.py`*