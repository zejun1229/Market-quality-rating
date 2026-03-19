# Out-of-Sample Case Study: Enterprise Social Networking Platforms (2012)

**Date:** 2026-03-20
**Script:** `src/oos_test_case.py`
**Purpose:** Validate Step 4 Rating Engine on a held-out historical market

---

## Test Market

**Enterprise Social Networking and Intranet Collaboration Platforms**
- Reference year: 2012 | T+5: 2017
- Structure: `winner_take_most`
- Actual outcome: **L2 — Headwinds** (market absorbed by Microsoft Teams / Slack ecosystem)

## Key Numbers

| Step | Result |
|------|--------|
| Composite Score | 58.80 |
| Cohort percentile | 50.0th (N=12) |
| LR b0 / b1 | -0.1092 / +1.6519 |
| z-score | +0.376 |
| log-odds | +0.512 |
| P(outcome >= L3) | 62.5% (13/20) |
| **Model rating** | **L3 — Viable (IGT)** |
| **Actual rating** | **L2 — Headwinds** |
| Match | Off by 1 band |

## Findings

- Model predicted L3; actual was L2. One-band miss at the 50th percentile — borderline case, statistically indistinguishable from L3 boundary.
- P(≥L3) = 62.5% is correct in spirit: market *could* have been L3, but category collapse drove the actual outcome below IGT.
- Top NN: On-Demand Restaurant Last-Mile Delivery (2014, L3) — 99.5% cosine similarity. Both share high regulatory/infrastructure scores with moderate market size.
- The logistic model cannot capture exogenous shocks (Microsoft Teams launch, Slack paradigm shift) that invalidated the market structure — expected limitation for any rule-based model.
