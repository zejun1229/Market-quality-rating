```
════════════════════════════════════════════════════════════════════════
  VELA MARKET QUALITY RATING  ·  ANALYST REPORT
════════════════════════════════════════════════════════════════════════


────────────────────────────────────────────────────────────────────────
  0. GENERAL QUALITY RATING
────────────────────────────────────────────────────────────────────────

  MARKET      On-Demand Home Services and Home Repair Marketplace
  DATE        2026-03-20
  REF YEAR    2013  (T+5 horizon: 2018)
  STRUCTURE   PLATFORM TWO SIDED

  RATING      L3 — VIABLE  [ ]
  SCORE       59.2 / 100
              [█████████████████░░░░░░░░░░░░░]
  PERCENTILE  59th within cohort  (Top 41% of 34 rated markets)

  PROBABILITY STATEMENT
  About 13 out of 20 markets with this structure and score profile
  achieved scale to L3 or above at T+5.  P(outcome ≥ L3) = 66.7%
  [b₀=+0.537  b₁=+1.818  z=+0.087  log-odds=+0.695]

────────────────────────────────────────────────────────────────────────
  1. DIMENSION SCORES
────────────────────────────────────────────────────────────────────────

  Dimension              Score  L-Rating  Confidence
  ────────────────────── ─────  ────────  ──────────
  Timing                    68        L3  HIGH    
  Competition               52        L2  HIGH    
  Market Size               45        L2  MEDIUM [!]
  Customer Readiness        62        L3  HIGH    
  Regulatory                42        L2  MEDIUM [!]
  Infrastructure            58        L3  HIGH    
  Market Structure          74        L3  HIGH    

  [!] = Confidence flag — see Section 5 for required analyst action.
  L-Rating key:  L5 ≥90  L4 ≥75  L3 ≥55  L2 ≥35  L1 <35

────────────────────────────────────────────────────────────────────────
  2. WHY THESE WEIGHTS WERE APPLIED
────────────────────────────────────────────────────────────────────────

  Structure type:  PLATFORM / TWO-SIDED
  Weights:  Timing 55%  ·  Competition 22%  ·  Market Size 23%

  Rationale:
  In two-sided marketplace models, Timing is the dominant causal driver
  because liquidity — the minimum viable density of supply and demand on both
  sides — is only achievable within a narrow adoption window. Entering too
  early (pre-smartphone ubiquity) or too late (post-Handy consolidation)
  produces structurally different outcomes. Market Size carries 23% weight
  because home services is a large but highly fragmented TAM; platforms that
  correctly sized their initial wedge (single city, single service category)
  outperformed those that over-expanded. Competition receives 22% weight:
  oligopoly dynamics favour the early-mover, but the winner-take-most outcome
  in this market was not inevitable — trust and supply quality were the actual
  differentiators, not raw network effects.

  Composite calculation:
  59.19 = (0.55 × 68) + (0.22 × 52) + (0.23 × 45)
       = 37.40 + 11.44 + 10.35

────────────────────────────────────────────────────────────────────────
  3. COMPARABLE MARKETS
────────────────────────────────────────────────────────────────────────

  [1] Online Student Tutor Matching Digital Marketplaces
      Year 2014  ·  Similarity 99.0%  ·  Rating at entry  L4
      Actual outcome  Mixed — category consolidated around 2 survivors
      Key lesson  Supply-quality enforcement and background-check investment were the decisive moats — platforms that commoditised labour lost to those that curated it.

  [2] Online Direct-to-Consumer Used Vehicle Retail Platfo
      Year 2017  ·  Similarity 98.8%  ·  Rating at entry  L4
      Actual outcome  Positive — acquired at 4x entry valuation
      Key lesson  Hyper-local density was a prerequisite for unit economics; premature geographic expansion destroyed gross margin before the model was proven.

  [3] Digital KYC and AML Identity Verification Software
      Year 2019  ·  Similarity 98.8%  ·  Rating at entry  L4
      Actual outcome  Negative — wound down; gig-worker regulation triggered unit-economics collapse
      Key lesson  Trust-and-safety UX (ratings, insurance, escrow) converted one-time users into repeat customers and was the primary driver of NPS and retention.


────────────────────────────────────────────────────────────────────────
  4. WHAT WOULD CHANGE THIS RATING
────────────────────────────────────────────────────────────────────────

  UPGRADE TRIGGERS  (→ L4)
  ─────────────────────────────────────────────────────────────────
  U1  Market Size re-scored from 'small' to 'medium' (score ≥ 65):
      new composite ≈ 63.8 → likely crosses 70th pct threshold.
  U2  Regulatory re-classified to 'light_touch' (score ≥ 70):
      removes the primary downside risk to unit economics and bumps
      the Customer Readiness signal confidence to HIGH.
  U3  Evidence of dominant platform moat (supply lock-in, proprietary
      background-check infrastructure) → Competition score to 68+.

  DOWNGRADE TRIGGERS  (→ L2)
  ─────────────────────────────────────────────────────────────────
  D1  Regulatory confirmed as 'moderate_compliance' (Gemini signal):
      worker-classification laws (AB5-type) cap gross margin and force
      Customer Readiness re-score to ≤38 (demand cools) → composite
      ≈ 59.2, driving pct below 45th → L2.
  D2  Market Size confirmed 'micro' (Gemini signal): TAM below $500M
      at reference year → composite ≈ 53.9, pct falls to L2 band.
  D3  Timing re-classified to 'innovators': market too early for
      smartphone density required for on-demand liquidity → Timing
      score ≤40, composite ≈ 43.8.

────────────────────────────────────────────────────────────────────────
  5. ANALYST FLAGS — ACTIONS REQUIRED
────────────────────────────────────────────────────────────────────────

  FLAG 1  ·  PRIORITY MEDIUM
  Dimension   Market Size
  Confidence  MEDIUM  ·  Sources retrieved: 7 *** BELOW SOURCE THRESHOLD ***
  Disagreement
    Claude  →  small
    Gemini  →  micro
  Action required
    Pull primary market sizing reports (IBISWorld / Gartner 2013 home
    services). Determine whether $500M–$1B TAM figure references the
    addressable online-booked segment only or total home services spend.
    Re-score and re-run composite if 'small' is confirmed vs 'micro'.

  FLAG 2  ·  PRIORITY MEDIUM
  Dimension   Regulatory
  Confidence  MEDIUM  ·  Sources retrieved: 6 *** BELOW SOURCE THRESHOLD ***
  Disagreement
    Claude  →  light_touch
    Gemini  →  moderate_compliance
  Action required
    Review gig-economy labour law timeline: California AB5 (2019),
    Dynamex (2018), and analogous state-level developments AS OF 2013.
    If worker-classification risk was already legally material in 2013,
    upgrade Regulatory to 'moderate_compliance' and re-run LR model.


────────────────────────────────────────────────────────────────────────
  6. CONFIDENCE SUMMARY
────────────────────────────────────────────────────────────────────────

  Overall confidence       MEDIUM
  High-confidence dims     5 of 7
  Flagged dims             2  (Market Size, Regulatory)
  Total sources retrieved  73
  Below source threshold   2 dim(s): Market Size, Regulatory
                           (threshold = 8 sources)

  Rating confidence:  The L3 rating is PROVISIONAL pending resolution
  of the Market Size and Regulatory flags in Section 5. Resolution in
  favour of the Gemini signals (micro + moderate_compliance) would
  reduce the rating to L2. Resolution in favour of Claude signals
  (small + light_touch) would support an upgrade to L4.

════════════════════════════════════════════════════════════════════════
  END OF REPORT  ·  On-Demand Home Services and Home Repair Marketplac
════════════════════════════════════════════════════════════════════════
```
