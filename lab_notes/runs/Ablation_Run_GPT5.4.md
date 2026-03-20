# Vela MQR — Ablation Study Report  (GPT-5.4 Judge)

**Test market:** US App-Based On-Demand Black Car Transportation (2010)  
**Run:** `2026-03-19 07:15:07`  
**Paths:**
- **Path 1 — Baseline:** no judge; conflicts kept FLAGGED
- **Path 2 — Parametric:** GPT-5.4, internal knowledge only; may resolve or flag
- **Path 3 — Search:** GPT-5.4 + live web search; may resolve with citations or flag

---

## US App-Based On-Demand Black Car Transportation (2010)

### Role 2 Verification — Agreement & Source URLs

| Dimension | Claude | Gemini | Agreement | Verification URL |
|-----------|--------|--------|:---------:|-----------------|
| `timing` | `pre_chasm` | `pre_chasm` | HIGH | _no URL_ |
| `competition` | `nascent` | `nascent` | HIGH | _no URL_ |
| `market_size` | `micro` | `micro` | HIGH | [link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFeEUWcdfrSitIxfphbdm5p7uqDG6Q88qEVlK7xMNAJHEsgNYFVGSt8HvY80zON7oZjDlfStziIZ_wDd53sgWTthPpWrtBAVJx8Cb_qWaJaKuKMgtEgR1rlf6nfTy-zbaY_9N29p1OIsQ==) |
| `customer_readiness` | `adopting` | `adopting` | HIGH | _no URL_ |
| `regulatory` | `light_touch` | `heavy` | LOW | _no URL_ |
| `infrastructure` | `developing` | `emerging` | MEDIUM | _no URL_ |
| `market_structure` | `emerging` | `emerging` | HIGH | [link](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH5rbJ4eoS3BEN65T8l3nCh9EY6ZOO2y9ouG2__cY8auRXrk29gIV4ZAkj5frJo-dGplAJmbsfxQXCv0XHFoRXonKXnad-oRuSunSaopEyAsH-t-QHLdurXzfanhlJF3k0ysAhWpdU5xDyC2t2eGhH8KIh5o0oURLlOYY4=) |

---

### Path 1: Baseline
_No judge — conflicts kept FLAGGED_  
**Resolved:** 5  **Flagged:** 2

| Dimension | Final Classification | Status | Judge Rationale | Score |
|-----------|-------------------|:------:|-----------------|:-----:|
| `timing` | `pre_chasm` | ✓ resolved | — | 62 |
| `competition` | `nascent` | ✓ resolved | — | 78 |
| `market_size` | `micro` | ✓ resolved | — | 22 |
| `customer_readiness` | `adopting` | ✓ resolved | — | 72 |
| `regulatory` | `light_touch` / `heavy` | ⚑ FLAGGED | — | 48 |
| `infrastructure` | `developing` / `emerging` | ⚑ FLAGGED | — | 52 |
| `market_structure` | `emerging` | ✓ resolved | — | 68 |

**Mean score:** 57

### Path 2: Parametric
_GPT-5.4 parametric judge (no web search)_  
**Resolved:** 6  **Flagged:** 1

| Dimension | Final Classification | Status | Judge Rationale | Score |
|-----------|-------------------|:------:|-----------------|:-----:|
| `timing` | `pre_chasm` | ✓ resolved | Claude and Gemini agreed — no judge required. | 62 |
| `competition` | `nascent` | ✓ resolved | Claude and Gemini agreed — no judge required. | 78 |
| `market_size` | `micro` | ✓ resolved | Claude and Gemini agreed — no judge required. | 22 |
| `customer_readiness` | `adopting` | ✓ resolved | Claude and Gemini agreed — no judge required. | 72 |
| `regulatory` | `heavy` | ✓ resolved | By 2010 the post-crisis regulatory environment was clearly moving toward heavier oversight: in the U.S., the Dodd-Frank … | 28 |
| `infrastructure` | `developing` / `emerging` | ⚑ FLAGGED | For infrastructure in 2010, both labels can be defended because many economies commonly described as emerging had rapidl… | 55 |
| `market_structure` | `emerging` | ✓ resolved | Claude and Gemini agreed — no judge required. | 70 |

**Mean score:** 55

### Path 3: Search
_GPT-5.4 search judge (live web evidence)_  
**Resolved:** 7  **Flagged:** 0

| Dimension | Final Classification | Status | Judge Rationale | Search Source | Score |
|-----------|-------------------|:------:|-----------------|---------------|:-----:|
| `timing` | `pre_chasm` | ✓ resolved | Claude and Gemini agreed — no judge required. | _no URL_ | 62 |
| `competition` | `nascent` | ✓ resolved | Claude and Gemini agreed — no judge required. | _no URL_ | 82 |
| `market_size` | `micro` | ✓ resolved | Claude and Gemini agreed — no judge required. | _no URL_ | 18 |
| `customer_readiness` | `adopting` | ✓ resolved | Claude and Gemini agreed — no judge required. | _no URL_ | 72 |
| `regulatory` | `heavy` | ✓ resolved | Contemporaneous 2010 evidence shows the market faced active enforcement rather than permissive oversight: TechCrunch rep… | [link](https://techcrunch.com/2010/10/24/ubercab-ordered-to-cease-and-desist/) | 25 |
| `infrastructure` | `emerging` | ✓ resolved | Historical sources point to the US ride-hailing/TNC market’s infrastructure being nascent in 2010 rather than already de… | [link](https://rosap.ntl.bts.gov/view/dot/60060/dot_60060_DS1.pdf) | 52 |
| `market_structure` | `emerging` | ✓ resolved | Claude and Gemini agreed — no judge required. | _no URL_ | 58 |

**Mean score:** 53

---

### 3-Path Score Comparison

| Dimension | Baseline | Parametric | Search | Δ Param | Δ Search |
|-----------|:--------:|:----------:|:------:|:-------:|:--------:|
| `timing` | 62 | 62 | 62 | +0 | +0 |
| `competition` | 78 | 78 | 82 | +0 | +4 |
| `market_size` | 22 | 22 | 18 | +0 | -4 |
| `customer_readiness` | 72 | 72 | 72 | +0 | +0 |
| `regulatory` | 48 ⚑ | 28 | 25 | -20 | -23 |
| `infrastructure` | 52 ⚑ | 55 ⚑ | 52 | +3 | +0 |
| `market_structure` | 68 | 70 | 58 | +2 | -10 |
| **Mean** | **57** | **55** | **53** | **-2** | **-4** |

_⚑ = dimension was FLAGGED (unresolved conflict); score is conservative midpoint._

---

### GPT-5.4 Judge Logic — Resolved Dimensions

_Full rationale for every dimension the judge actively resolved or flagged._

#### GPT-5.4 Parametric

**`regulatory`** — ✓ resolved  
Original conflict: `light_touch` (A) vs `heavy` (B) — agreement=LOW  
Judge chose: **`heavy`** (Classifier B)  
Rationale: _By 2010 the post-crisis regulatory environment was clearly moving toward heavier oversight: in the U.S., the Dodd-Frank Act was enacted in July 2010, expanding systemic supervision, derivatives regulation, consumer protection, and resolution authority. Internationally, Basel III was agreed in 2010, raising bank capital and liquidity standards, so characterizing 2010 as light-touch is less historically accurate than heavy on the regulatory dimension._

**`infrastructure`** — ⚑ FLAGGED  
Original conflict: `developing` (A) vs `emerging` (B) — agreement=MEDIUM  
Judge: **could not resolve**  
Rationale: _For infrastructure in 2010, both labels can be defended because many economies commonly described as emerging had rapidly expanding transport, power, telecom, and urban systems, yet still faced major deficits in reliability, coverage, and quality compared with advanced markets. Historically, "emerging" often denoted faster modernization and investment momentum, while "developing" emphasized the still-incomplete basic infrastructure base; without a specific country or market, the record does not clearly favor one term over the other._

#### GPT-5.4 Search

**`regulatory`** — ✓ resolved  
Original conflict: `light_touch` (A) vs `heavy` (B) — agreement=LOW  
Judge chose: **`heavy`** (Classifier A)  
Rationale: _Contemporaneous 2010 evidence shows the market faced active enforcement rather than permissive oversight: TechCrunch reported on October 24, 2010 that the San Francisco Municipal Transportation Agency and the California Public Utilities Commission ordered UberCab to cease and desist operations in San Francisco. A later CPUC document also states that in October 2010 CPSD issued UberCab a cease-and-desist letter instructing it to stop advertising and operating as a passenger carrier for hire without Commission authorization, which supports a heavy regulatory classification for 2010._
Sources: [link](https://techcrunch.com/2010/10/24/ubercab-ordered-to-cease-and-desist/)

**`infrastructure`** — ✓ resolved  
Original conflict: `developing` (A) vs `emerging` (B) — agreement=MEDIUM  
Judge chose: **`emerging`** (Classifier B)  
Rationale: _Historical sources point to the US ride-hailing/TNC market’s infrastructure being nascent in 2010 rather than already developed: a U.S. DOT/BTS report states that the "nascent ridehailing market" was introduced in 2008 and that Uber only released its beta and began service in San Francisco in 2010, while the smartphone-app model itself is described as the defining infrastructure of the service. Pew’s 2010 mobile data report also shows the enabling consumer infrastructure was still early-stage in 2010—only 40% of adults used internet/email/IM on a mobile device and just 11% had purchased a product using their phone—supporting an "emerging" classification rather than "developing."_
Sources: [link](https://rosap.ntl.bts.gov/view/dot/60060/dot_60060_DS1.pdf)

---

### Role 3 Prompt Audit — Baseline Path

_Exact prompt sent to Claude (Role 3 Scorer). Proves that market name, base profile, and all identifying information are stripped before scoring. FLAGGED dimensions show both candidate values — the scorer cannot infer market identity from these._

```
=== ANONYMISED MARKET FEATURE MATRIX ===

  DIMENSION                 VERIFIED CLASSIFICATION                       INTER-SOURCE AGREEMENT
  ------------------------  --------------------------------------------  ----------------------
  timing                    pre_chasm                                     HIGH
  competition               nascent                                       HIGH
  market_size               micro                                         HIGH
  customer_readiness        adopting                                      HIGH
  regulatory                light_touch / heavy  [CONFLICT — UNRESOLVED]  LOW
  infrastructure            developing / emerging  [CONFLICT — UNRESOLVED]  MEDIUM
  market_structure          emerging                                      HIGH

=== SCORING TASK ===
Map each dimension's categorical value to a 0-100 integer using the numeric
scale in your system prompt. Score each dimension independently.
Base your scores ONLY on the categorical values above — do not infer market identity.
For any dimension marked [CONFLICT — UNRESOLVED], apply a conservative score
that reflects the uncertainty between the two candidate values shown.

REQUIRED OUTPUT: a JSON object with EXACTLY these 7 integer keys.
Do NOT add any other keys. Do NOT include tier labels, composite scores,
investment ratings, or any text outside the JSON object.
{
  "timing": 0,
  "competition": 0,
  "market_size": 0,
  "customer_readiness": 0,
  "regulatory": 0,
  "infrastructure": 0,
  "market_structure": 0
}

Replace each 0 with your integer score. Return nothing but this JSON object.
```

---

_Generated by `run_ablation_study.py` · 2026-03-19 07:15:07 · model: GPT-5.4_