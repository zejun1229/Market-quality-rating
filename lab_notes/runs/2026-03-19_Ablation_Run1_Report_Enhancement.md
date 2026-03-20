# Ablation Study — Report Enhancement & First Run
### Date: 2026-03-19 | Type: Code Update + Run

---

## Code Changes (run_ablation_study.py)

Restructured `generate_report()` to meet final reporting requirements:

1. **Per-Path Full Analysis** — Added three dedicated subsections per market:
   - **Path 1 (Baseline):** table of all 7 dimensions with classification, agreement, Gemini evidence summary, and Role 3 score
   - **Path 2 (Parametric Judge):** table with classification, source tag, GPT-4o rationale, and score for every dimension (agreed and disputed)
   - **Path 3 (Search Judge):** same as parametric plus search source URL column

2. **URL Audit Section** — Dedicated section at the bottom of the report listing all Gemini verification URLs across all markets in one consolidated table (market × dimension → URL)

3. **Role 3 Prompt Audit** — Already present; confirmed to show exact raw prompt with market name stripped

Model check: `gpt-4o` confirmed as the model in both `pipeline_step2b_parametric.py` and `pipeline_step2b_search.py` — no change needed.

---

## First Ablation Run — Results Summary

**Run ID:** `20260319_011521`
**Input:** `reference_population_scored.json` (3 markets)
**Report:** [Ablation_Run_20260319_011521.md](./Ablation_Run_20260319_011521.md)

### Conflict Resolutions

| Market | Dimension | Agreement | Claude | Gemini | Parametric → | Search → |
|--------|-----------|:---------:|--------|--------|:------------:|:--------:|
| Ride-hailing 2010 | regulatory | LOW | `light_touch` | `heavy` | `light_touch` (A) | `light_touch` (A) |
| Ride-hailing 2010 | infrastructure | MEDIUM | `developing` | `emerging` | `emerging` (B) | `emerging` (B) |
| Cloud Storage 2009 | timing | MEDIUM | `pre_chasm` | `early_chasm` | `early_chasm` (B) | `early_chasm` (B) |
| Cloud Storage 2009 | competition | MEDIUM | `nascent` | `fragmented` | `fragmented` (B) | `nascent` (A) |
| Cloud Storage 2009 | market_size | MEDIUM | `micro` | `small` | `small` (B) | `small` (B) |
| Marketing Auto 2012 | infrastructure | MEDIUM | `mature` | `developing` | `developing` (B) | `developing` (B) |

**Key finding:** Parametric and Search judges agreed on 5/6 resolutions. The single divergence was Cloud Storage `competition`: parametric chose Gemini's `fragmented`, while search-grounded GPT-4o chose Claude's `nascent` — web evidence supported the less-competitive 2009 cloud market characterisation.

### Score Impact (Mean across 7 dimensions)

| Market | Baseline | Parametric | Search | Δ Param | Δ Search |
|--------|:--------:|:----------:|:------:|:-------:|:--------:|
| Ride-hailing 2010 | 51 | 58 | 59 | +7 | +8 |
| Cloud Storage 2009 | 61 | 62 | 62 | +1 | +1 |
| Marketing Auto 2012 | 61 | 61 | 60 | 0 | -1 |

**Observation:** Tie-breaking had the largest impact on Ride-hailing (regulatory resolved from ambiguous to `light_touch` pushed score +7/+8). Cloud Storage and Marketing Auto were near-stable.

---

## Known Issue

`src/reference_population.json` (the working copy used as a path workaround for step2) was accidentally staged by `git add .` in the auto-push and is now committed to GitHub. It is safe data but was intended to be excluded. Will add it to `.gitignore` in a future cleanup.
