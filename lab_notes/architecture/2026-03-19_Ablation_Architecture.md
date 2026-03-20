# Ablation Study Architecture — Implementation
### Date: 2026-03-19 | Type: Code Addition (no run yet)

---

## Purpose

Implements a 3-path ablation study to test the effect of different
conflict-resolution strategies between Role 1 (Claude) and Role 2 (Gemini).
A 3rd independent LLM (OpenAI GPT-4o) is used as tie-breaker to avoid
circularity — neither Claude nor Gemini adjudicates their own disagreement.

---

## Files Added

| File | Role |
|------|------|
| `src/pipeline_step2b_parametric.py` | OpenAI GPT-4o tie-breaker, parametric knowledge only |
| `src/pipeline_step2b_search.py` | OpenAI GPT-4o tie-breaker, live web search via Responses API |
| `src/run_ablation_study.py` | Orchestration: runs all 3 paths, generates report, auto-pushes |

---

## Architecture

```
                    ┌─ [Path 1: Baseline] ──────────────────────────────────┐
                    │   Role1(Claude) → Role2(Gemini) → Role3(Claude scorer) │
                    └───────────────────────────────────────────────────────┘
reference_pop.json ─┤
                    ├─ [Path 2: Parametric] ─────────────────────────────────┐
                    │   Role1 → Role2 → Step2b_Parametric(GPT-4o) → Role3   │
                    │                   (no search, parametric recall only)   │
                    └───────────────────────────────────────────────────────┘
                    │
                    └─ [Path 3: Search] ─────────────────────────────────────┐
                        Role1 → Role2 → Step2b_Search(GPT-4o+web) → Role3   │
                                        (Responses API, web_search_preview)   │
                        └───────────────────────────────────────────────────┘
```

---

## Conflict-Resolution Logic (Step 2b scripts)

**Trigger:** Any dimension where Claude and Gemini did NOT achieve HIGH agreement (i.e. MEDIUM or LOW). Agreed dimensions (HIGH) are passed through unchanged.

**Parametric judge** (`pipeline_step2b_parametric.py`):
- Uses `gpt-4o` via Chat Completions API
- Prompt: given dimension, ref_year, and the two conflicting values — choose one using internal parametric knowledge only
- Must choose exactly between the two provided values — cannot hallucinate a third option
- Returns: `resolved_classification`, `chosen` (A or B), `rationale`

**Search judge** (`pipeline_step2b_search.py`):
- Uses `gpt-4o` via OpenAI Responses API with `{"type": "web_search_preview"}` tool
- Prompt: same as parametric but also includes `market_domain` for searchability
- Extracts `search_source` URL from API response annotations (`url_citation` type)
- Falls back to Chat Completions if Responses API is unavailable (older openai SDK)

Both scripts are standalone-runnable (`python src/pipeline_step2b_parametric.py`) and importable modules.

---

## Orchestration (`run_ablation_study.py`)

1. Loads Step 2 JSON (`reference_population_scored.json` preferred)
2. For each market, runs all 3 paths in sequence
3. Captures the exact Role 3 prompt for Market 1 / Baseline path (for audit)
4. Generates a Markdown report with:
   - Role 2 Verification URLs per dimension
   - Conflict Resolution Log (what each path resolved each disagreement to)
   - 3-column score table (Baseline | Parametric | Search) with Δ columns
   - Parametric and Search resolution rationale + search source URLs
   - **Role 3 Prompt Audit** — exact prompt proving market names stripped
5. Saves report to `lab_notes/Ablation_Run_YYYYMMDD_HHMMSS.md`
6. Appends entry to `LOGBOOK_MASTER.md`
7. Auto-runs: `git add .` → `git commit -m "Auto-commit: Ablation run ..."` → `git push`

---

## Known Disagreements in Current Data (reference_population_scored.json)

| Market | Dimension | Agreement | Claude | Gemini |
|--------|-----------|-----------|--------|--------|
| Ride-hailing 2010 | regulatory | LOW | `light_touch` | `heavy` |
| Ride-hailing 2010 | infrastructure | MEDIUM | `developing` | `emerging` |
| Cloud Storage 2009 | timing | MEDIUM | `pre_chasm` | `early_chasm` |
| Cloud Storage 2009 | competition | MEDIUM | `nascent` | `fragmented` |
| Cloud Storage 2009 | market_size | MEDIUM | `micro` | `small` |
| Marketing Auto 2012 | infrastructure | MEDIUM | `mature` | `developing` |

6 total disagreements across 3 markets. The ablation study will show whether GPT-4o's
resolution changes the Role 3 scores and in which direction.

---

## Prerequisites for Running

1. `OPENAI_API_KEY` must be set in `.env` (GPT-4o access required)
2. `openai >= 1.66.0` for Responses API (`pip install --upgrade openai`)
3. Run from project root or `src/`: `python src/run_ablation_study.py`
