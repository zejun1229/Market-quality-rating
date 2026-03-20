# Scale Pipeline Architecture — Implementation
### Date: 2026-03-19 | Type: Code Addition (not yet run)

---

## Purpose

Builds `src/run_scale_pipeline.py`: an autonomous orchestration script that
loops in batches of 15 until 120 unique markets have been generated (Role 1),
verified (Role 2), and scored (Role 3) using the Baseline method.

---

## Files Added

| File | Role |
|------|------|
| `src/run_scale_pipeline.py` | Master orchestrator: market suggestion, 3-step pipeline, auto-correction, incremental save, batch log, git push |

---

## Architecture

```
loop until 120 markets accepted:
  ├─ Suggest 15 seeds   (1 Claude call, returns {domain, ref_year, knowledge_brief})
  └─ For each seed:
       ├─ Step 1: generate_market_profile() + extract_dimension_scaled() × 7
       ├─ Step 2: process_market_verification() via Gemini + Google Search
       │    └─ GATE: reject if T+5 has < 3 grounded sources
       ├─ Step 3: run_step3_market() — anonymised Claude scorer
       └─ Incremental save to reference_population_master.json
  ├─ compute_batch_agreement() → mean score (HIGH=1.0, MEDIUM=0.5, LOW=0.0)
  ├─ If mean_score < 0.40 → run_meta_correction() → new extraction_addendum
  ├─ append_batch_log() → lab_notes/Scaling_Run_Log.md
  └─ auto_git_push() → "Auto-commit: Batch N completed"
```

---

## Design Decisions

### Import strategy
Core logic imported directly from existing pipeline modules rather than inlined:
- `DIMENSIONS`, `_call_with_retry` ← pipeline_step1.py
- `get_gemini_client`, `process_market_verification`, `ORDINAL_SCALES` ← pipeline_step2.py
- `build_feature_matrix`, `build_scoring_prompt`, `SYSTEM_PROMPT` ← pipeline_step3.py

Only `extract_dimension_scaled` is a new local function (wraps step1 logic with
an optional `correction_addendum` parameter for auto-correction).

### Market suggestion
A single Claude call proposes 15 diverse seeds per batch. Existing domains are
passed to Claude to prevent duplicates. Suggestions include a `knowledge_brief`
(3–5 verified facts) which is fed into `generate_market_profile()`.

### Auto-correction mechanism
- Trigger: batch `mean_score < 0.40` (same LOW boundary used by pipeline_step2)
- Action: `run_meta_correction()` — feeds conflicting Claude vs Gemini cases
  back to Claude, asks it to identify systematic biases and produce a 3–6 bullet
  calibration note
- Output: new `correction_addendum` string injected into all future
  `extract_dimension_scaled()` calls via a CALIBRATION NOTE block in the prompt
- Persisted in `run_metadata.prompt_corrections[]` so the pipeline can resume

### Acceptance gate
A market is only counted toward the 120 if:
1. Step 1 profile generation succeeds (no API error)
2. Step 2 T+5 outcome has ≥ 3 grounded sources (VERIFIED status)
3. Step 3 scoring returns ≥ 1 valid integer score

### Incremental save
`reference_population_master.json` is written after every individual accepted
market — no batch data is lost if the script crashes.

### Resumability
On restart, `load_master()` picks up from the last saved state.
`total_accepted = len(markets)`, `batch_num` and `correction_addendum` are
all restored from the JSON metadata.

---

## Batch Log Format (Scaling_Run_Log.md)

Each batch section contains:
- Summary header (acceptance count, agreement score, auto-correction flag)
- Auto-correction addendum (if triggered)
- Per-market:
  - T+5 outcome verification status + source count
  - **Role 2 Evidence & Verification URLs** table (clickable links, Claude + Gemini values)
  - **Role 3 Scores** table (verified classification → integer score per dim)
- **Role 3 Prompt Audit** (exact raw prompt from first market of batch)
- Per-dimension agreement breakdown table

---

## Output Files

| File | Description |
|------|-------------|
| `reference_population_master.json` | Full 120-market dataset (schema v2.0) |
| `lab_notes/Scaling_Run_Log.md` | Running batch-by-batch Markdown report |

---

## Prerequisites

- `.env`: `ANTHROPIC_API_KEY` + `GEMINI_API_KEY` (both already set)
- Run from project root: `python src/run_scale_pipeline.py`
- Estimated time: ~50–70 minutes for 120 markets (dominated by Gemini 3s/call delays)
- Resumable: re-run the same command after a crash — picks up from last save
