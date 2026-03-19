# v4 Architecture Overhaul: Role 0, Dedup, Timeout, Rich Display
**Date:** 2026-03-20
**Status:** Written, not yet run

---

## Motivation

After 34 markets saved across 2 batches, the empirical audit pipeline (v3) had three pain points:
1. **Repetitive markets** — Claude kept generating semantically similar domains (e.g. multiple food delivery variants)
2. **Hanging Gemini calls** — occasionally `process_market_verification()` would stall for 90+ seconds on a single market, blocking the entire worker thread
3. **No visibility** — no live feedback on which workers were at which stage

v4 addresses all three while maintaining full resumability, parallel execution (MAX_WORKERS=12), and incremental save.

---

## Changes

### Role 0: Gemini Seed Pre-Search
Before Claude generates a market, Gemini (with Google Search grounding) is asked to identify verifiable historical venture market seeds from 2009–2021. Returns `[{domain, ref_year, knowledge_brief}]` for `BATCH_SIZE * 2` candidates. Claude (Role 1) then generates a full profile for each seed, with the seed's `knowledge_brief` prepended to the prompt for grounding.

**Why Gemini first?** Claude's training data is strong on market narratives but Gemini+Search can directly cite verifiable digital footprints (CrunchBase rounds, news). Using Gemini to seed + Claude to elaborate gives us richer, more verifiable profiles.

### Strict Deduplication (3-layer)
1. **Role 0 prompt injection**: Gemini is shown the full registry of all market names already saved + blacklist → avoids suggesting duplicates
2. **Role 1 prompt injection**: Claude is shown the same registry list → avoids generating similar names
3. **Python SequenceMatcher**: After Role 1 returns, `_is_duplicate(mkt_name, registry, threshold=0.75)` checks fuzzy similarity. If ≥0.75 ratio → market skipped, not saved, not blacklisted

### Widened Target Window
`REF_YEARS = list(range(2009, 2022))` — was 2005–2019. Focus on "mainstream, highly-funded with massive digital footprints" markets where empirical verification (Gemini search) is most reliable.

### Hard Timeout + Blacklist
`run_step2_with_timeout(gemini_client, market, timeout_secs=20.0)` wraps the Gemini call in `asyncio.wait_for()`. Each worker creates its own `asyncio.new_event_loop()` to avoid cross-thread event loop conflicts.

On `asyncio.TimeoutError`:
- Market returned as `(None, "", mkt_name)` from worker
- Main thread adds `mkt_name` to `blacklist` set
- Persisted in `run_metadata.blacklist` in master JSON
- Injected into Role 0 + Role 1 prompts on next batch

### Rich Live Terminal Visualization
```
╔═══════════════════════════════════════════════╗
║  Vela MQR | Batch 3 | 34/120                  ║
║  [████████░░░░░░░░] 28.3%  ⊗ Dedup:2  ✗ BL:0 ║
╠═══════════════════════════════════════════════╣
║  Active Workers                               ║
║  #35  Role 2: Web Searching...                ║
║  #36  Role 1: Generating...                   ║
╠═══════════════════════════════════════════════╣
║  Recent Completions (last 5)                  ║
║  ✓ #34 On-Demand Urban Restaurant...  mean=68 ║
╚═══════════════════════════════════════════════╝
```
`make_display()` returns a Rich `Panel`. Main thread calls `live.update(make_display(...))` after each `as_completed` future. Workers call `_set_stage()` which updates `_worker_status` dict under `_status_lock`.

---

## Constants
| Constant | v3 | v4 |
|---|---|---|
| `REF_YEARS` | 2005–2019 | 2009–2021 |
| `GEMINI_TIMEOUT_SECS` | — (no timeout) | 20.0 |
| `DEDUP_THRESHOLD` | — (no dedup) | 0.75 |
| `MAX_WORKERS` | 12 | 12 |
| `BATCH_SIZE` | 15 | 15 |
| `TARGET_COUNT` | 120 | 120 |

---

## State on Entry
- 34 markets saved (market_001–market_034)
- Blacklist: `[]` (empty)
- `batches_completed`: 2
- Last `correction_addendum`: restored from `prompt_corrections[-1]` on resume

---

## Expected Behaviour
1. Resume detects 34 markets → skips to batch 3
2. Role 0: Gemini searches for 30 seeds (BATCH_SIZE×2)
3. Workers process 15 seeds in parallel
4. Each market: Role 0 seed → Role 1 generate → dedup check → Role 2 verify (20s timeout) → Role 3 score
5. Incremental save after each completion; Rich display refreshes
6. Batch mean computed → auto-correction if < 0.40
7. Repeat until 120 markets saved

---

## Files Changed
- `src/run_scale_pipeline.py` — complete rewrite (v4)
- `src/pipeline_step2.py` — consolidated single Gemini call, removed all `time.sleep()`
- `src/pipeline_step1.py` — removed `time.sleep(0.4)`, lean output instruction
