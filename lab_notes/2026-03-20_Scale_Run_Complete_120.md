# Scale Pipeline v4 — 120/120 Markets Complete

**Date:** 2026-03-20
**Run:** Batches 5–9 (resuming from 66 markets)

## Summary

The scale pipeline completed its full 120-market target. The pipeline ran 9 batches total (batches 1–4 from prior session, 5–9 this session). Data is fully persisted in `reference_population_master.json`.

| Metric | Value |
|--------|-------|
| Total markets saved | 120 / 120 |
| Batches run | 9 |
| Blacklisted (timeout) | 5 |
| Deduped (skipped) | 5 |

## Batch Agreement (this session)

| Batch | Agreement | Score | H / M / L |
|-------|-----------|-------|-----------|
| 5 | MEDIUM | 0.658 | 47 / 35 / 16 |
| 6 | MEDIUM | 0.699 | 53 / 31 / 14 |
| 7 | MEDIUM | 0.626 | 41 / 32 / 18 |
| 8 | MEDIUM | 0.595 | 33 / 34 / 17 |
| 9 | HIGH | 0.714 | 4 / 2 / 1 |

## Bug Fixed

A `UnicodeEncodeError` on the Rich live-display final render (Windows GBK encoding rejecting `✓` U+2713) caused exit code 1 after all data was already saved. Fixed by replacing `✓` with `+` in `make_display()` (`run_scale_pipeline.py:189`). Data integrity unaffected.

## Notes

- Market size dimension showed the most Claude/Gemini disagreement (e.g., Claude: micro vs Gemini: massive/large) — consistent with prior batches.
- Dedup catch rate: 5/125 attempted = 4%, reasonable for this corpus size.
