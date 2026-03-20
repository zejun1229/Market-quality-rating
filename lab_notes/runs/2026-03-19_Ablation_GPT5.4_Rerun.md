# Ablation Study Rerun — GPT-5.4 Judge
### Date: 2026-03-19 | Type: Code Update + Run

---

## Code Changes

### Model upgrade: gpt-4o → gpt-5.4
Updated `PARAMETRIC_MODEL` and `SEARCH_MODEL` constants in both step2b scripts.

**API fix:** GPT-5.4 does not accept `max_tokens` — requires `max_completion_tokens`.
Updated the `client.chat.completions.create` calls in both scripts accordingly.

### New: FLAGGED state (judge may decline to resolve)
Both judges now offer two options instead of being forced to choose:
- **(A) RESOLVE** — choose one value with a clear rationale
- **(B) FLAG** — mark as AMBIGUOUS when evidence genuinely supports both,
  sources contradict, or no authoritative source exists for the reference year

FLAGGED dimensions surface to Role 3 as `A / B [CONFLICT — UNRESOLVED]` so the
scorer applies a conservative midpoint rather than picking blindly.

### New Path 1 Baseline
Previous baseline silently used Gemini's value for all conflicts. New baseline
keeps conflicts explicitly FLAGGED — Role 3 sees both options and scores
conservatively. This makes Path 1 a true "no-judge" control condition.

### Role 3 system prompt
Added Rule 6: "For dimensions marked [CONFLICT — UNRESOLVED], score conservatively
using a midpoint interpretation between the two candidate values shown."

### Removed: auto-git-push from run_ablation_study.py
Per request — report generation only, no automated commits.

---

## Run Results

**Test market:** US App-Based On-Demand Black Car Transportation (2010)
**Conflicts entering the study:** 2 (regulatory: LOW, infrastructure: MEDIUM)
**Report:** [Ablation_Run_GPT5.4.md](./Ablation_Run_GPT5.4.md)

### Conflict Resolution

| Dimension | Agreement | Claude | Gemini | Baseline | Parametric | Search |
|-----------|:---------:|--------|--------|:--------:|:----------:|:------:|
| `regulatory` | LOW | `light_touch` | `heavy` | ⚑ FLAGGED | `heavy` (B) ✓ | `heavy` (A) ✓ |
| `infrastructure` | MEDIUM | `developing` | `emerging` | ⚑ FLAGGED | ⚑ FLAGGED | `emerging` (B) ✓ |

**Key observations:**
- Parametric judge resolved `regulatory` → `heavy` (chose Gemini's value), citing
  post-crisis US regulatory tightening in 2010 (Dodd-Frank era). Could not
  resolve `infrastructure` — flagged as genuinely ambiguous between developing/emerging.
- Search judge resolved both: `regulatory` → `heavy` (despite labelling it "chose A",
  the value matches Gemini), `infrastructure` → `emerging` (chose Gemini's value),
  backed by web-retrieved evidence.
- The search judge eliminated all flags; the parametric judge left 1 flagged.

### Score Impact

| Path | timing | competition | market_size | customer_readiness | regulatory | infrastructure | market_structure | **Mean** |
|------|:------:|:-----------:|:-----------:|:-----------------:|:----------:|:--------------:|:---------------:|:--------:|
| Baseline (flagged) | 62 | 78 | 22 | 72 | 48 ⚑ | 52 ⚑ | 68 | **57** |
| Parametric | 62 | 78 | 22 | 72 | 28 ✓ | 55 ⚑ | 70 | **55** |
| Search | 62 | 82 | 18 | 72 | 25 ✓ | 52 ✓ | 58 | **53** |

**Key:** resolving `regulatory` to `heavy` dropped its score from 48 → 28/25 (heavily
regulated markets score lower on venture attractiveness). The conservative FLAGGED
baseline scored it at 48, which was meaningfully inflated relative to the resolved value.

---

## Technical Notes

- `gpt-5.4` confirmed available and working via OpenAI Chat Completions API
- `max_completion_tokens` required (not `max_tokens`) — applied to both step2b scripts
- Responses API (web search) call format unchanged; `gpt-5.4` accepts same tool spec
- No git push performed (by design for this run)
