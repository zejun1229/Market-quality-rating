# Pipeline Upgrade — v3 Canonical Enums & Verification URLs
### Date: 2026-03-18 | Type: Code Change (no new run)

---

## Summary

Upgraded both pipeline scripts to enforce strict, canonical economic-framework enums across all 7 dimensions and to surface a `verification_url` per dimension in Step 2 output. **No new pipeline run was performed** — these changes take effect from the next run onward.

Schema version bumped: `1.1` → `2.0`

---

## Files Changed

| File | Nature of change |
|------|-----------------|
| `src/pipeline_step1.py` | `DIMENSIONS` options + scoring guides replaced; schema_version → `"2.0"`; banner → `v3 canonical enums` |
| `src/pipeline_step2.py` | `ORDINAL_SCALES` updated; `verify_dimension` prompt + return dict updated; `_parse_gemini_classification` updated; `print_final_report` dimension matrix updated |

---

## New Enum Schema (v2.0)

### `timing` — Rogers Diffusion of Innovations
| Value | Description |
|-------|-------------|
| `innovators` | First ~2.5%; technology enthusiasts; no social proof needed |
| `early_adopters` | Next ~13.5% (cum. ~16%); visionaries; opinion leaders |
| `early_majority` | Next ~34% (cum. ~50%); pragmatists; need proven ROI; chasm crossed |
| `late_majority` | Next ~34% (cum. ~84%); conservatives; adopt under peer pressure |
| `laggards` | Final ~16%; traditionalists; adopt only when legacy alternatives cease |

**Previous values:** `pre_chasm · early_chasm · early_majority · late_majority · peak`

---

### `competition` — Industrial Economics Market Structure
| Value | Description |
|-------|-------------|
| `monopoly` | Single seller; near-absolute pricing power; prohibitive entry barriers |
| `oligopoly` | 2–5 players control >70% share; strategic interdependence; high barriers |
| `monopolistic_competition` | Many firms; differentiated products; moderate price-setting ability |
| `perfect_competition` | Many sellers; homogeneous products; price-taking; near-zero margins |

**Previous values:** `nascent · fragmented · consolidating · consolidated · commoditized`

---

### `market_size` — Revenue-Band Tiers (current spend, NOT TAM)
| Value | Annual category revenue |
|-------|------------------------|
| `micro` | < $100M |
| `small` | $100M – $1B |
| `medium` | $1B – $10B |
| `large` | $10B – $100B |
| `massive` | > $100B |

**Previous values:** same except `mega` → `massive`

---

### `customer_readiness` — Gartner Hype Cycle
| Value | Description |
|-------|-------------|
| `innovation_trigger` | Breakthrough event; early POC; no viable product; extreme hype begins |
| `peak_of_inflated_expectations` | Amplified early wins; unrealistic expectations; early failures appearing |
| `trough_of_disillusionment` | Implementations fail; interest wanes; only committed buyers remain |
| `slope_of_enlightenment` | 2nd/3rd-gen products; cautious enterprise pilots; ROI materialising |
| `plateau_of_productivity` | Mainstream adoption; ROI demonstrable; sustainable market growth |

**Previous values:** `unaware · aware · interested · ready · adopting`

---

### `regulatory` — Compliance-Burden Ladder
| Value | Description |
|-------|-------------|
| `unregulated` | No sector-specific rules; general commercial law only |
| `light_touch` | Self-regulation or adjacent rules applied loosely; minimal overhead |
| `moderate_compliance` | Formal sector rules enacted; active enforcement; manageable burden |
| `heavily_regulated` | Substantial compliance burden affecting product design and sales cycles |
| `prohibitive` | Effectively blocked in key markets; licensing bans or moratoriums |

**Previous values:** `unregulated · light_touch · moderate · heavy · restricted`

---

### `infrastructure` — Technology Maturity Ladder (4 levels)
| Value | Description |
|-------|-------------|
| `nascent` | Critical enablers not yet built; fundamental R&D required |
| `emerging` | Enablers exist but unreliable/expensive; not production-ready |
| `developing` | Functional and accessible; cost/reliability still improving |
| `mature` | Robust, reliable, widely accessible; no meaningful constraint |

**Previous values:** `non_existent · emerging · developing · mature · commoditized`

---

### `market_structure` — Competitive Archetype Taxonomy (categorical)
| Value | Description |
|-------|-------------|
| `winner_take_most` | Network effects → single dominant platform captures majority of value |
| `platform_two_sided` | Marketplace connecting distinct buyer/seller groups; cross-side effects |
| `technology_enablement` | Horizontal infra/API layer enabling verticals; no direct end-customer |
| `fragmented_niche` | Many specialists serving sub-segments; no platform dynamics |
| `regulated_infrastructure` | Utility-like; regulated pricing/access; natural-monopoly tendencies |

**Previous values:** `undefined · emerging · forming · defined · mature`
> Note: this dimension is **categorical, not ordinal**. Agreement scoring uses exact match = HIGH; any mismatch scores by list-distance position (has limited semantic meaning).

---

## Step 2: Verification URL Extraction

Each dimension verification call in Step 2 now returns and displays a `verification_url`:

**Source priority:**
1. **Grounding metadata URL** (`urls[0]` from Google Search API response) — authoritative
2. **Gemini JSON `source_url`** field — Gemini's self-reported citation, used as fallback

**Output change:** The dimension matrix in the final report now includes a `Verification URL` column, enabling direct source auditing per dimension.

**JSON output change:** Each entry in `step2.dimension_verifications[dim]` now includes:
```json
{
  "verification_url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/...",
  "grounding_urls": ["url1", "url2", "url3"]
}
```

---

## Impact on Existing Data

- `reference_population.json` and `reference_population_v3.json` were generated under schema `v1.1` with old enum values. They remain valid for historical reference but are **not compatible** with the new schema.
- The next run should target a new output file (e.g. `reference_population_v4.json`) to avoid mixing schema versions.

---

## Compatibility Note

The Step 2 `ORDINAL_SCALES` are kept in sync with Step 1 `DIMENSIONS` options. If Step 1 options are ever modified again, `ORDINAL_SCALES` in Step 2 must be updated to match.
