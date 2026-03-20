# Vela MQR — Market Quality Rating Pipeline

A multi-agent pipeline that generates a grounded reference population of venture markets, assigns L1-L5 quality ratings, and runs a 2x4 ablation study to measure prediction accuracy against T+5 historical outcomes.

---

## Architecture

```
Role 0  Gemini + Google Search   Seed market generation (grounded web search)
Role 1  Claude Sonnet 4.6        T=0 market profile + 7 dimension scores
Role 2  Gemini + Google Search   Grounded verification of all 7 dimensions
Step 3  Python                   Composite scoring (causal weights by structure type)
Step 4  Python + sklearn         Percentile rating, logistic regression, nearest neighbours
```

### Pipeline Scripts (`src/`)

| Script | Purpose |
|--------|---------|
| `pipeline_step1.py` | Claude: market profile narrative + dimension classifications |
| `pipeline_step2.py` | Gemini: grounded dimension verification + agreement scoring |
| `pipeline_step3.py` | Composite score computation (structure-specific causal weights) |
| `pipeline_step4_rating.py` | Percentile rating, LR P(>=L3), cosine NN — accepts `--input` / `--output` |
| `run_scale_pipeline.py` | Autonomous orchestration: 350-market reference population generation |
| `pipeline_validation.py` | Mirror validation pipeline: 100-market holdout with T+5 ground truth |
| `performance_scorer.py` | Standalone T+5 scorer — `score_and_label(markets, bands='symmetric'|'absolute')` |
| `run_ablation_mqr.py` | 2x4 ablation harness: prediction reference size x calibration method |

### Analysis Scripts (`analysis/`)

| Script | Purpose |
|--------|---------|
| `eda_120_markets.py` | EDA on the initial 120-market population |
| `validation_10_markets.py` | Manual 10-market OOS validation (40% exact, 70% within ±1) |
| `oos_test_case.py` | Individual OOS case studies |
| `oos_report2.py` | Second OOS report generation |

---

## Data Artifacts (`data/`)

| File | Description |
|------|-------------|
| `reference_population_master.json` | 350 grounded reference markets (root — pipeline output) |
| `predict_120.json` | First 120 markets sliced for ablation predictor condition |
| `predict_350.json` | Full 350 markets for ablation predictor condition |
| `rated_120.json` | Step 4 output: 120-market cohort, L1-L5 assigned |
| `rated_350.json` | Step 4 output: 350-market cohort, L1-L5 assigned |
| `validation_population.json` | 100-market holdout with T+5 ground truth metrics |
| `final_rated_population.json` | Original 120-market rated output (legacy) |
| `reference_population_scored.json` | Intermediate scored population (legacy) |

---

## 2x4 Ablation Study

**Dimension 1 — Prediction Reference Size:** REF-120 vs REF-350
**Dimension 2 — Performance Calibration:** Absolute / Symmetric-10 / Symmetric-60 / Symmetric-100

### Results

| Predictor \ Calibration | Absolute | Sym-10 | Sym-60 | Sym-100 |
|---|---|---|---|---|
| **REF-120** | E=24.0% O=67.0% | E=30.0% O=40.0% | E=23.3% O=63.3% | E=26.0% O=64.0% |
| **REF-350** | E=21.0% O=58.0% | E=30.0% O=50.0% | **E=30.0% O=66.7%** | E=30.0% O=64.0% |

*E = Exact Match %, O = Within ±1 Band %*

**Optimal configuration: REF-350 x Sym-60** (30% exact, 66.7% within ±1).
Full results: `experiments/case_study_2x4/ablation_results.md`

---

## Setup

```bash
pip install anthropic google-genai rich numpy scipy scikit-learn python-dotenv
```

Create `.env` in the project root:
```
ANTHROPIC_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## Running the Scale Pipeline

```bash
# Generate / resume the 350-market reference population
python src/run_scale_pipeline.py

# Rate the population (produces data/rated_350.json)
python src/pipeline_step4_rating.py --input data/predict_350.json --output data/rated_350.json

# Generate 100-market validation holdout
python src/pipeline_validation.py

# Run the 2x4 ablation study
python src/run_ablation_mqr.py
```

---

## Project Logbook

See `lab_notes/LOGBOOK_MASTER.md` for full experiment history.

| Subfolder | Contents |
|-----------|---------|
| `lab_notes/architecture/` | Design docs for each pipeline component |
| `lab_notes/runs/` | Per-run execution logs |
| `lab_notes/reports/` | OOS case studies and EDA reports |
| `_review/` | Deprecated scripts and legacy data files (pending manual cleanup) |
