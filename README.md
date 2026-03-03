# SuSteelAible

Analyzing decarbonization pathways in the EU steel industry through emissions data and corporate sustainability report analysis.

---

## → [Results & Key Findings](results/RESULTS.md)

Start here if you want to see what the project found — technology lock-in, the carbon price paradox, barriers and motivators extracted from 200 sustainability reports, and interactive topic visualizations.

> Interactive HTML visualizations are in `results/topics/` (open locally after cloning). Final presentation: `results/final_presentation.pdf`.

---

## Overview

**SuSteelAible** combines **quantitative emissions analysis** with **NLP analysis of ~200 corporate sustainability reports** (2013–2025) to understand technology lock-in, climate commitments, and decarbonization barriers in European steel.

**Three-part analysis:**
1. **Emissions EDA** — Technology gap, carbon price paradox, Scope 2 trends
2. **Modeling** — Technology baseline, panel econometrics, action score framework
3. **NLP Pipeline** — ClimateBERT classification, RAG extraction, BERTopic clustering

**Key findings preview:**
- Technology choice (BF-BOF vs EAF) explains ~79% of emissions variance — the single largest driver
- Carbon prices rose 17× while sector-wide emission intensity stayed flat — structural technology lock-in
- Policy interventions (ETS, CBAM, Green Deal) show no significant within-firm intensity reductions
- Top barrier: limited availability of low-carbon steel inputs and certification costs
- Top motivator: internal emissions reduction commitments, ahead of regulatory compliance


## Quick Start

```bash
git clone git@github.com:am0ebe/SusteelAible.git
cd SusteelAible

python3.11 -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

pip install -e .

# spaCy language model (required for NLP pipeline)
python -m pip install -v https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Extract EDA data + BERT cache
unzip data.zip
# → data/EDA/   (emissions, external drivers, EU ETS data)
# → cache/      (preprocessed BERT JSONs for all 197 reports)

# (Optional) Groq API key — required for RAG extraction and topic labeling (NLP sections 3 & 4)
echo 'GROQ_API_KEY=your-key-here' > .env
```

Done. You can now run all analysis notebooks. The BERT cache lets you skip the slow preprocessing step and jump directly to RAG extraction or topic modeling.


## Source Reports (Optional)

The original PDF reports (~1.6 GB) are only needed to rerun preprocessing from scratch.

**Download:** <https://github.com/am0ebe/SusteelAible/releases/download/v1.1-data/reports.zip>

```bash
unzip reports.zip
# → data/reports/   (source PDFs, organized by company)
```


## Notebooks

<details>
<summary><b>01_eda/ — Exploratory Data Analysis</b></summary>

#### `EDA_emissions.ipynb`
Emission patterns across 13 European steel companies (2013–2024).
- Technology gap: BF-BOF (~1.4–2.5 tCO₂e/t) vs EAF (~0.08–0.5 tCO₂e/t)
- Time trends: Scope 1 flat; Scope 2 declining for EAF via grid decarbonization
- Carbon price paradox: 17× price increase, flat sector intensity
- Data loads via `scripts/data_loader.py`; requires `data/EDA/` from `data.zip`

#### `external_drivers_eda_and_model.ipynb`
Panel econometrics on how prices and policy affect emission intensity.
- Fixed effects: only firm age significant within-firm; policy variables not significant
- DiD: ETS, CBAM, Green Deal show no significant intensity reduction
- Granger causality: coal price → intensity (lag 2, p=0.0015); carbon price borderline

</details>

<details>
<summary><b>02_models/ — Baseline and Action Score Models</b></summary>

#### `baseline_model.ipynb`
Technology type (EAF vs BF-BOF) as a decision tree baseline — explains ~79% of emission intensity variance.

#### `predictions.ipynb`
Panel econometrics + ML to identify statistically significant emission drivers; scenario analysis for future emissions.

#### `action_score_concept.ipynb`
Composite 100-point score assessing decarbonization effort per company:
- Performance (30 pts), Trend (30 pts), Data Quality (15 pts), Technology (20 pts), Renewable (5 pts)

#### `action_score_temporal.ipynb`
Applies the action score framework to pre-COVID (2013–2019) vs. post-COVID (2020–2024) periods.

</details>

<details>
<summary><b>03_nlp/ — NLP Pipeline</b></summary>

Open `03_nlp/run_all.ipynb` — the notebook intro explains the full pipeline and guides you through each step.

**GPU:** ClimateBERT runs on PyTorch. GPU makes it significantly faster. Install PyTorch with CUDA **before** `pip install -e .` using the interactive installer at **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)**. The notebook will confirm which device was detected.

**Skip to RAG or topic modeling:** The `cache/` folder from `data.zip` contains preprocessed BERT outputs for all 197 reports. Steps 1–2 detect the cache and skip reprocessing automatically.

```
data/reports/ (PDFs)
    ↓ [1] ClimateBERT             → cache/*.json
    ↓ [2] BERT Visualization      → out/bert/
    ↓ [3] RAG Extraction          → out/rag/
    ↓ [4] Topic Modeling          → out/topics/run_XX/
```

</details>


## Results

Pre-computed outputs committed to `results/` — browse without running the pipeline:

| Folder | Contents |
|--------|----------|
| `results/bert/` | Talk score trends, sentiment, net-zero funnel, word clouds (PNG + CSV) |
| `results/rag/` | Per-company barriers & motivators CSVs (15 companies) |
| `results/topics2/` | BERTopic deliverable: merged top-6 topics, interactive HTML visualizations |
| `results/topics1/` | Full topic set (17 barriers, 16 motivators) for deeper exploration |
| `results/final_presentation.pdf` | Final project presentation |

**→ [Full findings summary](results/RESULTS.md)**


## Project Structure

```
├── 01_eda/                     # Exploratory data analysis
│   ├── EDA_emissions.ipynb
│   └── external_drivers_eda_and_model.ipynb
│
├── 02_models/                  # Baseline and action score models
│   ├── baseline_model.ipynb
│   ├── predictions.ipynb
│   ├── action_score_concept.ipynb
│   └── action_score_temporal.ipynb
│
├── 03_nlp/                     # NLP pipeline
│   └── run_all.ipynb           # Main pipeline notebook (run this)
│
├── scripts/                    # Shared data utilities (used by 01_eda/, 02_models/)
│
├── data.zip                    # EDA data + BERT cache (~56MB) → data/EDA/ + cache/
├── cache/                      # BERT-preprocessed JSONs (from data.zip)
├── data/EDA/                   # Emissions + external driver data (from data.zip)
├── data/reports/               # Source PDFs (from reports.zip, optional)
│
└── results/                    # Pre-computed outputs
    ├── RESULTS.md              # ← Key findings summary
    ├── bert/
    ├── rag/
    ├── topics1/                # Full topic set (all topics, pre-deliverable)
    ├── topics2/                # Deliverable: merged top-6 topics
    └── final_presentation.pdf
```


## Requirements

- Python 3.11+
- Full dependencies in `pyproject.toml`
- Recommended: NVIDIA GPU (CUDA) or Apple Silicon

Install extras:
```bash
pip install -e ".[dev]"   # JupyterLab + ipywidgets
pip install -e ".[gpu]"   # GPU-accelerated FAISS (NVIDIA only)
```


## Team

**SuSteelAible** — March 2026

[@am0ebe](https://github.com/am0ebe) · [@calluna-borealis](https://github.com/calluna-borealis) · [@dzyen](https://github.com/dzyen) · [@aposkoub92](https://github.com/aposkoub92) · [@MJR-data](https://github.com/MJR-data)

Questions or bugs? [Open an issue](https://github.com/am0ebe/SusteelAible/issues).
