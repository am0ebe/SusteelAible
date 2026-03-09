# SuSteelAible

Analyzing decarbonization pathways in the European steel industry through emissions data and corporate sustainability reports.

---

## Overview

SuSteelAible combines **quantitative emissions analysis** with **natural language processing (NLP) of ~200 corporate sustainability and annual reports (2013–2025)** to investigate technology lock-in, climate commitments, and decarbonization barriers in the European steel sector.

The project integrates firm-level emissions data with large-scale text analysis to understand how **technology, policy signals, and corporate narratives interact in shaping industrial decarbonization trajectories.**

➡️ A detailed explanation of the research design, dataset construction, and full analytical results is available in **`project_overview.md`**.

## Analytical Framework

The analysis is structured in three main components:

**1. Emissions Analysis** 
- Technology gap between **BF–BOF** and **EAF** production routes
- The **carbon price paradox** in the EU ETS
- Scope 2 electricity trends and energy-system exposure

**2. Econometric Modeling** 
- Technology baseline models explaining emission intensity
- Panel regressions evaluating policy sensitivity
- **Action Score framework** for measuring operational decarbonization readiness

**3. NLP Pipeline** 
- **ClimateBERT** classification of climate-related disclosures
- **RAG (Retrieval-Augmented Generation)** for extracting barriers and motivators
- **BERTopic** clustering to map decarbonization discourse

## → [Key Findings](results/RESULTS.md)

Full results are available in **[results/RESULTS.md](results/RESULTS.md)**. The most important findings include:

- **Technology dominates emissions outcomes:**
Production technology (BF–BOF vs EAF) explains ~80% of variation in emissions intensity.
- **Carbon price escalation without intensity reduction:**
EU ETS prices increased roughly 17× (€5 → €85+), while sectoral emissions intensity remained largely unchanged—suggesting **structural technology lock-in rather than incentive failure.**
- **Limited measurable policy effects:**
ETS reforms, CBAM, and the EU Green Deal show **no statistically significant within-firm emissions reductions** during the study period.
- **Action scores split along technological lines:**
EAF producers: 50–93 points
BF–BOF producers: 12–40 points
- **Top reported barrier:**
Limited availability of **low-carbon steel inputs and certification costs.**
- **Top reported motivator:**
**Internal emissions targets**, often adopted ahead of regulatory requirements.

## Results and Visualization
Additional outputs and interactive materials:
- **Topic visualizations:**
`results/topics2/` (open HTML locally). 
- **Project Presentation:** 
`results/final_presentation.pdf`. 
- **Transition Tracker animation (Action vs Communication):** 
`results/models/talk_vs_action.mp4`.

---

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
#   The cache lets you skip ClimateBERT preprocessing and jump straight to RAG/topic modeling.

# (Optional) Groq API key — required for RAG extraction and topic labeling (NLP sections 3 & 4)
echo 'GROQ_API_KEY=your-key-here' > .env

# (NLP only) GPU strongly recommended for ClimateBERT. Install PyTorch with CUDA *before* pip install -e .
# → https://pytorch.org/get-started/locally/
```

Done. You can now run all analysis notebooks.


## Source Reports (Optional)

The original PDF reports (~1.6 GB) are only needed to rerun preprocessing from scratch.

**Download:** <https://github.com/am0ebe/SusteelAible/releases/download/v1.1-data/reports.zip>

```bash
unzip reports.zip
# → data/reports/   (source PDFs, organized by company)
```


## Project Structure

```
├── 01_eda/                     # Exploratory data analysis
│   ├── EDA_emissions.ipynb                      # Emission patterns, tech gap, carbon price paradox
│   └── external_drivers_eda_and_model.ipynb     # Panel econometrics + DiD on ETS/CBAM/Green Deal
│
├── 02_models/                  # Baseline and action score models
│   ├── baseline_model.ipynb                     # Decision tree baseline — tech type explains ~79% variance
│   ├── predictions.ipynb                        # PanelOLS + Random Forest; scenario projections to 2035
│   ├── action_score_concept.ipynb               # 100-pt decarbonization score per company
│   └── action_score_temporal.ipynb              # Pre/post-COVID comparison + Talk vs Action animation
│
├── 03_nlp/                     # NLP pipeline
│   ├── run_all.ipynb                            # Main pipeline notebook (run this)
│   ├── preprocessing.py                         # PDF → text chunks
│   ├── bert_1.py                                # ClimateBERT classification (5 scores per chunk)
│   ├── bert_2.py                                # BERT score visualizations + CSV export
│   ├── rag.py                                   # FAISS retrieval + LLM extraction of barriers/motivators
│   ├── llm_extract.py                           # Exhaustive LLM extraction (base class for rag.py)
│   ├── topic_modelling.py                       # BERTopic clustering + LLM labeling
│   ├── topic_gridsearch.py                      # Grid search over embedding/HDBSCAN/UMAP params
│   ├── model_test.py                            # Quick LLM evaluation for extraction quality
│   ├── data_loader.py                           # Shared cache loader (prep + bert JSON files)
│   └── gpu_utils.py                             # GPU/device management utilities
│
├── scripts/                    # Shared data utilities (used by 01_eda/, 02_models/)
│
├── data.zip                    # EDA data + BERT cache (~56MB) → data/EDA/ + cache/
├── cache/                      # BERT-preprocessed JSONs (from data.zip)
├── data/EDA/                   # Emissions + external driver data (from data.zip)
├── data/reports/               # Source PDFs (from reports.zip, optional)
│
└── results/                    # Pre-computed outputs (browse without running the pipeline)
    ├── RESULTS.md              # ← Key findings summary
    ├── bert/                   # Talk score trends, sentiment, net-zero funnel, word clouds (PNG + CSV)
    ├── rag/                    # Per-company barriers & motivators CSVs (15 companies)
    ├── topics1/                # Full topic set (17 barriers, 16 motivators)
    ├── topics2/                # Deliverable: merged top-6 topics + interactive HTML visualizations
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
