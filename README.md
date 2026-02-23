# SuSteelAible

Analyzing decarbonization pathways in the EU steel industry through integrated emissions data and corporate sustainability report analysis.

![net-zero wordclouds](wordcloud.png "net-zero wordclouds")


## Overview

This project combines **quantitative emissions analysis** with **qualitative text analysis** of ~200 corporate sustainability reports (2013–2025) to understand technology lock-in, climate commitments, and decarbonization barriers in European steel.

**Key findings:**

- Technology choice (BF-BOF vs EAF) explains ~79% of emissions variance
- Process emissions (Scope 1) remain largely stable—real progress comes from grid decarbonization (Scope 2)
- ClimateBERT analysis reveals trends in climate discourse specificity, commitment language, and net-zero focus

## Quick Start

```bash
# Clone repository
git clone git@github.com:am0ebe/SusteelAible.git
cd SusteelAible

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install as package
pip install -e .

# Download spaCy language model
python -m pip install -v  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Extract data
unzip data.zip

# (Optional) Add Groq API key — required for RAG extraction and topic labeling (sections 3 & 4)
echo 'GROQ_API_KEY=your-key-here' > .env
```

✅ **Done!** You can now run all analysis notebooks.

## Analyze Source Reports

Want to see the original sustainability reports we analyzed?

**Download reports (~1.6GB):**
<https://github.com/am0ebe/SusteelAible/releases/download/v1.1-data/reports.zip>



```bash
# Reports will be extracted to `data/reports/`
unzip reports.zip
```

<details>
<summary><h2 style="display: inline; cursor: pointer;"> EDA</h2></summary>

*(section in progress — Irene)*

<!-- TODO: overview of 01_eda/ notebooks, key findings, how to run -->

</details>

<details>
<summary><h2 style="display: inline; cursor: pointer;"> NLP</h2></summary>

### GPU (recommended)

ClimateBERT runs on PyTorch — GPU makes it significantly faster, CPU works but is much slower. GPU is the single biggest setup decision for this pipeline.

Install PyTorch with CUDA **before** `pip install -e .` (pip won't upgrade an existing torch installation). Use the interactive installer at **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)** to get the right command for your OS and CUDA version. Section 1 of `run_all.ipynb` will confirm which device was detected when you run it.

### Run the pipeline

Open `03_nlp/run_all.ipynb` — the notebook intro explains the full pipeline and guides you through each step.

</details>


## Project Structure
```
├── README.md
├── pyproject.toml
├── reports.zip
│
├── 01_eda/                    # Exploratory data analysis
│   ├── basic_EDA.ipynb
│   └── baseline_model.ipynb
│
├── 02_models/                 # Baseline emissions models
│
├── 03_nlp/                    # NLP, BERT, and RAG pipelines
│   │
│   ├── run_all.ipynb          # Main pipeline notebook (run this)
│   ├── preprocessing.py       # PDF → text chunks
│   ├── bert_1.py              # ClimateBERT classification (5 models)
│   ├── bert_2.py              # Visualization & CSV export
│   ├── llm_extract.py         # Exhaustive LLM extraction pipeline
│   ├── rag.py                 # FAISS-based RAG extraction
│   ├── topic_modelling.py     # BERTopic clustering & LLM labeling
│   ├── topic_gridsearch.py    # Staged HDBSCAN/UMAP hyperparameter search
│   ├── data_loader.py         # Cache loading utilities
│   ├── gpu_utils.py           # GPU device management
│   │
│   └── cache/                 # Cached / preprocessed model outputs
│
├── out/                       # Generated plots and visualizations
└── data/
    └── reports/               # Source PDF reports (from reports.zip)
```
## Requirements

- Python 3.11+ (declared in `pyproject.toml`, enforced by pip)
- See `pyproject.toml` for full dependencies
- Optional: NVIDIA GPU (CUDA) or Apple Silicon for faster processing

## Misc

Optional install extras (defined in `pyproject.toml`):
- `pip install -e ".[dev]"` — JupyterLab + ipywidgets (suppresses tqdm warnings in notebooks)
- `pip install -e ".[gpu]"` — GPU-accelerated FAISS (faster RAG retrieval, NVIDIA only)

## Team

**SuSteelAible** — March 2026

[@am0ebe](https://github.com/am0ebe) · [@calluna-borealis](https://github.com/calluna-borealis) · [@dzyen](https://github.com/dzyen) · [@aposkoub92](https://github.com/aposkoub92) · [@MJR-data
](https://github.com/MJR-data
)

## Contact

Questions, bugs, or suggestions? Feel free to [open an issue](https://github.com/am0ebe/SusteelAible/issues) or reach out to the team. We're happy to help!
