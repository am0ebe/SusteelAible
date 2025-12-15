# SuSteelAible

Analyzing decarbonization pathways in the EU steel industry through integrated emissions data and corporate sustainability report analysis.

## Overview

This project combines **quantitative emissions analysis** with **qualitative text analysis** of ~200 corporate sustainability reports (2013–2024) to understand technology lock-in, climate commitments, and decarbonization barriers in European steel.

**Key findings:**
- Technology choice (BF-BOF vs EAF) explains ~79% of emissions variance
- Process emissions (Scope 1) remain largely stable—real progress comes from grid decarbonization (Scope 2)
- ClimateBERT analysis reveals trends in climate discourse specificity, commitment language, and net-zero focus

## Quick Start

```bash
# Clone repository
git clone https://github.com/am0ebe/SusteelAible
cd susteelaible

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (BERT models), ensure CUDA is installed
# Tested on NVIDIA GPUs with 4GB+ VRAM
```

<details>
<summary><strong>OS-specific notes</strong></summary>

**Linux:** Works out of the box. For CUDA issues, check `nvidia-smi` and ensure drivers are loaded.

**macOS:** CPU-only for BERT models (no CUDA). Processing will be slower but functional.

**Windows:** Use PowerShell or Git Bash. Activate venv with `.venv\Scripts\activate`. For CUDA, install via [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

</details>

## 📦 Data Setup

### Standard Setup

```bash
# After cloning and pip install:
unzip data.zip
```

This gives you:
- `data/EDA/` — emissions data for EDA & baseline notebooks
- `cache/` — pre-processed ClimateBERT results (~200 reports)

You can now run **all notebooks except `bert_1.ipynb`** (the processing pipeline).

### Full Pipeline (Optional)

To re-run the ClimateBERT pipeline from scratch (~2-3 hours, GPU recommended):

```bash
# Download source PDFs from Releases (~1.6GB)
wget https://github.com/am0ebe/SusteelAible/releases/download/v1.0/reports.zip
unzip reports.zip
```

Then delete `cache/*` and run `bert_1.ipynb` to process all PDFs → generates fresh `cache/*.json` files.

## Project Structure

```
├── basic_EDA.ipynb        # Exploratory analysis (emissions)
├── baseline_model.ipynb   # Technology baseline model
├── bert_1.ipynb           # ClimateBERT pipeline (requires reports/)
├── bert_2.ipynb           # Report aggregation & visualization
├── rag_barriers.ipynb     # RAG-based barrier analysis (coming soon)
├── data.zip               # ⬅ Unzip first! Contains EDA + cache
├── data/
│   ├── EDA/               # Emissions data (from data.zip)
│   └── reports/           # PDFs (download from Releases)
├── cache/                 # ClimateBERT results (from data.zip)
├── monitor_gpu.py         # GPU monitoring utility
└── requirements.txt
```

## Notebooks

Each notebook contains its own setup instructions and documentation.

| Notebook | Focus |
|----------|-------|
| `basic_EDA.ipynb` | Emissions trends, technology lock-in analysis |
| `baseline_model.ipynb` | Decision tree baseline (EAF vs BF-BOF) |
| `bert_1.ipynb` | PDF extraction → translation → ClimateBERT pipeline |
| `bert_2.ipynb` | Aggregation, Talk Score, sentiment trends |
| `rag_barriers.ipynb` | Internal barrier identification via RAG |

## Analysis Tracks

### 1. Emissions Analysis
- **EDA**: Scope 1/2 trends across companies and technologies
- **Baseline model**: Technology alone explains 79% of variance (p<0.001)
- **External factors**: Carbon prices, electricity costs, policy shocks

### 2. Report Text Analysis
- **Pipeline**: Extract → Filter → Translate (Helsinki-NLP) → ClimateBERT
- **Metrics**: Volume, Specificity, Commitment, Sentiment, Net-Zero focus
- **Talk Score**: Composite metric (20% volume + 40% specificity + 40% commitment)

### 3. RAG Barrier Analysis
<!-- TODO: Add description -->

## Helper Scripts

**`monitor_gpu.py`** — Real-time GPU monitoring during BERT inference
```bash
python monitor_gpu.py  # Run in separate terminal during processing
```

## Requirements

Key dependencies (see `requirements.txt` for full list):
- Python 3.11+
- PyTorch 2.9+ with CUDA 13.0
- Transformers 4.57+
- PyMuPDF, pdfplumber (PDF extraction)
- spaCy, langid (NLP utilities)

## Team

**SuSteelAible** — December 2025

---

*For questions about specific notebooks, see the documentation within each file.*
