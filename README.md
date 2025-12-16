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
git clone git@github.com:am0ebe/SusteelAible.git
cd SusteelAible

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Extract data
unzip data.zip
```

✅ **Done!** You can now run all analysis notebooks.

## Analyze Source Reports

Want to see the original sustainability reports we analyzed?

```bash
# Download reports (~1.6GB)
wget https://github.com/am0ebe/SusteelAible/releases/download/v1.0/reports.zip
unzip reports.zip
# Reports will be extracted to `data/reports/`
```

<!-- <details>
<summary><strong>Full Pipeline Setup (optional) — Click to expand</strong></summary> -->
<details>
<summary><h2 style="display: inline; cursor: pointer;"> Full Pipeline Setup (optional) — Click to expand</h2></summary>

### 1. GPU Setup (recommended)

GPU acceleration speeds up `bert_1.ipynb` dramatically:

- **With GPU:** ~2-3 hours
- **Without GPU:** ~8+ hours (CPU fallback works, just slower)

**Installation:**

**Linux/Windows (NVIDIA GPU):**

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**macOS (Apple Silicon):**

```bash
# MPS (Metal) support is built-in, just install PyTorch
pip install torch torchvision
```

**Verify GPU detection:**

Open `bert_1.ipynb` and run the first cell:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

Expected output:

- NVIDIA GPU: `✅ CUDA GPU detected!`
- Apple Silicon: `✅ Apple Silicon (MPS) detected!`
- CPU only: `⚠️ No GPU detected - will use CPU (slower)`

### 2. Run Pipeline

Open `bert_1.ipynb` and run the second cell.
Once it finishes, run `bert_2.ipynb` to produce plots.

</details>


## Project Structure

After `unzip data.zip`:

```
├── data/
│   ├── EDA/               # Emissions data
│   └── reports/           # Source PDFs (only if running full pipeline)
├── cache/                 # Pre-processed ClimateBERT results
├── out/                   # Generated plots and visualizations
│
├── basic_EDA.ipynb        # Emissions analysis
├── baseline_model.ipynb   # Technology baseline model
├── bert_1.ipynb           # PDF processing pipeline (optional)
├── bert_2.ipynb           # Report analysis & visualization
├── rag_barriers.ipynb     # RAG barrier analysis (coming soon)
│
├── monitor_gpu.py         # GPU monitoring utility
└── requirements.txt
```

## Notebooks — What to Run

### Emissions Analysis Track

| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `basic_EDA.ipynb` | Explore emissions trends, technology lock-in (Scope 1 vs 2) |
| 2 | `baseline_model.ipynb` | Baseline model: technology explains 79% of variance |

### Text Analysis Track

| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `bert_1.ipynb` | Process PDFs → ClimateBERT *(skip if using cache)* |
| 2 | `bert_2.ipynb` | Analyze cached results: Talk Score, sentiment, trends |
| 3 | `rag_barriers.ipynb` | RAG-based barrier identification *(coming soon)* |

## Helper Scripts

**`monitor_gpu.py`** — Real-time GPU monitoring during BERT inference

```bash
python monitor_gpu.py  # Run in separate terminal during processing
```

## Project Structure

```
├── data/                  # -> after unzip data.zip
│   ├── EDA/               # Emissions data
│   └── reports/           # Source PDFs (-> only if downloaded)
├── cache/                 # Pre-processed ClimateBERT results
├── out/                   # Generated plots and visualizations
│
├── basic_EDA.ipynb        # Emissions analysis
├── baseline_model.ipynb   # Technology baseline model
├── bert_1.ipynb           # PDF processing pipeline (optional)
├── bert_2.ipynb           # Report analysis & visualization
├── rag_barriers.ipynb     # RAG barrier analysis (coming soon)
│
├── monitor_gpu.py         # GPU monitoring utility
└── requirements.txt
```

## Notebooks — What to Run

### Emissions Analysis Track

| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `basic_EDA.ipynb` | Explore emissions trends, technology lock-in (Scope 1 vs 2) |
| 2 | `baseline_model.ipynb` | Baseline model: technology explains 79% of variance |

### Text Analysis Track

| Order | Notebook | What it does |
|-------|----------|--------------|
| 1 | `bert_1.ipynb` | Process PDFs → ClimateBERT *(skip if using cache)* |
| 2 | `bert_2.ipynb` | Analyze cached results: Talk Score, sentiment, trends |
| 3 | `rag_barriers.ipynb` | RAG-based barrier identification *(coming soon)* |


## Helper Scripts

**`monitor_gpu.py`** — Real-time GPU monitoring during BERT inference

```bash
python monitor_gpu.py  # Run in separate terminal during processing
```

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependencies
- Optional: NVIDIA GPU (CUDA) or Apple Silicon for faster processing

## Team

**SuSteelAible** — December 2025

[@am0ebe](https://github.com/am0ebe) · [@calluna-borealis](https://github.com/calluna-borealis) · [contributors welcome]

## Contact

Questions, bugs, or suggestions? Feel free to [open an issue](https://github.com/am0ebe/SusteelAible/issues) or reach out to the team. We're happy to help!

---

*For questions about specific notebooks, see the documentation within each file.*
