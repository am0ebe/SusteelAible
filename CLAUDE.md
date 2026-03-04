# CLAUDE.md

## Quick Context

**SuSteelAible**: NLP pipeline analyzing ~200 EU steel company sustainability reports (2013-2025) to extract decarbonization barriers & motivators.

**Pipeline**: `PDFs → Preprocessing → BERT Analysis → RAG Extraction → Topic Modeling`

**Main notebook**: `03_nlp/run_all.ipynb` - runs the full pipeline

**Package layout**: directory is `03_nlp/` on disk, imports as `from nlp import ...` via `pyproject.toml` `package-dir` mapping.

## Code Conventions

- **Style**: PEP 8, 4 spaces, long lines OK, double quotes
- **Naming**: `snake_case` functions, `PascalCase` classes, `SCREAMING_SNAKE` constants
- **Config**: Use dataclasses (`RagConfig`, `TopicModelConfig`, etc.)
- **GPU**: Always use `GPUManager` for device handling
- **Logging**: `print(f"✓ ...")` success, `print(f"⚠️ ...")` warnings
- **Git**: Don't include yourself in commit messages
- **imports**: At top of file.

## Pipeline Modules

| Module | Purpose | Config Class |
|--------|---------|--------------|
| `preprocessing.py` | PDF → chunks (600-1600 chars) | `PreprocessingConfig` |
| `bert_1.py` | ClimateBERT classification (5 models) | `BERTConfig` |
| `bert_2.py` | Visualization & CSV export | - |
| `llm_extract.py` | Exhaustive LLM extraction (`ExtractPipeline`) | `RagConfig` |
| `rag.py` | FAISS retrieval extraction (`RAGPipeline`, extends `ExtractPipeline`) | `RagConfig` |
| `topic_modelling.py` | BERTopic clustering & labeling | `TopicModelConfig` |
| `topic_gridsearch.py` | Staged embedding/HDBSCAN/UMAP grid search (`TopicGridSearch`) | `TopicModelConfig` (base) |

## LLM Extraction Pipeline

**Purpose**: Extract barriers & motivators from BERT-filtered chunks using LLM (Ollama or Groq).

**Key classes/functions**:
```python
from nlp import RagConfig, load_pipeline

# Exhaustive (all chunks batched)
config = RagConfig(
    llm_provider="groq", model="llama-3.1-8b-instant",
    ctx=128000,
    batch_size=15,                # Auto-calculated from ctx if not set
    min_detector_score=0.7,
)

# RAG (FAISS retrieval)
config = RagConfig(
    approach="rag", llm_provider="groq", model="llama-3.1-8b-instant",
    ctx=128000, top_k=20,
)

pipeline = load_pipeline(config)  # picks ExtractPipeline or RAGPipeline
pipeline.extract_all_companies()
```

**Batching**: `batch_size` auto-calculated from `ctx` if not set (capped at 32k for batch calc). Larger context window = bigger batches = fewer LLM calls.


## Data Volumes (verified from cache)

| Stage | Count |
|---|---|
| PDFs | 180 unique files across 197 company-year reports |
| Companies | 15 EU steel producers (IDs 001–016, no 011) |
| Year range | 2013–2025 |
| Chunks after preprocessing | 36 444 |
| Chunks after BERT climate filter (detector ≥ 0.70) | 14 545 (39.9 %) |
| Barriers extracted (RAG) | 1 698 |
| Motivators extracted (RAG) | 1 255 |
| Barrier topics (after merging) | 16 |
| Motivator topics (after merging) | 6 |

Chunk sizes are configured in **chars** (`PreprocessingConfig.min_chunk_chars`, `max_chunk_chars`), not tokens. Target: 600–1 400 chars ≈ 150–350 tokens (≈ 4 chars/token rule of thumb).

**BERT scores usage**: only `detector_score` is used as a pipeline filter (threshold 0.70). `specificity`, `commitment`, `sentiment`, `netzero` scores are stored in `cache/*_bert.json` as metadata but are not used to filter chunks fed to RAG.

## Cache & Data Flow

```
data/reports/*.pdf
    ↓ preprocessing.py
cache/{company}_{year}_prep.json     # Raw chunks
    ↓ bert_1.py
cache/{company}_{year}_bert.json     # + BERT scores (detector, specificity, commitment, sentiment, netzero)
    ↓ llm_extract.py / rag.py (loads bert.json)
out/barriers_*.csv, motivators_*.csv # Extracted items with metadata
    ↓ topic_modelling.py
out/topics/                          # BERTopic models, visualizations
```

## PipelineState

Cross-session config and derived state for the topic modeling notebook. Persisted to `out/topics/state.json`.

**Fields**:
- Static config (set in notebook): `data_folder`, `output_folder`, `embedding_model`, `batch_size`, `llm_provider`, `llm_model`
- Dynamic state (auto-saved by pipeline): `run_dir`, `category_overrides`

**Pattern**: all section 4 functions accept `state=` and extract what they need:
```python
state = PipelineState.load_or_create("../out/topics/state.json", embedding_model="...", ...)
gs = TopicGridSearch(state=state)
results = run_topic_modeling_pipeline(state=state)
merge_topics_pipeline(category="barriers", topics_to_merge=[...], state=state)
generate_deliverable_viz(state=state, top_n=6)
```

`load_or_create` kwargs always override saved values — notebook cell is source of truth for static config; JSON persists derived state across sessions. Manual overrides: set fields directly and call `state.save()`.

## Common Tasks

**Check what's cached**:
```python
from nlp import load_bert_cache
docs = load_bert_cache('../cache')
print(f"{len(docs)} documents loaded")
```

**Debug OOM**:
```python
gpu = GPUManager()
gpu.emergency_cleanup(models={'model': model})
```

## Workflow

- **Session start**: run `git log --oneline -10` to understand recent context before touching anything.
- **Iterative feedback loop**: implement → user reviews → correct → repeat. Don't over-build before getting feedback.
- **Flag uncertainty**: when making a bigger change, call out the parts that are guesswork or trade-offs so review effort goes where it matters.
- **Proactive suggestions**: if something in the workflow, collaboration, or codebase looks improvable, surface it — don't wait to be asked.
- **End of session**: ask "what should we write down?" before closing — decisions and rationale that only live in chat or git history don't survive to the next session.
- **CLAUDE.md is the memory that matters most**: architectural decisions, "why" behind choices, things that keep going wrong. Update it when something stable is figured out.

## Decisions & Rationale

**`03_nlp/` directory name**: renamed from `nlp/` to restore numbered project structure (`01_eda/`, `02_models/`, `03_nlp/`). Python can't import a module starting with a digit, so `pyproject.toml` maps `nlp = "03_nlp"` under `[tool.setuptools.package-dir]`. All imports remain `from nlp import ...`.

**Snowflake for FAISS retrieval** (`rag.py`): chosen based on overall MTEB aggregate score — not clustering-specific. As it turns out, high aggregate MTEB rank doesn't imply good clustering (Snowflake clustering score is actually weak). For topic modeling, a separate model is chosen via grid search.

**Topic modeling best params** (from 3-stage grid search, `run_02`): granite embedding (`ibm-granite/granite-embedding-english-r2`) for both categories. Barriers: `umap 5/15, hdbscan min_cluster_size=25, min_samples=5, eom`. Motivators: `umap 5/25, hdbscan min_cluster_size=16, min_samples=3, eom`. Stored in the manual overrides cell in `run_all.ipynb` (commented) and in `state.json` after grid search.

**Grid search → PipelineState flow**: each stage auto-saves `gs.category_overrides` to `state.json`. Stage 2 reads the winning embedding model from state (set by stage 1) instead of a hardcode. The one manual step is reviewing `gs_stage1.csv` and confirming/overriding the auto-pick before running stage 2.

**`_suggest_best` removed from `TopicGridSearch`**: was auto-locking a single winner which was misleading (tiny DBCV differences swung the pick via outlier penalty). Replaced with `_print_candidates` which prints top 3 with flags and a what-to-look-for guide. Stage 1 still auto-sets `gs.locked["embedding_model"]` to its best pick; user overrides in stage 2 cell if needed.

**`merge_topics_pipeline` includes aggregation CSV saving**: yearly and company-year CSVs are regenerated inside the pipeline function after every merge — no manual loop needed in the notebook.

**`viz()` always receives full doc arrays**: BERTopic aligns docs/embeddings with `self.topics_` by index position internally. The `topics=` parameter controls display only. Passing pre-filtered arrays caused length mismatches. Same applies to `visualize_document_datamap` which now also receives `topics=`.

**Cache embeddings, always refit BERTopic**: encoding is the slow step; BERTopic fit is fast enough that re-running it each time isn't a problem. `cache_embeddings=True` skips the slow part while keeping full flexibility to tune HDBSCAN/UMAP params without re-encoding.

**Batch size capped at 32k for calculation**: ctx=128k was causing very large batches which hit Groq rate limits almost immediately. Cap prevents this regardless of context window size.

**Gemini removed**: no free tier in EU. Don't re-add.

**Exhaustive extraction approach (llm_extract.py) kept but not primary**: original idea was high-recall extraction with chunk provenance, then dedup/frequency analysis feeding into topic modeling. Abandoned as primary path because: (a) too slow to run locally, hit Groq limits, (b) match rate was 30–70% — too variable to trust. Work is preserved but RAG is the active approach. Provenance/chunk_id was dropped when bullet format replaced the `[id]|||text` parser.

**BERT filtering before LLM** (`min_detector_score=0.7`): keeps only climate-relevant chunks, cutting the corpus significantly before hitting the LLM. Threshold of 0.7 is the ClimateBERT default — not tuned here. Going higher (0.8–0.9) removes very few additional chunks so wasn't pursued. Some non-decarbonization climate chunks slip through; accepted as good-enough pre-filter. Other BERT scores (specificity, commitment) could further filter but haven't been applied yet.

**RAG template extraction and topic modeling**: RAG LLM outputs near-identical sentence shells varying only one slot word (e.g. "Complexity of integrating [biomass/MSW/industrial waste] with existing energy systems"). Because sentence transformers encode full sentence meaning, template structure dominates over the slot word — embeddings cluster correctly by theme but appear as artificial density. Result: BERTopic may split one real topic into several near-duplicate fragments. Mitigation: merge these post-hoc. Not a modeling bug — the sentences genuinely mean the same thing.

**Documentation artifacts**: `results/workflow.md` contains full pipeline technical specs in tables (corpus, preprocessing, BERT, RAG, topic modeling). `results/workflow.drawio` is a draw.io diagram mimicking the bla.png academic style (colored dashed section borders, numbered badges, parallelogram data nodes). Open at app.diagrams.net.

**Topic labeling flow**: BERTopic c-TF-IDF produces keywords (statistical, no LLM). Keywords are fed to LLM in one batched call → LLM returns "N: Label" lines → labels stored in `labels.csv`. The `keywords` column in the CSV is the raw pre-filter keywords; `KEYWORD_STOPWORDS` filters them before the LLM sees them (and also removes them from the CountVectorizer vocabulary).

## Engineering Rules

- Do NOT refactor unrelated code
- Do NOT rename public functions unless instructed
- Do NOT write overly defensive code.
- Do NOT split lines over many lines
- Preserve existing behavior unless explicitly told otherwise
- Prefer simple inline logic over abstractions
- Duplication only OK if it improves clarity
- After task completion: Review changes against this document and list + fix any violations before finalizing
