# allows "from nlp import" in notebooks

from .preprocessing import PDFPreprocessor, preprocess_pdfs
from .bert_1 import ClimateBERTAnalyzer, analyze_reports
from .bert_2 import ClimateBERTVisualizer, visualize_results
from .gpu_utils import GPUManager, clear_gpu_memory

from .data_loader import (
    CacheLoader,
    CachedDocument,
    BERTAnalyzedDocument,
    load_prep_cache,
    load_bert_cache,

    # CSV/JSON loading utilities
    DataLoader,
    load_csv_data,
    load_json_data,
)
from .llm_extract import (
    Config,
    ExtractPipeline,
    load_pipeline
)
from .rag import RAGPipeline

from .topic_modelling import (
    TopicModeler,
    TopicModelConfig,
    run_topic_modeling_pipeline,
    merge_topics_pipeline,
    latest_run_dir,
    run_grid_search,
    aggregate_by_year,
    aggregate_by_category,
    aggregate_by_company_year,
)
from .topic_gridsearch import TopicGridSearch
