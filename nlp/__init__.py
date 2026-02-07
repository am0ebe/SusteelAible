from .preprocessing import PDFPreprocessor, preprocess_pdfs
from .bert_1 import ClimateBERTAnalyzer, analyze_reports
from .bert_2 import ClimateBERTVisualizer, visualize_results
# allows "from nlp import" in notebooks
from .gpu_utils import GPUManager, clear_gpu_memory
# JSON cache loading utilities (shared by bert_2 and rag_pipeline)
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
from .rag_1 import (
    RAGPipeline,
    RAGConfig,
    quick_start,
    analyze_token_usage,
)
# Topic modeling with BERTopic (HDBSCAN + UMAP + KeyBERTInspired)
from .rag_2 import (
    TopicModeler,
    TopicModelConfig,
    run_topic_modeling_pipeline,
    aggregate_by_year,
    aggregate_by_category,
)
