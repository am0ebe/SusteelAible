from .preprocessing import PDFPreprocessor, preprocess_pdfs
from .bert_1 import ClimateBERTAnalyzer, analyze_reports
from .bert_2 import ClimateBERTVisualizer, visualize_results
# allows "from nlp import" in notebooks

from .utils import GPUManager

# JSON cache loading utilities (shared by bert_2 and rag_pipeline)
from .json_loader import (
    CacheLoader,
    CachedDocument,
    BERTAnalyzedDocument,
    load_prep_cache,
    load_bert_cache,
)
