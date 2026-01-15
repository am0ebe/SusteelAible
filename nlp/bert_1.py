"""
ClimateBERT Analysis Pipeline
=============================

Analyzes PDF reports using ClimateBERT models for climate-related text classification.

Pipeline: Preprocess (via preprocessing.py) → Filter → Analyze → Save

Usage:
    from bert_1 import ClimateBERTAnalyzer, analyze_reports

    # Quick start
    stats = analyze_reports("../data/reports")

    # Detailed usage
    analyzer = ClimateBERTAnalyzer()
    analyzer.set_pdf_path("report.pdf")
    result = analyzer.run_full_pipeline()
"""

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm
from transformers import pipeline

from nlp.gpu_utils import GPUManager
from nlp.preprocessing import PDFPreprocessor, PreprocessingConfig, ProcessedDocument
from nlp.json_loader import CacheLoader

# Suppress all the noisy warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# CONFIGURATION
# =============================================================================

CLIMATEBERT_MODELS = {
    "detector": "climatebert/distilroberta-base-climate-detector",
    "specificity": "climatebert/distilroberta-base-climate-specificity",
    "sentiment": "climatebert/distilroberta-base-climate-sentiment",
    "commitment": "climatebert/distilroberta-base-climate-commitment",
    "netzero": "climatebert/netzero-reduction",
}


@dataclass
class BERTConfig:
    """Configuration for ClimateBERT analysis pipeline."""

    cache_dir: str = "cache"
    batch_size: int = 32
    translate_to_english: bool = True
    verbose: bool = False

    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Create PreprocessingConfig for BERT requirements."""
        return PreprocessingConfig(
            min_chunk_chars=600,
            max_chunk_chars=1600,
            detect_language=True,
        )


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class ClimateBERTAnalyzer:
    """
    Analyzes PDF reports through ClimateBERT pipeline.

    Pipeline: Preprocess → Filter (climate detection) → Analyze (4 models)
    """

    def __init__(self, config: Optional[BERTConfig] = None, quiet: bool = False):
        self.config = config or BERTConfig()
        self.quiet = quiet

        # Setup directories
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache loader for file operations
        self.cache_loader = CacheLoader(self.config.cache_dir)

        # Initialize components (quiet mode for preprocessing to avoid extra prints)
        self.gpu = GPUManager()
        self.preprocessor = PDFPreprocessor(
            self.config.get_preprocessing_config(),
            quiet=quiet
        )
        self.models: Dict = {}

        # Current document state
        self.pdf_path: Optional[Path] = None
        self.prep_cache: Optional[Dict] = None
        self.bert_cache: Optional[Dict] = None

        if not quiet:
            print(f"✓ {self.gpu}")

    def _log(self, msg: str):
        if self.config.verbose and not self.quiet:
            print(msg)

    # -------------------------------------------------------------------------
    # Caching (using CacheLoader)
    # -------------------------------------------------------------------------

    def _get_cache_path(self, suffix: str) -> Path:
        """Get cache file path for current PDF."""
        if not self.pdf_path:
            raise ValueError("No PDF path set")
        return self.cache_loader.get_cache_path(self.pdf_path.stem, suffix)

    def _load_cache(self, suffix: str) -> Optional[Dict]:
        """Load cache file if exists."""
        if not self.pdf_path:
            raise ValueError("No PDF path set")
        self._log(f"  Loading cache: {self.pdf_path.stem}_{suffix}.json")
        return self.cache_loader.load_single_cache(self.pdf_path.stem, suffix)

    def _save_cache(self, suffix: str, data: Dict):
        """Save data to cache file."""
        if not self.pdf_path:
            raise ValueError("No PDF path set")
        self.cache_loader.save_single_cache(self.pdf_path.stem, suffix, data)

    # -------------------------------------------------------------------------
    # PDF Processing (delegates to preprocessing module)
    # -------------------------------------------------------------------------

    def set_pdf_path(self, pdf_path: str):
        """Set current PDF and reset caches."""
        new_path = Path(pdf_path)

        # Validate path
        if not new_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if new_path.is_dir():
            raise ValueError(
                f"Expected PDF file, got directory: {pdf_path}. Use process_pdfs() for folders."
            )
        if new_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        if self.pdf_path != new_path:
            self.pdf_path = new_path
            self.prep_cache = None
            self.bert_cache = None

    def preprocess_pdf(
        self, step_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict:
        """
        Extract, clean, chunk, and translate PDF using preprocessing module.

        Args:
            step_callback: Optional callback function(step_name, step_num) for progress updates

        Returns cached data if available.
        """
        # Return cached if available
        if self.prep_cache:
            return self.prep_cache

        # Try loading from file cache
        cached = self._load_cache("prep")
        if cached:
            self.prep_cache = cached
            return cached

        # Process PDF using shared preprocessor (with step callback)
        doc: ProcessedDocument = self.preprocessor.process_pdf(
            self.pdf_path,
            chunk_method="semantic",
            translate_to_english=self.config.translate_to_english,
            show_progress=False,  # Disable internal progress bar
            step_callback=step_callback,
        )

        # Build chunk IDs
        chunk_ids = [
            f"{doc.company_id}_{idx:03d}" if doc.company_id else f"chunk_{idx:03d}"
            for idx in range(len(doc.chunks))
        ]

        # Build chunk pairs (original + translated)
        if doc.translated and doc.original_chunks:
            chunk_pairs = [
                {"original": orig, "translated": trans}
                for orig, trans in zip(doc.original_chunks, doc.chunks)
            ]
        else:
            chunk_pairs = [{"original": chunk, "translated": chunk}
                           for chunk in doc.chunks]

        # Create cache data
        self.prep_cache = {
            "pdf_path": str(self.pdf_path),
            "company": doc.company_name,
            "company_id": doc.company_id,
            "year": doc.year,
            "language": doc.language,
            "translated": doc.translated,
            "extraction_method": doc.extraction_method,
            "num_pages": doc.num_pages,
            "num_chunks": len(doc.chunks),
            "chunks": doc.chunks,
            "chunk_ids": chunk_ids,
            "chunk_pairs": chunk_pairs,
            "processed_at": datetime.now().isoformat(),
        }

        self._save_cache("prep", self.prep_cache)
        return self.prep_cache

    # -------------------------------------------------------------------------
    # Model Management
    # -------------------------------------------------------------------------

    def _load_model(self, model_key: str):
        """Load a ClimateBERT model."""
        if model_key in self.models:
            return self.models[model_key]

        self._log(f"  Loading {model_key} model...")
        self.models[model_key] = pipeline(
            "text-classification",
            model=CLIMATEBERT_MODELS[model_key],
            device=self.gpu.device_id,
            batch_size=self.config.batch_size,
        )
        return self.models[model_key]

    def _unload_model(self, model_key: str):
        """Unload a model to free memory."""
        if model_key in self.models:
            del self.models[model_key]
            self.gpu.clear()

    def _unload_all_models(self):
        """Unload all models."""
        for key in list(self.models.keys()):
            del self.models[key]
        self.models = {}
        self.gpu.clear()

    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------

    def _run_model_on_chunks(
        self,
        chunks: List[Dict],
        model_key: str,
        label_field: str,
        score_field: str,
    ):
        """Run a model on chunks and update them with results."""
        model = self._load_model(model_key)
        texts = [c["text"] for c in chunks]

        # Track truncation for warning
        truncation_count = 0

        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i: i + self.config.batch_size]
            batch_chunks = chunks[i: i + self.config.batch_size]

            try:
                results = model(batch_texts, truncation=True, max_length=512)
                for chunk, result, text in zip(batch_chunks, results, batch_texts):
                    chunk[label_field] = result["label"]
                    chunk[score_field] = result["score"]
                    # Check if text was likely truncated (rough estimate: ~4 chars per token)
                    if len(text) > 512 * 4:
                        truncation_count += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.gpu.clear()
                else:
                    raise

        # Print warning if truncation occurred
        if truncation_count > 0 and not self.quiet:
            print(
                f"  ⚠️ {truncation_count} chunks exceeded 512 tokens (truncated by model)")

    def filter_climate_chunks(self) -> Dict:
        """Filter chunks to keep only climate-related content."""
        # Check cache
        if self.bert_cache and self.bert_cache.get("filtered"):
            return self.bert_cache

        cached = self._load_cache("bert")
        if cached and cached.get("filtered"):
            self.bert_cache = cached
            return cached

        # Ensure preprocessing is done
        if not self.prep_cache:
            self.preprocess_pdf()

        chunks = self.prep_cache["chunks"]
        chunk_ids = self.prep_cache.get(
            "chunk_ids", [f"chunk_{i:03d}" for i in range(len(chunks))]
        )

        # Run climate detector
        detector = self._load_model("detector")
        climate_chunks = []

        for i in range(0, len(chunks), self.config.batch_size):
            batch_texts = chunks[i: i + self.config.batch_size]
            batch_ids = chunk_ids[i: i + self.config.batch_size]

            try:
                results = detector(
                    batch_texts, truncation=True, max_length=512)
                for text, chunk_id, result in zip(batch_texts, batch_ids, results):
                    if result["label"].lower() in ["climate", "yes", "climate-related"]:
                        climate_chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "company": self.prep_cache.get("company"),
                                "company_id": self.prep_cache.get("company_id"),
                                "year": self.prep_cache.get("year"),
                                "text": text,
                                "detector_label": result["label"],
                                "detector_score": result["score"],
                            }
                        )
            except RuntimeError:
                self.gpu.clear()

        # Calculate stats
        kept_pct = len(climate_chunks) / len(chunks) * 100 if chunks else 0

        self.bert_cache = {
            "pdf_path": str(self.pdf_path),
            "company": self.prep_cache.get("company"),
            "company_id": self.prep_cache.get("company_id"),
            "year": self.prep_cache.get("year"),
            "language": self.prep_cache.get("language"),
            "translated": self.prep_cache.get("translated"),
            "filtered": True,
            "filtered_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "climate_chunks": len(climate_chunks),
            "kept_percentage": kept_pct,
            "chunks": climate_chunks,
        }

        self._save_cache("bert", self.bert_cache)
        return self.bert_cache

    def analyze_specificity(self) -> Dict:
        """Analyze climate chunks for specificity."""
        if self.bert_cache and self.bert_cache.get("specificity_analyzed"):
            return self.bert_cache

        if not self.bert_cache or not self.bert_cache.get("filtered"):
            self.filter_climate_chunks()

        self._unload_model("detector")
        self._run_model_on_chunks(
            self.bert_cache["chunks"],
            "specificity",
            "specificity_label",
            "specificity_score",
        )

        self.bert_cache["specificity_analyzed"] = True
        self.bert_cache["specificity_analyzed_at"] = datetime.now().isoformat()
        self._save_cache("bert", self.bert_cache)
        return self.bert_cache

    def analyze_sentiment(self) -> Dict:
        """Analyze climate chunks for sentiment."""
        if self.bert_cache and self.bert_cache.get("sentiment_analyzed"):
            return self.bert_cache

        if not self.bert_cache or not self.bert_cache.get("specificity_analyzed"):
            self.analyze_specificity()

        self._unload_model("specificity")
        self._run_model_on_chunks(
            self.bert_cache["chunks"], "sentiment", "sentiment_label", "sentiment_score"
        )

        self.bert_cache["sentiment_analyzed"] = True
        self.bert_cache["sentiment_analyzed_at"] = datetime.now().isoformat()
        self._save_cache("bert", self.bert_cache)
        return self.bert_cache

    def analyze_commitment(self) -> Dict:
        """Analyze climate chunks for commitment statements."""
        if self.bert_cache and self.bert_cache.get("commitment_analyzed"):
            return self.bert_cache

        if not self.bert_cache or not self.bert_cache.get("sentiment_analyzed"):
            self.analyze_sentiment()

        self._unload_model("sentiment")
        self._run_model_on_chunks(
            self.bert_cache["chunks"],
            "commitment",
            "commitment_label",
            "commitment_score",
        )

        self.bert_cache["commitment_analyzed"] = True
        self.bert_cache["commitment_analyzed_at"] = datetime.now().isoformat()
        self._save_cache("bert", self.bert_cache)
        return self.bert_cache

    def analyze_netzero(self) -> Dict:
        """Analyze climate chunks for net-zero reduction mentions."""
        if self.bert_cache and self.bert_cache.get("netzero_analyzed"):
            return self.bert_cache

        if not self.bert_cache or not self.bert_cache.get("commitment_analyzed"):
            self.analyze_commitment()

        chunks = self.bert_cache["chunks"]
        if not chunks:
            self.bert_cache["netzero_analyzed"] = True
            self.bert_cache["netzero_count"] = 0
            self.bert_cache["netzero_pct"] = 0
            self._save_cache("bert", self.bert_cache)
            return self.bert_cache

        self._unload_model("commitment")
        self._run_model_on_chunks(
            chunks, "netzero", "netzero_label", "netzero_score")

        # Calculate net-zero stats
        netzero_chunks = [c for c in chunks if c.get(
            "netzero_label") == "reduction"]

        self.bert_cache["netzero_analyzed"] = True
        self.bert_cache["netzero_analyzed_at"] = datetime.now().isoformat()
        self.bert_cache["netzero_count"] = len(netzero_chunks)
        self.bert_cache["netzero_pct"] = (
            len(netzero_chunks) / len(chunks) * 100 if chunks else 0
        )

        self._save_cache("bert", self.bert_cache)
        return self.bert_cache

    def run_full_pipeline(self) -> Dict:
        """Run complete analysis pipeline on current PDF."""
        self.preprocess_pdf()
        self.filter_climate_chunks()
        self.analyze_specificity()
        self.analyze_sentiment()
        self.analyze_commitment()
        self.analyze_netzero()
        self._unload_all_models()
        return self.bert_cache

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    def process_pdfs(self, path: str, skip_errors: bool = True) -> Dict:
        """
        Process single PDF or directory of PDFs.

        Args:
            path: Path to PDF file or directory
            skip_errors: Continue on errors

        Returns:
            Stats dictionary with processing summary
        """
        target = Path(path)

        # Validate path
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Get PDF files
        if target.is_file():
            if target.suffix.lower() != ".pdf":
                raise ValueError(f"Not a PDF file: {path}")
            pdf_files = [target]
            root = target.parent
        else:
            pdf_files = sorted(target.rglob("*.pdf"))
            root = target
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in: {path}")

        # Print header
        print(f"\n{'='*60}")
        print("CLIMATEBERT ANALYZER")
        print(f"{'='*60}")
        print(f"  PDFs: {len(pdf_files)}")
        print(f"  Cache: {self.cache_dir}")
        print(f"  {self.gpu}")
        print(
            f"  Translation: {'enabled' if self.config.translate_to_english else 'disabled'}")
        print(f"{'='*60}\n")

        # Stats tracking
        stats = {
            "total": len(pdf_files),
            "processed": 0,
            "cached": 0,
            "translated": 0,
            "errors": [],
            "start_time": time.time(),
        }

        # Steps definition
        PREP_STEPS = ["Extract", "Clean", "Chunk", "Lang", "Translate"]
        BERT_STEPS = ["Filter", "Specific", "Sentiment", "Commit", "NetZero"]
        TOTAL_STEPS = len(PREP_STEPS) + len(BERT_STEPS)

        # Single progress bar - more reliable across terminals
        pbar = tqdm(
            total=len(pdf_files),
            desc="Processing",
            unit="pdf",
            ncols=80,
        )

        def update_step(step_num: int, step_name: str, filename: str):
            """Update progress bar description with current step."""
            # Truncate filename
            max_len = 25
            if len(filename) > max_len:
                short = "..." + filename[-(max_len - 3):]
            else:
                short = filename
            pbar.set_description(
                f"{short} [{step_num}/{TOTAL_STEPS} {step_name}]")

        for idx, pdf_file in enumerate(pdf_files):
            display_name = str(pdf_file.relative_to(root))

            try:
                self.set_pdf_path(str(pdf_file))

                # Check if already fully processed (BERT cache complete)
                bert_cache_path = self._get_cache_path("bert")
                if bert_cache_path.exists():
                    cached = self._load_cache("bert")
                    if cached and cached.get("netzero_analyzed"):
                        stats["cached"] += 1
                        stats["processed"] += 1
                        if cached.get("translated"):
                            stats["translated"] += 1
                        update_step(TOTAL_STEPS, "Cached", display_name)
                        pbar.update(1)
                        continue

                # Check if preprocessing is cached
                prep_cache_path = self._get_cache_path("prep")
                if prep_cache_path.exists():
                    self.prep_cache = self._load_cache("prep")
                    update_step(len(PREP_STEPS), "PrepOK", display_name)
                    if self.prep_cache.get("translated"):
                        stats["translated"] += 1
                else:
                    # Run preprocessing with step callback
                    def prep_callback(step_name: str, step_num: int):
                        update_step(step_num, step_name, display_name)

                    self.preprocess_pdf(step_callback=prep_callback)
                    if self.prep_cache.get("translated"):
                        stats["translated"] += 1

                # BERT analysis steps
                offset = len(PREP_STEPS)

                update_step(offset + 1, "Filter", display_name)
                self.filter_climate_chunks()

                update_step(offset + 2, "Specific", display_name)
                self.analyze_specificity()

                update_step(offset + 3, "Sentiment", display_name)
                self.analyze_sentiment()

                update_step(offset + 4, "Commit", display_name)
                self.analyze_commitment()

                update_step(offset + 5, "NetZero", display_name)
                self.analyze_netzero()

                stats["processed"] += 1
                update_step(TOTAL_STEPS, "Done", display_name)
                self._unload_all_models()

            except Exception as e:
                stats["errors"].append(
                    {"file": str(pdf_file), "error": str(e)})
                update_step(0, "ERROR", display_name)
                if not skip_errors:
                    pbar.close()
                    raise
                self.gpu.emergency_cleanup(self.models)
                self.models = {}

            pbar.update(1)

        pbar.close()

        # Final stats
        stats["elapsed_time"] = time.time() - stats["start_time"]
        stats["avg_time"] = stats["elapsed_time"] / \
            len(pdf_files) if pdf_files else 0

        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"  Processed: {stats['processed']}/{stats['total']}")
        print(f"  From cache: {stats['cached']}")
        print(f"  Translated: {stats['translated']}")
        print(f"  Errors: {len(stats['errors'])}")
        print(
            f"  Time: {stats['elapsed_time']/60:.1f} min ({stats['avg_time']:.1f}s/pdf)")
        print(f"{'='*60}\n")

        if stats["errors"]:
            print("Errors:")
            for err in stats["errors"][:5]:
                print(f"  • {err['file']}: {err['error'][:50]}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more")

        return stats

    def cleanup(self):
        """Release all resources."""
        self._unload_all_models()
        self.gpu.emergency_cleanup()
        self.preprocessor.cleanup()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def analyze_reports(
    path: str,
    cache_dir: str = "cache",
    translate: bool = True,
    skip_errors: bool = True,
) -> Dict:
    """
    Convenience function to analyze PDF reports.

    Args:
        path: Path to PDF or directory
        cache_dir: Cache directory
        translate: Translate non-English documents
        skip_errors: Continue on errors

    Returns:
        Processing statistics
    """
    config = BERTConfig(cache_dir=cache_dir, translate_to_english=translate)
    analyzer = ClimateBERTAnalyzer(config, quiet=True)
    try:
        return analyzer.process_pdfs(path, skip_errors=skip_errors)
    finally:
        analyzer.cleanup()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("ClimateBERT Analysis Pipeline")
    print("=" * 60)
    print("\nUsage:")
    print("  from bert_1 import ClimateBERTAnalyzer, analyze_reports")
    print()
    print("  # Process folder")
    print("  stats = analyze_reports('data/reports/Baosteel')")
    print()
    print("  # Single file")
    print("  analyzer = ClimateBERTAnalyzer()")
    print("  analyzer.set_pdf_path('report.pdf')")
    print("  result = analyzer.run_full_pipeline()")
