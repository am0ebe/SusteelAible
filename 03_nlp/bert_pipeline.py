"""
ClimateBERT Analysis Pipeline
=============================

Processes PDF reports through ClimateBERT models for climate-related text analysis.

Pipeline: Extract → Translate (via preprocessing) → Filter → Analyze

Output: JSON files with scored climate chunks

Usage:
    from bert_pipeline import ClimateBERTAnalyzer

    analyzer = ClimateBERTAnalyzer()
    stats = analyzer.process_pdfs("../data/reports")

    # Or single file with detailed output
    analyzer.set_pdf_path("report.pdf")
    result = analyzer.run_full_pipeline()
"""

import os
import gc
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from tqdm.auto import tqdm
from transformers import pipeline

from preprocessing import (
    PDFPreprocessor,
    PreprocessingConfig,
    ProcessedDocument,
    get_device,
    get_gpu_info,
    clear_gpu_memory,
)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BERTConfig:
    """Configuration for ClimateBERT analysis pipeline."""

    # Paths
    cache_dir: str = "cache"
    reports_folder: str = "../data/reports"

    # Processing
    batch_size: int = 32

    # Translation (now handled by preprocessing)
    translate_to_english: bool = True

    # GPU
    use_torch_compile: bool = True

    # Output
    verbose: bool = False

    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Create PreprocessingConfig matching BERT requirements."""
        return PreprocessingConfig(
            min_chunk_chars=600,
            max_chunk_chars=1600,
            detect_language=True,
        )


# ClimateBERT model names
CLIMATEBERT_MODELS = {
    'detector': 'climatebert/distilroberta-base-climate-detector',
    'specificity': 'climatebert/distilroberta-base-climate-specificity',
    'sentiment': 'climatebert/distilroberta-base-climate-sentiment',
    'commitment': 'climatebert/distilroberta-base-climate-commitment',
    'netzero': 'climatebert/netzero-reduction',
}


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class ClimateBERTAnalyzer:
    """
    Processes PDF reports through ClimateBERT pipeline.

    Pipeline: Extract → Translate (shared) → Filter → Analyze
    """

    def __init__(self, config: Optional[BERTConfig] = None):
        self.config = config or BERTConfig()

        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.preprocessor = PDFPreprocessor(
            self.config.get_preprocessing_config())

        self.pdf_path: Optional[Path] = None
        self.prep_data: Optional[Dict] = None
        self.bert_data: Optional[Dict] = None

        self.device = get_device()
        self.device_id = 0 if self.device.type == "cuda" else -1
        self.models: Dict = {}

        gpu_info = get_gpu_info()
        if gpu_info:
            print(
                f"✓ GPU: {gpu_info['name']} ({gpu_info['total_memory_gb']:.1f}GB)")
            self._clear_gpu_memory()
        else:
            print("✓ Running on CPU")

    def _log(self, msg: str):
        if self.config.verbose:
            print(msg)

    # -------------------------------------------------------------------------
    # GPU Memory Management
    # -------------------------------------------------------------------------

    def _clear_gpu_memory(self):
        clear_gpu_memory()

    def _emergency_gpu_cleanup(self):
        if self.device.type != "cuda":
            return

        for name in list(self.models.keys()):
            try:
                self.models[name].model.cpu()
                del self.models[name]
            except:
                pass
        self.models = {}

        gc.collect()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    obj.data = obj.data.cpu()
            except:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _get_cache_path(self, suffix: str) -> Path:
        if not self.pdf_path:
            raise ValueError("No PDF path set")
        return self.cache_dir / f"{self.pdf_path.stem}_{suffix}.json"

    def _load_prep_cache(self) -> Optional[Dict]:
        if self.prep_data is not None:
            return self.prep_data
        cache_file = self._get_cache_path('prep')
        if cache_file.exists():
            self._log(f"  Loading cached prep: {cache_file.name}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.prep_data = json.load(f)
            return self.prep_data
        return None

    def _save_prep_cache(self):
        if self.prep_data is None:
            return
        cache_file = self._get_cache_path('prep')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.prep_data, f, ensure_ascii=False, indent=2)

    def _load_bert_cache(self) -> Optional[Dict]:
        if self.bert_data is not None:
            return self.bert_data
        cache_file = self._get_cache_path('bert')
        if cache_file.exists():
            self._log(f"  Loading cached BERT: {cache_file.name}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.bert_data = json.load(f)
            return self.bert_data
        return None

    def _save_bert_cache(self):
        if self.bert_data is None:
            return
        cache_file = self._get_cache_path('bert')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.bert_data, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # PDF Processing (using shared preprocessing with translation)
    # -------------------------------------------------------------------------

    def set_pdf_path(self, pdf_path: str):
        new_path = Path(pdf_path)
        if self.pdf_path != new_path:
            self.pdf_path = new_path
            self.prep_data = None
            self.bert_data = None

    def extract_and_translate_pdf(self) -> Dict:
        """Extract PDF and translate if needed (using shared preprocessor)."""
        if self.prep_data is not None:
            return self.prep_data

        cached = self._load_prep_cache()
        if cached:
            return cached

        # Use shared preprocessor with translation
        doc = self.preprocessor.process_pdf(
            self.pdf_path,
            chunk_method="semantic",
            translate_to_english=self.config.translate_to_english,
        )

        chunk_ids = [
            f"{doc.company_id}_{idx:03d}" if doc.company_id else f"chunk_{idx:03d}"
            for idx in range(len(doc.chunks))
        ]

        # Build chunk pairs for compatibility
        if doc.translated and doc.original_chunks:
            chunk_pairs = [
                {"original": o, "translated": t}
                for o, t in zip(doc.original_chunks, doc.chunks)
            ]
        else:
            chunk_pairs = [
                {"original": c, "translated": c}
                for c in doc.chunks
            ]

        self.prep_data = {
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
            "extracted_at": str(datetime.now())
        }

        self._save_prep_cache()
        return self.prep_data

    # -------------------------------------------------------------------------
    # ClimateBERT Analysis Methods
    # -------------------------------------------------------------------------

    def load_model(self, model_key: str):
        if model_key in self.models:
            return self.models[model_key]

        if self.models:
            self._clear_gpu_memory()

        model_name = CLIMATEBERT_MODELS[model_key]
        self._log(f"  Loading {model_key} model...")

        self.models[model_key] = pipeline(
            "text-classification",
            model=model_name,
            device=self.device_id,
            batch_size=self.config.batch_size
        )
        return self.models[model_key]

    def _unload_model(self, model_key: str):
        if model_key in self.models:
            del self.models[model_key]
            self._clear_gpu_memory()

    def filter_climate_chunks(self) -> Dict:
        """Filter chunks to keep only climate-related content."""
        if self.bert_data and self.bert_data.get('filtered'):
            return self.bert_data

        cached = self._load_bert_cache()
        if cached and cached.get('filtered'):
            return cached

        if not self.prep_data:
            self.prep_data = self._load_prep_cache()
            if not self.prep_data:
                raise FileNotFoundError(
                    "Run extract_and_translate_pdf() first")

        chunks = self.prep_data['chunks']
        chunk_ids = self.prep_data.get(
            'chunk_ids', [f"chunk_{i:03d}" for i in range(len(chunks))])
        source = "translated" if self.prep_data.get(
            'translated') else "original"

        self._clear_gpu_memory()

        detector = self.load_model('detector')
        climate_chunks = []

        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            batch_ids = chunk_ids[i:i + self.config.batch_size]
            try:
                results = detector(batch, truncation=True, max_length=512)
                for chunk, chunk_id, result in zip(batch, batch_ids, results):
                    if result['label'].lower() in ['climate', 'yes', 'climate-related']:
                        climate_chunks.append({
                            'chunk_id': chunk_id,
                            'company': self.prep_data.get('company'),
                            'company_id': self.prep_data.get('company_id'),
                            'year': self.prep_data.get('year'),
                            'text': chunk,
                            'detector_score': result['score'],
                            'detector_label': result['label']
                        })
            except:
                self._clear_gpu_memory()

        kept_pct = len(climate_chunks) / len(chunks) * 100 if chunks else 0

        self.bert_data = {
            'pdf_path': str(self.pdf_path),
            'company': self.prep_data.get('company'),
            'company_id': self.prep_data.get('company_id'),
            'year': self.prep_data.get('year'),
            'language': self.prep_data.get('language', 'unknown'),
            'source': source,
            'filtered': True,
            'filtered_at': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'climate_chunks': len(climate_chunks),
            'kept_percentage': kept_pct,
            'chunks': climate_chunks
        }

        self._save_bert_cache()
        return self.bert_data

    def analyze_specificity(self) -> Dict:
        """Analyze climate chunks for specificity."""
        if self.bert_data and self.bert_data.get('specificity_analyzed'):
            return self.bert_data
        if not self.bert_data:
            self.bert_data = self._load_bert_cache()
        if not self.bert_data:
            self.filter_climate_chunks()

        chunks = self.bert_data['chunks']
        self._unload_model('detector')

        specificity = self.load_model('specificity')

        for i in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[i:i + self.config.batch_size]
            batch_texts = [c['text'] for c in batch_chunks]
            try:
                results = specificity(
                    batch_texts, truncation=True, max_length=512)
                for chunk, result in zip(batch_chunks, results):
                    chunk['specificity_label'] = result['label']
                    chunk['specificity_score'] = result['score']
            except:
                self._clear_gpu_memory()

        self.bert_data['specificity_analyzed'] = True
        self.bert_data['specificity_analyzed_at'] = datetime.now().isoformat()
        self._save_bert_cache()
        return self.bert_data

    def analyze_sentiment(self) -> Dict:
        """Analyze climate chunks for sentiment."""
        if self.bert_data and self.bert_data.get('sentiment_analyzed'):
            return self.bert_data
        if not self.bert_data:
            self.bert_data = self._load_bert_cache()
        if not self.bert_data or not self.bert_data.get('specificity_analyzed'):
            self.analyze_specificity()

        chunks = self.bert_data['chunks']
        self._unload_model('specificity')

        sentiment = self.load_model('sentiment')

        for i in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[i:i + self.config.batch_size]
            batch_texts = [c['text'] for c in batch_chunks]
            try:
                results = sentiment(
                    batch_texts, truncation=True, max_length=512)
                for chunk, result in zip(batch_chunks, results):
                    chunk['sentiment_label'] = result['label']
                    chunk['sentiment_score'] = result['score']
            except:
                self._clear_gpu_memory()

        self.bert_data['sentiment_analyzed'] = True
        self.bert_data['sentiment_analyzed_at'] = datetime.now().isoformat()
        self._save_bert_cache()
        return self.bert_data

    def analyze_commitments(self) -> Dict:
        """Analyze climate chunks for commitment statements."""
        if self.bert_data and self.bert_data.get('commitment_analyzed'):
            return self.bert_data
        if not self.bert_data:
            self.bert_data = self._load_bert_cache()
        if not self.bert_data or not self.bert_data.get('sentiment_analyzed'):
            self.analyze_sentiment()

        chunks = self.bert_data['chunks']
        self._unload_model('sentiment')

        commitment = self.load_model('commitment')

        for i in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[i:i + self.config.batch_size]
            batch_texts = [c['text'] for c in batch_chunks]
            try:
                results = commitment(
                    batch_texts, truncation=True, max_length=512)
                for chunk, result in zip(batch_chunks, results):
                    chunk['commitment_label'] = result['label']
                    chunk['commitment_score'] = result['score']
            except:
                self._clear_gpu_memory()

        self.bert_data['commitment_analyzed'] = True
        self.bert_data['commitment_analyzed_at'] = datetime.now().isoformat()
        self._save_bert_cache()
        return self.bert_data

    def analyze_netzero(self) -> Dict:
        """Analyze climate chunks for net-zero reduction mentions."""
        if self.bert_data and self.bert_data.get('netzero_analyzed'):
            return self.bert_data
        if not self.bert_data:
            self.bert_data = self._load_bert_cache()
        if not self.bert_data or not self.bert_data.get('commitment_analyzed'):
            self.analyze_commitments()

        chunks = self.bert_data['chunks']
        if not chunks:
            self.bert_data['netzero_analyzed'] = True
            self.bert_data['netzero_count'] = 0
            self.bert_data['netzero_pct'] = 0
            self._save_bert_cache()
            return self.bert_data

        self._unload_model('commitment')

        netzero = self.load_model('netzero')

        for i in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[i:i + self.config.batch_size]
            batch_texts = [c['text'] for c in batch_chunks]
            try:
                results = netzero(batch_texts, truncation=True, max_length=512)
                for chunk, result in zip(batch_chunks, results):
                    chunk['netzero_label'] = result['label']
                    chunk['netzero_score'] = result['score']
            except Exception as e:
                self._log(f"  ⚠ Netzero batch error: {e}")
                self._clear_gpu_memory()

        n0_chunks = [c for c in chunks if c.get(
            'netzero_label') == 'reduction']

        self.bert_data['netzero_analyzed'] = True
        self.bert_data['netzero_analyzed_at'] = datetime.now().isoformat()
        self.bert_data['netzero_count'] = len(n0_chunks)
        self.bert_data['netzero_pct'] = len(
            n0_chunks) / len(chunks) * 100 if chunks else 0
        self._save_bert_cache()
        return self.bert_data

    def run_full_pipeline(self) -> Dict:
        """Run complete analysis pipeline on current PDF."""
        self.extract_and_translate_pdf()
        self.filter_climate_chunks()
        self.analyze_specificity()
        self.analyze_sentiment()
        self.analyze_commitments()
        self.analyze_netzero()
        return self.bert_data

    def get_results(self) -> Optional[Dict]:
        if not self.bert_data:
            self.bert_data = self._load_bert_cache()
        if not self.bert_data:
            print(f"❌ No analysis found")
            return None
        required = ['filtered', 'specificity_analyzed',
                    'sentiment_analyzed', 'commitment_analyzed']
        missing = [r for r in required if not self.bert_data.get(r)]
        if missing:
            print(f"⚠ Incomplete: {', '.join(missing)}")
        return self.bert_data

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    def process_pdfs(self, path: str, skip_errors: bool = True) -> Dict:
        """
        Process single PDF or directory of PDFs.

        Args:
            path: Path to PDF file or directory
            skip_errors: Continue on errors (default: True)

        Returns:
            Stats dictionary with processing summary
        """
        target = Path(path)
        if not target.exists():
            print(f"❌ Path not found: {path}")
            return None

        if target.is_file():
            if target.suffix.lower() != '.pdf':
                print(f"❌ Not a PDF: {path}")
                return None
            pdf_files = [target]
            root = target.parent
        else:
            pdf_files = sorted(target.rglob("*.pdf"))
            root = target

        if not pdf_files:
            print(f"❌ No PDFs found")
            return None

        n_pdfs = len(pdf_files)

        print(f"\n{'='*60}")
        print(f"CLIMATEBERT ANALYZER")
        print(f"{'='*60}")
        print(f"  PDFs to process: {n_pdfs}")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  GPU: {'Yes' if self.device.type == 'cuda' else 'No (CPU)'}")
        print(
            f"  Translation: {'enabled' if self.config.translate_to_english else 'disabled'}")
        print(f"{'='*60}\n")

        stats = {
            'total': n_pdfs, 'extracted': 0, 'translated': 0,
            'filtered': 0, 'analyzed': 0, 'skipped': 0, 'errors': [],
            'start_time': time.time()
        }

        STEPS = ['Extract', 'Filter', 'Specificity',
                 'Sentiment', 'Commitment', 'NetZero']

        pbar_files = tqdm(
            total=n_pdfs, desc="Files", unit="pdf", position=0, leave=True,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        pbar_steps = tqdm(
            total=len(STEPS), desc="Starting...", position=1, leave=False,
            dynamic_ncols=True, bar_format='{desc} [{bar}] {n}/{total}'
        )

        def update_step(step_idx: int, step_name: str, filename: str):
            if len(filename) > 40:
                filename = "..." + filename[-37:]
            pbar_steps.n = step_idx
            pbar_steps.set_description(f"{filename} → {step_name}")
            pbar_steps.refresh()

        for pdf_file in pdf_files:
            relative_path = pdf_file.relative_to(root)
            display_name = str(relative_path)

            pbar_steps.n = 0
            pbar_steps.refresh()

            try:
                self.set_pdf_path(str(pdf_file))

                # Check if already processed
                bert_cache = self._get_cache_path('bert')
                if bert_cache.exists():
                    with open(bert_cache, 'r') as f:
                        cached = json.load(f)
                    if cached.get('netzero_analyzed'):
                        stats['skipped'] += 1
                        stats['extracted'] += 1
                        stats['filtered'] += 1
                        stats['analyzed'] += 1
                        if cached.get('source') == 'translated':
                            stats['translated'] += 1
                        pbar_steps.n = len(STEPS)
                        pbar_steps.set_description(
                            f"{display_name[-30:]} → Cached ✓")
                        pbar_steps.refresh()
                        pbar_files.update(1)
                        continue

                update_step(1, STEPS[0], display_name)
                result = self.extract_and_translate_pdf()
                stats['extracted'] += 1
                if result.get('translated'):
                    stats['translated'] += 1

                update_step(2, STEPS[1], display_name)
                self.filter_climate_chunks()
                stats['filtered'] += 1

                update_step(3, STEPS[2], display_name)
                self.analyze_specificity()

                update_step(4, STEPS[3], display_name)
                self.analyze_sentiment()

                update_step(5, STEPS[4], display_name)
                self.analyze_commitments()

                update_step(6, STEPS[5], display_name)
                self.analyze_netzero()
                stats['analyzed'] += 1

                pbar_steps.n = len(STEPS)
                pbar_steps.set_description(f"{display_name[-30:]} → Done ✓")
                pbar_steps.refresh()

                self._clear_gpu_memory()

            except Exception as e:
                stats['errors'].append(
                    {'file': str(relative_path), 'error': str(e)})
                pbar_steps.set_description(f"{display_name[-30:]} → Error ✗")
                pbar_steps.refresh()
                if not skip_errors:
                    pbar_steps.close()
                    pbar_files.close()
                    raise
                self._emergency_gpu_cleanup()

            pbar_files.update(1)

        pbar_steps.close()
        pbar_files.close()

        stats['elapsed_time'] = time.time() - stats['start_time']
        stats['avg_time_per_pdf'] = stats['elapsed_time'] / \
            n_pdfs if n_pdfs else 0

        print(f"\n{'='*60}")
        print(f"COMPLETE")
        print(f"{'='*60}")
        print(f"  Processed:   {stats['extracted']}/{stats['total']} PDFs")
        print(f"  From cache:  {stats['skipped']}")
        print(f"  Translated:  {stats['translated']}")
        print(f"  Analyzed:    {stats['analyzed']}")
        print(f"  Errors:      {len(stats['errors'])}")
        print(f"  Total time:  {stats['elapsed_time']/60:.1f} min")
        print(f"  Avg per PDF: {stats['avg_time_per_pdf']:.1f}s")
        print(f"{'='*60}\n")

        if stats['errors']:
            print("Errors encountered:")
            for err in stats['errors'][:5]:
                print(f"  • {err['file']}: {err['error'][:60]}")
            if len(stats['errors']) > 5:
                print(f"  ... and {len(stats['errors'])-5} more")

        return stats

    def cleanup(self):
        """Release all resources."""
        self._emergency_gpu_cleanup()
        self.preprocessor.cleanup()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_reports(
    path: str,
    cache_dir: str = "cache",
    translate: bool = True,
    skip_errors: bool = True
) -> Dict:
    """
    Convenience function to analyze PDF reports.

    Args:
        path: Path to PDF or directory
        cache_dir: Cache directory for results
        translate: Whether to translate non-English documents
        skip_errors: Continue on errors

    Returns:
        Processing statistics
    """
    config = BERTConfig(
        cache_dir=cache_dir,
        translate_to_english=translate,
    )
    analyzer = ClimateBERTAnalyzer(config)
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
    print("  from bert_pipeline import ClimateBERTAnalyzer, analyze_reports")
    print()
    print("  # Quick start")
    print("  stats = analyze_reports('../data/reports')")
    print()
    print("  # Detailed usage")
    print("  analyzer = ClimateBERTAnalyzer()")
    print("  analyzer.set_pdf_path('report.pdf')")
    print("  result = analyzer.run_full_pipeline()")
