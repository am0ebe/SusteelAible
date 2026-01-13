"""
JSON Cache Loader Module
========================

Shared utilities for loading and parsing preprocessed JSON cache files.
Used by both BERT visualization and RAG pipelines.

Supports two cache file types:
- *_prep.json: Preprocessing results (chunks, metadata, language info)
- *_bert.json: BERT analysis results (includes classification scores)

Usage:
    from nlp.json_loader import CacheLoader, load_prep_cache, load_bert_cache

    # Load all prep files
    loader = CacheLoader(cache_dir="cache")
    documents = loader.load_prep_files()

    # Load all bert files
    bert_data = loader.load_bert_files()

    # Convenience functions
    documents = load_prep_cache("cache")
    bert_data = load_bert_cache("cache")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CachedDocument:
    """Represents a document loaded from prep.json cache."""

    pdf_path: str
    company: Optional[str] = None
    company_id: Optional[str] = None
    year: Optional[int] = None
    language: str = "en"
    translated: bool = False
    extraction_method: str = "unknown"
    num_pages: int = 0
    num_chunks: int = 0
    chunks: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    chunk_pairs: List[Dict] = field(default_factory=list)
    processed_at: Optional[str] = None

    @classmethod
    def from_json(cls, data: Dict) -> "CachedDocument":
        """Create from JSON data."""
        return cls(
            pdf_path=data.get("pdf_path", ""),
            company=data.get("company"),
            company_id=data.get("company_id"),
            year=data.get("year"),
            language=data.get("language", "en"),
            translated=data.get("translated", False),
            extraction_method=data.get("extraction_method", "unknown"),
            num_pages=data.get("num_pages", 0),
            num_chunks=data.get("num_chunks", 0),
            chunks=data.get("chunks", []),
            chunk_ids=data.get("chunk_ids", []),
            chunk_pairs=data.get("chunk_pairs", []),
            processed_at=data.get("processed_at"),
        )

    def to_langchain_documents(self) -> List:
        """Convert to LangChain Document format."""
        from langchain.schema import Document

        docs = []
        for i, chunk_text in enumerate(self.chunks):
            chunk_id = self.chunk_ids[i] if i < len(
                self.chunk_ids) else f"chunk_{i:03d}"
            docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": chunk_id,
                    "source_path": self.pdf_path,
                    "company_name": self.company,
                    "company_id": self.company_id,
                    "year": str(self.year) if self.year else None,
                    "language": self.language,
                    "translated": self.translated,
                    "chunk_index": i,
                },
            ))
        return docs


@dataclass
class BERTAnalyzedDocument:
    """Represents a document loaded from bert.json cache with analysis scores."""

    pdf_path: str
    company: Optional[str] = None
    company_id: Optional[str] = None
    year: Optional[int] = None
    language: str = "en"
    translated: bool = False

    # Filtering info
    total_chunks: int = 0
    climate_chunks: int = 0
    kept_percentage: float = 0.0

    # Analysis flags
    filtered: bool = False
    specificity_analyzed: bool = False
    sentiment_analyzed: bool = False
    commitment_analyzed: bool = False
    netzero_analyzed: bool = False

    # Net-zero stats
    netzero_count: int = 0
    netzero_pct: float = 0.0

    # Analyzed chunks with scores
    chunks: List[Dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Dict) -> "BERTAnalyzedDocument":
        """Create from JSON data."""
        return cls(
            pdf_path=data.get("pdf_path", ""),
            company=data.get("company"),
            company_id=data.get("company_id"),
            year=data.get("year"),
            language=data.get("language", "en"),
            translated=data.get("translated", False),
            total_chunks=data.get("total_chunks", 0),
            climate_chunks=data.get("climate_chunks", 0),
            kept_percentage=data.get("kept_percentage", 0.0),
            filtered=data.get("filtered", False),
            specificity_analyzed=data.get("specificity_analyzed", False),
            sentiment_analyzed=data.get("sentiment_analyzed", False),
            commitment_analyzed=data.get("commitment_analyzed", False),
            netzero_analyzed=data.get("netzero_analyzed", False),
            netzero_count=data.get("netzero_count", 0),
            netzero_pct=data.get("netzero_pct", 0.0),
            chunks=data.get("chunks", []),
        )

    def get_chunk_texts(self) -> List[str]:
        """Get just the text from each chunk."""
        return [c.get("text", "") for c in self.chunks]

    def to_langchain_documents(self) -> List:
        """Convert to LangChain Document format with BERT scores in metadata."""
        from langchain.schema import Document

        docs = []
        for i, chunk in enumerate(self.chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{i:03d}")
            docs.append(Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "chunk_id": chunk_id,
                    "source_path": self.pdf_path,
                    "company_name": self.company,
                    "company_id": self.company_id,
                    "year": str(self.year) if self.year else None,
                    "language": self.language,
                    "translated": self.translated,
                    "chunk_index": i,
                    # BERT scores
                    "detector_score": chunk.get("detector_score", 0),
                    "specificity_label": chunk.get("specificity_label"),
                    "specificity_score": chunk.get("specificity_score", 0),
                    "sentiment_label": chunk.get("sentiment_label"),
                    "sentiment_score": chunk.get("sentiment_score", 0),
                    "commitment_label": chunk.get("commitment_label"),
                    "commitment_score": chunk.get("commitment_score", 0),
                    "netzero_label": chunk.get("netzero_label"),
                    "netzero_score": chunk.get("netzero_score", 0),
                },
            ))
        return docs


# =============================================================================
# MAIN LOADER CLASS
# =============================================================================

class CacheLoader:
    """
    Loads and parses JSON cache files from preprocessing and BERT analysis.

    Supports:
    - *_prep.json: Raw preprocessed chunks
    - *_bert.json: BERT-analyzed chunks with classification scores
    """

    def __init__(self, cache_dir: str = "cache", exclude_year_gte: Optional[int] = None):
        """
        Initialize the cache loader.

        Args:
            cache_dir: Directory containing cache files
            exclude_year_gte: Exclude documents with year >= this value
        """
        self.cache_dir = Path(cache_dir)
        self.exclude_year_gte = exclude_year_gte

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    def _should_exclude(self, year: Optional[int]) -> bool:
        """Check if document should be excluded based on year."""
        if self.exclude_year_gte is None or year is None:
            return False
        return year >= self.exclude_year_gte

    # -------------------------------------------------------------------------
    # Prep File Loading
    # -------------------------------------------------------------------------

    def load_prep_files(self) -> List[CachedDocument]:
        """Load all *_prep.json files."""
        prep_files = list(self.cache_dir.glob("*_prep.json"))

        if not prep_files:
            raise FileNotFoundError(
                f"No prep cache files found in {self.cache_dir}")

        documents = []
        for fp in prep_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                doc = CachedDocument.from_json(data)

                if self._should_exclude(doc.year):
                    continue

                documents.append(doc)

            except Exception as e:
                print(f"⚠️ Error loading {fp.name}: {e}")
                continue

        return documents

    def load_prep_as_langchain(self) -> List:
        """Load prep files and convert to LangChain Documents."""
        documents = self.load_prep_files()
        all_docs = []
        for doc in documents:
            all_docs.extend(doc.to_langchain_documents())
        return all_docs

    # -------------------------------------------------------------------------
    # BERT File Loading
    # -------------------------------------------------------------------------

    def load_bert_files(self) -> List[BERTAnalyzedDocument]:
        """Load all *_bert.json files."""
        bert_files = list(self.cache_dir.glob("*_bert.json"))

        if not bert_files:
            raise FileNotFoundError(
                f"No BERT cache files found in {self.cache_dir}")

        documents = []
        for fp in bert_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                doc = BERTAnalyzedDocument.from_json(data)

                if self._should_exclude(doc.year):
                    continue

                documents.append(doc)

            except Exception as e:
                print(f"⚠️ Error loading {fp.name}: {e}")
                continue

        return documents

    def load_bert_as_langchain(self) -> List:
        """Load BERT files and convert to LangChain Documents with scores."""
        documents = self.load_bert_files()
        all_docs = []
        for doc in documents:
            all_docs.extend(doc.to_langchain_documents())
        return all_docs

    def load_bert_raw(self) -> List[Dict]:
        """Load BERT files as raw dictionaries (for bert_2 visualization)."""
        bert_files = list(self.cache_dir.glob("*_bert.json"))

        if not bert_files:
            return []

        raw_data = []
        for fp in bert_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                year = data.get("year")
                if self._should_exclude(year):
                    continue

                raw_data.append(data)

            except Exception as e:
                print(f"⚠️ Error loading {fp.name}: {e}")
                continue

        return raw_data

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_companies(self, source: str = "prep") -> List[str]:
        """Get sorted list of unique company IDs."""
        if source == "prep":
            documents = self.load_prep_files()
            return sorted(set(d.company_id for d in documents if d.company_id))
        else:
            documents = self.load_bert_files()
            return sorted(set(d.company_id for d in documents if d.company_id))

    def get_years_for_company(self, company_id: str, source: str = "prep") -> List[int]:
        """Get sorted list of years for a specific company."""
        if source == "prep":
            documents = self.load_prep_files()
        else:
            documents = self.load_bert_files()

        return sorted(set(
            d.year for d in documents
            if d.company_id == company_id and d.year is not None
        ))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of available cache files."""
        prep_files = list(self.cache_dir.glob("*_prep.json"))
        bert_files = list(self.cache_dir.glob("*_bert.json"))

        return {
            "cache_dir": str(self.cache_dir),
            "prep_files": len(prep_files),
            "bert_files": len(bert_files),
            "exclude_year_gte": self.exclude_year_gte,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_prep_cache(
    cache_dir: str = "cache",
    exclude_year_gte: Optional[int] = None,
    as_langchain: bool = False
) -> List:
    """
    Convenience function to load prep cache files.

    Args:
        cache_dir: Directory containing cache files
        exclude_year_gte: Exclude documents with year >= this value
        as_langchain: If True, return LangChain Documents

    Returns:
        List of CachedDocument or LangChain Documents
    """
    loader = CacheLoader(cache_dir, exclude_year_gte)
    if as_langchain:
        return loader.load_prep_as_langchain()
    return loader.load_prep_files()


def load_bert_cache(
    cache_dir: str = "cache",
    exclude_year_gte: Optional[int] = None,
    as_langchain: bool = False,
    raw: bool = False
) -> List:
    """
    Convenience function to load BERT cache files.

    Args:
        cache_dir: Directory containing cache files
        exclude_year_gte: Exclude documents with year >= this value
        as_langchain: If True, return LangChain Documents
        raw: If True, return raw dictionaries (for visualization)

    Returns:
        List of BERTAnalyzedDocument, LangChain Documents, or raw dicts
    """
    loader = CacheLoader(cache_dir, exclude_year_gte)
    if raw:
        return loader.load_bert_raw()
    if as_langchain:
        return loader.load_bert_as_langchain()
    return loader.load_bert_files()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("JSON Cache Loader Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from nlp.json_loader import CacheLoader, load_prep_cache, load_bert_cache")
    print()
    print("  # Load prep files")
    print("  documents = load_prep_cache('cache')")
    print()
    print("  # Load as LangChain documents for RAG")
    print("  lc_docs = load_prep_cache('cache', as_langchain=True)")
    print()
    print("  # Load BERT files for visualization")
    print("  bert_data = load_bert_cache('cache', raw=True)")
